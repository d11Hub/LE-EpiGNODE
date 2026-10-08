
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training import checkpoints as utils


class ResMLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, activation):
        super().__init__()
        if activation == 'relu':
            Act = lambda: nn.ReLU(inplace=False)
        elif activation == 'tanh':
            Act = lambda: nn.Tanh()
        else:
            Act = lambda: nn.Softplus(beta=1.0)

        self.input_layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Act()
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                Act()
            )
            for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_size, out_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation == 'relu':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    gain = nn.init.calculate_gain('tanh') if activation == 'tanh' else 1.0
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output_layer(x)


class LE_EpiGNN(nn.Module):
    SCENARIO = None

    def __init__(self, args, t, datas, U_0, C_mat, dist_mat, static_pop_city_raw=None):
        'Initialize model settings and input tensors.'
        super(LE_EpiGNN, self).__init__()
        self.args = args
        self.scenario = self.SCENARIO
        if self.scenario not in {"eulerian_BRGM", "eulerian_CRGM"}:
            raise ValueError(f"Unknown Eulerian scenario: {self.scenario}")
        self.device = torch.device(args.device)


        if not getattr(args, 'inference_only', False):
            if not 0 < args.validation_time < args.training_time <= len(t):
                raise ValueError("Expected 0 < validation_time < training_time <= data length")
            self.val_start = args.training_time - args.validation_time
            self.t_train = t[:args.training_time + 1].clone().detach().to(self.device).float()
            self.t_val = t[self.val_start:args.training_time].clone().detach().to(
                self.device).float()
        self.t_test = t.clone().detach().to(self.device).float()


        if not getattr(args, 'inference_only', False):
            self.train_data = torch.tensor(datas[:args.training_time, :, :], dtype=torch.float32, device=self.device)
            self.val_data = torch.tensor(datas[self.val_start:args.training_time, :, :],
                                         dtype=torch.float32, device=self.device)


        self.t_min = float(t[0].item())
        self.t_max = float(t[-1].item())
        self.num_cities = args.num_cities
        self.num_ages = args.num_ages
        self.pop_ref = args.pop_ref
        self.re_factor = args.re_factor
        if self.scenario == "eulerian_CRGM":
            if static_pop_city_raw is None:
                raise ValueError("CRGM requires static_pop_city_raw")
            self.register_buffer("static_pop_city_raw",
                                 torch.tensor(static_pop_city_raw, dtype=torch.float32, device=self.device))


        self.use_GE = (self.num_cities > 1)  

        default_mob_path = os.path.join(
            "data",
            getattr(args, 'dataset', ''),
            "pretrained_MLP_mob.pth"
        )
        self.pretrained_mob_path = (
                getattr(args, 'pretrained_mob_path', '') or
                getattr(args, 'mob_ckpt_path', '') or
                default_mob_path
        )

        N_nodes = self.num_cities * self.num_ages


        C_mat = np.asarray(C_mat, dtype=np.float32)
        if C_mat.shape != (N_nodes, N_nodes):
            raise ValueError(
                f"C_mat must have shape {(N_nodes, N_nodes)}; got {C_mat.shape}"
            )
        self.register_buffer("C_mat", torch.tensor(C_mat, dtype=torch.float32, device=self.device))
        dist_mat = np.asarray(dist_mat, dtype=np.float32)
        self.register_buffer("dist_mat", torch.tensor(dist_mat, dtype=torch.float32, device=self.device))

        edge_mask = torch.nonzero(self.C_mat > 0, as_tuple=False)  
        self.register_buffer("edge_index_L", edge_mask.t().long())  


        senders_idx, receivers_idx = self.edge_index_L
        self.register_buffer("C_edge", self.C_mat[senders_idx, receivers_idx].unsqueeze(-1))


        self.c_hidden_dim = getattr(args, "c_hidden_dim", 8)

        self.C_edge_linear = nn.Sequential(
            nn.Linear(1, self.c_hidden_dim, bias=True),
            nn.LayerNorm(self.c_hidden_dim),
            nn.Linear(self.c_hidden_dim, 1, bias=True)
        )
        for m in self.C_edge_linear:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


        U0_tensor = torch.tensor(U_0, dtype=torch.float32, device=self.device)
        self.register_buffer("S_0", U0_tensor[:, 0].clone())
        self.I_0 = nn.Parameter(U0_tensor[:, 1].clone(), requires_grad=True)
        self.register_buffer("R_0", U0_tensor[:, 2].clone())


        self.gamma = torch.tensor(args.gamma, dtype=torch.float32, device=self.device)  


        if self.use_GE:
            self.eta = torch.tensor([1.0], device=self.device)
            self.alpha_E_raw = nn.Parameter(torch.tensor(300.0, device=self.device))
            self.eps_E = torch.tensor(0.0, device=self.device)
        else:
            self.register_buffer("eta_raw", torch.zeros(self.num_ages, dtype=torch.float32, device=self.device))
            self.register_buffer("alpha_E_raw", torch.tensor(0.0, dtype=torch.float32, device=self.device))
            self.eps_E = torch.tensor(0.0, device=self.device)


        self.alpha_grav = torch.tensor(1.0, device=self.device)
        self.beta_grav = torch.tensor(1.0, device=self.device)
        self.lambda_grav = torch.tensor(0.012, device=self.device)


        self.MLP_L = ResMLP(
            in_size=6 + 1 + 1,
            hidden_size=args.gl_hidden_dim,
            out_size=1,
            num_layers=args.gl_num_layers,
            activation='softplus'
        )


        self.MLP_mob = None
        if self.use_GE:
            self.MLP_mob = ResMLP(
                in_size=3 +1 ,
                hidden_size=getattr(args, 'mob_hidden_dim', 64),
                out_size=1,
                num_layers=getattr(args, 'mob_num_layers', 2),
                activation='softplus'
            )

            if not os.path.exists(self.pretrained_mob_path):
                raise FileNotFoundError(f"Pretrained MLP_mob weights not found: {self.pretrained_mob_path}")

            ckpt = torch.load(self.pretrained_mob_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and 'MLP_mob_state_dict' in ckpt:
                mob_state = ckpt['MLP_mob_state_dict']
            else:
                mob_state = ckpt

            self.MLP_mob.load_state_dict(mob_state, strict=True)

            for p in self.MLP_mob.parameters():
                p.requires_grad = False

            self.MLP_mob.eval()

        self.to(self.device)


    @property
    def alpha_E(self):
        if self.use_GE:
            return F.softplus(self.alpha_E_raw)
        else:
            return torch.tensor(0.0, device=self.device)

    @property
    def U_0(self):
        return torch.stack([self.S_0, self.I_0, self.R_0], dim=1)


    def compute_B_matrix(self, y, t):
        "Construct predicted city mobility from current states; return mobility and population aggregates."
        S, I, R = y[:, 0], y[:, 1], y[:, 2]

        if not self.use_GE:
            pop_patch = S + I + R
            pop_patch_raw = pop_patch / self.re_factor
            pop_patch_raw_2d = pop_patch_raw.view(self.num_cities, self.num_ages)
            pop_city_raw = pop_patch_raw_2d.sum(dim=1)
            B_pred = torch.zeros(self.num_cities, self.num_cities, device=self.device)
            return B_pred, pop_city_raw, pop_patch_raw_2d

        pop_patch = S + I + R


        pop_patch_raw = pop_patch / self.re_factor
        pop_patch_raw_2d = pop_patch_raw.view(self.num_cities, self.num_ages)
        pop_city_raw = pop_patch_raw_2d.sum(dim=1)

        I_patch_raw = I / self.re_factor
        I_city_raw = I_patch_raw.view(self.num_cities, self.num_ages).sum(dim=1)
        gammaI_city_raw = self.gamma * I_city_raw

        population = (self.static_pop_city_raw if self.scenario == "eulerian_CRGM"
                      else pop_city_raw)
        dens = gammaI_city_raw / (population + 1e-12)

        B_pred = torch.zeros(self.num_cities, self.num_cities, device=self.device)

        for l in range(self.num_cities):
            for k in range(self.num_cities):
                if l == k:
                    continue

                P_l_norm = population[l] / self.pop_ref
                P_k_norm = population[k] / self.pop_ref

                grav = (
                        (P_l_norm ** self.alpha_grav) *
                        (P_k_norm ** self.beta_grav) *
                        torch.exp(-self.lambda_grav * self.dist_mat[l, k])
                )
                if self.scenario == "eulerian_CRGM":
                    t_norm = 10 * (t - self.t_min) / max(self.t_max - self.t_min, 1.0)
                else:
                    t_norm = 0.001 * (t - self.t_min) / (
                            self.t_max - self.t_min + 1e-12
                    )

                feat_mob = torch.stack([dens[l], dens[k], grav, t_norm])

                b_val = F.softplus(self.MLP_mob(feat_mob))
                B_pred[l, k] = b_val.squeeze()

        return B_pred, pop_city_raw, pop_patch_raw_2d

    def predict_B_series(self, state_series, t_series):
        """
        state_series: [T_seq, N, 3]
        t_series:     [T_seq]
        return:       [T_seq, num_cities, num_cities]
        """
        if not self.use_GE:
            T_seq = state_series.shape[0]
            return torch.zeros(T_seq, self.num_cities, self.num_cities, device=self.device)

        B_list = []
        for y_t, t_t in zip(state_series, t_series):
            B_t, _, _ = self.compute_B_matrix(y_t, t_t)
            B_list.append(B_t)
        return torch.stack(B_list, dim=0)


    def compute_GE_GL(self, y, t):
        N = self.num_cities * self.num_ages
        senders_idx, receivers_idx = self.edge_index_L
        E = senders_idx.size(0)


        v_i = y[senders_idx]  
        v_j = y[receivers_idx]  


        c_emb = self.C_edge_linear(self.C_edge)
        t_norm = 0.001 * (t - self.t_min) / (self.t_max - self.t_min + 1e-12)

        t_feat = t_norm.reshape(1, 1).expand(E, 1)
        feat_L = torch.cat([v_i, v_j, c_emb, t_feat], dim=-1)  

        out_L = self.MLP_L(feat_L)
        GL_flat = F.softplus(out_L).squeeze(-1)  


        G_L_raw = torch.zeros(N, N, device=self.device)
        G_L_raw[senders_idx, receivers_idx] = GL_flat


        pop = y[:, 0] + y[:, 1] + y[:, 2]  
        G_L = G_L_raw / (pop[:, None] + 1e-9)  

        if not self.use_GE:
            G_E = torch.zeros(N, N, device=self.device)
        else:
            B_t, pop_city_raw, pop_patch_raw_2d = self.compute_B_matrix(y, t)

            G_E = torch.zeros(N, N, device=self.device)

            eta_share = self.eta / (torch.sum(self.eta) + 1e-12)

            for l in range(self.num_cities):

                P_l = (self.static_pop_city_raw[l] if self.scenario == "eulerian_CRGM"
                       else pop_city_raw[l]) + 1e-12

                for k in range(self.num_cities):
                    if l == k:
                        continue


                    M_lk = self.alpha_E * B_t[l, k]


                    base_rate = M_lk / P_l + self.eps_E


                    for a in range(self.num_ages):
                        i = l * self.num_ages + a
                        j = k * self.num_ages + a
                        G_E[i, j] = eta_share[a] * base_rate


            G_E.fill_diagonal_(0.0)
            row_sums = G_E.sum(dim=1)
            G_E = G_E - torch.diag(row_sums)

        return G_E, G_L


    def GNNSIRSystem(self, t, y):
        y = y.to(self.device)
        S, I, R = y[:, 0], y[:, 1], y[:, 2]

        G_E, G_L = self.compute_GE_GL(y, t)


        dS_dt = G_E.T @ S - G_L.T @ I
        dI_dt = G_E.T @ I + G_L.T @ I - self.gamma * I
        dR_dt = G_E.T @ R + self.gamma * I

        return torch.stack([dS_dt, dI_dt, dR_dt], dim=1)


    def compute_loss(self, pred_u, true_u, ic_weight=0.0, include_ic=False):
        "Compare gamma*I_pred with last-channel observed gammaI using peak-normalized MSE at node or city level."
        I_pred = pred_u[:, :, 1]
        gammaI_pred = self.gamma * I_pred
        gammaI_true = true_u[:, :, 3]

        eps = 1e-6


        if self.args.loss_level == 'city':
            gammaI_pred = gammaI_pred.view(-1, self.num_cities, self.num_ages).sum(dim=2)  
            gammaI_true = gammaI_true.view(-1, self.num_cities, self.num_ages).sum(dim=2)  


        scale = torch.amax(gammaI_true, dim=0, keepdim=True).detach() + eps
        scale_raw = torch.sqrt_(scale)

        gammaI_pred_norm = gammaI_pred / scale
        gammaI_true_norm = gammaI_true / scale
        loss_norm = torch.mean((gammaI_pred_norm - gammaI_true_norm) ** 2)
        loss_raw = torch.mean((gammaI_pred - gammaI_true) ** 2)
        loss =  loss_raw


        if include_ic:
            ic_loss = torch.mean((gammaI_pred[0] - gammaI_true[0]) ** 2)
            loss = loss + ic_weight * ic_loss

        return loss


    def train_series(self, exp_dir):
        num_epochs = self.args.epochs
        base_lr = self.args.lr

        param_groups = [
            {'params': list(self.MLP_L.parameters()), 'lr': base_lr},
            {'params': [self.I_0], 'lr': base_lr * 0.0},
            {'params': list(self.C_edge_linear.parameters()), 'lr': base_lr},
        ]

        if self.use_GE:

            param_groups.append({'params': [self.alpha_E_raw], 'lr': base_lr})

        optimizer = optim.Adam(param_groups, lr=base_lr)


        best_loss = float('inf')
        loss_history = []
        eta_trace, alpha_E_trace, eps_E_trace = [], [], []

        train_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()

            pred_u = odeint(self.GNNSIRSystem, self.U_0, self.t_train, method='dopri5', rtol=1e-5, atol=1e-8)

            val_u = pred_u[self.val_start:self.args.training_time, :, :]

            train_loss = self.compute_loss(
                pred_u[:-1, :, :],
                self.train_data,
                ic_weight=self.args.ic_weight,
                include_ic=True
            )

            val_loss = self.compute_loss(
                val_u,
                self.val_data,
                ic_weight=0.0,
                include_ic=False
            )

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.args.grad_clip)
            if val_loss < best_loss:
                best_loss = val_loss.item()
                best_epoch = epoch
                utils.save_checkpoint(
                    self, optimizer, best_epoch, best_loss,
                    os.path.join(exp_dir, "checkpoints"),
                    is_best=True
                )

            optimizer.step()


            if self.args.use_lr_increase:
                if (epoch + 1) >= self.args.lr_increase_start and\
                        (epoch + 1 - self.args.lr_increase_start) % self.args.lr_increase_every == 0:
                    for pg in optimizer.param_groups:
                        old_lr = pg['lr']
                        new_lr = min(old_lr * self.args.lr_increase_factor, self.args.lr_max)
                        pg['lr'] = new_lr
                    print(f">>> Epoch {epoch + 1}: learning rate increased to {optimizer.param_groups[0]['lr']:.2e}")

            current_lr = optimizer.param_groups[0]['lr']
            loss_history.append({
                'Loss': train_loss.item(),
                'Val_Loss': val_loss.item(),
                'Learning Rate': current_lr
            })


            eta_trace.append(self.eta.detach().cpu().numpy())
            alpha_E_trace.append(self.alpha_E.detach().cpu().item())
            eps_E_trace.append(self.eps_E.detach().cpu().item())

            utils.save_checkpoint(self, optimizer, epoch + 1, train_loss, os.path.join(exp_dir, "checkpoints"))

            epoch_duration = time.time() - epoch_start_time

            eta_str = "[" + ", ".join([f"{e:.3f}" for e in self.eta.detach().cpu().tolist()]) + "]"

            print(
                f"Epoch {epoch + 1:03d} | Loss={train_loss.item():.4e} | "
                f"Val={val_loss.item():.4e} | lr={current_lr:.2e} | "
                f"alpha={self.alpha_E.item():.2f} | eps={self.eps_E.item():.6f} | "
                f"eta={eta_str} | "
                f"Time={epoch_duration:.2f}s"
            )

        print(f"\nTraining complete; best weights after {best_epoch} updates, lowest monitoring loss: {best_loss:.4e}")
        return loss_history, eta_trace, alpha_E_trace, eps_E_trace


    def predict(self, best_model_path):
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.eval()

        t_pre = self.t_test.to(self.device)
        U_0 = self.U_0.to(self.device)

        with torch.no_grad():
            pred_u = odeint(self.GNNSIRSystem, U_0, t_pre, method='dopri5', rtol=1e-5, atol=1e-8)

        I_pred = pred_u[:, :, 1]
        gammaI = (self.gamma * I_pred).unsqueeze(-1)
        pred_u_extend = torch.cat([pred_u, gammaI], dim=-1)

        GE_list, GL_list = [], []
        for t_val, state in zip(t_pre, pred_u):
            GE_t, GL_t = self.compute_GE_GL(state, t_val)
            GE_list.append(GE_t.detach().cpu().numpy())
            GL_list.append(GL_t.detach().cpu().numpy())
        B_pred = self.predict_B_series(pred_u, t_pre)

        return pred_u_extend.detach().cpu().numpy(), np.array(GE_list), np.array(GL_list), B_pred.detach().cpu().numpy()
