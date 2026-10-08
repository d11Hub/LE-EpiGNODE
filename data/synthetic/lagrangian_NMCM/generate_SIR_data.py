
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp


mpl.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MixedEulerianLagrangianEpidemicModel:
    def __init__(self):

        

        self.num_cities = 1
        self.num_ages = 4
        self.k = self.num_cities * self.num_ages
        self.output_dir = "Dataset_LE_1City_4Age"


        self.use_GE = (self.num_cities > 1)


        

        self.beta = 0.01       
        self.gamma = 0.02
        self.alpha_E = 10000.0       
        self.re_factor = 1e-4


        self.p_S = 1.0
        self.q_I = 0.4


        

        self.eta = np.array([0.18, 0.30, 0.28, 0.24], dtype=float)


        

        

        

        

        


        

        
        pop_city_raw = np.array([
            [82000, 102000, 130000, 65000]   
        ], dtype=float)


        I0_city_raw = np.array([
            [1900, 1000, 1200, 1300]
        ], dtype=float)

        assert pop_city_raw.shape == (self.num_cities, self.num_ages)
        assert I0_city_raw.shape == (self.num_cities, self.num_ages)

        self.S0 = (pop_city_raw - I0_city_raw).reshape(-1) * self.re_factor
        self.I0 = I0_city_raw.reshape(-1) * self.re_factor
        self.R0 = np.zeros(self.k, dtype=float)


        

        self.alpha_grav = 1.0
        self.beta_grav = 1.0
        self.lambda_grav = 0.012
        self.kappa_grav = 40.0
        self.pop_ref = 1e6


        self.rho_src = 6.0
        self.rho_dst = 10.0

        self.contact_mats = [
            np.array([
                [3.20, 2.92, 2.79, 3.17],
                [2.88, 3.00, 2.54, 3.13],
                [2.71, 2.56, 2.50, 2.81],
                [3.23, 3.22, 2.84, 3.50]
            ], dtype=float)
        ]
        self.city_contact_scale = np.array([1.0], dtype=float)


    

    def compute_baidu_migration(self, pop_city_raw, gammaI_city_raw):
        """
        B_lk = kappa * (P_l/pop_ref)^alpha * (P_k/pop_ref)^beta * exp(-lambda*d_lk)
                     * exp(-rho_src * gammaI_l / P_l - rho_dst * gammaI_k / P_k)
        """
        if not self.use_GE:
            return np.zeros((self.num_cities, self.num_cities), dtype=float)

        B_matrix = np.zeros((self.num_cities, self.num_cities), dtype=float)

        for l in range(self.num_cities):
            for k in range(self.num_cities):
                if l == k:
                    continue

                dist_lk = self.dist_matrix[l, k]
                P_l = pop_city_raw[l]
                P_k = pop_city_raw[k]

                P_l_norm = P_l / self.pop_ref
                P_k_norm = P_k / self.pop_ref

                gravity_lk = (
                    (P_l_norm ** self.alpha_grav) *
                    (P_k_norm ** self.beta_grav) *
                    np.exp(-self.lambda_grav * dist_lk)
                )

                dens_l = gammaI_city_raw[l] / (P_l + 1e-12)
                dens_k = gammaI_city_raw[k] / (P_k + 1e-12)

                epidemic_factor = np.exp(
                    - self.rho_src * dens_l
                    - self.rho_dst * dens_k
                )

                B_matrix[l, k] = self.kappa_grav * gravity_lk * epidemic_factor

        return B_matrix

    def add_gaussian_noise(self, data, noise_level=0.05, seed=2026, nonnegative=True):
        "Add Gaussian noise scaled by the peak of each series; use the supplied seed and optionally clip to nonnegative values."
        rng = np.random.default_rng(seed)

        data = np.asarray(data, dtype=float)


        
        scale = np.max(np.abs(data), axis=0, keepdims=True)
        scale = np.maximum(scale, 1e-12)

        noise = rng.normal(
            loc=0.0,
            scale=noise_level * scale,
            size=data.shape
        )

        noisy_data = data + noise

        if nonnegative:
            noisy_data = np.clip(noisy_data, 0.0, None)

        return noisy_data


    

    def compute_operators(self, y):
        'Initialize model settings and input tensors.'
        X_S = y[:self.k]
        X_I = y[self.k: 2 * self.k]
        X_R = y[2 * self.k: 3 * self.k]


        

        pop_age = X_S + X_I + X_R
        pop_age_raw = pop_age / self.re_factor
        pop_city_raw_2d = pop_age_raw.reshape(self.num_cities, self.num_ages)
        pop_city_raw = pop_city_raw_2d.sum(axis=1)

        I_age_raw = X_I / self.re_factor
        I_city_raw_2d = I_age_raw.reshape(self.num_cities, self.num_ages)
        I_city_raw = I_city_raw_2d.sum(axis=1)
        gammaI_city_raw = self.gamma * I_city_raw


        

        

        G_L = np.zeros((self.k, self.k), dtype=float)

        for c in range(self.num_cities):
            idx = slice(c * self.num_ages, (c + 1) * self.num_ages)

            S_city = X_S[idx]  
            I_city = X_I[idx]
            S_safe = np.maximum(S_city, 0.0)
            I_safe = np.maximum(I_city, 0.0)
            pop_city_age = pop_age[idx]
            C_city = self.contact_mats[c] * self.city_contact_scale[c]


            G_L_block_raw = self.beta * (
                    ((I_safe ** self.q_I)[:, None] * C_city) @ np.diag(S_safe ** self.p_S)
            )

            G_L_block = G_L_block_raw / (pop_city_age[:, None] + 1e-9)
            G_L[idx, idx] = G_L_block


        

        if self.use_GE:
            B_matrix = self.compute_baidu_migration(pop_city_raw, gammaI_city_raw)
        else:
            B_matrix = np.zeros((self.num_cities, self.num_cities), dtype=float)


        

        

        
        if self.use_GE:
            G_E = np.zeros((self.k, self.k), dtype=float)
            eta_share = self.eta / (np.sum(self.eta) + 1e-12)

            for l in range(self.num_cities):
                P_l = pop_city_raw[l] + 1e-12

                for k in range(self.num_cities):
                    if l == k:
                        continue

                    M_lk = self.alpha_E * B_matrix[l, k]
                    base_rate = M_lk / P_l

                    for a in range(self.num_ages):
                        i = l * self.num_ages + a
                        j = k * self.num_ages + a
                        G_E[i, j] = eta_share[a] * base_rate

            np.fill_diagonal(G_E, 0.0)
            row_sums = G_E.sum(axis=1)
            np.fill_diagonal(G_E, -row_sums)
        else:
            G_E = np.zeros((self.k, self.k), dtype=float)

        return G_L, G_E, B_matrix


    

    def sir_rhs(self, t, y):
        G_L, G_E, _ = self.compute_operators(y)

        X_S = y[:self.k]
        X_I = y[self.k: 2 * self.k]
        X_R = y[2 * self.k: 3 * self.k]


        


        
        dX_S = G_E.T @ X_S - G_L.T @ X_I
        dX_I = G_E.T @ X_I + G_L.T @ X_I - self.gamma * X_I
        dX_R = G_E.T @ X_R + self.gamma * X_I

        return np.concatenate([dX_S, dX_I, dX_R])


    

    def generate_dataset(self, t_max=200):
        print("Integrating the coupled Eulerian and Lagrangian ODE ...")
        t_span = (0.0, float(t_max))
        t_eval = np.arange(0, t_max + 1, 1, dtype=float)
        X0 = np.concatenate([self.S0, self.I0, self.R0])

        solution = solve_ivp(
            fun=self.sir_rhs,
            t_span=t_span,
            y0=X0,
            t_eval=t_eval,
            method="LSODA",
            rtol=1e-5,
            atol=1e-8
        )

        if not solution.success:
            raise RuntimeError(f"ODE integration failed: {solution.message}")

        y_scaled = solution.y
        self.t = solution.t


        

        
        self.S = y_scaled[:self.k, :].T
        self.I = y_scaled[self.k: 2 * self.k, :].T
        self.R = y_scaled[2 * self.k: 3 * self.k, :].T


        self.I_true = self.I.copy()


        self.I = self.add_gaussian_noise(
            self.I,
            noise_level=0.05,  
            seed=42,
            nonnegative=True
        )


        self.gammaI = self.I * self.gamma

        GL_list, GE_list, Baidu_list = [], [], []

        for idx in range(len(self.t)):
            G_L_t, G_E_t, B_t = self.compute_operators(y_scaled[:, idx])
            GL_list.append(G_L_t)
            GE_list.append(G_E_t)
            Baidu_list.append(B_t)

        self.G_L_series = np.stack(GL_list, axis=0)
        self.G_E_age_series = np.stack(GE_list, axis=0)
        self.Baidu_series = np.stack(Baidu_list, axis=0)


        

        T_len = len(self.t)
        self.G_E_city_series = np.zeros((T_len, self.num_cities, self.num_cities))

        for l in range(self.num_cities):
            for k in range(self.num_cities):
                if l != k:
                    for a in range(self.num_ages):
                        self.G_E_city_series[:, l, k] += self.G_E_age_series[
                            :, l * self.num_ages + a, k * self.num_ages + a
                        ]

        for i in range(T_len):
            np.fill_diagonal(self.G_E_city_series[i], 0.0)
            row_sums = self.G_E_city_series[i].sum(axis=1)
            np.fill_diagonal(self.G_E_city_series[i], -row_sums)

        self.save_data()
        print(f"Data saved to directory: {self.output_dir}")


    

    def save_data(self):
        os.makedirs(self.output_dir, exist_ok=True)


        

        np.save(os.path.join(self.output_dir, "G_L_series.npy"), self.G_L_series)

        if self.use_GE:
            np.save(os.path.join(self.output_dir, "G_E_series.npy"), self.G_E_city_series)
            np.save(os.path.join(self.output_dir, "G_E_age_series.npy"), self.G_E_age_series)
            np.save(os.path.join(self.output_dir, "Baidu_Migration_series.npy"), self.Baidu_series)

        states_age = np.stack([self.S, self.I, self.R, self.gammaI], axis=-1)
        np.save(os.path.join(self.output_dir, "states_age.npy"), states_age)

        S_city = self.S.reshape(len(self.t), self.num_cities, self.num_ages).sum(axis=2)
        I_city = self.I.reshape(len(self.t), self.num_cities, self.num_ages).sum(axis=2)
        R_city = self.R.reshape(len(self.t), self.num_cities, self.num_ages).sum(axis=2)
        gammaI_city = self.gammaI.reshape(len(self.t), self.num_cities, self.num_ages).sum(axis=2)

        states_city = np.stack([S_city, I_city, R_city, gammaI_city], axis=-1)
        np.save(os.path.join(self.output_dir, "states_city.npy"), states_city)


        

        
        gammaI_city_real = gammaI_city / self.re_factor
        df_gammaI_city = pd.DataFrame(
            gammaI_city_real,
            columns=[f"city_{i + 1}" for i in range(self.num_cities)]
        )
        df_gammaI_city.to_excel(
            os.path.join(self.output_dir, "true_city_gammaI.xlsx"),
            index=False
        )

        states_age_real = states_age / self.re_factor
        age_excel_path = os.path.join(self.output_dir, "true_age_time_series.xlsx")
        with pd.ExcelWriter(age_excel_path) as writer:
            for i in range(self.k):
                df_age = pd.DataFrame(
                    states_age_real[:, i, :],
                    columns=["S", "I", "R", "gammaI"]
                )
                df_age.to_excel(writer, index=False, sheet_name=f"age_{i + 1}")

        T_len = len(self.t)
        edge_columns = [f"{i + 1}->{j + 1}" for i in range(self.k) for j in range(self.k)]

        GL_flat = self.G_L_series.reshape(T_len, self.k * self.k)
        pd.DataFrame(GL_flat, columns=edge_columns).to_excel(
            os.path.join(self.output_dir, "true_edges_GL.xlsx"),
            index=False
        )

        GE_flat = self.G_E_age_series.reshape(T_len, self.k * self.k)
        pd.DataFrame(GE_flat, columns=edge_columns).to_excel(
            os.path.join(self.output_dir, "true_edges_GE.xlsx"),
            index=False
        )

        mob_columns = [
            f"{i + 1}->{j + 1}"
            for i in range(self.num_cities)
            for j in range(self.num_cities)
        ]
        B_flat = self.Baidu_series.reshape(T_len, self.num_cities * self.num_cities)
        pd.DataFrame(B_flat, columns=mob_columns).to_excel(
            os.path.join(self.output_dir, "true_Baidu_mobility.xlsx"),
            index=False
        )

        true_city_path = os.path.join(self.output_dir, "true_city_time_series.xlsx")
        with pd.ExcelWriter(true_city_path) as writer:
            for c in range(self.num_cities):
                df_city = pd.DataFrame(
                    states_city[:, c, :],
                    columns=["S", "I", "R", "gammaI"]
                )
                df_city_real = df_city / self.re_factor
                df_city_real.to_excel(writer, index=False, sheet_name=f"city_{c + 1}")

        gl_city_path = os.path.join(self.output_dir, "true_GL_by_city.xlsx")
        with pd.ExcelWriter(gl_city_path) as writer:
            for c in range(self.num_cities):
                idx0 = c * self.num_ages
                idx1 = (c + 1) * self.num_ages

                GL_city_block = self.G_L_series[:, idx0:idx1, idx0:idx1]
                gl_columns = [
                    f"age_{i + 1}->age_{j + 1}"
                    for i in range(self.num_ages)
                    for j in range(self.num_ages)
                ]
                GL_city_flat = GL_city_block.reshape(len(self.t), self.num_ages * self.num_ages)

                pd.DataFrame(GL_city_flat, columns=gl_columns).to_excel(
                    writer, index=False, sheet_name=f"city_{c + 1}"
                )

        ge_age_path = os.path.join(self.output_dir, "true_GE_by_age.xlsx")
        with pd.ExcelWriter(ge_age_path) as writer:
            for a in range(self.num_ages):
                city_indices = [c * self.num_ages + a for c in range(self.num_cities)]


                GE_age_block = self.G_E_age_series[:, city_indices, :][:, :, city_indices]

                ge_columns = [
                    f"city_{i + 1}->city_{j + 1}"
                    for i in range(self.num_cities)
                    for j in range(self.num_cities)
                ]
                GE_age_flat = GE_age_block.reshape(len(self.t), self.num_cities * self.num_cities)

                pd.DataFrame(GE_age_flat, columns=ge_columns).to_excel(
                    writer, index=False, sheet_name=f"age_{a + 1}"
                )


def plot_operator_series(t, G_series, title_prefix, skip_diag=False):
    T_len, N, _ = G_series.shape
    fig, axes = plt.subplots(
        N, N,
        figsize=(3 * N, 3 * N),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        sharex=True, sharey=True
    )

    if N == 1:
        axes = np.array([[axes]])

    if skip_diag:
        mask = ~np.eye(N, dtype=bool)
        vals = G_series[:, mask]
        max_val, min_val = np.max(vals), np.min(vals)
    else:
        max_val, min_val = np.max(G_series), np.min(G_series)

    for i in range(N):
        for j in range(N):
            ax = axes[i, j]
            if not (skip_diag and i == j):
                color = "red" if i == j else "green"
                ax.plot(t, G_series[:, i, j], "-", color=color, linewidth=1.5)

            pad = 0.05 * (max_val - min_val + 1e-12)
            ax.set_ylim(min_val - pad, max_val + pad)
            ax.tick_params(
                direction="out", length=3, width=1.0, labelsize=7,
                bottom=True, left=(j == 0),
                labelbottom=(i == N - 1), labelleft=(j == 0)
            )
            if i == N - 1:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment("right")

    fig.text(0.5, 0.02, "Time (days)", ha="center", va="center", fontsize=14)
    fig.text(0.02, 0.5, f"{title_prefix} element values", ha="center", va="center",
             rotation="vertical", fontsize=14)

    legend_elements = [plt.Line2D([0], [0], color="green", linestyle="-", label="Off-diagonal elements (i != j)")]
    if not skip_diag:
        legend_elements.insert(0, plt.Line2D([0], [0], color="red", linestyle="-", label="Diagonal elements (i = j)"))

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
        frameon=True,
        edgecolor="#666666",
        fontsize=12
    )
    plt.suptitle(f"{title_prefix} over time", y=0.98, fontsize=16)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.08)
    plt.show()


def plot_all_results(model):
    t = model.t


    

    I_real = model.I / model.re_factor


    plt.figure(figsize=(10, 6))
    I_city = I_real.reshape(len(t), model.num_cities, model.num_ages).sum(axis=2)
    for c in range(model.num_cities):
        plt.plot(t, I_city[:, c], label=f'City {c + 1}', linewidth=2)
    plt.title("Total infected population (I) by city over time", fontsize=15)
    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Total infected population (people)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


    I_reshaped = I_real.reshape(len(t), model.num_cities, model.num_ages)
    for c in range(model.num_cities):
        plt.figure(figsize=(10, 5))
        for a in range(model.num_ages):
            plt.plot(t, I_reshaped[:, c, a], label=f'Age group {a + 1}', linewidth=2)
        plt.title(f"City {c + 1}: infected population (I) by age group", fontsize=14)
        plt.xlabel("Time (days)", fontsize=12)
        plt.ylabel("Infected population (people)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


    


    


    for c in range(model.num_cities):
        idx_list = [c * model.num_ages + a for a in range(model.num_ages)]
        GL_sub = model.G_L_series[:, idx_list, :][:, :, idx_list]
        plot_operator_series(t, GL_sub, f"City {c + 1}: local infection operator $G_L$ (4x4)", skip_diag=False)

def print_peak_times(model):
    t = model.t


    I_real = model.I / model.re_factor   


    I_reshaped = I_real.reshape(len(t), model.num_cities, model.num_ages)

    print("\n" + "=" * 80)
    print("Peak infection times by city and age group")
    print("=" * 80)

    for c in range(model.num_cities):
        print(f"\nCity {c + 1}:")
        for a in range(model.num_ages):
            series = I_reshaped[:, c, a]
            peak_idx = np.argmax(series)
            peak_time = t[peak_idx]
            peak_value = series[peak_idx]
            print(f"  Age group {a + 1}: peak day = {int(peak_time)}, peak infections = {peak_value:.2f}")

    print("\n" + "=" * 80)
    print("Peak total infection times by city")
    print("=" * 80)

    I_city = I_reshaped.sum(axis=2)   

    for c in range(model.num_cities):
        series = I_city[:, c]
        peak_idx = np.argmax(series)
        peak_time = t[peak_idx]
        peak_value = series[peak_idx]
        print(f"City {c + 1}: peak day = {int(peak_time)}, peak infections = {peak_value:.2f}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Generate this synthetic scenario in a new directory outside the package.")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    output = options.output.absolute().resolve()
    package = Path(__file__).resolve().parents[3]
    if output == package or package in output.parents:
        parser.error("Generated data must be outside the package")
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    if any(path.is_symlink() for path in (options.output.absolute(), *options.output.absolute().parents)):
        parser.error("Symlinked output paths are not supported")
    model = MixedEulerianLagrangianEpidemicModel()
    model.output_dir = str(output)
    model.generate_dataset(t_max=200)
    print_peak_times(model)
    plot_all_results(model)
