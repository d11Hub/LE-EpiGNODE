
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


LAGRANGIAN_TYPE = "time_decay"


EULERIAN_TYPE = "baseline"


LAGRANGIAN_LABELS = {
    "bilinear": "Bilinear",
    "time_decay": "Time-decaying",
    "exponential": "Inhibitory",
    "nonlinear": "Nonlinear",
}

EULERIAN_LABELS = {
    "baseline": "Baseline gravity migration",
    "controlled": "Controlled gravity migration",
}


mpl.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams["axes.unicode_minus"] = False


class MixedEulerianLagrangianEpidemicModel:
    def __init__(
        self,
        lagrangian_type=LAGRANGIAN_TYPE,
        eulerian_type=EULERIAN_TYPE,
    ):

        


        if lagrangian_type not in LAGRANGIAN_LABELS:
            raise ValueError(
                f"Unknown Lagrangian mechanism: {lagrangian_type}; "
                f"available values: {list(LAGRANGIAN_LABELS)}"
            )

        if eulerian_type not in EULERIAN_LABELS:
            raise ValueError(
                f"Unknown Eulerian mechanism: {eulerian_type}; "
                f"available values: {list(EULERIAN_LABELS)}"
            )

        self.lagrangian_type = lagrangian_type
        self.eulerian_type = eulerian_type

        self.lagrangian_label = LAGRANGIAN_LABELS[lagrangian_type]
        self.eulerian_label = EULERIAN_LABELS[eulerian_type]


        


        self.num_cities = 4
        self.num_ages = 4
        self.k = self.num_cities * self.num_ages

        self.output_dir = (
            f"Dataset_LE_4City_4Age"
        )

        self.use_GE = self.num_cities > 1


        


        self.beta = 0.02
        self.gamma = 0.021


        
        self.alpha_E = 5000.0

        self.re_factor = 1e-4


        


        
        self.r_L = 0.0035


        
        self.alpha_I = 0.045


        
        self.p_S = 1.0
        self.q_I = 0.2


        


        
        self.r_E = 0.03


        


        self.eta = np.array(
            [0.106, 0.685, 0.156, 0.053],
            dtype=float,
        )


        


        self.dist_matrix = np.array(
            [
                [0.0, 120.0, 170.0, 310.0],
                [120.0, 0.0, 130.0, 220.0],
                [170.0, 130.0, 0.0, 160.0],
                [310.0, 220.0, 160.0, 0.0],
            ],
            dtype=float,
        )

        assert self.dist_matrix.shape == (
            self.num_cities,
            self.num_cities,
        )


        


        pop_city_raw = np.array(
            [
                [120000, 165000, 190000, 115000],
                [110000, 155000, 180000, 105000],
                [100000, 145000, 170000, 95000],
                [115000, 160000, 185000, 110000],
            ],
            dtype=float,
        )


        
        I0_city_raw = np.array(
            [
                [1900, 1000, 1200, 1300],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )

        assert pop_city_raw.shape == (
            self.num_cities,
            self.num_ages,
        )

        assert I0_city_raw.shape == (
            self.num_cities,
            self.num_ages,
        )

        self.pop_city_raw = pop_city_raw.copy()
        self.I0_city_raw = I0_city_raw.copy()


        self.static_pop_city_raw = pop_city_raw.sum(axis=1)

        self.S0 = (
            pop_city_raw - I0_city_raw
        ).reshape(-1) * self.re_factor

        self.I0 = (
            I0_city_raw.reshape(-1)
            * self.re_factor
        )

        self.R0 = np.zeros(
            self.k,
            dtype=float,
        )


        


        self.alpha_grav = 1.0
        self.beta_grav = 1.0
        self.lambda_grav = 0.012
        self.kappa_grav = 100.0
        self.pop_ref = 1e6


        self.rho_src = 6.0
        self.rho_dst = 10.0


        


        C_base = 0.5 * np.array(
            [
                [3.20, 2.92, 2.79, 3.17],
                [3.10, 3.00, 2.84, 3.13],
                [3.21, 3.16, 2.90, 3.31],
                [3.23, 2.90, 2.74, 3.00],
            ],
            dtype=float,
        )

        self.contact_mats = [
            C_base.copy(),
            C_base.copy(),
            C_base.copy(),
            C_base.copy(),
        ]

        self.city_contact_scale = np.array(
            [1.00, 0.98, 0.95, 0.92],
            dtype=float,
        )


    


    def compute_baidu_migration(
        self,
        t,
        gammaI_city_raw,
    ):
        """
        B_lk =
            kappa
            * (P_l / pop_ref)^alpha
            * (P_k / pop_ref)^beta
            * exp(-lambda * d_lk)
            * exp(
                -rho_src * gammaI_l / P_l
                -rho_dst * gammaI_k / P_k
            )
            * time_factor
        """

        if not self.use_GE:
            return np.zeros(
                (
                    self.num_cities,
                    self.num_cities,
                ),
                dtype=float,
            )

        B_matrix = np.zeros(
            (
                self.num_cities,
                self.num_cities,
            ),
            dtype=float,
        )


        
        if self.eulerian_type == "baseline":
            time_factor = 1.0

        elif self.eulerian_type == "controlled":
            time_factor = np.exp(
                -self.r_E * t
            )

        else:
            raise ValueError(
                f"Unknown Eulerian mechanism: "
                f"{self.eulerian_type}"
            )

        for l in range(self.num_cities):
            for k in range(self.num_cities):
                if l == k:
                    continue

                dist_lk = self.dist_matrix[l, k]

                P_l = self.static_pop_city_raw[l]
                P_k = self.static_pop_city_raw[k]

                P_l_norm = P_l / self.pop_ref
                P_k_norm = P_k / self.pop_ref

                gravity_lk = (
                    (P_l_norm ** self.alpha_grav)
                    * (P_k_norm ** self.beta_grav)
                    * np.exp(
                        -self.lambda_grav
                        * dist_lk
                    )
                )

                dens_l = (
                    gammaI_city_raw[l]
                    / (P_l + 1e-12)
                )

                dens_k = (
                    gammaI_city_raw[k]
                    / (P_k + 1e-12)
                )

                epidemic_factor = np.exp(
                    -self.rho_src * dens_l
                    -self.rho_dst * dens_k
                )

                B_matrix[l, k] = (
                    self.kappa_grav
                    * gravity_lk
                    * epidemic_factor
                    * time_factor
                )

        return B_matrix

    def compute_GE_from_baidu(
            self,
            B_matrix,
    ):
        "Construct age-level GE from the supplied raw or noisy city mobility matrix."

        G_E = np.zeros(
            (
                self.k,
                self.k,
            ),
            dtype=float,
        )

        if not self.use_GE:
            return G_E

        eta_share = (
                self.eta
                / (
                        np.sum(self.eta)
                        + 1e-12
                )
        )

        for l in range(self.num_cities):
            P_l = (
                    self.static_pop_city_raw[l]
                    + 1e-12
            )

            for k in range(self.num_cities):
                if l == k:
                    continue

                M_lk = (
                        self.alpha_E
                        * B_matrix[l, k]
                )

                base_rate = M_lk / P_l

                for a in range(self.num_ages):
                    i = (
                            l * self.num_ages
                            + a
                    )

                    j = (
                            k * self.num_ages
                            + a
                    )

                    G_E[i, j] = (
                            eta_share[a]
                            * base_rate
                    )

        np.fill_diagonal(
            G_E,
            0.0,
        )

        row_sums = G_E.sum(axis=1)

        np.fill_diagonal(
            G_E,
            -row_sums,
        )

        return G_E

    def aggregate_GE_city_series(
            self,
            G_E_age_series,
    ):
        "Aggregate same-age intercity GE entries to the city level."

        T_len = G_E_age_series.shape[0]

        G_E_city_series = np.zeros(
            (
                T_len,
                self.num_cities,
                self.num_cities,
            ),
            dtype=float,
        )

        for l in range(self.num_cities):
            for k in range(self.num_cities):
                if l == k:
                    continue

                for a in range(self.num_ages):
                    G_E_city_series[
                    :,
                    l,
                    k,
                    ] += G_E_age_series[
                         :,
                         l * self.num_ages + a,
                         k * self.num_ages + a,
                         ]

        for t_idx in range(T_len):
            np.fill_diagonal(
                G_E_city_series[t_idx],
                0.0,
            )

            row_sums = (
                G_E_city_series[t_idx]
                .sum(axis=1)
            )

            np.fill_diagonal(
                G_E_city_series[t_idx],
                -row_sums,
            )

        return G_E_city_series


    


    def add_gaussian_noise(
        self,
        data,
        noise_level=0.05,
        seed=2026,
        nonnegative=True,
    ):
        rng = np.random.default_rng(seed)

        data = np.asarray(
            data,
            dtype=float,
        )

        scale = np.max(
            np.abs(data),
            axis=0,
            keepdims=True,
        )

        scale = np.maximum(
            scale,
            1e-12,
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_level * scale,
            size=data.shape,
        )

        noisy_data = data + noise

        if nonnegative:
            noisy_data = np.clip(
                noisy_data,
                0.0,
                None,
            )

        return noisy_data


    


    def compute_operators(
        self,
        t,
        y,
    ):
        'Initialize model settings and input tensors.'

        X_S = y[:self.k]
        X_I = y[self.k:2 * self.k]
        X_R = y[2 * self.k:3 * self.k]


        


        pop_age = X_S + X_I + X_R

        I_age_raw = (
            X_I / self.re_factor
        )

        I_city_raw = I_age_raw.reshape(
            self.num_cities,
            self.num_ages,
        ).sum(axis=1)

        gammaI_city_raw = (
            self.gamma * I_city_raw
        )


        


        G_L = np.zeros(
            (
                self.k,
                self.k,
            ),
            dtype=float,
        )

        for c in range(self.num_cities):
            idx = slice(
                c * self.num_ages,
                (c + 1) * self.num_ages,
            )

            S_city = np.maximum(
                X_S[idx],
                0.0,
            )

            I_city = np.maximum(
                X_I[idx],
                0.0,
            )

            pop_city_age = np.maximum(
                pop_age[idx],
                1e-12,
            )

            C_city = (
                self.contact_mats[c]
                * self.city_contact_scale[c]
            )


            


            if self.lagrangian_type == "bilinear":

                G_L_block_raw = (
                    self.beta
                    * C_city
                    @ np.diag(S_city)
                )

            elif self.lagrangian_type == "time_decay":

                beta_t = (
                    self.beta
                    * np.exp(-self.r_L * t)
                )

                G_L_block_raw = (
                    beta_t
                    * C_city
                    @ np.diag(S_city)
                )

            elif self.lagrangian_type == "exponential":

                suppression = np.exp(
                    -self.alpha_I * I_city
                )

                G_L_block_raw = (
                    self.beta
                    * (
                        suppression[:, None]
                        * C_city
                    )
                    @ np.diag(S_city)
                )

            elif self.lagrangian_type == "nonlinear":

                G_L_block_raw = (
                    self.beta
                    * (
                        (
                            I_city ** self.q_I
                        )[:, None]
                        * C_city
                    )
                    @ np.diag(
                        S_city ** self.p_S
                    )
                )

            else:
                raise ValueError(
                    f"Unknown Lagrangian mechanism: "
                    f"{self.lagrangian_type}"
                )

            G_L_block = (
                G_L_block_raw
                / pop_city_age[:, None]
            )

            G_L[idx, idx] = G_L_block


        


        if self.use_GE:
            B_matrix = (
                self.compute_baidu_migration(
                    t,
                    gammaI_city_raw,
                )
            )
        else:
            B_matrix = np.zeros(
                (
                    self.num_cities,
                    self.num_cities,
                ),
                dtype=float,
            )


        


        G_E = self.compute_GE_from_baidu(
            B_matrix
        )

        return G_L, G_E, B_matrix


    


    def sir_rhs(
        self,
        t,
        y,
    ):

        
        G_L, G_E, _ = (
            self.compute_operators(
                t,
                y,
            )
        )

        X_S = y[:self.k]
        X_I = y[self.k:2 * self.k]
        X_R = y[2 * self.k:3 * self.k]

        infection = G_L.T @ X_I

        dX_S = (
            G_E.T @ X_S
            - infection
        )

        dX_I = (
            G_E.T @ X_I
            + infection
            - self.gamma * X_I
        )

        dX_R = (
            G_E.T @ X_R
            + self.gamma * X_I
        )

        return np.concatenate(
            [
                dX_S,
                dX_I,
                dX_R,
            ]
        )


    


    def generate_dataset(
        self,
        t_max=200,
    ):
        print("=" * 70)
        print(
            f"Generating: "
            f"{self.lagrangian_label}"
            f" + "
            f"{self.eulerian_label}"
        )
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)

        t_span = (
            0.0,
            float(t_max),
        )

        t_eval = np.arange(
            0,
            t_max + 1,
            1,
            dtype=float,
        )

        X0 = np.concatenate(
            [
                self.S0,
                self.I0,
                self.R0,
            ]
        )

        solution = solve_ivp(
            fun=self.sir_rhs,
            t_span=t_span,
            y0=X0,
            t_eval=t_eval,
            method="LSODA",
            rtol=1e-6,
            atol=1e-9,
        )

        if not solution.success:
            raise RuntimeError(
                f"ODE integration failed: "
                f"{solution.message}"
            )


        solution.y[:, 0] = X0

        y_scaled = solution.y
        self.t = solution.t

        self.S = (
            y_scaled[:self.k, :].T
        )

        self.I_true = (
            y_scaled[
                self.k:2 * self.k,
                :
            ].T
        )

        self.R = (
            y_scaled[
                2 * self.k:3 * self.k,
                :
            ].T
        )


        self.I = self.add_gaussian_noise(
            self.I_true,
            noise_level=0.05,
            seed=42,
            nonnegative=True,
        )

        self.gammaI = (
                self.gamma * self.I
        )


        

        

        Baidu_true_list = []
        G_L_true_list = []

        for idx, t in enumerate(self.t):
            G_L_t, _, _ = self.compute_operators(
                t,
                y_scaled[:, idx]
            )

            G_L_true_list.append(G_L_t)
            X_I_true = y_scaled[
                       self.k:2 * self.k,
                       idx,
                       ]

            I_age_raw = (
                    X_I_true
                    / self.re_factor
            )

            I_city_raw = I_age_raw.reshape(
                self.num_cities,
                self.num_ages,
            ).sum(axis=1)

            gammaI_city_raw = (
                    self.gamma
                    * I_city_raw
            )

            B_t = self.compute_baidu_migration(
                t,
                gammaI_city_raw,
            )

            Baidu_true_list.append(B_t)

        Baidu_true_series = np.stack(
            Baidu_true_list,
            axis=0,
        )
        self.G_L_series = np.stack(
            G_L_true_list,
            axis=0,
        )

        
        self.true_GL_by_city = np.stack(
            [
                self.G_L_series[
                :,
                c * self.num_ages:(c + 1) * self.num_ages,
                c * self.num_ages:(c + 1) * self.num_ages,
                ]
                for c in range(self.num_cities)
            ],
            axis=1,
        )


        

        

        self.Baidu_series = self.add_gaussian_noise(
            Baidu_true_series,
            noise_level=0.05,
            seed=43,
            nonnegative=True,
        )


        for t_idx in range(len(self.t)):
            np.fill_diagonal(
                self.Baidu_series[t_idx],
                0.0,
            )


        


        GE_age_list = []

        for t_idx in range(len(self.t)):
            G_E_t = self.compute_GE_from_baidu(
                self.Baidu_series[t_idx]
            )

            GE_age_list.append(G_E_t)

        self.G_E_age_series = np.stack(
            GE_age_list,
            axis=0,
        )

        self.G_E_city_series = self.aggregate_GE_city_series(
            self.G_E_age_series
        )


        


        self.save_data()

        print(
            f"Data saved to: "
            f"{self.output_dir}"
        )


    


    def save_data(self):
        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        np.save(
            os.path.join(
                self.output_dir,
                "G_L_series.npy",
            ),
            self.G_L_series,
        )

        np.save(
            os.path.join(
                self.output_dir,
                "true_GL_by_city.npy",
            ),
            self.true_GL_by_city,
        )

        


        np.save(
            os.path.join(
                self.output_dir,
                "time.npy",
            ),
            self.t,
        )

        


        np.save(
            os.path.join(
                self.output_dir,
                "Baidu_Migration_noisy_series.npy",
            ),
            self.Baidu_series,
        )


        


        np.save(
            os.path.join(
                self.output_dir,
                "G_E_age_from_noisy_Baidu_series.npy",
            ),
            self.G_E_age_series,
        )

        np.save(
            os.path.join(
                self.output_dir,
                "G_E_city_from_noisy_Baidu_series.npy",
            ),
            self.G_E_city_series,
        )


        

        

        

        

        states_age_noisy = np.stack(
            [
                self.S,
                self.I,
                self.R,
                self.gammaI,
            ],
            axis=-1,
        )

        np.save(
            os.path.join(
                self.output_dir,
                "states_age_noisy.npy",
            ),
            states_age_noisy,
        )


        


        T_len = len(self.t)

        S_city = self.S.reshape(
            T_len,
            self.num_cities,
            self.num_ages,
        ).sum(axis=2)

        I_city = self.I.reshape(
            T_len,
            self.num_cities,
            self.num_ages,
        ).sum(axis=2)

        R_city = self.R.reshape(
            T_len,
            self.num_cities,
            self.num_ages,
        ).sum(axis=2)

        gammaI_city = self.gammaI.reshape(
            T_len,
            self.num_cities,
            self.num_ages,
        ).sum(axis=2)

        states_city_noisy = np.stack(
            [
                S_city,
                I_city,
                R_city,
                gammaI_city,
            ],
            axis=-1,
        )

        np.save(
            os.path.join(
                self.output_dir,
                "states_city_noisy.npy",
            ),
            states_city_noisy,
        )


        


        states_age_real = (
                states_age_noisy
                / self.re_factor
        )

        age_excel_path = os.path.join(
            self.output_dir,
            "noisy_age_time_series.xlsx",
        )

        with pd.ExcelWriter(
                age_excel_path
        ) as writer:
            for node in range(self.k):
                city = (
                        node // self.num_ages
                        + 1
                )

                age = (
                        node % self.num_ages
                        + 1
                )

                df_age = pd.DataFrame(
                    states_age_real[
                    :,
                    node,
                    :
                    ],
                    columns=[
                        "S",
                        "I_noisy",
                        "R",
                        "gammaI_noisy",
                    ],
                )


                df_age.insert(
                    0,
                    "time",
                    self.t,
                )

                df_age.to_excel(
                    writer,
                    index=False,
                    sheet_name=(
                        f"city_{city}_age_{age}"
                    ),
                )


        


        city_excel_path = os.path.join(
            self.output_dir,
            "noisy_city_time_series.xlsx",
        )

        with pd.ExcelWriter(
                city_excel_path
        ) as writer:
            for city in range(
                    self.num_cities
            ):
                df_city = pd.DataFrame(
                    states_city_noisy[
                    :,
                    city,
                    :
                    ]
                    / self.re_factor,
                    columns=[
                        "S",
                        "I_noisy",
                        "R",
                        "gammaI_noisy",
                    ],
                )

                df_city.insert(
                    0,
                    "time",
                    self.t,
                )

                df_city.to_excel(
                    writer,
                    index=False,
                    sheet_name=(
                        f"city_{city + 1}"
                    ),
                )

        


        T_len = len(self.t)

        mob_columns = [
            f"{i + 1}->{j + 1}"
            for i in range(self.num_cities)
            for j in range(self.num_cities)
        ]

        B_flat = self.Baidu_series.reshape(
            T_len,
            self.num_cities * self.num_cities,
        )

        df_baidu = pd.DataFrame(
            B_flat,
            columns=mob_columns,
        )

        df_baidu.insert(
            0,
            "time",
            self.t,
        )

        df_baidu.to_excel(
            os.path.join(
                self.output_dir,
                "noisy_Baidu_mobility.xlsx",
            ),
            index=False,
        )


        


        GE_city_flat = self.G_E_city_series.reshape(
            T_len,
            self.num_cities * self.num_cities,
        )

        df_ge_city = pd.DataFrame(
            GE_city_flat,
            columns=mob_columns,
        )

        df_ge_city.insert(
            0,
            "time",
            self.t,
        )

        df_ge_city.to_excel(
            os.path.join(
                self.output_dir,
                "GE_city_from_noisy_Baidu.xlsx",
            ),
            index=False,
        )


        


        edge_columns = [
            f"{i + 1}->{j + 1}"
            for i in range(self.k)
            for j in range(self.k)
        ]

        GE_age_flat = self.G_E_age_series.reshape(
            T_len,
            self.k * self.k,
        )

        df_ge_age = pd.DataFrame(
            GE_age_flat,
            columns=edge_columns,
        )

        df_ge_age.insert(
            0,
            "time",
            self.t,
        )

        df_ge_age.to_excel(
            os.path.join(
                self.output_dir,
                "GE_age_from_noisy_Baidu.xlsx",
            ),
            index=False,
        )

        


        gl_city_path = os.path.join(
            self.output_dir,
            "true_GL_by_city.xlsx",
        )

        with pd.ExcelWriter(
            gl_city_path
        ) as writer:
            for c in range(self.num_cities):

                GL_city_block = self.true_GL_by_city[
                    :,
                    c,
                    :,
                    :,
                ]

                gl_columns = [
                    f"age_{i + 1}->age_{j + 1}"
                    for i in range(self.num_ages)
                    for j in range(self.num_ages)
                ]

                GL_city_flat = GL_city_block.reshape(
                    len(self.t),
                    self.num_ages * self.num_ages,
                )

                df_gl_city = pd.DataFrame(
                    GL_city_flat,
                    columns=gl_columns,
                )

                df_gl_city.insert(
                    0,
                    "time",
                    self.t,
                )

                df_gl_city.to_excel(
                    writer,
                    index=False,
                    sheet_name=f"city_{c + 1}",
                )


        

        

        

        ge_age_path = os.path.join(
            self.output_dir,
            "true_GE_by_age.xlsx",
        )

        with pd.ExcelWriter(
            ge_age_path
        ) as writer:
            for a in range(self.num_ages):

                city_indices = [
                    c * self.num_ages + a
                    for c in range(self.num_cities)
                ]

                GE_age_block = self.G_E_age_series[
                    :,
                    city_indices,
                    :,
                ][
                    :,
                    :,
                    city_indices,
                ]

                ge_columns = [
                    f"city_{i + 1}->city_{j + 1}"
                    for i in range(self.num_cities)
                    for j in range(self.num_cities)
                ]

                GE_age_flat = GE_age_block.reshape(
                    len(self.t),
                    self.num_cities * self.num_cities,
                )

                df_ge_age_by_age = pd.DataFrame(
                    GE_age_flat,
                    columns=ge_columns,
                )

                df_ge_age_by_age.insert(
                    0,
                    "time",
                    self.t,
                )

                df_ge_age_by_age.to_excel(
                    writer,
                    index=False,
                    sheet_name=f"age_{a + 1}",
                )


def plot_operator_series(
    t,
    G_series,
    title_prefix,
    skip_diag=False,
):
    _, N, _ = G_series.shape

    fig, axes = plt.subplots(
        N,
        N,
        figsize=(3 * N, 3 * N),
        gridspec_kw={
            "wspace": 0.05,
            "hspace": 0.05,
        },
        sharex=True,
        sharey=True,
    )

    if N == 1:
        axes = np.array([[axes]])

    if skip_diag:
        mask = ~np.eye(
            N,
            dtype=bool,
        )

        vals = G_series[:, mask]
        max_val = np.max(vals)
        min_val = np.min(vals)

    else:
        max_val = np.max(G_series)
        min_val = np.min(G_series)

    for i in range(N):
        for j in range(N):
            ax = axes[i, j]

            if not (
                skip_diag
                and i == j
            ):
                color = (
                    "red"
                    if i == j
                    else "green"
                )

                ax.plot(
                    t,
                    G_series[:, i, j],
                    "-",
                    color=color,
                    linewidth=1.5,
                )

            pad = (
                0.05
                * (
                    max_val
                    - min_val
                    + 1e-12
                )
            )

            ax.set_ylim(
                min_val - pad,
                max_val + pad,
            )

            ax.tick_params(
                direction="out",
                length=3,
                width=1.0,
                labelsize=7,
                bottom=True,
                left=(j == 0),
                labelbottom=(i == N - 1),
                labelleft=(j == 0),
            )

    fig.text(
        0.5,
        0.02,
        "Time (days)",
        ha="center",
        fontsize=14,
    )

    fig.text(
        0.02,
        0.5,
        f"{title_prefix} element values",
        ha="center",
        rotation="vertical",
        fontsize=14,
    )

    plt.suptitle(
        f"{title_prefix} over time",
        y=0.98,
        fontsize=16,
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.92,
        top=0.90,
        bottom=0.08,
    )

    plt.show()


def plot_all_results(model):
    t = model.t


    I_noisy_real = (
        model.I
        / model.re_factor
    )

    I_reshaped = I_noisy_real.reshape(
        len(t),
        model.num_cities,
        model.num_ages,
    )


    


    plt.figure(figsize=(10, 6))

    I_city = I_reshaped.sum(axis=2)

    for city in range(
        model.num_cities
    ):
        plt.plot(
            t,
            I_city[:, city],
            label=f"City {city + 1}",
            linewidth=1.5,
        )

    plt.title(
        (
            f"{model.lagrangian_label}"
            f" + "
            f"{model.eulerian_label}\n"
            f"Noisy total infected population by city"
        ),
        fontsize=15,
    )

    plt.xlabel("Time (days)")
    plt.ylabel("Noisy infected population (people)")
    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.6,
    )

    plt.tight_layout()
    plt.show()


    


    for city in range(
        model.num_cities
    ):
        plt.figure(figsize=(10, 5))

        for age in range(
            model.num_ages
        ):
            plt.plot(
                t,
                I_reshaped[
                    :,
                    city,
                    age,
                ],
                label=(
                    f"Age group {age + 1}"
                ),
                linewidth=1.5,
            )

        plt.title(
            (
                f"{model.lagrangian_label}"
                f" + "
                f"{model.eulerian_label}\n"
                f"City {city + 1}"
                f"Noisy infected population by age group"
            )
        )

        plt.xlabel("Time (days)")
        plt.ylabel("Noisy infected population (people)")
        plt.legend()

        plt.grid(
            True,
            linestyle="--",
            alpha=0.6,
        )

        plt.tight_layout()
        plt.show()

        


        plot_operator_series(
            t,
            model.Baidu_series,
            "Noisy mobility index matrix",
            skip_diag=True,
        )


        


        plot_operator_series(
            t,
            model.G_E_city_series,
            "City-level G_E operator from noisy mobility",
            skip_diag=False,
        )


def print_peak_times(model):
    t = model.t

    I_real = (
            model.I
            / model.re_factor
    )

    I_reshaped = I_real.reshape(
        len(t),
        model.num_cities,
        model.num_ages,
    )

    print("\n" + "=" * 80)
    print(
        f"{model.lagrangian_label}"
        f" + "
        f"{model.eulerian_label}"
    )
    print("Peak infections by city and age group")
    print("=" * 80)

    for city in range(
        model.num_cities
    ):
        print(
            f"\nCity {city + 1}: "
        )

        for age in range(
            model.num_ages
        ):
            series = I_reshaped[
                :,
                city,
                age,
            ]

            peak_idx = int(
                np.argmax(series)
            )

            print(
                f"  Age group {age + 1}: "
                f"Peak time = day "
                f"{int(t[peak_idx])}; "
                f"peak infected population = "
                f"{series[peak_idx]:.2f}"
            )

    print("\n" + "=" * 80)
    print("Peak total infections by city")
    print("=" * 80)

    I_city = I_reshaped.sum(axis=2)

    for city in range(
        model.num_cities
    ):
        series = I_city[:, city]

        peak_idx = int(
            np.argmax(series)
        )

        print(
            f"City {city + 1}: "
            f"Peak time = day "
            f"{int(t[peak_idx])}; "
            f"peak infected population = "
            f"{series[peak_idx]:.2f}"
        )


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
    model = MixedEulerianLagrangianEpidemicModel(
        lagrangian_type=LAGRANGIAN_TYPE,
        eulerian_type=EULERIAN_TYPE,
    )
    model.output_dir = str(output)

    model.generate_dataset(
        t_max=200
    )

    print_peak_times(model)

    plot_all_results(model)
