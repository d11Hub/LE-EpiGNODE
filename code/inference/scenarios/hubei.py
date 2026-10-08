"""Checkpoint initialization for hubei."""
import os
import random
import numpy as np
import pandas as pd
import torch
from models.real.hubei import LE_EpiGNN as BaseModel
LE_EpiGNN = BaseModel

DEFAULTS = {'dataset': 'Hubei_3City_1Age', 'num_cities': 3, 'num_ages': 1, 'loss_level': 'age', 'training_time': 15, 'gl_hidden_dim': 64, 'gl_num_layers': 10, 'c_hidden_dim': 9, 'random_seed': 9, 're_factor': 0.0001, 'gamma_init': '0.10,0.78,0.76', 'sigma': 1.0 / 5.2, 'alpha_E_init': 22500.0, 'eps_E_init': 0.0001, 'pop_ref': 5000000.0, 'mob_hidden_dim': 256, 'mob_num_layers': 15, 'mob_ckpt_path': '', 'S0_lr_factor': 0.25, 'E0_lr_factor': 1.5, 'I0_lr_factor': 1.5, 'gamma_lr_factor': 3.0, 'contact_values': '65,14,66', 's0_ratio': 1.0, 'i0_min': 0.0, 's0_values': '120000,10000,9000', 'e0_values': '22,3,5', 'i0_values': '520,1,1', 'r0_values': '0, 0, 0', 'rho_src': 3.0, 'rho_dst': 10.0, 'alpha_grav': 0.2, 'beta_grav': 1.8, 'lambda_grav': 0.003}

def expand_float_list(value, n, name):
    """Broadcast one float to n entries, accept n entries, and reject other lengths."""
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [float(x) for x in value]
    else:
        vals = [float(x.strip()) for x in str(value).split(',') if x.strip()]
    if len(vals) == 1:
        vals = vals * n
    elif len(vals) != n:
        raise ValueError(f'{name} must have length 1 or num_cities={n}; got {len(vals)}: {vals}')
    return vals

def build_model(args):
    data_dir = args.data_dir
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data_dir = args.data_dir
    data_path = os.path.join(data_dir, 'states_age.npy')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing required input: {data_path}")
    datas_raw = np.load(data_path).astype(np.float32)
    datas = datas_raw * args.re_factor
    t = torch.arange(datas.shape[0])
    T, N, D = datas.shape
    args.num_ages = 1
    args.num_cities = N // args.num_ages
    city_order_path = os.path.join(data_dir, 'city_population_order.csv')
    if not os.path.exists(city_order_path):
        raise FileNotFoundError(f"Missing required input: {city_order_path}")
    city_info = pd.read_csv(city_order_path)
    args.city_names = city_info['city'].astype(str).tolist()
    pop_vec = city_info['population'].astype(float).values.astype(np.float32)
    if len(args.city_names) != args.num_cities:
        raise ValueError(f'city_population_order.csv has {len(args.city_names)} cities, but states_age.npy has {args.num_cities} cities')
    args.static_pop_city_raw = pop_vec.tolist()
    args.gamma_init = expand_float_list(args.gamma_init, args.num_cities, 'gamma_init')
    contact_values = expand_float_list(args.contact_values, args.num_cities, 'contact_values')
    dist_path = os.path.join(data_dir, 'dist_mat.npy')
    if not os.path.exists(dist_path):
        raise FileNotFoundError(f"Missing required input: {dist_path}")
    dist_mat = np.load(dist_path).astype(np.float32)
    if dist_mat.shape != (args.num_cities, args.num_cities):
        raise ValueError(f'dist_mat must have shape {(args.num_cities, args.num_cities)}; got {dist_mat.shape}')
    C_mat = np.zeros((N, N), dtype=np.float32)
    for c in range(args.num_cities):
        idx0 = c * args.num_ages
        idx1 = (c + 1) * args.num_ages
        C_mat[idx0:idx1, idx0:idx1] = contact_values[c] * np.eye(args.num_ages, dtype=np.float32)
    gamma_init_arr = np.array(args.gamma_init, dtype=np.float32)
    sigma_value = max(float(args.sigma), 1e-06)
    gammaI0_raw = datas_raw[0, :, -1].astype(np.float32)
    if args.i0_values.strip():
        I0_input = np.array([float(x.strip()) for x in args.i0_values.split(',') if x.strip()], dtype=np.float32)
        if len(I0_input) == 1:
            I0_input = np.repeat(I0_input, args.num_cities)
        elif len(I0_input) != args.num_cities:
            raise ValueError(f'i0_values must have length 1 or {args.num_cities}; got {len(I0_input)}')
    else:
        I0_input = gammaI0_raw / np.maximum(gamma_init_arr, 1e-06)
        I0_input = np.maximum(I0_input, args.i0_min)
    I0_input = np.maximum(I0_input, 0.0).astype(np.float32)
    if args.e0_values.strip():
        E0_input = np.array([float(x.strip()) for x in args.e0_values.split(',') if x.strip()], dtype=np.float32)
        if len(E0_input) == 1:
            E0_input = np.repeat(E0_input, args.num_cities)
        elif len(E0_input) != args.num_cities:
            raise ValueError(f'e0_values must have length 1 or {args.num_cities}; got {len(E0_input)}')
    else:
        E0_input = gammaI0_raw / sigma_value
    E0_input = np.maximum(E0_input, 0.0).astype(np.float32)
    if args.r0_values.strip():
        R0_input = np.array([float(x.strip()) for x in args.r0_values.split(',') if x.strip()], dtype=np.float32)
        if len(R0_input) == 1:
            R0_input = np.repeat(R0_input, args.num_cities)
        elif len(R0_input) != args.num_cities:
            raise ValueError(f'r0_values must have length 1 or {args.num_cities}; got {len(R0_input)}')
    else:
        R0_input = np.zeros(args.num_cities, dtype=np.float32)
    R0_input = np.maximum(R0_input, 0.0).astype(np.float32)
    if args.s0_values.strip():
        S0_input = np.array([float(x.strip()) for x in args.s0_values.split(',') if x.strip()], dtype=np.float32)
        if len(S0_input) == 1:
            S0_input = np.repeat(S0_input, args.num_cities)
        elif len(S0_input) != args.num_cities:
            raise ValueError(f's0_values must have length 1 or {args.num_cities}; got {len(S0_input)}')
    else:
        S0_input = pop_vec * args.s0_ratio - E0_input - I0_input - R0_input
    S0_input = np.maximum(S0_input, 1.0).astype(np.float32)
    U_0 = np.stack([S0_input, E0_input, I0_input, R0_input], axis=1) * args.re_factor
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat)
    return LE_model
