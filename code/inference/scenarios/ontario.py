"""Checkpoint initialization for ontario."""
import os
import random
import numpy as np
import pandas as pd
import torch
from models.real.ontario import LE_EpiGNN as BaseModel
LE_EpiGNN = BaseModel

DEFAULTS = {'dataset': 'Dataset_LE_1City_4Age', 'num_cities': 1, 'num_ages': 4, 'ode_rtol': 1e-05, 'ode_atol': 1e-08, 'random_seed': 3, 'gl_num_layers': 3, 'gl_hidden_dim': 64, 'c_hidden_dim': 6, 're_factor': 1.0283875e-05, 'gamma': [0.24691881, 0.43754812, 0.18519953, 0.09355255], 'sigma': [1 / 3.61, 1 / 3.61, 1 / 3.61, 1 / 3.61], 'pop_ref': 1000000.0, 'mob_hidden_dim': 128, 'mob_num_layers': 4, 'mob_ckpt_path': '', 'manual_S0': [73457.56, 246495.61, 135706.66, 46018.7], 'manual_E0': [1420.41, 1713.03, 1029.36, 714.62], 'manual_I0': [2079.65, 1659.51, 685.86, 327.14], 'manual_R0': None}

def load_ontario_prepared_data(args):
    'Load the dataset arrays.'
    data_dir = args.data_dir
    datas = np.load(os.path.join(data_dir, 'states_age.npy')).astype(np.float32)
    C_mat = np.load(os.path.join(data_dir, 'contact_matrix.npy')).astype(np.float32)
    group_pop = np.load(os.path.join(data_dir, 'group_population.npy')).astype(np.float32)
    dates = pd.to_datetime(pd.read_csv(os.path.join(data_dir, 'aligned_dates.csv'))['date'])
    group_names = ['0-19', '20-39', '40-59', '60+']
    if datas.ndim != 3 or datas.shape[1:] != (4, 5) or C_mat.shape != (4, 4):
        raise ValueError('Ontario processed arrays have unexpected dimensions')
    if len(dates) != len(datas) or group_pop.shape != (4,):
        raise ValueError('Ontario processed dates or population do not match the states')
    args.num_ages = len(group_names)
    r0 = args.manual_R0 if args.manual_R0 is not None else [0.0] * len(group_names)
    U_0 = np.stack([args.manual_S0, args.manual_E0, args.manual_I0, r0], axis=1).astype(np.float32)
    U_0 *= args.re_factor
    return (datas, C_mat, group_names, group_pop, dates, U_0)

def build_model(args):
    data_dir = args.data_dir
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    datas, C_mat, group_names, group_pop, dates, U_0 = load_ontario_prepared_data(args)
    t = torch.arange(datas.shape[0])
    T, N, D = datas.shape
    args.num_cities = 1
    args.num_ages = len(group_names)
    dist_mat = np.zeros((args.num_cities, args.num_cities), dtype=np.float32)
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat)
    return LE_model
