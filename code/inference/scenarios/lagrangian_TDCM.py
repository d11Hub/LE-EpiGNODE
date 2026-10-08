"""Checkpoint initialization for lagrangian_TDCM."""
import os
import random
import numpy as np
import pandas as pd
import torch
from models.synthetic.lagrangian import LE_EpiGNN as BaseModel

class LE_EpiGNN(BaseModel):
    SCENARIO = 'lagrangian_TDCM'

DEFAULTS = {'dataset': 'Dataset_LE_1City_4Age', 'num_cities': 1, 'num_ages': 4, 'random_seed': 10, 'gl_num_layers': 6, 'gl_hidden_dim': 64, 'c_hidden_dim': 5, 're_factor': 0.0001, 'gamma': 0.02, 'pop_ref': 1000000.0, 'mob_hidden_dim': 128, 'mob_num_layers': 4, 'mob_ckpt_path': ''}

def build_model(args):
    data_dir = args.data_dir
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data_path = os.path.join(data_dir, 'states_age.npy')
    datas = np.load(data_path).astype(np.float32)
    t = torch.arange(datas.shape[0])
    T, N, D = datas.shape
    dist_mat = np.array([[0.0, 120.0, 170.0, 310.0], [120.0, 0.0, 130.0, 220.0], [170.0, 130.0, 0.0, 160.0], [310.0, 220.0, 160.0, 0.0]], dtype=float)
    C_base = np.array([[3.2, 2.92, 2.79, 3.17], [2.88, 3.0, 2.54, 3.13], [2.71, 2.56, 2.5, 2.81], [3.23, 3.22, 2.84, 3.5]], dtype=np.float32)
    C_base = C_base
    city_contact_scale = np.array([1.0], dtype=np.float32)
    C_blocks = [C_base * city_contact_scale[c] for c in range(args.num_cities)]
    C_mat = np.zeros((N, N), dtype=np.float32)
    for c in range(args.num_cities):
        idx0 = c * args.num_ages
        idx1 = (c + 1) * args.num_ages
        C_mat[idx0:idx1, idx0:idx1] = C_blocks[c]
    S0_input = np.array([82000, 102000, 130000, 65000], dtype=np.float32)
    I0_input = np.array([2000, 1100, 1000, 1200], dtype=np.float32)
    R0_input = np.zeros(N, dtype=np.float32)
    U_0 = np.stack([S0_input, I0_input, R0_input], axis=1) * args.re_factor
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat)
    return LE_model
