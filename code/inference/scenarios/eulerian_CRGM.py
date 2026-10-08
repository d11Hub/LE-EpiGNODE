"""Checkpoint initialization for eulerian_CRGM."""
import os
import random
import numpy as np
import pandas as pd
import torch
from models.synthetic.eulerian import LE_EpiGNN as BaseModel

class LE_EpiGNN(BaseModel):
    SCENARIO = 'eulerian_CRGM'

DEFAULTS = {'dataset': 'Dataset_E_4City_1Age', 'num_cities': 4, 'num_ages': 1, 'random_seed': 2, 'gl_num_layers': 3, 'gl_hidden_dim': 64, 'c_hidden_dim': 8, 're_factor': 1e-05, 'gamma': 0.08, 'pop_ref': 100000.0, 'mob_hidden_dim': 128, 'mob_num_layers': 5, 'mob_ckpt_path': ''}

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
    dist_mat = np.array([[0.0, 120.0, 210.0, 260.0], [120.0, 0.0, 150.0, 180.0], [210.0, 150.0, 0.0, 140.0], [260.0, 180.0, 140.0, 0.0]], dtype=np.float32)
    C_blocks = [np.array([[1.0]], dtype=np.float32), np.array([[1.1]], dtype=np.float32), np.array([[1.1]], dtype=np.float32), np.array([[1.1]], dtype=np.float32)]
    C_mat = np.zeros((N, N), dtype=np.float32)
    for c in range(args.num_cities):
        idx0 = c * args.num_ages
        idx1 = (c + 1) * args.num_ages
        C_mat[idx0:idx1, idx0:idx1] = C_blocks[c]
    S0_input = np.array([130000, 100000, 90000, 80000], dtype=np.float32)
    I0_input = np.array([500, 0, 0, 0], dtype=np.float32)
    R0_input = np.zeros(N, dtype=np.float32)
    U_0 = np.stack([S0_input, I0_input, R0_input], axis=1) * args.re_factor
    static_pop_city_raw = S0_input.reshape(args.num_cities, args.num_ages).sum(axis=1)
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat, static_pop_city_raw)
    return LE_model
