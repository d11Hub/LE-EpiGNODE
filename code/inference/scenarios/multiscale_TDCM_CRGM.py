"""Checkpoint initialization for multiscale_TDCM_CRGM."""
import os
import random
import numpy as np
import pandas as pd
import torch
from models.synthetic.multiscale import LE_EpiGNN as BaseModel

class LE_EpiGNN(BaseModel):
    SCENARIO = 'multiscale_TDCM_CRGM'

DEFAULTS = {'dataset': 'Dataset_LE_4City_4Age', 'num_cities': 4, 'num_ages': 4, 'random_seed': 10, 'gl_num_layers': 6, 'gl_hidden_dim': 64, 'c_hidden_dim': 5, 're_factor': 0.0001, 'gamma': 0.021, 'pop_ref': 1000000.0, 'mob_hidden_dim': 64, 'mob_num_layers': 2, 'mob_ckpt_path': ''}

def build_model(args):
    data_dir = args.data_dir
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data_path = os.path.join(data_dir, 'states_age_noisy.npy')
    datas = np.load(data_path).astype(np.float32)
    t = torch.arange(datas.shape[0])
    T, N, D = datas.shape
    dist_mat = np.array([[0.0, 120.0, 170.0, 310.0], [120.0, 0.0, 130.0, 220.0], [170.0, 130.0, 0.0, 160.0], [310.0, 220.0, 160.0, 0.0]], dtype=np.float32)
    C_base = 0.5 * np.array([[3.2, 2.92, 2.79, 3.17], [3.1, 3.0, 2.84, 3.13], [3.21, 3.16, 2.9, 3.31], [3.23, 2.9, 2.74, 3.0]], dtype=np.float32)
    city_contact_scale = np.array([1.0, 0.98, 0.95, 0.92], dtype=np.float32)
    C_blocks = [C_base * city_contact_scale[c] for c in range(args.num_cities)]
    C_mat = np.zeros((N, N), dtype=np.float32)
    for c in range(args.num_cities):
        idx0 = c * args.num_ages
        idx1 = (c + 1) * args.num_ages
        C_mat[idx0:idx1, idx0:idx1] = C_blocks[c]
    S0_input = np.array([120000, 165000, 190000, 115000, 110000, 155000, 180000, 105000, 100000, 145000, 170000, 95000, 115000, 160000, 185000, 110000], dtype=np.float32)
    I0_input = np.array([1900, 1000, 1200, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    R0_input = np.zeros(N, dtype=np.float32)
    U_0 = np.stack([S0_input, I0_input, R0_input], axis=1) * args.re_factor
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat)
    return LE_model
