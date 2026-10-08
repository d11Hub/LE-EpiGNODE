
import os
import argparse
import numpy as np
import torch
import random
import pandas as pd

import utils
from model import LE_EpiGNN
from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser('LE-EpiGNN')
    parser.add_argument('--save', type=str, default='experiments/', help='Set save.')
    parser.add_argument('--dataset', type=str, default='Dataset_LE_4City_4Age', help='Set dataset.')

    parser.add_argument('--num_cities', type=int, default=4, help='Set num cities.')
    parser.add_argument('--num_ages', type=int, default=4, help='Set num ages.')
    parser.add_argument('--loss_level', type=str, choices=['age', 'city'], default='age', help='Set loss level.')

    parser.add_argument('--epochs', type=int, default=20, help='Set epochs.')
    parser.add_argument('--training_time', type=int, default=56, help='Set training time.')
    parser.add_argument('--validation_time', type=int, default=None, help='Set validation time.')
    parser.add_argument('--lr', type=float, default=1e-4,  help='Set lr.')
    parser.add_argument('--warmup_epochs', type=int, default=1000, help='Set warmup epochs.')
    parser.add_argument('--patience', type=int, default=30,help='Set patience.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Set grad clip.')
    parser.add_argument('--lr_factor', type=float, default=0.8, help='Set lr factor.')
    parser.add_argument('--ic_weight', type=float, default=1.0, help='Set ic weight.')

    parser.add_argument('-r', '--random_seed', type=int, default=10, help='Set random seed.')
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gl_num_layers",  type=int, default=6, help='Set gl num layers.')
    parser.add_argument('--gl_hidden_dim', type=int, default=64, help='Set gl hidden dim.')
    parser.add_argument('--c_hidden_dim', type=int, default=5, help='Set c hidden dim.')
    parser.add_argument('--re_factor', type=float, default=1e-4, help='Set re factor.')

    parser.add_argument('--gamma', type=float, default=0.021, help='Set gamma.')
    parser.add_argument('--pop_ref', type=float, default=1e6, help='Set pop ref.')
    parser.add_argument('--mob_hidden_dim', type=int, default=64, help='Set mob hidden dim.')
    parser.add_argument('--mob_num_layers', type=int, default=2, help='Set mob num layers.')
    parser.add_argument('--mob_ckpt_path', type=str, default='', help='Set mob ckpt path.')

    parser.add_argument('--use_lr_increase', type=str2bool, default=True,
                        help='Set use lr increase.')
    parser.add_argument('--lr_increase_start', type=int, default=500,
                        help='Set lr increase start.')
    parser.add_argument('--lr_increase_every', type=int, default=250,
                        help='Set lr increase every.')
    parser.add_argument('--lr_increase_factor', type=float, default=2,
                        help='Set lr increase factor.')
    parser.add_argument('--lr_max', type=float, default=1e-4,
                        help='Set lr max.')


    from training.settings import apply_defaults
    apply_defaults(parser, __file__)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.validation_time is None:
        args.validation_time = (3 * args.training_time + 9) // 10


    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    data_tag = args.dataset
    save_root = os.path.join(args.save, data_tag)
    exp_dir = utils.setup_experiment_dir(save_root)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    utils.save_config_to_yaml(args, exp_dir)


    data_path = os.path.join("data", data_tag, "states_age_noisy.npy") 
    datas = np.load(data_path).astype(np.float32)


    t = torch.arange(datas.shape[0])
    T, N, D = datas.shape
    print(f"Loaded data: {T} time steps, {N} nodes")


    dist_mat = np.array([
        [0.0, 120.0, 170.0, 310.0],
        [120.0, 0.0, 130.0, 220.0],
        [170.0, 130.0, 0.0, 160.0],
        [310.0, 220.0, 160.0, 0.0]
    ], dtype=np.float32)

    C_base = 0.5 * np.array([
        [3.20, 2.92, 2.79, 3.17],
        [3.10, 3.00, 2.84, 3.13],
        [3.21, 3.16, 2.90, 3.31],
        [3.23, 2.90, 2.74, 3.00]
    ], dtype=np.float32)

    city_contact_scale = np.array(
        [1.00, 0.98, 0.95, 0.92],
        dtype=np.float32
    )

    C_blocks = [
        C_base * city_contact_scale[c]
        for c in range(args.num_cities)
    ]


    C_mat = np.zeros((N, N), dtype=np.float32)
    for c in range(args.num_cities):
        idx0 = c * args.num_ages
        idx1 = (c + 1) * args.num_ages
        C_mat[idx0:idx1, idx0:idx1] = C_blocks[c]


    S0_input = np.array([
        120000, 165000, 190000, 115000,
        110000, 155000, 180000, 105000,
        100000, 145000, 170000, 95000,
        115000, 160000, 185000, 110000
    ], dtype=np.float32)


    I0_input = np.array([
        1900, 1000, 1200, 1300,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    ], dtype=np.float32)


    R0_input = np.zeros(N, dtype=np.float32)


    U_0 = np.stack([S0_input, I0_input, R0_input], axis=1) * args.re_factor


    print("Initializing LE-EpiGNN ...")
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat)


    from training.outputs import finish_training
    args.data_dir = os.path.abspath(os.path.join('data', args.dataset))
    finish_training(LE_model, args, exp_dir)
