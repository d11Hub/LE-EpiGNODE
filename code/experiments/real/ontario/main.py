
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
    parser.add_argument('--dataset', type=str, default='Dataset_LE_1City_4Age', help='Set dataset.')

    parser.add_argument('--num_cities', type=int, default=1, help='Set num cities.')
    parser.add_argument('--num_ages', type=int, default=4, help='Set num ages.')
    parser.add_argument('--loss_level', type=str, choices=['age', 'city'], default='age', help='Set loss level.')

    parser.add_argument('--epochs', type=int, default=360, help='Set epochs.')
    parser.add_argument('--training_time', type=int, default=25, help='Set training time.')
    parser.add_argument('--validation_time', type=int, default=None, help='Set validation time.')
    parser.add_argument('--lr', type=float, default=5e-6, help='Set lr.')
    parser.add_argument('--warmup_epochs', type=int, default=1000, help='Set warmup epochs.')
    parser.add_argument('--patience', type=int, default=15,help='Set patience.')
    parser.add_argument('--ode_rtol', type=float, default=1e-5, help='Set ode rtol.')
    parser.add_argument('--ode_atol', type=float, default=1e-8, help='Set ode atol.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Set grad clip.')
    parser.add_argument('--lr_factor', type=float, default=0.8, help='Set lr factor.')
    parser.add_argument('--S0_lr_factor', type=float, default=380, help='Set S0 lr factor.')
    parser.add_argument('--E0_lr_factor', type=float, default=0.5,help='Set E0 lr factor.')
    parser.add_argument('--I0_lr_factor', type=float, default=0.5, help='Set I0 lr factor.')
    parser.add_argument('--gamma_lr_factor', type=float, default=0.5, help='Set gamma lr factor.')
    parser.add_argument('--ic_weight', type=float, default=0.001, help='Set ic weight.')

    parser.add_argument('-r', '--random_seed', type=int, default=3, help='Set random seed.')
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gl_num_layers",  type=int, default=3, help='Set gl num layers.')
    parser.add_argument('--gl_hidden_dim', type=int, default=64, help='Set gl hidden dim.')
    parser.add_argument('--c_hidden_dim', type=int, default=6, help='Set c hidden dim.')
    parser.add_argument('--re_factor', type=float, default=1.0283875e-5, help='Set re factor.')

    parser.add_argument(
        '--gamma',
        type=float,
        nargs='+',
        default=[0.24691881, 0.43754812, 0.18519953, 0.09355255],
        help='Set gamma.'
    )

    parser.add_argument(
        '--sigma',
        type=float,
        nargs='+',
        default=[1/3.61, 1/3.61, 1/3.61, 1/3.61],
        help='Set sigma.'
    )

    parser.add_argument('--pop_ref', type=float, default=1e6, help='Set pop ref.')
    parser.add_argument('--mob_hidden_dim', type=int, default=128, help='Set mob hidden dim.')
    parser.add_argument('--mob_num_layers', type=int, default=4, help='Set mob num layers.')
    parser.add_argument('--mob_ckpt_path', type=str, default='', help='Set mob ckpt path.')

    parser.add_argument('--case_excel', type=str, default='Ontario_data_20211115_20220301.xlsx',
                        help='Set case excel.')
    parser.add_argument('--case_sheet', type=str, default='SmoothedData',
                        choices=['RawData', 'SmoothedData'],
                        help='Set case sheet.')
    parser.add_argument('--date_start', type=str, default='2021-11-28',
                        help='Set date start.')
    parser.add_argument('--date_end', type=str, default='2022-02-27',
                        help='Set date end.')
    parser.add_argument('--contact_excel', type=str, default='Ontario_grouped_contact_matrix.xlsx',
                        help='Set contact excel.')
    parser.add_argument('--contact_sheet', type=str, default='GroupedContactMatrix',
                        help='Set contact sheet.')
    parser.add_argument('--pop_sheet', type=str, default='GroupPopulation',
                        help='Set pop sheet.')


    parser.add_argument('--case_multiplier', type=float, default=1.0,
                        help='Set case multiplier.')


    parser.add_argument(
        '--manual_S0',
        type=float,
        nargs='+',
        default=[73457.56, 246495.61, 135706.66, 46018.70],
        help='Set manual S0.'
    )

    parser.add_argument(
        '--manual_E0',
        type=float,
        nargs='+',
        default=[1420.41, 1713.03, 1029.36, 714.62],
        help='Set manual E0.'
    )

    parser.add_argument(
        '--manual_I0',
        type=float,
        nargs='+',
        default=[2079.65, 1659.51, 685.86, 327.14],
        help='Set manual I0.'
    )

    parser.add_argument(
        '--manual_R0',
        type=float,
        nargs='+',
        default=None,
        help='Set manual R0.'
    )


    parser.add_argument('--save_real_npy', type=str2bool, default=True,
                        help='Set save real npy.')


    from training.settings import apply_defaults
    apply_defaults(parser, __file__)
    return parser.parse_args()


def load_ontario_prepared_data(args):
    'Load the dataset arrays.'
    data_dir = os.path.join("data", args.dataset)
    datas = np.load(os.path.join(data_dir, "states_age.npy")).astype(np.float32)
    C_mat = np.load(os.path.join(data_dir, "contact_matrix.npy")).astype(np.float32)
    group_pop = np.load(os.path.join(data_dir, "group_population.npy")).astype(np.float32)
    dates = pd.to_datetime(pd.read_csv(os.path.join(data_dir, "aligned_dates.csv"))["date"])
    group_names = ["0-19", "20-39", "40-59", "60+"]
    if datas.ndim != 3 or datas.shape[1:] != (4, 5) or C_mat.shape != (4, 4):
        raise ValueError("Ontario processed arrays have unexpected dimensions")
    if len(dates) != len(datas) or group_pop.shape != (4,):
        raise ValueError("Ontario processed dates or population do not match the states")
    args.num_ages = len(group_names)
    r0 = args.manual_R0 if args.manual_R0 is not None else [0.0] * len(group_names)
    U_0 = np.stack([args.manual_S0, args.manual_E0, args.manual_I0, r0], axis=1).astype(np.float32)
    U_0 *= args.re_factor
    return datas, C_mat, group_names, group_pop, dates, U_0


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


    datas, C_mat, group_names, group_pop, dates, U_0 = load_ontario_prepared_data(args)
    utils.save_config_to_yaml(args, exp_dir)


    t = torch.arange(datas.shape[0])
    T, N, D = datas.shape

    args.num_cities = 1
    args.num_ages = len(group_names)

    print(f"Loaded real data: {T} time steps, {N} age groups, {D} state channels")


    dist_mat = np.zeros((args.num_cities, args.num_cities), dtype=np.float32)


    print("Initializing LE-EpiGNN ...")
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat)


    from training.outputs import finish_training
    args.data_dir = os.path.abspath(os.path.join('data', args.dataset))
    finish_training(LE_model, args, exp_dir)
