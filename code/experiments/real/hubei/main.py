import os
import argparse
import numpy as np
import torch
import random
import pandas as pd
from source_schema import display_name

import utils
from model import LE_EpiGNN
from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser('LE-EpiGNN')
    parser.add_argument('--save', type=str, default='experiments/', help='Set save.')
    parser.add_argument('--dataset', type=str, default='Hubei_3City_1Age', help='Set dataset.')

    parser.add_argument('--num_cities', type=int, default=3, help='Set num cities.')
    parser.add_argument('--num_ages', type=int, default=1, help='Set num ages.')
    parser.add_argument('--loss_level', type=str, choices=['age', 'city'], default='age', help='Set loss level.')

    parser.add_argument('--epochs', type=int, default=414, help='Set epochs.')
    parser.add_argument('--training_time', type=int, default=15, help='Set training time.')
    parser.add_argument('--validation_time', type=int, default=None, help='Set validation time.')
    parser.add_argument('--lr', type=float, default=8e-5, help='Set lr.')
    parser.add_argument('--warmup_epochs', type=int, default=1000, help='Set warmup epochs.')
    parser.add_argument('--patience', type=int, default=30, help='Set patience.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Set grad clip.')
    parser.add_argument('--lr_factor', type=float, default=0.8, help='Set lr factor.')
    parser.add_argument('--ic_weight', type=float, default=40.0, help='Set ic weight.')

    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--gl_hidden_dim', type=int, default=64, help='Set gl hidden dim.')
    parser.add_argument("--gl_num_layers", type=int, default=10, help='Set gl num layers.')
    parser.add_argument('--c_hidden_dim', type=int, default=9, help='Set c hidden dim.')
    parser.add_argument('-r', '--random_seed', type=int, default=9, help='Set random seed.')
    parser.add_argument('--re_factor', type=float, default=1e-4, help='Set re factor.')

    parser.add_argument(
        '--gamma_init',
        type=str,
        default='0.10,0.78,0.76',
        help='Set gamma init.'
    )


    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0 / 5.2,
        help='Set sigma.'
    )

    parser.add_argument('--alpha_E_init', type=float, default=22500.00, help='Set alpha E init.')
    parser.add_argument('--eps_E_init', type=float, default=0.0001, help='Set eps E init.')
    parser.add_argument('--pop_ref', type=float, default=5e6, help='Set pop ref.')
    parser.add_argument('--mob_hidden_dim', type=int, default=256, help='Set mob hidden dim.')
    parser.add_argument('--mob_num_layers', type=int, default=15, help='Set mob num layers.')
    parser.add_argument('--mob_ckpt_path', type=str, default='', help='Set mob ckpt path.')
    parser.add_argument('--S0_lr_factor', type=float, default=0.25, help='Set S0 lr factor.')
    parser.add_argument('--E0_lr_factor', type=float, default=1.5, help='Set E0 lr factor.')
    parser.add_argument('--I0_lr_factor', type=float, default=1.5, help='Set I0 lr factor.')
    parser.add_argument('--gamma_lr_factor', type=float, default=3.0, help='Set gamma lr factor.')

    parser.add_argument(
        '--use_lr_increase',
        type=str2bool,
        default=True,
        help='Set use lr increase.'
    )
    parser.add_argument(
        '--lr_increase_start',
        type=int,
        default=500,
        help='Set lr increase start.'
    )
    parser.add_argument(
        '--lr_increase_every',
        type=int,
        default=250,
        help='Set lr increase every.'
    )
    parser.add_argument(
        '--lr_increase_factor',
        type=float,
        default=2,
        help='Set lr increase factor.'
    )
    parser.add_argument(
        '--lr_max',
        type=float,
        default=1e-4,
        help='Set lr max.'
    )

    parser.add_argument(
        "--contact_values",
        type=str,
        default="65,14,66",
        help='Set contact values.'
    )

    parser.add_argument(
        "--s0_ratio",
        type=float,
        default=1.0,
        help='Set s0 ratio.'
    )

    parser.add_argument(
        "--i0_min",
        type=float,
        default=0.0,
        help='Set i0 min.'
    )

    parser.add_argument(
        "--s0_values",
        type=str,
        default="120000,10000,9000",
        help='Set s0 values.'
    )

    parser.add_argument(
        "--e0_values",
        type=str,
        default="22,3,5",
        help='Set e0 values.'
    )

    parser.add_argument(
        "--i0_values",
        type=str,
        default="520,1,1",
        help='Set i0 values.'
    )

    parser.add_argument(
        "--r0_values",
        type=str,
        default="0, 0, 0",
        help='Set r0 values.'
    )

    parser.add_argument("--rho_src", type=float, default=3.0)
    parser.add_argument("--rho_dst", type=float, default=10.0)
    parser.add_argument("--alpha_grav", type=float, default=0.2)
    parser.add_argument("--beta_grav", type=float, default=1.8)
    parser.add_argument("--lambda_grav", type=float, default=0.003)


    from training.settings import apply_defaults
    apply_defaults(parser, __file__)
    return parser.parse_args()


def expand_float_list(value, n, name):
    "Broadcast one float to n entries, accept n entries, and reject other lengths."
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [float(x) for x in value]
    else:
        vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]

    if len(vals) == 1:
        vals = vals * n
    elif len(vals) != n:
        raise ValueError(
            f"{name} must have length 1 or num_cities={n}; got {len(vals)}: {vals}"
        )

    return vals


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
    data_dir = os.path.join("data", data_tag)
    data_path = os.path.join(data_dir, "states_age.npy")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing required input: {data_path}"
        )

    datas_raw = np.load(data_path).astype(np.float32)
    datas = datas_raw * args.re_factor

    t = torch.arange(datas.shape[0])
    T, N, D = datas.shape


    args.num_ages = 1
    args.num_cities = N // args.num_ages


    if args.training_time >= T:
        args.training_time = max(2, int(T * 0.7))
        args.validation_time = (3 * args.training_time + 9) // 10
    if not 0 < args.validation_time < args.training_time <= T:
        raise ValueError("Expected 0 < validation_time < training_time <= data length")

    print(f"Loaded data: T={T}, nodes={N}, cities={args.num_cities}, age groups={args.num_ages}")
    print(f"Training days={args.training_time}, monitoring days={args.validation_time}")


    city_order_path = os.path.join(data_dir, "city_population_order.csv")
    if not os.path.exists(city_order_path):
        raise FileNotFoundError(
            f"Missing required input: {city_order_path}"
        )

    city_info = pd.read_csv(city_order_path)
    args.city_names = city_info["city"].astype(str).tolist()
    pop_vec = city_info["population"].astype(float).values.astype(np.float32)

    if len(args.city_names) != args.num_cities:
        raise ValueError(
            f"city_population_order.csv has {len(args.city_names)} cities, "
            f"but states_age.npy has {args.num_cities} cities"
        )

    args.static_pop_city_raw = pop_vec.tolist()

    args.gamma_init = expand_float_list(args.gamma_init, args.num_cities, "gamma_init")
    contact_values = expand_float_list(args.contact_values, args.num_cities, "contact_values")

    print("City order:", display_name(args.city_names))
    print("City population:", pop_vec)
    print("gamma_init:", args.gamma_init)
    print("contact_values:", contact_values)
    print("Fixed sigma:", args.sigma)


    save_root = os.path.join(args.save, data_tag)
    exp_dir = utils.setup_experiment_dir(save_root)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    utils.save_config_to_yaml(args, exp_dir)


    dist_path = os.path.join(data_dir, "dist_mat.npy")
    if not os.path.exists(dist_path):
        raise FileNotFoundError(
            f"Missing required input: {dist_path}"
        )

    dist_mat = np.load(dist_path).astype(np.float32)

    if dist_mat.shape != (args.num_cities, args.num_cities):
        raise ValueError(
            f"dist_mat must have shape {(args.num_cities, args.num_cities)}; got {dist_mat.shape}"
        )


    C_mat = np.zeros((N, N), dtype=np.float32)

    for c in range(args.num_cities):
        idx0 = c * args.num_ages
        idx1 = (c + 1) * args.num_ages
        C_mat[idx0:idx1, idx0:idx1] = contact_values[c] * np.eye(args.num_ages, dtype=np.float32)


    gamma_init_arr = np.array(args.gamma_init, dtype=np.float32)
    sigma_value = max(float(args.sigma), 1e-6)

    gammaI0_raw = datas_raw[0, :, -1].astype(np.float32)


    if args.i0_values.strip():
        I0_input = np.array(
            [float(x.strip()) for x in args.i0_values.split(",") if x.strip()],
            dtype=np.float32
        )
        if len(I0_input) == 1:
            I0_input = np.repeat(I0_input, args.num_cities)
        elif len(I0_input) != args.num_cities:
            raise ValueError(
                f"i0_values must have length 1 or {args.num_cities}; got {len(I0_input)}"
            )
    else:
        I0_input = gammaI0_raw / np.maximum(gamma_init_arr, 1e-6)
        I0_input = np.maximum(I0_input, args.i0_min)

    I0_input = np.maximum(I0_input, 0.0).astype(np.float32)


    if args.e0_values.strip():
        E0_input = np.array(
            [float(x.strip()) for x in args.e0_values.split(",") if x.strip()],
            dtype=np.float32
        )
        if len(E0_input) == 1:
            E0_input = np.repeat(E0_input, args.num_cities)
        elif len(E0_input) != args.num_cities:
            raise ValueError(
                f"e0_values must have length 1 or {args.num_cities}; got {len(E0_input)}"
            )
    else:
        E0_input = gammaI0_raw / sigma_value

    E0_input = np.maximum(E0_input, 0.0).astype(np.float32)


    if args.r0_values.strip():
        R0_input = np.array(
            [float(x.strip()) for x in args.r0_values.split(",") if x.strip()],
            dtype=np.float32
        )

        if len(R0_input) == 1:
            R0_input = np.repeat(R0_input, args.num_cities)
        elif len(R0_input) != args.num_cities:
            raise ValueError(
                f"r0_values must have length 1 or {args.num_cities}; got {len(R0_input)}"
            )
    else:
        R0_input = np.zeros(args.num_cities, dtype=np.float32)

    R0_input = np.maximum(R0_input, 0.0).astype(np.float32)


    if args.s0_values.strip():
        S0_input = np.array(
            [float(x.strip()) for x in args.s0_values.split(",") if x.strip()],
            dtype=np.float32
        )

        if len(S0_input) == 1:
            S0_input = np.repeat(S0_input, args.num_cities)
        elif len(S0_input) != args.num_cities:
            raise ValueError(
                f"s0_values must have length 1 or {args.num_cities}; got {len(S0_input)}"
            )
    else:
        S0_input = pop_vec * args.s0_ratio - E0_input - I0_input - R0_input

    S0_input = np.maximum(S0_input, 1.0).astype(np.float32)

    print("Constructed initial states:")
    for c, name in enumerate(args.city_names):
        print(
            f"  {display_name(name)}: S0={S0_input[c]:.4f}, "
            f"E0={E0_input[c]:.4f}, "
            f"I0={I0_input[c]:.4f}, "
            f"R0={R0_input[c]:.4f}"
        )

    U_0 = np.stack([S0_input, E0_input, I0_input, R0_input], axis=1) * args.re_factor


    print("Initializing LE-EpiGNN-SEIR ...")
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat)


    from training.outputs import finish_training
    args.data_dir = os.path.abspath(os.path.join('data', args.dataset))
    finish_training(LE_model, args, exp_dir)
