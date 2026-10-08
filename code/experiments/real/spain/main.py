
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
    parser.add_argument('--dataset', type=str, default='Spain_4Region_R1_R4', help='Set dataset.')

    parser.add_argument('--num_cities', type=int, default=4, help='Set num cities.')
    parser.add_argument('--num_ages', type=int, default=3, help='Set num ages.')
    parser.add_argument('--loss_level', type=str, choices=['age', 'city'], default='age', help='Set loss level.')

    parser.add_argument('--epochs', type=int, default=98, help='Set epochs.')
    parser.add_argument('--training_time', type=int, default=20, help='Set training time.')
    parser.add_argument('--validation_time', type=int, default=None, help='Set validation time.')
    parser.add_argument('--lr', type=float, default=0.00034896149312711093, help='Set lr.')

    parser.add_argument('--warmup_epochs', type=int, default=1000, help='Set warmup epochs.')
    parser.add_argument('--patience', type=int, default=30, help='Set patience.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Set grad clip.')
    parser.add_argument('--lr_factor', type=float, default=0.8, help='Set lr factor.')

    parser.add_argument(
        '--ic_weight',
        type=float,
        default=0.13242639182475752,
        help='Set ic weight.'
    )

    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument(
        '--gl_hidden_dim',
        type=int,
        default=64,
        help='Set gl hidden dim.'
    )

    parser.add_argument(
        "--gl_num_layers",
        type=int,
        default=5,
        help='Set gl num layers.'
    )

    parser.add_argument(
        '--c_hidden_dim',
        type=int,
        default=4,
        help='Set c hidden dim.'
    )

    parser.add_argument(
        '-r',
        '--random_seed',
        type=int,
        default=8,
        help='Set random seed.'
    )

    parser.add_argument(
        '--re_factor',
        type=float,
        default=4.629680785916101e-05,
        help='Set re factor.'
    )

    parser.add_argument(
        "--gamma_init",
        type=str,
        default=(
            "0.27287395392098274,"
            "0.20308836622116583,"
            "0.4356073007728257,"
            "0.10925454311570595,"
            "0.12120827867089094,"
            "0.16207766677655694,"
            "0.24363212649515667,"
            "0.20117940203898285,"
            "0.11384306252533437,"
            "0.3779637960655219,"
            "0.4914470581793447,"
            "0.28819499522543457"
        ),
    )


    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0 / 5.2,
        help='Set sigma.'
    )

    parser.add_argument(
        '--alpha_E_init',
        type=float,
        default=0.06392274673368706,
        help='Set alpha E init.'
    )

    parser.add_argument(
        '--eps_E_init',
        type=float,
        default=5.348036274296318e-12,
        help='Set eps E init.'
    )

    parser.add_argument(
        '--pretrained_MLP_mob_path',
        type=str,
        default='',
        help=(
            'Set pretrained MLP mob path.'
            "When empty, load "
            "data/<dataset>/pretrained_MLP_mob.pth"
        )
    )

    parser.add_argument(
        "--S0_lr_factor",
        type=float,
        default=9.202222460247059,
    )
    parser.add_argument(
        "--E0_lr_factor",
        type=float,
        default=17.614135604155905,
    )

    parser.add_argument(
        "--I0_lr_factor",
        type=float,
        default=5.248857296225804,
    )

    parser.add_argument(
        "--gamma_lr_factor",
        type=float,
        default=0.04516359966900622,
    )

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
        "--eta_values",
        type=str,
        default=(
            "0.3589332240561754,"
            "0.24079838933578698,"
            "0.40026838660803765"
        ),
    )
    parser.add_argument(
        "--age_population_ratio",
        type=str,
        default="0.3333333333,0.3333333333,0.3333333334",
        help='Set age population ratio.'
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
        default=(
            "8754.196137908459,"
            "24353.690161700644,"
            "50182.48160138278,"
            "42509.987799538685,"
            "50708.22125218894,"
            "98609.0471611192,"
            "17974.1093624509,"
            "8860.152808506891,"
            "39397.616680248946,"
            "10849.320758930775,"
            "312.6229597798789,"
            "1010.7240612102931"
        ),
    )

    parser.add_argument(
        "--e0_values",
        type=str,
        default=(
            "18.974420310569727,"
            "2425.928259289346,"
            "392.6565383731958,"
            "954.5845344448434,"
            "7105.481481875951,"
            "3422.7394312395554,"
            "77.05656267777404,"
            "2313.0282699637223,"
            "2709.455444417754,"
            "136.59108206619777,"
            "401.87126784112166,"
            "19.404589410621146"
        ),
    )
    parser.add_argument(
        "--i0_values",
        type=str,
        default=(
            "484.99257429859006,"
            "383.7419900887593,"
            "2.4290278791791935,"
            "805.9390966052825,"
            "1048.97815044621,"
            "1822.0007465700264,"
            "0.9538102893528242,"
            "0.00618105759227543,"
            "6.639527349217693e-06,"
            "0.1224131857724681,"
            "111.5661334812281,"
            "1.0116379380848612"
        ),
    )

    parser.add_argument(
        "--r0_values",
        type=str,
        default="0",
        help='Set r0 values.'
    )


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
            f"{name} must have length 1 or target length={n}; "
            f"got {len(vals)}: {vals}"
        )

    return vals
def prepare_mob_checkpoint(args, data_dir):
    "Validate mobility weights, city order, input features, edge_bias, output_mode, edge_scale and gravity settings."
    default_path = os.path.join(
        data_dir,
        "pretrained_MLP_mob.pth"
    )

    ckpt_path = (
        args.pretrained_MLP_mob_path
        if args.pretrained_MLP_mob_path
        else default_path
    )

    ckpt_path = os.path.abspath(ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Pretrained MLP_mob checkpoint not found:\n"
            f"{ckpt_path}\n"
            'Check the required dataset and model files.'
            "to regenerate pretrained_MLP_mob.pth."
        )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(ckpt, dict):
        raise TypeError(
            'This checkpoint contains only a provided state_dict; '
            "edge_scale, output_mode and feature settings are missing."
            'Check the required dataset and model files.'
        )

    required_keys = [
        "MLP_mob_state_dict",
        "num_cities",
        "city_names",
        "enhanced_features",
        "use_city_id",
        "weekly_time_features",
        "time_feature_mode",
        "long_period_days",
        "use_edge_bias",
        "output_mode",
        "edge_scale",
        "pop_ref",
        "alpha_grav",
        "beta_grav",
        "lambda_grav",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in ckpt
    ]

    if missing_keys:
        raise KeyError(
            "Pretrained checkpoint is missing fields:\n"
            f"{missing_keys}\n"
            'Check the required dataset and model files.'
            'Check the required dataset and model files.'
        )


    ckpt_num_cities = int(ckpt["num_cities"])

    if ckpt_num_cities != args.num_cities:
        raise ValueError(
            "Pretrained and main models have different city counts:\n"
            f"checkpoint: {ckpt_num_cities}\n"
            f"main.py:    {args.num_cities}"
        )


    ckpt_city_names = [
        str(x) for x in ckpt["city_names"]
    ]

    main_city_names = [
        str(x) for x in args.city_names
    ]

    if ckpt_city_names != main_city_names:
        raise ValueError(
            "Pretrained and main models have different city orders:\n"
            f"checkpoint: {ckpt_city_names}\n"
            f"main.py:    {main_city_names}"
        )


    output_mode = str(ckpt["output_mode"])

    if output_mode not in {"raw", "edge_scaled"}:
        raise ValueError(
            f"Unsupported output_mode={output_mode}"
        )

    num_edges = (
        args.num_cities
        * (args.num_cities - 1)
    )

    edge_scale = torch.as_tensor(
        ckpt["edge_scale"],
        dtype=torch.float32
    ).view(-1)

    if edge_scale.numel() != num_edges:
        raise ValueError(
            "Incorrect checkpoint edge_scale length: "
            f"expected {num_edges}; "
            f"got {edge_scale.numel()}."
        )

    if torch.any(edge_scale <= 0):
        raise ValueError(
            "Checkpoint edge_scale must be strictly positive."
        )


    if bool(ckpt["use_edge_bias"]):
        if ckpt.get("edge_bias") is None:
            raise ValueError(
                "Checkpoint specifies use_edge_bias=True "
                "but has no edge_bias."
            )

        edge_bias = torch.as_tensor(
            ckpt["edge_bias"],
            dtype=torch.float32
        ).view(-1)

        if edge_bias.numel() != num_edges:
            raise ValueError(
                "Incorrect checkpoint edge_bias length: "
                f"expected {num_edges}; "
                f"got {edge_bias.numel()}."
            )


    args.pretrained_MLP_mob_path = ckpt_path

    print("\n========== Pretrained mobility model checked ==========")
    print(f"checkpoint: {ckpt_path}")
    print(f"City order: {ckpt_city_names}")
    print(f"output_mode: {output_mode}")
    print(
        "edge_scale range: "
        f"[{edge_scale.min().item():.6e}, "
        f"{edge_scale.max().item():.6e}]"
    )
    print(
        "Feature settings: "
        f"enhanced={bool(ckpt['enhanced_features'])}, "
        f"city_id={bool(ckpt['use_city_id'])}, "
        f"weekly={bool(ckpt['weekly_time_features'])}, "
        f"time_mode={ckpt['time_feature_mode']}, "
        f"long_period={float(ckpt['long_period_days'])}"
    )

    return ckpt


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
    if datas_raw.ndim != 3:
        raise ValueError(
            "states_age.npy must be a 3D array; "
            "[T, N, 5],"
            f"got shape={datas_raw.shape}"
        )

    if D != 5:
        raise ValueError(
            "SEIR requires five channels in states_age.npy; "
            "[S, E, I, R, gammaI],"
            f"got D={D}"
        )

    if not np.all(np.isfinite(datas_raw)):
        raise ValueError(
            "states_age.npy contains NaN or Inf."
        )

    age_order_path = os.path.join(data_dir, "age_order.csv")
    if not os.path.exists(age_order_path):
        raise FileNotFoundError(f"Missing required input: {age_order_path}")
    age_info = pd.read_csv(age_order_path)
    args.age_names = age_info["age_name"].astype(str).tolist()
    args.num_ages = len(args.age_names)
    if args.num_ages <= 0 or N % args.num_ages != 0:
        raise ValueError(f"Node count N={N} is not divisible by age count {args.num_ages}.")
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

    if N != args.num_cities * args.num_ages:
        raise ValueError(
            f"states_age.npy has {N} nodes, but regions times ages = "
            f"{args.num_cities}×{args.num_ages}={args.num_cities * args.num_ages}"
        )

    if len(args.city_names) != args.num_cities:
        raise ValueError(
            f"city_population_order.csv has {len(args.city_names)} cities, "
            f"but states_age.npy has {args.num_cities} cities"
        )

    args.static_pop_city_raw = pop_vec.tolist()


    mob_ckpt = prepare_mob_checkpoint(
        args,
        data_dir
    )


    args.gamma_init = expand_float_list(
        args.gamma_init,
        N,
        "gamma_init"
    )
    args.eta_values = expand_float_list(args.eta_values, args.num_ages, "eta_values")
    age_population_ratio = np.asarray(
        expand_float_list(args.age_population_ratio, args.num_ages, "age_population_ratio"),
        dtype=np.float32
    )
    if np.any(age_population_ratio < 0) or age_population_ratio.sum() <= 0:
        raise ValueError("age_population_ratio must be nonnegative with a positive sum.")
    age_population_ratio = age_population_ratio / age_population_ratio.sum()

    print("NUTS1 region order:", args.city_names)
    print("Age group order:", args.age_names)
    print("Region population:", pop_vec)
    print("gamma_init:", args.gamma_init)
    print("eta_values:", args.eta_values)
    print("age_population_ratio:", age_population_ratio.tolist())
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


    if not np.all(np.isfinite(dist_mat)):
        raise ValueError(
            "dist_mat.npy contains NaN or Inf."
        )

    if np.any(dist_mat < 0):
        raise ValueError(
            "dist_mat.npy must contain nonnegative distances."
        )

    if not np.allclose(
            np.diag(dist_mat),
            0.0,
            atol=1e-6
    ):
        raise ValueError(
            "dist_mat.npy must have a zero diagonal."
        )

    if dist_mat.shape != (args.num_cities, args.num_cities):
        raise ValueError(
            f"dist_mat must have shape {(args.num_cities, args.num_cities)}; got {dist_mat.shape}"
        )

    contact_path = os.path.join(data_dir, "contact_mat.npy")
    if not os.path.exists(contact_path):
        raise FileNotFoundError(f"Missing required input: {contact_path}")
    C_mat = np.load(contact_path).astype(np.float32)
    if C_mat.shape != (N, N):
        raise ValueError(f"contact_mat must have shape {(N, N)}; got {C_mat.shape}")


    gamma_init_node = np.asarray(
        args.gamma_init,
        dtype=np.float32
    )

    if gamma_init_node.shape != (N,):
        raise ValueError(
            f"gamma_init must contain N={N} node values; "
            f"got shape={gamma_init_node.shape}"
        )

    if np.any(gamma_init_node <= 0.0) or np.any(gamma_init_node >= 1.0):
        raise ValueError(
            "Every gamma_init value must be in (0, 1)."
        )

    sigma_value = max(float(args.sigma), 1e-6)

    gammaI0_raw = datas_raw[0, :, -1].astype(np.float32)


    if args.i0_values.strip():
        I0_input = np.asarray(expand_float_list(args.i0_values, N, "i0_values"), dtype=np.float32)
    else:
        I0_input = gammaI0_raw / np.maximum(gamma_init_node, 1e-6)
        I0_input = np.maximum(I0_input, args.i0_min)

    I0_input = np.maximum(I0_input, 0.0).astype(np.float32)


    if args.e0_values.strip():
        E0_input = np.asarray(expand_float_list(args.e0_values, N, "e0_values"), dtype=np.float32)
    else:
        E0_input = gammaI0_raw / sigma_value

    E0_input = np.maximum(E0_input, 0.0).astype(np.float32)


    if args.r0_values.strip():
        R0_input = np.asarray(expand_float_list(args.r0_values, N, "r0_values"), dtype=np.float32)
    else:
        R0_input = np.zeros(N, dtype=np.float32)

    R0_input = np.maximum(R0_input, 0.0).astype(np.float32)


    if args.s0_values.strip():
        S0_input = np.asarray(expand_float_list(args.s0_values, N, "s0_values"), dtype=np.float32)
    else:
        population_nodes = (pop_vec[:, None] * age_population_ratio[None, :]).reshape(-1)
        S0_input = population_nodes * args.s0_ratio - E0_input - I0_input - R0_input

    S0_input = np.maximum(S0_input, 1.0).astype(np.float32)

    print("Node parameter initial values:")

    for c, city_name in enumerate(args.city_names):
        for a, age_name in enumerate(args.age_names):
            idx = c * args.num_ages + a

            print(
                f"  [{idx:02d}] {city_name}/{age_name}: "
                f"gamma={gamma_init_node[idx]:.6f}, "
                f"S0={S0_input[idx]:.4f}, "
                f"E0={E0_input[idx]:.4f}, "
                f"I0={I0_input[idx]:.4f}, "
                f"R0={R0_input[idx]:.4f}"
            )

    U_0 = np.stack([S0_input, E0_input, I0_input, R0_input], axis=1) * args.re_factor


    print("Initializing LE-EpiGNN-SEIR ...")
    LE_model = LE_EpiGNN(args, t, datas, U_0, C_mat, dist_mat)


    from training.outputs import finish_training
    args.data_dir = os.path.abspath(os.path.join('data', args.dataset))
    finish_training(LE_model, args, exp_dir)
