'Discover processed scenarios and construct scenario models.'
import importlib
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

PACKAGE = Path(__file__).resolve().parents[2]


def discover():
    scenarios = {}
    for config in sorted((PACKAGE / "results").glob("*/*/config.yaml")):
        scenario = config.parent.name
        kind = config.parent.parent.name
        scenarios[scenario] = (kind, PACKAGE / "data" / kind / scenario, config.parent)
    return scenarios


def build_model(scenario, device):
    kind, data, archive = discover()[scenario]
    module = importlib.import_module("inference.scenarios." + scenario)
    config = yaml.safe_load((archive / "config.yaml").read_text(encoding="utf-8"))
    settings = dict(module.DEFAULTS)
    settings.update(config)
    settings.update(device=device, data_dir=str(data), inference_only=True)

    mobility = str(archive / "pretrained_MLP_mob.pth")
    settings.update(pretrained_mob_path=mobility, pretrained_MLP_mob_path=mobility, mob_ckpt_path=mobility)
    args = SimpleNamespace(**settings)
    if not torch.device(device).type == "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu")
    return module.build_model(args), args
