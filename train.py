"""Prepare a scenario working directory or run model training."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent


def discover():
    return {p.parent.name: (p.parent, PACKAGE / "data" / p.parent.parent.name / p.parent.name,
                           PACKAGE / "results" / p.parent.parent.name / p.parent.name)
            for p in sorted((PACKAGE / "code/experiments").glob("*/*/main.py"))}


def training_config(scenario, sources):
    return PACKAGE / "configs/training" / sources[0].parent.name / (scenario + ".yaml")


def config_arguments(config_file, device):
    config = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    arguments = []
    for key, value in config.items():
        if key in {"city_names", "age_names", "static_pop_city_raw"} or value is None:
            continue
        if key == "device":
            value = device
        if key == "save":
            value = "experiments/"
        arguments.append("--" + key)
        if isinstance(value, list) and key in {"gamma_init", "eta_values"}:
            arguments.append(",".join(str(item) for item in value))
        elif isinstance(value, list):
            arguments.extend(str(item) for item in value)
        elif isinstance(value, bool):
            arguments.append(str(value).lower())
        else:
            arguments.append(str(value))
    return arguments


def prepare(scenario, destination, sources):
    code, data, results = sources
    work = Path(destination).resolve() / scenario
    if work.exists():
        raise FileExistsError("Working directory already exists: {}".format(work))
    shutil.copytree(code, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ["models", "training", "inference"]:
        shutil.copytree(PACKAGE / "code" / name, work / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = yaml.safe_load(training_config(scenario, sources).read_text(encoding="utf-8"))
    target = work / "data" / config["dataset"]
    shutil.copytree(data, target, ignore=shutil.ignore_patterns("*.py", "__pycache__", "*.pyc"))
    for weights in results.glob("pretrained*.pth"):
        shutil.copy2(weights, target / weights.name)
    shutil.copy2(training_config(scenario, sources), work / "training.yaml")
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List scenario IDs")
    action.add_argument("--prepare", metavar="ID", help="Prepare a scenario working directory")
    action.add_argument("--run", metavar="ID", help="Prepare and train a scenario")
    parser.add_argument("--output", type=Path, default=PACKAGE.parent / "training_runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, help="Override the training epoch count")
    options = parser.parse_args()
    sources = discover()
    if options.list:
        print("\n".join(sorted(sources)))
        return
    scenario = options.prepare or options.run
    if scenario not in sources:
        parser.error("Unknown scenario; use --list")
    if options.epochs is not None and options.epochs < 1:
        parser.error("--epochs must be positive")
    destination = Path(os.path.abspath(options.output))
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        parser.error("Symlinked output paths are not supported")
    destination = destination.resolve()
    if destination == PACKAGE or PACKAGE in destination.parents:
        parser.error("Training output must be outside the package")
    work = prepare(scenario, destination, sources[scenario])
    print("Prepared {}".format(work), flush=True)
    if options.run:
        command = [sys.executable, "-B", "main.py", *config_arguments(work / "training.yaml", options.device)]
        if options.epochs is not None:
            command.extend(["--epochs", str(options.epochs)])
        subprocess.run(command, cwd=work, check=True)


if __name__ == "__main__":
    main()
