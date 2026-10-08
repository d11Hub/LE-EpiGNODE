"""Create experiment directories and record settings and console output."""
from datetime import datetime
from pathlib import Path
import argparse
import sys
import uuid
import yaml


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, message):
        self.console.write(message)
        self.log.write(message)
        return len(message)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def setup_experiment_dir(save_path):
    identifier = "exp_" + datetime.now().strftime("%Y%m%d-%H%M") + "_" + uuid.uuid4().hex[:6]
    folder = Path(save_path) / identifier
    folder.mkdir(parents=True, exist_ok=False)
    log = (folder / "training.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    print("Experiment directory: {}".format(folder), flush=True)
    return str(folder)


def save_config_to_yaml(args, exp_dir):
    with (Path(exp_dir) / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False, allow_unicode=False)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes", "y", "1"}:
        return True
    if value.lower() in {"false", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")
