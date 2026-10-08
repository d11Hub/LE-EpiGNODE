"""Load default arguments for a scenario."""
from pathlib import Path
import yaml


def apply_defaults(parser, script_file):
    script = Path(script_file).resolve()
    path = script.parent / "training.yaml"
    if not path.is_file():
        path = script.parents[4] / "configs/training" / script.parent.parent.name / (script.parent.name + ".yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    destinations = {action.dest for action in parser._actions}
    defaults = {key: value for key, value in config.items() if key in destinations and value is not None}
    defaults["save"] = "experiments/"
    for key in ["gamma_init", "eta_values"]:
        if isinstance(defaults.get(key), list):
            defaults[key] = ",".join(str(x) for x in defaults[key])
    parser.set_defaults(**defaults)
