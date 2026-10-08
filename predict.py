'Load provided best weights and save gammaI predictions and English figures.'
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "code"))

from inference.runtime import discover, build_model
from inference.prediction_output import write_prediction
from inference.plot_predictions import plot_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="Scenario ID or all")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device: cuda:0 or cpu")
    parser.add_argument("--output", type=Path, default=PACKAGE.parent / "predictions")
    parser.add_argument("--overwrite", action="store_true", help="Replace selected prediction directories")
    options = parser.parse_args()
    sources = discover()
    if options.list:
        print("\n".join(sorted(sources)))
        return
    if options.scenario != "all" and options.scenario not in sources:
        parser.error("Choose a scenario ID or all; use --list to display IDs")
    selected = sorted(sources) if options.scenario == "all" else [options.scenario]
    raw_output = Path(os.path.abspath(options.output))
    if any(path.is_symlink() for path in (raw_output, *raw_output.parents)):
        parser.error("Symlinked output paths are not supported")
    output = raw_output.resolve()
    if output == PACKAGE or PACKAGE in output.parents:
        parser.error("Prediction output must be outside the package")
    targets = {name: output / sources[name][0] / name for name in selected}
    allowed = {"pred_u.npy", "pred_u_city.npy", "GE_city.npy", "GL_age.npy", "pred_mobility.npy",
               "case_forecast.png", "city_forecast.png", "mobility_forecast.png"}
    for target in targets.values():
        if any(path.is_symlink() for path in (target, *target.parents)):
            parser.error("Symlinked output paths are not supported")
        if target.exists() and not options.overwrite:
            parser.error("Output already exists: {}. Use another output or --overwrite".format(target))
        if target.exists() and (not target.is_dir() or any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in target.iterdir())):
            parser.error("Refusing to replace a directory containing non-prediction files: {}".format(target))
    for name in selected:
        kind, data, archive = sources[name]
        print("Predicting {} on {}".format(name, options.device), flush=True)
        model, args = build_model(name, options.device)
        with tempfile.TemporaryDirectory(prefix="le_epignode_inference_") as folder:
            generated = Path(folder)
            write_prediction(model, archive / "best_model.pth", generated, args.num_cities, args.num_ages)
            plot_predictions(name, kind, data, archive / "config.yaml", generated)
            target = targets[name]
            if target.exists():
                if output not in target.resolve().parents:
                    raise ValueError("Refusing to replace a directory outside the output root")
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(generated, target)
        del model
        print("Saved {}".format(target), flush=True)


if __name__ == "__main__":
    main()
