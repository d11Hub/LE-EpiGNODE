"""Train a model and save its best-model forecasts."""
from pathlib import Path
import csv
import numpy as np
import torch
from inference.prediction_output import write_prediction
from inference.plot_predictions import plot_predictions
from .experiment_utils import save_config_to_yaml


def finish_training(model, args, exp_dir):
    result = model.train_series(exp_dir)
    history = result[0]
    if len(history) != args.epochs:
        raise RuntimeError("Training did not complete the requested epoch count")
    losses = np.array([row["Val_Loss"] for row in history], dtype=float)
    if not np.isfinite(losses).all() or not np.isfinite([row["Loss"] for row in history]).all():
        raise RuntimeError("Nonfinite training or monitoring loss")
    folder = Path(exp_dir)
    with (folder / "loss_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = folder / "checkpoints/best_model.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] != int(np.argmin(losses)) or saved["loss"] != float(losses.min()):
        raise RuntimeError("The best checkpoint does not match the minimum monitoring loss")
    save_config_to_yaml(args, folder)
    output = folder / "predictions"
    write_prediction(model, checkpoint, output, args.num_cities, args.num_ages)
    kind = "real" if model.SCENARIO in {"ontario", "hubei", "spain"} else "synthetic"
    plot_predictions(model.SCENARIO, kind, Path(args.data_dir), folder / "config.yaml", output)
    print("Best checkpoint epoch: {}".format(saved["epoch"]), flush=True)
    print("Predictions: {}".format(output), flush=True)
