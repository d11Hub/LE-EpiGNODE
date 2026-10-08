"""Save the best monitored model."""
from pathlib import Path
import torch


def save_checkpoint(model, optimizer, epoch, loss, save_dir, is_best=False):
    if not is_best:
        return
    folder = Path(save_dir)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "loss": loss,
                "model_config": vars(model.args)}, folder / "best_model.pth")
