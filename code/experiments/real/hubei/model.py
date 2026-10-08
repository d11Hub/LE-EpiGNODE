'Scenario model settings.'
import sys
from pathlib import Path
for parent in Path(__file__).absolute().parents:
    if (parent / "models").is_dir():
        sys.path.insert(0, str(parent))
        break
else:
    raise ImportError("Prepare this experiment with train.py")
from models.real.hubei import LE_EpiGNN as _BaseModel, ResMLP

class LE_EpiGNN(_BaseModel):
    SCENARIO = 'hubei'
