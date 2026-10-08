"""Source-table aliases stored outside executable code."""
import json
from pathlib import Path
def source_label(key):
    here = Path(__file__).absolute().parent
    candidates = [here / "source_schema.json"]
    candidates += list((here / "data").glob("*/source_schema.json"))
    for parent in here.parents:
        candidates.append(parent / "data" / "real" / "spain" / "source_schema.json")
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))[key]
    raise FileNotFoundError("Missing source_schema.json; prepare the scenario before reading source tables")
