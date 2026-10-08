"""Load Hubei city aliases for display."""
import json
from pathlib import Path
def source_label(key):
    here = Path(__file__).absolute().parent
    candidates = [here / "source_schema.json"]
    candidates += list((here / "data").glob("*/source_schema.json"))
    for parent in here.parents:
        candidates.append(parent / "data" / "real" / "hubei" / "source_schema.json")
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))[key]
    raise FileNotFoundError("City alias mapping not found: source_schema.json")


def display_name(value):
    """Map a city name or a sequence of city names to English display names."""
    aliases = source_label("city_aliases")
    if isinstance(value, (list, tuple)):
        return [aliases.get(str(item), str(item)) for item in value]
    return aliases.get(str(value), str(value))
