# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO / "tests" / "qualification" / "evol_x" / "data"


def load_evol_x_data_document(filename: str) -> dict[str, Any]:
    path = _DATA_DIR / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "source_authority" not in payload:
        raise ValueError(f"{filename}: missing source_authority")
    if "schema_version" not in payload:
        raise ValueError(f"{filename}: missing schema_version")
    return payload
