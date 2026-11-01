# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO / "tests" / "qualification" / "evol_x" / "data"

_OWNER_REQUIRED_KEYS = ("owner_id", "domain", "paths_prefix")


def validate_evol_x_data_document(filename: str, payload: dict[str, Any]) -> dict[str, Any]:
    if "source_authority" not in payload:
        raise ValueError(f"{filename}: missing source_authority")
    if "schema_version" not in payload:
        raise ValueError(f"{filename}: missing schema_version")
    if filename.startswith("owner_rules"):
        for owner in payload.get("owners", ()):
            missing = [k for k in _OWNER_REQUIRED_KEYS if k not in owner]
            if missing or not owner.get("paths_prefix"):
                raise ValueError(f"{filename}: invalid owner row {owner!r}")
    if filename.startswith("exclusion_register"):
        for entry in payload.get("entries", ()):
            if entry.get("status") not in {"ACCEPTED", "OPEN", "IN_SCOPE_CLASSIFIED"}:
                raise ValueError(f"{filename}: invalid exclusion status {entry!r}")
    return payload


def load_evol_x_data_document(filename: str, *, data_dir: Path | None = None) -> dict[str, Any]:
    path = (data_dir or _DATA_DIR) / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_evol_x_data_document(filename, payload)
