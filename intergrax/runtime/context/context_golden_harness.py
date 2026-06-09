# © Artur Czarnecki. All rights reserved.

"""Context regression golden harness (IDEAL-16.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_context_golden_cases(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / "tests" / "fixtures" / "context_golden" / "cases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("context golden cases must be a list")
    return cases
