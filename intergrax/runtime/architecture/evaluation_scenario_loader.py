# © Artur Czarnecki. All rights reserved.

"""Golden scenario library loader (IDEAL-25.1)."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.runtime.architecture.evaluation_assets import ScenarioLibraryAsset


def default_scenario_library_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "eval_scenarios" / "library.v1.json"


def load_scenario_library(repo_root: Path) -> ScenarioLibraryAsset:
    path = default_scenario_library_path(repo_root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ScenarioLibraryAsset.model_validate(raw)
