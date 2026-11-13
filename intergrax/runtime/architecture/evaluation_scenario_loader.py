# © Artur Czarnecki. All rights reserved.

"""Golden scenario library loader (IDEAL-25.1)."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.contracts.evaluation_contracts import EvalDataset, EvalScenario
from intergrax.runtime.architecture.evaluation_assets import ScenarioLibraryAsset


def default_scenario_library_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "eval_scenarios" / "library.v1.json"


def load_scenario_library(repo_root: Path) -> ScenarioLibraryAsset:
    path = default_scenario_library_path(repo_root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ScenarioLibraryAsset.model_validate(raw)


def scenario_library_to_eval_dataset(library: ScenarioLibraryAsset) -> EvalDataset:
    """Typed dataset view of a scenario library: stable scenario identity, content digest."""
    return EvalDataset(
        dataset_id=library.library_id,
        version=library.version,
        scenarios=tuple(
            EvalScenario(
                scenario_id=case.scenario_id,
                description=case.description,
                risk_tags=tuple(case.risk_tags),
            )
            for case in library.scenarios
        ),
    )


def load_scenario_library_dataset(repo_root: Path) -> EvalDataset:
    return scenario_library_to_eval_dataset(load_scenario_library(repo_root))
