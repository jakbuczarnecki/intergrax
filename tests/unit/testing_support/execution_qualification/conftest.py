# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture
def run_artifact_root(repo_root: Path, tmp_path: Path) -> Path:
    root = repo_root / "build" / "qualification" / f"unit-{tmp_path.name}"
    root.mkdir(parents=True, exist_ok=True)
    return root
