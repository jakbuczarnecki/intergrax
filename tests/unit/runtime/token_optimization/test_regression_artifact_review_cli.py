# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-2G: diagnostic artifact review CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.fixture_dataset import (
    load_token_regression_fixture_dataset,
)
from intergrax.runtime.token_optimization.regression import (
    run_token_regression_benchmark_execution,
)
from intergrax.runtime.token_optimization.regression_diagnostics import (
    write_token_regression_diagnostic_artifacts,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "review_token_regression_artifacts.py"
_DATASET_PATH = (
    _REPO_ROOT / "benchmarks" / "token_optimization" / "fixtures" / "regression_synthetic_v1"
)


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *args],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _generate_artifact_folder(tmp_path: Path) -> Path:
    dataset = load_token_regression_fixture_dataset(_DATASET_PATH)
    execution = run_token_regression_benchmark_execution(fixtures=dataset.fixtures)
    write_token_regression_diagnostic_artifacts(execution, tmp_path)
    return tmp_path


def test_review_cli_human_output(tmp_path: Path) -> None:
    artifact_dir = _generate_artifact_folder(tmp_path)

    completed = _run_script(str(artifact_dir))

    assert completed.returncode == 0
    assert "Token regression diagnostic artifact review" in completed.stdout
    assert "Status:" in completed.stdout
    assert "Top savings:" in completed.stdout
    assert "Safety checks:" in completed.stdout
    assert "Marketing interpretation:" in completed.stdout


def test_review_cli_json_output(tmp_path: Path) -> None:
    artifact_dir = _generate_artifact_folder(tmp_path)

    completed = _run_script(str(artifact_dir), "--json")

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["artifact_kind"] == "token_regression_artifact_review"


def test_review_cli_missing_dir_fails() -> None:
    missing_path = _REPO_ROOT / ".artifacts" / "token_optimization" / "missing_review_dir"

    completed = _run_script(str(missing_path))

    assert completed.returncode == 1
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert "fail" in combined.lower() or "missing" in combined.lower()
