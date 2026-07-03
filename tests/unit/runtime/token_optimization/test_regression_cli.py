# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-2D: benchmark CLI report/gate output tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_token_regression_benchmarks.py"
_FIXTURE_DATASET = (
    "benchmarks/token_optimization/fixtures/regression_synthetic_v1"
)

_UNSAFE_KEYS = frozenset(
    {
        "content",
        "raw_content",
        "original_content",
        "optimized_content",
        "prompt",
        "messages",
        "document",
        "documents",
        "memory",
        "memory_content",
        "summary_text",
        "tool_schema",
        "tool_catalog",
        "context",
        "context_pack",
        "fragments",
        "evidence",
        "payload",
        "body",
        "raw_context",
        "raw_prompt",
        "raw_document",
        "tool_args",
        "chunks",
        "event",
        "signal",
    }
)


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *args],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_collect_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_collect_keys(nested))
    return keys


def test_report_output_contains_human_header() -> None:
    completed = _run_script("--report")
    assert completed.returncode == 0
    assert "Token regression benchmark report" in completed.stdout
    assert "fixtures=7 passed=7 failed=0" in completed.stdout


def test_report_json_output_is_valid_and_contains_results() -> None:
    completed = _run_script("--report-json")
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["total_fixtures"] == 7
    assert payload["failed"] == 0
    assert len(payload["results"]) == 7
    assert _collect_keys(payload).isdisjoint(_UNSAFE_KEYS)


def test_gate_output_contains_status_pass() -> None:
    completed = _run_script("--gate")
    assert completed.returncode == 0
    assert "Token regression gate" in completed.stdout
    assert "status=pass" in completed.stdout


def test_gate_json_output_is_valid_and_contains_status_pass() -> None:
    completed = _run_script("--gate-json")
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "pass"
    assert payload["failed"] == 0
    assert _collect_keys(payload).isdisjoint(_UNSAFE_KEYS)


def test_gate_with_impossible_ratio_threshold_fails() -> None:
    completed = _run_script("--gate", "--min-total-saved-ratio", "1.0")
    assert completed.returncode != 0
    assert "total_saved_ratio_below_threshold" in completed.stdout


def test_gate_json_with_impossible_ratio_threshold_fails() -> None:
    completed = _run_script("--gate-json", "--min-total-saved-ratio", "1.0")
    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "fail"
    reason_codes = {failure["reason_code"] for failure in payload["failures"]}
    assert "total_saved_ratio_below_threshold" in reason_codes


def test_multiple_output_modes_fail_fast() -> None:
    completed = _run_script("--report", "--gate")
    assert completed.returncode != 0
    combined = f"{completed.stdout}\n{completed.stderr}".lower()
    assert "not allowed with" in combined or "mutually exclusive" in combined


def test_threshold_flags_are_ignored_outside_gate_modes() -> None:
    completed = _run_script("--report", "--min-total-saved-ratio", "1.0")
    assert completed.returncode == 0
    assert "total_saved_ratio_below_threshold" not in completed.stdout


def test_report_with_fixture_dataset_exits_zero() -> None:
    completed = _run_script("--report", "--fixture-dataset", _FIXTURE_DATASET)
    assert completed.returncode == 0
    assert "Token regression benchmark report" in completed.stdout


def test_gate_with_fixture_dataset_exits_zero() -> None:
    completed = _run_script("--gate", "--fixture-dataset", _FIXTURE_DATASET)
    assert completed.returncode == 0
    assert "status=pass" in completed.stdout


def test_report_json_with_fixture_dataset_is_safe() -> None:
    completed = _run_script("--report-json", "--fixture-dataset", _FIXTURE_DATASET)
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["failed"] == 0
    assert len(payload["results"]) in {7, 8}
    assert _collect_keys(payload).isdisjoint(_UNSAFE_KEYS)


def test_gate_json_with_fixture_dataset_passes_and_is_safe() -> None:
    completed = _run_script("--gate-json", "--fixture-dataset", _FIXTURE_DATASET)
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "pass"
    assert payload["failed"] == 0
    assert _collect_keys(payload).isdisjoint(_UNSAFE_KEYS)
