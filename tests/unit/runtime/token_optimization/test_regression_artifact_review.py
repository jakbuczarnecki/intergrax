# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-2G: diagnostic artifact review tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intergrax.runtime.token_optimization.fixture_dataset import (
    load_token_regression_fixture_dataset,
)
from intergrax.runtime.token_optimization.regression import (
    run_token_regression_benchmark_execution,
)
from intergrax.runtime.token_optimization.regression_artifact_review import (
    review_token_regression_artifacts,
)
from intergrax.runtime.token_optimization.regression_diagnostics import (
    write_token_regression_diagnostic_artifacts,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATASET_PATH = (
    _REPO_ROOT / "benchmarks" / "token_optimization" / "fixtures" / "regression_synthetic_v1"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _generate_artifact_folder(tmp_path: Path) -> Path:
    dataset = load_token_regression_fixture_dataset(_DATASET_PATH)
    execution = run_token_regression_benchmark_execution(fixtures=dataset.fixtures)
    write_token_regression_diagnostic_artifacts(execution, tmp_path)
    return tmp_path


def _issue_codes(review: dict[str, Any]) -> set[str]:
    return {issue["code"] for issue in review["issues"]}


def _case_review_by_fixture(
    review: dict[str, Any],
    fixture_id: str,
) -> dict[str, Any] | None:
    for entry in review["top_savings"]:
        if entry["fixture_id"] == fixture_id:
            return entry
    for entry in review["safety_checks"]:
        if entry["fixture_id"] == fixture_id:
            return entry
    return None


def _find_change_type(review: dict[str, Any], fixture_id: str) -> str | None:
    for entry in review["top_savings"]:
        if entry["fixture_id"] == fixture_id:
            return entry["change_type"]
    for entry in review["safety_checks"]:
        if entry["fixture_id"] == fixture_id:
            return entry["change_type"]
    return None


def test_review_passes_generated_artifact_folder_with_warnings(tmp_path: Path) -> None:
    artifact_dir = _generate_artifact_folder(tmp_path)

    review = review_token_regression_artifacts(artifact_dir)

    assert review["status"] == "pass_with_warnings"
    assert review["summary"]["total_fixtures"] == 8
    assert "dominant_savings_case" in _issue_codes(review)
    truncation_issues = [
        issue
        for issue in review["issues"]
        if issue["code"] == "likely_truncation"
        and issue.get("fixture_id") == "context_pack.long_workspace_document"
    ]
    assert truncation_issues


def test_review_detects_missing_manifest(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "summary.json",
        {
            "schema_version": 1,
            "artifact_kind": "token_regression_diagnostic_summary",
            "total_fixtures": 0,
            "passed": 0,
            "failed": 0,
            "total_baseline_tokens": 0,
            "total_optimized_tokens": 0,
            "total_saved_tokens": 0,
            "total_saved_ratio": 0.0,
            "metadata": {},
            "cases": [],
        },
    )

    review = review_token_regression_artifacts(tmp_path)

    assert review["status"] == "fail"
    assert "missing_manifest" in _issue_codes(review)


def test_review_detects_missing_case_file(tmp_path: Path) -> None:
    case_path = "cases/missing.case.json"
    _write_json(
        tmp_path / "summary.json",
        {
            "schema_version": 1,
            "artifact_kind": "token_regression_diagnostic_summary",
            "total_fixtures": 1,
            "passed": 1,
            "failed": 0,
            "total_baseline_tokens": 10,
            "total_optimized_tokens": 8,
            "total_saved_tokens": 2,
            "total_saved_ratio": 0.2,
            "metadata": {},
            "cases": [
                {
                    "fixture_id": "missing.case",
                    "source_type": "tool_schema",
                    "passed": True,
                    "baseline_tokens": 10,
                    "optimized_tokens": 8,
                    "saved_tokens": 2,
                    "saved_ratio": 0.2,
                    "validation_status": "passed",
                    "fallback_status": False,
                    "receipt_present": True,
                    "case_artifact": case_path,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "artifact_kind": "token_regression_diagnostic_manifest",
            "summary_artifact": "summary.json",
            "case_count": 1,
            "cases": [case_path],
        },
    )

    review = review_token_regression_artifacts(tmp_path)

    assert review["status"] == "fail"
    assert "missing_case_artifact" in _issue_codes(review)


def test_review_detects_mojibake(tmp_path: Path) -> None:
    case_path = "cases/tool_schema.encoding_issue.json"
    case_payload = {
        "schema_version": 1,
        "artifact_kind": "token_regression_diagnostic_case",
        "fixture_id": "tool_schema.encoding_issue",
        "source_type": "tool_schema",
        "passed": True,
        "metadata": {
            "eval_case": "compactable",
            "protected_value_count": 0,
        },
        "input": {"original_tool_catalog": "safe input"},
        "output": {"optimized_tool_catalog": "truncated text ends with â€¦"},
        "metrics": {
            "baseline_tokens": 10,
            "optimized_tokens": 8,
            "saved_tokens": 2,
            "saved_ratio": 0.2,
        },
        "validation": {
            "validation_status": "passed",
            "fallback_used": False,
            "receipt_present": True,
            "strategy": "tool_schema.compaction",
        },
        "expectation": {
            "passed": True,
            "failure_reasons": [],
        },
    }
    _write_json(tmp_path / "cases" / "tool_schema.encoding_issue.json", case_payload)
    _write_json(
        tmp_path / "summary.json",
        {
            "schema_version": 1,
            "artifact_kind": "token_regression_diagnostic_summary",
            "total_fixtures": 1,
            "passed": 1,
            "failed": 0,
            "total_baseline_tokens": 10,
            "total_optimized_tokens": 8,
            "total_saved_tokens": 2,
            "total_saved_ratio": 0.2,
            "metadata": {},
            "cases": [
                {
                    "fixture_id": "tool_schema.encoding_issue",
                    "source_type": "tool_schema",
                    "passed": True,
                    "baseline_tokens": 10,
                    "optimized_tokens": 8,
                    "saved_tokens": 2,
                    "saved_ratio": 0.2,
                    "validation_status": "passed",
                    "fallback_status": False,
                    "receipt_present": True,
                    "case_artifact": case_path,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "artifact_kind": "token_regression_diagnostic_manifest",
            "summary_artifact": "summary.json",
            "case_count": 1,
            "cases": [case_path],
        },
    )

    review = review_token_regression_artifacts(tmp_path)

    assert review["status"] == "pass_with_warnings"
    assert "possible_encoding_issue" in _issue_codes(review)


def test_review_classifies_fallback_case(tmp_path: Path) -> None:
    artifact_dir = _generate_artifact_folder(tmp_path)
    review = review_token_regression_artifacts(artifact_dir)

    change_type = _find_change_type(review, "memory_summary.fallback_validation")
    assert change_type == "expected_fallback"


def test_review_classifies_protected_case(tmp_path: Path) -> None:
    artifact_dir = _generate_artifact_folder(tmp_path)
    review = review_token_regression_artifacts(artifact_dir)

    protected_dates = _find_change_type(review, "memory_summary.protected_dates")
    protected_description = _find_change_type(
        review,
        "tool_schema.protected_description",
    )
    assert protected_dates == "protected_content_preserved"
    assert protected_description == "protected_content_preserved"


def test_review_json_is_serializable(tmp_path: Path) -> None:
    artifact_dir = _generate_artifact_folder(tmp_path)
    review = review_token_regression_artifacts(artifact_dir)

    serialized = json.dumps(review, sort_keys=True)
    assert serialized
