# © Artur Czarnecki. All rights reserved.

"""Diagnostic artifact writer for token regression benchmark executions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from intergrax.runtime.token_optimization.regression import (
    TokenRegressionCaseExecution,
    TokenRegressionExecution,
)

_DIAGNOSTIC_JSON_KWARGS: dict[str, Any] = {
    "indent": 2,
    "sort_keys": True,
    "ensure_ascii": False,
}


def write_token_regression_diagnostic_artifacts(
    execution: TokenRegressionExecution,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write self-contained diagnostic artifacts for a benchmark execution."""
    root = Path(output_dir)
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    sorted_cases = sorted(execution.cases, key=lambda case: case.fixture_id)
    case_artifact_paths: list[str] = []

    for case in sorted_cases:
        relative_path = f"cases/{case.fixture_id}.json"
        case_path = root / relative_path
        case_payload = _build_case_artifact(case)
        _write_json(case_path, case_payload)
        case_artifact_paths.append(relative_path)

    summary_path = root / "summary.json"
    summary_payload = _build_summary_artifact(execution, case_artifact_paths)
    _write_json(summary_path, summary_payload)

    manifest_payload = {
        "schema_version": 1,
        "artifact_kind": "token_regression_diagnostic_manifest",
        "summary_artifact": "summary.json",
        "case_count": len(case_artifact_paths),
        "cases": case_artifact_paths,
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest_payload)

    return {
        "output_dir": str(root),
        "summary_artifact": str(summary_path),
        "manifest_artifact": str(manifest_path),
        "case_count": len(case_artifact_paths),
        "case_artifacts": [str(root / path) for path in case_artifact_paths],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, **_DIAGNOSTIC_JSON_KWARGS) + "\n",
        encoding="utf-8",
    )


def _build_summary_artifact(
    execution: TokenRegressionExecution,
    case_artifact_paths: list[str],
) -> dict[str, Any]:
    summary = execution.summary
    cases_by_id = {case.fixture_id: case for case in execution.cases}
    case_entries: list[dict[str, Any]] = []

    for relative_path in case_artifact_paths:
        fixture_id = Path(relative_path).stem
        case = cases_by_id[fixture_id]
        result = case.result
        case_entries.append(
            {
                "fixture_id": result.fixture_id,
                "source_type": result.source_type,
                "passed": result.passed,
                "baseline_tokens": result.baseline_tokens,
                "optimized_tokens": result.optimized_tokens,
                "saved_tokens": result.saved_tokens,
                "saved_ratio": result.saved_ratio,
                "validation_status": result.validation_status,
                "fallback_status": result.fallback_status,
                "receipt_present": result.receipt_present,
                "case_artifact": relative_path,
            }
        )

    return {
        "schema_version": 1,
        "artifact_kind": "token_regression_diagnostic_summary",
        "total_fixtures": summary.total_fixtures,
        "passed": summary.passed,
        "failed": summary.failed,
        "total_baseline_tokens": summary.total_baseline_tokens,
        "total_optimized_tokens": summary.total_optimized_tokens,
        "total_saved_tokens": summary.total_saved_tokens,
        "total_saved_ratio": summary.total_saved_ratio,
        "metadata": dict(summary.metadata),
        "cases": case_entries,
    }


def _build_case_artifact(case: TokenRegressionCaseExecution) -> dict[str, Any]:
    result = case.result
    input_payload, output_payload = _extract_input_output(case)

    return {
        "schema_version": 1,
        "artifact_kind": "token_regression_diagnostic_case",
        "fixture_id": case.fixture_id,
        "source_type": case.source_type,
        "passed": result.passed,
        "metadata": dict(result.metadata),
        "input": input_payload,
        "output": output_payload,
        "metrics": {
            "baseline_tokens": result.baseline_tokens,
            "optimized_tokens": result.optimized_tokens,
            "saved_tokens": result.saved_tokens,
            "saved_ratio": result.saved_ratio,
        },
        "validation": {
            "validation_status": result.validation_status,
            "fallback_used": result.fallback_status,
            "receipt_present": result.receipt_present,
            "strategy": result.strategy,
        },
        "expectation": {
            "passed": result.passed,
            "failure_reasons": list(result.failure_reasons),
        },
    }


def _extract_input_output(
    case: TokenRegressionCaseExecution,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if case.outcome is None:
        return {}, {}

    source_type = case.source_type
    if source_type == "context_pack":
        return _extract_context_pack_io(case.outcome)
    if source_type == "memory_summary":
        return _extract_memory_summary_io(case.outcome)
    if source_type == "tool_schema":
        return _extract_tool_schema_io(case.outcome)

    return _extract_generic_io(case.outcome)


def _extract_context_pack_io(outcome: object) -> tuple[dict[str, Any], dict[str, Any]]:
    original_fragments = getattr(outcome, "original_fragments", None) or ()
    optimized_fragments = getattr(outcome, "optimized_fragments", None) or ()

    input_payload: dict[str, Any] = {
        "fragments": [_fragment_to_input_dict(fragment) for fragment in original_fragments],
        "original_content": _coerce_str(getattr(outcome, "original_content", "")),
    }
    output_payload: dict[str, Any] = {
        "fragments": [_fragment_to_output_dict(fragment) for fragment in optimized_fragments],
        "optimized_content": _coerce_str(getattr(outcome, "optimized_content", "")),
    }
    return input_payload, output_payload


def _fragment_to_input_dict(fragment: object) -> dict[str, Any]:
    return {
        "fragment_id": getattr(fragment, "fragment_id", ""),
        "required": bool(getattr(fragment, "required", False)),
        "metadata": dict(getattr(fragment, "metadata", {}) or {}),
        "original_content": _coerce_str(getattr(fragment, "content", "")),
    }


def _fragment_to_output_dict(fragment: object) -> dict[str, Any]:
    return {
        "fragment_id": getattr(fragment, "fragment_id", ""),
        "required": bool(getattr(fragment, "required", False)),
        "metadata": dict(getattr(fragment, "metadata", {}) or {}),
        "optimized_content": _coerce_str(getattr(fragment, "content", "")),
    }


def _extract_memory_summary_io(outcome: object) -> tuple[dict[str, Any], dict[str, Any]]:
    original = (
        getattr(outcome, "original_summary", None)
        or getattr(outcome, "original_content", "")
    )
    optimized = (
        getattr(outcome, "optimized_summary", None)
        or getattr(outcome, "optimized_content", "")
    )
    return (
        {"original_summary": _coerce_str(original)},
        {"optimized_summary": _coerce_str(optimized)},
    )


def _extract_tool_schema_io(outcome: object) -> tuple[dict[str, Any], dict[str, Any]]:
    original = (
        getattr(outcome, "original_catalog", None)
        or getattr(outcome, "original_content", "")
    )
    optimized = (
        getattr(outcome, "optimized_catalog", None)
        or getattr(outcome, "optimized_content", "")
    )
    return (
        {"original_tool_catalog": _coerce_str(original)},
        {"optimized_tool_catalog": _coerce_str(optimized)},
    )


def _extract_generic_io(outcome: object) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {"original_content": _coerce_str(getattr(outcome, "original_content", ""))},
        {"optimized_content": _coerce_str(getattr(outcome, "optimized_content", ""))},
    )


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)
