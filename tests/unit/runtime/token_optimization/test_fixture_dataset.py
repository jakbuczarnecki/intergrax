# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-2E-B: file-backed regression fixture dataset loader tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intergrax.runtime.token_optimization.fixture_dataset import (
    load_token_regression_fixture_dataset,
)
from intergrax.runtime.token_optimization.regression import (
    run_token_regression_benchmarks,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATASET_PATH = (
    _REPO_ROOT / "benchmarks" / "token_optimization" / "fixtures" / "regression_synthetic_v1"
)
_REQUIRED_FIXTURE_IDS = frozenset(
    {
        "tool_schema.compact_catalog",
        "tool_schema.protected_description",
        "context_pack.compact_fragments",
        "context_pack.protected_evidence",
        "memory_summary.compact_summary",
        "memory_summary.protected_dates",
        "memory_summary.fallback_validation",
    }
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


@pytest.fixture(name="dataset")
def fixture_dataset():
    return load_token_regression_fixture_dataset(_DATASET_PATH)


def test_loader_exposes_dataset_id_and_version(dataset) -> None:
    assert dataset.dataset_id == "regression_synthetic_v1"
    assert dataset.dataset_version == "1.0.0"
    assert dataset.metadata["dataset_id"] == "regression_synthetic_v1"
    assert dataset.metadata["dataset_version"] == "1.0.0"


def test_loader_discovers_fixture_json_recursively(dataset) -> None:
    fixture_paths = {fixture.metadata["fixture_path"] for fixture in dataset.fixtures}
    assert "cases/tool_schema/compact_catalog/fixture.json" in fixture_paths
    assert "cases/context_pack/compact_fragments/fixture.json" in fixture_paths
    assert "cases/memory_summary/fallback_validation/fixture.json" in fixture_paths


def test_fixture_count_is_seven_or_eight(dataset) -> None:
    assert len(dataset.fixtures) in {7, 8}


def test_required_fixture_ids_are_present(dataset) -> None:
    fixture_ids = {fixture.fixture_id for fixture in dataset.fixtures}
    assert _REQUIRED_FIXTURE_IDS.issubset(fixture_ids)


def test_fixture_ids_are_unique(dataset) -> None:
    fixture_ids = [fixture.fixture_id for fixture in dataset.fixtures]
    assert len(fixture_ids) == len(set(fixture_ids))


def test_tool_schema_fixture_runs_successfully(dataset) -> None:
    summary = run_token_regression_benchmarks(fixtures=dataset.fixtures)
    result = next(
        item for item in summary.results if item.fixture_id == "tool_schema.compact_catalog"
    )
    assert result.passed is True
    assert result.saved_tokens >= 1


def test_context_pack_fixture_runs_successfully(dataset) -> None:
    summary = run_token_regression_benchmarks(fixtures=dataset.fixtures)
    result = next(
        item for item in summary.results if item.fixture_id == "context_pack.compact_fragments"
    )
    assert result.passed is True
    assert result.saved_tokens >= 1


def test_memory_summary_fixture_runs_successfully(dataset) -> None:
    summary = run_token_regression_benchmarks(fixtures=dataset.fixtures)
    result = next(
        item for item in summary.results if item.fixture_id == "memory_summary.compact_summary"
    )
    assert result.passed is True
    assert result.receipt_present is True


def test_fallback_validation_fixture_maps_policy_and_passes(dataset) -> None:
    fixture = next(
        item for item in dataset.fixtures if item.fixture_id == "memory_summary.fallback_validation"
    )
    assert fixture.expectation.expect_fallback is True
    assert fixture.expectation.expected_validation_status == "failed"
    assert fixture.expectation.allow_fallback is True
    assert fixture.expectation.expect_validation_pass is False

    summary = run_token_regression_benchmarks(fixtures=dataset.fixtures)
    result = next(
        item for item in summary.results if item.fixture_id == "memory_summary.fallback_validation"
    )
    assert result.passed is True
    assert result.validation_status == "failed"
    assert result.fallback_status is True
    assert result.saved_tokens == 0


def test_expectations_are_mapped_correctly(dataset) -> None:
    compact_tool = next(
        item for item in dataset.fixtures if item.fixture_id == "tool_schema.compact_catalog"
    )
    assert compact_tool.expectation.expected_min_saved_tokens == 1
    assert compact_tool.expectation.expected_min_saved_ratio == 0.05
    assert compact_tool.expectation.require_receipt is True
    assert compact_tool.expectation.expect_validation_pass is True
    assert compact_tool.expectation.allow_fallback is False
    assert compact_tool.expectation.expect_fallback is False

    protected_tool = next(
        item for item in dataset.fixtures if item.fixture_id == "tool_schema.protected_description"
    )
    assert protected_tool.expectation.expected_max_saved_tokens == 0
    assert protected_tool.expectation.expected_max_saved_ratio == 0.0


def test_protected_values_reflected_in_metadata(dataset) -> None:
    protected_dates = next(
        item for item in dataset.fixtures if item.fixture_id == "memory_summary.protected_dates"
    )
    assert protected_dates.metadata["protected_value_count"] == 1

    compact_summary = next(
        item for item in dataset.fixtures if item.fixture_id == "memory_summary.compact_summary"
    )
    assert compact_summary.metadata["protected_value_count"] == 0


def test_loader_does_not_require_global_manifest_or_expectations(dataset) -> None:
    assert not (_DATASET_PATH / "manifest.json").exists()
    assert not (_DATASET_PATH / "expectations.json").exists()
    assert len(dataset.fixtures) >= 7


def test_report_metadata_does_not_contain_unsafe_keys(dataset) -> None:
    summary = run_token_regression_benchmarks(fixtures=dataset.fixtures)
    payload = {
        "results": [
            {
                "fixture_id": result.fixture_id,
                "metadata": dict(result.metadata),
            }
            for result in summary.results
        ]
    }
    assert _collect_keys(payload).isdisjoint(_UNSAFE_KEYS)


def test_unsupported_semantic_validation_hook_raises() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_id": "tmp",
                    "dataset_version": "0.0.1",
                    "case_discovery": {
                        "strategy": "recursive_fixture_json",
                        "root": "cases",
                    },
                }
            ),
            encoding="utf-8",
        )
        case_dir = root / "cases" / "memory_summary" / "bad_hook"
        case_dir.mkdir(parents=True)
        (case_dir / "input.txt").write_text("summary text", encoding="utf-8")
        (case_dir / "fixture.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fixture_id": "memory_summary.bad_hook",
                    "source_type": "memory_summary",
                    "eval_case": "fallback",
                    "category": "memory",
                    "expected_behavior": "test",
                    "description": "test",
                    "input": {"format": "plain_text", "file": "input.txt"},
                    "optimizer": {
                        "kind": "memory_summary",
                        "semantic_validation_hook": "reject",
                    },
                    "protected_values": [],
                    "expected": {
                        "receipt": "required",
                        "validation": "pass_like",
                        "fallback": "forbidden",
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="semantic_validation_hook"):
            load_token_regression_fixture_dataset(root)
