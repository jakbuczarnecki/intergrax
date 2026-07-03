# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-2F: diagnostic regression artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.fixture_dataset import (
    load_token_regression_fixture_dataset,
)
from intergrax.runtime.token_optimization.regression import (
    TokenRegressionExpectation,
    TokenRegressionFixture,
    TokenRegressionSourceType,
    run_token_regression_benchmark_execution,
)
from intergrax.runtime.token_optimization.regression_diagnostics import (
    write_token_regression_diagnostic_artifacts,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATASET_PATH = (
    _REPO_ROOT / "benchmarks" / "token_optimization" / "fixtures" / "regression_synthetic_v1"
)


def _load_dataset_execution():
    dataset = load_token_regression_fixture_dataset(_DATASET_PATH)
    return run_token_regression_benchmark_execution(fixtures=dataset.fixtures)


def test_diagnostic_artifacts_write_summary_and_case_files(tmp_path: Path) -> None:
    execution = _load_dataset_execution()
    write_token_regression_diagnostic_artifacts(execution, tmp_path)

    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "cases" / "context_pack.compact_fragments.json").is_file()

    for case in execution.cases:
        assert (tmp_path / "cases" / f"{case.fixture_id}.json").is_file()


def test_context_pack_case_artifact_contains_input_output_and_metrics(tmp_path: Path) -> None:
    execution = _load_dataset_execution()
    write_token_regression_diagnostic_artifacts(execution, tmp_path)

    case_path = tmp_path / "cases" / "context_pack.compact_fragments.json"
    payload = json.loads(case_path.read_text(encoding="utf-8"))

    assert payload["input"]["fragments"]
    assert payload["output"]["fragments"]
    assert all("original_content" in fragment for fragment in payload["input"]["fragments"])
    assert all("optimized_content" in fragment for fragment in payload["output"]["fragments"])

    all_original = " ".join(
        fragment["original_content"] for fragment in payload["input"]["fragments"]
    )
    assert "Local Workspace keeps imported project documents searchable" in all_original

    assert payload["metrics"]["saved_tokens"] >= 1
    assert payload["validation"]["receipt_present"] is True
    assert payload["expectation"]["passed"] is True


def test_memory_summary_case_artifact_contains_input_output_and_metrics(tmp_path: Path) -> None:
    execution = _load_dataset_execution()
    write_token_regression_diagnostic_artifacts(execution, tmp_path)

    case_path = tmp_path / "cases" / "memory_summary.compact_summary.json"
    payload = json.loads(case_path.read_text(encoding="utf-8"))

    assert payload["input"]["original_summary"]
    assert payload["output"]["optimized_summary"]
    assert "baseline_tokens" in payload["metrics"]
    assert "optimized_tokens" in payload["metrics"]
    assert "saved_tokens" in payload["metrics"]
    assert "saved_ratio" in payload["metrics"]


def test_tool_schema_case_artifact_contains_input_output_and_metrics(tmp_path: Path) -> None:
    execution = _load_dataset_execution()
    write_token_regression_diagnostic_artifacts(execution, tmp_path)

    case_path = tmp_path / "cases" / "tool_schema.compact_catalog.json"
    payload = json.loads(case_path.read_text(encoding="utf-8"))

    assert payload["input"]["original_tool_catalog"]
    assert payload["output"]["optimized_tool_catalog"]
    assert "baseline_tokens" in payload["metrics"]
    assert "optimized_tokens" in payload["metrics"]
    assert "saved_tokens" in payload["metrics"]
    assert "saved_ratio" in payload["metrics"]


def test_diagnostic_artifacts_execute_each_fixture_once(tmp_path: Path) -> None:
    calls = 0

    def counting_runner(_counter: object) -> object:
        nonlocal calls
        calls += 1
        return _MinimalOutcome()

    fixture = TokenRegressionFixture(
        fixture_id="synthetic.single_pass",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Count runner invocations.",
        expectation=TokenRegressionExpectation(
            require_receipt=False,
            expect_validation_pass=False,
            allow_fallback=True,
        ),
        runner=counting_runner,
    )

    execution = run_token_regression_benchmark_execution(fixtures=(fixture,))
    write_token_regression_diagnostic_artifacts(execution, tmp_path)

    assert calls == 1
    assert (tmp_path / "cases" / "synthetic.single_pass.json").is_file()


def test_runner_error_case_artifact_is_written(tmp_path: Path) -> None:
    fixture = TokenRegressionFixture(
        fixture_id="synthetic.runner_boom",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Runner raises.",
        expectation=TokenRegressionExpectation(require_receipt=False),
        runner=lambda _counter: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    execution = run_token_regression_benchmark_execution(fixtures=(fixture,))
    write_token_regression_diagnostic_artifacts(execution, tmp_path)

    case_path = tmp_path / "cases" / "synthetic.runner_boom.json"
    assert case_path.is_file()
    payload = json.loads(case_path.read_text(encoding="utf-8"))

    assert payload["passed"] is False
    assert payload["validation"]["validation_status"] == "runner_error"
    assert payload["input"] == {}
    assert payload["output"] == {}
    assert any("boom" in reason for reason in payload["expectation"]["failure_reasons"])


class _MinimalOutcome:
    """Minimal stand-in outcome for single-pass counting test."""

    original_content = "one two three four"
    optimized_content = "one two"
    original_tokens = 4
    optimized_tokens = 2
    saved_tokens = 2
    saved_ratio = 0.5
    validation_status = "passed"
    fallback_status = False
    receipt = object()
    strategy = None
