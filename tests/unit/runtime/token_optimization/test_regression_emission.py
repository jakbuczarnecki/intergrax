# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-1E: policy-gated regression benchmark emission wrapper tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.token_optimization.domain_events import (
    TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND,
    TOKEN_OPTIMIZATION_SIGNAL_PAYLOAD_SCHEMA_ID,
    TokenOptimizationSignalPayloadV1,
    register_token_optimization_domain_signal,
)
from intergrax.runtime.token_optimization.emission import (
    TokenOptimizationEmissionPolicy,
    TokenOptimizationEmissionStatus,
)
from intergrax.runtime.token_optimization.regression import (
    default_token_counter,
    run_token_regression_benchmarks,
)
from intergrax.runtime.token_optimization.regression_emission import (
    TokenRegressionEmissionRunResult,
    run_token_regression_benchmarks_with_emission,
)
from intergrax.runtime.token_optimization.signals import TokenOptimizationSignalType

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REGRESSION_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "token_optimization" / "regression.py"
)
_BENCHMARK_SCRIPT = _REPO_ROOT / "scripts" / "check_token_regression_benchmarks.py"
_SCOPE_GUARD_MODULES = (_REGRESSION_MODULE, _BENCHMARK_SCRIPT)
_SCOPE_GUARD_IMPORTS = (
    "token_optimization.regression_emission",
    "token_optimization.emission",
    "regression_emission",
)


@pytest.fixture(autouse=True)
def _register_token_optimization_domain_kind() -> None:
    clear_event_kind_registry()
    register_token_optimization_domain_signal()
    yield
    clear_event_kind_registry()


def _emit_context() -> tuple[EmitContext, RuntimeEventBus]:
    bus = RuntimeEventBus(record_history=True)
    ctx = EmitContext(
        task_id="task-1",
        run_id="run-1",
        tenant_id="tenant-a",
        bus=bus,
    )
    return ctx, bus


def _enabled_emission_policy() -> TokenOptimizationEmissionPolicy:
    return TokenOptimizationEmissionPolicy(enabled=True)


def test_default_policy_emits_zero_events_and_returns_skipped_statuses() -> None:
    ctx, bus = _emit_context()

    run_result = run_token_regression_benchmarks_with_emission(ctx)

    assert isinstance(run_result, TokenRegressionEmissionRunResult)
    assert run_result.emitted_event_count == 0
    assert len(run_result.result_emissions) == len(run_result.summary.results)
    assert run_result.summary_emission is not None
    assert all(
        emission.status == TokenOptimizationEmissionStatus.SKIPPED_DISABLED
        for emission in run_result.result_emissions
    )
    assert (
        run_result.summary_emission.status
        == TokenOptimizationEmissionStatus.SKIPPED_DISABLED
    )
    assert len(bus.history) == 0


def test_enabled_policy_emits_one_event_per_result_plus_summary() -> None:
    ctx, bus = _emit_context()

    run_result = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=_enabled_emission_policy(),
    )

    expected_count = len(run_result.summary.results) + 1
    assert run_result.emitted_event_count == expected_count
    assert all(emission.emitted for emission in run_result.result_emissions)
    assert run_result.summary_emission is not None
    assert run_result.summary_emission.emitted is True
    assert len(bus.history) == expected_count


def test_emit_results_false_emits_only_summary() -> None:
    ctx, bus = _emit_context()

    run_result = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=_enabled_emission_policy(),
        emit_results=False,
    )

    assert run_result.result_emissions == ()
    assert run_result.summary_emission is not None
    assert run_result.summary_emission.emitted is True
    assert run_result.emitted_event_count == 1
    assert len(bus.history) == 1


def test_emit_summary_false_emits_only_results() -> None:
    ctx, bus = _emit_context()

    run_result = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=_enabled_emission_policy(),
        emit_summary=False,
    )

    assert len(run_result.result_emissions) == len(run_result.summary.results)
    assert run_result.summary_emission is None
    assert run_result.emitted_event_count == len(run_result.summary.results)
    assert len(bus.history) == len(run_result.summary.results)


def test_policy_gate_disables_result_emission() -> None:
    ctx, bus = _emit_context()
    policy = TokenOptimizationEmissionPolicy(
        enabled=True,
        emit_regression_results=False,
    )

    run_result = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=policy,
    )

    assert all(
        emission.status == TokenOptimizationEmissionStatus.SKIPPED_KIND_DISABLED
        for emission in run_result.result_emissions
    )
    assert run_result.summary_emission is not None
    assert run_result.summary_emission.emitted is True
    assert run_result.emitted_event_count == 1
    assert len(bus.history) == 1


def test_policy_gate_disables_summary_emission() -> None:
    ctx, bus = _emit_context()
    policy = TokenOptimizationEmissionPolicy(
        enabled=True,
        emit_regression_summaries=False,
    )

    run_result = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=policy,
    )

    assert all(emission.emitted for emission in run_result.result_emissions)
    assert run_result.summary_emission is not None
    assert (
        run_result.summary_emission.status
        == TokenOptimizationEmissionStatus.SKIPPED_KIND_DISABLED
    )
    assert run_result.emitted_event_count == len(run_result.summary.results)
    assert len(bus.history) == len(run_result.summary.results)


def test_dry_run_policy_emits_zero_events() -> None:
    ctx, bus = _emit_context()
    policy = TokenOptimizationEmissionPolicy(enabled=True, dry_run=True)

    run_result = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=policy,
    )

    assert run_result.emitted_event_count == 0
    assert all(
        emission.status == TokenOptimizationEmissionStatus.DRY_RUN
        for emission in run_result.result_emissions
    )
    assert (
        run_result.summary_emission is not None
        and run_result.summary_emission.status == TokenOptimizationEmissionStatus.DRY_RUN
    )
    assert all(isinstance(emission.payload, TokenOptimizationSignalPayloadV1) for emission in run_result.result_emissions)
    assert len(bus.history) == 0


def test_emitted_events_use_token_optimization_signal_kind_and_schema() -> None:
    ctx, bus = _emit_context()

    run_result = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=_enabled_emission_policy(),
    )

    for emission in run_result.result_emissions:
        assert emission.event is not None
        assert emission.event.event_type == RuntimeEventType.DOMAIN_SIGNAL
        assert emission.event.event_kind == TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND
        assert (
            emission.event.payload["payload_schema_id"]
            == TOKEN_OPTIMIZATION_SIGNAL_PAYLOAD_SCHEMA_ID
        )
        assert emission.signal.signal_type is TokenOptimizationSignalType.REGRESSION_RESULT

    assert run_result.summary_emission is not None
    assert run_result.summary_emission.event is not None
    assert run_result.summary_emission.event.event_kind == TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND
    assert (
        run_result.summary_emission.signal.signal_type
        is TokenOptimizationSignalType.REGRESSION_SUMMARY
    )
    assert len(bus.history) == run_result.emitted_event_count


def test_wrapper_returns_same_summary_as_core_runner() -> None:
    ctx, _bus = _emit_context()
    expected = run_token_regression_benchmarks(token_counter=default_token_counter)

    run_result = run_token_regression_benchmarks_with_emission(ctx)

    assert run_result.summary.total_fixtures == expected.total_fixtures
    assert run_result.summary.passed == expected.passed
    assert run_result.summary.failed == expected.failed
    assert len(run_result.summary.results) == len(expected.results)


@pytest.mark.parametrize("module_path", _SCOPE_GUARD_MODULES, ids=lambda p: p.name)
def test_regression_core_and_benchmark_script_do_not_import_emission_wrapper(
    module_path: Path,
) -> None:
    source = module_path.read_text(encoding="utf-8")

    for import_name in _SCOPE_GUARD_IMPORTS:
        assert import_name not in source
