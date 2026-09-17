# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001C-H — public observation grouping SPI controls central Problem grouping."""

from __future__ import annotations

import pytest

from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_bridge import (
    build_reliability_diagnostic_emitter,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_strategy_composition import (
    build_reliability_case_default_grouping_strategy,
)
from tests.unit.erl_diagnostics_plugins.correlation_grouping_strategy import (
    CorrelationGroupingStrategy,
)
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    build_diagnostic_orchestrator_stack_for_tests,
    query_all_occurrences_for_problem,
    query_all_problems_for_tenant,
)
from tests.unit.runtime.diagnostics.reliability.test_erl_diag_001c_grouping_occurrence import (
    _TENANT_A,
    _TENANT_B,
    _observation,
)

pytestmark = pytest.mark.unit


def _stack_with_observation_grouping(
    observation_grouping: CorrelationGroupingStrategy,
):
    orchestrator, persistence, read_service, occurrence_persistence = (
        build_diagnostic_orchestrator_stack_for_tests(
            observation_grouping=observation_grouping,
        )
    )
    emitter = build_reliability_diagnostic_emitter(
        orchestrator,
        observation_grouping=observation_grouping,
    )
    return emitter, persistence, occurrence_persistence


def test_custom_correlation_strategy_merges_distinct_cases() -> None:
    plugin = CorrelationGroupingStrategy()
    emitter, persistence, occurrence_persistence = _stack_with_observation_grouping(plugin)
    emitter.emit(
        _observation(
            reliability_case_id="CASE-A",
            correlation_id="CORR-X",
            observation_id="OBS-A",
        ),
    )
    emitter.emit(
        _observation(
            reliability_case_id="CASE-B",
            correlation_id="CORR-X",
            observation_id="OBS-B",
        ),
    )

    problems = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems) == 1
    assert problems[0].occurrence_count == 2
    occurrences = query_all_occurrences_for_problem(
        occurrence_persistence,
        tenant_id=_TENANT_A,
        problem_id=problems[0].problem_id,
    )
    assert len(occurrences) == 2


def test_default_strategy_keeps_distinct_cases_separate() -> None:
    orchestrator, persistence, _, _ = build_diagnostic_orchestrator_stack_for_tests()
    emitter = build_reliability_diagnostic_emitter(orchestrator)
    emitter.emit(
        _observation(
            reliability_case_id="CASE-A",
            correlation_id="CORR-X",
            observation_id="OBS-A",
        ),
    )
    emitter.emit(
        _observation(
            reliability_case_id="CASE-B",
            correlation_id="CORR-X",
            observation_id="OBS-B",
        ),
    )
    assert len(query_all_problems_for_tenant(persistence, _TENANT_A)) == 2


def test_custom_strategy_replay_and_second_observation() -> None:
    plugin = CorrelationGroupingStrategy()
    emitter, persistence, occurrence_persistence = _stack_with_observation_grouping(plugin)
    obs_a = _observation(
        reliability_case_id="CASE-A",
        correlation_id="CORR-X",
        observation_id="OBS-A",
    )
    emitter.emit(obs_a)
    emitter.emit(obs_a)
    problems = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems) == 1
    assert problems[0].occurrence_count == 1

    emitter.emit(
        _observation(
            reliability_case_id="CASE-B",
            correlation_id="CORR-X",
            observation_id="OBS-B",
        ),
    )
    problems = query_all_problems_for_tenant(persistence, _TENANT_A)
    assert len(problems) == 1
    assert problems[0].occurrence_count == 2


def test_custom_strategy_tenant_isolation() -> None:
    plugin = CorrelationGroupingStrategy()
    emitter, persistence, _ = _stack_with_observation_grouping(plugin)
    emitter.emit(
        _observation(
            tenant_id=_TENANT_A,
            reliability_case_id="CASE-A",
            correlation_id="CORR-SHARED",
            observation_id="OBS-A",
        ),
    )
    emitter.emit(
        _observation(
            tenant_id=_TENANT_B,
            reliability_case_id="CASE-B",
            correlation_id="CORR-SHARED",
            observation_id="OBS-B",
        ),
    )
    assert len(query_all_problems_for_tenant(persistence, _TENANT_A)) == 1
    assert len(query_all_problems_for_tenant(persistence, _TENANT_B)) == 1


def test_plugin_strategy_does_not_import_runtime_package() -> None:
    import importlib
    from pathlib import Path

    mod = importlib.import_module(
        "tests.unit.erl_diagnostics_plugins.correlation_grouping_strategy",
    )
    source_path = mod.__file__
    assert source_path is not None
    text = Path(source_path).read_text(encoding="utf-8")
    assert "intergrax.runtime" not in text


def test_shared_composition_wires_matching_batch_and_observation_strategies() -> None:
    plugin = CorrelationGroupingStrategy()
    batch, obs = build_reliability_case_default_grouping_strategy(observation_grouping=plugin)
    assert batch is not None
    assert obs is plugin
