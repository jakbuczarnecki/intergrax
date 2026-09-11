# © Artur Czarnecki. All rights reserved.

"""DIAG R3 — multi-agent failure localization qualification (R3-A1–A7)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.contracts.multi_agent_failure_localization import (
    DiagnosticFailureTopologyCompleteness,
    FailureBoundaryCertainty,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFindingKind
from testing_support.runtime.multi_agent_failure_localization_r1_harness import (
    build_multi_agent_failure_localization_scenario,
    open_lineage_chain,
    open_lineage_fan_out,
    record_execution_failed,
)

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _boundary_ids(assessment) -> frozenset[str]:
    analysis = assessment.failure_boundary_analysis
    assert analysis is not None
    return frozenset(
        str(boundary.execution_id) for boundary in analysis.topology.failed_boundaries
    )


def test_r3_a1_exact_child_boundary() -> None:
    e1, e2, e3, e4 = [mint_execution_id() for _ in range(4)]
    scenario = build_multi_agent_failure_localization_scenario()
    open_lineage_chain(scenario, (e1, e2, e3, e4))
    record_execution_failed(scenario, e4)

    assessment = scenario.assess()
    assert _boundary_ids(assessment) == frozenset({str(e4)})


def test_r3_a2_nested_depth_three() -> None:
    e1, e2, e3, e4 = [mint_execution_id() for _ in range(4)]
    scenario = build_multi_agent_failure_localization_scenario()
    open_lineage_chain(scenario, (e1, e2, e3, e4))
    record_execution_failed(scenario, e4)

    analysis = scenario.assess().failure_boundary_analysis
    assert analysis is not None
    topology = analysis.topology
    assert _boundary_ids(scenario.assess()) == frozenset({str(e4)})
    assert {str(x) for x in topology.affected_executions} == {str(e1), str(e2), str(e3)}


def test_r3_a3_sibling_isolation() -> None:
    e1, e2, e3, e4 = [mint_execution_id() for _ in range(4)]
    scenario = build_multi_agent_failure_localization_scenario()
    open_lineage_fan_out(
        scenario,
        e1,
        ((e2, e1), (e3, e1), (e4, e1)),
    )
    record_execution_failed(scenario, e3)

    topology = scenario.assess().failure_boundary_analysis.topology
    assert _boundary_ids(scenario.assess()) == frozenset({str(e3)})
    assert {str(x) for x in topology.healthy_executions} == {str(e2), str(e4)}
    assert {str(x) for x in topology.affected_executions} == {str(e1)}


def test_r3_a4_multi_failure() -> None:
    e1, e2, e3, e4 = [mint_execution_id() for _ in range(4)]
    scenario = build_multi_agent_failure_localization_scenario()
    open_lineage_fan_out(
        scenario,
        e1,
        ((e2, e1), (e3, e1), (e4, e1)),
    )
    record_execution_failed(scenario, e2)
    record_execution_failed(scenario, e3)

    topology = scenario.assess().failure_boundary_analysis.topology
    assert _boundary_ids(scenario.assess()) == frozenset({str(e2), str(e3)})
    assert str(e4) in {str(x) for x in topology.healthy_executions}


def test_r3_a5_no_causal_inference_parent_not_boundary() -> None:
    e1, e2, e3, e4 = [mint_execution_id() for _ in range(4)]
    scenario = build_multi_agent_failure_localization_scenario()
    open_lineage_chain(scenario, (e1, e2, e3, e4))
    record_execution_failed(scenario, e4)

    analysis = scenario.assess().failure_boundary_analysis
    assert analysis is not None
    boundaries = analysis.topology.failed_boundaries
    assert str(e1) not in {str(b.execution_id) for b in boundaries}
    assert str(e1) in {str(x) for x in analysis.topology.affected_executions}
    assert not hasattr(analysis, "root_cause_execution_id")
    assert not hasattr(analysis, "cause_execution_ids")


def test_r3_a6_missing_evidence() -> None:
    e1, e2, e3 = [mint_execution_id() for _ in range(3)]
    scenario = build_multi_agent_failure_localization_scenario()
    open_lineage_chain(scenario, (e1, e2, e3))

    analysis = scenario.assess().failure_boundary_analysis
    assert analysis is not None
    assert analysis.boundaries == ()
    assert analysis.topology.failed_boundaries == ()
    assert (
        analysis.topology.completeness
        is DiagnosticFailureTopologyCompleteness.UNAVAILABLE
    )
    failure_findings = [
        f for f in scenario.assess().findings if f.kind is DiagnosticFindingKind.EXECUTION_FAILED
    ]
    assert failure_findings == []


def test_r3_a7_lineage_unavailable_preserves_failure_evidence() -> None:
    e_fail = mint_execution_id()
    scenario = build_multi_agent_failure_localization_scenario(with_lineage=False)
    record_execution_failed(scenario, e_fail)

    analysis = scenario.assess().failure_boundary_analysis
    assert analysis is not None
    assert len(analysis.topology.failed_boundaries) == 1
    boundary = analysis.topology.failed_boundaries[0]
    assert str(boundary.execution_id) == str(e_fail)
    assert boundary.certainty is FailureBoundaryCertainty.PROVEN
    assert (
        analysis.topology.completeness
        is DiagnosticFailureTopologyCompleteness.UNAVAILABLE
    )
    assert analysis.topology.affected_executions == ()
