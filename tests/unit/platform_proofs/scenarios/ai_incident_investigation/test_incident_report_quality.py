# © Artur Czarnecki. All rights reserved.

from __future__ import annotations
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_fixture_runtime_bundle, build_runtime_bundle

import re

import pytest

from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator_evidence import (
    evaluator_pass_summary,
    is_private_truth_check_id,
    project_scenario_evaluation_to_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evidence_builder import (
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    FORBIDDEN_LEAK_MARKERS,
    ScenarioVariant,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.report_sections import (
    build_incident_report_sections,
    incident_report_extra_css,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    execute_resolved_skeleton,
)
from scripts.proof.intergrax_platform_proof_html_renderer import (
    assert_no_external_report_dependencies,
    render_platform_proof_report,
)

pytestmark = pytest.mark.unit

_LEAK_RE = re.compile("|".join(re.escape(marker) for marker in FORBIDDEN_LEAK_MARKERS), re.I)
_CORRUPTION_RE = re.compile(
    r"Traceback|NoneType|undefined|\[object Object\]|private_truth|expected_hypothesis|initiating_factor_code",
    re.I,
)


async def _build_report_pair() -> tuple[str, str, object, object]:
    resolved_fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    resolved_bundle = resolved_fixture_bundle.bundle
    unresolved_fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    unresolved_bundle = unresolved_fixture_bundle.bundle
    resolved_result = await execute_resolved_skeleton(resolved_bundle)
    unresolved_result = await execute_resolved_skeleton(unresolved_bundle)
    resolved_evaluation = evaluate_scenario_run(resolved_result, resolved_fixture_bundle.fixture)
    unresolved_evaluation = evaluate_scenario_run(unresolved_result, unresolved_fixture_bundle.fixture)

    resolved_evidence = build_platform_proof_evidence(
        resolved_result,
        evaluation=resolved_evaluation,
        variant=ScenarioVariant.RESOLVED,
        source_revision="testsha",
    )
    unresolved_evidence = build_platform_proof_evidence(
        unresolved_result,
        evaluation=unresolved_evaluation,
        variant=ScenarioVariant.UNRESOLVED,
        source_revision="testsha",
    )

    resolved_html = render_platform_proof_report(
        resolved_evidence,
        domain_sections=build_incident_report_sections(
            result=resolved_result,
            evaluation=resolved_evaluation,
            evidence=resolved_evidence,
            variant=ScenarioVariant.RESOLVED,
        ),
        extra_css=incident_report_extra_css(),
    )
    unresolved_html = render_platform_proof_report(
        unresolved_evidence,
        domain_sections=build_incident_report_sections(
            result=unresolved_result,
            evaluation=unresolved_evaluation,
            evidence=unresolved_evidence,
            variant=ScenarioVariant.UNRESOLVED,
        ),
        extra_css=incident_report_extra_css(),
    )
    return resolved_html, unresolved_html, resolved_evaluation, unresolved_evaluation


def _decision_section(html: str) -> str:
    start = html.index('id="incident-investigation-result"')
    end = html.index('id="incident-defensibility"', start)
    return html[start:end]


@pytest.mark.asyncio
async def test_resolved_report_60_second_reader_contract() -> None:
    html, _, evaluation, _ = await _build_report_pair()
    decision = _decision_section(html).lower()
    passed_count, total_count, _ = evaluator_pass_summary(evaluation)

    for phrase in (
        "proof result",
        "pass",
        "incident outcome",
        "resolved",
        "production overload",
        "correlation",
        "comparison",
        "attendance",
        "telemetry",
        "h1",
        "superseded",
        "h2",
        "rejected",
        "h3",
        "supported",
        "independent",
        "evaluator",
        f"{passed_count}/{total_count}",
    ):
        assert phrase in decision, phrase


@pytest.mark.asyncio
async def test_unresolved_report_60_second_reader_contract() -> None:
    _, html, _, evaluation = await _build_report_pair()
    decision = _decision_section(html).lower()
    passed_count, total_count, _ = evaluator_pass_summary(evaluation)

    for phrase in (
        "proof result",
        "pass",
        "incident outcome",
        "unresolved",
        "no root-cause diagnosis accepted",
        "telemetry",
        "unavailable",
        "h1",
        "superseded",
        "h2",
        "rejected",
        "h3",
        "insufficient evidence",
        "challenge remains open",
        "pass means the proof behaved correctly",
        "did not justify accepting a root-cause diagnosis",
        f"{passed_count}/{total_count}",
    ):
        assert phrase in decision, phrase


@pytest.mark.asyncio
async def test_cross_path_contamination_absent_from_decision_sections() -> None:
    resolved_html, unresolved_html, _, _ = await _build_report_pair()
    resolved_decision = _decision_section(resolved_html).lower()
    unresolved_decision = _decision_section(unresolved_html).lower()

    assert "no root-cause diagnosis accepted" not in resolved_decision
    assert "decisive telemetry unavailable" not in resolved_decision
    assert "challenge remains open" not in resolved_decision

    assert "h3 supported" not in unresolved_decision
    assert "challenge was satisfied" not in unresolved_decision
    assert "telemetry showed" not in unresolved_decision


@pytest.mark.asyncio
async def test_evaluator_evidence_matches_runtime_evaluation() -> None:
    resolved_fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    resolved_bundle = resolved_fixture_bundle.bundle
    unresolved_fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    unresolved_bundle = unresolved_fixture_bundle.bundle
    resolved_result = await execute_resolved_skeleton(resolved_bundle)
    unresolved_result = await execute_resolved_skeleton(unresolved_bundle)
    resolved_evaluation = evaluate_scenario_run(resolved_result, resolved_fixture_bundle.fixture)
    unresolved_evaluation = evaluate_scenario_run(unresolved_result, unresolved_fixture_bundle.fixture)

    resolved_evidence = build_platform_proof_evidence(
        resolved_result,
        evaluation=resolved_evaluation,
        variant=ScenarioVariant.RESOLVED,
        source_revision="testsha",
    )
    unresolved_evidence = build_platform_proof_evidence(
        unresolved_result,
        evaluation=unresolved_evaluation,
        variant=ScenarioVariant.UNRESOLVED,
        source_revision="testsha",
    )

    for evidence, evaluation in (
        (resolved_evidence, resolved_evaluation),
        (unresolved_evidence, unresolved_evaluation),
    ):
        assert evidence.evaluator is not None
        passed_checks = [c for c in evidence.evaluator.checks if c.passed]
        failed_checks = [c for c in evidence.evaluator.checks if not c.passed]
        assert len(passed_checks) == len(evaluation.checks)
        assert len(failed_checks) == len(evaluation.failures)
        assert tuple(c.check_id for c in passed_checks) == evaluation.checks
        assert tuple(c.check_id for c in failed_checks) == evaluation.failures
        assert evidence.evaluator.failure_reasons == evaluation.failures


@pytest.mark.asyncio
async def test_private_truth_labels_not_exposed_in_decision_sections() -> None:
    resolved_html, unresolved_html, _, _ = await _build_report_pair()
    for html in (resolved_html, unresolved_html):
        decision = _decision_section(html).lower()
        assert "private_truth_consistent" not in decision
        assert "expected_hypothesis" not in decision
        assert not _LEAK_RE.search(decision)
        assert not _CORRUPTION_RE.search(decision)


@pytest.mark.asyncio
async def test_report_safe_no_external_dependencies() -> None:
    resolved_html, unresolved_html, _, _ = await _build_report_pair()
    for html in (resolved_html, unresolved_html):
        assert_no_external_report_dependencies(html)
        assert "<script" not in html.lower()


@pytest.mark.asyncio
async def test_path_specific_limitations_in_evidence() -> None:
    resolved_fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    resolved_bundle = resolved_fixture_bundle.bundle
    unresolved_fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    unresolved_bundle = unresolved_fixture_bundle.bundle
    resolved_result = await execute_resolved_skeleton(resolved_bundle)
    unresolved_result = await execute_resolved_skeleton(unresolved_bundle)
    resolved_evaluation = evaluate_scenario_run(resolved_result, resolved_fixture_bundle.fixture)
    unresolved_evaluation = evaluate_scenario_run(unresolved_result, unresolved_fixture_bundle.fixture)

    resolved_evidence = build_platform_proof_evidence(
        resolved_result,
        evaluation=resolved_evaluation,
        variant=ScenarioVariant.RESOLVED,
        source_revision="testsha",
    )
    unresolved_evidence = build_platform_proof_evidence(
        unresolved_result,
        evaluation=unresolved_evaluation,
        variant=ScenarioVariant.UNRESOLVED,
        source_revision="testsha",
    )

    assert "RESOLVED path only" in resolved_evidence.limitations[0]
    assert "UNRESOLVED path only" in unresolved_evidence.limitations[0]
    assert resolved_result.outcome == OUTCOME_RESOLVED
    assert unresolved_result.outcome == OUTCOME_UNRESOLVED


def test_private_truth_check_ids_are_report_safe_labels() -> None:
    projected = project_scenario_evaluation_to_evidence(
        type(
            "E",
            (),
            {
                "passed": True,
                "checks": ("private_truth_consistent",),
                "failures": (),
            },
        )()
    )
    label = projected.checks[0].label.lower()
    assert "private_truth" not in label
    assert is_private_truth_check_id("private_truth_consistent")
