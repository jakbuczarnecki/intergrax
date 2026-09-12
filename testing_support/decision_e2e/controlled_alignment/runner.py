# © Artur Czarnecki. All rights reserved.

"""Controlled alignment qualification execution (fixture path, production contracts)."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    trace_reader_from_composition,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    ScenarioExecutionResult,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    FixtureRuntimeBundle,
    build_fixture_runtime_bundle,
)
from testing_support.decision_e2e.controlled_alignment.evidence import (
    ControlledAlignmentRepairEvidence,
    extract_repair_evidence,
)
from testing_support.decision_e2e.controlled_alignment.scenario import (
    ControlledAlignmentScenario,
    MODEL_OVERCOMMIT_SCENARIO,
)
from testing_support.decision_e2e.controlled_alignment.stimulus_llm import (
    ModelOvercommitStimulusLLM,
)


@dataclass(frozen=True, slots=True)
class ControlledAlignmentRunResult:
    scenario: ControlledAlignmentScenario
    execution: ScenarioExecutionResult
    trace_events: tuple[dict[str, object], ...]
    repair_evidence: ControlledAlignmentRepairEvidence
    revision_system_contents: tuple[str, ...]


def _trace_events_for_execution(
    fixture_bundle: FixtureRuntimeBundle,
    result: ScenarioExecutionResult,
) -> tuple[dict[str, object], ...]:
    provenance = result.execution_provenance
    if provenance is None:
        return ()
    reader = trace_reader_from_composition(fixture_bundle.bundle.runtime_composition)
    if reader is None:
        return ()
    persisted = reader.read_run(
        str(provenance.platform_run_id),
        provenance.execution_tenant_id,
    )
    return tuple(dict(item) for item in persisted.events if isinstance(item, dict))


def _revision_system_contents(
    llm: ModelOvercommitStimulusLLM,
) -> tuple[str, ...]:
    return tuple(
        message.content or ""
        for message in llm.revision_messages
        if message.role == "system"
    )


async def run_controlled_alignment_scenario(
    scenario: ControlledAlignmentScenario,
    *,
    fixture_bundle: FixtureRuntimeBundle | None = None,
    llm: ModelOvercommitStimulusLLM | None = None,
) -> ControlledAlignmentRunResult:
    stimulus_llm = llm or ModelOvercommitStimulusLLM()
    resolved_fixture = fixture_bundle or build_fixture_runtime_bundle(
        llm_adapter_override=stimulus_llm,
    )
    execution = await execute_resolved_skeleton(resolved_fixture.bundle)
    trace_events = _trace_events_for_execution(resolved_fixture, execution)
    revision_contents = _revision_system_contents(stimulus_llm)
    repair = extract_repair_evidence(
        trace_events=trace_events,
        execution=execution,
        revision_system_contents=revision_contents,
        expected_direction=scenario.expected_direction,
    )
    return ControlledAlignmentRunResult(
        scenario=scenario,
        execution=execution,
        trace_events=trace_events,
        repair_evidence=repair,
        revision_system_contents=revision_contents,
    )


async def run_model_overcommit_controlled_qualification() -> ControlledAlignmentRunResult:
    return await run_controlled_alignment_scenario(MODEL_OVERCOMMIT_SCENARIO)


def run_result_to_run_record(result: ControlledAlignmentRunResult) -> dict[str, object]:
    provenance = result.execution.execution_provenance
    run_id = str(provenance.platform_run_id) if provenance is not None else ""
    return {
        "run_id": run_id,
        "scenario_id": result.scenario.scenario_id,
        "outcome": result.execution.outcome,
        "revision_pass": result.execution.revision_pass,
        "evaluator_loop_iterations": result.execution.evaluator_loop_iterations,
        "repair_status": result.repair_evidence.repair_status.value,
        "trace_events": [dict(item) for item in result.trace_events],
        "alignment_evidence": {
            "direction": (
                result.repair_evidence.alignment.correction_direction.value
                if result.repair_evidence.alignment.correction_direction
                else ""
            ),
            "correctable": result.repair_evidence.alignment.correctable,
            "revision_context_valid": result.repair_evidence.revision_context_valid,
            "attempt_indices": list(result.repair_evidence.attempt_indices),
        },
        "execution_summary": {
            "outcome": result.execution.outcome,
            "revision_pass": result.execution.revision_pass,
            "evaluator_loop_iterations": result.execution.evaluator_loop_iterations,
            "critic_verdict_passed": result.execution.critic_verdict_passed,
        },
    }


__all__ = [
    "ControlledAlignmentRunResult",
    "run_controlled_alignment_scenario",
    "run_model_overcommit_controlled_qualification",
    "run_result_to_run_record",
]
