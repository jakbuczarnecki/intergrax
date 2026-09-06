# © Artur Czarnecki. All rights reserved.

"""Typed DS-E2E-12 model qualification matrix runner (test/qualification side only)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import StrEnum

from testing_support.decision_e2e.preflight import verify_required_ollama_models
from testing_support.decision_e2e.scenario_qualification import (
    ScenarioQualificationAttempt,
    run_ai_incident_live_qualification,
)


class ModelQualificationFailureClass(StrEnum):
    QUALIFICATION_CANDIDATE_PASS = "QUALIFICATION_CANDIDATE_PASS"
    MODEL_CONTRACT_FAILURE = "MODEL_CONTRACT_FAILURE"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RESOURCE_INSUFFICIENT = "RESOURCE_INSUFFICIENT"
    CONFIGURATION_BLOCKED = "CONFIGURATION_BLOCKED"
    PLATFORM_DEFECT = "PLATFORM_DEFECT"


@dataclass(frozen=True, slots=True)
class ModelQualificationCandidate:
    provider_id: str
    model_id: str
    availability: str = "LOCAL_AVAILABLE"


@dataclass(frozen=True, slots=True)
class ModelQualificationObservation:
    provider_id: str
    model_id: str
    executed: bool
    decision_path_exercised: bool
    evaluation_passed: bool
    failure_class: ModelQualificationFailureClass
    used_mock_provider: bool
    revision_occurred: bool
    tools_used: int
    final_critic_verdict: bool | None
    terminal_outcome: str | None
    initial_critic_errors: tuple[str, ...]
    evaluator_failures: tuple[str, ...]
    block_reason: str | None
    wall_clock_sec: float | None = None
    model_invocation_count: int | None = None
    revision_iterations: int | None = None


def _provider_error_markers(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return any(
        token in lowered
        for token in (
            "connection",
            "timeout",
            "unreachable",
            "adapter",
            "parse",
            "json",
            "structured",
            "ollama api",
        )
    )


def _classify_attempt(
    attempt: ScenarioQualificationAttempt,
    *,
    provider_id: str,
    model_id: str,
) -> ModelQualificationFailureClass:
    evidence = attempt.evidence
    if evidence.used_mock_provider:
        return ModelQualificationFailureClass.CONFIGURATION_BLOCKED
    if not evidence.executed:
        reason = (evidence.block_reason or attempt.error or "").lower()
        if "missing required ollama models" in reason or "model" in reason and "missing" in reason:
            return ModelQualificationFailureClass.MODEL_UNAVAILABLE
        if _provider_error_markers(attempt.error or evidence.block_reason):
            return ModelQualificationFailureClass.PROVIDER_FAILURE
        if "resource" in reason or "memory" in reason or "vram" in reason:
            return ModelQualificationFailureClass.RESOURCE_INSUFFICIENT
        if "configuration" in reason or "credential" in reason or "adapter unavailable" in reason:
            return ModelQualificationFailureClass.CONFIGURATION_BLOCKED
        return ModelQualificationFailureClass.PROVIDER_FAILURE
    if evidence.decision_path_exercised is False:
        return ModelQualificationFailureClass.PLATFORM_DEFECT
    if attempt.evaluation_passed:
        return ModelQualificationFailureClass.QUALIFICATION_CANDIDATE_PASS
    if _provider_error_markers(attempt.error):
        return ModelQualificationFailureClass.PROVIDER_FAILURE
    return ModelQualificationFailureClass.MODEL_CONTRACT_FAILURE


def _observation_from_attempt(
    attempt: ScenarioQualificationAttempt,
    *,
    provider_id: str,
    model_id: str,
    wall_clock_sec: float,
) -> ModelQualificationObservation:
    evidence = attempt.evidence
    failure_class = _classify_attempt(
        attempt,
        provider_id=provider_id,
        model_id=model_id,
    )
    initial_critic_errors: tuple[str, ...] = ()
    revision_occurred = False
    tools_used = 0
    final_critic_verdict: bool | None = None
    terminal_outcome = evidence.outcome
    revision_iterations: int | None = None
    model_invocation_count: int | None = None

    if evidence.executed and attempt.error is None:
        # Outcome fields are only present on successful execution path.
        pass

    if attempt.error and failure_class is ModelQualificationFailureClass.MODEL_CONTRACT_FAILURE:
        initial_critic_errors = tuple(attempt.error.split("; "))

    return ModelQualificationObservation(
        provider_id=provider_id,
        model_id=model_id,
        executed=evidence.executed,
        decision_path_exercised=bool(evidence.decision_path_exercised),
        evaluation_passed=attempt.evaluation_passed,
        failure_class=failure_class,
        used_mock_provider=evidence.used_mock_provider,
        revision_occurred=revision_occurred,
        tools_used=tools_used,
        final_critic_verdict=final_critic_verdict,
        terminal_outcome=terminal_outcome,
        initial_critic_errors=initial_critic_errors,
        evaluator_failures=initial_critic_errors,
        block_reason=evidence.block_reason or attempt.error,
        wall_clock_sec=wall_clock_sec,
        model_invocation_count=model_invocation_count,
        revision_iterations=revision_iterations,
    )


async def run_model_qualification_diagnostic(
    candidate: ModelQualificationCandidate,
    *,
    env_prefix: str = "INTERGRAX_LLM",
) -> ModelQualificationObservation:
    """Run one DS-E2E-12 diagnostic execution for a provider/model candidate."""
    if candidate.provider_id == "ollama":
        ok, reason = verify_required_ollama_models(frozenset({candidate.model_id}))
        if not ok:
            return ModelQualificationObservation(
                provider_id=candidate.provider_id,
                model_id=candidate.model_id,
                executed=False,
                decision_path_exercised=False,
                evaluation_passed=False,
                failure_class=ModelQualificationFailureClass.MODEL_UNAVAILABLE,
                used_mock_provider=False,
                revision_occurred=False,
                tools_used=0,
                final_critic_verdict=None,
                terminal_outcome=None,
                initial_critic_errors=(),
                evaluator_failures=(),
                block_reason=reason,
                wall_clock_sec=0.0,
            )

    prior_provider = os.environ.get(f"{env_prefix}_PROVIDER")
    prior_model = os.environ.get(f"{env_prefix}_MODEL")
    os.environ[f"{env_prefix}_PROVIDER"] = candidate.provider_id
    os.environ[f"{env_prefix}_MODEL"] = candidate.model_id
    os.environ["INTERGRAX_DECISION_E2E_QUALIFICATION"] = "1"

    started = time.perf_counter()
    try:
        attempt = await run_ai_incident_live_qualification()
    finally:
        if prior_provider is None:
            os.environ.pop(f"{env_prefix}_PROVIDER", None)
        else:
            os.environ[f"{env_prefix}_PROVIDER"] = prior_provider
        if prior_model is None:
            os.environ.pop(f"{env_prefix}_MODEL", None)
        else:
            os.environ[f"{env_prefix}_MODEL"] = prior_model

    observation = _observation_from_attempt(
        attempt,
        provider_id=candidate.provider_id,
        model_id=candidate.model_id,
        wall_clock_sec=time.perf_counter() - started,
    )
    return _enrich_from_scenario_result(observation, attempt)


def _enrich_from_scenario_result(
    observation: ModelQualificationObservation,
    attempt: ScenarioQualificationAttempt,
) -> ModelQualificationObservation:
    if not attempt.evidence.executed:
        return observation
    failures = tuple(
        part.strip()
        for part in (attempt.error or "").split(";")
        if part.strip()
    )
    if not failures:
        return observation
    return ModelQualificationObservation(
        provider_id=observation.provider_id,
        model_id=observation.model_id,
        executed=observation.executed,
        decision_path_exercised=observation.decision_path_exercised,
        evaluation_passed=observation.evaluation_passed,
        failure_class=observation.failure_class,
        used_mock_provider=observation.used_mock_provider,
        revision_occurred=observation.revision_occurred,
        tools_used=observation.tools_used,
        final_critic_verdict=observation.final_critic_verdict,
        terminal_outcome=observation.terminal_outcome,
        initial_critic_errors=failures,
        evaluator_failures=failures,
        block_reason=observation.block_reason,
        wall_clock_sec=observation.wall_clock_sec,
        model_invocation_count=observation.model_invocation_count,
        revision_iterations=observation.revision_iterations,
    )
