# © Artur Czarnecki. All rights reserved.

"""Scenario execution exception causal classification tests (DS-E2E-15J-C1)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.decision_system.completion_eligibility import (
    CompletionEligibilityDecision,
    CompletionEligibilityStatus,
)
from intergrax.decision_system.evidence_requirements import EvidenceRequirementId
from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureOwner,
    DecisionFailureReason,
)
from intergrax.llm_adapters.contracts.strict_tool_call_validation import (
    StrictToolContractValidationError,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
    CompletionReconciliationDiagnostic,
    CompletionReconciliationError,
    CompletionReconciliationFailureReason,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
    PreReconciliationRecoveryStatus,
    PreReconciliationTransitionDecision,
    PreReconciliationTransitionOutcome,
    PreReconciliationValidationError,
)
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_completion_gate import (
    CompletionEligibilityBlockedError,
)
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_requirement_semantics import (
    AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID,
    AI_INCIDENT_TELEMETRY_EVIDENCE_REQUIREMENT_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    TERMINAL_STATE_NOT_ACCEPTED,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_UNRESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
)
from testing_support.decision_e2e.failure_observation_adapter import (
    ProviderInfrastructureFacts,
    observation_from_scenario_execution_exception,
    provider_infrastructure_facts_from_execution_error,
)


class CompletelyUnknownExecutionError(Exception):
    """Synthetic unknown exception for fail-closed classification proof."""


class ProviderProtocolTransportError(Exception):
    """Synthetic provider protocol transport failure."""


@dataclass(frozen=True, slots=True)
class _ClassificationExpectation:
    category: DecisionFailureCategory
    reason: DecisionFailureReason
    boundary: DecisionFailureBoundary
    owner: DecisionFailureOwner


def _classify_exception(exc: BaseException) -> _ClassificationExpectation:
    observation = observation_from_scenario_execution_exception(exc)
    result = classify_decision_failure(observation)
    assert result is not None
    return _ClassificationExpectation(
        category=result.category,
        reason=result.reason,
        boundary=result.boundary,
        owner=result.owner,
    )


def _legacy_provider_network_classification(
    exc: BaseException,
) -> _ClassificationExpectation:
    """Pre-C1 fallback that mislabeled unknown exceptions as provider network failure."""
    return _ClassificationExpectation(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.NETWORK_FAILURE,
        boundary=DecisionFailureBoundary.HOST_EXECUTION,
        owner=DecisionFailureOwner.PROVIDER,
    )


def test_pre_reconciliation_validation_error_unchanged() -> None:
    exc = PreReconciliationValidationError(
        PreReconciliationTransitionDecision(
            outcome=PreReconciliationTransitionOutcome.REJECTED,
            validation_errors=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
            recovery_status=PreReconciliationRecoveryStatus.BUDGET_EXHAUSTED,
            revision_budget_remaining=0,
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
            recovery_attempted=False,
        )
    )
    result = _classify_exception(exc)
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.MODEL_BEHAVIOR,
        reason=DecisionFailureReason.UNSUPPORTED_COMPLETION,
        boundary=DecisionFailureBoundary.PHASE_VALIDATION,
        owner=DecisionFailureOwner.MODEL,
    )


def test_completion_reconciliation_error_unchanged() -> None:
    exc = CompletionReconciliationError(
        CompletionReconciliationFailureReason.VALIDATION_ERRORS_PRESENT,
        diagnostic=CompletionReconciliationDiagnostic(
            model_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            validation_errors=("some_error",),
            evidence_gathering_stop_reason="planner_final_answer",
        ),
    )
    result = _classify_exception(exc)
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.MODEL_BEHAVIOR,
        reason=DecisionFailureReason.UNSUPPORTED_COMPLETION,
        boundary=DecisionFailureBoundary.COMPLETION_RECONCILIATION,
        owner=DecisionFailureOwner.MODEL,
    )


def test_terminal_acceptance_runtime_error_unchanged() -> None:
    exc = RuntimeError(TERMINAL_STATE_NOT_ACCEPTED)
    result = _classify_exception(exc)
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.TERMINAL_ACCEPTANCE_CONTRACT_VIOLATION,
        boundary=DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
        owner=DecisionFailureOwner.EXECUTION_ENGINE,
    )


def test_strict_tool_contract_validation_error_not_provider_network() -> None:
    exc = StrictToolContractValidationError(
        "tool_call.arguments: expected type integer"
    )
    result = _classify_exception(exc)
    assert result.category is not DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
    assert result.reason is not DecisionFailureReason.NETWORK_FAILURE
    assert result.owner is not DecisionFailureOwner.PROVIDER
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.STRICT_TOOL_CONTRACT_VIOLATION,
        boundary=DecisionFailureBoundary.STRICT_TOOL_PROJECTION,
        owner=DecisionFailureOwner.EXECUTION_ENGINE,
    )


@pytest.mark.parametrize(
    "requirement_id",
    (
        AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID,
        AI_INCIDENT_TELEMETRY_EVIDENCE_REQUIREMENT_ID,
    ),
)
def test_completion_eligibility_blocked_maps_from_typed_decision(
    requirement_id: EvidenceRequirementId,
) -> None:
    exc = CompletionEligibilityBlockedError(
        CompletionEligibilityDecision(
            status=CompletionEligibilityStatus.INELIGIBLE,
            unresolved_mandatory_requirement_ids=(requirement_id,),
            satisfied_mandatory_count=0,
            waived_mandatory_count=0,
            optional_unsatisfied_count=0,
        )
    )
    result = _classify_exception(exc)
    assert result.category is not DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
    assert result.reason is not DecisionFailureReason.NETWORK_FAILURE
    assert result.owner is not DecisionFailureOwner.PROVIDER
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.MODEL_BEHAVIOR,
        reason=DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING,
        boundary=DecisionFailureBoundary.EVIDENCE_LIFECYCLE,
        owner=DecisionFailureOwner.MODEL,
    )


def test_connection_refused_maps_to_provider_network_failure() -> None:
    exc = ConnectionRefusedError("connection refused")
    result = _classify_exception(exc)
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.NETWORK_FAILURE,
        boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        owner=DecisionFailureOwner.PROVIDER,
    )


def test_timeout_maps_to_provider_timeout() -> None:
    exc = TimeoutError("provider request timed out")
    result = _classify_exception(exc)
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.TIMEOUT,
        boundary=DecisionFailureBoundary.HOST_EXECUTION,
        owner=DecisionFailureOwner.PROVIDER,
    )


def test_http_429_maps_to_provider_rate_limit() -> None:
    facts = provider_infrastructure_facts_from_execution_error(
        exc=RuntimeError("rate limited"),
        http_status=429,
    )
    assert facts == ProviderInfrastructureFacts(rate_limit=True)
    from testing_support.decision_e2e.failure_observation_adapter import (
        observation_from_provider_infrastructure_facts,
    )

    result = classify_decision_failure(
        observation_from_provider_infrastructure_facts(
            ProviderInfrastructureFacts(rate_limit=True),
            boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        )
    )
    assert result is not None
    assert result.category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
    assert result.reason is DecisionFailureReason.RATE_LIMIT
    assert result.boundary is DecisionFailureBoundary.PROVIDER_BINDING
    assert result.owner is DecisionFailureOwner.PROVIDER


def test_http_5xx_maps_to_provider_server_error() -> None:
    facts = provider_infrastructure_facts_from_execution_error(
        exc=RuntimeError("upstream failure"),
        http_status=503,
    )
    assert facts == ProviderInfrastructureFacts(server_error=True)
    assert facts is not None
    from testing_support.decision_e2e.failure_observation_adapter import (
        observation_from_provider_infrastructure_facts,
    )

    result = classify_decision_failure(
        observation_from_provider_infrastructure_facts(facts)
    )
    assert result is not None
    assert result.category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
    assert result.reason is DecisionFailureReason.PROVIDER_SERVER_ERROR
    assert result.boundary is DecisionFailureBoundary.HOST_EXECUTION
    assert result.owner is DecisionFailureOwner.PROVIDER


def test_provider_protocol_transport_error_maps_to_protocol_failure() -> None:
    exc = ProviderProtocolTransportError("provider protocol transport failure")
    result = _classify_exception(exc)
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.PROVIDER_PROTOCOL_ERROR,
        boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        owner=DecisionFailureOwner.PROVIDER,
    )


def test_unknown_exception_fail_closed_not_provider_network() -> None:
    exc = CompletelyUnknownExecutionError("no causal mapping")
    result = _classify_exception(exc)
    assert result == _ClassificationExpectation(
        category=DecisionFailureCategory.UNCLASSIFIED,
        reason=DecisionFailureReason.UNCLASSIFIED,
        boundary=DecisionFailureBoundary.HOST_EXECUTION,
        owner=DecisionFailureOwner.DECISION_SYSTEM,
    )


def test_provider_helper_unknown_error_type_returns_none() -> None:
    facts = provider_infrastructure_facts_from_execution_error(
        exc=CompletelyUnknownExecutionError("unknown"),
    )
    assert facts is None


def test_typed_domain_exception_precedence_over_provider_mapper() -> None:
    exc = StrictToolContractValidationError("malformed tool arguments")
    facts = provider_infrastructure_facts_from_execution_error(exc=exc)
    assert facts is None
    result = _classify_exception(exc)
    assert result.category is DecisionFailureCategory.PLATFORM_CONTRACT


def test_l1r2_replay_strict_tool_and_completion_eligibility_reclassified() -> None:
    replay_cases = (
        StrictToolContractValidationError("tool_call.arguments: expected type integer"),
        CompletionEligibilityBlockedError(
            CompletionEligibilityDecision(
                status=CompletionEligibilityStatus.INELIGIBLE,
                unresolved_mandatory_requirement_ids=(
                    AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID,
                ),
                satisfied_mandatory_count=0,
                waived_mandatory_count=0,
                optional_unsatisfied_count=0,
            )
        ),
    )
    reclassified = 0
    for exc in replay_cases:
        old = _legacy_provider_network_classification(exc)
        new = _classify_exception(exc)
        assert old.category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
        assert new.category is not DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
        reclassified += 1
    assert reclassified == len(replay_cases)


def test_c1_diagnostic_artifacts_written(tmp_path: Path) -> None:
    session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_root = tmp_path / "qualification" / "DS-E2E-15J-C1" / session_id
    artifact_root.mkdir(parents=True)

    rows: list[dict[str, str]] = []
    matrix_cases: tuple[tuple[BaseException, str, str], ...] = (
        (
            StrictToolContractValidationError("expected type integer"),
            "intergrax.llm_adapters",
            "model emitted schema-invalid tool call",
        ),
        (
            CompletionEligibilityBlockedError(
                CompletionEligibilityDecision(
                    status=CompletionEligibilityStatus.INELIGIBLE,
                    unresolved_mandatory_requirement_ids=(
                        AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID,
                    ),
                    satisfied_mandatory_count=0,
                    waived_mandatory_count=0,
                    optional_unsatisfied_count=0,
                )
            ),
            "platform_proofs.evidence_completion_gate",
            "mandatory evidence unresolved at completion gate",
        ),
        (ConnectionRefusedError("refused"), "provider transport", "connection refused"),
        (TimeoutError("timed out"), "provider transport", "request timeout"),
        (ProviderProtocolTransportError("protocol"), "provider transport", "protocol failure"),
        (CompletelyUnknownExecutionError("unknown"), "unknown", "no typed evidence"),
    )

    for exc, producer, meaning in matrix_cases:
        old = _legacy_provider_network_classification(exc)
        new = _classify_exception(exc)
        rows.append(
            {
                "exception_type": type(exc).__name__,
                "producer": producer,
                "typed_payload": meaning,
                "old_category": old.category.value,
                "old_reason": old.reason.value,
                "old_boundary": old.boundary.value,
                "old_owner": old.owner.value,
                "new_category": new.category.value,
                "new_reason": new.reason.value,
                "new_boundary": new.boundary.value,
                "new_owner": new.owner.value,
                "causal_evidence": meaning,
            }
        )

    inventory = {
        "known_domain_exceptions": [
            "PreReconciliationValidationError",
            "CompletionReconciliationError",
            "CompletionEligibilityBlockedError",
        ],
        "known_model_tool_contract_exceptions": [
            "StrictToolContractValidationError",
        ],
        "known_provider_infrastructure_exceptions": [
            "ConnectionError",
            "TimeoutError",
            "ProviderProtocolTransportError",
        ],
        "unknown_fallback": ["CompletelyUnknownExecutionError"],
    }
    (artifact_root / "exception_inventory.json").write_text(
        json.dumps(inventory, indent=2),
        encoding="utf-8",
    )

    matrix_path = artifact_root / "classification_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    replay_rows = rows[:2]
    replay_path = artifact_root / "l1r2_exception_replay.csv"
    with replay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0].keys()))
        writer.writeheader()
        writer.writerows(replay_rows)

    report = (
        "# DS-E2E-15J-C1 Qualification Exception Classification\n\n"
        "REPLAY / DIAGNOSTIC ONLY\n\n"
        f"- matrix rows: {len(rows)}\n"
        f"- l1r2 replay rows: {len(replay_rows)}\n"
    )
    (artifact_root / "report.md").write_text(report, encoding="utf-8")
    manifest = "\n".join(
        sorted(path.name for path in artifact_root.iterdir() if path.is_file())
    )
    (artifact_root / "artifact-manifest.txt").write_text(manifest, encoding="utf-8")

    assert any(
        row["new_category"] != DecisionFailureCategory.PROVIDER_INFRASTRUCTURE.value
        for row in rows
        if row["exception_type"] == "CompletelyUnknownExecutionError"
    )
