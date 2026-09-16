# © Artur Czarnecki. All rights reserved.

"""GR-7-A2 — External Work → ExternalEffectOutcome projection."""

from __future__ import annotations

import pytest

from external_contractor_adapter.external_effect_outcome_projection import (
    ExternalWorkSideEffectObservation,
    external_work_provider_mutation_attempted,
    project_external_work_side_effect_to_effect_outcome,
)
from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.external_work import ExternalWorkErrorCode
from intergrax.contracts.governed_proof import compose_governed_proof_profile
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision

pytestmark = pytest.mark.unit

_ALLOW = PolicyDecision(
    action=PolicyAction.ALLOW,
    reason="allow",
    enforcement_level=EnforcementLevel.MANDATORY,
    policy_rule_id="test.allow",
    policy_bundle_id="b",
    policy_bundle_version="1",
    policy_bundle_digest="sha256:" + ("ab" * 32),
    decision_id="pol-1",
)
_DENY = _ALLOW.model_copy(
    update={
        "action": PolicyAction.DENY,
        "reason": "deny",
        "decision_id": "pol-deny",
    },
)
_PROOF = compose_governed_proof_profile(
    principal_id="p",
    task_id="task-1",
    run_id="run-1",
    action="external_work.create",
    provider_id="prov",
    policy_action=PolicyAction.ALLOW,
    idempotency_key="idem-1",
    correlation_id="corr-1",
)


def _obs(**updates: object) -> ExternalWorkSideEffectObservation:
    base = ExternalWorkSideEffectObservation(
        provider_mutation_attempted=True,
        adapter_result=ExternalWorkAdapterResult(
            used=True,
            reason="mapped",
            policy_decision=_ALLOW,
            proof=_PROOF,
        ),
    )
    return base.model_copy(update=updates)


def test_success_projection() -> None:
    outcome = project_external_work_side_effect_to_effect_outcome(_obs())
    assert outcome is ExternalEffectOutcome.SUCCESS


def test_definitive_failure_projection() -> None:
    outcome = project_external_work_side_effect_to_effect_outcome(
        _obs(
            adapter_result=ExternalWorkAdapterResult(
                used=False,
                reason="external_work_error",
                error_code=ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE,
                error_retryable=False,
            ),
        ),
    )
    assert outcome is ExternalEffectOutcome.FAILURE


def test_unknown_projection_uses_contract_error_code() -> None:
    outcome = project_external_work_side_effect_to_effect_outcome(
        _obs(
            adapter_result=ExternalWorkAdapterResult(
                used=False,
                reason="external_work_error",
                error_code=ExternalWorkErrorCode.PROVIDER_OUTCOME_UNCERTAIN,
                error_retryable=False,
            ),
        ),
    )
    assert outcome is ExternalEffectOutcome.UNKNOWN


def test_retryable_transient_is_failure_not_unknown() -> None:
    outcome = project_external_work_side_effect_to_effect_outcome(
        _obs(
            adapter_result=ExternalWorkAdapterResult(
                used=False,
                reason="external_work_error",
                error_code=ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE,
                error_retryable=True,
            ),
        ),
    )
    assert outcome is ExternalEffectOutcome.FAILURE


def test_no_attempt_returns_none() -> None:
    assert (
        project_external_work_side_effect_to_effect_outcome(
            ExternalWorkSideEffectObservation(
                provider_mutation_attempted=False,
                adapter_result=ExternalWorkAdapterResult(
                    used=False,
                    reason="policy_denied",
                    policy_decision=_DENY,
                ),
            ),
        )
        is None
    )


def test_provider_mutation_attempted_respects_policy_deny() -> None:
    result = ExternalWorkAdapterResult(
        used=False,
        reason="deny",
        policy_decision=_DENY,
        error_code=ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE,
    )
    assert not external_work_provider_mutation_attempted(result, policy_denied=True)
    assert not external_work_provider_mutation_attempted(result, policy_denied=False)
