# © Artur Czarnecki. All rights reserved.

"""PC-4: atomic GovernedExecutionResult consistency."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.evaluated_policy_decision import EvaluatedPolicyDecision
from intergrax.contracts.governed_execution_result import GovernedExecutionResult
from intergrax.contracts.governed_proof import GovernedProofProfile
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

_T0 = datetime(2026, 7, 21, 10, 0, 0, tzinfo=timezone.utc)
_DIGEST = "sha256:" + ("ab" * 32)


def _decision(*, action: str = "CREATE_EXTERNAL_WORK") -> EvaluatedPolicyDecision:
    d = PolicyDecision(
        action=PolicyAction.ALLOW,
        policy_rule_id="r.create",
        policy_bundle_id="b1",
        policy_bundle_version="1",
        policy_bundle_digest=_DIGEST,
        decision_id="d1",
    )
    return EvaluatedPolicyDecision(
        decision=d,
        bundle_id="b1",
        bundle_version="1",
        bundle_digest=_DIGEST,
        matched_rule_id="r.create",
        evaluated_at=_T0,
        request_digest=_DIGEST,
    )


def _proof(**overrides) -> GovernedProofProfile:
    base = dict(
        principal_id="u1",
        tenant_id="ten1",
        task_id="t1",
        run_id="r1",
        action="CREATE_EXTERNAL_WORK",
        resource="scope",
        provider_id="prov",
        policy_action=PolicyAction.ALLOW,
        policy_rule_id="r.create",
        policy_reason="ok",
        correlation_id="c1",
        idempotency_key="i1",
    )
    base.update(overrides)
    return GovernedProofProfile.model_validate(base)


def _invocation(**overrides) -> ProviderInvocation:
    base = dict(
        invocation_id="inv-1",
        provider_id="prov",
        operation="create_work",
        task_id="t1",
        run_id="r1",
        correlation_id="c1",
        idempotency_key="i1",
        request_digest=_DIGEST,
        started_at=_T0,
    )
    base.update(overrides)
    return ProviderInvocation.model_validate(base)


def _outcome(**overrides) -> ProviderInvocationOutcome:
    base = dict(
        invocation_id="inv-1",
        status=ProviderInvocationStatus.SUCCEEDED,
        completed_at=_T0,
    )
    base.update(overrides)
    return ProviderInvocationOutcome.model_validate(base)


def _ger(**overrides) -> GovernedExecutionResult:
    base = dict(
        execution_id="exec-1",
        task_id="t1",
        run_id="r1",
        principal_id="u1",
        tenant_id="ten1",
        correlation_id="c1",
        idempotency_key="i1",
        action="CREATE_EXTERNAL_WORK",
        evaluated_policy_decision=_decision(),
        provider_invocation=_invocation(),
        provider_outcome=_outcome(),
        proof=_proof(),
        execution_started_at=_T0,
        execution_completed_at=_T0,
    )
    base.update(overrides)
    return GovernedExecutionResult.model_validate(base)


def test_consistent_result_ok() -> None:
    assert _ger().execution_id == "exec-1"


def test_reject_mismatched_task() -> None:
    with pytest.raises(ValueError, match="task_id_inconsistent"):
        _ger(proof=_proof(task_id="other"))


def test_reject_mismatched_invocation() -> None:
    with pytest.raises(ValueError, match="invocation_id_outcome_mismatch"):
        _ger(provider_outcome=_outcome(invocation_id="inv-other"))


def test_reject_cross_execution_proof_decision() -> None:
    with pytest.raises(ValueError, match="proof_policy_rule_mismatch"):
        _ger(proof=_proof(policy_rule_id="other-rule"))
