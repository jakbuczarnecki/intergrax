# © Artur Czarnecki. All rights reserved.

"""Single reproducible governed-execution demonstration for partner validation.

Path (offline, deterministic fake provider — no network):

```text
CREATE_EXTERNAL_WORK (policy ALLOW → proof)
  → QUOTE governed continuation surface
  → QuoteAcceptanceEvidence
  → ACCEPT_QUOTE (policy ALLOW → provider → GovernedProofProfile)
```

Run::

    uv run pytest agents/external_contractor_adapter/tests/test_partner_validation_demo.py -q
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from external_contractor_adapter.external_work_adapter import (
    META_CORRELATION_ID,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    ExternalWorkAdapter,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CREATE_EXTERNAL_WORK,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from external_contractor_adapter.tests.fakes.deterministic_side_effect_policy import (
    DeterministicMeaningfulSideEffectPolicy,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.governed_continuation import ContinuationReason
from intergrax.contracts.governed_proof import EVIDENCE_KIND_QUOTE_ACCEPTANCE
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction

_DIGEST = "sha256:" + ("ab" * 32)
_T0 = datetime(2026, 7, 20, 18, 0, 0, tzinfo=timezone.utc)
_TASK_ID = "task-partner-demo"
_RUN_ID = "run-partner-demo"
_PROVIDER_ID = "gec3_deterministic_fake"
_CORR_ID = "corr-partner-demo"
_CREATE_IDEMP = "idem-partner-create"
_ACCEPT_IDEMP = "idem-partner-accept"


def _meta() -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER_ID,
        META_SCOPE_DESCRIPTION: "partner validation demo scope",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: _CREATE_IDEMP,
        META_CORRELATION_ID: _CORR_ID,
        "external_work.budget_limit": MoneyAmount(
            amount=Decimal("25.00"), currency="USD"
        ),
        "external_work.principal_id": "partner-demo-user",
        "external_work.tenant_id": "partner-demo-tenant",
    }


@pytest.mark.unit
@pytest.mark.gate
def test_partner_validation_governed_execution_demo() -> None:
    """Observable create → continuation → accept → GovernedProofProfile."""
    policy = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)
    adapter = ExternalWorkAdapter(
        DeterministicExternalWorkFake(),
        side_effect_policy=policy,
    )

    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            metadata=_meta(),
        ),
        principal_id="partner-demo-user",
        tenant_id="partner-demo-tenant",
    )
    assert created.used is True
    assert created.proof is not None
    assert created.proof.action == ACTION_CREATE_EXTERNAL_WORK
    assert created.policy_decision is not None
    assert created.policy_decision.action is PolicyAction.ALLOW

    surfaced = adapter.with_continuation_surface(created, run_id=_RUN_ID)
    assert surfaced.continuation is not None
    assert surfaced.continuation.reason is ContinuationReason.QUOTE
    assert surfaced.continuation.task_id == _TASK_ID
    assert surfaced.continuation.run_id == _RUN_ID
    assert surfaced.continuation.run_id != surfaced.continuation.task_id
    # Continuation surfaces a blocker — it must not itself mutate via accept.
    assert policy.calls[-1].action == ACTION_CREATE_EXTERNAL_WORK

    assert created.quote is not None
    acceptance = QuoteAcceptanceEvidence.model_validate(
        {
            "acceptance_id": "acc-partner-demo",
            "quote_id": created.quote.quote_id,
            "quote_version": 1,
            "scope_digest": _DIGEST,
            "actor": ActorIdentity(
                kind=ActorKind.USER,
                actor_id="partner-demo-user",
                tenant_id="partner-demo-tenant",
            ),
            "accepted_at": _T0 + timedelta(minutes=5),
            "hitl_decision_id": "hdec-partner-demo",
            "interrupt_id": "intr-partner-demo",
            "policy_decision_ref": "pol-partner-demo",
        }
    )
    accepted = adapter.forward_quote_acceptance(
        created.snapshot.correlation,  # type: ignore[union-attr]
        acceptance,
        idempotency_key=_ACCEPT_IDEMP,
        principal_id="partner-demo-user",
        tenant_id="partner-demo-tenant",
    )
    assert accepted.used is True
    assert accepted.proof is not None
    proof = accepted.proof
    assert proof.task_id == _TASK_ID
    assert proof.run_id == _RUN_ID
    assert proof.provider_id == _PROVIDER_ID
    assert proof.action == ACTION_ACCEPT_QUOTE
    assert accepted.policy_decision is not None
    assert accepted.policy_decision.action is PolicyAction.ALLOW
    assert proof.policy_action is PolicyAction.ALLOW
    assert proof.policy_rule_id == accepted.policy_decision.policy_rule_id
    assert proof.governance_evidence is not None
    assert proof.governance_evidence.kind == EVIDENCE_KIND_QUOTE_ACCEPTANCE
    assert proof.governance_evidence.evidence_id == "acc-partner-demo"
    assert proof.correlation_id == _CORR_ID
    assert proof.idempotency_key == _ACCEPT_IDEMP
    assert proof.continuation_reason is ContinuationReason.QUOTE

    # Ordering: ACCEPT_QUOTE policy before any accept-side proof composition.
    accept_reqs = [c for c in policy.calls if c.action == ACTION_ACCEPT_QUOTE]
    assert len(accept_reqs) == 1
    assert accept_reqs[0].run_id == _RUN_ID
    assert accept_reqs[0].task_id == _TASK_ID
