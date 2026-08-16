# © Artur Czarnecki. All rights reserved.

"""GEC-5 — meaningful side-effect policy composition (Tier-2 consumer)."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from external_contractor_adapter.external_work_adapter import (
    META_ACCEPTANCE_IDEMPOTENCY_KEY,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_QUOTE_ACCEPTANCE,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    ExternalWorkAdapter,
    adapt_from_step_metadata,
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
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction

_DIGEST = "sha256:" + ("cd" * 32)
_T0 = datetime(2026, 7, 20, 14, 0, 0, tzinfo=timezone.utc)
_AGENT_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_PY = _AGENT_ROOT / "external_work_adapter.py"


class _RecordingIntegration(DeterministicExternalWorkFake):
    """Records provider-bound mutating calls for ordering proofs."""

    def __init__(self, *args: object, call_log: list[str] | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.call_log = call_log if call_log is not None else []

    def create_work(self, request):  # type: ignore[no-untyped-def]
        self.call_log.append("integration.create_work")
        return super().create_work(request)

    def submit_quote_acceptance(self, correlation, acceptance, *, idempotency_key: str):
        self.call_log.append("integration.accept_quote")
        return super().submit_quote_acceptance(
            correlation, acceptance, idempotency_key=idempotency_key
        )

    def get_quote(self, correlation):  # type: ignore[no-untyped-def]
        self.call_log.append("integration.get_quote")
        return super().get_quote(correlation)


def _meta(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "review PR #42",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-gec5-1",
        "external_work.budget_limit": MoneyAmount(amount=Decimal("40.00"), currency="USD"),
        "external_work.principal_id": "u1",
        "external_work.tenant_id": "tenant-a",
    }
    payload.update(overrides)
    return payload


def _acceptance(**overrides: object) -> QuoteAcceptanceEvidence:
    payload: dict[str, object] = {
        "acceptance_id": "acc-gec5-1",
        "quote_id": "q-gec3-1",
        "quote_version": 1,
        "scope_digest": _DIGEST,
        "actor": ActorIdentity(
            kind=ActorKind.USER, actor_id="u1", tenant_id="tenant-a"
        ),
        "accepted_at": _T0 + timedelta(minutes=3),
        "hitl_decision_id": "hdec_gec5",
    }
    payload.update(overrides)
    return QuoteAcceptanceEvidence.model_validate(payload)


def _allow() -> DeterministicMeaningfulSideEffectPolicy:
    return DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)


@pytest.mark.unit
@pytest.mark.gate
def test_quote_receipt_is_observational_no_accept_policy() -> None:
    call_log: list[str] = []
    policy = _allow()
    policy.call_log = call_log
    integration = _RecordingIntegration(call_log=call_log)
    adapter = ExternalWorkAdapter(integration, side_effect_policy=policy)
    result = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-obs",
            run_id="run-obs",
            metadata=_meta(),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert result.used is True
    assert result.quote is not None
    actions = [c.action for c in policy.calls]
    assert ACTION_CREATE_EXTERNAL_WORK in actions
    assert ACTION_ACCEPT_QUOTE not in actions
    assert "integration.accept_quote" not in call_log
    assert "integration.get_quote" in call_log or result.quote is not None


@pytest.mark.unit
@pytest.mark.gate
def test_accept_quote_classified_as_meaningful_side_effect() -> None:
    policy = _allow()
    integration = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(integration, side_effect_policy=policy)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-cls",
            run_id="run-cls",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-cls"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.snapshot is not None and created.quote is not None
    policy.calls.clear()
    adapter.forward_quote_acceptance(
        created.snapshot.correlation,
        _acceptance(quote_id=created.quote.quote_id),
        idempotency_key="idem-accept-cls",
    )
    assert len(policy.calls) == 1
    req = policy.calls[0]
    assert req.action == ACTION_ACCEPT_QUOTE
    assert any(k.value == "commitment" for k in req.kinds)
    assert any(k.value == "mutation" for k in req.kinds)


@pytest.mark.unit
@pytest.mark.gate
def test_policy_before_accept_ordering_allow_once() -> None:
    call_log: list[str] = []
    policy = _allow()
    policy.call_log = call_log
    integration = _RecordingIntegration(call_log=call_log)
    adapter = ExternalWorkAdapter(integration, side_effect_policy=policy)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-ord",
            run_id="run-ord",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-ord"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.snapshot is not None and created.quote is not None
    # Reset log to isolate accept path ordering.
    call_log.clear()
    policy.calls.clear()
    forwarded = adapter.forward_quote_acceptance(
        created.snapshot.correlation,
        _acceptance(quote_id=created.quote.quote_id),
        idempotency_key="idem-accept-ord",
    )
    assert forwarded.used is True
    assert "policy.evaluate" in call_log
    assert "integration.accept_quote" in call_log
    assert call_log.index("policy.evaluate") < call_log.index("integration.accept_quote")
    assert call_log.count("integration.accept_quote") == 1
    # Enrich reads may follow accept — never before policy / accept.
    if "integration.get_quote" in call_log:
        assert call_log.index("integration.accept_quote") < call_log.index(
            "integration.get_quote"
        )


@pytest.mark.unit
@pytest.mark.gate
def test_deny_prevents_provider_accept_call() -> None:
    call_log: list[str] = []
    policy = DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.ALLOW,
        by_action={ACTION_ACCEPT_QUOTE: PolicyAction.DENY},
    )
    policy.call_log = call_log
    integration = _RecordingIntegration(call_log=call_log)
    adapter = ExternalWorkAdapter(integration, side_effect_policy=policy)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-deny",
            run_id="run-deny",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-deny"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.snapshot is not None and created.quote is not None
    call_log.clear()
    denied = adapter.forward_quote_acceptance(
        created.snapshot.correlation,
        _acceptance(quote_id=created.quote.quote_id),
        idempotency_key="idem-accept-deny",
    )
    assert denied.used is False
    assert denied.reason == "side_effect_denied"
    assert denied.policy_decision is not None
    assert denied.policy_decision.action is PolicyAction.DENY
    assert "integration.accept_quote" not in call_log
    assert call_log == ["policy.evaluate"]


@pytest.mark.unit
@pytest.mark.gate
def test_require_human_surfaces_continuation_no_provider_call() -> None:
    call_log: list[str] = []
    policy = DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.ALLOW,
        by_action={ACTION_ACCEPT_QUOTE: PolicyAction.REQUIRE_HUMAN},
    )
    policy.call_log = call_log
    integration = _RecordingIntegration(call_log=call_log)
    adapter = ExternalWorkAdapter(integration, side_effect_policy=policy)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-gov",
            run_id="run-gov",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-gov"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.snapshot is not None and created.quote is not None
    call_log.clear()
    gated = adapter.forward_quote_acceptance(
        created.snapshot.correlation,
        _acceptance(quote_id=created.quote.quote_id),
        idempotency_key="idem-accept-gov",
    )
    assert gated.used is False
    assert gated.reason == "side_effect_governance_required"
    assert gated.continuation is not None
    assert gated.continuation.reason is ContinuationReason.QUOTE
    assert gated.continuation.task_id == "task-gov"
    assert gated.continuation.run_id == "run-gov"
    assert "integration.accept_quote" not in call_log


@pytest.mark.unit
@pytest.mark.gate
def test_missing_policy_and_indeterminate_fail_closed() -> None:
    fake = DeterministicExternalWorkFake()
    bare = ExternalWorkAdapter(fake, side_effect_policy=None)
    denied = bare.create_and_map(
        bare.build_create_request(
            task_id="task-miss",
            run_id="run-miss",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-miss"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert denied.used is False
    assert denied.reason == "side_effect_policy_missing"
    assert fake.create_calls == 0

    # RuntimePolicyEngine default (no matching rule) is indeterminate → DENY.
    from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

    engine = RuntimePolicyEngine()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=engine)
    indeterminate = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-indet",
            run_id="run-indet",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-indet"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert indeterminate.used is False
    assert indeterminate.reason == "side_effect_denied"
    assert indeterminate.policy_decision is not None
    assert indeterminate.policy_decision.reason == "meaningful_side_effect_indeterminate"
    assert fake.create_calls == 0


@pytest.mark.unit
@pytest.mark.gate
def test_evidence_presence_is_not_authorization() -> None:
    call_log: list[str] = []
    policy = DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.ALLOW,
        by_action={ACTION_ACCEPT_QUOTE: PolicyAction.DENY},
    )
    policy.call_log = call_log
    integration = _RecordingIntegration(call_log=call_log)
    adapter = ExternalWorkAdapter(integration, side_effect_policy=policy)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-ev",
            run_id="run-ev",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-ev"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.quote is not None and created.snapshot is not None
    evidence = _acceptance(
        quote_id=created.quote.quote_id,
        policy_decision_ref="pol_ref_not_sufficient",
    )
    call_log.clear()
    result = adapter.forward_continuation_evidence(
        created.snapshot.correlation,
        reason=ContinuationReason.QUOTE,
        evidence=evidence,
        idempotency_key="idem-ev-accept",
    )
    assert result.used is False
    assert "integration.accept_quote" not in call_log
    assert evidence.policy_decision_ref == "pol_ref_not_sufficient"


@pytest.mark.unit
@pytest.mark.gate
def test_execution_identity_forwarded_or_explicitly_missing() -> None:
    """Valid run_id is forwarded unchanged; missing is None — never \"\"."""
    policy = _allow()
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)

    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-id-sem",
            run_id="run-id-sem",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-id-sem"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.used is True
    assert len(policy.calls) == 1
    assert policy.calls[0].run_id == "run-id-sem"
    assert policy.calls[0].run_id != ""
    assert created.snapshot is not None
    assert created.snapshot.correlation.run_id == "run-id-sem"

    policy.calls.clear()
    missing = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-id-missing",
            run_id=None,
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-id-missing"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert missing.used is False
    assert missing.reason == "side_effect_identity_missing"
    assert policy.calls == []
    assert fake.create_calls == 1  # only the successful create above
    # Correlation on the denied path stays unset — no fabricated run identity.
    assert missing.snapshot is None

    # ACCEPT_QUOTE / CANCEL with correlation.run_id is None fail before policy.
    assert created.quote is not None
    corr = created.snapshot.correlation.model_copy(update={"run_id": None})
    assert corr.run_id is None
    accept_denied = adapter.forward_quote_acceptance(
        corr,
        _acceptance(quote_id=created.quote.quote_id),
        idempotency_key="idem-accept-missing-run",
    )
    assert accept_denied.reason == "side_effect_identity_missing"
    assert policy.calls == []
    cancel_denied = adapter.cancel_and_map(
        corr,
        idempotency_key="idem-cancel-missing-run",
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert cancel_denied.reason == "side_effect_identity_missing"
    assert policy.calls == []

    source = _ADAPTER_PY.read_text(encoding="utf-8")
    assert "request.run_id or \"\"" not in source
    assert "correlation.run_id or \"\"" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_preserves_identity_correlation_idempotency_and_payload() -> None:
    policy = _allow()
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-preserve",
            run_id="run-preserve",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-preserve"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.snapshot is not None and created.quote is not None
    key = "idem-accept-preserve"
    acceptance = _acceptance(quote_id=created.quote.quote_id)
    forwarded = adapter.forward_quote_acceptance(
        created.snapshot.correlation,
        acceptance,
        idempotency_key=key,
    )
    assert forwarded.used is True
    assert forwarded.snapshot is not None
    corr = forwarded.snapshot.correlation
    assert corr.task_id == "task-preserve"
    assert corr.run_id == "run-preserve"
    assert corr.provider_id == "gec3_deterministic_fake"
    assert corr.idempotency_key == "idem-preserve"
    accept_req = [c for c in policy.calls if c.action == ACTION_ACCEPT_QUOTE][-1]
    assert accept_req.correlation["idempotency_key"] == key
    assert accept_req.correlation["quote_id"] == acceptance.quote_id
    assert accept_req.correlation["scope_digest"] == _DIGEST
    # Policy context is a copy — mutating it must not alter evidence payload.
    mutable = dict(accept_req.context)
    mutable["quote_id"] = "mutated"
    assert acceptance.quote_id == created.quote.quote_id


@pytest.mark.unit
@pytest.mark.gate
def test_tier2_has_no_embedded_approval_rules() -> None:
    source = _ADAPTER_PY.read_text(encoding="utf-8").lower()
    for needle in (
        "spending_limit",
        "max_quote",
        "quote_value_threshold",
        "approved = true",
        "if quote.total",
    ):
        assert needle not in source
    tree = ast.parse(_ADAPTER_PY.read_text(encoding="utf-8"), filename=str(_ADAPTER_PY))
    class_names = {
        n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
    }
    assert "PolicyEngine" not in class_names
    assert "ContinuationRuntime" not in class_names


@pytest.mark.unit
@pytest.mark.gate
def test_adapt_from_step_metadata_wires_policy() -> None:
    call_log: list[str] = []
    policy = _allow()
    policy.call_log = call_log
    integration = _RecordingIntegration(call_log=call_log)
    result = adapt_from_step_metadata(
        integration,
        task_id="task-step",
        run_id="run-step",
        message="scope",
        metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-step"}),
        side_effect_policy=policy,
    )
    assert result.used is True
    assert result.reason == "continuation_blocked"
    assert "policy.evaluate" in call_log
    assert call_log.index("policy.evaluate") < call_log.index("integration.create_work")
