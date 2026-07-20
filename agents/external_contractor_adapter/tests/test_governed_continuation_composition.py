# © Artur Czarnecki. All rights reserved.

"""GEC-4 — Tier-2 composition of Governed Continuation (mapping only)."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
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
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from external_contractor_adapter.tests.fakes.deterministic_side_effect_policy import (
    DeterministicMeaningfulSideEffectPolicy,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import ExternalWorkStatus, QuoteAcceptanceEvidence
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    attach_continuation_refs_to_quote_acceptance,
    compose_continuation_interrupt,
    continuation_evidence_refs_from_quote_acceptance,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler

_DIGEST = "sha256:" + ("cd" * 32)
_T0 = datetime(2026, 7, 20, 14, 0, 0, tzinfo=timezone.utc)
_AGENT_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_PY = _AGENT_ROOT / "external_work_adapter.py"
_DOMAIN_PY = _AGENT_ROOT / "steps" / "domain_job.py"
_FORBIDDEN_RUNTIME_NAMES = {
    "ContinuationRuntime",
    "ContinuationEngine",
    "ContinuationManager",
    "QuoteRuntime",
    "QuoteLifecycleEngine",
    "GovernanceRuntime",
}


def _meta(**overrides: object) -> dict[str, object]:
    from decimal import Decimal

    payload: dict[str, object] = {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "review PR #42",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-gec4-1",
        "external_work.budget_limit": MoneyAmount(
            amount=Decimal("40.00"), currency="USD"
        ),
    }
    payload.update(overrides)
    return payload



def _allow_policy() -> DeterministicMeaningfulSideEffectPolicy:
    return DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)


def _adapter(fake: DeterministicExternalWorkFake | None = None) -> ExternalWorkAdapter:
    return ExternalWorkAdapter(
        fake or DeterministicExternalWorkFake(),
        side_effect_policy=_allow_policy(),
    )


def _acceptance(**overrides: object) -> QuoteAcceptanceEvidence:
    payload: dict[str, object] = {
        "acceptance_id": "acc-gec4-1",
        "quote_id": "q-gec3-1",
        "quote_version": 1,
        "scope_digest": _DIGEST,
        "actor": ActorIdentity(
            kind=ActorKind.USER, actor_id="u1", tenant_id="tenant-a"
        ),
        "accepted_at": _T0 + timedelta(minutes=3),
        "hitl_decision_id": "hdec_gec4",
        "interrupt_id": "int_gec4",
    }
    payload.update(overrides)
    return QuoteAcceptanceEvidence.model_validate(payload)


@pytest.mark.unit
@pytest.mark.gate
def test_continuation_blocker_surfaced_for_quote() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = _adapter(fake)
    mapped = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-gec4",
            run_id="run-gec4",
            metadata=_meta(),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    blocker = adapter.surface_continuation_blocker(mapped, run_id="run-gec4")
    assert blocker is not None
    assert blocker.reason is ContinuationReason.QUOTE
    assert blocker.task_id == "task-gec4"
    assert blocker.run_id == "run-gec4"
    assert blocker.run_id != blocker.task_id
    assert blocker.correlation["external_task_id"].startswith("ext-gec3-")
    assert blocker.context["quote_id"] == "q-gec3-1"
    surfaced = adapter.with_continuation_surface(mapped, run_id="run-gec4")
    assert surfaced.reason == "continuation_blocked"
    assert surfaced.continuation is not None
    assert surfaced.continuation.reason is ContinuationReason.QUOTE
    assert surfaced.continuation.run_id == "run-gec4"


@pytest.mark.unit
@pytest.mark.gate
def test_continuation_preserves_distinct_task_and_run_identity() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = _adapter(fake)
    mapped = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-123",
            run_id="run-456",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-identity"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    continuation = adapter.surface_continuation_blocker(mapped, run_id="run-456")
    assert continuation is not None
    assert continuation.task_id == "task-123"
    assert continuation.run_id == "run-456"
    assert continuation.run_id != continuation.task_id
    # Correlation fields remain unchanged (optional run_id forwarded as stored).
    assert continuation.correlation["task_id"] == "task-123"
    assert continuation.correlation["run_id"] == "run-456"


@pytest.mark.unit
@pytest.mark.gate
def test_missing_run_id_fails_closed_without_task_fallback() -> None:
    fake = DeterministicExternalWorkFake()
    policy = _allow_policy()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    # GEC-5: meaningful create requires real Nexus run_id (fail closed).
    denied_create = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-no-run",
            run_id=None,
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-no-run"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert denied_create.used is False
    assert denied_create.reason == "side_effect_identity_missing"
    # Create with a real run_id, then prove continuation surface never fabricates one.
    mapped = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-no-run",
            run_id="run-for-snapshot",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-no-run"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert mapped.snapshot is not None
    assert mapped.snapshot.correlation.task_id == "task-no-run"
    blocker = adapter.surface_continuation_blocker(mapped, run_id="")
    assert blocker is None
    result = adapter.with_continuation_surface(mapped, run_id=None)
    assert result.continuation is None
    assert result.used is False
    assert result.error_code is not None
    assert result.error_code.value == "invalid_request"
    assert result.error_message is not None
    assert "run identity" in result.error_message.lower()
    assert result.reason == "continuation_correlation_failed"
    summary = result.to_domain_summary()
    assert summary.get("continuation") is None
    via_step = adapt_from_step_metadata(
        fake,
        task_id="task-no-run-step",
        run_id=None,
        message="scope",
        metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-no-run-step"}),
        side_effect_policy=policy,
    )
    assert via_step.used is False
    assert via_step.continuation is None
    assert via_step.reason == "side_effect_identity_missing"
    source = _ADAPTER_PY.read_text(encoding="utf-8")
    assert "correlation.run_id or correlation.task_id" not in source
    assert "or correlation.task_id" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_interrupt_composition_reuses_nexus_handler() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = _adapter(fake)
    mapped = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-int",
            run_id="run-int",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-int"}),
        ),
        principal_id="u1",
        tenant_id="tenant-a",
    )
    blocker = adapter.surface_continuation_blocker(mapped, run_id="run-int")
    assert blocker is not None
    assert blocker.run_id == "run-int"
    interrupt = compose_continuation_interrupt(blocker)
    resolution = ExecutionInterruptHandler().resolve_interrupt(interrupt)
    assert resolution.should_pause is True
    assert resolution.interrupt is not None
    assert resolution.interrupt.interrupt_id == interrupt.interrupt_id


@pytest.mark.unit
@pytest.mark.gate
def test_correlation_preserved_across_continuation_surface() -> None:
    fake = DeterministicExternalWorkFake()
    result = adapt_from_step_metadata(
        fake,
        task_id="task-corr",
        run_id="run-corr",
        message="scope",
        metadata=_meta(
            **{
                META_IDEMPOTENCY_KEY: "idem-corr",
                "external_work.principal_id": "u1",
                "external_work.tenant_id": "tenant-a",
            }
        ),
        side_effect_policy=_allow_policy(),
    )
    assert result.used is True
    assert result.reason == "continuation_blocked"
    assert result.snapshot is not None
    assert result.continuation is not None
    corr = result.snapshot.correlation
    assert result.continuation.correlation["external_task_id"] == corr.external_task_id
    assert result.continuation.correlation["idempotency_key"] == "idem-corr"
    assert result.continuation.task_id == "task-corr"
    assert result.continuation.run_id == "run-corr"
    assert result.continuation.run_id != result.continuation.task_id


@pytest.mark.unit
@pytest.mark.gate
def test_continuation_evidence_propagation_without_tier2_governance() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = _adapter(fake)
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
    assert created.snapshot is not None and created.quote is not None
    evidence = attach_continuation_refs_to_quote_acceptance(
        _acceptance(quote_id=created.quote.quote_id),
        hitl_decision_id="hdec_prop",
        interrupt_id="int_prop",
        policy_decision_ref="pol_prop",
    )
    refs = continuation_evidence_refs_from_quote_acceptance(evidence)
    forwarded = adapter.forward_continuation_evidence(
        created.snapshot.correlation,
        reason=ContinuationReason.QUOTE,
        evidence=evidence,
        idempotency_key="idem-accept-gec4",
    )
    assert forwarded.used is True
    assert forwarded.status == ExternalWorkStatus.ACCEPTED
    assert refs.hitl_decision_id == "hdec_prop"
    assert refs.interrupt_id == "int_prop"
    # Tier-2 must not interpret governance — evidence refs pass through unchanged.
    assert evidence.policy_decision_ref == "pol_prop"


@pytest.mark.unit
@pytest.mark.gate
def test_tier2_never_evaluates_governance_or_resumes() -> None:
    source = _ADAPTER_PY.read_text(encoding="utf-8")
    for needle in (
        "ExecutionInterruptHandler",
        "resolve_decision",
        "resolve_interrupt",
        "HumanPauseCoordinator",
        "evaluate_interrupt",
        "HumanResponseVerdict",
    ):
        assert needle not in source
    # May compose MeaningfulSideEffectEvaluator; must not embed approval rules.
    assert "spending_limit" not in source.lower()
    assert "quote_value_threshold" not in source.lower()
    assert "does not own governance" in source.lower() or "never decide" in source.lower()
    assert "forward_continuation_evidence" in source


@pytest.mark.unit
@pytest.mark.gate
def test_no_duplicate_runtime_or_quote_lifecycle_engine() -> None:
    for path in (_ADAPTER_PY, _DOMAIN_PY):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        assert not (class_names & _FORBIDDEN_RUNTIME_NAMES)
        for forbidden in _FORBIDDEN_RUNTIME_NAMES:
            assert forbidden not in source


@pytest.mark.unit
@pytest.mark.gate
def test_no_transport_coupling_in_continuation_path() -> None:
    tree = ast.parse(_ADAPTER_PY.read_text(encoding="utf-8"), filename=str(_ADAPTER_PY))
    forbidden_roots = {"httpx", "requests", "aiohttp", "urllib3", "http"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_roots
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_roots
    source_lower = _ADAPTER_PY.read_text(encoding="utf-8").lower()
    for needle in ("json-rpc", "jsonrpc", "partner sdk"):
        assert needle not in source_lower


@pytest.mark.unit
@pytest.mark.gate
def test_adapt_metadata_resume_forwards_evidence() -> None:
    fake = DeterministicExternalWorkFake()
    policy = _allow_policy()
    boot = ExternalWorkAdapter(fake, side_effect_policy=policy).create_and_map(
        ExternalWorkAdapter(fake, side_effect_policy=policy).build_create_request(
            task_id="task-resume",
            run_id="run-resume",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-resume"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert boot.quote is not None
    result = adapt_from_step_metadata(
        fake,
        task_id="task-resume",
        run_id="run-resume",
        message="scope",
        metadata=_meta(
            **{
                META_IDEMPOTENCY_KEY: "idem-resume",
                META_QUOTE_ACCEPTANCE: _acceptance(quote_id=boot.quote.quote_id),
                META_ACCEPTANCE_IDEMPOTENCY_KEY: "idem-accept-resume",
            }
        ),
        side_effect_policy=policy,
    )
    assert result.used is True
    assert result.status == ExternalWorkStatus.ACCEPTED
    assert result.continuation is None
    assert result.reason == "mapped"


@pytest.mark.unit
@pytest.mark.gate
def test_non_quote_continuation_reason_not_owned_by_adapter() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = _adapter(fake)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-sec",
            run_id="run-sec",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-sec"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert created.snapshot is not None and created.quote is not None
    result = adapter.forward_continuation_evidence(
        created.snapshot.correlation,
        reason=ContinuationReason.SECURITY,
        evidence=_acceptance(quote_id=created.quote.quote_id),
        idempotency_key="idem-sec-accept",
    )
    assert result.used is False
    assert result.reason == "continuation_reason_unsupported"
