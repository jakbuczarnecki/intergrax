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
    adapter = ExternalWorkAdapter(fake)
    mapped = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-gec4",
            run_id="run-gec4",
            metadata=_meta(),
        )
    )
    blocker = adapter.surface_continuation_blocker(mapped)
    assert blocker is not None
    assert blocker.reason is ContinuationReason.QUOTE
    assert blocker.correlation["external_task_id"].startswith("ext-gec3-")
    assert blocker.context["quote_id"] == "q-gec3-1"
    surfaced = adapter.with_continuation_surface(mapped)
    assert surfaced.reason == "continuation_blocked"
    assert surfaced.continuation is not None
    assert surfaced.continuation.reason is ContinuationReason.QUOTE


@pytest.mark.unit
@pytest.mark.gate
def test_interrupt_composition_reuses_nexus_handler() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake)
    mapped = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-int",
            run_id="run-int",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-int"}),
        )
    )
    blocker = adapter.surface_continuation_blocker(mapped)
    assert blocker is not None
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
        metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-corr"}),
    )
    assert result.used is True
    assert result.reason == "continuation_blocked"
    assert result.snapshot is not None
    assert result.continuation is not None
    corr = result.snapshot.correlation
    assert result.continuation.correlation["external_task_id"] == corr.external_task_id
    assert result.continuation.correlation["idempotency_key"] == "idem-corr"
    assert result.continuation.task_id == "task-corr"


@pytest.mark.unit
@pytest.mark.gate
def test_continuation_evidence_propagation_without_tier2_governance() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-ev",
            run_id="run-ev",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-ev"}),
        ),
        enrich=False,
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
        "PolicyEngine",
        "evaluate_interrupt",
        "HumanResponseVerdict",
    ):
        assert needle not in source
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
    boot = ExternalWorkAdapter(fake).create_and_map(
        ExternalWorkAdapter(fake).build_create_request(
            task_id="task-resume",
            run_id="run-resume",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-resume"}),
        ),
        enrich=False,
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
    )
    assert result.used is True
    assert result.status == ExternalWorkStatus.ACCEPTED
    assert result.continuation is None
    assert result.reason == "mapped"


@pytest.mark.unit
@pytest.mark.gate
def test_non_quote_continuation_reason_not_owned_by_adapter() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake)
    created = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-sec",
            run_id="run-sec",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-sec"}),
        ),
        enrich=False,
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
