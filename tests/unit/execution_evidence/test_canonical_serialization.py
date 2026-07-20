# © Artur Czarnecki. All rights reserved.

"""Canonical serialization determinism for governed execution boundary events."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from intergrax.contracts.execution_evidence.boundary_event import (
    ExecutionBoundaryEvent,
    GovernedProofSection,
    PolicyDecisionSection,
    ProviderInvocationSection,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.attestation.canonical_json import (
    canonical_json_bytes,
    stable_payload_hash,
)

pytestmark = [pytest.mark.unit]

_T0 = datetime(2026, 7, 20, 18, 0, 0, tzinfo=timezone.utc)


def _event(**overrides: object) -> ExecutionBoundaryEvent:
    base: dict[str, object] = {
        "event_id": "ebe-fixed",
        "occurred_at": _T0,
        "task_id": "task-1",
        "run_id": "run-1",
        "correlation_id": "corr-1",
        "idempotency_key": "idem-1",
        "principal_id": "user-1",
        "tenant_id": "tenant-1",
        "actor": "host",
        "provider_id": "provider-1",
        "action": "CREATE_EXTERNAL_WORK",
        "policy": PolicyDecisionSection(
            bundle_id="bundle-1",
            bundle_version="1.0.0",
            bundle_digest="sha256:" + ("ab" * 32),
            rule_id="rule.create",
            action=PolicyAction.ALLOW,
            decision_id="pol:create",
        ),
        "provider_invocation": ProviderInvocationSection(
            operation="create_work",
            invocation_id="ext-1",
            outcome="success",
            completed_at=_T0,
        ),
        "governed_proof": GovernedProofSection(
            proof_id="proof-1",
            proof_digest="sha256:" + ("cd" * 32),
            proof={"task_id": "task-1", "run_id": "run-1"},
        ),
    }
    base.update(overrides)
    return ExecutionBoundaryEvent.model_validate(base)


def test_semantically_identical_events_same_canonical_bytes() -> None:
    a = _event()
    b = _event()
    assert canonical_json_bytes(a.canonical_payload()) == canonical_json_bytes(
        b.canonical_payload()
    )
    assert stable_payload_hash(a.canonical_payload()) == stable_payload_hash(
        b.canonical_payload()
    )


def test_meaningful_field_change_changes_digest() -> None:
    base = stable_payload_hash(_event().canonical_payload())
    changed = stable_payload_hash(_event(task_id="task-2").canonical_payload())
    assert base != changed


def test_dict_insertion_order_does_not_change_digest() -> None:
    payload = _event().canonical_payload()
    reordered = {k: payload[k] for k in reversed(list(payload.keys()))}
    assert stable_payload_hash(payload) == stable_payload_hash(reordered)


def test_timestamp_normalization_deterministic() -> None:
    e1 = _event(occurred_at=_T0)
    e2 = _event(occurred_at=datetime.fromisoformat(_T0.isoformat()))
    assert canonical_json_bytes(e1.canonical_payload()) == canonical_json_bytes(
        e2.canonical_payload()
    )


def test_decimal_values_remain_exact_in_nested_proof() -> None:
    proof = {"amount": str(Decimal("10.50")), "currency": "USD"}
    e = _event(
        governed_proof=GovernedProofSection(
            proof_id="proof-1",
            proof_digest="sha256:" + ("cd" * 32),
            proof=proof,
        )
    )
    text = canonical_json_bytes(e.canonical_payload()).decode("utf-8")
    assert "10.50" in text
    assert stable_payload_hash(e.canonical_payload()) == stable_payload_hash(
        e.canonical_payload()
    )
