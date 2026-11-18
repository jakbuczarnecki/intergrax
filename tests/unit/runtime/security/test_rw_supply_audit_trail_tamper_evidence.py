# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-B Part E: audit-trail tamper evidence (NT-RW-SUP-07), listing and redaction (NT-RW-SUP-17).

Both trails are exercised: the in-memory trail and the evidence-plane trail over a temporary SQLite
runtime event store (in-process fixture, no live backend). Secret-shaped values are generated at
run time and never printed.
"""

from __future__ import annotations

import dataclasses
import json
import secrets
import sqlite3
from pathlib import Path

import pytest
from intergrax.runtime.events.store import resolve_runtime_event_persistence
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.security import security_audit_trail as trail_module
from intergrax.runtime.security.security_audit_trail import (
    EvidencePlaneSecurityAuditTrail,
    SecurityAuditTrail,
)

pytestmark = [pytest.mark.unit]

_TENANT = "tenant-a"


def _evidence_trail(db_path: Path) -> EvidencePlaneSecurityAuditTrail:
    port = resolve_runtime_event_persistence(db_path=db_path)
    assert port is not None
    return EvidencePlaneSecurityAuditTrail(port)


def _append_three(trail: SecurityAuditTrail | EvidencePlaneSecurityAuditTrail) -> None:
    for index in range(3):
        trail.append(
            tenant_id=_TENANT,
            action=f"action-{index}",
            actor_id="operator-1",
            resource=f"resource-{index}",
            metadata={"index": index},
        )


def _report(trail: SecurityAuditTrail | EvidencePlaneSecurityAuditTrail) -> object:
    verify = getattr(trail, "verify_chain", None)
    assert callable(verify), "the trail exposes no chain verification (EX-D6-01)"
    return verify(_TENANT)


def _stored_rows(db_path: Path) -> list[tuple[int, str]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, event_json FROM runtime_events ORDER BY id").fetchall()
    conn.close()
    return [(int(row_id), str(text)) for row_id, text in rows]


def _credential_shaped_value() -> str:
    return "".join(("s", "k", "-")) + secrets.token_hex(16)


# --------------------------------------------------------------------- NT-RW-SUP-07 (EX-D6-01, F-D6-02)


def test_sup_07a_modified_entry_detected(tmp_path: Path) -> None:
    memory = SecurityAuditTrail()
    _append_three(memory)
    entries = memory._entries
    entries[1] = dataclasses.replace(entries[1], metadata={"index": 99})
    report = _report(memory)
    assert getattr(report, "intact") is False
    assert getattr(report, "first_broken_index") == 1

    db_path = tmp_path / "evidence.db"
    durable = _evidence_trail(db_path)
    _append_three(durable)
    row_id, text = _stored_rows(db_path)[1]
    data = json.loads(text)
    data["payload"]["data"]["action"] = "action-forged"
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE runtime_events SET event_json = ? WHERE id = ?", (json.dumps(data), row_id))
    conn.close()
    report = _report(_evidence_trail(db_path))
    assert getattr(report, "intact") is False
    assert getattr(report, "first_broken_index") == 1
    assert getattr(report, "reason") == "entry_digest_mismatch"


def test_sup_07b_removed_interior_entry_detected(tmp_path: Path) -> None:
    memory = SecurityAuditTrail()
    _append_three(memory)
    del memory._entries[1]
    report = _report(memory)
    assert getattr(report, "intact") is False
    assert getattr(report, "first_broken_index") == 1

    db_path = tmp_path / "evidence.db"
    durable = _evidence_trail(db_path)
    _append_three(durable)
    row_id, _text = _stored_rows(db_path)[1]
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM runtime_events WHERE id = ?", (row_id,))
    conn.close()
    report = _report(_evidence_trail(db_path))
    assert getattr(report, "intact") is False
    assert getattr(report, "reason") == "prev_digest_mismatch"


def test_sup_07c_clean_chain_verifies(tmp_path: Path) -> None:
    memory = SecurityAuditTrail()
    _append_three(memory)
    report = _report(memory)
    assert getattr(report, "intact") is True
    assert getattr(report, "verified_count") == 3

    db_path = tmp_path / "evidence.db"
    durable = _evidence_trail(db_path)
    _append_three(durable)
    first = _report(durable)
    restarted = _evidence_trail(db_path)
    second = _report(restarted)
    assert getattr(first, "intact") is True and getattr(second, "intact") is True
    assert getattr(first, "verified_count") == 3
    assert getattr(first, "head_digest") == getattr(second, "head_digest") == restarted.head_digest(_TENANT)
    entries = restarted.list_entries(tenant_id=_TENANT)
    assert entries[0].prev_digest == trail_module.SECURITY_AUDIT_CHAIN_GENESIS
    assert [e.prev_digest for e in entries[1:]] == [e.entry_digest for e in entries[:-1]]
    # A restarted trail continues the stored chain.
    restarted.append(tenant_id=_TENANT, action="action-3", actor_id="operator-1", resource="resource-3")
    assert getattr(_report(_evidence_trail(db_path)), "verified_count") == 4


# --------------------------------------------------------------------- NT-RW-SUP-17 (F-D6-03)


def test_sup_17a_listing_reports_possible_truncation() -> None:
    store = InMemoryRuntimeEventStore()
    port = resolve_runtime_event_persistence(implementation=store)
    assert port is not None
    trail = EvidencePlaneSecurityAuditTrail(port)
    for index in range(999):
        trail.append(tenant_id=_TENANT, action="a", actor_id="u", resource=f"r-{index}")
    assert len(trail.list_entries(tenant_id=_TENANT)) == 999
    trail.append(tenant_id=_TENANT, action="a", actor_id="u", resource="r-999")
    truncated = getattr(trail_module, "AuditTrailListingTruncated", None)
    assert truncated is not None, "a listing at the port limit must not silently return the oldest entries"
    with pytest.raises(truncated):
        trail.list_entries(tenant_id=_TENANT)


def test_sup_17b_in_memory_value_redacted() -> None:
    value = _credential_shaped_value()
    trail = SecurityAuditTrail()
    entry = trail.append(tenant_id=_TENANT, action="a", actor_id="u", resource="r", metadata={"note": value})
    stored = trail.list_entries(tenant_id=_TENANT)[0]
    leaked = value in json.dumps(stored.metadata) or value in json.dumps(entry.metadata)
    assert not leaked, "secret-shaped value stored unredacted (value not shown)"
    redacted = stored.metadata["note"] == "[REDACTED:secret-shaped]"
    assert redacted, "secret-shaped value not replaced by the marker (value not shown)"


def test_sup_17c_evidence_plane_value_redacted(tmp_path: Path) -> None:
    value = _credential_shaped_value()
    db_path = tmp_path / "evidence.db"
    trail = _evidence_trail(db_path)
    trail.append(tenant_id=_TENANT, action="a", actor_id="u", resource="r", metadata={"note": value, "count": 2})
    stored = _evidence_trail(db_path).list_entries(tenant_id=_TENANT)[0]
    redacted = stored.metadata["note"] == "[REDACTED:secret-shaped]"
    assert redacted, "secret-shaped value not replaced by the marker (value not shown)"
    assert stored.metadata["count"] == 2
    leaked = any(value in text for _row_id, text in _stored_rows(db_path))
    assert not leaked, "secret-shaped value stored unredacted (value not shown)"
