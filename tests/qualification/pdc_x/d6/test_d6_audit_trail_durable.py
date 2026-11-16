# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R020: the security audit trail is durable, append-only and tenant-scoped."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from intergrax.applications._shared.security_audit_trail_wiring import (
    resolve_security_audit_trail_wiring,
)
from intergrax.core.security.secret_safety import SecretSafetyValidationError
from intergrax.core.security.security_configuration import (
    AUDIT_REGIONS_INVALID,
    AUDIT_SINK_MISSING,
    SecurityConfigurationError,
)
from intergrax.runtime.security.security_audit_trail import (
    EvidencePlaneSecurityAuditTrail,
    SecurityAuditTrail,
)

from ._support import generated_secret, lab_env, open_evidence_plane, product_env

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _trail(db_path: Path) -> EvidencePlaneSecurityAuditTrail:
    return EvidencePlaneSecurityAuditTrail(open_evidence_plane(db_path))


def test_d6_r020_audit_trail_survives_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.db"
    first = _trail(db_path)
    first.append(
        tenant_id="tenant-a",
        action="credential.revoked",
        actor_id="svc.security",
        resource="credential/vendor",
        metadata={"reason": "rotation"},
    )
    first.append(
        tenant_id="tenant-a",
        action="policy.changed",
        actor_id="admin-1",
        resource="policy/egress",
    )

    # simulated restart: a brand-new trail and a brand-new store handle on the same database
    restarted = _trail(db_path)
    entries = restarted.list_entries(tenant_id="tenant-a")
    assert [entry.action for entry in entries] == ["credential.revoked", "policy.changed"]
    assert entries[0].metadata == {"reason": "rotation"}
    assert entries[0].actor_id == "svc.security"
    assert all(entry.entry_id.startswith("audit_") for entry in entries)


def test_d6_r020_trail_is_tenant_scoped(tmp_path: Path) -> None:
    trail = _trail(tmp_path / "evidence.db")
    trail.append(tenant_id="tenant-a", action="a.one", actor_id="u", resource="r")
    trail.append(tenant_id="tenant-b", action="b.one", actor_id="u", resource="r")
    assert [e.action for e in trail.list_entries(tenant_id="tenant-a")] == ["a.one"]
    assert [e.action for e in trail.list_entries(tenant_id="tenant-b")] == ["b.one"]
    assert trail.list_entries(tenant_id="tenant-c") == ()
    with pytest.raises(ValueError):
        trail.list_entries(tenant_id=None)
    with pytest.raises(ValueError):
        trail.append(tenant_id="  ", action="x", actor_id="u", resource="r")


def test_d6_r020_trail_is_append_only(tmp_path: Path) -> None:
    trail = _trail(tmp_path / "evidence.db")
    for name in ("update", "delete", "remove", "clear", "truncate", "rewrite"):
        assert not hasattr(trail, name)
    first = trail.append(tenant_id="tenant-a", action="x.one", actor_id="u", resource="r")
    trail.append(tenant_id="tenant-a", action="x.two", actor_id="u", resource="r")
    ids = [entry.entry_id for entry in trail.list_entries(tenant_id="tenant-a")]
    assert ids[0] == first.entry_id and len(ids) == 2


def test_d6_r020_entries_use_the_canonical_evidence_store_not_a_security_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "evidence.db"
    _trail(db_path).append(tenant_id="tenant-a", action="x.one", actor_id="u", resource="r")
    connection = sqlite3.connect(db_path)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        connection.close()
    assert tables, "runtime event store must own the schema"
    assert not any("security" in table.lower() or "audit" in table.lower() for table in tables)


def test_d6_r020_secret_like_metadata_is_rejected_and_never_persisted(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.db"
    trail = _trail(db_path)
    secret = generated_secret()
    with pytest.raises(SecretSafetyValidationError):
        trail.append(
            tenant_id="tenant-a",
            action="x.one",
            actor_id="u",
            resource="r",
            metadata={"api_key": secret},
        )
    assert trail.list_entries(tenant_id="tenant-a") == ()
    assert secret.encode("utf-8") not in db_path.read_bytes()


def test_d6_r020_wiring_persists_through_the_evidence_plane(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.db"
    env = product_env()
    wiring = resolve_security_audit_trail_wiring(env, evidence_persistence=open_evidence_plane(db_path))
    assert wiring.enabled is True and wiring.trail is not None
    assert wiring.report is not None and wiring.report.replicated and len(wiring.report.regions) >= 2

    restarted = _trail(db_path)
    actions = [e.action for e in restarted.list_entries(tenant_id=env.profile_id)]
    assert actions == ["audit_trail.bootstrap", "audit_trail.ready"]


def test_d6_r020_wiring_without_a_sink_fails_closed_no_memory_fallback() -> None:
    with pytest.raises(SecurityConfigurationError) as excinfo:
        resolve_security_audit_trail_wiring(product_env())
    assert excinfo.value.reason_code == AUDIT_SINK_MISSING


@pytest.mark.parametrize(
    "regions",
    [["eu-central-1"], ["eu-central-1", "eu-central-1"], ["eu-central-1", " "], []],
)
def test_d6_r020_malformed_regions_fail_closed(regions: list[str], tmp_path: Path) -> None:
    env = product_env(audit_trail_regions=regions)
    with pytest.raises(SecurityConfigurationError) as excinfo:
        resolve_security_audit_trail_wiring(env, evidence_persistence=open_evidence_plane(tmp_path / "evidence.db"))
    assert excinfo.value.reason_code == AUDIT_REGIONS_INVALID


def test_d6_r020_lab_profile_and_disabled_flag_stay_disabled() -> None:
    assert resolve_security_audit_trail_wiring(lab_env()).enabled is False
    assert resolve_security_audit_trail_wiring(product_env(immutable_audit_trail_enabled=False)).enabled is False


def test_d6_r020_in_memory_trail_remains_available_for_tests() -> None:
    memory = SecurityAuditTrail()
    memory.append(tenant_id="t", action="a", actor_id="u", resource="r")
    assert len(memory.list_entries(tenant_id="t")) == 1


def test_d6_r020_persisted_payload_is_json_only(tmp_path: Path) -> None:
    db_path = tmp_path / "evidence.db"
    trail = _trail(db_path)
    entry = trail.append(
        tenant_id="tenant-a",
        action="x.one",
        actor_id="u",
        resource="r",
        metadata={"nested": {"count": 3}, "when": Path("a/b")},
    )
    restored = trail.list_entries(tenant_id="tenant-a")[0]
    assert restored.entry_id == entry.entry_id
    assert restored.metadata["nested"] == {"count": 3}
    assert json.dumps(restored.metadata)
