# © Artur Czarnecki. All rights reserved.

"""PDC-D6 F-D6-02 (relabel part) — the multi-region trail does not claim more than it proves.

``MultiRegionSecurityAuditTrail`` keeps per-region lists of entry ids in the same process and compares
those lists with each other and with a sealed prefix of the same lists. It never reads the stored
entries back and the entries carry no hash chain or signature. Tamper evidence is an excluded finding
(missing feature); these tests pin the corrected wording and the unchanged behaviour, so a future
tamper-evidence ticket has to change them deliberately.
"""

from __future__ import annotations

import pytest
from intergrax.runtime.security import multi_region_audit_trail as module
from intergrax.runtime.security.multi_region_audit_trail import (
    MultiRegionAuditTrailReport,
    MultiRegionSecurityAuditTrail,
)
from intergrax.runtime.security.security_audit_trail import SecurityAuditTrail

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _flat(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def _trail() -> tuple[MultiRegionSecurityAuditTrail, SecurityAuditTrail]:
    primary = SecurityAuditTrail()
    return MultiRegionSecurityAuditTrail(regions=("eu", "us"), primary=primary), primary


def _append(trail: MultiRegionSecurityAuditTrail, index: int) -> None:
    trail.append(tenant_id="tenant-a", action=f"action.{index}", actor_id="svc", resource="res")


def test_module_and_class_docstrings_say_the_regions_are_in_process_bookkeeping() -> None:
    for text in (module.__doc__, MultiRegionSecurityAuditTrail.__doc__):
        assert "in-process" in _flat(text)
    assert "does not detect" in _flat(MultiRegionSecurityAuditTrail.__doc__)
    assert "replicated across configured regions" not in _flat(MultiRegionSecurityAuditTrail.__doc__)
    assert "immutable multi-region security audit trail" not in _flat(module.__doc__)


def test_the_report_class_and_its_flags_describe_what_they_measure() -> None:
    assert "in-process" in _flat(MultiRegionAuditTrailReport.__doc__)
    for name in ("replicated", "immutable"):
        description = _flat(MultiRegionAuditTrailReport.model_fields[name].description)
        assert "in-process" in description, name
        assert "does not prove" in description, name


def test_verify_replication_still_reports_the_consistency_of_the_bookkeeping() -> None:
    trail, _ = _trail()
    for index in range(3):
        _append(trail, index)
    trail.seal_prefix()
    _append(trail, 3)

    report = trail.verify_replication()

    assert report.replicated is True
    assert report.immutable is True
    assert [replica.entry_count for replica in report.regions] == [4, 4]
    assert [replica.region_id for replica in report.regions] == ["eu", "us"]


def test_a_tampered_sealed_prefix_of_the_bookkeeping_is_still_detected() -> None:
    trail, _ = _trail()
    for index in range(3):
        _append(trail, index)
    trail.seal_prefix()
    trail._region_entry_ids["us"][0] = "audit_forged"

    report = trail.verify_replication()

    assert report.immutable is False


def test_deleting_stored_entries_is_not_detected_by_the_bookkeeping_documented_limit() -> None:
    """Pins the documented limit: the flags do not read the stored entries (tamper evidence is excluded)."""
    trail, primary = _trail()
    for index in range(3):
        _append(trail, index)
    trail.seal_prefix()
    primary._entries.clear()

    report = trail.verify_replication()

    assert (report.replicated, report.immutable) == (True, True)
    assert trail.list_entries(tenant_id="tenant-a") == ()
