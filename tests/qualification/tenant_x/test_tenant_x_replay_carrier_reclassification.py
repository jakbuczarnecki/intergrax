# © Artur Czarnecki. All rights reserved.

"""TENANT-X replay carrier reclassification (T-IGX-041-I): the strengthened delta check fails closed.

The tests drive the pure helper ``replay_carrier_delta_violations`` with the real discovered
set (one cached scan) and with mutated copies of it. They prove that an unclassified extra
carrier, a swap with an unchanged count, a vanished classified id, a removed id that is still
present, a missing role or reason and a mutated pin each produce a violation.
"""

from __future__ import annotations

import hashlib

import pytest
from tests.qualification.tenant_x import _replay_x_carrier_reconciliation as rec
from tests.qualification.tenant_x._inventory import class_field_tenant_surface_ids

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

# SHA-256 of the sorted, newline-joined 19 ids of ``REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS`` as
# merged before T-IGX-041-I: the set must stay byte-identical.
_BASE_IDS_SHA256 = "f6c8c18e9fa2299fbb886e1f3b1ae68177e961f0576158f15908a68154395ef1"
_EXTRA_ID = "intergrax/_t_igx_041_i_probe.py::ProbeRecord.tenant_id::class.field.tenant"


@pytest.fixture(scope="module")
def discovered() -> frozenset[str]:
    return class_field_tenant_surface_ids()


def _violations(discovered: frozenset[str], **overrides: object) -> list[str]:
    params: dict[str, object] = {"discovered": discovered, "pin_total": rec.TENANT_X_REPLAY_HEAD_CARRIER_TOTAL}
    params.update(overrides)
    return rec.replay_carrier_delta_violations(**params)  # type: ignore[arg-type]


def test_real_data_has_no_violation(discovered: frozenset[str]) -> None:
    assert _violations(discovered) == []
    assert len(discovered) == rec.TENANT_X_REPLAY_HEAD_CARRIER_TOTAL


def test_unclassified_extra_carrier_fails_by_count_and_digest(discovered: frozenset[str]) -> None:
    found = _violations(discovered | {_EXTRA_ID})
    assert any("delta" in v for v in found), found
    assert any("digest" in v for v in found), found


def test_swap_with_unchanged_count_fails_through_the_digest(discovered: frozenset[str]) -> None:
    classified = rec.REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS | frozenset(rec.REPLAY_P7_ADDED_CLASSIFICATION)
    victim = min(discovered - classified)  # an unclassified carrier of the original 800
    swapped = (discovered - {victim}) | {_EXTRA_ID}
    assert len(swapped) == len(discovered)
    found = _violations(swapped)
    assert found and all("digest" in v for v in found), found


def test_classified_carrier_that_is_no_longer_discovered_fails(discovered: frozenset[str]) -> None:
    added = min(rec.REPLAY_P7_ADDED_CLASSIFICATION)
    base = min(rec.REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS)
    for gone in (added, base):
        found = _violations(discovered - {gone})
        assert any("not discovered" in v and gone in v for v in found), (gone, found)


def test_removed_carrier_that_is_still_discovered_fails(discovered: frozenset[str]) -> None:
    removed = min(rec.REPLAY_P7_REMOVED_SURFACE_IDS)
    found = _violations(discovered | {removed})
    assert any("still discovered" in v and removed in v for v in found), found


def test_removed_carrier_is_absent_at_head(discovered: frozenset[str]) -> None:
    assert rec.REPLAY_P7_REMOVED_SURFACE_IDS
    assert rec.REPLAY_P7_REMOVED_SURFACE_IDS.isdisjoint(discovered)


@pytest.mark.parametrize("bad_role", ["", "AUTHORITY", "tenant_scoped", "TENANT_SCOPED "])
def test_role_outside_the_closed_vocabulary_fails(discovered: frozenset[str], bad_role: str) -> None:
    sid = min(rec.REPLAY_P7_ADDED_CLASSIFICATION)
    mutated = dict(rec.REPLAY_P7_ADDED_CLASSIFICATION)
    mutated[sid] = (bad_role, "a reason")
    found = _violations(discovered, added=mutated)
    assert any("role" in v and sid in v for v in found), found


@pytest.mark.parametrize("bad_reason", ["", "   ", "\n"])
def test_missing_reason_fails(discovered: frozenset[str], bad_reason: str) -> None:
    sid = min(rec.REPLAY_P7_ADDED_CLASSIFICATION)
    mutated = dict(rec.REPLAY_P7_ADDED_CLASSIFICATION)
    mutated[sid] = (rec.REPLAY_P7_ROLE_TENANT_FREE, bad_reason)
    found = _violations(discovered, added=mutated)
    assert any("reason" in v and sid in v for v in found), found


def test_classification_overlap_is_rejected(discovered: frozenset[str]) -> None:
    base_id = min(rec.REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS)
    mutated = dict(rec.REPLAY_P7_ADDED_CLASSIFICATION)
    mutated[base_id] = (rec.REPLAY_P7_ROLE_TENANT_SCOPED, "already part of the 19-id set")
    assert any("overlap" in v for v in _violations(discovered, added=mutated))
    removed = rec.REPLAY_P7_REMOVED_SURFACE_IDS | {min(rec.REPLAY_P7_ADDED_CLASSIFICATION)}
    assert any("overlap" in v for v in _violations(discovered, removed=removed))


def test_pin_arithmetic_holds_for_the_real_data_and_fails_for_a_mutated_pin(discovered: frozenset[str]) -> None:
    expected = (
        rec.TENANT_X_FINAL_IMMUTABLE_CARRIER_TOTAL
        + len(rec.REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS)
        + len(rec.REPLAY_P7_ADDED_CLASSIFICATION)
        - len(rec.REPLAY_P7_REMOVED_SURFACE_IDS)
    )
    assert rec.TENANT_X_REPLAY_HEAD_CARRIER_TOTAL == expected
    for mutated_pin in (expected - 1, expected + 1):
        found = _violations(discovered, pin_total=mutated_pin)
        assert any("pin" in v for v in found), (mutated_pin, found)


def test_wrong_ids_digest_fails(discovered: frozenset[str]) -> None:
    found = _violations(discovered, ids_sha256="0" * 64)
    assert any("digest" in v for v in found), found


def test_the_nineteen_id_set_is_unchanged_and_still_discovered(discovered: frozenset[str]) -> None:
    base = rec.REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS
    assert len(base) == 19
    assert hashlib.sha256("\n".join(sorted(base)).encode("utf-8")).hexdigest() == _BASE_IDS_SHA256
    assert base <= discovered


def test_every_added_carrier_is_a_discovered_class_field_surface(discovered: frozenset[str]) -> None:
    for sid, (role, reason) in rec.REPLAY_P7_ADDED_CLASSIFICATION.items():
        assert sid in discovered
        assert sid.endswith("::class.field.tenant")
        assert role in rec.REPLAY_P7_ROLES
        assert reason.strip()


def test_gate_helper_raises_on_a_wrong_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    rec.assert_replay_carrier_delta_classified()  # passes at head
    monkeypatch.setattr(rec, "TENANT_X_REPLAY_CLASS_FIELD_IDS_SHA256", "0" * 64)
    with pytest.raises(AssertionError, match="digest"):
        rec.assert_replay_carrier_delta_classified()
