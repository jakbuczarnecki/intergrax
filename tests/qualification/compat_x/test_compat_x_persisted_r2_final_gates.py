# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-FINAL current-head reconciliation gates and adversarial probes."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.qualification.compat_x._compat_x_persisted_r2_final import (
    LegacyObligationFingerprint,
    PersistedFamilyFingerprint,
    R2FinalDrift,
    build_persisted_r2_final_report,
    compute_r2_final_drift,
    current_family_fingerprints,
    current_obligation_fingerprints,
    load_accepted_r2_baseline,
    persisted_r2_final_gate_passes,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_r2_final_gate_passes_on_current_head() -> None:
    report = build_persisted_r2_final_report()
    assert persisted_r2_final_gate_passes(report)
    assert report.legacy_obligation_count == 16
    assert report.unclassified_candidates == 0
    assert report.mechanical_family_ids == report.inventory_family_ids
    assert len(report.inventory_family_ids) == 28
    assert not report.drift.family_additions
    assert not report.drift.family_removals
    assert not report.drift.family_fingerprint_drift


def test_cx_r2_final_probe_a_added_family_fails() -> None:
    baseline = load_accepted_r2_baseline()
    families = current_family_fingerprints()
    families["synthetic.added_family"] = PersistedFamilyFingerprint(
        family_id="synthetic.added_family",
        version_disposition="EXPLICIT_PAYLOAD_VERSION",
        current_version="synthetic.v1",
        version_owner_path="intergrax/synthetic.py",
        migration_disposition="CURRENT_ONLY_FAIL_CLOSED",
        migration_owner_path=None,
        writer_paths=("intergrax/synthetic.py",),
        reader_paths=("intergrax/synthetic.py",),
        storage_kind="FILE_JSON",
        canonical_payload_type="Synthetic",
        owner_module_path="intergrax/synthetic.py",
    )
    drift = compute_r2_final_drift(
        baseline,
        frozenset(families),
        families,
        current_obligation_fingerprints(),
    )
    assert "synthetic.added_family" in drift.family_additions
    report = build_persisted_r2_final_report(family_fingerprints=families)
    assert not persisted_r2_final_gate_passes(report)


def test_cx_r2_final_probe_b_removed_family_fails() -> None:
    baseline = load_accepted_r2_baseline()
    families = current_family_fingerprints()
    removed_id = next(iter(baseline.accepted_persisted_family_ids))
    del families[removed_id]
    drift = compute_r2_final_drift(
        baseline,
        frozenset(families),
        families,
        current_obligation_fingerprints(),
    )
    assert removed_id in drift.family_removals
    report = build_persisted_r2_final_report(family_fingerprints=families)
    assert not persisted_r2_final_gate_passes(report)


def test_cx_r2_final_probe_c_same_count_substitution_fails() -> None:
    baseline = load_accepted_r2_baseline()
    families = current_family_fingerprints()
    removed_id = "memory.user_profile"
    added_id = "synthetic.substitute_family"
    assert removed_id in families
    removed_fp = families.pop(removed_id)
    families[added_id] = removed_fp.__class__(
        family_id=added_id,
        version_disposition=removed_fp.version_disposition,
        current_version=removed_fp.current_version,
        version_owner_path=removed_fp.version_owner_path,
        migration_disposition=removed_fp.migration_disposition,
        migration_owner_path=removed_fp.migration_owner_path,
        writer_paths=removed_fp.writer_paths,
        reader_paths=removed_fp.reader_paths,
        storage_kind=removed_fp.storage_kind,
        canonical_payload_type=removed_fp.canonical_payload_type,
        owner_module_path=removed_fp.owner_module_path,
    )
    assert len(families) == 28
    drift = compute_r2_final_drift(
        baseline,
        frozenset(families),
        families,
        current_obligation_fingerprints(),
    )
    assert removed_id in drift.family_removals
    assert added_id in drift.family_additions
    report = build_persisted_r2_final_report(family_fingerprints=families)
    assert not persisted_r2_final_gate_passes(report)


def test_cx_r2_final_probe_d_version_owner_drift_fails() -> None:
    families = current_family_fingerprints()
    target = families["memory.user_profile"]
    families["memory.user_profile"] = replace(
        target, version_owner_path="intergrax/synthetic_version_owner.py"
    )
    report = build_persisted_r2_final_report(family_fingerprints=families)
    assert "memory.user_profile" in report.drift.family_fingerprint_drift
    assert not persisted_r2_final_gate_passes(report)


def test_cx_r2_final_probe_e_migration_owner_drift_fails() -> None:
    families = current_family_fingerprints()
    target = families["memory.user_profile"]
    families["memory.user_profile"] = replace(
        target, migration_owner_path="intergrax/synthetic_migration_owner.py"
    )
    report = build_persisted_r2_final_report(family_fingerprints=families)
    assert not persisted_r2_final_gate_passes(report)


def test_cx_r2_final_probe_f_obligation_disposition_drift_fails() -> None:
    obligations = current_obligation_fingerprints()
    key = "memory.user_profile::LEGACY_UNVERSIONED_PRE_V1"
    original = obligations[key]
    obligations[key] = replace(original, disposition="CURRENT_ONLY_FAIL_CLOSED")
    report = build_persisted_r2_final_report(obligation_fingerprints=obligations)
    assert key in report.drift.obligation_fingerprint_drift
    assert not persisted_r2_final_gate_passes(report)


def test_cx_r2_final_probe_g_sanctioned_reader_drift_fails() -> None:
    obligations = current_obligation_fingerprints()
    key = "memory.user_profile::LEGACY_UNVERSIONED_PRE_V1"
    original = obligations[key]
    obligations[key] = replace(
        original, sanctioned_reader_symbols=("synthetic_reader",)
    )
    report = build_persisted_r2_final_report(obligation_fingerprints=obligations)
    assert not persisted_r2_final_gate_passes(report)


def test_cx_r2_final_probe_h_constituent_gate_failure_injected() -> None:
    good = build_persisted_r2_final_report()
    broken = replace(
        good,
        drift=R2FinalDrift(
            family_additions=(),
            family_removals=(),
            family_fingerprint_drift=(),
            obligation_additions=(),
            obligation_removals=(),
            obligation_fingerprint_drift=("synthetic::obligation",),
        ),
    )
    assert not persisted_r2_final_gate_passes(broken)


def test_cx_r2_final_baseline_fixture_matches_accepted_family_list() -> None:
    baseline = load_accepted_r2_baseline()
    assert len(baseline.accepted_persisted_family_ids) == 28
    assert len(baseline.accepted_legacy_obligation_fingerprints) == 16
    provenance = baseline.provenance
    assert provenance["durable_inventory_accepted_sha"] == "78bd502b5da5284b1bdaae2e321b7076fa22227d"
    assert provenance["schema_gate_accepted_sha"] == "bb5900cd9a502f3e967b88ee54e4e18f6ff101cb"
    assert provenance["legacy_migration_accepted_sha"] == "2e7d6173cc13bd1fb5ae48ef02292ee76bd22137"
