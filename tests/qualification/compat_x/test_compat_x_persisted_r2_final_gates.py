# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-FINAL current-head reconciliation gates and adversarial probes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.qualification.compat_x._compat_x_persisted_r2_final import (
    AcceptedR2Baseline,
    EXPECTED_ACCEPTED_R2_BASELINE_PROVENANCE,
    LegacyObligationFingerprint,
    PersistedFamilyFingerprint,
    R2FinalDrift,
    build_persisted_r2_final_report,
    compute_r2_final_drift,
    current_family_fingerprints,
    current_obligation_fingerprints,
    discover_r2_final_baseline_semantic_typing_violations,
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
    assert baseline.provenance == EXPECTED_ACCEPTED_R2_BASELINE_PROVENANCE
    assert discover_r2_final_baseline_semantic_typing_violations() == ()


def test_cx_r2_final_probe_i_bad_provenance_sha_fails() -> None:
    baseline = load_accepted_r2_baseline()
    bad_provenance = replace(
        baseline.provenance,
        closure_fingerprint_baseline_sha="0000000000000000000000000000000000000000",
    )
    bad_baseline = replace(baseline, provenance=bad_provenance)
    report = build_persisted_r2_final_report(baseline=bad_baseline)
    assert not report.baseline_provenance_valid
    assert not persisted_r2_final_gate_passes(report)


def test_cx_r2_final_probe_j_malformed_family_baseline_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        Path(__file__).resolve().parents[2]
        .joinpath("fixtures", "compat_x", "r2_final_accepted_baseline.json")
        .read_text(encoding="utf-8")
    )
    family_id = "memory.user_profile"
    del fixture["accepted_family_fingerprints"][family_id]["version_disposition"]
    path = tmp_path / "bad_baseline.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_accepted_r2_baseline(path)


def test_cx_r2_final_probe_k_family_id_key_mismatch_fails(tmp_path: Path) -> None:
    baseline = load_accepted_r2_baseline()
    families = dict(baseline.accepted_family_fingerprints)
    target = families["memory.user_profile"]
    families["memory.user_profile"] = replace(target, family_id="synthetic.mismatched_id")
    broken = AcceptedR2Baseline(
        provenance=baseline.provenance,
        accepted_persisted_family_ids=baseline.accepted_persisted_family_ids,
        accepted_family_fingerprints=families,
        accepted_legacy_obligation_fingerprints=baseline.accepted_legacy_obligation_fingerprints,
    )
    with pytest.raises(ValueError, match="family fingerprint key"):
        build_persisted_r2_final_report(baseline=broken)


def test_cx_r2_final_probe_l_obligation_key_mismatch_fails() -> None:
    baseline = load_accepted_r2_baseline()
    obligations = dict(baseline.accepted_legacy_obligation_fingerprints)
    key = "memory.user_profile::LEGACY_UNVERSIONED_PRE_V1"
    original = obligations[key]
    obligations[key] = replace(original, obligation_key="synthetic::wrong_key")
    broken = AcceptedR2Baseline(
        provenance=baseline.provenance,
        accepted_persisted_family_ids=baseline.accepted_persisted_family_ids,
        accepted_family_fingerprints=baseline.accepted_family_fingerprints,
        accepted_legacy_obligation_fingerprints=obligations,
    )
    with pytest.raises(ValueError, match="obligation fingerprint key"):
        build_persisted_r2_final_report(baseline=broken)
