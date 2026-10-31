# © Artur Czarnecki. All rights reserved.

"""B6-F10 adversarial gates — injected defects must yield computed FAIL (no declarative PASS)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.qualification.ebh_7._ebh7_f10_canonical_inputs import (
    F10Inputs,
    VendorConcern,
    cached_canonical_f10_inputs,
)
from tests.qualification.ebh_7._ebh7_f10_ecap_normative_matrix import (
    DISPOSITION_BLOCKED,
    DISPOSITION_QUALIFIED,
    certify_f10,
)
from tests.qualification.ebh_7._ebh7_f10_types import ChangeClass, EcapGate, GateVerdict, ResultStatus

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def _base() -> F10Inputs:
    return cached_canonical_f10_inputs()


def _assert_fails(inputs: F10Inputs, obligation_id: str, reason_fragment: str) -> None:
    cert = certify_f10(inputs)
    result = next((r for r in cert.results if r.obligation.obligation_id == obligation_id), None)
    assert result is not None, f"obligation {obligation_id} not produced"
    assert result.status is ResultStatus.FAIL, f"{obligation_id} not FAIL: {result.status}"
    assert any(reason_fragment in reason for reason in result.reasons), result.reasons
    assert cert.verdict(result.obligation.gate).verdict is GateVerdict.BLOCKED
    assert cert.disposition == DISPOSITION_BLOCKED


def test_f10_adv_c1_missing_family_in_catalog_fails() -> None:
    base = _base()
    inputs = replace(base, ebh6_families=tuple(f for f in base.ebh6_families if f.family_id != "BND"))
    _assert_fails(inputs, "C1-FAM-BND", "N/A invalid")
    _assert_fails(inputs, "C1-FAMILY-ID-IDENTITY", "ID mismatch")


def test_f10_adv_c1_missing_tier_fails() -> None:
    base = _base()
    inputs = replace(base, tiers=tuple(t for t in base.tiers if t.tier_id != "Tier-2"))
    _assert_fails(inputs, "C1-TIER-SET", "canonical tier set mismatch")


def test_f10_adv_c1_correct_count_with_swapped_id_fails() -> None:
    base = _base()
    swapped = tuple("XYZ" if fid == "OWN" else fid for fid in base.ebh7_family_matrix_ids)
    assert len(swapped) == len(base.ebh7_family_matrix_ids)
    _assert_fails(replace(base, ebh7_family_matrix_ids=swapped), "C1-FAMILY-ID-IDENTITY", "ID mismatch")


def test_f10_adv_c2_missing_dup_class_fails() -> None:
    base = _base()
    inputs = replace(base, dup_classes=tuple(c for c in base.dup_classes if c != "DUP-06"))
    _assert_fails(inputs, "C2-DUP-CLASS-IDENTITY", "class set mismatch")
    _assert_fails(inputs, "C2-DUP-06", "missing from DUP_CLASSES SSOT")


def test_f10_adv_c2_subset_declared_as_full_fails() -> None:
    base = _base()
    inputs = replace(base, rf_families=tuple(f for f in base.rf_families if f.family_id != "RF-TEN-01"))
    _assert_fails(inputs, "C2-SURFACE-tenant-context", "subset claimed as full")


def test_f10_adv_c3_missing_vbn_domain_fails() -> None:
    base = _base()
    domain = "5-vector-database"
    inputs = replace(
        base,
        vendor_concerns=tuple(c for c in base.vendor_concerns if c.domain != domain),
        domain_na={k: v for k, v in base.domain_na.items() if k != domain},
    )
    _assert_fails(inputs, f"C3-VBN-{domain}", "neither covered")


def test_f10_adv_c3_unknown_concern_fails() -> None:
    base = _base()
    rogue = VendorConcern("rogue.quantum_backend", "99-quantum", "owner", "composer", "INTERNAL", "ev", "")
    inputs = replace(base, vendor_concerns=(*base.vendor_concerns, rogue))
    _assert_fails(inputs, "C3-CONCERN-rogue.quantum_backend", "unknown concern")


def test_f10_adv_c3_missing_owner_or_classification_fails() -> None:
    base = _base()
    first = base.vendor_concerns[0]
    stripped = replace(first, semantic_owner="", classification="")
    inputs = replace(base, vendor_concerns=(stripped, *base.vendor_concerns[1:]))
    _assert_fails(inputs, f"C3-CONCERN-{first.concern_id}", "missing owner")
    _assert_fails(inputs, f"C3-CONCERN-{first.concern_id}", "unknown classification")


def test_f10_adv_c4_unclassified_change_fails() -> None:
    base = _base()
    inputs = replace(base, changed_paths=(*base.changed_paths, "intergrax/quantum/new_mechanism.py"))
    _assert_fails(inputs, "C4-UNCLASSIFIED", "unclassified change intergrax/quantum/new_mechanism.py")


def test_f10_adv_c4_class_without_required_regression_fails() -> None:
    base = _base()
    regressions = dict(base.c4_class_regressions)
    regressions[ChangeClass.VENDOR_BACKEND] = ()
    inputs = replace(base, c4_class_regressions=regressions)
    _assert_fails(inputs, "C4-CLASS-vendor-backend", "has no required regression")


def test_f10_adv_missing_execution_attestation_fails() -> None:
    _assert_fails(replace(_base(), profile_argv=()), "C4-CLASS-vendor-backend", "not executed by MANIFEST")


def test_f10_adv_wrong_execution_sha_fails() -> None:
    inputs = replace(_base(), identity_violations=("intergrax/runtime/nexus/tools/invoker.py",))
    _assert_fails(inputs, "C3-PROHIBITED-ZERO", "executed tree differs from TEST_EXECUTION_SHA")


def test_f10_adv_c5_pass_despite_blocker_fails() -> None:
    base = _base()
    inputs = replace(
        base,
        ebh6_families=tuple(f for f in base.ebh6_families if f.family_id != "BND"),
        report_text=base.report_text + f"\n| ECAP-C1 | PASS |\n{DISPOSITION_QUALIFIED}\n",
    )
    cert = certify_f10(inputs)
    assert cert.verdict(EcapGate.C1).verdict is GateVerdict.BLOCKED
    _assert_fails(inputs, "C5-NO-FALSE-PASS", "NORMATIVELY QUALIFIED while C1–C4 blocked")
    _assert_fails(inputs, "C5-NO-FALSE-PASS", "static claim")


def test_f10_adv_matrix_disk_differs_from_generated_fails() -> None:
    inputs = replace(_base(), matrix_disk_text="# tampered\n")
    _assert_fails(inputs, "C5-MATRIX-SYNC", "differs from generated")
