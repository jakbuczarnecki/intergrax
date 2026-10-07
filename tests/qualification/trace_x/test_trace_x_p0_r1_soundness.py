# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P0-R1 closed-world soundness, provenance, and child-mapping gates."""

from __future__ import annotations

import inspect
import subprocess

import pytest

from tests.qualification.trace_x._trace_x_p0_support import (
    ARCHITECTURE_LOCK,
    FRZ_TO_CHILD,
    MANDATORY_FRZ_TRC_IDS,
    TRACE_MECHANISM_CLASS_REGISTRY,
    TRACE_X_CHILD_DECOMPOSITION,
    TRACE_X_P0_AUDITED_HEAD,
    TRACE_X_P0_START_HEAD,
    TraceXChildId,
    _SENSITIVE_CLASS_SUFFIXES,
    assert_frz_to_child_mapping_consistent,
    assert_registry_no_unexplained_orphans,
    assert_registry_references_valid_surfaces,
    assert_reverse_reconstruction_child_mapping_consistent,
    assert_sensitive_classes_explicitly_classified,
    discover_sensitive_classes,
    repo_root,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO_ROOT = repo_root()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()


def test_txp0_r1_q01_all_sensitive_categories_enforced() -> None:
    discovered = discover_sensitive_classes()
    suffix_hits = {name for name in discovered for suffix in _SENSITIVE_CLASS_SUFFIXES if name.endswith(suffix)}
    assert suffix_hits, "expected at least one sensitive suffix discovery"
    assert_sensitive_classes_explicitly_classified(discovered)


@pytest.mark.parametrize(
    ("unknown_name", "suffix"),
    [
        ("UnknownReconstructor", "Reconstructor"),
        ("UnknownCausalEvidence", "CausalEvidence"),
        ("UnknownLineageReader", "LineageReader"),
        ("UnknownLineageWriter", "LineageWriter"),
        ("UnknownLineagePersistence", "LineagePersistence"),
    ],
    ids=[
        "reconstructor",
        "causal_evidence",
        "lineage_reader",
        "lineage_writer",
        "lineage_persistence",
    ],
)
def test_txp0_r1_q02_through_q06_unknown_sensitive_mechanism_fails(
    unknown_name: str,
    suffix: str,
) -> None:
    assert unknown_name.endswith(suffix)
    assert unknown_name not in TRACE_MECHANISM_CLASS_REGISTRY
    synthetic = {unknown_name: ["synthetic/qualification_fixture.py"]}
    with pytest.raises(AssertionError, match="unclassified sensitive"):
        assert_sensitive_classes_explicitly_classified(synthetic)


def test_txp0_r1_q07_registry_references_valid_tx_surfaces() -> None:
    assert_registry_references_valid_surfaces()


def test_txp0_r1_q08_no_unexplained_registry_orphans() -> None:
    discovered = discover_sensitive_classes()
    assert_registry_no_unexplained_orphans(discovered)


def test_txp0_r1_q09_provenance_test_truthfully_named() -> None:
    from tests.qualification.trace_x import _trace_x_p0_qualification_tests as gates

    assert hasattr(gates, "test_txp0_q01_task_provenance_anchor_preserved")
    assert not hasattr(gates, "test_txp0_q01_current_head_anchor")


def test_txp0_r1_q10_no_descendant_as_current_head_gate() -> None:
    source = inspect.getsource(
        __import__(
            "tests.qualification.trace_x._trace_x_p0_qualification_tests",
            fromlist=["test_txp0_q01_task_provenance_anchor_preserved"],
        ).test_txp0_q01_task_provenance_anchor_preserved,
    )
    assert "TRACE_X_P0_AUDITED_HEAD" not in source
    assert "is-ancestor" not in source
    head = _git_head()
    if head != TRACE_X_P0_AUDITED_HEAD:
        assert head != TRACE_X_P0_START_HEAD or TRACE_X_P0_START_HEAD != TRACE_X_P0_AUDITED_HEAD


def test_txp0_r1_q11_frz_child_mapping_unique() -> None:
    assert_frz_to_child_mapping_consistent()
    owners = list(FRZ_TO_CHILD.values())
    assert len(owners) == len(MANDATORY_FRZ_TRC_IDS)


def test_txp0_r1_q12_p1_covers_trc_02_and_12() -> None:
    p1 = next(c for c in TRACE_X_CHILD_DECOMPOSITION if c.child_id == TraceXChildId.P1.value)
    assert set(p1.frz_criteria) == {"FRZ-TRC-02", "FRZ-TRC-12"}


def test_txp0_r1_q13_p2_covers_trc_01() -> None:
    p2 = next(c for c in TRACE_X_CHILD_DECOMPOSITION if c.child_id == TraceXChildId.P2.value)
    assert p2.frz_criteria == ("FRZ-TRC-01",)


def test_txp0_r1_q14_p3_covers_trc_03_04_06() -> None:
    p3 = next(c for c in TRACE_X_CHILD_DECOMPOSITION if c.child_id == TraceXChildId.P3.value)
    assert set(p3.frz_criteria) == {"FRZ-TRC-03", "FRZ-TRC-04", "FRZ-TRC-06"}


def test_txp0_r1_q15_p4_covers_trc_05() -> None:
    p4 = next(c for c in TRACE_X_CHILD_DECOMPOSITION if c.child_id == TraceXChildId.P4.value)
    assert p4.frz_criteria == ("FRZ-TRC-05",)


def test_txp0_r1_q16_p5_covers_trc_07_08_11() -> None:
    p5 = next(c for c in TRACE_X_CHILD_DECOMPOSITION if c.child_id == TraceXChildId.P5.value)
    assert set(p5.frz_criteria) == {"FRZ-TRC-07", "FRZ-TRC-08", "FRZ-TRC-11"}


def test_txp0_r1_q17_p6_covers_trc_09_10() -> None:
    p6 = next(c for c in TRACE_X_CHILD_DECOMPOSITION if c.child_id == TraceXChildId.P6.value)
    assert set(p6.frz_criteria) == {"FRZ-TRC-09", "FRZ-TRC-10"}


def test_txp0_r1_q18_reverse_mappings_consistent() -> None:
    assert_reverse_reconstruction_child_mapping_consistent()


def test_txp0_r1_q19_trace_x_cert_covers_all_12() -> None:
    cert = next(c for c in TRACE_X_CHILD_DECOMPOSITION if c.child_id == TraceXChildId.CERT.value)
    assert set(cert.frz_criteria) == set(MANDATORY_FRZ_TRC_IDS)


def test_txp0_r1_q20_no_architecture_lock_regression() -> None:
    assert len(ARCHITECTURE_LOCK) >= 9
    lock_ids = {entry.lock_id for entry in ARCHITECTURE_LOCK}
    assert "TX-LOCK-04" in lock_ids
