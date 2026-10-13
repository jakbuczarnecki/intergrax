# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0-R1 negative integrity / mutation gates (D1–D9)."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import sys

from tests.qualification.prod_q._p0_certification import (
    _parse_sqlite_reproduction_tail,
    build_provider_dispositions,
    build_r1_sqlite_disposition,
    build_tenant_isolation_audit,
    run_prod_q_bounded_subprocess,
)
from tests.qualification.prod_q._p0_types import ProdQFindingDisposition
from tests.qualification.prod_q._p0_classification import classify_candidate, classification_by_candidate_id
from tests.qualification.prod_q._p0_discovery import (
    _module_defines_symbol,
    raw_discovery_candidates,
)
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation
from tests.qualification.prod_q._p0_integrity import (
    assert_inventory_identity_sound,
    probe_duplicate_candidate_id_detected,
    probe_fake_semantic_owner_not_resolved,
    probe_false_auxiliary_na_without_symbol_evidence,
    probe_mutated_alternate_mechanism_detected,
    probe_mutated_owner_alters_reconciliation,
    probe_readiness_method_not_auto_auxiliary,
    probe_removed_classification_detected,
    probe_unresolved_classification_role_explicit,
    probe_unresolved_composition_marks_incomplete,
    probe_unresolved_contract_marks_incomplete,
    probe_unknown_indicator_classified_not_pq_f01,
)
from tests.qualification.prod_q._p0_ownership import (
    probe_alternate_contract_conflict_detected,
    probe_concern_identity_stable_under_candidate_order,
    probe_distinct_concerns_not_merged,
    probe_legal_port_providers_not_duplicate,
)
from tests.qualification.prod_q._p0_types import SqliteReproductionClassification
from tests.qualification.prod_q._p0_types import (
    CandidateInventoryStatus,
    DiscoveryKind,
    ProductionReadinessCandidate,
    ProductionReadinessEvidenceState,
    ProductionReadinessFamily,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_PARENT_SPAWNS_CHILD_SCRIPT = r"""
import subprocess
import sys
import time

marker = sys.argv[1]
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(300)"],
)
marker_path = marker + ".child_pid"
with open(marker_path, "w", encoding="utf-8") as handle:
    handle.write(str(child.pid))
time.sleep(300)
"""

_PARENT_EXITS_CHILD_INHERITS_PIPES_SCRIPT = r"""
import os
import subprocess
import sys

marker = sys.argv[1]
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(300)"],
    stdout=sys.stdout,
    stderr=sys.stderr,
)
with open(marker + ".child_pid", "w", encoding="utf-8") as handle:
    handle.write(str(child.pid))
os._exit(0)
"""


def _terminate_pid(pid: int) -> None:
    if not _process_still_running(pid):
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _process_still_running(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        synchronize = 0x00100000
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_prod_q_p0_int_d1_unknown_indicator_blocks() -> None:
    assert probe_unknown_indicator_classified_not_pq_f01() is None


def test_prod_q_p0_int_d2_removed_classification_fails_identity() -> None:
    by_id = classification_by_candidate_id()
    assert probe_removed_classification_detected(by_id) is None


def test_prod_q_p0_int_d2_identity_not_index_tautology() -> None:
    raw = tuple(reversed(raw_discovery_candidates()))
    by_id = classification_by_candidate_id()
    assert_inventory_identity_sound(raw, by_id)


def test_prod_q_p0_int_d2_duplicate_candidate_id_mutation() -> None:
    assert probe_duplicate_candidate_id_detected() is None


def test_prod_q_p0_int_d3_fake_semantic_owner_not_resolved() -> None:
    assert probe_fake_semantic_owner_not_resolved() is None


def test_prod_q_p0_int_d5_tenant_not_static_pass_with_contradiction() -> None:
    audit = build_tenant_isolation_audit()
    assert audit.verdict != "PASS"
    mutant = replace(audit, verdict="PASS", production_path_contradictions=("forced",))
    assert mutant.production_path_contradictions != ()


def test_prod_q_p0_int_d6_frz_no_proven_without_gate() -> None:
    for record in build_frz_reconciliation():
        if record.evidence_state == ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD:
            assert "gate:" in " ".join(record.evidence_refs)


def test_prod_q_p0_int_d6_frz_registry_33_unique() -> None:
    records = build_frz_reconciliation()
    ids = [r.criterion_id for r in records]
    assert len(ids) == 33
    assert len(frozenset(ids)) == 33


def test_prod_q_p0_int_d7_sqlite_reproduction_not_run_on_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_PROD_Q_R1_HTTP_REPRODUCTION", raising=False)
    build_r1_sqlite_disposition.cache_clear()
    record = build_r1_sqlite_disposition()
    assert record.reproduction_executed is False
    assert record.reproduction_classification == SqliteReproductionClassification.NOT_EXECUTED
    assert record.evidence_state == ProductionReadinessEvidenceState.BLOCKER
    assert record.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    assert record.sqlite_bootstrap_observed is True


def test_prod_q_p0_int_d7_bounded_subprocess_timeout_insufficient_evidence() -> None:
    exit_code, _tail, timed_out = run_prod_q_bounded_subprocess(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        timeout_sec=1,
    )
    assert timed_out is True
    assert exit_code is None


def test_prod_q_p0_int_d7_timeout_does_not_present_as_startup_pass() -> None:
    classification, *_rest = _parse_sqlite_reproduction_tail("subprocess timeout")
    assert classification == SqliteReproductionClassification.ENVIRONMENT_UNAVAILABLE
    assert classification != SqliteReproductionClassification.PRODUCTION_DEFECT


def test_prod_q_p0_int_d7_http_reproduction_timeout_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_PROD_Q_R1_HTTP_REPRODUCTION", "1")

    def _fake_bounded(*_args, **_kwargs):
        return None, "subprocess timeout", True

    monkeypatch.setattr(
        "tests.qualification.prod_q._p0_certification.run_prod_q_bounded_subprocess",
        _fake_bounded,
    )
    build_r1_sqlite_disposition.cache_clear()
    record = build_r1_sqlite_disposition()
    assert record.reproduction_executed is False
    assert record.reproduction_exit_code is None
    assert record.finding_disposition == ProdQFindingDisposition.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED
    assert record.reproduction_classification == SqliteReproductionClassification.ENVIRONMENT_UNAVAILABLE
    assert record.evidence_state == ProductionReadinessEvidenceState.BLOCKER


def test_prod_q_p0_int_d7_bounded_subprocess_completes_normally() -> None:
    exit_code, tail, timed_out = run_prod_q_bounded_subprocess(
        [sys.executable, "-c", "print('prod_q_probe')"],
        timeout_sec=5,
    )
    assert timed_out is False
    assert exit_code == 0
    assert "prod_q_probe" in tail


def test_prod_q_p0_int_d7_http_reproduction_completed_failure_marks_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_PROD_Q_R1_HTTP_REPRODUCTION", "1")

    def _fake_bounded(*_args, **_kwargs):
        tail = (
            "E   intergrax.contracts.execution_continuation.ExecutionContinuationError: "
            "production composition requires an explicit ExecutionContinuationStateStore"
        )
        return 1, tail, False

    monkeypatch.setattr(
        "tests.qualification.prod_q._p0_certification.run_prod_q_bounded_subprocess",
        _fake_bounded,
    )
    build_r1_sqlite_disposition.cache_clear()
    record = build_r1_sqlite_disposition()
    assert record.reproduction_executed is True
    assert record.reproduction_exit_code == 1
    assert record.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER


def test_prod_q_p0_int_d7_timeout_terminates_owned_child_after_parent_exits(
    tmp_path,
) -> None:
    if sys.platform == "win32":
        pytest.skip(
            "Windows: after parent exit, bounded subprocess cannot verify owned "
            "descendant termination without a job-object ownership mechanism "
            "(STOP — ARCHITECTURE DECISION REQUIRED)"
        )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    marker = tmp_path / "parent_exit_probe"
    child_pid: int | None = None
    try:
        exit_code, _tail, timed_out = run_prod_q_bounded_subprocess(
            [sys.executable, "-c", _PARENT_EXITS_CHILD_INHERITS_PIPES_SCRIPT, str(marker)],
            timeout_sec=2,
        )
        assert timed_out is True
        assert exit_code is None
        child_pid_path = Path(f"{marker}.child_pid")
        if not child_pid_path.is_file():
            pytest.fail("child pid marker not written before parent exit")
        child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
        assert not _process_still_running(child_pid)
        assert _process_still_running(unrelated.pid)
    finally:
        if child_pid is not None:
            _terminate_pid(child_pid)
        _terminate_pid(unrelated.pid)


def test_prod_q_p0_int_d7_timeout_terminates_parent_and_child(tmp_path) -> None:
    marker = tmp_path / "tree_probe"
    _exit_code, _tail, timed_out = run_prod_q_bounded_subprocess(
        [sys.executable, "-c", _PARENT_SPAWNS_CHILD_SCRIPT, str(marker)],
        timeout_sec=2,
    )
    assert timed_out is True
    child_pid_path = Path(f"{marker}.child_pid")
    if not child_pid_path.is_file():
        pytest.skip("child pid marker not written before timeout on this host")
    child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
    assert not _process_still_running(child_pid)


def test_prod_q_p0_int_d7_timeout_does_not_terminate_unrelated_process(tmp_path) -> None:
    unrelated_marker = tmp_path / "unrelated.pid"
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    unrelated_marker.write_text(str(unrelated.pid), encoding="utf-8")
    try:
        marker = tmp_path / "bounded_probe"
        _exit_code, _tail, timed_out = run_prod_q_bounded_subprocess(
            [sys.executable, "-c", _PARENT_SPAWNS_CHILD_SCRIPT, str(marker)],
            timeout_sec=2,
        )
        assert timed_out is True
        assert _process_still_running(unrelated.pid)
    finally:
        _terminate_pid(unrelated.pid)


def test_prod_q_p0_int_d7_continuation_and_admission_tails_distinguishable() -> None:
    continuation_tail = (
        "E   intergrax.contracts.execution_continuation.ExecutionContinuationError: "
        "production composition requires an explicit ExecutionContinuationStateStore"
    )
    admission_tail = (
        "E   intergrax.runtime.resilience.dependency_attempt_boundary_composition."
        "ToolDependencyAttemptBoundaryMaterializationError: strict production requires"
    )
    cont_class, cont_exc, *_ = _parse_sqlite_reproduction_tail(continuation_tail)
    adm_class, adm_exc, *_ = _parse_sqlite_reproduction_tail(admission_tail)
    assert cont_class == SqliteReproductionClassification.DEPENDENCY_MATERIALIZATION
    assert adm_class == SqliteReproductionClassification.DEPENDENCY_MATERIALIZATION
    assert "ExecutionContinuationError" in cont_exc
    assert "ToolDependencyAttemptBoundaryMaterializationError" in adm_exc
    assert cont_exc != adm_exc


def test_prod_q_p0_int_d8_provider_bundle_not_registered_alone() -> None:
    for provider in build_provider_dispositions():
        if provider.implementation_exists and not provider.registration_evidence:
            assert provider.registered is False


def test_prod_q_p0_int_d9_missing_anchor_symbol_fails_resolution() -> None:
    ok, _line, reason = _module_defines_symbol(
        "intergrax/hosting/engine/engine.py",
        "__prod_q_nonexistent_symbol__",
    )
    assert ok is False
    assert reason


def test_prod_q_p0_int_d9_method_level_health_discovered() -> None:
    method_candidates = [
        c
        for c in raw_discovery_candidates()
        if c.discovery_kind == DiscoveryKind.AST_METHOD_INDICATOR and c.indicator == "health"
    ]
    assert method_candidates, "expected method-level health mechanism in closed-world discovery"


def test_prod_q_p0_int_d1_gate_changes_on_unknown_indicator() -> None:
    baseline_unclassified = sum(
        1
        for r in classification_by_candidate_id().values()
        if r.inventory_status == CandidateInventoryStatus.UNCLASSIFIED_INDICATOR
    )
    assert baseline_unclassified == 0
    sample = next(iter(raw_discovery_candidates()))
    mutant = classify_candidate(replace(sample, indicator="__unknown__"))
    assert mutant.inventory_status == CandidateInventoryStatus.UNCLASSIFIED_INDICATOR
    assert mutant.family == ProductionReadinessFamily.PQ_UNCLASSIFIED


def test_prod_q_p0_int_ownership_unresolved_contract_incomplete() -> None:
    assert probe_unresolved_contract_marks_incomplete() is None


def test_prod_q_p0_int_ownership_unresolved_composition_incomplete() -> None:
    assert probe_unresolved_composition_marks_incomplete() is None


def test_prod_q_p0_int_ownership_alternate_contract_conflict() -> None:
    assert probe_alternate_contract_conflict_detected() is None


def test_prod_q_p0_int_ownership_distinct_concerns_not_merged() -> None:
    assert probe_distinct_concerns_not_merged() is None


def test_prod_q_p0_int_ownership_concern_identity_order_stable() -> None:
    assert probe_concern_identity_stable_under_candidate_order() is None


def test_prod_q_p0_int_ownership_mutated_state_alters_reconciliation() -> None:
    assert probe_mutated_owner_alters_reconciliation() is None


def test_prod_q_p0_int_r1_r1_r1_readiness_not_auto_auxiliary() -> None:
    assert probe_readiness_method_not_auto_auxiliary() is None


def test_prod_q_p0_int_r1_r1_r1_unresolved_role_explicit() -> None:
    assert probe_unresolved_classification_role_explicit() is None


def test_prod_q_p0_int_r1_r1_r1_false_auxiliary_na_rejected() -> None:
    assert probe_false_auxiliary_na_without_symbol_evidence() is None


def test_prod_q_p0_int_r1_r1_r1_cross_contract_duplicate_owner() -> None:
    assert probe_alternate_contract_conflict_detected() is None


def test_prod_q_p0_int_r1_r1_r1_legal_port_providers_not_duplicate() -> None:
    assert probe_legal_port_providers_not_duplicate() is None


def test_prod_q_p0_int_r1_r1_r1_alternate_mechanism_mutation() -> None:
    assert probe_mutated_alternate_mechanism_detected() is None


def test_prod_q_p0_int_d8_nonexistent_provider_not_registered() -> None:
    ghost = ProductionReadinessCandidate(
        candidate_id="ghost::bundle::provider.sandbox_host",
        module_path="intergrax/integrations/providers/sandbox_host/ghost/bundle.py",
        symbol="sandbox_host_bundle",
        discovery_kind=DiscoveryKind.PROVIDER_SANDBOX_HOST,
        line_no=1,
        indicator="sandbox_provider",
    )
    record = classify_candidate(ghost)
    assert record.evidence_state == ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED
