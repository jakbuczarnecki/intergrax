# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P4 — determinism, environment classification & reproducibility (Q09, Q10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations.providers.relational_store.sqlite.config import (
    SQLiteIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    resolve_sqlite_store_paths,
)
from tests.integration.runtime.sandbox.reference_substrate.backend import (
    build_reference_backend_or_skip_reason,
)
from tests.integration.runtime.sandbox.reference_substrate.preflight import (
    ReferenceSubstratePreflight,
    evaluate_reference_substrate_preflight,
)
from tests.qualification.governance.pytest_node_integrity import (
    PytestExecuteSummary,
    pytest_nodes_missing_from_collection,
)
from tests.qualification.prod_q._p0_certification import build_r1_sqlite_disposition
from tests.qualification.prod_q._p0_types import ProductionReadinessEvidenceState
from tests.qualification.qual_x._p4_determinism_environment_catalog import (
    QUAL_X_P4_CLEAN_CHECKOUT_PROCEDURE,
    QUAL_X_P4_IN_SCOPE_ENVIRONMENT_OWNERS,
    QUAL_X_P4_START_HEAD,
    QUAL_X_P4_W2_C_DETERMINISM_DEBT,
    QualXP4DeterminismDisposition,
    qual_x_p4_all_gate_pytest_node_ids,
)
from tests.qualification.qual_x._p4_determinism_environment_support import (
    QualXP4Outcome,
    classify_relational_bootstrap_test_environment,
    default_relational_path_for_data_dir,
    mandatory_outcome_from_pytest_execute_summary,
    mandatory_outcome_from_r1_sqlite_disposition,
    mandatory_qualification_allows_pass,
    prerequisite_outcome_from_reference_substrate_preflight,
    snapshot_r1_sqlite_disposition,
    sqlite_header_classification_stable_probe,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_qual_x_p4_start_head_documented() -> None:
    assert QUAL_X_P4_START_HEAD == "d75fe6354e02d9b6fd7239fa463fadf60d5f286f"


def test_qual_x_p4_in_scope_environment_owners_are_unique() -> None:
    owner_ids = [row.owner_id for row in QUAL_X_P4_IN_SCOPE_ENVIRONMENT_OWNERS]
    assert len(owner_ids) == len(frozenset(owner_ids))


def test_qual_x_p4_owner_ssot_modules_exist() -> None:
    missing: list[str] = []
    for owner in QUAL_X_P4_IN_SCOPE_ENVIRONMENT_OWNERS:
        if not (_REPO_ROOT / owner.ssot_module).is_file():
            missing.append(owner.ssot_module)
        if not (_REPO_ROOT / owner.primary_gate_module).is_file():
            missing.append(owner.primary_gate_module)
    assert missing == []


def test_qual_x_p4_r1_sqlite_mandatory_gate_cannot_pass_while_blocker() -> None:
    build_r1_sqlite_disposition.cache_clear()
    disposition = build_r1_sqlite_disposition()
    outcome = mandatory_outcome_from_r1_sqlite_disposition(disposition)
    assert disposition.evidence_state is ProductionReadinessEvidenceState.BLOCKER
    assert not mandatory_qualification_allows_pass(outcome)


def test_qual_x_p4_r1_sqlite_missing_http_reproduction_is_not_executed() -> None:
    build_r1_sqlite_disposition.cache_clear()
    disposition = build_r1_sqlite_disposition()
    assert disposition.reproduction_executed is False
    assert mandatory_outcome_from_r1_sqlite_disposition(disposition) is QualXP4Outcome.NOT_EXECUTED


def test_qual_x_p4_relational_env_isolates_path_from_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    default_path = default_relational_path_for_data_dir(data_dir)
    isolated = tmp_path / "isolated-relational.db"
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(isolated))
    paths = resolve_sqlite_store_paths(SQLiteIntegrationConfig(data_dir=data_dir))
    assert paths.relational == isolated
    assert paths.relational != default_path
    outcome = classify_relational_bootstrap_test_environment(
        relational_path=paths.relational,
        default_relational_path=default_path,
        env_value=str(isolated),
    )
    assert outcome is QualXP4Outcome.PASS


def test_qual_x_p4_relational_missing_env_on_polluted_default_not_pass(
    tmp_path: Path,
) -> None:
    polluted = tmp_path / "intergrax.db"
    polluted.write_bytes(b"%3|not-sqlite|FAIL|artifact")
    outcome = classify_relational_bootstrap_test_environment(
        relational_path=tmp_path / "unused.db",
        default_relational_path=polluted,
        env_value=None,
    )
    assert outcome is QualXP4Outcome.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED


def test_qual_x_p4_relational_forbidden_fallback_to_default_when_env_set(
    tmp_path: Path,
) -> None:
    default_path = tmp_path / "intergrax.db"
    outcome = classify_relational_bootstrap_test_environment(
        relational_path=default_path,
        default_relational_path=default_path,
        env_value=str(default_path),
    )
    assert outcome is QualXP4Outcome.INVALID_QUALIFICATION


def test_qual_x_p4_phys_preflight_non_ok_is_environment_issue() -> None:
    blocked = ReferenceSubstratePreflight(
        ok=False,
        block_reason="synthetic missing privilege",
        wsl_detected=False,
        linux_kernel="",
        ip_available=False,
        netns_usable=False,
        firewall_backend=None,
        privileged=False,
        python_available=False,
    )
    outcome = prerequisite_outcome_from_reference_substrate_preflight(blocked)
    assert outcome is QualXP4Outcome.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED
    assert not mandatory_qualification_allows_pass(outcome)


def test_qual_x_p4_phys_preflight_ok_prerequisite_only_not_full_phys_pass() -> None:
    ready = ReferenceSubstratePreflight(
        ok=True,
        block_reason=None,
        wsl_detected=True,
        linux_kernel="6.1.0",
        ip_available=True,
        netns_usable=True,
        firewall_backend="nftables",
        privileged=True,
        python_available=True,
    )
    assert prerequisite_outcome_from_reference_substrate_preflight(ready) is QualXP4Outcome.PASS


def test_qual_x_p4_backend_builder_returns_skip_reason_when_preflight_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = ReferenceSubstratePreflight(
        ok=False,
        block_reason="synthetic block",
        wsl_detected=False,
        linux_kernel="",
        ip_available=False,
        netns_usable=False,
        firewall_backend=None,
        privileged=False,
        python_available=False,
    )

    def _blocked_preflight() -> ReferenceSubstratePreflight:
        return blocked

    monkeypatch.setattr(
        "tests.integration.runtime.sandbox.reference_substrate.backend.evaluate_reference_substrate_preflight",
        _blocked_preflight,
    )
    result = build_reference_backend_or_skip_reason()
    assert isinstance(result, str)
    assert "synthetic block" in result


def test_qual_x_p4_pytest_mandatory_summary_skip_is_invalid_qualification() -> None:
    summary = PytestExecuteSummary(passed=1, skipped=1)
    outcome = mandatory_outcome_from_pytest_execute_summary(summary, expected_passed=2)
    assert outcome is QualXP4Outcome.INVALID_QUALIFICATION


def test_qual_x_p4_pytest_mandatory_summary_failure_is_test_failure() -> None:
    summary = PytestExecuteSummary(passed=1, failed=1)
    outcome = mandatory_outcome_from_pytest_execute_summary(summary, expected_passed=2)
    assert outcome is QualXP4Outcome.TEST_FAILURE


def test_qual_x_p4_pytest_mandatory_summary_all_passed_is_pass() -> None:
    summary = PytestExecuteSummary(passed=3)
    outcome = mandatory_outcome_from_pytest_execute_summary(summary, expected_passed=3)
    assert outcome is QualXP4Outcome.PASS


def test_qual_x_p4_p4_gate_nodes_are_collectable() -> None:
    node_ids = qual_x_p4_all_gate_pytest_node_ids()
    missing, proc = pytest_nodes_missing_from_collection(node_ids, _REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert missing == frozenset()


def test_qual_x_p4_w2_c_disposition_tracked_not_p4_corrected() -> None:
    debt = QUAL_X_P4_W2_C_DETERMINISM_DEBT
    assert debt.disposition is QualXP4DeterminismDisposition.TRACKED_FREEZE_DEBT
    assert debt.p4_scope_correction is False
    assert (_REPO_ROOT / debt.pytest_node_id.split("::", 1)[0]).is_file()


def test_qual_x_p4_determinism_repeat_r1_disposition_snapshot_stable() -> None:
    build_r1_sqlite_disposition.cache_clear()
    first = snapshot_r1_sqlite_disposition(build_r1_sqlite_disposition())
    build_r1_sqlite_disposition.cache_clear()
    second = snapshot_r1_sqlite_disposition(build_r1_sqlite_disposition())
    assert first == second


def test_qual_x_p4_determinism_repeat_sqlite_header_classifier_stable() -> None:
    payload = b"SQLite format 3\x00"
    assert sqlite_header_classification_stable_probe(payload) == "SQLITE_HEADER"
    assert sqlite_header_classification_stable_probe(payload) == "SQLITE_HEADER"


def test_qual_x_p4_clean_checkout_procedure_declares_partial_verification() -> None:
    assert "PARTIAL" in QUAL_X_P4_CLEAN_CHECKOUT_PROCEDURE.verification_status
    assert QUAL_X_P4_CLEAN_CHECKOUT_PROCEDURE.execution_command.startswith("uv run pytest")


def test_qual_x_p4_synthetic_stale_mandatory_node_would_fail_collectability() -> None:
    stale = "tests/qualification/qual_x/synthetic_p4_deleted_gate.py::test_never_existed"
    missing, _proc = pytest_nodes_missing_from_collection(
        (*qual_x_p4_all_gate_pytest_node_ids(), stale),
        _REPO_ROOT,
    )
    assert stale in missing


def test_qual_x_p4_live_preflight_contract_matches_platform() -> None:
    preflight = evaluate_reference_substrate_preflight()
    outcome = prerequisite_outcome_from_reference_substrate_preflight(preflight)
    if preflight.ok:
        assert outcome is QualXP4Outcome.PASS
    else:
        assert outcome is QualXP4Outcome.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED
