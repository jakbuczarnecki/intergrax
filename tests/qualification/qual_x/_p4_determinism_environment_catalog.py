# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P4 — determinism, environment classification & reproducibility (Q09, Q10)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

QUAL_X_P4_START_HEAD: str = "d75fe6354e02d9b6fd7239fa463fadf60d5f286f"

_P4_TEST = "tests/qualification/qual_x/test_qual_x_p4_determinism_environment_classification_reproducibility.py"
_PROD_Q_R1_R3 = (
    "tests/unit/applications/governed_contractor/test_prod_q_r1_r3_startup_sqlite_isolation.py"
)
_PROD_Q_P0_FAST = "tests/qualification/prod_q/test_prod_q_p0_fast.py"
_PHYSQ_PREFLIGHT = (
    "tests/integration/runtime/sandbox/reference_substrate/"
    "test_reference_physical_egress_qualification.py"
)
_W2_C = "tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py"


def _nid(module: str, test_name: str) -> str:
    return f"{module}::{test_name}"


class QualXP4DeterminismDisposition(StrEnum):
    COLLECTABLE_DETERMINISTIC_STATIC = "COLLECTABLE_DETERMINISTIC_STATIC"
    TRACKED_FREEZE_DEBT = "TRACKED_FREEZE_DEBT"
    HISTORICAL_BATCH_ONLY = "HISTORICAL_BATCH_ONLY"


@dataclass(frozen=True, slots=True)
class QualXP4EnvironmentOwner:
    owner_id: str
    semantic_owner: str
    ssot_module: str
    primary_gate_module: str
    reconciliation_class: str
    frz_criteria: tuple[str, ...]


QUAL_X_P4_IN_SCOPE_ENVIRONMENT_OWNERS: tuple[QualXP4EnvironmentOwner, ...] = (
    QualXP4EnvironmentOwner(
        owner_id="PROD-Q-R1-SQLITE",
        semantic_owner="R1-SQLITE-ENV-01 disposition & isolation evidence",
        ssot_module="tests/qualification/prod_q/_p0_certification.py",
        primary_gate_module=_PROD_Q_P0_FAST,
        reconciliation_class="typed_disposition_fail_closed_blocker",
        frz_criteria=("FRZ-REG-08", "FRZ-PRD-04"),
    ),
    QualXP4EnvironmentOwner(
        owner_id="RELATIONAL-SQLITE-PATHS",
        semantic_owner="INTERGRAX_RELATIONAL_DB path resolution",
        ssot_module="intergrax/integrations/providers/relational_store/sqlite/paths.py",
        primary_gate_module=_PROD_Q_R1_R3,
        reconciliation_class="env_var_isolation_before_bootstrap",
        frz_criteria=("FRZ-REG-08",),
    ),
    QualXP4EnvironmentOwner(
        owner_id="AW-7C-PHYSQ-PREFLIGHT",
        semantic_owner="Reference substrate physical qualification preflight",
        ssot_module=(
            "tests/integration/runtime/sandbox/reference_substrate/preflight.py"
        ),
        primary_gate_module=_PHYSQ_PREFLIGHT,
        reconciliation_class="evaluate_preflight_fail_closed_before_backend",
        frz_criteria=("FRZ-REG-08", "FRZ-REG-06"),
    ),
    QualXP4EnvironmentOwner(
        owner_id="QUAL-X-PYTEST-INTEGRITY",
        semantic_owner="Mandatory gate collection / execute summary parsing",
        ssot_module="tests/qualification/governance/pytest_node_integrity.py",
        primary_gate_module=_P4_TEST,
        reconciliation_class="collect_only_and_execute_summary_truthfulness",
        frz_criteria=("FRZ-REG-07", "FRZ-REG-08", "FRZ-REG-06"),
    ),
)


@dataclass(frozen=True, slots=True)
class QualXP4W2CDebtRecord:
    debt_id: str
    pytest_node_id: str
    checklist_authority: str
    disposition: QualXP4DeterminismDisposition
    p4_scope_correction: bool
    summary: str


QUAL_X_P4_W2_C_DETERMINISM_DEBT: QualXP4W2CDebtRecord = QualXP4W2CDebtRecord(
    debt_id="W2-C-RETRY-STORM-THREAD-WARNING",
    pytest_node_id=_nid(_W2_C, "test_retry_storm_caps_provider_calls"),
    checklist_authority="PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST FRZ-REG-06",
    disposition=QualXP4DeterminismDisposition.TRACKED_FREEZE_DEBT,
    p4_scope_correction=False,
    summary=(
        "PytestUnhandledThreadExceptionWarning after rate-limit rejection; "
        "bounded-call assertion PASS — hygiene debt, not P4 implementation scope"
    ),
)


@dataclass(frozen=True, slots=True)
class QualXP4CleanCheckoutStep:
    order: int
    instruction: str


@dataclass(frozen=True, slots=True)
class QualXP4CleanCheckoutProcedure:
    """Local-only P4 verification recipe — not full FRZ-REG-07 platform certification."""

    code_revision_command: str
    required_env_defaults: tuple[tuple[str, str], ...]
    prohibited_fallbacks: tuple[str, ...]
    collection_command: str
    execution_command: str
    repeatability_command: str
    evidence_to_retain: tuple[str, ...]
    verification_status: str
    steps: tuple[QualXP4CleanCheckoutStep, ...]


QUAL_X_P4_CLEAN_CHECKOUT_PROCEDURE: QualXP4CleanCheckoutProcedure = QualXP4CleanCheckoutProcedure(
    code_revision_command="git rev-parse HEAD",
    required_env_defaults=(
        ("INTERGRAX_PROD_Q_R1_HTTP_REPRODUCTION", "unset (fast disposition; NOT full R1 HTTP replay)"),
        ("INTERGRAX_RELATIONAL_DB", "unset for P4 batch; set to tmp path for relational bootstrap tests"),
    ),
    prohibited_fallbacks=(
        "Implicit use of polluted build/intergrax.db when INTERGRAX_RELATIONAL_DB is required",
        "Treating pytest skip/xfail on mandatory P4 nodes as architecture PASS",
        "Representing AW-7C PHYSQ PASS without Linux privileged preflight",
    ),
    collection_command=(
        "uv run pytest -p no:xdist --collect-only -q "
        "tests/qualification/qual_x/test_qual_x_p4_determinism_environment_classification_reproducibility.py"
    ),
    execution_command=(
        "uv run pytest -p no:xdist "
        "tests/qualification/qual_x/test_qual_x_p4_determinism_environment_classification_reproducibility.py"
    ),
    repeatability_command=(
        "uv run pytest -p no:xdist "
        "tests/qualification/qual_x/test_qual_x_p4_determinism_environment_classification_reproducibility.py "
        "-k determinism_repeat"
    ),
    evidence_to_retain=(
        "git rev-parse HEAD",
        "pytest stdout/stderr with pass/fail/skip counts",
        "R1-SQLITE disposition record fields from P4 gates (no secrets)",
    ),
    verification_status=(
        "PARTIAL — P4 batch demonstrated on implementation worktree; "
        "not independent clean-checkout FRZ-REG-07 closure"
    ),
    steps=(
        QualXP4CleanCheckoutStep(1, "git fetch origin development && git rev-parse HEAD"),
        QualXP4CleanCheckoutStep(2, "uv sync (existing lockfile; no Docker)"),
        QualXP4CleanCheckoutStep(3, "Run collection_command; expect all P4 nodes collected"),
        QualXP4CleanCheckoutStep(4, "Run execution_command; expect exit 0"),
        QualXP4CleanCheckoutStep(5, "Optional: repeatability_command for determinism_repeat subset"),
        QualXP4CleanCheckoutStep(
            6,
            "Archive logs under .tmp/session/QUAL-X-P4-<HEAD>/run.log",
        ),
    ),
)


QUAL_X_P4_DETERMINISM_REPEAT_NODE_IDS: tuple[str, ...] = (
    _nid(_P4_TEST, "test_qual_x_p4_determinism_repeat_r1_disposition_snapshot_stable"),
    _nid(_P4_TEST, "test_qual_x_p4_determinism_repeat_sqlite_header_classifier_stable"),
)


def qual_x_p4_all_gate_pytest_node_ids() -> tuple[str, ...]:
    return (
        _nid(_P4_TEST, "test_qual_x_p4_start_head_documented"),
        _nid(_P4_TEST, "test_qual_x_p4_in_scope_environment_owners_are_unique"),
        _nid(_P4_TEST, "test_qual_x_p4_owner_ssot_modules_exist"),
        _nid(_P4_TEST, "test_qual_x_p4_r1_sqlite_mandatory_gate_cannot_pass_while_blocker"),
        _nid(_P4_TEST, "test_qual_x_p4_r1_sqlite_missing_http_reproduction_is_not_executed"),
        _nid(_P4_TEST, "test_qual_x_p4_relational_env_isolates_path_from_default"),
        _nid(_P4_TEST, "test_qual_x_p4_relational_missing_env_on_polluted_default_not_pass"),
        _nid(_P4_TEST, "test_qual_x_p4_relational_forbidden_fallback_to_default_when_env_set"),
        _nid(_P4_TEST, "test_qual_x_p4_phys_preflight_non_ok_is_environment_issue"),
        _nid(_P4_TEST, "test_qual_x_p4_phys_preflight_ok_prerequisite_only_not_full_phys_pass"),
        _nid(_P4_TEST, "test_qual_x_p4_backend_builder_returns_skip_reason_when_preflight_blocks"),
        _nid(_P4_TEST, "test_qual_x_p4_pytest_mandatory_summary_skip_is_invalid_qualification"),
        _nid(_P4_TEST, "test_qual_x_p4_pytest_mandatory_summary_failure_is_test_failure"),
        _nid(_P4_TEST, "test_qual_x_p4_pytest_mandatory_summary_all_passed_is_pass"),
        _nid(_P4_TEST, "test_qual_x_p4_p4_gate_nodes_are_collectable"),
        _nid(_P4_TEST, "test_qual_x_p4_w2_c_disposition_tracked_not_p4_corrected"),
        _nid(_P4_TEST, "test_qual_x_p4_determinism_repeat_r1_disposition_snapshot_stable"),
        _nid(_P4_TEST, "test_qual_x_p4_determinism_repeat_sqlite_header_classifier_stable"),
        _nid(_P4_TEST, "test_qual_x_p4_synthetic_stale_mandatory_node_would_fail_collectability"),
        _nid(_P4_TEST, "test_qual_x_p4_clean_checkout_procedure_declares_partial_verification"),
        _nid(_P4_TEST, "test_qual_x_p4_live_preflight_contract_matches_platform"),
    )
