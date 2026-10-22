# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P4 consumers — reuse PROD-Q, PHYSQ preflight, and pytest integrity owners."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from intergrax.integrations.providers.relational_store.sqlite.config import (
    SQLiteIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    resolve_sqlite_store_paths,
)
from tests.integration.runtime.sandbox.reference_substrate.preflight import (
    ReferenceSubstratePreflight,
)
from tests.qualification.governance.pytest_node_integrity import PytestExecuteSummary
from tests.qualification.prod_q._p0_certification import (
    classify_sqlite_header_bytes,
    probe_sqlite_header_at_path,
)
from tests.qualification.prod_q._p0_types import (
    ProductionReadinessEvidenceState,
    ProdQFindingDisposition,
    R1SqliteEnv01Disposition,
    SqliteReproductionClassification,
)


class QualXP4Outcome(StrEnum):
    PASS = "PASS"
    TEST_FAILURE = "TEST_FAILURE"
    ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED = "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED"
    NOT_EXECUTED = "NOT_EXECUTED"
    INVALID_QUALIFICATION = "INVALID_QUALIFICATION"


def mandatory_qualification_allows_pass(outcome: QualXP4Outcome) -> bool:
    return outcome is QualXP4Outcome.PASS


def mandatory_outcome_from_r1_sqlite_disposition(
    disposition: R1SqliteEnv01Disposition,
) -> QualXP4Outcome:
    """Map existing PROD-Q R1 disposition — no parallel decision authority."""
    if disposition.finding_disposition is ProdQFindingDisposition.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED:
        return QualXP4Outcome.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED
    if disposition.evidence_state is ProductionReadinessEvidenceState.BLOCKER:
        if not disposition.reproduction_executed:
            return QualXP4Outcome.NOT_EXECUTED
        if disposition.reproduction_classification in (
            SqliteReproductionClassification.ENVIRONMENT_UNAVAILABLE,
            SqliteReproductionClassification.TEST_ISOLATION_OR_HARNESS,
        ):
            return QualXP4Outcome.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED
        if disposition.reproduction_classification is SqliteReproductionClassification.PRODUCTION_DEFECT:
            return QualXP4Outcome.TEST_FAILURE
        return QualXP4Outcome.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED
    if (
        disposition.reproduction_executed
        and disposition.reproduction_classification
        is SqliteReproductionClassification.DEPENDENCY_MATERIALIZATION
        and disposition.evidence_state is ProductionReadinessEvidenceState.PROVEN_CURRENT_HEAD
    ):
        return QualXP4Outcome.PASS
    return QualXP4Outcome.INVALID_QUALIFICATION


def prerequisite_outcome_from_reference_substrate_preflight(
    preflight: ReferenceSubstratePreflight,
) -> QualXP4Outcome:
    if preflight.ok:
        return QualXP4Outcome.PASS
    return QualXP4Outcome.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED


def mandatory_outcome_from_pytest_execute_summary(
    summary: PytestExecuteSummary,
    *,
    expected_passed: int,
    mandatory_skip_forbidden: bool = True,
) -> QualXP4Outcome:
    if summary.errors > 0:
        return QualXP4Outcome.INVALID_QUALIFICATION
    if summary.failed > 0:
        return QualXP4Outcome.TEST_FAILURE
    if mandatory_skip_forbidden and summary.skipped > 0:
        return QualXP4Outcome.INVALID_QUALIFICATION
    if summary.xfailed > 0:
        return QualXP4Outcome.INVALID_QUALIFICATION
    if summary.passed < expected_passed:
        return QualXP4Outcome.NOT_EXECUTED
    if summary.passed == expected_passed:
        return QualXP4Outcome.PASS
    return QualXP4Outcome.INVALID_QUALIFICATION


def default_relational_path_for_data_dir(data_dir: Path) -> Path:
    paths = resolve_sqlite_store_paths(SQLiteIntegrationConfig(data_dir=data_dir))
    return paths.relational


def classify_relational_bootstrap_test_environment(
    *,
    relational_path: Path,
    default_relational_path: Path,
    env_value: str | None,
) -> QualXP4Outcome:
    if env_value:
        if relational_path.resolve() == default_relational_path.resolve():
            return QualXP4Outcome.INVALID_QUALIFICATION
        return QualXP4Outcome.PASS
    probe = probe_sqlite_header_at_path(default_relational_path)
    if probe not in ("MISSING", "SQLITE_HEADER"):
        return QualXP4Outcome.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED
    return QualXP4Outcome.NOT_EXECUTED


@dataclass(frozen=True, slots=True)
class R1DispositionSnapshot:
    finding_id: str
    evidence_state: str
    reproduction_executed: bool
    reproduction_classification: str
    default_db_header_probe: str
    finding_disposition: str


def snapshot_r1_sqlite_disposition(
    disposition: R1SqliteEnv01Disposition,
) -> R1DispositionSnapshot:
    return R1DispositionSnapshot(
        finding_id=disposition.finding_id,
        evidence_state=disposition.evidence_state.value,
        reproduction_executed=disposition.reproduction_executed,
        reproduction_classification=disposition.reproduction_classification.value,
        default_db_header_probe=disposition.default_db_header_probe,
        finding_disposition=disposition.finding_disposition.value,
    )


def sqlite_header_classification_stable_probe(payload: bytes) -> str:
    return classify_sqlite_header_bytes(payload)
