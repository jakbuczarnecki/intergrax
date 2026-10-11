# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-P0 persisted boundary inventory & migration ownership gates."""

from __future__ import annotations

import pytest

from tests.qualification.compat_x._compat_x_persisted_discovery import (
    mechanical_closed_world_family_ids,
    persisted_closed_world_parity_holds,
)
from tests.qualification.compat_x._compat_x_persisted_gates import (
    build_persisted_r2_p0_qualification_report,
    persisted_r2_p0_inventory_gate_passes,
)
from tests.qualification.compat_x._compat_x_persisted_historical_five import (
    HistoricalFindingAssessment,
    historical_persisted_five_reconciliation,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_remediation import persisted_r2_remediation_waves
from tests.qualification.compat_x._compat_x_persisted_synthetic import (
    synthetic_envelope_inheritance_proven,
    synthetic_implicit_fallback_allowed,
    synthetic_service_false_positive_classification,
)
from tests.qualification.compat_x._compat_x_persisted_types import (
    MigrationDisposition,
    PersistedVersionDisposition,
)
from tests.qualification.compat_x._compat_x_versioning_gates import build_r1_qualification_report

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_r2_p0_closed_world_inventory_gate() -> None:
    assert persisted_closed_world_parity_holds()
    assert persisted_r2_p0_inventory_gate_passes()
    report = build_persisted_r2_p0_qualification_report()
    assert report.metrics.unknown_unclassified_count == 0
    assert report.metrics.duplicate_version_authority_count == 0
    assert report.metrics.duplicate_migration_authority_count == 0
    assert len(PERSISTED_SCHEMA_FAMILIES) == len(mechanical_closed_world_family_ids())


def test_cx_r2_p0_r1_continuity() -> None:
    r1 = build_r1_qualification_report()
    assert r1.public_stable_noncompliant_version_identity_count == 0
    assert r1.unclassified_obligation_count == 0
    assert r1.persisted_without_version_finding_count == 5


def test_cx_r2_p0_historical_five_reconciliation() -> None:
    rows = historical_persisted_five_reconciliation()
    assert len(rows) == 5
    by_class = {r.finding_class: r for r in rows}
    assert (
        by_class["AgentPackageContractAuthorityService"].assessment
        == HistoricalFindingAssessment.FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR
    )
    assert (
        by_class["PackageAgentContractAuthorityError"].assessment
        == HistoricalFindingAssessment.FALSE_POSITIVE_EXCEPTION
    )
    assert (
        by_class["RuntimeRevisionService"].assessment
        == HistoricalFindingAssessment.FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR
    )
    assert (
        by_class["DocumentStoreSuspendedExecutionOperationStore"].assessment
        == HistoricalFindingAssessment.REAL_DURABLE_BOUNDARY_STORE
    )
    assert (
        by_class["RuntimeEventPublisher"].assessment
        == HistoricalFindingAssessment.FALSE_POSITIVE_PUBLISHER_PROTOCOL
    )


def test_cx_r2_p0_adversarial_a_service_false_positive() -> None:
    probe = synthetic_service_false_positive_classification()
    assert probe.payload_disposition == PersistedVersionDisposition.EXPLICIT_PAYLOAD_VERSION
    assert probe.orchestrator_disposition == PersistedVersionDisposition.NOT_A_PERSISTED_SCHEMA


def test_cx_r2_p0_adversarial_b_versioned_round_trip_family_present() -> None:
    family = next(
        f for f in PERSISTED_SCHEMA_FAMILIES if f.family_id == "agent_distribution.runtime_revision"
    )
    assert family.current_version == "runtime_revision.v1"
    assert family.writer_paths and family.reader_paths


def test_cx_r2_p0_adversarial_c_unversioned_durable_payload_flagged() -> None:
    missing = [
        f
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.version_disposition == PersistedVersionDisposition.MISSING_VERSION
    ]
    assert any(f.family_id == "agent_distribution.agent_artifact_metadata" for f in missing)


def test_cx_r2_p0_adversarial_d_unknown_version_fail_closed_runtime_revision() -> None:
    family = next(
        f for f in PERSISTED_SCHEMA_FAMILIES if f.family_id == "agent_distribution.runtime_revision"
    )
    assert family.migration_disposition == MigrationDisposition.CURRENT_ONLY_FAIL_CLOSED


def test_cx_r2_p0_adversarial_e_explicit_migration_owner_autonomous_work() -> None:
    family = next(
        f for f in PERSISTED_SCHEMA_FAMILIES if f.family_id == "autonomous_work.postgresql_container"
    )
    assert family.migration_disposition == MigrationDisposition.EXPLICIT_MIGRATION
    assert family.migration_owner_path is not None


def test_cx_r2_p0_adversarial_f_implicit_fallback_rejected() -> None:
    assert synthetic_implicit_fallback_allowed() is False


def test_cx_r2_p0_adversarial_g_no_duplicate_migration_authority() -> None:
    report = build_persisted_r2_p0_qualification_report()
    assert report.metrics.duplicate_migration_authority_count == 0


def test_cx_r2_p0_adversarial_h_container_not_payload_version() -> None:
    container = next(
        f for f in PERSISTED_SCHEMA_FAMILIES if f.family_id == "agent_distribution.sqlite_container"
    )
    payload = next(
        f
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.family_id == "agent_distribution.agent_artifact_metadata"
    )
    assert container.version_disposition == PersistedVersionDisposition.STORE_CONTAINER_VERSION
    assert payload.version_disposition == PersistedVersionDisposition.MISSING_VERSION


def test_cx_r2_p0_adversarial_i_envelope_inheritance() -> None:
    assert synthetic_envelope_inheritance_proven() is True
    inner = next(
        f
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.family_id == "runtime.suspended_execution_operation_descriptor"
    )
    assert inner.version_disposition == PersistedVersionDisposition.INHERITED_ENVELOPE_VERSION


def test_cx_r2_p0_adversarial_j_historical_not_payload_identity() -> None:
    report = build_persisted_r2_p0_qualification_report()
    assert report.service_misclassified_as_payload == 0


def test_cx_r2_p0_remediation_waves_non_empty_r2_r1() -> None:
    waves = persisted_r2_remediation_waves()
    assert waves[0].wave_id == "COMPAT-X-R2-R1"
    assert len(waves[0].items) >= 1
    assert any(
        item.family_id == "agent_distribution.package_agent_contract_authority_record"
        for item in waves[0].items
    )
