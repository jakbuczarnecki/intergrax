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
from tests.qualification.compat_x._compat_x_persisted_contract_authority import (
    build_contract_authority_durability_evidence,
)
from tests.qualification.compat_x._compat_x_persisted_discovery import (
    build_persisted_mechanical_discovery_report,
)
from tests.qualification.compat_x._compat_x_persisted_classifier import classify_durable_candidate
from tests.qualification.compat_x._compat_x_persisted_mechanical_scan import (
    iter_production_python_modules,
)
from tests.qualification.compat_x._compat_x_persisted_synthetic import (
    synthetic_cache_name_not_false_excluded,
    synthetic_document_store_unknown_fails,
    synthetic_durable_file_state_detected,
    synthetic_durable_state_file_not_cache_excluded,
    synthetic_envelope_inheritance_proven,
    synthetic_evidence_export_excluded,
    synthetic_exception_near_persistence_not_payload_family,
    synthetic_implicit_fallback_allowed,
    synthetic_pickle_to_redis_detected,
    synthetic_pickle_to_sqlite_detected,
    synthetic_redis_classified_family_passes,
    synthetic_redis_unknown_family_fails_closed_world,
    synthetic_serialization_only_excluded,
    synthetic_service_false_positive_classification,
    synthetic_unknown_sqlite_module_fails_closed_world,
    synthetic_unknown_sql_table_fails,
)
from tests.qualification.compat_x._compat_x_persisted_types import DurableExclusionReason
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


def test_cx_r2_p0_adversarial_a_unknown_sqlite_fails_closed_world() -> None:
    assert synthetic_unknown_sqlite_module_fails_closed_world()


def test_cx_r2_p0_adversarial_a2_service_false_positive() -> None:
    probe = synthetic_service_false_positive_classification()
    assert probe.orchestrator_disposition == PersistedVersionDisposition.NOT_A_PERSISTED_SCHEMA


def test_cx_r2_p0_adversarial_c_serialization_only_excluded() -> None:
    assert synthetic_serialization_only_excluded()


def test_cx_r2_p0_adversarial_f_document_store_unknown_fails() -> None:
    assert synthetic_document_store_unknown_fails()


def test_cx_r2_p0_adversarial_g_unknown_sql_table_fails() -> None:
    assert synthetic_unknown_sql_table_fails()


def test_cx_r2_p0_mechanical_discovery_closed_world_counts() -> None:
    report = build_persisted_mechanical_discovery_report()
    assert len(report.unclassified_candidate_ids) == 0
    assert report.persisted_member_count + report.exclusion_count == len(report.raw_candidates)


def test_cx_r2_p0_sqlite_contract_authority_gap() -> None:
    evidence = build_contract_authority_durability_evidence()
    assert not evidence.in_scope_blocker


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
    assert missing == []


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
    assert payload.version_disposition == PersistedVersionDisposition.EXPLICIT_PAYLOAD_VERSION


def test_cx_r2_p0_adversarial_i_envelope_inheritance() -> None:
    assert synthetic_envelope_inheritance_proven() is True
    inner = next(
        f
        for f in PERSISTED_SCHEMA_FAMILIES
        if f.family_id == "runtime.suspended_execution_operation_descriptor"
    )
    assert inner.version_disposition == PersistedVersionDisposition.INHERITED_ENVELOPE_VERSION


def test_cx_r2_p0_adversarial_e_exception_not_payload_family() -> None:
    assert synthetic_exception_near_persistence_not_payload_family()


def test_cx_r2_p0_adversarial_j_historical_not_payload_identity() -> None:
    report = build_persisted_r2_p0_qualification_report()
    assert report.service_misclassified_as_payload == 0


_REQUIRED_HEAD_MODULES: tuple[str, ...] = (
    "intergrax/runtime/execution/continuation/durable_state_file.py",
    "intergrax/runtime/tools/sqlite_idempotency_store.py",
    "intergrax/distributed/providers/redis_idempotency_store.py",
    "intergrax/distributed/providers/redis_kv_store.py",
)


def test_cx_r2_p0_r2_required_head_module_probes() -> None:
    report = build_persisted_mechanical_discovery_report()
    by_module: dict[str, list] = {m: [] for m in _REQUIRED_HEAD_MODULES}
    for c in report.raw_candidates:
        if c.module_path in by_module:
            by_module[c.module_path].append(c)
    for module_path, raw in by_module.items():
        assert raw, f"missing durable candidates for {module_path}"
        for candidate in raw:
            cls = classify_durable_candidate(candidate)
            assert cls.exclusion_reason != DurableExclusionReason.CACHE_NON_DURABLE, module_path
            assert cls.family_id is not None or cls.exclusion_reason is not None, module_path


def test_cx_r2_p0_adversarial_k_redis_unknown_fails_closed_world() -> None:
    assert synthetic_redis_unknown_family_fails_closed_world()


def test_cx_r2_p0_adversarial_l_redis_classified_passes() -> None:
    assert synthetic_redis_classified_family_passes()


def test_cx_r2_p0_adversarial_m_durable_file_detected() -> None:
    assert synthetic_durable_file_state_detected()


def test_cx_r2_p0_adversarial_n_evidence_export_excluded() -> None:
    assert synthetic_evidence_export_excluded()


def test_cx_r2_p0_adversarial_o_pickle_sqlite_detected() -> None:
    assert synthetic_pickle_to_sqlite_detected()


def test_cx_r2_p0_adversarial_p_pickle_redis_detected() -> None:
    assert synthetic_pickle_to_redis_detected()


def test_cx_r2_p0_adversarial_q_cache_name_not_false_excluded() -> None:
    assert synthetic_cache_name_not_false_excluded()


def test_cx_r2_p0_adversarial_r_durable_state_file_regression() -> None:
    assert synthetic_durable_state_file_not_cache_excluded()


def test_cx_r2_p0_r2_production_module_scan_non_zero() -> None:
    assert len(iter_production_python_modules()) > 0


def test_cx_r2_p0_remediation_waves_r2_r2_has_migration_items() -> None:
    waves = persisted_r2_remediation_waves()
    assert waves[0].wave_id == "COMPAT-X-R2-R1"
    assert waves[1].wave_id == "COMPAT-X-R2-R2"
    assert len(waves[1].items) >= 1
