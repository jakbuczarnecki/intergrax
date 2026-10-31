# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R3-R1 envelope version authority and observability provenance gates."""

from __future__ import annotations

import pytest
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import (
    RUNTIME_EVENT_SCHEMA_VERSION,
    RuntimeEvent,
    parse_runtime_event_payload,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.runtime_event import RuntimeEvent as RuntimeRuntimeEvent
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError, assert_runtime_event_schema
from intergrax.runtime.observability.export_boundary import (
    envelope_from_runtime_event,
    envelope_from_runtime_event_source,
)
from intergrax.runtime.observability.runtime_event_export_mapping import (
    runtime_event_export_source_from_event,
)
from intergrax.runtime.observability.runtime_event_export_models import (
    RuntimeEventExportSource,
)
from intergrax.runtime.schema.registry import (
    PREVIEW_RUNTIME_SCHEMA_VERSIONS,
    RUNTIME_SCHEMA_REGISTRY,
    validate_schema_version,
)
from testing_support.runtime_events import with_preferred_canonical_payload
from tests.qualification.compat_x._compat_x_closed_world import (
    build_closed_world_report,
    build_closed_world_report_with_extra_candidates,
    discover_candidates_from_source,
)
from tests.qualification.compat_x._compat_x_event_evolution_inventory import (
    count_canonical_production_raw_store_bypasses,
    discover_runtime_event_persistence_call_sites,
    event_evolution_inventory,
    event_evolution_r3_p0_gate_passes,
    synthetic_probe_k_unknown_persistence_append_gate,
    synthetic_second_evolution_authority_owner,
    validate_production_persistence_call_site_classification,
)
from tests.qualification.compat_x._compat_x_event_evolution_r3_r1 import (
    RuntimeEventReaderClassificationEvidence,
    RuntimeEventReaderDisposition,
    compute_event_evolution_r3_r1_metrics,
    count_module_wide_constructor_positive_evidence_rules,
    count_runtime_event_v2_semantic_authority_literals,
    count_stale_constructor_positive_evidence_rules,
    discover_persisted_runtime_event_read_paths,
    discover_runtime_event_constructor_sites,
    discover_runtime_event_reader_sites,
    event_evolution_r3_r1_gate_passes,
    minimal_runtime_event_payload_dict,
    observability_runtime_event_v2_literal_sites,
    production_runtime_event_v1_literal_sites,
    production_runtime_event_v2_literal_sites,
    registry_runtime_event_version_parity_violations,
    runtime_event_default_schema_version,
    synthetic_persisted_reconstruction_in_trusted_module_constructor_site,
    synthetic_persisted_runtime_event_model_validate_bypass_site,
    synthetic_registry_drift_violations,
    synthetic_runtime_event_kwargs_raw_reconstruction_site,
    synthetic_sanctioned_runtime_event_constructor_site,
    synthetic_second_v2_literal_owner,
    synthetic_stale_constructor_positive_evidence_rule,
    synthetic_unknown_persisted_reader_site,
    synthetic_unknown_qualname_in_trusted_module_constructor_site,
    synthetic_unknown_runtime_event_constructor_site,
    validate_constructor_positive_evidence_rule_integrity,
    validate_runtime_event_registry_parity,
)
from tests.qualification.compat_x._compat_x_semantic_identity_reconciliation import (
    ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES,
    ACCEPTED_P0_HISTORICAL_INVENTORY_SOURCE_SHA,
    ACCEPTED_P0_STABLE_INVENTORY_SHA256,
    R3_RUNTIME_EVENT_SCHEMA_VERSION_CANDIDATE_ID,
    REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256,
    REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES,
    RUNTIME_EVENT_MODEL_VALIDATE_CURRENT_CANDIDATE_ID,
    RUNTIME_EVENT_MODEL_VALIDATE_HISTORICAL_CANDIDATE_ID,
    RUNTIME_EVENT_MODEL_VALIDATE_STABLE_IDENTITY,
    _remainder_candidates_minus_r3,
    build_r3_runtime_event_schema_version_candidate_delta,
    build_synthetic_r3_inventory_balanced_add_remove_report,
    build_synthetic_r3_inventory_equal_count_substitution_report,
    build_synthetic_r3_inventory_hidden_removal_report,
    inventory_identity_fingerprint,
    stable_inventory_candidate_identity,
    stable_inventory_identity_set,
    synthetic_wire_persistence_lineno_only_pair,
    synthetic_wire_persistence_semantic_variant,
    validate_current_head_compat_closed_world_inventory,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def _canonical_write_event(**overrides: object) -> RuntimeEvent:
    event = RuntimeRuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
    )
    if overrides:
        event = event.model_copy(update=overrides)
    return with_preferred_canonical_payload(event)


def test_cx_r3_r1_q1_exactly_one_v2_semantic_authority() -> None:
    assert count_runtime_event_v2_semantic_authority_literals() == 1
    sites = production_runtime_event_v2_literal_sites()
    assert len(sites) == 1
    assert sites[0][0] == "intergrax/contracts/runtime_event.py"


def test_cx_r3_r1_q2_registry_parity() -> None:
    assert registry_runtime_event_version_parity_violations() == ()
    assert RUNTIME_SCHEMA_REGISTRY["runtime_event"] == RUNTIME_EVENT_SCHEMA_VERSION


def test_cx_r3_r1_q3_model_default_parity() -> None:
    assert runtime_event_default_schema_version() == RUNTIME_EVENT_SCHEMA_VERSION


def test_cx_r3_r1_q4_parser_accepts_current() -> None:
    data = minimal_runtime_event_payload_dict(RUNTIME_EVENT_SCHEMA_VERSION)
    event = parse_runtime_event_payload(data)
    assert event.schema_version == RUNTIME_EVENT_SCHEMA_VERSION


def test_cx_r3_r1_q5_retired_v1_rejected() -> None:
    with pytest.raises(ValueError, match="retired"):
        parse_runtime_event_payload(minimal_runtime_event_payload_dict("runtime_event.v1"))


def test_cx_r3_r1_q6_future_version_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_runtime_event_payload(minimal_runtime_event_payload_dict("runtime_event.v99"))


def test_cx_r3_r1_q7_write_current_accepted() -> None:
    assert_runtime_event_schema(_canonical_write_event())


def test_cx_r3_r1_q8_write_v1_and_future_rejected() -> None:
    with pytest.raises(RuntimeEventSchemaError):
        assert_runtime_event_schema(_canonical_write_event(schema_version="runtime_event.v1"))
    with pytest.raises(RuntimeEventSchemaError):
        assert_runtime_event_schema(_canonical_write_event(schema_version="runtime_event.v99"))


def test_cx_r3_r1_q9_preview_cannot_widen_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        PREVIEW_RUNTIME_SCHEMA_VERSIONS,
        "runtime_event",
        frozenset({"runtime_event.v99"}),
    )
    with pytest.raises(RuntimeEventSchemaError):
        assert_runtime_event_schema(_canonical_write_event(schema_version="runtime_event.v99"))
    assert not validate_schema_version("runtime_event", "runtime_event.v99")


def test_cx_r3_r1_q10_preview_cannot_widen_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        PREVIEW_RUNTIME_SCHEMA_VERSIONS,
        "runtime_event",
        frozenset({"runtime_event.v99"}),
    )
    with pytest.raises(ValueError, match="unsupported"):
        parse_runtime_event_payload(minimal_runtime_event_payload_dict("runtime_event.v99"))


def test_cx_r3_r1_q11_observability_provenance_chain() -> None:
    event = _canonical_write_event(tenant_id="tenant-provenance")
    source = runtime_event_export_source_from_event(event)
    envelope = envelope_from_runtime_event_source(source)
    assert source.source_schema_id == event.schema_version
    assert envelope.source_schema_id == event.schema_version
    assert envelope.tenant_id == event.tenant_id


def test_cx_r3_r1_q12_observability_no_v2_literal() -> None:
    assert observability_runtime_event_v2_literal_sites() == ()


def test_cx_r3_r1_q13_p0_combined_gate_passes() -> None:
    passed, violations = event_evolution_r3_p0_gate_passes()
    assert passed, violations


def test_cx_r3_r1_q14_production_persistence_bypass_zero() -> None:
    sites = discover_runtime_event_persistence_call_sites()
    assert validate_production_persistence_call_site_classification(sites) == ()
    assert count_canonical_production_raw_store_bypasses(sites) == 0


def test_cx_r3_r1_q15_no_second_evolution_authority() -> None:
    inv = event_evolution_inventory()
    concerns = [row.concern for row in inv.owner_matrix]
    assert len(concerns) == len(set(concerns))


def test_cx_r3_r1_combined_gate_passes_on_current_head() -> None:
    passed, violations = event_evolution_r3_r1_gate_passes()
    assert passed, violations


def test_cx_r3_r1_probe_b_registry_drift_would_fail() -> None:
    assert registry_runtime_event_version_parity_violations() == ()
    drift_violations = synthetic_registry_drift_violations()
    assert drift_violations
    drifted = {**RUNTIME_SCHEMA_REGISTRY, "runtime_event": "runtime_event.v99"}
    assert validate_runtime_event_registry_parity(drifted) == drift_violations
    passed, gate_violations = event_evolution_r3_r1_gate_passes(registry=drifted)
    assert not passed
    assert gate_violations


def test_cx_r3_r1_probe_i_synthetic_persisted_model_validate_bypass_fails_gate() -> None:
    site = synthetic_persisted_runtime_event_model_validate_bypass_site()
    assert site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED
    passed, violations = event_evolution_r3_r1_gate_passes(extra_reader_sites=(site,))
    assert not passed
    assert violations


def test_cx_r3_r1_probe_j_unknown_reader_unclassified_fails_gate() -> None:
    site = synthetic_unknown_persisted_reader_site()
    assert site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED
    passed, violations = event_evolution_r3_r1_gate_passes(extra_reader_sites=(site,))
    assert not passed
    assert violations


def test_cx_r3_r1_probe_j_second_evolution_authority_injection_fails() -> None:
    assert synthetic_second_evolution_authority_owner()


def test_cx_r3_r1_probe_a_second_v2_literal_fails() -> None:
    violations = synthetic_second_v2_literal_owner(
        extra_site=("intergrax/synthetic_probe_module.py", 1),
    )
    assert violations


def test_cx_r3_r1_probe_c_future_read_fail_closed() -> None:
    with pytest.raises(ValueError):
        parse_runtime_event_payload(minimal_runtime_event_payload_dict("runtime_event.v99"))


def test_cx_r3_r1_probe_d_future_write_fail_closed() -> None:
    with pytest.raises(RuntimeEventSchemaError):
        assert_runtime_event_schema(_canonical_write_event(schema_version="runtime_event.v99"))


def test_cx_r3_r1_probe_f_observability_synthetic_source_preserved() -> None:
    source = RuntimeEventExportSource(
        source_schema_id="runtime_event.synthetic_probe",
        event_id="evt-1",
        run_id="run-1",
        task_id="task-1",
        event_type="task_created",
    )
    envelope = envelope_from_runtime_event_source(source)
    assert envelope.source_schema_id == "runtime_event.synthetic_probe"


def test_cx_r3_r1_probe_g_source_mismatch_not_overwritten() -> None:
    event = _canonical_write_event()
    source = runtime_event_export_source_from_event(event).model_copy(
        update={"source_schema_id": "runtime_event.synthetic_probe"},
    )
    envelope = envelope_from_runtime_event_source(source)
    assert envelope.source_schema_id == "runtime_event.synthetic_probe"
    assert envelope.source_schema_id != RUNTIME_EVENT_SCHEMA_VERSION or (
        source.source_schema_id == RUNTIME_EVENT_SCHEMA_VERSION
    )


def test_cx_r3_r1_probe_h_p0_bypass_regression() -> None:
    assert synthetic_probe_k_unknown_persistence_append_gate() == ()


def test_cx_r3_r1_missing_envelope_version_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_runtime_event_payload(minimal_runtime_event_payload_dict(None))


def test_cx_r3_r1_envelope_from_runtime_event_end_to_end() -> None:
    event = _canonical_write_event()
    envelope = envelope_from_runtime_event(event)
    assert envelope.source_schema_id == event.schema_version


def test_cx_r3_r1_v1_literal_only_in_contract_owner() -> None:
    sites = production_runtime_event_v1_literal_sites()
    assert len(sites) == 1
    assert sites[0][0] == "intergrax/contracts/runtime_event.py"


def test_cx_r3_r1_persisted_readers_use_canonical_parser() -> None:
    readers = discover_persisted_runtime_event_read_paths()
    assert "intergrax/runtime/events/stores/sqlite_runtime_event_store.py" in readers
    assert "intergrax/runtime/events/stores/document_backed_runtime_event_store.py" in readers


def test_cx_r3_r1_reader_sites_closed_world_classified() -> None:
    sites = discover_runtime_event_reader_sites()
    assert sites
    assert all(site.disposition != RuntimeEventReaderDisposition.UNCLASSIFIED for site in sites)
    canonical = [site for site in sites if site.disposition == RuntimeEventReaderDisposition.CANONICAL_PARSER]
    assert len(canonical) == 2
    modules = {site.module_path for site in canonical}
    assert modules == {
        "intergrax/runtime/events/stores/document_backed_runtime_event_store.py",
        "intergrax/runtime/events/stores/sqlite_runtime_event_store.py",
    }


def test_cx_r3_r1_constructor_sites_closed_world_evidence_backed() -> None:
    ctor_sites = discover_runtime_event_constructor_sites()
    assert ctor_sites
    assert all(site.disposition != RuntimeEventReaderDisposition.UNCLASSIFIED for site in ctor_sites)
    assert all(
        site.classification_evidence != RuntimeEventReaderClassificationEvidence.UNSPECIFIED for site in ctor_sites
    )
    metrics = compute_event_evolution_r3_r1_metrics()
    assert metrics.constructor_unclassified == 0
    assert metrics.constructor_evidence_less_producer == 0
    assert metrics.constructor_candidate_count == len(ctor_sites)
    assert metrics.constructor_positively_classified == len(ctor_sites)


def test_cx_r3_r1_probe_k_runtime_event_kwargs_raw_reconstruction_fails_gate() -> None:
    site = synthetic_runtime_event_kwargs_raw_reconstruction_site()
    assert site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED
    passed, violations = event_evolution_r3_r1_gate_passes(extra_reader_sites=(site,))
    assert not passed
    assert violations


def test_cx_r3_r1_probe_l_unknown_constructor_unclassified_fails_gate() -> None:
    site = synthetic_unknown_runtime_event_constructor_site()
    assert site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED
    passed, violations = event_evolution_r3_r1_gate_passes(extra_reader_sites=(site,))
    assert not passed
    assert violations


def test_cx_r3_r1_probe_m_sanctioned_constructor_passes_classification() -> None:
    site = synthetic_sanctioned_runtime_event_constructor_site()
    assert site.disposition == RuntimeEventReaderDisposition.NON_PERSISTED_CONSTRUCTION
    assert site.classification_evidence == RuntimeEventReaderClassificationEvidence.PLATFORM_SIGNAL_EVENT_FACTORY
    passed, violations = event_evolution_r3_r1_gate_passes(extra_reader_sites=(site,))
    assert passed, violations


def test_cx_r3_r1_constructor_evidence_rule_integrity_on_current_head() -> None:
    assert count_module_wide_constructor_positive_evidence_rules() == 0
    assert count_stale_constructor_positive_evidence_rules() == 0
    assert validate_constructor_positive_evidence_rule_integrity() == ()


def test_cx_r3_r1_probe_n_unknown_qualname_in_trusted_module_fails_gate() -> None:
    site = synthetic_unknown_qualname_in_trusted_module_constructor_site()
    assert site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED
    passed, violations = event_evolution_r3_r1_gate_passes(extra_reader_sites=(site,))
    assert not passed
    assert violations


def test_cx_r3_r1_probe_o_stale_positive_constructor_rule_fails_integrity() -> None:
    stale_rule = synthetic_stale_constructor_positive_evidence_rule()
    violations = validate_constructor_positive_evidence_rule_integrity(extra_rules=(stale_rule,))
    assert violations


def test_cx_r3_r1_probe_persisted_reconstruction_in_trusted_module_fails_gate() -> None:
    site = synthetic_persisted_reconstruction_in_trusted_module_constructor_site()
    assert site.disposition == RuntimeEventReaderDisposition.UNCLASSIFIED
    passed, violations = event_evolution_r3_r1_gate_passes(extra_reader_sites=(site,))
    assert not passed
    assert violations


def test_cx_r3_r1_compat_current_head_inventory_reconciled() -> None:
    delta = build_r3_runtime_event_schema_version_candidate_delta()
    assert R3_RUNTIME_EVENT_SCHEMA_VERSION_CANDIDATE_ID in delta.added_candidate_ids
    assert delta.removed_candidate_ids == ()
    assert delta.current_remainder_fingerprint == REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256
    assert delta.historical_remainder_fingerprint == REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256
    assert validate_current_head_compat_closed_world_inventory() == ()


def test_cx_r3_r1_r3_candidate_semantic_identity_and_version_literal() -> None:
    delta = build_r3_runtime_event_schema_version_candidate_delta()
    assert delta.r3_reconciled_candidate_id == R3_RUNTIME_EVENT_SCHEMA_VERSION_CANDIDATE_ID
    assert RUNTIME_EVENT_SCHEMA_VERSION == "runtime_event.v2"


def test_cx_r3_r1_probe_q_equal_count_substitution_fails_inventory() -> None:
    report = build_synthetic_r3_inventory_equal_count_substitution_report()
    assert len(report.raw_candidates) == ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES
    violations = validate_current_head_compat_closed_world_inventory(report)
    assert violations
    assert any("fingerprint" in v for v in violations)


def test_cx_r3_r1_probe_r_hidden_removal_fails_inventory() -> None:
    report = build_synthetic_r3_inventory_hidden_removal_report()
    assert len(report.raw_candidates) == REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES
    violations = validate_current_head_compat_closed_world_inventory(report)
    assert violations


def test_cx_r3_r1_probe_s_balanced_add_remove_fails_inventory() -> None:
    report = build_synthetic_r3_inventory_balanced_add_remove_report()
    assert len(report.raw_candidates) == ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES
    violations = validate_current_head_compat_closed_world_inventory(report)
    assert violations
    assert any("fingerprint" in v for v in violations)


def test_cx_r3_r1_historical_inventory_provenance_constants() -> None:
    canonical = "0e5262199a6c2f1737cf8ff48def53945ec02974"
    assert ACCEPTED_P0_HISTORICAL_INVENTORY_SOURCE_SHA == canonical
    assert ACCEPTED_P0_STABLE_INVENTORY_SHA256 == ("7a39eec8a37f4af21d9a6aca90981a4e6f5ed83fcdbb5feb5cea633efc52ca80")
    import tests.qualification.compat_x._compat_x_semantic_identity_reconciliation as recon

    assert not hasattr(recon, "ACCEPTED_P0_RAW_CANDIDATE_IDS_SHA256")
    assert not hasattr(recon, "ACCEPTED_P0_RAW_CANDIDATE_IDS_FINGERPRINT_SOURCE_SHA")


def test_cx_r3_r1_probe_t_wire_lineno_only_stable_identity_normalized() -> None:
    old, new = synthetic_wire_persistence_lineno_only_pair()
    assert stable_inventory_candidate_identity(old) == RUNTIME_EVENT_MODEL_VALIDATE_STABLE_IDENTITY
    assert stable_inventory_candidate_identity(new) == RUNTIME_EVENT_MODEL_VALIDATE_STABLE_IDENTITY
    assert RUNTIME_EVENT_MODEL_VALIDATE_HISTORICAL_CANDIDATE_ID != RUNTIME_EVENT_MODEL_VALIDATE_CURRENT_CANDIDATE_ID


def test_cx_r3_r1_probe_u_semantic_wire_movement_changes_fingerprint() -> None:
    base_set = stable_inventory_identity_set(_remainder_candidates_minus_r3(build_closed_world_report()))
    base_fp = inventory_identity_fingerprint(base_set)
    variant = synthetic_wire_persistence_semantic_variant(
        semantic_identity="wire:intergrax/contracts/runtime_event.py:module_level:model_validate:99",
        candidate_id="wire.intergrax.contracts.runtime_event.py::module_level.model_validate:149:99",
    )
    variant_identity = stable_inventory_candidate_identity(variant)
    assert variant_identity not in base_set
    mutated = base_set | {variant_identity}
    assert inventory_identity_fingerprint(mutated) != base_fp


def test_cx_r3_r1_positional_only_runtime_event_model_validate_drift_evidence() -> None:
    """Audit record: raw candidate_id lineno drift is non-semantic (SOURCE-LINE-ONLY)."""
    assert RUNTIME_EVENT_MODEL_VALIDATE_HISTORICAL_CANDIDATE_ID.endswith(":150:1")
    assert RUNTIME_EVENT_MODEL_VALIDATE_CURRENT_CANDIDATE_ID.endswith(":149:1")


def test_cx_r3_r1_probe_p_unknown_extra_compat_candidate_fails_inventory() -> None:
    source = """
COMMON_SCHEMA_VERSION = "probe.compat.extra.v1"

class SyntheticCompatProbe:
    schema_version = COMMON_SCHEMA_VERSION
"""
    module = "synthetic/qualification/r3_r1_compat_closed_world_probe.py"
    extra = tuple(discover_candidates_from_source(module, source))
    assert extra
    report = build_closed_world_report_with_extra_candidates(extra)
    violations = validate_current_head_compat_closed_world_inventory(report)
    assert violations
