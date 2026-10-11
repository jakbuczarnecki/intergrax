# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R3-P0 event evolution closed-world inventory and policy lock gates."""

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_event import parse_runtime_event_payload
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payload_registry import (
    UnknownPayloadSchemaError,
    validate_payload_envelope,
)
from intergrax.runtime.events.payloads import CANONICAL_PAYLOAD_TYPES, RuntimeEventPayload
from intergrax.runtime.events.runtime_event_payload_policy import (
    PayloadWriteMode,
    iter_runtime_event_payload_policies,
)
from intergrax.runtime.schema.registry import RUNTIME_SCHEMA_REGISTRY, validate_schema_version

from tests.qualification.compat_x._compat_x_event_evolution_inventory import (
    EventCompatibilityDisposition,
    RuntimeEventWritePathDisposition,
    compute_event_evolution_r3_p0_metrics,
    count_canonical_production_raw_store_bypasses,
    discover_runtime_event_persistence_call_sites,
    envelope_fail_closed_probes,
    event_evolution_inventory,
    event_evolution_r3_p0_gate_passes,
    inventory_payload_schema_ids,
    probe_extension_duplicate_canonical_override,
    registered_canonical_payload_schema_ids,
    synthetic_canonical_composition_raw_store_bypass_probe,
    synthetic_inventory_missing_classification,
    synthetic_inventory_phantom_schema_id,
    synthetic_inventory_phantom_schema_row_probe,
    synthetic_second_evolution_authority_owner,
    synthetic_two_current_write_versions_for_family,
    synthetic_undispositioned_historical_version,
    synthetic_write_policy_unregistered_schema_probe,
    validate_canonical_production_guard_chains,
    validate_inventory_closed_world,
    validate_inventory_closed_world_for_extra_class,
    validate_production_persistence_call_site_classification,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_r3_p0_closed_world_inventory_gate() -> None:
    violations = validate_inventory_closed_world()
    assert violations == ()
    metrics = compute_event_evolution_r3_p0_metrics()
    assert metrics.unclassified_payload_classes == 0
    assert metrics.duplicate_schema_ids == 0
    assert metrics.duplicate_family_membership == 0
    assert metrics.duplicate_write_ownership == 0


def test_cx_r3_p0_q1_every_canonical_payload_in_one_family() -> None:
    inv = event_evolution_inventory()
    seen: dict[type[RuntimeEventPayload], str] = {}
    for family in inv.payload_families:
        for version in family.versions:
            assert version.payload_type not in seen
            seen[version.payload_type] = family.family_id
    assert len(seen) == len(CANONICAL_PAYLOAD_TYPES)


def test_cx_r3_p0_q2_inventory_schema_ids_match_registry() -> None:
    inv = event_evolution_inventory()
    assert inventory_payload_schema_ids(inv) == registered_canonical_payload_schema_ids()


def test_cx_r3_p0_q3_registry_classes_in_inventory() -> None:
    inv = event_evolution_inventory()
    inv_classes = {v.payload_type for f in inv.payload_families for v in f.versions}
    assert inv_classes == set(CANONICAL_PAYLOAD_TYPES)


def test_cx_r3_p0_q4_runtime_event_type_write_policy_unique() -> None:
    policies = list(iter_runtime_event_payload_policies())
    assert len(policies) == len(RuntimeEventType)
    assert {event_type for event_type, _ in policies} == set(RuntimeEventType)


def test_cx_r3_p0_q5_strict_write_schemas_registered() -> None:
    reg = registered_canonical_payload_schema_ids()
    for event_type, entry in iter_runtime_event_payload_policies():
        if entry.write_mode != PayloadWriteMode.STRICT_SPINE_SCHEMA:
            continue
        assert entry.schema_id is not None
        assert entry.schema_id in reg, event_type.value


def test_cx_r3_p0_q6_multi_version_family_write_rules() -> None:
    inv = event_evolution_inventory()
    for family in inv.payload_families:
        if len(family.versions) <= 1:
            continue
        active = [
            v.schema_id
            for v in family.versions
            if v.compatibility_disposition == EventCompatibilityDisposition.CURRENT_WRITE_AND_READ
        ]
        if family.allows_multiple_active_write_versions:
            assert len(active) >= 1, family.family_id
        else:
            assert len(active) == 1, family.family_id


def test_cx_r3_p0_q7_historical_supported_read_explicit() -> None:
    inv = event_evolution_inventory()
    for family in inv.payload_families:
        for version in family.versions:
            assert version.compatibility_disposition != EventCompatibilityDisposition.UNKNOWN_FAIL_CLOSED


def test_cx_r3_p0_q8_envelope_matches_runtime_registry() -> None:
    current = RUNTIME_SCHEMA_REGISTRY["runtime_event"]
    assert current == "runtime_event.v2"
    assert validate_schema_version("runtime_event", current)
    inv = event_evolution_inventory()
    current_rows = [
        r for r in inv.envelope_records if r.schema_version == current
    ]
    assert len(current_rows) == 1


def test_cx_r3_p0_q9_runtime_event_v1_not_supported_read() -> None:
    assert not validate_schema_version("runtime_event", "runtime_event.v1")
    inv = event_evolution_inventory()
    retired = [
        r
        for r in inv.envelope_records
        if r.schema_version == "runtime_event.v1"
    ]
    assert len(retired) == 1
    assert retired[0].compatibility_disposition == EventCompatibilityDisposition.RETIRED_FAIL_CLOSED


def test_cx_r3_p0_q10_envelope_missing_unknown_fail_closed() -> None:
    probes = envelope_fail_closed_probes()
    assert len(probes) == 3
    joined = " ".join(probes).lower()
    assert "missing" in joined or "expected" in joined
    assert "retired" in joined or "unsupported" in joined


def test_cx_r3_p0_q11_unknown_payload_schema_fail_closed() -> None:
    with pytest.raises(UnknownPayloadSchemaError):
        validate_payload_envelope(
            {
                "payload_schema_id": "unknown.compat_x_probe.v1",
                "payload_schema_version": 1,
                "data": {},
            }
        )


def test_cx_r3_p0_q12_extension_duplicate_override_fail_closed() -> None:
    message = probe_extension_duplicate_canonical_override()
    assert message is not None
    assert "duplicate" in message.lower()


def test_cx_r3_p0_q13_no_duplicate_evolution_policy_owners() -> None:
    inv = event_evolution_inventory()
    concerns = [row.concern for row in inv.owner_matrix]
    assert len(concerns) == len(set(concerns))


def test_cx_r3_p0_q14_qualification_not_version_authority() -> None:
    inv = event_evolution_inventory()
    for row in inv.owner_matrix:
        assert "tests/qualification" not in row.semantic_owner_path


def test_cx_r3_p0_q15_tenant_continuity_policy_evidence() -> None:
    """Compatibility policy documents representation-only evolution; no tenant rewrite hooks."""
    inv = event_evolution_inventory()
    tenant_rules = [r for r in inv.extension_contract.rules if "tenant" in r.lower()]
    assert len(tenant_rules) == 1
    for family in inv.payload_families:
        for version in family.versions:
            assert not version.adaptation_required
            assert not version.migration_required


def test_cx_r3_p0_probe_a_unclassified_payload_fails() -> None:
    class _SyntheticOrphan(RuntimeEventPayload):
        schema_id = "synthetic.orphan_probe.v1"
        schema_version = 1

    violations = validate_inventory_closed_world_for_extra_class(_SyntheticOrphan)
    assert any("probe" in v or "unclassified" in v for v in violations)


def test_cx_r3_p0_probe_b_phantom_schema_not_in_registry() -> None:
    phantom = synthetic_inventory_phantom_schema_id()
    assert phantom not in registered_canonical_payload_schema_ids()
    violations = synthetic_inventory_phantom_schema_row_probe()
    assert violations
    assert not any("did not fail closed" in v for v in violations)


def test_cx_r3_p0_probe_c_two_current_writes_disallowed_family_fails() -> None:
    violations = synthetic_two_current_write_versions_for_family()
    assert violations


def test_cx_r3_p0_probe_d_undispositioned_version_fails() -> None:
    violations = synthetic_undispositioned_historical_version()
    assert violations


def test_cx_r3_p0_probe_e_unknown_envelope_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_runtime_event_payload(
            {
                "schema_version": "runtime_event.v99",
                "task_id": "task_test",
                "run_id": "run_test",
                "attempt_id": "attempt_test",
                "execution_id": "execution_test",
                "event_type": "task_created",
                "phase": "planning",
            }
        )


def test_cx_r3_p0_probe_f_runtime_event_v1_fail_closed() -> None:
    with pytest.raises(ValueError, match="retired"):
        parse_runtime_event_payload(
            {
                "schema_version": "runtime_event.v1",
                "task_id": "task_test",
                "run_id": "run_test",
                "attempt_id": "attempt_test",
                "execution_id": "execution_test",
                "event_type": "task_created",
                "phase": "planning",
            }
        )


def test_cx_r3_p0_probe_g_unknown_payload_fail_closed() -> None:
    with pytest.raises(UnknownPayloadSchemaError):
        validate_payload_envelope(
            {"payload_schema_id": "nope.v1", "payload_schema_version": 1, "data": {}}
        )


def test_cx_r3_p0_probe_h_extension_duplicate_id() -> None:
    assert probe_extension_duplicate_canonical_override() is not None


def test_cx_r3_p0_probe_i_write_policy_unregistered_schema_detection() -> None:
    violations = synthetic_write_policy_unregistered_schema_probe()
    assert violations
    assert any("unregistered" in v for v in violations)


def test_cx_r3_p0_probe_j_second_evolution_authority_fails() -> None:
    violations = synthetic_second_evolution_authority_owner()
    assert violations


def test_cx_r3_p0_metrics_snapshot() -> None:
    metrics = compute_event_evolution_r3_p0_metrics()
    assert metrics.runtime_event_type_count == len(RuntimeEventType)
    assert metrics.canonical_payload_class_count == len(CANONICAL_PAYLOAD_TYPES)
    assert metrics.multi_version_family_count == 2
    assert metrics.strict_production_write_policies == len(RuntimeEventType) - 1
    assert metrics.extension_event_kind_policies == 1


def test_cx_r3_p0_payload_version_explicit_in_inventory() -> None:
    inv = event_evolution_inventory()
    for family in inv.payload_families:
        for version in family.versions:
            assert version.payload_schema_version == int(version.payload_type.schema_version)


def test_cx_r3_p0_mechanical_persistence_discovery_classified() -> None:
    sites = discover_runtime_event_persistence_call_sites()
    assert sites
    assert validate_production_persistence_call_site_classification(sites) == ()
    assert count_canonical_production_raw_store_bypasses(sites) == 0
    assert all(
        site.disposition != RuntimeEventWritePathDisposition.UNCLASSIFIED for site in sites
    )


def test_cx_r3_p0_canonical_composition_guard_chains() -> None:
    assert validate_canonical_production_guard_chains() == ()


def test_cx_r3_p0_probe_bypass_raw_store_without_validator() -> None:
    violations = synthetic_canonical_composition_raw_store_bypass_probe()
    assert violations
    assert any("ValidatingRuntimeEventPersistence" in v for v in violations)


def test_cx_r3_p0_combined_gate_passes() -> None:
    passed, violations = event_evolution_r3_p0_gate_passes()
    assert passed, violations
