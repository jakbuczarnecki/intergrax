# © Artur Czarnecki. All rights reserved.

"""Canonical PT-T01 matrix: Campaign B NT-01..NT-16 tracked pytest nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MatrixStatus = Literal["ACTIVE", "HELD_OUT", "CONDITIONAL"]


@dataclass(frozen=True, slots=True)
class PtT01Row:
    nt_id: str
    finding_id: str
    status: MatrixStatus
    tracked_pytest: tuple[str, ...]
    owner: str
    held_out_reason: str = ""
    baseline_red: str = ""


PT_T01_MATRIX: tuple[PtT01Row, ...] = (
    PtT01Row(
        "NT-01",
        "A4-044",
        "ACTIVE",
        ("tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py",),
        "T-IGX-043-A",
    ),
    PtT01Row(
        "NT-02",
        "A4-015",
        "ACTIVE",
        (
            "tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py::test_tier2_build_context_takes_posture_from_the_host",
        ),
        "T-IGX-043-A",
    ),
    PtT01Row(
        "NT-03",
        "A4-026",
        "ACTIVE",
        ("tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py",),
        "T-IGX-043-C",
    ),
    PtT01Row(
        "NT-04a",
        "A4-027",
        "ACTIVE",
        ("tests/unit/applications/test_strict_production_serving_execution_tenant_authority.py",),
        "T-IGX-043-J",
    ),
    PtT01Row("NT-04b", "A4-026", "HELD_OUT", (), "T-IGX-051-J", held_out_reason="detector scope"),
    PtT01Row(
        "NT-05",
        "A4-037",
        "ACTIVE",
        (
            "tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py::test_injected_human_decision_store_reaches_tool_wiring_context",
        ),
        "T-IGX-043-A",
    ),
    PtT01Row(
        "NT-06",
        "A4-007",
        "ACTIVE",
        (
            "tests/unit/agents/test_wa_agent_import_rule.py",
            "tests/unit/runtime/architecture/test_wa_boundary_inversion_gate.py",
        ),
        "T-IGX-043-G",
    ),
    PtT01Row("NT-07", "A2-049", "HELD_OUT", (), "T-IGX-051-J", held_out_reason="trace classifier"),
    PtT01Row(
        "NT-08",
        "A6-007",
        "CONDITIONAL",
        ("tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py",),
        "T-IGX-043-I",
        held_out_reason="PT-WD owns",
    ),
    PtT01Row(
        "NT-09",
        "X-ESC-E2",
        "ACTIVE",
        ("tests/unit/runtime/execution/test_decision_finalization_persistence.py",),
        "T-IGX-043-J",
    ),
    PtT01Row("NT-10", "X-TLX", "ACTIVE", ("tests/unit/runtime/tools/test_tools_side_effect_safety.py",), "T-IGX-043-E"),
    PtT01Row(
        "NT-11",
        "A8-020",
        "ACTIVE",
        (
            "tests/unit/rag/vectorstore/test_collection_access_policy.py::test_bound_tenant_rejects_mismatched_document_before_provider",
        ),
        "T-IGX-043-J",
    ),
    PtT01Row(
        "NT-12",
        "X-A4-016",
        "ACTIVE",
        (
            "tests/unit/applications/test_wb_llm_resolver_delegation.py::test_foreign_tenant_run_does_not_receive_the_routing_record",
        ),
        "T-IGX-043-H",
    ),
    PtT01Row("NT-13", "X-A8-003", "ACTIVE", ("tests/unit/contracts/test_wb_profile_projection.py",), "T-IGX-043-H"),
    PtT01Row(
        "NT-14",
        "X-S2b",
        "ACTIVE",
        (
            "tests/unit/tools/providers/rag/test_rag_retrieve.py::test_rag_retrieve_fails_closed_without_request_or_bound_tenant",
        ),
        "T-IGX-043-G",
    ),
    PtT01Row(
        "NT-15",
        "X-ESC-E4",
        "ACTIVE",
        (
            "tests/unit/experiments/test_wf_experiment_store_tenant.py",
            "tests/unit/applications/test_harness_auth.py::test_sec_r1_idp_only_dependency_fail_closed",
        ),
        "T-IGX-043-J",
    ),
    PtT01Row("NT-15e", "X-ESC-E5", "HELD_OUT", (), "PT-WF-follow-on", held_out_reason="ERL tenant"),
    PtT01Row(
        "NT-16",
        "A7-031",
        "ACTIVE",
        (
            "tests/qualification/pdc_x/d6/test_xd06_provider_boundary.py::test_xd06_tenant_credential_is_not_released_for_another_tenant_scope",
        ),
        "T-IGX-051-A",
    ),
)

A4_006_TRACKED = "tests/unit/scripts/test_check_runtime_event_tenant_propagation.py"

EXPECTED_NT_IDS = frozenset(row.nt_id for row in PT_T01_MATRIX)

ACTIVE_TRACKED_PYTEST_ARGS: tuple[str, ...] = tuple(
    path for row in PT_T01_MATRIX if row.status == "ACTIVE" for path in row.tracked_pytest
) + (A4_006_TRACKED,)
