# © Artur Czarnecki. All rights reserved.

"""GOV-X2 qualification-only invariant matrix (GX2-01..GX2-20) — mechanical evidence SSOT."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

GOV_X2_START_HEAD: Final[str] = "ccf1449144d256ffbe92ddba2450b38ff6dfae99"


class GovX2InvariantResult(StrEnum):
    PASS = "PASS"
    NOT_APPLICABLE_WITH_EVIDENCE = "N/A — WITH EVIDENCE"


@dataclass(frozen=True, slots=True)
class GovX2InvariantEvidence:
    invariant_id: str
    summary: str
    result: GovX2InvariantResult
    primary_pytest_node_ids: tuple[str, ...]
    supporting_artifacts: tuple[str, ...] = ()
    notes: str = ""


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


_Q_AUTH = "tests/qualification/governance/test_governance_e2e_authorization.py"
_GR2_LAUNCHER = "tests/unit/runtime/governance/test_gr2_r3_root_execution_launcher.py"
_GR2_ADMISSION = (
    "tests/unit/runtime/governance/test_runtime_execution_policy_admission.py"
)
_GR3 = "tests/unit/runtime/governance/test_gr3_canonical_inner_enforcement.py"
_GR6_R1 = "tests/unit/runtime/policy/test_gr6_r1_decision_requirement_enforcement.py"
_MP4R7 = "tests/unit/mp4r7/test_enterprise_integration_qualification.py"
_PG_C = "tests/unit/agents/external_contractor_adapter/test_pg_fix_c_scoped_approval_grant.py"
_G5C2 = "tests/unit/runtime/human/test_g5c2b1_governed_continuation_grant.py"
_GR12_FINAL = (
    "tests/qualification/governance/gr12/test_gr12_final_control_plane_qualification.py"
)
_GR13 = (
    "tests/qualification/governance/gr13/test_gr13_governance_evidence_proof_matrix.py"
)
_GR11 = (
    "tests/qualification/governance/gr11/test_gr11_plugin_enterprise_certification.py"
)
_GR10_BATCH = (
    "tests/qualification/governance/strategy/test_gr10_qualification_batch.py"
)
_TOOL = "tests/qualification/governance/test_governance_e2e_pluginability.py"
_HITL = "tests/qualification/governance/test_governance_e2e_hitl.py"
_DECISION = "tests/qualification/governance/test_governance_e2e_decision.py"
_FAIL_CLOSED = "tests/qualification/governance/test_governance_e2e_fail_closed.py"


GOV_X2_INVARIANT_CATALOG: tuple[GovX2InvariantEvidence, ...] = (
    GovX2InvariantEvidence(
        "GX2-01",
        "Governance ≠ Execution — separate admission vs permission",
        GovX2InvariantResult.PASS,
        (
            _nid(_Q_AUTH, "test_scenario_a_root_admission_allow_single_intake"),
            _nid(_Q_AUTH, "test_scenario_b_root_admission_deny_zero_intake"),
        ),
        ("docs/project/maintainers/qualification/GOV_X1_GOVERNANCE_AUTHORITY_BOUNDARY_RECERTIFICATION.md",),
    ),
    GovX2InvariantEvidence(
        "GX2-02",
        "Proposal ≠ permission",
        GovX2InvariantResult.PASS,
        (
            _nid(
                _GR6_R1,
                "test_direct_boundary_required_without_material_denies_no_effect",
            ),
            _nid(_DECISION, "test_scenario_l_stale_proposal_fail_closed"),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-03",
        "Permission ≠ execution without canonical admission",
        GovX2InvariantResult.PASS,
        (
            _nid(
                _GR2_LAUNCHER,
                "test_launcher_deny_skips_intake[root.execution.agent]",
            ),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-04",
        "Human approval does not bypass fresh Governance evaluation",
        GovX2InvariantResult.PASS,
        (
            _nid(
                _MP4R7,
                "test_mp4r7_human_approve_governance_deny_prevents_continuation_and_operation",
            ),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-05",
        "Evidence does not grant or widen authority",
        GovX2InvariantResult.PASS,
        (_nid(_GR13, "test_g13_08_evidence_non_authoritative_proof_nodes_registered"),),
        ("tests/qualification/governance/gr13/catalog.py",),
    ),
    GovX2InvariantEvidence(
        "GX2-06",
        "Decision acceptance does not grant execution permission",
        GovX2InvariantResult.PASS,
        (
            _nid(
                _GR6_R1,
                "test_direct_boundary_required_without_material_denies_no_effect",
            ),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-07",
        "Child/downstream authority only narrows",
        GovX2InvariantResult.PASS,
        (_nid(_GR11, "test_gr11_g19_decision_requirement_not_execution_authority"),),
        ("tests/qualification/governance/gr11/catalog.py",),
    ),
    GovX2InvariantEvidence(
        "GX2-08",
        "Tenant scope cannot widen downstream",
        GovX2InvariantResult.PASS,
        (_nid(_MP4R7, "test_mp4r7_cross_tenant_fail_closed"),),
    ),
    GovX2InvariantEvidence(
        "GX2-09",
        "Fresh authorization immediately before meaningful side effect",
        GovX2InvariantResult.PASS,
        (
            _nid(_GR6_R1, "test_coordinator_with_valid_material_and_governance_allow"),
            _nid(
                _GR12_FINAL,
                "test_gr12_final_f21_stale_revision_negative_proofs_registered",
            ),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-10",
        "Stale authorization cannot be reused",
        GovX2InvariantResult.PASS,
        (
            _nid(_MP4R7, "test_mp4r7_stale_proposal_fail_closed"),
            _nid(_PG_C, "test_c11_fresh_deny_over_grant_blocks_provider"),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-11",
        "Missing HITL material cannot become approval",
        GovX2InvariantResult.PASS,
        (_nid(_GR12_FINAL, "test_gr12_final_f22_hitl_does_not_equate_approval_with_permission"),),
    ),
    GovX2InvariantEvidence(
        "GX2-12",
        "REQUIRE_HUMAN → pause / zero effect until resolution",
        GovX2InvariantResult.PASS,
        (_nid(_GR3, "test_deny_and_require_human_execute_zero_effects"),),
    ),
    GovX2InvariantEvidence(
        "GX2-13",
        "Continuation lifecycle Execution-owned",
        GovX2InvariantResult.PASS,
        (_nid(_G5C2, "test_task_mismatch_fails_closed"),),
        ("tests/qualification/governance/gr11/catalog.py",),
    ),
    GovX2InvariantEvidence(
        "GX2-14",
        "Tool planning/access ≠ tool invocation authorization",
        GovX2InvariantResult.PASS,
        (
            _nid(
                _TOOL,
                "test_scenario_plugin_custom_runtime_execution_policy_admission_via_composition_launcher",
            ),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-15",
        "Provider result cannot mint Governance permission",
        GovX2InvariantResult.PASS,
        (
            _nid(
                "applications/governed_contractor_application/tests/host/test_gr7_a4_unknown_host_state_separation.py",
                "test_map_provider_invocation_outcome_status_to_host_state",
            ),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-16",
        "Control-plane mutation — single sanctioned permission authority",
        GovX2InvariantResult.PASS,
        (_nid(_GR12_FINAL, "test_gr12_final_f02_applicable_rows_are_qualified"),),
    ),
    GovX2InvariantEvidence(
        "GX2-17",
        "Runtime extensions cannot self-expand authority",
        GovX2InvariantResult.PASS,
        (_nid(_GR11, "test_gr11_g21_dynamic_registration_classified_and_composition_time_evidence"),),
    ),
    GovX2InvariantEvidence(
        "GX2-18",
        "Alternate governance/execution path count = 0 (certified closed-world)",
        GovX2InvariantResult.PASS,
        (
            _nid(
                _GR12_FINAL,
                "test_gr12_final_f24_catalogued_surfaces_match_bypass_inventory",
            ),
            _nid(_GR10_BATCH, "test_gr10_all_catalog_pytest_node_ids_are_collectable"),
        ),
    ),
    GovX2InvariantEvidence(
        "GX2-19",
        "Authorization/effect identity correlated",
        GovX2InvariantResult.PASS,
        (_nid(_GR3, "test_exact_four_id_match_allow_executes_once"),),
    ),
    GovX2InvariantEvidence(
        "GX2-20",
        "Protected successful effect has evidence attributable to authorization",
        GovX2InvariantResult.PASS,
        (
            _nid(_GR13, "test_g13_06_shared_strategy_rows_prove_same_canonical_module"),
            _nid(_MP4R7, "test_mp4r7_evidence_failure_preserves_primary"),
        ),
    ),
)


GOV_X2_E2E_CLASS_NODE_IDS: Final[tuple[str, ...]] = (
    _nid(_Q_AUTH, "test_scenario_a_root_admission_allow_single_intake"),
    _nid(_GR3, "test_exact_four_id_match_allow_executes_once"),
    _nid(_GR6_R1, "test_coordinator_with_valid_material_and_governance_allow"),
    _nid(_TOOL, "test_scenario_plugin_allowing_vs_denying_admission_via_composition"),
    _nid(_MP4R7, "test_mp4r7_success_e2e"),
    _nid(
        "applications/governed_contractor_application/tests/host/test_gr7_a3_durable_provider_invocation.py",
        "test_success_persists_intent_before_outcome_and_ger",
    ),
    _nid(_GR12_FINAL, "test_gr12_final_f02_applicable_rows_are_qualified"),
    _nid(_GR11, "test_gr11_g19_decision_requirement_not_execution_authority"),
    _nid(_HITL, "test_scenario_n_through_o_mp4r7_require_human_then_resume"),
    _nid(_FAIL_CLOSED, "test_failure_matrix_catalog_documents_policy_evaluation_fail_closed"),
)


def gov_x2_all_proof_pytest_node_ids() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in GOV_X2_INVARIANT_CATALOG:
        for node_id in entry.primary_pytest_node_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered.append(node_id)
    for node_id in GOV_X2_E2E_CLASS_NODE_IDS:
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
    return tuple(ordered)
