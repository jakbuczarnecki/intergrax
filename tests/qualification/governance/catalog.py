# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 scenario catalog — maps matrix rows to canonical pytest evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GovFinal4ScenarioResult(StrEnum):
    QUALIFIED = "QUALIFIED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "N/A"
    GAP = "GAP"


@dataclass(frozen=True, slots=True)
class GovFinal4ScenarioEvidence:
    scenario_id: str
    title: str
    expected: str
    result: GovFinal4ScenarioResult
    primary_pytest_node_ids: tuple[str, ...]
    notes: str = ""


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


_Q_AUTH = "tests/qualification/governance/test_governance_e2e_authorization.py"
_GR2_LAUNCHER = "tests/unit/runtime/governance/test_gr2_r3_root_execution_launcher.py"
_GR2_ADMISSION = "tests/unit/runtime/governance/test_runtime_execution_policy_admission.py"
_GR2_GOV = "tests/unit/runtime/governance/test_gr2_execution_admission_governance.py"
_GR3 = "tests/unit/runtime/governance/test_gr3_canonical_inner_enforcement.py"
_PG_B = "tests/unit/runtime/policy/test_pg_fix_b_side_effect_policy_precedence.py"
_PG_A = "tests/unit/agents/external_contractor_adapter/test_pg_fix_a_side_effect_boundary.py"
_GR6_R1 = "tests/unit/runtime/policy/test_gr6_r1_decision_requirement_enforcement.py"
_GR6_RS1 = "tests/unit/runtime/test_gr6_rs1_decision_subject_resource_binding.py"
_GR6_ACTION = "tests/unit/contracts/test_gr6_arch_canonical_action_identity.py"
_MP4R7 = "tests/unit/mp4r7/test_enterprise_integration_qualification.py"
_PG_C = "tests/unit/agents/external_contractor_adapter/test_pg_fix_c_scoped_approval_grant.py"
_G5C2 = "tests/unit/runtime/human/test_g5c2b1_governed_continuation_grant.py"
_GR7_A3 = (
    "applications/governed_contractor_application/tests/host/test_gr7_a3_durable_provider_invocation.py"
)
_GR7_A4 = (
    "applications/governed_contractor_application/tests/host/test_gr7_a4_unknown_host_state_separation.py"
)
_GR7_A6 = (
    "applications/governed_contractor_application/tests/host/test_gr7_a6_provider_reconciliation.py"
)
_GR7_A7 = (
    "applications/governed_contractor_application/tests/host/test_gr7_a7_provider_recovery.py"
)
_MSE_POLICY = "tests/unit/runtime/policy/test_meaningful_side_effect_policy.py"
_GR3_PLUGIN = "tests/unit/runtime/governance/test_gr3_r3_explicit_plugin_selection_semantics.py"
_GR2_PLUGIN = _GR2_LAUNCHER


GOV_FINAL_4_SCENARIO_CATALOG: tuple[GovFinal4ScenarioEvidence, ...] = (
    GovFinal4ScenarioEvidence(
        "A",
        "Root execution admission ALLOW",
        "admitted → exactly one intake",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_Q_AUTH, "test_scenario_a_root_admission_allow_single_intake"),
            _nid(_GR2_LAUNCHER, "test_launcher_allow_runs_admission_and_intake_once[ROOT_AGENT]"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "B",
        "Root execution admission DENY",
        "0 intake / 0 execution starts",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_Q_AUTH, "test_scenario_b_root_admission_deny_zero_intake"),
            _nid(_GR2_LAUNCHER, "test_launcher_deny_skips_intake[ROOT_AGENT]"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "C",
        "Root admission policy failure fail-closed",
        "no default ALLOW",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_GR2_ADMISSION, "test_evaluator_unconfigured_fail_closed"),
            _nid(_GR2_ADMISSION, "test_unavailable_adapter_fail_closed"),
            _nid(_GR2_ADMISSION, "test_evaluator_no_matching_rule_indeterminate"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "D",
        "Inner governance ALLOW",
        "operation executes exactly once",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR3, "test_exact_four_id_match_allow_executes_once"),),
    ),
    GovFinal4ScenarioEvidence(
        "E",
        "Inner governance DENY",
        "0 consequential execution",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR3, "test_deny_and_require_human_execute_zero_effects"),),
    ),
    GovFinal4ScenarioEvidence(
        "F",
        "Invalid / missing execution identity",
        "fail closed, 0 effect",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_GR3, "test_no_active_execution_blocks_effect"),
            _nid(_GR3, "test_identity_mismatch_blocks_effect[task_id]"),
            _nid(_GR3, "test_identity_mismatch_blocks_effect[run_id]"),
            _nid(_GR3, "test_identity_mismatch_blocks_effect[attempt_id]"),
            _nid(_GR3, "test_identity_mismatch_blocks_effect[execution_id]"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "G",
        "Policy precedence (specific DENY over broad ALLOW)",
        "DENY",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_PG_B, "test_b1_historical_exploit_wildcard_allow_specific_deny"),
            _nid(_PG_B, "test_b12_external_work_integration_broad_allow_cannot_bypass_specific_deny"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "H",
        "Explicit typed action/resource match",
        "no suffix/reflection semantics",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_MSE_POLICY, "test_action_filtering"),
            _nid(_PG_A, "test_a5_deny_blocks_provider"),
        ),
        notes="AST gates in GR-4 architecture tests supplement explicit matching.",
    ),
    GovFinal4ScenarioEvidence(
        "I",
        "MSE NOT_REQUIRED without Decision material",
        "still passes Governance",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR6_R1, "test_not_required_without_material_preserves_governance_allow"),),
    ),
    GovFinal4ScenarioEvidence(
        "J",
        "MSE REQUIRED + valid material + Governance ALLOW",
        "execute",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_GR6_R1, "test_coordinator_with_valid_material_and_governance_allow"),
            _nid(_MP4R7, "test_mp4r7_success_e2e"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "K",
        "REQUIRED Decision material missing",
        "DENY / fail closed",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR6_R1, "test_direct_boundary_required_without_material_denies_no_effect"),),
    ),
    GovFinal4ScenarioEvidence(
        "L",
        "Stale / wrong Decision material",
        "fail closed",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_MP4R7, "test_mp4r7_stale_proposal_fail_closed"),
            _nid(_GR6_RS1, "test_wrong_resource_denied_at_boundary_no_effect"),
            _nid(_GR6_R1, "test_four_id_mismatch_on_material_denied"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "M",
        "Decision accepted + fresh Governance DENY",
        "0 effect",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_MP4R7, "test_mp4r7_human_approve_governance_deny_prevents_continuation_and_operation"),),
    ),
    GovFinal4ScenarioEvidence(
        "N",
        "Governance REQUIRE_HUMAN",
        "pause / 0 effect",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_MP4R7, "test_mp4r7_success_e2e"),),
        notes="REQUIRE_HUMAN path exercised before human approval in MP-4R7 success flow.",
    ),
    GovFinal4ScenarioEvidence(
        "O",
        "Human APPROVED + fresh ALLOW → resume once",
        "exactly one protected operation",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_MP4R7, "test_mp4r7_success_e2e"),),
    ),
    GovFinal4ScenarioEvidence(
        "P",
        "Human APPROVED + fresh DENY",
        "no resume effect",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_MP4R7, "test_mp4r7_human_approve_governance_deny_prevents_continuation_and_operation"),),
    ),
    GovFinal4ScenarioEvidence(
        "Q",
        "Human REJECT",
        "no unauthorized resume",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_MP4R7, "test_mp4r7_human_reject_e2e"),),
    ),
    GovFinal4ScenarioEvidence(
        "R",
        "Stale / mismatched grant",
        "0 effect",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_PG_C, "test_c10_wrong_policy_bundle_blocks_provider"),
            _nid(_PG_C, "test_c11_fresh_deny_over_grant_blocks_provider"),
            _nid(_G5C2, "test_task_mismatch_fails_closed"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "S",
        "Grant consumed once",
        "cannot reuse for second effect",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_PG_C, "test_c14_grant_consumed_before_provider_callback"),),
    ),
    GovFinal4ScenarioEvidence(
        "T",
        "Governance ALLOW → Reliability intent before mutation",
        "durable SUCCESS path",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR7_A3, "test_success_persists_intent_before_outcome_and_ger"),),
    ),
    GovFinal4ScenarioEvidence(
        "U",
        "Governance DENY → no ProviderInvocation intent",
        "0 provider mutation",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(
                "applications/governed_contractor_application/tests/host/test_gr7_a2_external_work_erl_bridge.py",
                "test_governance_deny_zero_provider_calls_no_admission",
            ),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "V",
        "Provider UNKNOWN durable truth",
        "UNKNOWN ≠ DENY",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR7_A4, "test_map_provider_invocation_outcome_status_to_host_state"),),
    ),
    GovFinal4ScenarioEvidence(
        "W",
        "UNKNOWN reconciliation read-only",
        "no blind retry",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR7_A6, "test_accept_unknown_still_unknown_when_quote_not_accepted"),),
    ),
    GovFinal4ScenarioEvidence(
        "X",
        "Recovery / repeat eligibility",
        "same logical identity, new physical invocation id",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR7_A7, "test_repeat_creates_new_physical_invocation_id"),),
    ),
    GovFinal4ScenarioEvidence(
        "Y",
        "Crash ambiguity (intent without outcome)",
        "fail-safe recovery path",
        GovFinal4ScenarioResult.PARTIAL,
        (_nid(_GR7_A4, "test_crash_ambiguity_intent_without_outcome_not_explicit_unknown"),),
        notes="Host-qualified; not all strategies.",
    ),
    GovFinal4ScenarioEvidence(
        "Z",
        "Cross-tenant fail closed",
        "0 effect",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_MP4R7, "test_mp4r7_cross_tenant_fail_closed"),
            _nid(_GR6_R1, "test_tenant_aware_requirement"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "RB",
        "Resource binding coherence",
        "mismatch → 0 effect",
        GovFinal4ScenarioResult.QUALIFIED,
        (_nid(_GR6_RS1, "test_wrong_resource_denied_at_boundary_no_effect"),),
    ),
    GovFinal4ScenarioEvidence(
        "POB",
        "Provider operation binding SSOT",
        "external_work.* → canonical ops",
        GovFinal4ScenarioResult.QUALIFIED,
        (
            _nid(_GR6_ACTION, "test_external_work_actions_are_valid_decision_execution_action_kinds"),
            _nid(_GR6_ACTION, "test_decision_material_bound_action_matches_side_effect_action"),
        ),
    ),
    GovFinal4ScenarioEvidence(
        "CP",
        "Control-plane mutation",
        "NOT QUALIFIED — GR-12 OPEN",
        GovFinal4ScenarioResult.GAP,
        (),
        notes="NOT QUALIFIED — GR-12 OPEN per architecture SSOT.",
    ),
)


@dataclass(frozen=True, slots=True)
class GovFinal4FailureEvidence:
    failure_window: str
    expected_behavior: str
    effect_called: bool
    durable_truth: str
    pytest_node_ids: tuple[str, ...]


GOV_FINAL_4_FAILURE_CATALOG: tuple[GovFinal4FailureEvidence, ...] = (
    GovFinal4FailureEvidence(
        "policy evaluation exception",
        "fail closed DENY",
        False,
        "none",
        (_nid(_GR6_R1, "test_policy_failure_fails_closed"),),
    ),
    GovFinal4FailureEvidence(
        "inner guard violation",
        "fail closed",
        False,
        "none",
        (_nid(_GR3, "test_no_active_execution_blocks_effect"),),
    ),
    GovFinal4FailureEvidence(
        "provider invocation intent persistence failure",
        "no provider mutation",
        False,
        "none",
        (_nid(_GR7_A3, "test_intent_persistence_failure_zero_provider_calls"),),
    ),
    GovFinal4FailureEvidence(
        "provider exception after grant",
        "grant consumed; no success claim",
        False,
        "none",
        (_nid(_PG_C, "test_c13_provider_failure_leaves_grant_consumed"),),
    ),
    GovFinal4FailureEvidence(
        "outcome persistence failure",
        "primary error preserved",
        False,
        "ambiguous / recovery",
        (_nid(_GR7_A3, "test_outcome_persistence_failure_single_provider_call"),),
    ),
    GovFinal4FailureEvidence(
        "reconciliation probe failure",
        "STILL_UNKNOWN / no mutation",
        False,
        "UNKNOWN",
        (_nid(_GR7_A6, "test_probe_read_failure_not_confirmed_failure"),),
    ),
    GovFinal4FailureEvidence(
        "evidence / observer failure",
        "permission unchanged",
        False,
        "none",
        (_nid(_MP4R7, "test_mp4r7_evidence_failure_preserves_primary"),),
    ),
)


def scenario_ids_a_through_z() -> frozenset[str]:
    return frozenset(entry.scenario_id for entry in GOV_FINAL_4_SCENARIO_CATALOG if entry.scenario_id.isalpha())
