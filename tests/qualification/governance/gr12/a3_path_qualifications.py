# © Artur Czarnecki. All rights reserved.

"""GR-12-A3 execution-proof references for core control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from intergrax.agent_distribution.control_plane_governance import (
    MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION,
    MUTATION_TYPE_ADMIT_RUNTIME_REVISION,
    MUTATION_TYPE_BIND_AGENT,
    MUTATION_TYPE_BUILD_RUNTIME_REVISION,
    MUTATION_TYPE_COMPLETE_DRAIN,
    MUTATION_TYPE_DISABLE_BINDING,
    MUTATION_TYPE_ENABLE_BINDING,
    MUTATION_TYPE_INSTALL_AGENT,
    MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE,
    MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION,
    MUTATION_TYPE_UPDATE_BINDING_CONFIG,
)


@dataclass(frozen=True, slots=True)
class Gr12A3PathProofBundle:
    path_id: str
    allow: tuple[str, ...]
    deny: tuple[str, ...]
    tenant: tuple[str, ...]
    stale: tuple[str, ...]
    hitl: tuple[str, ...]
    evidence: tuple[str, ...]
    pluginability: str | None = None

    @property
    def primary_proof(self) -> str:
        return self.allow[0]

    def all_proof_nodes(self) -> tuple[str, ...]:
        nodes: list[str] = []
        for group in (
            self.allow,
            self.deny,
            self.tenant,
            self.stale,
            self.hitl,
            self.evidence,
        ):
            nodes.extend(group)
        if self.pluginability:
            nodes.append(self.pluginability)
        return tuple(nodes)


GR12_A3_AD_PATH_TO_MUTATION_TYPE: Final[dict[str, str]] = {
    "CP-AD-ACTIVATE": MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION,
    "CP-AD-ROLLBACK": MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION,
    "CP-AD-INSTALL": MUTATION_TYPE_INSTALL_AGENT,
    "CP-AD-BIND": MUTATION_TYPE_BIND_AGENT,
    "CP-AD-BINDING-CONFIG": MUTATION_TYPE_UPDATE_BINDING_CONFIG,
    "CP-AD-ENABLE-BINDING": MUTATION_TYPE_ENABLE_BINDING,
    "CP-AD-DISABLE-BINDING": MUTATION_TYPE_DISABLE_BINDING,
    "CP-AD-ADMIT": MUTATION_TYPE_ADMIT_RUNTIME_REVISION,
    "CP-AD-BUILD": MUTATION_TYPE_BUILD_RUNTIME_REVISION,
    "CP-AD-DRAIN": MUTATION_TYPE_COMPLETE_DRAIN,
    "CP-AD-POST-CUTOVER-FAIL": MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE,
}

_AD_CP = "tests/unit/agent_distribution/test_agent_distribution_control_plane_governance.py"
_AD_DS = "tests/unit/agent_distribution/test_agent_distribution_desired_state_remediation.py"
_AD_ACT = "tests/unit/agent_distribution/test_agent_distribution_activation_remediation.py"
_AD_BUILD = "tests/unit/agent_distribution/test_agent_distribution_build_remediation.py"
_AD_DRAIN = "tests/unit/agent_distribution/test_agent_distribution_drain_recovery_remediation.py"
_AD_TENANT = "tests/unit/agent_distribution/test_agent_distribution_tenant_denial_remediation.py"

_AHI = "tests/unit/runtime/adaptive/test_ahi_control_plane_governance.py"
_ECP = "tests/unit/runtime/capacity/test_ecp_control_plane_governance.py"
_TC_CANCEL = "tests/unit/applications/test_task_control_governed_cancel.py"
_TC_RESUME = "tests/unit/applications/test_task_control_governed_resume.py"
_TC_AUTONOMY = "tests/unit/applications/test_task_control_governed_autonomy.py"
_TC_HOST = "tests/unit/applications/test_task_control_host_composition.py"
_A2 = "tests/qualification/governance/gr12/test_gr12_a2_mandatory_cla04_composition.py"

GR12_A3_CORE_PATH_PROOFS: Final[tuple[Gr12A3PathProofBundle, ...]] = (
    Gr12A3PathProofBundle(
        path_id="CP-AD-ACTIVATE",
        allow=(f"{_AD_CP}::test_ad1_activation_allow_commits_once",),
        deny=(f"{_AD_CP}::test_ad2_activation_deny_zero_commits",),
        tenant=(f"{_AD_CP}::test_ad7_wrong_tenant_authority_denies_without_mutation",),
        stale=(f"{_AD_CP}::test_ad8_stale_state_cas_rejects_after_allow",),
        hitl=(f"{_AD_CP}::test_ad3_activation_require_human_zero_commits_preserves_scope",),
        evidence=(f"{_AD_CP}::test_ad10_mutation_id_stable_on_retry",),
        pluginability=f"{_AD_ACT}::test_r2_tenant_scoped_evaluator_delegates_via_typed_protocol",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-ROLLBACK",
        allow=(f"{_AD_CP}::test_ad4_rollback_allow_commits_once",),
        deny=(f"{_AD_CP}::test_ad5_rollback_deny_zero_commits",),
        tenant=(f"{_AD_TENANT}::test_te1_wrong_tenant_install_blocked_before_lookup",),
        stale=(f"{_AD_ACT}::test_adr14_cas_regression_after_allow",),
        hitl=(f"{_AD_CP}::test_ad6_rollback_require_human_zero_commits",),
        evidence=(f"{_AD_CP}::test_ad11_activation_and_rollback_digest_distinct",),
        pluginability=f"{_AD_ACT}::test_r2_tenant_scoped_evaluator_delegates_via_typed_protocol",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-INSTALL",
        allow=(f"{_AD_DS}::test_ads1_install_allow_one_mutation_sequence",),
        deny=(f"{_AD_DS}::test_ads2_install_deny_zero_mutations",),
        tenant=(f"{_AD_DS}::test_ads4_install_tenant_mismatch_zero_mutations",),
        stale=(f"{_AD_DS}::test_ads12_update_config_cas_after_authorization",),
        hitl=(f"{_AD_DS}::test_ads3_install_require_human_zero_mutations_preserves_scope",),
        evidence=(f"{_AD_DS}::test_ads23_mutation_id_preserved_exactly",),
        pluginability=f"{_AD_DS}::test_ads18_no_policy_tenant_match_deny",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-BIND",
        allow=(f"{_AD_DS}::test_ads6_bind_allow_one_create",),
        deny=(f"{_AD_DS}::test_ads7_bind_deny_zero_create",),
        tenant=(f"{_AD_TENANT}::test_te2_wrong_tenant_bind_blocked_before_lookup",),
        stale=(f"{_AD_DS}::test_ads12_update_config_cas_after_authorization",),
        hitl=(f"{_AD_DS}::test_ads3_install_require_human_zero_mutations_preserves_scope",),
        evidence=(f"{_AD_DS}::test_ads8_bind_target_identity_changes_request_digest",),
        pluginability=f"{_AD_DS}::test_ads18_no_policy_tenant_match_deny",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-BINDING-CONFIG",
        allow=(f"{_AD_DS}::test_ads9_update_config_allow_one_update",),
        deny=(f"{_AD_DS}::test_ads10_update_config_deny_zero_updates",),
        tenant=(f"{_AD_TENANT}::test_te3_wrong_tenant_update_enable_disable_blocked_before_lookup",),
        stale=(f"{_AD_DS}::test_ads12_update_config_cas_after_authorization",),
        hitl=(f"{_AD_DS}::test_ads3_install_require_human_zero_mutations_preserves_scope",),
        evidence=(f"{_AD_DS}::test_ads13_config_digest_changes_target_request_digest",),
        pluginability=f"{_AD_DS}::test_ads18_no_policy_tenant_match_deny",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-ENABLE-BINDING",
        allow=(f"{_AD_DS}::test_ads14_enable_allow_once",),
        deny=(f"{_AD_DS}::test_ads15_enable_deny_zero_mutation",),
        tenant=(f"{_AD_TENANT}::test_te3_wrong_tenant_update_enable_disable_blocked_before_lookup",),
        stale=(f"{_AD_DS}::test_ads12_update_config_cas_after_authorization",),
        hitl=(f"{_AD_DS}::test_ads3_install_require_human_zero_mutations_preserves_scope",),
        evidence=(f"{_AD_DS}::test_ads17_enable_vs_disable_different_digest",),
        pluginability=f"{_AD_DS}::test_ads18_no_policy_tenant_match_deny",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-DISABLE-BINDING",
        allow=(f"{_AD_DS}::test_ads16_disable_allow_once",),
        deny=(f"{_AD_DS}::test_ads15_enable_deny_zero_mutation",),
        tenant=(f"{_AD_TENANT}::test_te3_wrong_tenant_update_enable_disable_blocked_before_lookup",),
        stale=(f"{_AD_DS}::test_ads12_update_config_cas_after_authorization",),
        hitl=(f"{_AD_DS}::test_ads3_install_require_human_zero_mutations_preserves_scope",),
        evidence=(f"{_AD_DS}::test_ads17_enable_vs_disable_different_digest",),
        pluginability=f"{_AD_DS}::test_ads18_no_policy_tenant_match_deny",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-ADMIT",
        allow=(f"{_AD_BUILD}::test_r1_7_reference_admission_allow_persists_revision",),
        deny=(f"{_AD_BUILD}::test_r1_8_reference_admission_deny_zero_revision_writes",),
        tenant=(f"{_AD_BUILD}::test_adb4_tenant_mismatch_before_revision_lookup",),
        stale=(f"{_AD_BUILD}::test_adb17_concurrent_revision_claim_domain_conflict",),
        hitl=(f"{_AD_BUILD}::test_r1_9_reference_admission_require_human_zero_writes",),
        evidence=(f"{_AD_BUILD}::test_adb20_authorization_evidence_binds_request",),
        pluginability=f"{_AD_BUILD}::test_adb6_missing_policy_deny_zero_writes",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-BUILD",
        allow=(f"{_AD_BUILD}::test_adb1_build_allow_persists_once",),
        deny=(f"{_AD_BUILD}::test_adb2_build_deny_zero_writes",),
        tenant=(f"{_AD_BUILD}::test_adb4_tenant_mismatch_before_revision_lookup",),
        stale=(f"{_AD_BUILD}::test_adb17_concurrent_revision_claim_domain_conflict",),
        hitl=(f"{_AD_BUILD}::test_adb3_require_human_zero_writes_preserves_scope",),
        evidence=(f"{_AD_BUILD}::test_adb20_authorization_evidence_binds_request",),
        pluginability=f"{_AD_BUILD}::test_adb7_policy_failure_deny_zero_writes",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-DRAIN",
        allow=(f"{_AD_DRAIN}::test_dr1_complete_drain_allow_stops_once",),
        deny=(f"{_AD_DRAIN}::test_dr2_complete_drain_deny_zero_effects",),
        tenant=(f"{_AD_DRAIN}::test_dr4_drain_tenant_mismatch_blocked",),
        stale=(f"{_AD_DRAIN}::test_dr7_drain_record_revision_binding",),
        hitl=(f"{_AD_DRAIN}::test_dr3_complete_drain_require_human_zero_effects",),
        evidence=(f"{_AD_DRAIN}::test_dr8_drain_policy_identity_changes_digest",),
        pluginability=f"{_AD_DRAIN}::test_dr5_drain_missing_policy_denies",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AD-POST-CUTOVER-FAIL",
        allow=(f"{_AD_DRAIN}::test_dr11_failure_mark_is_control_plane_mutation",),
        deny=(f"{_AD_DRAIN}::test_dr12_failure_mark_wrong_tenant_denies",),
        tenant=(f"{_AD_DRAIN}::test_dr12_failure_mark_wrong_tenant_denies",),
        stale=(f"{_AD_DRAIN}::test_dr9_drain_timeout_mark_failed_after_authorization",),
        hitl=(f"{_AD_DRAIN}::test_dr3_complete_drain_require_human_zero_effects",),
        evidence=(f"{_AD_DRAIN}::test_dr18_failure_evidence_ref_not_authority",),
        pluginability=f"{_AD_DRAIN}::test_dr6_drain_policy_failure_denies",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AHI-APPLY",
        allow=(f"{_AHI}::test_ahicpm1_apply_allow_executes_with_evidence",),
        deny=(f"{_AHI}::test_ahicpm2_apply_deny_has_zero_writes",),
        tenant=(f"{_AHI}::test_ahicpm4_wrong_tenant_blocked_before_write",),
        stale=(f"{_AHI}::test_ahicpm8_stale_state_conflict_does_not_overwrite_pointer",),
        hitl=(f"{_AHI}::test_ahicpm3_apply_require_human_has_zero_effects",),
        evidence=(f"{_AHI}::test_ahicpm6_caller_mutation_id_in_evidence",),
        pluginability=f"{_AHI}::test_ahicpm18_single_cpm_evaluation_per_mutation",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-AHI-ROLLBACK",
        allow=(f"{_AHI}::test_ahicpm10_rollback_allow_uses_canonical_previous",),
        deny=(f"{_AHI}::test_ahicpm11_rollback_deny_zero_effects",),
        tenant=(f"{_AHI}::test_ahicpm4_wrong_tenant_blocked_before_write",),
        stale=(f"{_AHI}::test_ahicpm22_rollback_mid_flight_conflict_zero_partial_writes",),
        hitl=(f"{_AHI}::test_ahicpm12_rollback_require_human_zero_effects",),
        evidence=(f"{_AHI}::test_ahicpm16_separate_rollback_mutation_id",),
        pluginability=f"{_AHI}::test_ahicpm18_single_cpm_evaluation_per_mutation",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-ECP-SCALE-K8S",
        allow=(f"{_ECP}::test_ecp_cpm1_allow_k8s_exact_target_and_evidence",),
        deny=(f"{_ECP}::test_ecp_cpm2_deny_k8s_zero_scale_calls",),
        tenant=(f"{_ECP}::test_ecp_cpm4_wrong_tenant_blocked_before_provider_mutation",),
        stale=(f"{_ECP}::test_ecp_cpm8_stale_k8s_state_blocks_apply",),
        hitl=(f"{_ECP}::test_ecp_cpm3_require_human_k8s_zero_scale_calls",),
        evidence=(f"{_ECP}::test_ecp_cpm6_caller_mutation_id_in_evidence",),
        pluginability=f"{_ECP}::test_ecp_r1_standalone_k8s_allow_without_execution_identity",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-ECP-SCALE-CELERY",
        allow=(f"{_ECP}::test_ecp_cpm9_allow_celery_exact_target_and_evidence",),
        deny=(f"{_ECP}::test_ecp_cpm10_deny_celery_zero_worker_mutation",),
        tenant=(f"{_ECP}::test_ecp_cpm4_wrong_tenant_blocked_before_provider_mutation",),
        stale=(f"{_ECP}::test_ecp_cpm32_stale_state_scheduler_blocked_without_cpm_evidence",),
        hitl=(f"{_ECP}::test_ecp_cpm26_require_human_zero_provider_effect",),
        evidence=(f"{_ECP}::test_ecp_cpm6_caller_mutation_id_in_evidence",),
        pluginability=f"{_ECP}::test_ecp_r1_standalone_k8s_deny_without_execution_identity",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-TASK-CANCEL",
        allow=(f"{_TC_CANCEL}::test_taskcpm_c1_allow_matching_binding_requests_cancel_once",),
        deny=(f"{_TC_CANCEL}::test_taskcpm_c2_deny_zero_cancellation_effect",),
        tenant=(f"{_TC_CANCEL}::test_taskcpm_c4_wrong_tenant_zero_cancellation_effect",),
        stale=(f"{_TC_CANCEL}::test_taskcpm_c6_binding_changes_after_authorization_zero_effect",),
        hitl=(f"{_TC_CANCEL}::test_taskcpm_c3_require_human_zero_cancellation_effect",),
        evidence=(f"{_TC_CANCEL}::test_taskcpm_c10_authorization_evidence_binds_exact_scope",),
        pluginability=f"{_A2}::test_gr12_a2_external_evaluator_injected_without_domain_changes",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-TASK-RESUME",
        allow=(f"{_TC_RESUME}::test_taskcpm_r1_allow_exact_checkpoint_invokes_runner_once",),
        deny=(f"{_TC_RESUME}::test_taskcpm_r7_deny_zero_runner_invocation",),
        tenant=(f"{_TC_RESUME}::test_taskcpm_r6_wrong_tenant_zero_runner_invocation",),
        stale=(f"{_TC_RESUME}::test_taskcpm_r16_resume_token_stale_zero_runner",),
        hitl=(f"{_TC_RESUME}::test_taskcpm_r8_require_human_zero_runner_invocation",),
        evidence=(f"{_TC_RESUME}::test_taskcpm_r2_caller_mutation_id_preserved",),
        pluginability=f"{_A2}::test_gr12_a2_external_evaluator_injected_without_domain_changes",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-TASK-AUTONOMY",
        allow=(f"{_TC_AUTONOMY}::test_taskcpm_a1_allow_matching_binding_changes_autonomy_once",),
        deny=(f"{_TC_AUTONOMY}::test_taskcpm_a8_deny_zero_mutation_with_evidence",),
        tenant=(f"{_TC_AUTONOMY}::test_taskcpm_a6_wrong_tenant_zero_mutation",),
        stale=(f"{_TC_AUTONOMY}::test_taskcpm_a12_current_autonomy_changes_after_allow_zero_mutation",),
        hitl=(f"{_TC_AUTONOMY}::test_taskcpm_a9_require_human_zero_mutation_with_scope",),
        evidence=(f"{_TC_AUTONOMY}::test_taskcpm_a10b_authorization_evidence_binds_exact_scope",),
        pluginability=f"{_TC_AUTONOMY}::test_taskcpm_a19_product_host_uses_canonical_bundle_authority",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-REF-ACTIVATE",
        allow=(f"{_AD_ACT}::test_adr7_reference_production_allow_commits",),
        deny=(f"{_AD_ACT}::test_adr8_reference_production_deny_zero_commit",),
        tenant=(f"{_AD_ACT}::test_adr10_reference_production_wrong_tenant_zero_commit",),
        stale=(f"{_AD_ACT}::test_adr14_cas_regression_after_allow",),
        hitl=(f"{_AD_ACT}::test_adr11_reference_production_require_human_zero_commit",),
        evidence=(f"{_AD_ACT}::test_adr12_reference_path_uses_distinct_admission_and_activation_mutation_ids",),
        pluginability=f"{_AD_ACT}::test_r2_tenant_scoped_evaluator_delegates_via_typed_protocol",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-HOST-BOUNDARY-OPTIONAL",
        allow=(f"{_A2}::test_gr12_a2_product_task_control_wiring_requires_canonical_boundary",),
        deny=(f"{_A2}::test_gr12_a2_ecp_product_without_authority_fails_at_wiring",),
        tenant=(f"{_TC_HOST}::test_taskcpm_h1_product_host_runtime_exposes_canonical_boundary",),
        stale=(f"{_TC_CANCEL}::test_taskcpm_c6_binding_changes_after_authorization_zero_effect",),
        hitl=(f"{_TC_CANCEL}::test_taskcpm_c3_require_human_zero_cancellation_effect",),
        evidence=(f"{_A2}::test_gr12_a2_r2_product_host_composition_exposes_cla04_boundary",),
        pluginability=f"{_A2}::test_gr12_a2_external_evaluator_injected_without_domain_changes",
    ),
    Gr12A3PathProofBundle(
        path_id="CP-ECP-BOUNDARY-OPTIONAL",
        allow=(f"{_ECP}::test_ecp_cpm16_product_without_authority_fails_at_wiring",),
        deny=(f"{_A2}::test_gr12_a2_ecp_enabled_without_boundary_fails_at_wiring",),
        tenant=(f"{_ECP}::test_ecp_cpm4_wrong_tenant_blocked_before_provider_mutation",),
        stale=(f"{_ECP}::test_ecp_cpm8_stale_k8s_state_blocks_apply",),
        hitl=(f"{_ECP}::test_ecp_cpm3_require_human_k8s_zero_scale_calls",),
        evidence=(f"{_ECP}::test_ecp_cpm15_production_supplied_deny_policy_zero_provider_effect",),
        pluginability=f"{_ECP}::test_ecp_cpm14_production_missing_policy_fails_closed",
    ),
)


GR12_A3_QUALIFIED_PATH_IDS: Final[frozenset[str]] = frozenset(
    bundle.path_id for bundle in GR12_A3_CORE_PATH_PROOFS
)


def gr12_a3_proof_bundle(path_id: str) -> Gr12A3PathProofBundle:
    for bundle in GR12_A3_CORE_PATH_PROOFS:
        if bundle.path_id == path_id:
            return bundle
    raise KeyError(path_id)
