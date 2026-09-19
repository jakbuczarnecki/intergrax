# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 scenario catalog (import/map only)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.memory_behavior.contracts import BehaviorScenarioCategory


@dataclass(frozen=True, slots=True)
class MemAudit6ScenarioRef:
    scenario_id: str
    category: BehaviorScenarioCategory
    pytest_node: str


def _node(module: str, test_name: str) -> str:
    return f"tests/qualification/memory_behavior/{module}::{test_name}"


MEM_AUDIT_6_SCENARIOS: tuple[MemAudit6ScenarioRef, ...] = (
    MemAudit6ScenarioRef("USER-01", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_01_basic_remember_recall")),
    MemAudit6ScenarioRef("USER-02", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_02_irrelevant_memory_not_displacing")),
    MemAudit6ScenarioRef("USER-03", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_03_top_k_respected")),
    MemAudit6ScenarioRef("USER-04", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_04_empty_query_no_semantic_recall")),
    MemAudit6ScenarioRef("USER-05", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_05_disabled_semantic_fallback")),
    MemAudit6ScenarioRef("USER-06", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_06_forget_hard_gate")),
    MemAudit6ScenarioRef("USER-09", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_09_supersession_lineage")),
    MemAudit6ScenarioRef("USER-10", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_10_supersession_recall_prefers_new")),
    MemAudit6ScenarioRef("USER-11", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_11_self_supersession_rejected")),
    MemAudit6ScenarioRef("USER-12", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_12_supersession_missing_target")),
    MemAudit6ScenarioRef("USER-16", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_16_provenance_preserved")),
    MemAudit6ScenarioRef("USER-18", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_18_governance_deny_zero_side_effects")),
    MemAudit6ScenarioRef("USER-20", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_20_conflicting_facts_unresolved")),
    MemAudit6ScenarioRef("USER-21", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_21_supersession_resolves_conflict")),
    MemAudit6ScenarioRef("USER-22", BehaviorScenarioCategory.USER, _node("test_mem_final_audit_6_user_behavior.py", "test_user_22_deterministic_ordering")),
    MemAudit6ScenarioRef("SEC-01", BehaviorScenarioCategory.SECURITY, _node("test_mem_final_audit_6_security_behavior.py", "test_identity_user_spoof_denied")),
    MemAudit6ScenarioRef("SEC-02", BehaviorScenarioCategory.SECURITY, _node("test_mem_final_audit_6_security_behavior.py", "test_identity_tenant_spoof_denied")),
    MemAudit6ScenarioRef("USER-23", BehaviorScenarioCategory.SECURITY, _node("test_mem_final_audit_6_security_behavior.py", "test_cross_user_isolation")),
    MemAudit6ScenarioRef("USER-24", BehaviorScenarioCategory.SECURITY, _node("test_mem_final_audit_6_security_behavior.py", "test_cross_tenant_isolation")),
    MemAudit6ScenarioRef("SEC-03", BehaviorScenarioCategory.SECURITY, _node("test_mem_final_audit_6_security_behavior.py", "test_cross_tenant_shared_backing_reverse_direction")),
    MemAudit6ScenarioRef(
        "HARNESS-01",
        BehaviorScenarioCategory.HARNESS_INTEGRITY,
        _node("test_mem_final_audit_6r_harness.py", "test_behavior_runner_aggregates_real_violation_ledger"),
    ),
    MemAudit6ScenarioRef("P-01", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, _node("test_mem_final_audit_6_projection_behavior.py", "test_p01_partial_projection_remember")),
    MemAudit6ScenarioRef("P-02", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, _node("test_mem_final_audit_6_projection_behavior.py", "test_p02_partial_projection_forget")),
    MemAudit6ScenarioRef("P-03", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, _node("test_mem_final_audit_6_projection_behavior.py", "test_p03_reconciliation_repair")),
    MemAudit6ScenarioRef("P-04", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, _node("test_mem_final_audit_6_projection_behavior.py", "test_p04_reconciliation_idempotent")),
    MemAudit6ScenarioRef("USER-08", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, _node("test_mem_final_audit_6_projection_behavior.py", "test_user_08_stale_vector_after_delete")),
    MemAudit6ScenarioRef("P-05", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, _node("test_mem_final_audit_6_projection_behavior.py", "test_p05_reconciliation_projection_failure")),
    MemAudit6ScenarioRef("SESSION-01", BehaviorScenarioCategory.SESSION, _node("test_mem_final_audit_6_session_behavior.py", "test_session_01_basic_episodic_recall")),
    MemAudit6ScenarioRef("SESSION-02", BehaviorScenarioCategory.SESSION, _node("test_mem_final_audit_6_session_behavior.py", "test_session_02_session_isolation")),
    MemAudit6ScenarioRef("SESSION-03", BehaviorScenarioCategory.SESSION, _node("test_mem_final_audit_6_session_behavior.py", "test_session_03_cross_session_when_enabled")),
    MemAudit6ScenarioRef("SESSION-05", BehaviorScenarioCategory.SESSION, _node("test_mem_final_audit_6_session_behavior.py", "test_session_05_scope_authority_on_recall")),
    MemAudit6ScenarioRef(
        "TASK-01",
        BehaviorScenarioCategory.TASK,
        _node("test_mem_final_audit_6_task_behavior.py", "test_task_01_remember_and_capability_read"),
    ),
    MemAudit6ScenarioRef("TASK-02", BehaviorScenarioCategory.TASK, _node("test_mem_final_audit_6_task_behavior.py", "test_task_02_forget")),
    MemAudit6ScenarioRef("TASK-03", BehaviorScenarioCategory.TASK, _node("test_mem_final_audit_6_task_behavior.py", "test_task_03_namespace_isolation")),
    MemAudit6ScenarioRef("TASK-05", BehaviorScenarioCategory.TASK, _node("test_mem_final_audit_6_task_behavior.py", "test_task_05_tenant_scope_on_task")),
)
