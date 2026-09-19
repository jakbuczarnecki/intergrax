# © Artur Czarnecki. All rights reserved.

"""Explicit registry of real behavioral qualification cases (aggregate runner source of truth)."""

from __future__ import annotations

from tests.qualification.memory_behavior.contracts import (
    BehaviorEvalCase,
    BehaviorGateKind,
    BehaviorScenarioCategory,
)
from tests.qualification.memory_behavior.scenarios.projection import (
    run_p01_partial_projection_remember,
    run_p02_partial_projection_forget,
    run_p03_reconciliation_repair,
    run_p04_reconciliation_idempotent,
    run_p05_reconciliation_projection_failure,
    run_user_08_stale_vector_after_delete,
)
from tests.qualification.memory_behavior.scenarios.security import (
    run_sec_01_identity_user_spoof_denied,
    run_sec_02_identity_tenant_spoof_denied,
    run_sec_03_cross_tenant_shared_backing_reverse_direction,
    run_user_23_cross_user_isolation,
    run_user_24_cross_tenant_isolation,
)
from tests.qualification.memory_behavior.scenarios.session import (
    run_session_01_basic_episodic_recall,
    run_session_02_session_isolation,
    run_session_03_cross_session_when_enabled,
    run_session_05_scope_authority_on_recall,
)
from tests.qualification.memory_behavior.scenarios.task import (
    run_task_01_remember_and_capability_read,
    run_task_02_forget,
    run_task_03_namespace_isolation,
    run_task_05_tenant_scope_on_task,
)
from tests.qualification.memory_behavior.scenarios.user import (
    run_user_01_basic_remember_recall,
    run_user_02_irrelevant_memory_not_displacing,
    run_user_03_top_k_respected,
    run_user_04_empty_query_no_semantic_recall,
    run_user_05_disabled_semantic_fallback,
    run_user_06_forget_hard_gate,
    run_user_09_supersession_lineage,
    run_user_10_supersession_recall_prefers_new,
    run_user_11_self_supersession_rejected,
    run_user_12_supersession_missing_target,
    run_user_16_provenance_preserved,
    run_user_18_governance_deny_zero_side_effects,
    run_user_20_conflicting_facts_unresolved,
    run_user_21_supersession_resolves_conflict,
    run_user_22_deterministic_ordering,
)

MEM_AUDIT_6_BEHAVIOR_CASES: tuple[BehaviorEvalCase, ...] = (
    BehaviorEvalCase("USER-01", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_01_basic_remember_recall),
    BehaviorEvalCase("USER-02", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_02_irrelevant_memory_not_displacing),
    BehaviorEvalCase("USER-03", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_03_top_k_respected),
    BehaviorEvalCase("USER-04", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_04_empty_query_no_semantic_recall),
    BehaviorEvalCase("USER-05", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_05_disabled_semantic_fallback),
    BehaviorEvalCase("USER-06", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_06_forget_hard_gate),
    BehaviorEvalCase("USER-09", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_09_supersession_lineage),
    BehaviorEvalCase("USER-10", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_10_supersession_recall_prefers_new),
    BehaviorEvalCase("USER-11", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_11_self_supersession_rejected),
    BehaviorEvalCase("USER-12", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_12_supersession_missing_target),
    BehaviorEvalCase("USER-16", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_16_provenance_preserved),
    BehaviorEvalCase("USER-18", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_18_governance_deny_zero_side_effects),
    BehaviorEvalCase("USER-20", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_20_conflicting_facts_unresolved),
    BehaviorEvalCase("USER-21", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_21_supersession_resolves_conflict),
    BehaviorEvalCase("USER-22", BehaviorScenarioCategory.USER, BehaviorGateKind.HARD, run_user_22_deterministic_ordering),
    BehaviorEvalCase("SEC-01", BehaviorScenarioCategory.SECURITY, BehaviorGateKind.HARD, run_sec_01_identity_user_spoof_denied),
    BehaviorEvalCase("SEC-02", BehaviorScenarioCategory.SECURITY, BehaviorGateKind.HARD, run_sec_02_identity_tenant_spoof_denied),
    BehaviorEvalCase("USER-23", BehaviorScenarioCategory.SECURITY, BehaviorGateKind.HARD, run_user_23_cross_user_isolation),
    BehaviorEvalCase("USER-24", BehaviorScenarioCategory.SECURITY, BehaviorGateKind.HARD, run_user_24_cross_tenant_isolation),
    BehaviorEvalCase("SEC-03", BehaviorScenarioCategory.SECURITY, BehaviorGateKind.HARD, run_sec_03_cross_tenant_shared_backing_reverse_direction),
    BehaviorEvalCase("P-01", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, BehaviorGateKind.HARD, run_p01_partial_projection_remember),
    BehaviorEvalCase("P-02", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, BehaviorGateKind.HARD, run_p02_partial_projection_forget),
    BehaviorEvalCase("P-03", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, BehaviorGateKind.HARD, run_p03_reconciliation_repair),
    BehaviorEvalCase("P-04", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, BehaviorGateKind.HARD, run_p04_reconciliation_idempotent),
    BehaviorEvalCase("USER-08", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, BehaviorGateKind.HARD, run_user_08_stale_vector_after_delete),
    BehaviorEvalCase("P-05", BehaviorScenarioCategory.PROJECTION_LIFECYCLE, BehaviorGateKind.HARD, run_p05_reconciliation_projection_failure),
    BehaviorEvalCase("SESSION-01", BehaviorScenarioCategory.SESSION, BehaviorGateKind.HARD, run_session_01_basic_episodic_recall),
    BehaviorEvalCase("SESSION-02", BehaviorScenarioCategory.SESSION, BehaviorGateKind.HARD, run_session_02_session_isolation),
    BehaviorEvalCase("SESSION-03", BehaviorScenarioCategory.SESSION, BehaviorGateKind.HARD, run_session_03_cross_session_when_enabled),
    BehaviorEvalCase("SESSION-05", BehaviorScenarioCategory.SESSION, BehaviorGateKind.HARD, run_session_05_scope_authority_on_recall),
    BehaviorEvalCase("TASK-01", BehaviorScenarioCategory.TASK, BehaviorGateKind.HARD, run_task_01_remember_and_capability_read),
    BehaviorEvalCase("TASK-02", BehaviorScenarioCategory.TASK, BehaviorGateKind.HARD, run_task_02_forget),
    BehaviorEvalCase("TASK-03", BehaviorScenarioCategory.TASK, BehaviorGateKind.HARD, run_task_03_namespace_isolation),
    BehaviorEvalCase("TASK-05", BehaviorScenarioCategory.TASK, BehaviorGateKind.HARD, run_task_05_tenant_scope_on_task),
)
