# © Artur Czarnecki. All rights reserved.

"""UE-11A — typed Unified Execution validation matrix and gate helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_ALLOWED_PROOF_PREFIXES: tuple[str, ...] = ("tests/",)


class ValidationDomain(Enum):
    ROOT_STRATEGY = "ROOT_STRATEGY"
    LIFECYCLE = "LIFECYCLE"
    FAIL_CLOSED = "FAIL_CLOSED"
    CHILD_EXECUTION = "CHILD_EXECUTION"
    CONCURRENCY = "CONCURRENCY"
    RECOVERY = "RECOVERY"
    OBSERVABILITY = "OBSERVABILITY"
    DIAGNOSTICS = "DIAGNOSTICS"
    ANTI_BYPASS = "ANTI_BYPASS"
    PRODUCTION_SCENARIO = "PRODUCTION_SCENARIO"


class ValidationStatus(Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    GAP = "gap"


class ValidationProofKind(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    ACCEPTANCE = "acceptance"
    ARCHITECTURE_GATE = "architecture_gate"


class GapTarget(Enum):
    UE_11B = "UE-11B"
    UE_11C = "UE-11C"
    UE_11D = "UE-11D"
    UE_11E = "UE-11E"
    UE_11F = "UE-11F"
    UE_11G = "UE-11G"


@dataclass(frozen=True, slots=True)
class ValidationProof:
    kind: ValidationProofKind
    path: str
    test_name: str


@dataclass(frozen=True, slots=True)
class ValidationCapability:
    capability_id: str
    domain: ValidationDomain
    status: ValidationStatus
    proofs: tuple[ValidationProof, ...]
    gap_target: GapTarget | None = None


def _proof(
    kind: ValidationProofKind,
    path: str,
    test_name: str,
) -> ValidationProof:
    return ValidationProof(kind=kind, path=path, test_name=test_name)


def _unit(path: str, test_name: str) -> ValidationProof:
    return _proof(ValidationProofKind.UNIT, path, test_name)


def _gate(path: str, test_name: str) -> ValidationProof:
    return _proof(ValidationProofKind.ARCHITECTURE_GATE, path, test_name)


def _covered(
    capability_id: str,
    domain: ValidationDomain,
    *proofs: ValidationProof,
) -> ValidationCapability:
    return ValidationCapability(
        capability_id=capability_id,
        domain=domain,
        status=ValidationStatus.COVERED,
        proofs=proofs,
        gap_target=None,
    )


def _partial(
    capability_id: str,
    domain: ValidationDomain,
    gap_target: GapTarget,
    *proofs: ValidationProof,
) -> ValidationCapability:
    return ValidationCapability(
        capability_id=capability_id,
        domain=domain,
        status=ValidationStatus.PARTIAL,
        proofs=proofs,
        gap_target=gap_target,
    )


def _gap(
    capability_id: str,
    domain: ValidationDomain,
    gap_target: GapTarget,
) -> ValidationCapability:
    return ValidationCapability(
        capability_id=capability_id,
        domain=domain,
        status=ValidationStatus.GAP,
        proofs=(),
        gap_target=gap_target,
    )


UNIFIED_EXECUTION_VALIDATION_MATRIX: tuple[ValidationCapability, ...] = (
    # ROOT_STRATEGY — stack proven with probe backends; canonical root E2E deferred to UE-11B.
    _partial(
        "root.inference.end_to_end",
        ValidationDomain.ROOT_STRATEGY,
        GapTarget.UE_11B,
        _unit(
            "tests/unit/runtime/execution/test_execution_runtime.py",
            "test_inference_root_runtime_binds_identity_authority_budget",
        ),
        _unit(
            "tests/unit/runtime/execution/test_strategy_execution_router.py",
            "test_inference_router_delegates_only_to_inference_executor",
        ),
    ),
    _partial(
        "root.agentic.end_to_end",
        ValidationDomain.ROOT_STRATEGY,
        GapTarget.UE_11B,
        _unit(
            "tests/unit/runtime/execution/test_execution_runtime.py",
            "test_agentic_root_runtime_binds_identity_authority_budget",
        ),
        _unit(
            "tests/unit/runtime/execution/test_strategy_execution_router.py",
            "test_agentic_router_delegates_only_to_agent_executor",
        ),
    ),
    _partial(
        "root.orchestration.end_to_end",
        ValidationDomain.ROOT_STRATEGY,
        GapTarget.UE_11B,
        _unit(
            "tests/unit/runtime/execution/test_execution_runtime.py",
            "test_orchestration_root_runtime_nexus_receives_active_context",
        ),
        _unit(
            "tests/unit/runtime/execution/test_strategy_execution_router.py",
            "test_orchestration_router_delegates_only_to_orchestration_executor",
        ),
    ),
    # LIFECYCLE
    _covered(
        "lifecycle.root_execution_id_single",
        ValidationDomain.LIFECYCLE,
        _gate(
            "tests/unit/runtime/architecture/test_ue_10r2_single_canonical_root_execution_id_gate.py",
            "test_execution_runtime_execute_does_not_mint_execution_id",
        ),
        _gate(
            "tests/unit/runtime/architecture/test_ue_10r2_single_canonical_root_execution_id_gate.py",
            "test_runtime_module_mints_execution_id_only_in_mint_root_execution_identity",
        ),
    ),
    _covered(
        "lifecycle.root_execution_id_platform_owned",
        ValidationDomain.LIFECYCLE,
        _gate(
            "tests/unit/runtime/architecture/test_ue_10r3_platform_owned_root_identity_gate.py",
            "test_facade_execute_does_not_accept_root_execution_context",
        ),
        _unit(
            "tests/unit/runtime/execution/test_execution_facade.py",
            "test_facade_mints_platform_execution_id_not_supplied_by_caller",
        ),
    ),
    _covered(
        "lifecycle.identity_consistent_through_backend",
        ValidationDomain.LIFECYCLE,
        _unit(
            "tests/unit/runtime/execution/test_execution_boundary.py",
            "test_boundary_hook_and_delegate_see_exact_identity",
        ),
    ),
    _covered(
        "lifecycle.authority_bound_before_strategy",
        ValidationDomain.LIFECYCLE,
        _unit(
            "tests/unit/runtime/execution/test_execution_facade.py",
            "test_facade_root_budget_and_authority_visible_before_strategy",
        ),
    ),
    _covered(
        "lifecycle.budget_bound_before_strategy",
        ValidationDomain.LIFECYCLE,
        _unit(
            "tests/unit/runtime/execution/test_execution_facade.py",
            "test_facade_root_budget_and_authority_visible_before_strategy",
        ),
    ),
    _covered(
        "lifecycle.runtime_event_same_execution_id",
        ValidationDomain.LIFECYCLE,
        _unit(
            "tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",
            "test_root_execution_event_execution_id",
        ),
    ),
    _covered(
        "lifecycle.parent_child_execution_tree",
        ValidationDomain.LIFECYCLE,
        _unit(
            "tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py",
            "test_child_parent_link",
        ),
        _unit(
            "tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py",
            "test_nested_parent_chain",
        ),
    ),
    # FAIL_CLOSED
    _covered(
        "fail_closed.missing_identity",
        ValidationDomain.FAIL_CLOSED,
        _unit(
            "tests/unit/runtime/execution/test_execution_runtime.py",
            "test_nexus_without_active_identity_fails",
        ),
    ),
    _covered(
        "fail_closed.missing_execution_id",
        ValidationDomain.FAIL_CLOSED,
        _unit(
            "tests/unit/runtime/nexus/execution/test_ue_8ar1_execution_tree_authority.py",
            "test_graph_executor_without_active_execution_id_fails_closed",
        ),
        _unit(
            "tests/unit/runtime/execution/test_child_execution.py",
            "test_child_runner_fails_without_active_execution_id",
        ),
    ),
    _covered(
        "fail_closed.missing_authority",
        ValidationDomain.FAIL_CLOSED,
        _unit(
            "tests/unit/runtime/execution/test_execution_runtime.py",
            "test_nexus_without_active_authority_fails",
        ),
        _unit(
            "tests/unit/runtime/nexus/execution/test_ue_10r4_graph_authority.py",
            "test_graph_executor_without_active_authority_fails_closed",
        ),
    ),
    _covered(
        "fail_closed.missing_budget",
        ValidationDomain.FAIL_CLOSED,
        _unit(
            "tests/unit/runtime/execution/test_execution_runtime.py",
            "test_nexus_without_active_budget_fails",
        ),
        _unit(
            "tests/unit/runtime/execution/budget/test_ue_8b1r2_preserve_run_budget_through_nexus_entry.py",
            "test_active_execution_without_budget_context_fails_closed",
        ),
    ),
    _covered(
        "fail_closed.identity_mismatch",
        ValidationDomain.FAIL_CLOSED,
        _unit(
            "tests/unit/runtime/execution/test_orchestration.py",
            "test_nexus_fails_closed_on_active_identity_mismatch",
        ),
    ),
    _gap(
        "fail_closed.budget_execution_id_mismatch",
        ValidationDomain.FAIL_CLOSED,
        GapTarget.UE_11C,
    ),
    _covered(
        "fail_closed.authority_metadata_mismatch",
        ValidationDomain.FAIL_CLOSED,
        _unit(
            "tests/unit/runtime/nexus/execution/test_ue_10r4_graph_authority.py",
            "test_graph_executor_rejects_task_authority_metadata_mismatch",
        ),
    ),
    _covered(
        "fail_closed.strategy_backend_missing",
        ValidationDomain.FAIL_CLOSED,
        _unit(
            "tests/unit/runtime/execution/test_orchestration.py",
            "test_orchestration_router_fails_closed_when_strategy_is_not_orchestration",
        ),
    ),
    # CHILD_EXECUTION
    _covered(
        "child.unique_execution_id",
        ValidationDomain.CHILD_EXECUTION,
        _unit(
            "tests/unit/runtime/execution/test_graph_executor_child_execution.py",
            "test_parallel_nodes_receive_unique_child_execution_ids",
        ),
    ),
    _covered(
        "child.parent_execution_id_link",
        ValidationDomain.CHILD_EXECUTION,
        _unit(
            "tests/unit/runtime/execution/test_execution_boundary.py",
            "test_boundary_binds_parent_execution_id_for_child_identity",
        ),
    ),
    _covered(
        "child.authority_derived_from_parent",
        ValidationDomain.CHILD_EXECUTION,
        _unit(
            "tests/unit/runtime/nexus/execution/test_ue_8ar1_execution_tree_authority.py",
            "test_child_without_delegation_inherits_parent_authority",
        ),
    ),
    _covered(
        "child.authority_no_escalation",
        ValidationDomain.CHILD_EXECUTION,
        _unit(
            "tests/unit/runtime/execution/authority/test_child_execution_authority_policy.py",
            "test_default_policy_nested_child_overreach_denied",
        ),
    ),
    _covered(
        "child.budget_derived_from_parent",
        ValidationDomain.CHILD_EXECUTION,
        _unit(
            "tests/unit/runtime/execution/budget/test_child_execution_budget.py",
            "test_root_shared_child_shared_nested_shared",
        ),
    ),
    _covered(
        "child.routes_through_boundary",
        ValidationDomain.CHILD_EXECUTION,
        _unit(
            "tests/unit/runtime/execution/test_graph_executor_child_execution.py",
            "test_single_node_runs_as_child_of_orchestration_execution",
        ),
    ),
    _covered(
        "child.routes_through_strategy_router",
        ValidationDomain.CHILD_EXECUTION,
        _unit(
            "tests/unit/runtime/execution/test_graph_executor_child_execution.py",
            "test_child_agentic_execution_routes_through_strategy_router",
        ),
    ),
    # CONCURRENCY
    _partial(
        "concurrency.parallel_root_identity_isolation",
        ValidationDomain.CONCURRENCY,
        GapTarget.UE_11D,
        _unit(
            "tests/unit/runtime/execution/test_execution_boundary.py",
            "test_parallel_boundaries_isolate_execution_ids",
        ),
    ),
    _gap(
        "concurrency.parallel_root_authority_isolation",
        ValidationDomain.CONCURRENCY,
        GapTarget.UE_11D,
    ),
    _partial(
        "concurrency.parallel_root_budget_isolation",
        ValidationDomain.CONCURRENCY,
        GapTarget.UE_11D,
        _unit(
            "tests/unit/runtime/execution/budget/test_ue_8b2_runtime_consumption.py",
            "test_different_runs_isolated",
        ),
    ),
    _covered(
        "concurrency.parallel_child_identity_isolation",
        ValidationDomain.CONCURRENCY,
        _unit(
            "tests/unit/runtime/execution/test_child_execution.py",
            "test_parallel_children_isolate_execution_ids",
        ),
    ),
    _covered(
        "concurrency.parallel_child_budget_isolation",
        ValidationDomain.CONCURRENCY,
        _unit(
            "tests/unit/runtime/execution/budget/test_ue_8b1r1_shared_under_reserved.py",
            "test_parallel_shared_under_reserved_cannot_oversubscribe",
        ),
    ),
    _partial(
        "concurrency.no_context_cross_talk",
        ValidationDomain.CONCURRENCY,
        GapTarget.UE_11D,
        _unit(
            "tests/unit/runtime/background_execution/test_ue_9a_background_identity_redelivery.py",
            "test_contextvar_identity_does_not_leak_between_attempts",
        ),
    ),
    # RECOVERY
    _covered(
        "recovery.retry_execution_semantics",
        ValidationDomain.RECOVERY,
        _unit(
            "tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",
            "test_local_retry_preserves_execution_id",
        ),
    ),
    _covered(
        "recovery.resume_run_continuity",
        ValidationDomain.RECOVERY,
        _unit(
            "tests/unit/runtime/execution/test_execution_runtime.py",
            "test_resume_root_execution_id_matches_identity_through_lifecycle",
        ),
    ),
    _covered(
        "recovery.resume_attempt_semantics",
        ValidationDomain.RECOVERY,
        _unit(
            "tests/unit/runtime/execution/test_orchestration.py",
            "test_resume_checkpoint_preserves_attempt_mints_fresh_execution_id",
        ),
    ),
    _covered(
        "recovery.resume_fresh_execution_id",
        ValidationDomain.RECOVERY,
        _unit(
            "tests/unit/runtime/execution/test_orchestration.py",
            "test_resume_checkpoint_preserves_attempt_mints_fresh_execution_id",
        ),
    ),
    _partial(
        "recovery.resume_execution_tree_continuity",
        ValidationDomain.RECOVERY,
        GapTarget.UE_11E,
        _unit(
            "tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py",
            "test_completed_child_skipped_on_resume",
        ),
    ),
    _covered(
        "recovery.redelivery_run_continuity",
        ValidationDomain.RECOVERY,
        _unit(
            "tests/unit/runtime/background_execution/test_ue_9a_background_identity_redelivery.py",
            "test_redelivery_preserves_run_and_task_but_mints_new_attempt_and_execution",
        ),
    ),
    _covered(
        "recovery.redelivery_attempt_semantics",
        ValidationDomain.RECOVERY,
        _unit(
            "tests/unit/runtime/background_execution/test_ue_9a_background_identity_redelivery.py",
            "test_three_consecutive_redeliveries_keep_run_with_distinct_attempts",
        ),
    ),
    _covered(
        "recovery.redelivery_budget_continuity",
        ValidationDomain.RECOVERY,
        _unit(
            "tests/unit/runtime/execution/budget/test_ue_9ar1_preserve_run_budget_across_redelivery.py",
            "test_attempt_two_has_new_attempt_id_but_same_run_budget_state",
        ),
    ),
    _covered(
        "recovery.redelivery_no_execution_id_reuse",
        ValidationDomain.RECOVERY,
        _unit(
            "tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",
            "test_redelivery_uses_new_execution_id",
        ),
    ),
    # OBSERVABILITY
    _covered(
        "obs.runtime_event_execution_id",
        ValidationDomain.OBSERVABILITY,
        _unit(
            "tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",
            "test_root_execution_event_execution_id",
        ),
    ),
    _covered(
        "obs.runtime_event_attempt_id",
        ValidationDomain.OBSERVABILITY,
        _unit(
            "tests/unit/contracts/test_trace_1b_identity.py",
            "test_runtime_event_rejects_missing_attempt_id",
        ),
        _unit(
            "tests/unit/contracts/test_trace_1b_identity.py",
            "test_emit_context_propagates_to_platform_event",
        ),
    ),
    _covered(
        "obs.runtime_event_run_id",
        ValidationDomain.OBSERVABILITY,
        _unit(
            "tests/unit/contracts/test_trace_1b_identity.py",
            "test_emit_context_propagates_to_platform_event",
        ),
    ),
    _covered(
        "obs.child_event_parent_correlation",
        ValidationDomain.OBSERVABILITY,
        _unit(
            "tests/unit/runtime/events/test_ue_9b_runtime_event_execution_id.py",
            "test_with_parent_preserves_child_execution_id",
        ),
    ),
    _covered(
        "obs.no_identity_minting",
        ValidationDomain.OBSERVABILITY,
        _gate(
            "tests/unit/runtime/architecture/test_ue_9br1_runtime_event_legacy_retirement_gate.py",
            "test_production_runtime_event_emitters_do_not_mint_execution_id",
        ),
    ),
    _covered(
        "obs.otlp_execution_identity_correlation",
        ValidationDomain.OBSERVABILITY,
        _unit(
            "tests/unit/runtime/observability/test_otlp_exporter.py",
            "test_runtime_event_identity_is_exported_in_otlp_attributes",
        ),
    ),
    # DIAGNOSTICS
    _covered(
        "diag.terminal_failure_correlates_run",
        ValidationDomain.DIAGNOSTICS,
        _unit(
            "tests/unit/runtime/diagnostics/test_terminal_execution_diagnostic_trigger.py",
            "test_trigger_builds_single_execution_orchestration_request",
        ),
    ),
    _partial(
        "diag.terminal_failure_correlates_execution",
        ValidationDomain.DIAGNOSTICS,
        GapTarget.UE_11F,
        _unit(
            "tests/unit/runtime/diagnostics/test_diagnostic_subsystem_failure_evidence.py",
            "test_failure_event_preserves_execution_identity",
        ),
    ),
    _partial(
        "diag.consumes_observability_evidence",
        ValidationDomain.DIAGNOSTICS,
        GapTarget.UE_11F,
        _unit(
            "tests/unit/runtime/diagnostics/test_diagnostic_assessment.py",
            "test_causal_without_runtime_emits_proven_finding",
        ),
    ),
    _partial(
        "diag.no_execution_lifecycle_ownership",
        ValidationDomain.DIAGNOSTICS,
        GapTarget.UE_11F,
        _unit(
            "tests/unit/runtime/architecture/test_diag_foundation_4_entrypoint_consistency.py",
            "test_df4_nexus_loop_is_single_terminal_diagnostic_emitter",
        ),
    ),
    # ANTI_BYPASS
    _covered(
        "anti_bypass.single_strategy_resolver_owner",
        ValidationDomain.ANTI_BYPASS,
        _gate(
            "tests/unit/runtime/architecture/test_ue_9d_legacy_execution_retirement_gate.py",
            "test_strategy_resolver_is_owned_by_canonical_router",
        ),
    ),
    _covered(
        "anti_bypass.no_direct_strategic_backend_execute",
        ValidationDomain.ANTI_BYPASS,
        _gate(
            "tests/unit/runtime/architecture/test_ue_9d_legacy_execution_retirement_gate.py",
            "test_strategic_backends_execute_only_through_canonical_router",
        ),
    ),
    _covered(
        "anti_bypass.no_nexus_root_lifecycle",
        ValidationDomain.ANTI_BYPASS,
        _gate(
            "tests/unit/runtime/architecture/test_ue_10r1_nexus_lifecycle_retirement_gate.py",
            "test_nexus_loop_has_no_root_lifecycle_mint_or_bind",
        ),
    ),
    _covered(
        "anti_bypass.no_graph_root_authority_bootstrap",
        ValidationDomain.ANTI_BYPASS,
        _gate(
            "tests/unit/runtime/architecture/test_ue_10r4_graph_authority_fail_closed_gate.py",
            "test_graph_executor_has_no_root_authority_fallback",
        ),
    ),
    _covered(
        "anti_bypass.public_root_execution_id_not_injectable",
        ValidationDomain.ANTI_BYPASS,
        _gate(
            "tests/unit/runtime/architecture/test_ue_10r3_platform_owned_root_identity_gate.py",
            "test_facade_execute_does_not_accept_root_execution_context",
        ),
        _unit(
            "tests/unit/runtime/execution/test_execution_facade.py",
            "test_facade_mints_platform_execution_id_not_supplied_by_caller",
        ),
    ),
    _covered(
        "anti_bypass.no_dynamic_imports_execution",
        ValidationDomain.ANTI_BYPASS,
        _gate(
            "tests/unit/runtime/architecture/test_ue_10r41_execution_import_hygiene_gate.py",
            "test_execution_package_has_no_local_imports",
        ),
    ),
    _covered(
        "anti_bypass.no_forbidden_reflection_execution",
        ValidationDomain.ANTI_BYPASS,
        _gate(
            "tests/unit/runtime/architecture/test_ue_10r4_graph_authority_fail_closed_gate.py",
            "test_execution_package_has_no_forbidden_quality_constructions",
        ),
    ),
    # PRODUCTION_SCENARIO — expected gaps; closed in UE-11G.
    _gap(
        "scenario.real_inference",
        ValidationDomain.PRODUCTION_SCENARIO,
        GapTarget.UE_11G,
    ),
    _gap(
        "scenario.real_agentic",
        ValidationDomain.PRODUCTION_SCENARIO,
        GapTarget.UE_11G,
    ),
    _gap(
        "scenario.real_orchestration",
        ValidationDomain.PRODUCTION_SCENARIO,
        GapTarget.UE_11G,
    ),
    _gap(
        "scenario.failure_recovery_end_to_end",
        ValidationDomain.PRODUCTION_SCENARIO,
        GapTarget.UE_11G,
    ),
    _gap(
        "scenario.observability_diagnostics_end_to_end",
        ValidationDomain.PRODUCTION_SCENARIO,
        GapTarget.UE_11G,
    ),
)

REQUIRED_CAPABILITY_IDS: frozenset[str] = frozenset(
    {
        "root.inference.end_to_end",
        "root.agentic.end_to_end",
        "root.orchestration.end_to_end",
        "lifecycle.root_execution_id_single",
        "lifecycle.root_execution_id_platform_owned",
        "lifecycle.identity_consistent_through_backend",
        "lifecycle.authority_bound_before_strategy",
        "lifecycle.budget_bound_before_strategy",
        "lifecycle.runtime_event_same_execution_id",
        "lifecycle.parent_child_execution_tree",
        "fail_closed.missing_identity",
        "fail_closed.missing_execution_id",
        "fail_closed.missing_authority",
        "fail_closed.missing_budget",
        "fail_closed.identity_mismatch",
        "fail_closed.budget_execution_id_mismatch",
        "fail_closed.authority_metadata_mismatch",
        "fail_closed.strategy_backend_missing",
        "child.unique_execution_id",
        "child.parent_execution_id_link",
        "child.authority_derived_from_parent",
        "child.authority_no_escalation",
        "child.budget_derived_from_parent",
        "child.routes_through_boundary",
        "child.routes_through_strategy_router",
        "concurrency.parallel_root_identity_isolation",
        "concurrency.parallel_root_authority_isolation",
        "concurrency.parallel_root_budget_isolation",
        "concurrency.parallel_child_identity_isolation",
        "concurrency.parallel_child_budget_isolation",
        "concurrency.no_context_cross_talk",
        "recovery.retry_execution_semantics",
        "recovery.resume_run_continuity",
        "recovery.resume_attempt_semantics",
        "recovery.resume_fresh_execution_id",
        "recovery.resume_execution_tree_continuity",
        "recovery.redelivery_run_continuity",
        "recovery.redelivery_attempt_semantics",
        "recovery.redelivery_budget_continuity",
        "recovery.redelivery_no_execution_id_reuse",
        "obs.runtime_event_execution_id",
        "obs.runtime_event_attempt_id",
        "obs.runtime_event_run_id",
        "obs.child_event_parent_correlation",
        "obs.no_identity_minting",
        "obs.otlp_execution_identity_correlation",
        "diag.terminal_failure_correlates_run",
        "diag.terminal_failure_correlates_execution",
        "diag.consumes_observability_evidence",
        "diag.no_execution_lifecycle_ownership",
        "anti_bypass.single_strategy_resolver_owner",
        "anti_bypass.no_direct_strategic_backend_execute",
        "anti_bypass.no_nexus_root_lifecycle",
        "anti_bypass.no_graph_root_authority_bootstrap",
        "anti_bypass.public_root_execution_id_not_injectable",
        "anti_bypass.no_dynamic_imports_execution",
        "anti_bypass.no_forbidden_reflection_execution",
        "scenario.real_inference",
        "scenario.real_agentic",
        "scenario.real_orchestration",
        "scenario.failure_recovery_end_to_end",
        "scenario.observability_diagnostics_end_to_end",
    }
)

REQUIRED_DOMAINS: frozenset[ValidationDomain] = frozenset(ValidationDomain)

_VALID_GAP_TARGETS: frozenset[GapTarget] = frozenset(GapTarget)

_TEST_SYMBOL_CACHE: dict[str, frozenset[str]] = {}


def repo_root() -> Path:
    return _REPO_ROOT


def matrix_capability_ids() -> frozenset[str]:
    return frozenset(entry.capability_id for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX)


def matrix_domains() -> frozenset[ValidationDomain]:
    return frozenset(entry.domain for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX)


def count_by_status() -> dict[ValidationStatus, int]:
    counts = {status: 0 for status in ValidationStatus}
    for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX:
        counts[entry.status] += 1
    return counts


def count_by_domain() -> dict[ValidationDomain, dict[ValidationStatus, int]]:
    result: dict[ValidationDomain, dict[ValidationStatus, int]] = {
        domain: {status: 0 for status in ValidationStatus} for domain in ValidationDomain
    }
    for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX:
        result[entry.domain][entry.status] += 1
    return result


def gap_backlog() -> dict[GapTarget, tuple[str, ...]]:
    backlog: dict[GapTarget, list[str]] = {target: [] for target in GapTarget}
    for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX:
        if entry.status in {ValidationStatus.PARTIAL, ValidationStatus.GAP}:
            assert entry.gap_target is not None
            backlog[entry.gap_target].append(entry.capability_id)
    return {target: tuple(sorted(ids)) for target, ids in backlog.items()}


def _collect_test_symbols(path: Path) -> frozenset[str]:
    normalized = path.as_posix()
    cached = _TEST_SYMBOL_CACHE.get(normalized)
    if cached is not None:
        return cached
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(item.name)
    result = frozenset(symbols)
    _TEST_SYMBOL_CACHE[normalized] = result
    return result


def _proof_path_allowed(proof_path: str) -> bool:
    normalized = proof_path.replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in _ALLOWED_PROOF_PREFIXES)


def validate_unified_execution_matrix(
    *,
    repo_root_path: Path | None = None,
) -> list[str]:
    """Return human-readable gate violations; empty list means PASS."""
    root = repo_root_path or _REPO_ROOT
    violations: list[str] = []

    seen_ids: set[str] = set()
    for entry in UNIFIED_EXECUTION_VALIDATION_MATRIX:
        if entry.capability_id in seen_ids:
            violations.append(f"duplicate capability_id: {entry.capability_id}")
        seen_ids.add(entry.capability_id)

        if entry.domain not in REQUIRED_DOMAINS:
            violations.append(f"unknown domain for {entry.capability_id}: {entry.domain}")

        if entry.status is ValidationStatus.COVERED:
            if entry.gap_target is not None:
                violations.append(
                    f"{entry.capability_id}: COVERED must not have gap_target"
                )
            if not entry.proofs:
                violations.append(f"{entry.capability_id}: COVERED requires proofs")
        elif entry.status is ValidationStatus.PARTIAL:
            if entry.gap_target is None:
                violations.append(f"{entry.capability_id}: PARTIAL requires gap_target")
            if not entry.proofs:
                violations.append(f"{entry.capability_id}: PARTIAL requires proofs")
            elif entry.gap_target not in _VALID_GAP_TARGETS:
                violations.append(
                    f"{entry.capability_id}: invalid gap_target {entry.gap_target}"
                )
        elif entry.status is ValidationStatus.GAP:
            if entry.gap_target is None:
                violations.append(f"{entry.capability_id}: GAP requires gap_target")
            if entry.proofs:
                violations.append(f"{entry.capability_id}: GAP must not list proofs")
            elif entry.gap_target not in _VALID_GAP_TARGETS:
                violations.append(
                    f"{entry.capability_id}: invalid gap_target {entry.gap_target}"
                )

        for proof in entry.proofs:
            if not _proof_path_allowed(proof.path):
                violations.append(
                    f"{entry.capability_id}: proof path not under tests/: {proof.path}"
                )
            absolute = root / proof.path
            if not absolute.is_file():
                violations.append(
                    f"{entry.capability_id}: missing proof file {proof.path}"
                )
                continue
            symbols = _collect_test_symbols(absolute)
            if proof.test_name not in symbols:
                violations.append(
                    f"{entry.capability_id}: missing test symbol "
                    f"{proof.test_name} in {proof.path}"
                )

    missing_required = REQUIRED_CAPABILITY_IDS - seen_ids
    for capability_id in sorted(missing_required):
        violations.append(f"missing required capability: {capability_id}")

    extra_ids = seen_ids - REQUIRED_CAPABILITY_IDS
    for capability_id in sorted(extra_ids):
        violations.append(f"unknown capability not in required set: {capability_id}")

    present_domains = matrix_domains()
    for domain in REQUIRED_DOMAINS:
        if domain not in present_domains:
            violations.append(f"missing required domain: {domain.value}")

    return violations
