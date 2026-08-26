#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-26.1 — umbrella gate for post-L3 ideal architecture closeout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def _run(script: str, *extra: str) -> int:
    script_path = str(REPO_ROOT / "scripts" / script)
    for cmd in (
        ["uv", "run", "python", script_path, *extra],
        [PYTHON, script_path, *extra],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            return 0
    print(f"FAILED: {script}", file=sys.stderr)
    return 1


def main() -> int:
    scripts = [
        ("harness_maturity_report.py", ("--enforce-l3-critical",)),
        ("check_shadow_eval_automation.py", ()),
        ("check_product_release_eval_gate.py", ()),
        ("check_registry_snapshot_diff.py", ()),
        ("check_context_golden.py", ()),
        ("check_agents_lifecycle_metadata.py", ()),
        ("check_structured_output_gate.py", ()),
        ("check_product_security_wiring.py", ()),
        ("check_sandbox_policy_wiring.py", ()),
        ("check_agent_promotion_eval_gate.py", ()),
        ("check_l4_runtime_evidence.py", ()),
        ("check_tenant_storage_isolation.py", ()),
        ("check_context_drift_monitoring.py", ()),
        ("check_modality_live_endpoints.py", ()),
        ("check_deploy_slo_evidence.py", ()),
        ("check_product_intake_parity.py", ()),
        ("check_reasoning_failure_taxonomy.py", ()),
        ("check_reasoning_gates.py", ()),
        ("check_product_long_running_resume.py", ()),
        ("check_swarm_coordination_templates.py", ()),
        ("check_tool_mcp_schema_export.py", ()),
        ("check_entity_graph_memory_wiring.py", ()),
        ("check_semantic_compression_profile.py", ()),
        ("check_partial_results_reference_hosts.py", ()),
        ("check_human_review_sample_queue.py", ()),
        ("check_bounded_policy_learning.py", ()),
        ("check_checkpoint_introspection_api.py", ()),
        ("check_evaluator_loop_graph_template.py", ()),
        ("check_delegation_budget_enforcement.py", ()),
        ("check_execution_strategy_hook.py", ()),
        ("check_oversized_tool_lint.py", ()),
        ("check_langgraph_skill_pack_import.py", ()),
        ("check_skill_selection_hook.py", ()),
        ("check_cost_forecast_wiring.py", ()),
        ("check_modality_product_worker_pool.py", ()),
        ("check_on_call_ownership_model.py", ()),
        ("check_prompt_approval_wiring.py", ()),
        ("check_prompt_compare_api.py", ()),
        ("check_cross_host_agent_certification.py", ()),
        ("check_capability_negotiation.py", ()),
        ("check_cost_optimization_wiring.py", ()),
        ("check_policy_change_impact_cli.py", ()),
        ("check_health_dashboard_contracts.py", ()),
        ("check_agent_simulator_wiring.py", ()),
        ("check_architecture_debt_burn_down.py", ()),
        ("check_compliance_profile_wiring.py", ()),
        ("check_live_model_routing_wiring.py", ()),
        ("check_llm_routing_rules.py", ()),
        ("check_llm_routing_context_wiring.py", ()),
        ("check_llm_profile_runtime.py", ()),
        ("check_llm_catalog_miss_observability.py", ()),
        ("check_rag_hierarchical_bootstrap.py", ()),
        ("check_rag_catalog_poisoning_defense.py", ()),
        ("check_tenant_fairness_quotas.py", ()),
        ("check_architecture_boundary_chaos.py", ()),
        ("check_plan_scorecard_sync.py", ()),
        ("check_multi_agent_contention_simulation.py", ()),
        ("check_trace_explorer_wiring.py", ()),
        ("check_replay_environment_wiring.py", ()),
        ("check_quarterly_strategy_review.py", ()),
        ("check_architecture_health_metrics.py", ()),
        ("check_production_capacity_adapters.py", ()),
        ("check_critical_action_signing.py", ()),
        ("check_immutable_security_audit_trail.py", ()),
        ("check_integration_marketplace_catalog.py", ()),
        ("check_catalog_hot_reload.py", ()),
        ("check_graph_editor_wiring.py", ()),
        ("check_capability_marketplace_readiness.py", ()),
        ("check_product_observability_dashboard.py", ()),
        ("check_governance_health_dashboard.py", ()),
        ("check_lkw_hybrid_daemon.py", ()),
        ("check_business_agent_certification.py", ()),
        ("check_pre_context_policy_wiring.py", ()),
        ("check_tool_injection_defense.py", ()),
        ("phase_v_capability_graph_guard.py", ("--enforce",)),
    ]
    exit_code = 0
    for script, extra in scripts:
        exit_code = exit_code or _run(script, *extra)
    test_path = REPO_ROOT / "tests" / "unit" / "runtime" / "architecture" / "test_audit_ideal_depth_gate.py"
    for cmd in (
        ["uv", "run", "pytest", str(test_path), "-q", "-m", "gate and no_ci"],
        [PYTHON, "-m", "pytest", str(test_path), "-q", "-m", "gate and no_ci"],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            break
    else:
        print("FAILED: test_audit_ideal_depth_gate.py", file=sys.stderr)
        exit_code = 1
    if exit_code == 0:
        print("OK: AUDIT-IDEAL gate checks passed")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
