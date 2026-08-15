# © Artur Czarnecki. All rights reserved.
"""Shared constants for harness architecture audit orchestration."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DOMAIN_ORDER: tuple[str, ...] = (
    "PLATFORM_FOUNDATION",
    "UNIFIED_EXECUTION_RUNTIME",
    "ORCHESTRATION",
    "NEXUS_EXECUTION_FLOW",
    "REASONING_AND_COGNITION",
    "AGENT_CONTRACTS_AND_ASSEMBLY",
    "LLM_ADAPTERS",
    "TOOLS",
    "CODE_CRAFT",
    "SKILLS",
    "INTEGRATIONS",
    "RAG",
    "MEMORY",
    "CONTEXT_ENGINEERING",
    "MODALITY",
    "OBSERVABILITY",
    "RELIABILITY_FAILURE_AND_HITL",
    "CRITIC_VERIFICATION",
    "ADAPTIVE_HARNESS_INTELLIGENCE",
    "ELASTIC_CAPACITY_AND_SCALING",
    "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    "TIER3_APPLICATION_ENVIRONMENT",
    "APPLICATION_HOSTING",
    "UNIFIED_CONTEXT_LIFECYCLE",
)

RESULTS_ROOT = REPO_ROOT / "docs" / "audit_results"
AUDIT_GUIDES = REPO_ROOT / "docs" / "project" / "maintainers" / "audit"

ORCHESTRATOR_BY_MODE: dict[str, str] = {
    "audit_only": "docs/project/maintainers/audit/ORCHESTRATOR.md",
    "implement_plan": "docs/project/maintainers/audit/IMPLEMENT_ORCHESTRATOR.md",
    "layer_completion": "docs/project/maintainers/audit/LAYER_COMPLETION_ORCHESTRATOR.md",
    "idea_audit": "docs/project/maintainers/audit/IDEA_AUDIT_ORCHESTRATOR.md",
}

BOOTSTRAP_BY_MODE: dict[str, str] = {
    "audit_only": "docs/project/maintainers/bootstrap/01_audit_all_domains.txt",
    "audit_one": "docs/project/maintainers/bootstrap/02_audit_one_domain.txt",
    "implement_plan": "docs/project/maintainers/bootstrap/03_implement_plan_all_domains.txt",
    "implement_one": "docs/project/maintainers/bootstrap/04_implement_plan_one_domain.txt",
    "layer_completion": "docs/project/maintainers/bootstrap/05_closeout_all_domains.txt",
    "idea_audit": "docs/project/maintainers/bootstrap/idea_audit.txt",
}


def resolve_bootstrap(mode: str, single_domain: str | None) -> str:
    if single_domain:
        if mode == "audit_only":
            return BOOTSTRAP_BY_MODE["audit_one"]
        if mode == "implement_plan":
            return BOOTSTRAP_BY_MODE["implement_one"]
    return BOOTSTRAP_BY_MODE.get(mode, BOOTSTRAP_BY_MODE["audit_only"])


def default_domain_state() -> dict[str, object]:
    return {
        "status": "pending",
        "verdict": None,
        "p0_open": 0,
        "p1_open": 0,
        "result_md": None,
        "plan_updated": False,
    }


def build_progress_template(
    *,
    run_id: str,
    mode: str,
    scope: str,
    single_domain: str | None = None,
) -> dict[str, object]:
    orchestrator_key = mode if mode in ORCHESTRATOR_BY_MODE else "audit_only"
    domains: dict[str, object] = {}
    order = (single_domain,) if single_domain else DOMAIN_ORDER
    for domain in order:
        entry = default_domain_state()
        entry["result_md"] = f"docs/audit_results/{run_id}/{domain}.md"
        domains[domain] = entry

    return {
        "orchestrator": f"full_harness_{orchestrator_key}",
        "mode": orchestrator_key,
        "scope": scope,
        "single_domain": single_domain,
        "canonical": ORCHESTRATOR_BY_MODE.get(orchestrator_key, ORCHESTRATOR_BY_MODE["audit_only"]),
        "bootstrap": resolve_bootstrap(orchestrator_key, single_domain),
        "run_id": run_id,
        "results_dir": f"docs/audit_results/{run_id}",
        "started_at": run_id,
        "last_updated": run_id,
        "completed_at": None,
        "current_domain": order[0],
        "notes": "",
        "domain_order": list(order),
        "domains": domains,
    }
