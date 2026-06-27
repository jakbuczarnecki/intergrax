# © Artur Czarnecki. All rights reserved.
"""One-shot generator for SK-EXP3 per-skill USAGE.md files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "intergrax" / "skills" / "providers"

SKILLS: dict[str, dict[str, object]] = {
    "cost.budget_guardian": {
        "bundle": "cost",
        "risk": "medium",
        "purpose": "Run budget enforcement for governed agent hosts.",
        "works": "cost.* tools via CostBackend; resolved at SkillResolver registration.",
        "use": "cost_skill_profile(); enable on trusted operator hosts.",
        "get": "Quota checks before expensive tool loops.",
        "tools": [
            ("cost.check_quota", "Check remaining quota"),
            ("cost.get_run_budget", "Fetch run budget"),
            ("cost.forecast_spend", "Forecast spend trajectory"),
        ],
        "related": ["billing.usage_tracker", "ops.trace_debug"],
    },
    "identity.access_checker": {
        "bundle": "identity",
        "risk": "medium",
        "purpose": "Identity and tenancy verification for multi-tenant hosts.",
        "works": "identity.* via IdentityProvider integration.",
        "use": "identity_skill_profile(); wire identity_provider slug.",
        "get": "Token verification without agent-local auth code.",
        "tools": [
            ("identity.verify_token", "Validate bearer token"),
            ("identity.get_user", "Resolve user profile"),
            ("identity.list_tenants", "List accessible tenants"),
        ],
        "related": ["platform.secrets_flags", "hitl.approval_gate"],
    },
    "health.integration_probe": {
        "bundle": "health",
        "risk": "low",
        "purpose": "Integration health probes for operator dashboards.",
        "works": "health.check_* tools against wired integrations.",
        "use": "health_skill_profile(); run from harness operator agents.",
        "get": "Pre-flight backend readiness without custom scripts.",
        "tools": [
            ("health.check_integration", "Probe integration slug"),
            ("health.check_profile", "Validate environment profile"),
            ("health.check_relational_store", "Relational store probe"),
        ],
        "related": ["ops.trace_debug", "vector_store.admin"],
    },
    "context.token_planner": {
        "bundle": "context",
        "risk": "low",
        "purpose": "Context budget planning before LLM assembly.",
        "works": "context.estimate_tokens + context.summarize with memory.read fallback.",
        "use": "context_skill_profile(); pair with ContextProfile.budget on host.",
        "get": "Proactive trimming instead of hard failures.",
        "tools": [
            ("context.estimate_tokens", "Estimate token count"),
            ("context.summarize", "Summarize overflow text"),
            ("memory.read", "Read session context"),
        ],
        "related": ["memory.task_scratchpad", "rag.hybrid_qa"],
    },
    "memory.ltm_curator": {
        "bundle": "memory",
        "risk": "medium",
        "purpose": "Long-term memory fact curation across sessions.",
        "works": "ltm.write_fact + ltm.search via LTM store binding.",
        "use": "memory_skill_profile(); enable LTM on MemoryProfile.",
        "get": "Durable facts without ad-hoc memory tool lists.",
        "tools": [
            ("ltm.write_fact", "Persist durable fact"),
            ("ltm.search", "Search LTM index"),
            ("memory.read", "Read session context"),
        ],
        "related": ["memory.task_scratchpad", "memory.session_cleanup"],
    },
    "agent.roster_introspect": {
        "bundle": "agent",
        "risk": "low",
        "purpose": "Agent roster introspection for hub and concierge agents.",
        "works": "agent.list_agents + agent.get_contract + skill.resolve.",
        "use": "agent_roster_skill_profile(); platform hub hosts.",
        "get": "Self-describing harness without hardcoded agent lists.",
        "tools": [
            ("agent.list_agents", "List registered agents"),
            ("agent.get_contract", "Fetch agent contract"),
            ("skill.resolve", "Resolve skill pack"),
        ],
        "related": ["platform.concierge", "harness.skill_registry"],
    },
    "vector_store.admin": {
        "bundle": "vector_store",
        "risk": "low",
        "purpose": "Vector store administration separate from RAG ingest.",
        "works": "vector_store.* via VectorStore integration.",
        "use": "vector_store_skill_profile(); wire vector_store slug.",
        "get": "Backend health without destructive RAG ops.",
        "tools": [
            ("vector_store.list_collections", "List collections"),
            ("vector_store.count", "Count vectors"),
            ("vector_store.health", "Health probe"),
        ],
        "related": ["rag.index_admin", "rag.document_ingest"],
    },
    "eval.trajectory_judge": {
        "bundle": "eval",
        "risk": "medium",
        "purpose": "Trajectory-level eval judging for agent regression.",
        "works": "eval.judge + eval.record_observation + eval.trajectory.",
        "use": "eval_skill_profile(); wire eval harness backend.",
        "get": "Step-by-step eval without custom tool wiring.",
        "tools": [
            ("eval.judge", "Judge trajectory outcome"),
            ("eval.record_observation", "Record observation"),
            ("eval.trajectory", "Fetch trajectory"),
        ],
        "related": ["eval.score_logger", "eval.release_compare"],
    },
    "eval.release_compare": {
        "bundle": "eval",
        "risk": "low",
        "purpose": "Compare eval releases and export observation sets.",
        "works": "eval.compare_releases + eval.summarize_release + eval.export_observations.",
        "use": "eval_skill_profile(); CI eval gate hosts.",
        "get": "Release-over-release regression visibility.",
        "tools": [
            ("eval.compare_releases", "Compare two releases"),
            ("eval.summarize_release", "Summarize release metrics"),
            ("eval.export_observations", "Export observation bundle"),
        ],
        "related": ["eval.trajectory_judge", "ops.workflow_runner"],
    },
    "rag.retrieval_tuner": {
        "bundle": "rag",
        "risk": "medium",
        "purpose": "Retrieval tuning with preview and rerank before production queries.",
        "works": "rag.preview_retrieval + rag.rerank + rag.retrieve.",
        "use": "rag_skill_profile(); indexer and QA tuning agents.",
        "get": "Tuning loop without exposing all RAG admin tools.",
        "tools": [
            ("rag.preview_retrieval", "Preview retrieval candidates"),
            ("rag.rerank", "Rerank result set"),
            ("rag.retrieve", "Execute retrieval"),
        ],
        "related": ["rag.hybrid_qa", "rag.index_admin"],
    },
    "workspace.snapshot_manager": {
        "bundle": "workspace",
        "risk": "medium",
        "purpose": "Workspace snapshot and cleanup for long authoring sessions.",
        "works": "workspace.snapshot + list_files + delete_file.",
        "use": "lkw_skill_profile() or sandbox_skill_profile() hosts.",
        "get": "Checkpoint/rollback without host filesystem access.",
        "tools": [
            ("workspace.snapshot", "Create workspace snapshot"),
            ("workspace.list_files", "List workspace files"),
            ("workspace.delete_file", "Delete stale file"),
        ],
        "related": ["workspace.authoring", "storage.artifact_sync"],
    },
    "message_bus.task_admin": {
        "bundle": "message_bus",
        "risk": "medium",
        "purpose": "Message bus task queue administration.",
        "works": "message_bus.list_tasks + cancel + purge_completed.",
        "use": "message_bus_skill_profile(); operator cleanup agents.",
        "get": "Queue hygiene complement to async_runner.",
        "tools": [
            ("message_bus.list_tasks", "List queued tasks"),
            ("message_bus.cancel", "Cancel task"),
            ("message_bus.purge_completed", "Purge completed tasks"),
        ],
        "related": ["message_bus.async_runner", "ops.workflow_admin"],
    },
    "ops.workflow_admin": {
        "bundle": "ops",
        "risk": "medium",
        "purpose": "Workflow run administration: list, cancel, and inspect logs.",
        "works": "workflow.list_runs + cancel_run + fetch_logs.",
        "use": "ops_skill_profile(); batch orchestration hosts.",
        "get": "Ops visibility beyond workflow_runner trigger path.",
        "tools": [
            ("workflow.list_runs", "List workflow runs"),
            ("workflow.cancel_run", "Cancel in-flight run"),
            ("workflow.fetch_logs", "Fetch run logs"),
        ],
        "related": ["ops.workflow_runner", "eval.release_compare"],
    },
    "hitl.queue_manager": {
        "bundle": "hitl",
        "risk": "medium",
        "purpose": "HITL queue operations for operator dashboards.",
        "works": "hitl.list_for_task + summarize_queue + list_pending.",
        "use": "hitl_skill_profile(); HITL-enabled harness hosts.",
        "get": "Queue depth visibility without approval actions.",
        "tools": [
            ("hitl.list_for_task", "List decisions for task"),
            ("hitl.summarize_queue", "Summarize queue depth"),
            ("hitl.list_pending", "List pending decisions"),
        ],
        "related": ["hitl.approval_gate", "ops.incident_dispatch"],
    },
    "crm.account_lookup": {
        "bundle": "crm",
        "risk": "medium",
        "purpose": "CRM account research for support and sales agents.",
        "works": "crm.* via CRM integration backend.",
        "use": "crm_skill_profile(); wire crm integration slug.",
        "get": "Account context without vendor SDK in Tier-2.",
        "tools": [
            ("crm.get_account", "Fetch account record"),
            ("crm.list_contacts", "List account contacts"),
            ("crm.list_tickets", "List support tickets"),
        ],
        "related": ["collaboration.outreach", "dev.issue_triage"],
    },
    "billing.usage_tracker": {
        "bundle": "billing",
        "risk": "medium",
        "purpose": "Usage metering and run cost correlation.",
        "works": "billing.* + harness.get_run_cost for trace correlation.",
        "use": "billing_skill_profile(); platform metering hosts.",
        "get": "Chargeback visibility for multi-tenant deployments.",
        "tools": [
            ("billing.list_usage", "List usage records"),
            ("billing.record_usage", "Record usage event"),
            ("harness.get_run_cost", "Fetch run cost from trace"),
        ],
        "related": ["cost.budget_guardian", "metrics.run_observer"],
    },
    "metrics.run_observer": {
        "bundle": "metrics",
        "risk": "low",
        "purpose": "Runtime metrics queries correlated with traces.",
        "works": "metrics.query_instant/range + observability.query_traces.",
        "use": "metrics_skill_profile(); SRE operator agents.",
        "get": "Metrics + trace join without custom dashboards.",
        "tools": [
            ("metrics.query_instant", "Instant metric query"),
            ("metrics.query_range", "Range metric query"),
            ("observability.query_traces", "Correlate traces"),
        ],
        "related": ["ops.trace_debug", "eval.score_logger"],
    },
    "dev.issue_updater": {
        "bundle": "dev",
        "risk": "medium",
        "purpose": "Update existing tracker issues from agent remediation loops.",
        "works": "issues.update_issue + add_comment + get_issue.",
        "use": "ops_skill_profile(); skills=[DEV_ISSUE_UPDATER].",
        "get": "Close-the-loop updates complementing issue_creator.",
        "tools": [
            ("issues.update_issue", "Update issue fields"),
            ("issues.add_comment", "Add comment"),
            ("issues.get_issue", "Fetch issue details"),
        ],
        "related": ["dev.issue_creator", "dev.issue_triage"],
    },
    "collaboration.thread_reply": {
        "bundle": "collaboration",
        "risk": "medium",
        "purpose": "Email thread follow-up and reply drafting.",
        "works": "collaboration.reply_message + get/list messages.",
        "use": "Enable collaboration bundle on outreach hosts.",
        "get": "Thread continuity beyond initial outreach send.",
        "tools": [
            ("collaboration.reply_message", "Reply to thread"),
            ("collaboration.get_message", "Read message"),
            ("collaboration.list_messages", "List thread messages"),
        ],
        "related": ["collaboration.outreach", "collaboration.calendar"],
    },
    "ops.findings_review": {
        "bundle": "ops",
        "risk": "high",
        "purpose": "Security findings triage with scan, summarize, and notify.",
        "works": "security.summarize_findings + security.scan + notify.send.",
        "use": "ops_skill_profile(); restrict to HIGH risk trusted hosts.",
        "get": "Findings review loop distinct from security_audit scan-only path.",
        "tools": [
            ("security.summarize_findings", "Summarize scan findings"),
            ("security.scan", "Run security scan"),
            ("notify.send", "Alert owners"),
        ],
        "related": ["ops.security_audit", "ops.incident_dispatch"],
    },
}

TEMPLATE = """# `{skill_id}`

**Bundle:** `{bundle}` · **Version:** 1.0.0 · **Risk:** `{risk}`

## Purpose

{purpose}

## How it works

{works}

## How to use

{use}

## What you get

{get}

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
{tool_rows}

## Related skills

{related}
"""


def main() -> None:
    bundle_skills: dict[str, list[str]] = {}
    for skill_id, meta in SKILLS.items():
        bundle = str(meta["bundle"])
        path = ROOT / bundle / skill_id / "USAGE.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        tools = meta["tools"]
        assert isinstance(tools, list)
        tool_rows = "\n".join(f"| `{tid}` | {role} |" for tid, role in tools)
        related_list = meta["related"]
        assert isinstance(related_list, list)
        related = "\n".join(f"- `{item}`" for item in related_list)
        path.write_text(
            TEMPLATE.format(
                skill_id=skill_id,
                bundle=bundle,
                risk=meta["risk"],
                purpose=meta["purpose"],
                works=meta["works"],
                use=meta["use"],
                get=meta["get"],
                tool_rows=tool_rows,
                related=related,
            ),
            encoding="utf-8",
        )
        bundle_skills.setdefault(bundle, []).append(skill_id)

    for bundle, sids in bundle_skills.items():
        index_path = ROOT / bundle / "USAGE.md"
        existing = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        all_sids: set[str] = set(sids)
        if index_path.exists():
            for line in existing.splitlines():
                if line.startswith("| `") and "/USAGE.md)" in line:
                    all_sids.add(line.split("`")[1])
        lines = [f"# {bundle.title()} skill bundle", ""]
        if bundle in {
            "cost",
            "identity",
            "health",
            "context",
            "agent",
            "vector_store",
            "crm",
            "billing",
            "metrics",
        }:
            lines.append(
                f"**Bundle id:** `{bundle}` · **Plugin:** `{bundle.title()}SkillPlugin` · SK-EXP3"
            )
            lines.append("")
        lines.extend(["| skill_id | Guide |", "|----------|-------|"])
        for sid in sorted(all_sids):
            if (ROOT / bundle / sid / "USAGE.md").is_file():
                lines.append(f"| `{sid}` | [{sid}/USAGE.md]({sid}/USAGE.md) |")
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(SKILLS)} SK-EXP3 skill USAGE files")


if __name__ == "__main__":
    main()
