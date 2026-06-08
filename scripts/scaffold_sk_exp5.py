# © Artur Czarnecki. All rights reserved.
"""Scaffold SK-EXP5 — 50 high-value compositional skill packs."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "intergrax" / "skills" / "providers"

# bundle -> list of (skill_id, CONST, description, tool_ids, risk, tags)
EXTEND: dict[str, list[tuple]] = {
    "rag": [
        ("rag.semantic_qa", "RAG_SEMANTIC_QA", "Semantic Q&A with memory search and document fetch.", ("rag.retrieve", "rag.get_document", "memory.search"), "MEDIUM", ("rag", "semantic", "qa")),
        ("rag.ingest_pipeline", "RAG_INGEST_PIPELINE", "End-to-end ingest: parse, ingest, and index readiness check.", ("document.parse", "rag.ingest_document", "rag.check_index_status"), "MEDIUM", ("rag", "ingest", "pipeline")),
        ("rag.metadata_search", "RAG_METADATA_SEARCH", "Metadata-filtered document discovery without destructive ops.", ("rag.search_by_metadata", "rag.list_documents", "rag.describe_collection"), "LOW", ("rag", "metadata", "search")),
    ],
    "legal": [
        ("legal.redline_draft", "LEGAL_REDLINE_DRAFT", "Contract redline drafting with retrieval and workspace IO.", ("rag.retrieve", "workspace.read_file", "workspace.write_file"), "MEDIUM", ("legal", "redline", "draft")),
        ("legal.regulatory_scan", "LEGAL_REGULATORY_SCAN", "Regulatory lookup across web, wiki, and indexed corpus.", ("websearch.query", "knowledge.search", "rag.retrieve"), "MEDIUM", ("legal", "regulatory", "scan")),
        ("legal.obligation_tracker", "LEGAL_OBLIGATION_TRACKER", "Track contractual obligations in task memory and workspace.", ("memory.write", "memory.read", "workspace.write_file"), "LOW", ("legal", "obligation", "tracker")),
    ],
    "research": [
        ("research.deep_dive", "RESEARCH_DEEP_DIVE", "Deep web research with batch fetch and report workspace export.", ("websearch.fetch_batch", "websearch.read_url", "workspace.write_file"), "MEDIUM", ("research", "deep_dive", "web")),
        ("research.source_validator", "RESEARCH_SOURCE_VALIDATOR", "Validate sources against index and parse previews.", ("websearch.query", "rag.retrieve", "document.parse_preview"), "MEDIUM", ("research", "source", "validator")),
        ("research.report_compiler", "RESEARCH_REPORT_COMPILER", "Compile citation-backed reports from retrieval and web evidence.", ("rag.retrieve", "websearch.query", "workspace.write_file"), "MEDIUM", ("research", "report", "compiler")),
    ],
    "workspace": [
        ("workspace.draft_reviewer", "WORKSPACE_DRAFT_REVIEWER", "Read-only draft review with workspace search and memory context.", ("workspace.read_file", "workspace.search", "memory.read"), "LOW", ("workspace", "draft", "review")),
        ("workspace.artifact_exporter", "WORKSPACE_ARTIFACT_EXPORTER", "Export workspace artifacts to durable object storage.", ("workspace.export_artifact", "storage.put", "workspace.list_files"), "MEDIUM", ("workspace", "export", "artifact")),
    ],
    "memory": [
        ("memory.cross_turn_notes", "MEMORY_CROSS_TURN_NOTES", "Cross-turn note taking with list/read/write task memory.", ("memory.write", "memory.list_keys", "memory.read"), "LOW", ("memory", "notes", "cross_turn")),
        ("memory.fact_extractor", "MEMORY_FACT_EXTRACTOR", "Extract durable facts into LTM with context summarization.", ("ltm.write_fact", "memory.read", "context.summarize"), "MEDIUM", ("memory", "fact", "extractor")),
    ],
    "ops": [
        ("ops.oncall_runbook", "OPS_ONCALL_RUNBOOK", "On-call runbook: logs, traces, and stakeholder notification.", ("logs.search", "observability.query_traces", "notify.send"), "MEDIUM", ("ops", "oncall", "runbook")),
        ("ops.postmortem_writer", "OPS_POSTMORTEM_WRITER", "Postmortem drafting from harness run metadata and logs.", ("harness.get_run", "logs.search", "workspace.write_file"), "MEDIUM", ("ops", "postmortem", "writer")),
        ("ops.change_approver", "OPS_CHANGE_APPROVER", "Change approval loop: HITL pending, notify, workflow poll.", ("hitl.list_pending", "notify.send", "workflow.poll"), "HIGH", ("ops", "change", "approval")),
        ("ops.capacity_planner", "OPS_CAPACITY_PLANNER", "Capacity planning from metrics, cost forecast, and run history.", ("metrics.query_range", "cost.forecast_spend", "harness.list_runs"), "MEDIUM", ("ops", "capacity", "planner")),
    ],
    "dev": [
        ("dev.pr_reviewer", "DEV_PR_REVIEWER", "PR/issue review with search, fetch, and mail notification.", ("issues.search", "issues.get_issue", "collaboration.send_mail"), "MEDIUM", ("dev", "pr", "review")),
        ("dev.release_notes", "DEV_RELEASE_NOTES", "Release notes from issue search and workspace export.", ("issues.search", "workspace.write_file", "notify.send"), "LOW", ("dev", "release", "notes")),
        ("dev.sprint_planner", "DEV_SPRINT_PLANNER", "Sprint planning with issues, calendar, and scratchpad memory.", ("issues.search", "collaboration.list_calendar", "memory.write"), "MEDIUM", ("dev", "sprint", "planner")),
    ],
    "platform": [
        ("platform.runbook_hub", "PLATFORM_RUNBOOK_HUB", "Platform hub: skill resolve, agent roster, and retrieval.", ("skill.resolve", "agent.list_agents", "rag.retrieve"), "LOW", ("platform", "runbook", "hub")),
        ("platform.flag_rollout", "PLATFORM_FLAG_ROLLOUT", "Feature-flag rollout with metrics probe and notify.", ("platform.evaluate_feature_flag", "notify.send", "metrics.query_instant"), "MEDIUM", ("platform", "flag", "rollout")),
        ("platform.deploy_inspector", "PLATFORM_DEPLOY_INSPECTOR", "Deploy inspection: workflow runs, check suites, and logs.", ("platform.list_workflow_runs", "platform.list_check_suites", "logs.search"), "MEDIUM", ("platform", "deploy", "inspector")),
    ],
    "collaboration": [
        ("collaboration.meeting_brief", "COLLABORATION_MEETING_BRIEF", "Meeting brief from calendar, user profile, and workspace draft.", ("collaboration.list_calendar", "collaboration.get_user", "workspace.write_file"), "MEDIUM", ("collaboration", "meeting", "brief")),
        ("collaboration.stakeholder_ping", "COLLABORATION_STAKEHOLDER_PING", "Stakeholder outreach with CRM context, mail, and notify.", ("crm.get_account", "collaboration.send_mail", "notify.send"), "MEDIUM", ("collaboration", "stakeholder", "ping")),
    ],
    "data": [
        ("data.pipeline_probe", "DATA_PIPELINE_PROBE", "Data pipeline health: SQL probe, records query, store check.", ("database.query", "records.query", "health.check_relational_store"), "MEDIUM", ("data", "pipeline", "probe")),
        ("data.schema_documenter", "DATA_SCHEMA_DOCUMENTER", "Schema documentation for SQL and records stores.", ("database.describe_schema", "records.describe_collection", "workspace.write_file"), "LOW", ("data", "schema", "documenter")),
    ],
    "hitl": [
        ("hitl.escalation_router", "HITL_ESCALATION_ROUTER", "Escalate HITL queue depth to PagerDuty and notify.", ("hitl.summarize_queue", "pagerduty.trigger_incident", "notify.send"), "HIGH", ("hitl", "escalation", "router")),
        ("hitl.decision_auditor", "HITL_DECISION_AUDITOR", "Audit HITL decisions with trace correlation.", ("hitl.get_decision", "hitl.list_for_task", "observability.query_traces"), "MEDIUM", ("hitl", "decision", "auditor")),
    ],
    "graph": [
        ("graph.path_finder", "GRAPH_PATH_FINDER", "Graph path exploration with node fetch and session memory.", ("graph.run_query", "graph.get_node", "memory.read"), "MEDIUM", ("graph", "path", "finder")),
        ("graph.knowledge_linker", "GRAPH_KNOWLEDGE_LINKER", "Link graph entities to RAG grounding and LTM facts.", ("graph.run_query", "rag.retrieve", "ltm.write_fact"), "MEDIUM", ("graph", "knowledge", "linker")),
    ],
    "sandbox": [
        ("sandbox.test_runner", "SANDBOX_TEST_RUNNER", "Sandbox test execution with workspace input and error capture.", ("sandbox.exec", "workspace.read_file", "errors.capture"), "HIGH", ("sandbox", "test", "runner")),
        ("sandbox.refactor_loop", "SANDBOX_REFACTOR_LOOP", "Iterative refactor: exec, write, and workspace search.", ("sandbox.exec", "workspace.write_file", "workspace.search"), "HIGH", ("sandbox", "refactor", "loop")),
    ],
    "storage": [
        ("storage.backup_sync", "STORAGE_BACKUP_SYNC", "Backup sync between object storage and workspace snapshot.", ("storage.get", "storage.put", "workspace.snapshot"), "MEDIUM", ("storage", "backup", "sync")),
        ("storage.presigned_share", "STORAGE_PRESIGNED_SHARE", "Presigned URL sharing with existence check and notify.", ("storage.presigned_url", "storage.exists", "notify.send"), "MEDIUM", ("storage", "presigned", "share")),
    ],
    "message_bus": [
        ("message_bus.retry_handler", "MESSAGE_BUS_RETRY_HANDLER", "Retry failed async tasks via re-enqueue and status poll.", ("message_bus.get_status", "message_bus.enqueue", "notify.send"), "MEDIUM", ("message_bus", "retry", "handler")),
        ("message_bus.dead_letter", "MESSAGE_BUS_DEAD_LETTER", "Dead-letter hygiene: list, purge completed, and log search.", ("message_bus.list_tasks", "message_bus.purge_completed", "logs.search"), "MEDIUM", ("message_bus", "dead_letter", "hygiene")),
    ],
    "cache": [
        ("cache.warm_prefetch", "CACHE_WARM_PREFETCH", "Warm session cache from retrieval results.", ("cache.set", "cache.get", "rag.retrieve"), "LOW", ("cache", "warm", "prefetch")),
    ],
    "eval": [
        ("eval.baseline_runner", "EVAL_BASELINE_RUNNER", "Baseline eval recording with Braintrust and run listing.", ("eval.record_observation", "braintrust.log_eval", "harness.list_runs"), "LOW", ("eval", "baseline", "runner")),
        ("eval.regression_guard", "EVAL_REGRESSION_GUARD", "Regression guard: compare releases, summarize, and alert.", ("eval.compare_releases", "eval.summarize_release", "notify.send"), "MEDIUM", ("eval", "regression", "guard")),
    ],
    "modality": [
        ("modality.audio_transcript", "MODALITY_AUDIO_TRANSCRIPT", "Audio transcript pipeline with parse preview and workspace export.", ("speech.transcribe", "document.parse_preview", "workspace.write_file"), "MEDIUM", ("modality", "audio", "transcript")),
        ("modality.image_analyst", "MODALITY_IMAGE_ANALYST", "Image analysis with detect, OCR, and ingest path.", ("vision.detect", "vision.ocr_regions", "rag.ingest_document"), "MEDIUM", ("modality", "image", "analyst")),
    ],
    "notify": [
        ("notify.escalation_ladder", "NOTIFY_ESCALATION_LADDER", "Escalation ladder: schedule, send, and PagerDuty trigger.", ("notify.schedule", "notify.send", "pagerduty.trigger_incident"), "HIGH", ("notify", "escalation", "ladder")),
    ],
    "cost": [
        ("cost.chargeback_report", "COST_CHARGEBACK_REPORT", "Chargeback report from run budget, billing usage, and workspace export.", ("cost.get_run_budget", "billing.list_usage", "workspace.write_file"), "MEDIUM", ("cost", "chargeback", "report")),
    ],
    "identity": [
        ("identity.session_bootstrap", "IDENTITY_SESSION_BOOTSTRAP", "Bootstrap session from verified identity and memory seed.", ("identity.verify_token", "identity.get_user", "memory.write"), "MEDIUM", ("identity", "session", "bootstrap")),
    ],
    "health": [
        ("health.identity_probe", "HEALTH_IDENTITY_PROBE", "Extended health sweep: identity, cache, notify, wiki backends.", ("health.check_identity_provider", "health.check_key_value_cache", "health.check_notification_channel", "health.check_wiki_knowledge"), "LOW", ("health", "identity", "probe")),
    ],
    "filesystem": [
        ("filesystem.stat_auditor", "FILESYSTEM_STAT_AUDITOR", "Filesystem audit: stat, list, and read for operator hosts.", ("filesystem.stat", "filesystem.list", "filesystem.read_text"), "MEDIUM", ("filesystem", "stat", "auditor")),
    ],
    "harness": [
        ("harness.cost_analyst", "HARNESS_COST_ANALYST", "Run cost analysis with compare and instant metrics.", ("harness.get_run_cost", "harness.compare_runs", "metrics.query_instant"), "LOW", ("harness", "cost", "analyst")),
        ("harness.integration_sweep", "HARNESS_INTEGRATION_SWEEP", "Integration sweep with catalog introspection and skill resolve.", ("health.check_integration", "catalog.list_tools", "skill.resolve"), "LOW", ("harness", "integration", "sweep")),
    ],
    "agent": [
        ("agent.capability_mapper", "AGENT_CAPABILITY_MAPPER", "Map agent contracts to catalog tools and skill packs.", ("agent.get_contract", "skill.resolve", "catalog.describe_tool"), "LOW", ("agent", "capability", "mapper")),
    ],
}


def _manifest_block(const: str, skill_id: str, desc: str, tools: tuple, risk: str, tags: tuple) -> str:
    tools_repr = ", ".join(f'"{t}"' for t in tools)
    tags_repr = ", ".join(f'"{t}"' for t in tags)
    return f"""{const} = SkillManifest(
    skill_id="{skill_id}",
    version="1.0.0",
    description="{desc}",
    tool_ids=({tools_repr}),
    prompt_instruction_ids=("{skill_id}.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.{risk},
    tags=({tags_repr}),
)
"""


def _append_manifests(bundle_id: str, skills: list[tuple]) -> None:
    path = ROOT / bundle_id / "manifests.py"
    blocks = "\n\n".join(_manifest_block(s[1], s[0], s[2], s[3], s[4], s[5]) for s in skills)
    path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n\n" + blocks + "\n", encoding="utf-8")


def _regenerate_plugin(bundle_id: str) -> None:
    manifests_path = ROOT / bundle_id / "manifests.py"
    consts = re.findall(r"^([A-Z][A-Z0-9_]+) = SkillManifest\(", manifests_path.read_text(encoding="utf-8"), re.MULTILINE)
    class_name = "".join(p.capitalize() for p in bundle_id.split("_")) + "SkillPlugin"
    tuple_name = f"_{bundle_id.upper()}_MANIFESTS"
    imports = ",\n    ".join(consts)
    plugin_path = ROOT / bundle_id / "plugin.py"
    content = f'''# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.{bundle_id}.manifests import (
    {imports},
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

{tuple_name} = (
    {",\n    ".join(consts)},
)


class {class_name}:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="{bundle_id}",
            skill_ids=tuple(m.skill_id for m in {tuple_name}),
            status=SkillBundleStatus.STABLE,
            description="{bundle_id} skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return {tuple_name}

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in {tuple_name}:
            registry.register(manifest)
'''
    plugin_path.write_text(content, encoding="utf-8")


def main() -> None:
    total = 0
    for bundle_id, skills in EXTEND.items():
        _append_manifests(bundle_id, skills)
        _regenerate_plugin(bundle_id)
        total += len(skills)
        print(f"{bundle_id}: +{len(skills)}")
    print(f"SK-EXP5 total: {total} skills")


if __name__ == "__main__":
    main()
