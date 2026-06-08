# © Artur Czarnecki. All rights reserved.
"""One-shot generator for SK-EXP2 per-skill USAGE.md files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "intergrax" / "skills" / "providers"

SKILLS: dict[str, dict[str, object]] = {
    "rag.index_admin": {
        "bundle": "rag",
        "risk": "low",
        "purpose": "Vector index introspection for operators and indexer agents.",
        "works": "Read-only RAG admin tools resolved at registration via SkillResolver.",
        "use": "SkillProfile(enabled_bundles=['rag']); skills=[RAG_INDEX_ADMIN] on AgentContract.",
        "get": "Standard admin surface without ad-hoc tool lists on indexer agents.",
        "tools": [
            ("rag.list_collections", "List index collections"),
            ("rag.describe_collection", "Collection stats"),
            ("rag.check_index_status", "Readiness probe"),
            ("rag.list_documents", "Paginated document ids"),
        ],
        "related": ["rag.document_ingest", "rag.collection_lifecycle"],
    },
    "rag.collection_lifecycle": {
        "bundle": "rag",
        "risk": "high",
        "purpose": "Controlled index lifecycle: metadata search, delete, and purge.",
        "works": "HIGH risk tier; destructive tools gated by ToolProfile and policy.",
        "use": "Admin-only hosts; pair with rag.index_admin before purge.",
        "get": "Grouped destructive ops under one governed skill.",
        "tools": [
            ("rag.search_by_metadata", "Metadata filter scan"),
            ("rag.delete_documents", "Delete by document id"),
            ("rag.purge_collection", "Controlled collection purge"),
        ],
        "related": ["rag.index_admin", "rag.document_ingest"],
    },
    "sandbox.code_exec": {
        "bundle": "sandbox",
        "risk": "high",
        "purpose": "Sandboxed code execution with workspace IO for coding agents.",
        "works": "sandbox.exec + workspace read/write via ToolWiringContext.",
        "use": "sandbox_skill_profile(); wire sandbox_session on host.",
        "get": "Isolated exec without host filesystem access.",
        "tools": [
            ("sandbox.exec", "Run allowlisted sandbox operation"),
            ("workspace.read_file", "Read script/input"),
            ("workspace.write_file", "Write output"),
        ],
        "related": ["workspace.authoring", "ops.security_audit"],
    },
    "hitl.approval_gate": {
        "bundle": "hitl",
        "risk": "high",
        "purpose": "Human-in-the-loop approval for high-risk agent actions.",
        "works": "hitl.* via HumanDecisionStoreBinding; notify.send for alerts.",
        "use": "hitl_skill_profile(); enable HITL store on harness host.",
        "get": "Governed approval without per-agent HITL wiring.",
        "tools": [
            ("hitl.list_pending", "List pending decisions"),
            ("hitl.submit_response", "Submit human response"),
            ("hitl.get_decision", "Fetch decision record"),
            ("notify.send", "Alert stakeholder"),
        ],
        "related": ["ops.incident_dispatch", "legal.contract_review"],
    },
    "graph.entity_explorer": {
        "bundle": "graph",
        "risk": "medium",
        "purpose": "Knowledge graph traversal with RAG grounding.",
        "works": "graph.* via GraphStore; rag.retrieve for text grounding.",
        "use": "graph_skill_profile(); wire graph_store + vector_store.",
        "get": "Structured graph + unstructured retrieval in one pack.",
        "tools": [
            ("graph.run_query", "Run graph query"),
            ("graph.get_node", "Fetch node by id"),
            ("rag.retrieve", "Grounding retrieval"),
        ],
        "related": ["legal.case_research", "rag.hybrid_qa"],
    },
    "storage.artifact_sync": {
        "bundle": "storage",
        "risk": "medium",
        "purpose": "Object storage sync with shadow workspace import/export.",
        "works": "storage.* + workspace import/export via integrations.",
        "use": "Wire object_storage slug; enable storage + workspace tools.",
        "get": "Durable artifacts across runs without agent-local IO.",
        "tools": [
            ("storage.get", "Fetch object"),
            ("storage.put", "Upload object"),
            ("workspace.export_artifact", "Export to storage"),
            ("workspace.import_artifact", "Import from storage"),
        ],
        "related": ["workspace.authoring", "research.citation_synthesis"],
    },
    "message_bus.async_runner": {
        "bundle": "message_bus",
        "risk": "medium",
        "purpose": "Async background tasks via message bus queue.",
        "works": "message_bus.* via TaskQueue/MessageBus integration.",
        "use": "message_bus_skill_profile(); wire message_bus slug.",
        "get": "Long-running work without blocking sync Nexus loop.",
        "tools": [
            ("message_bus.enqueue", "Enqueue task"),
            ("message_bus.get_status", "Poll status"),
            ("message_bus.get_result", "Fetch result"),
        ],
        "related": ["ops.workflow_runner", "eval.score_logger"],
    },
    "cache.session_cache": {
        "bundle": "cache",
        "risk": "low",
        "purpose": "KV cache with task memory read for session acceleration.",
        "works": "cache.* via KeyValueCache; memory.read as fallback.",
        "use": "cache_skill_profile(); wire key_value_cache integration.",
        "get": "Fewer duplicate tool calls within a session.",
        "tools": [
            ("cache.get", "Read cache key"),
            ("cache.set", "Write cache key"),
            ("memory.read", "Session memory fallback"),
        ],
        "related": ["memory.task_scratchpad", "rag.hybrid_qa"],
    },
    "eval.score_logger": {
        "bundle": "eval",
        "risk": "low",
        "purpose": "Log eval scores to Braintrust and query correlated traces.",
        "works": "braintrust.log_eval + observability.query_traces.",
        "use": "eval_skill_profile(); wire braintrust observability backend.",
        "get": "Standard eval harness agent pack.",
        "tools": [
            ("braintrust.log_eval", "Log eval score"),
            ("observability.query_traces", "Correlate traces"),
        ],
        "related": ["ops.trace_debug", "ops.workflow_runner"],
    },
    "modality.speech_io": {
        "bundle": "modality",
        "risk": "medium",
        "purpose": "Speech transcribe and synthesize for voice agents.",
        "works": "speech.* via SpeechProviderBackend; requires ModalityProfile.",
        "use": "modality_skill_profile(); wire speech_provider integration.",
        "get": "Voice agents without vendor SDK in Tier-2.",
        "tools": [
            ("speech.transcribe", "Speech-to-text"),
            ("speech.synthesize", "Text-to-speech"),
        ],
        "related": ["harness.modality_smoke", "platform.concierge"],
    },
    "modality.vision_ocr": {
        "bundle": "modality",
        "risk": "medium",
        "purpose": "Vision OCR pipeline for document images.",
        "works": "vision.* + document.parse_preview via model_inference.",
        "use": "modality_skill_profile(); enable modality on lab host.",
        "get": "Multimodal path before rag.document_ingest.",
        "tools": [
            ("vision.ocr_regions", "OCR text regions"),
            ("vision.detect", "Region detection"),
            ("document.parse_preview", "Parse structure preview"),
        ],
        "related": ["rag.document_ingest", "harness.vision_qa"],
    },
    "notify.scheduled_alerts": {
        "bundle": "notify",
        "risk": "medium",
        "purpose": "Deferred notification scheduling with cancel and immediate send.",
        "works": "notify.schedule/list/cancel + notify.send.",
        "use": "notify_skill_profile(); wire notification_channel.",
        "get": "Time-shifted alerts for long agent workflows.",
        "tools": [
            ("notify.schedule", "Schedule delivery"),
            ("notify.list_scheduled", "List pending"),
            ("notify.cancel_scheduled", "Cancel schedule"),
            ("notify.send", "Immediate send"),
        ],
        "related": ["ops.incident_dispatch", "collaboration.outreach"],
    },
    "collaboration.calendar": {
        "bundle": "collaboration",
        "risk": "medium",
        "purpose": "Calendar scheduling via collaboration suite.",
        "works": "collaboration calendar tools via CollaborationSuite.",
        "use": "Enable collaboration bundle; wire collaboration_suite slug.",
        "get": "Meeting scheduling complement to email outreach skill.",
        "tools": [
            ("collaboration.list_calendar", "List calendar events"),
            ("collaboration.create_event", "Create meeting"),
            ("collaboration.get_user", "Resolve user profile"),
        ],
        "related": ["collaboration.outreach", "dev.issue_creator"],
    },
    "platform.secrets_flags": {
        "bundle": "platform",
        "risk": "high",
        "purpose": "Runtime secrets and feature-flag evaluation.",
        "works": "platform.get_secret + evaluate_feature_flag bindings.",
        "use": "Restrict to HIGH risk agents on trusted hosts.",
        "get": "Governed secrets/flags without env leakage in agents.",
        "tools": [
            ("platform.get_secret", "Fetch secret by key"),
            ("platform.evaluate_feature_flag", "Evaluate feature flag"),
        ],
        "related": ["platform.concierge", "ops.security_audit"],
    },
    "platform.cicd_inspector": {
        "bundle": "platform",
        "risk": "medium",
        "purpose": "CI/CD workflow and check-suite inspection.",
        "works": "platform CI tools via CiCdBackend integration.",
        "use": "platform_skill_profile(); wire ci_cd backend.",
        "get": "Agent-driven CI visibility for release automation.",
        "tools": [
            ("platform.list_workflow_runs", "List workflow runs"),
            ("platform.get_workflow_run", "Run details"),
            ("platform.list_check_suites", "List check suites"),
        ],
        "related": ["dev.issue_creator", "ops.workflow_runner"],
    },
    "data.records_query": {
        "bundle": "data",
        "risk": "medium",
        "purpose": "Document store query for non-relational records.",
        "works": "records.* via DocumentStore integration.",
        "use": "data_skill_profile(); wire document_store slug.",
        "get": "NoSQL complement to data.sql_analyst.",
        "tools": [
            ("records.query", "Query records"),
            ("records.get", "Fetch record"),
            ("records.describe_collection", "Collection schema"),
        ],
        "related": ["data.sql_analyst", "rag.hybrid_qa"],
    },
    "dev.issue_creator": {
        "bundle": "dev",
        "risk": "medium",
        "purpose": "Create tracker issues from agent findings with dedup notify.",
        "works": "issues.create_issue + search; notify on create.",
        "use": "ops_skill_profile(); skills=[DEV_ISSUE_CREATOR].",
        "get": "Discovery-to-ticket loop for automation agents.",
        "tools": [
            ("issues.create_issue", "Create issue"),
            ("issues.search", "Dedup search"),
            ("notify.send", "Notify assignee"),
        ],
        "related": ["dev.issue_triage", "platform.cicd_inspector"],
    },
    "memory.session_cleanup": {
        "bundle": "memory",
        "risk": "medium",
        "purpose": "Session memory hygiene: list, read, delete stale keys.",
        "works": "memory.delete_key is destructive; list/read support safe purge.",
        "use": "memory_skill_profile(); task KV enabled on MemoryProfile.",
        "get": "Prevents unbounded task memory in long sessions.",
        "tools": [
            ("memory.list_keys", "Enumerate keys"),
            ("memory.delete_key", "Delete record"),
            ("memory.read", "Read before delete"),
        ],
        "related": ["memory.task_scratchpad", "cache.session_cache"],
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
        lines = [f"# {bundle.title()} skill bundle", ""]
        if "SK-EXP2" not in existing and bundle in {
            "sandbox",
            "hitl",
            "graph",
            "storage",
            "message_bus",
            "cache",
            "eval",
            "modality",
            "notify",
        }:
            lines.append(f"**Bundle id:** `{bundle}` · **Plugin:** `{bundle.title()}SkillPlugin` · SK-EXP2")
            lines.append("")
        lines.extend(["| skill_id | Guide |", "|----------|-------|"])
        all_sids = sorted(set(sids))
        if index_path.exists():
            for line in existing.splitlines():
                if line.startswith("| `") and "/USAGE.md)" in line:
                    sid = line.split("`")[1]
                    if sid not in all_sids:
                        all_sids.append(sid)
            all_sids = sorted(set(all_sids))
        for sid in all_sids:
            if (ROOT / bundle / sid / "USAGE.md").is_file():
                lines.append(f"| `{sid}` | [{sid}/USAGE.md]({sid}/USAGE.md) |")
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(SKILLS)} SK-EXP2 skill USAGE files")


if __name__ == "__main__":
    main()
