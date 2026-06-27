# © Artur Czarnecki. All rights reserved.
"""Scaffold SK-EXP4 skill manifests, plugins, and bundle helpers."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "intergrax" / "skills" / "providers"

# (skill_id, const_name, description, tool_ids, risk, tags)
NEW_BUNDLE_SKILLS: dict[str, list[tuple]] = {
    "catalog": [
        (
            "catalog.tool_introspect",
            "CATALOG_TOOL_INTROSPECT",
            "Tool catalog introspection: list tools, describe contracts, resolve skills.",
            ("catalog.list_tools", "catalog.describe_tool", "skill.resolve"),
            "LOW",
            ("catalog", "introspection", "tools"),
        ),
    ],
    "cloud_platform": [
        (
            "cloud_platform.resolver",
            "CLOUD_PLATFORM_RESOLVER",
            "Cloud platform resolution: health probe, endpoint resolve, and integration check.",
            ("cloud_platform.health", "cloud_platform.resolve", "health.check_integration"),
            "MEDIUM",
            ("cloud_platform", "resolve", "health"),
        ),
    ],
    "code": [
        (
            "code.runner",
            "CODE_RUNNER",
            "Controlled code execution: script run, code exec, and sandbox operation listing.",
            ("code.exec", "script.run", "sandbox.list_operations"),
            "HIGH",
            ("code", "exec", "script"),
        ),
    ],
    "filesystem": [
        (
            "filesystem.local_io",
            "FILESYSTEM_LOCAL_IO",
            "Local filesystem IO: read/write text, glob paths, and list directories.",
            ("filesystem.read_text", "filesystem.write_text", "filesystem.glob", "filesystem.list"),
            "HIGH",
            ("filesystem", "local", "io"),
        ),
    ],
    "http": [
        (
            "http.api_client",
            "HTTP_API_CLIENT",
            "HTTP API client: outbound requests with error capture and log correlation.",
            ("http.request", "errors.capture", "logs.search"),
            "MEDIUM",
            ("http", "api", "client"),
        ),
    ],
    "interaction": [
        (
            "interaction.session_handler",
            "INTERACTION_SESSION_HANDLER",
            "User session handling: list sessions, read history, and post replies.",
            (
                "interaction.list_sessions",
                "interaction.get_session_history",
                "interaction.post_reply",
            ),
            "MEDIUM",
            ("interaction", "session", "handler"),
        ),
        (
            "interaction.input_capture",
            "INTERACTION_INPUT_CAPTURE",
            "Capture last user input, post reply, and persist to task memory.",
            ("interaction.get_last_input", "interaction.post_reply", "memory.write"),
            "MEDIUM",
            ("interaction", "input", "capture"),
        ),
    ],
    "jira": [
        (
            "jira.task_navigator",
            "JIRA_TASK_NAVIGATOR",
            "Jira task navigation: search tasks, fetch issues, and add comments.",
            ("jira.search_tasks", "jira.get_issue", "jira.add_comment"),
            "MEDIUM",
            ("jira", "tasks", "navigator"),
        ),
    ],
    "gitlab": [
        (
            "gitlab.issue_creator",
            "GITLAB_ISSUE_CREATOR",
            "GitLab issue creation with dedup search and stakeholder notification.",
            ("gitlab.create_issue", "issues.search", "notify.send"),
            "MEDIUM",
            ("gitlab", "issues", "create"),
        ),
    ],
    "ml": [
        (
            "ml.explain_predict",
            "ML_EXPLAIN_PREDICT",
            "ML inference with explainability: predict, explain, and batch predict.",
            ("ml.predict", "ml.explain", "ml.batch_predict"),
            "MEDIUM",
            ("ml", "predict", "explain"),
        ),
    ],
    "openai": [
        (
            "openai.vector_admin",
            "OPENAI_VECTOR_ADMIN",
            "OpenAI vector store admin: upload, clear, and file_search query.",
            (
                "openai.vector_store.upload",
                "openai.vector_store.clear",
                "openai.file_search.query",
            ),
            "HIGH",
            ("openai", "vector_store", "admin"),
        ),
    ],
}

EXTEND_MANIFESTS: dict[str, list[tuple]] = {
    "browser": [
        (
            "browser.interactive_run",
            "BROWSER_INTERACTIVE_RUN",
            "Interactive browser automation: run browser, fetch page, parse preview.",
            ("browser.run", "browser.fetch_page", "document.parse_preview"),
            "HIGH",
            ("browser", "interactive", "automation"),
        ),
    ],
    "cache": [
        (
            "cache.key_admin",
            "CACHE_KEY_ADMIN",
            "Cache key administration: list, get, and delete session cache keys.",
            ("cache.list_keys", "cache.get", "cache.delete"),
            "MEDIUM",
            ("cache", "admin", "keys"),
        ),
    ],
    "knowledge": [
        (
            "knowledge.confluence_navigator",
            "KNOWLEDGE_CONFLUENCE_NAVIGATOR",
            "Confluence deep navigation: get page, search pages, and cross-search.",
            ("confluence.get_page", "confluence.search_pages", "confluence.search"),
            "MEDIUM",
            ("knowledge", "confluence", "navigator"),
        ),
    ],
    "data": [
        (
            "data.sql_mutator",
            "DATA_SQL_MUTATOR",
            "SQL mutation runner: execute statements with schema guard and query fallback.",
            ("database.execute", "database.describe_schema", "database.query"),
            "HIGH",
            ("data", "sql", "mutator"),
        ),
        (
            "data.records_admin",
            "DATA_RECORDS_ADMIN",
            "Records store admin: put, delete, and count documents.",
            ("records.put", "records.delete", "records.count"),
            "HIGH",
            ("data", "records", "admin"),
        ),
    ],
    "eval": [
        (
            "eval.observation_browser",
            "EVAL_OBSERVATION_BROWSER",
            "Eval observation browser: list observations, record new, and correlate traces.",
            ("eval.list_observations", "eval.record_observation", "observability.query_traces"),
            "LOW",
            ("eval", "observations", "browser"),
        ),
    ],
    "harness": [
        (
            "harness.run_comparator",
            "HARNESS_RUN_COMPARATOR",
            "Harness run comparison: list runs, fetch details, and compare outcomes.",
            ("harness.list_runs", "harness.get_run", "harness.compare_runs"),
            "LOW",
            ("harness", "runs", "compare"),
        ),
        (
            "harness.run_exporter",
            "HARNESS_RUN_EXPORTER",
            "Harness run export: bundle export with events and run metadata.",
            ("harness.export_run_bundle", "harness.get_run_events", "harness.get_run"),
            "MEDIUM",
            ("harness", "runs", "export"),
        ),
    ],
    "health": [
        (
            "health.full_stack_probe",
            "HEALTH_FULL_STACK_PROBE",
            "Full-stack health probe: graph store, message bus, object storage, search provider.",
            (
                "health.check_graph_store",
                "health.check_message_bus",
                "health.check_object_storage",
                "health.check_search_provider",
            ),
            "LOW",
            ("health", "full_stack", "probe"),
        ),
    ],
    "ops": [
        (
            "ops.log_tail",
            "OPS_LOG_TAIL",
            "Live log tailing with search and error capture for incident response.",
            ("logs.tail", "logs.search", "errors.capture"),
            "MEDIUM",
            ("ops", "logs", "tail"),
        ),
        (
            "ops.incident_ack",
            "OPS_INCIDENT_ACK",
            "PagerDuty incident acknowledge with trigger and notify escalation path.",
            ("pagerduty.acknowledge_incident", "pagerduty.trigger_incident", "notify.send"),
            "HIGH",
            ("ops", "incident", "pagerduty"),
        ),
    ],
    "memory": [
        (
            "memory.semantic_search",
            "MEMORY_SEMANTIC_SEARCH",
            "Semantic memory search across session memory and LTM index.",
            ("memory.search", "memory.read", "ltm.search"),
            "LOW",
            ("memory", "semantic", "search"),
        ),
    ],
    "notify": [
        (
            "notify.batch_dispatch",
            "NOTIFY_BATCH_DISPATCH",
            "Batch notification dispatch with due scheduling and pending list.",
            ("notify.send_batch", "notify.dispatch_due", "notify.list_scheduled"),
            "MEDIUM",
            ("notify", "batch", "dispatch"),
        ),
    ],
    "platform": [
        (
            "platform.secret_admin",
            "PLATFORM_SECRET_ADMIN",
            "Secret lifecycle admin: put, delete, and get runtime secrets.",
            ("platform.put_secret", "platform.delete_secret", "platform.get_secret"),
            "HIGH",
            ("platform", "secrets", "admin"),
        ),
        (
            "platform.workflow_cancel",
            "PLATFORM_WORKFLOW_CANCEL",
            "CI workflow cancellation: cancel run, fetch details, list runs.",
            (
                "platform.cancel_workflow_run",
                "platform.get_workflow_run",
                "platform.list_workflow_runs",
            ),
            "HIGH",
            ("platform", "workflow", "cancel"),
        ),
    ],
    "storage": [
        (
            "storage.object_lifecycle",
            "STORAGE_OBJECT_LIFECYCLE",
            "Object storage lifecycle: exists check, presigned URLs, and delete.",
            ("storage.exists", "storage.presigned_url", "storage.delete"),
            "HIGH",
            ("storage", "lifecycle", "object"),
        ),
    ],
    "vector_store": [
        (
            "vector_store.purge",
            "VECTOR_STORE_PURGE",
            "Vector store purge: delete vectors with count and collection listing.",
            ("vector_store.delete", "vector_store.count", "vector_store.list_collections"),
            "HIGH",
            ("vector_store", "purge", "delete"),
        ),
    ],
    "modality": [
        (
            "modality.vision_segment",
            "MODALITY_VISION_SEGMENT",
            "Vision segmentation pipeline: segment regions, detect, and OCR.",
            ("vision.segment", "vision.detect", "vision.ocr_regions"),
            "MEDIUM",
            ("modality", "vision", "segment"),
        ),
    ],
    "research": [
        (
            "research.web_cache_admin",
            "RESEARCH_WEB_CACHE_ADMIN",
            "Web search cache admin: invalidate cache, query, and batch fetch.",
            ("websearch.invalidate_cache", "websearch.query", "websearch.fetch_batch"),
            "LOW",
            ("research", "web", "cache"),
        ),
    ],
}


def _risk_enum(risk: str) -> str:
    return f"SkillRiskTier.{risk}"


def _manifest_block(
    const: str,
    skill_id: str,
    desc: str,
    tools: tuple[str, ...],
    risk: str,
    tags: tuple[str, ...],
) -> str:
    tools_repr = ", ".join(f'"{t}"' for t in tools)
    tags_repr = ", ".join(f'"{t}"' for t in tags)
    return f"""{const} = SkillManifest(
    skill_id="{skill_id}",
    version="1.0.0",
    description="{desc}",
    tool_ids=({tools_repr}),
    prompt_instruction_ids=("{skill_id}.system",),
    policy_fragment_id=None,
    risk_tier={_risk_enum(risk)},
    tags=({tags_repr}),
)
"""


def _write_new_bundle(bundle_id: str, skills: list[tuple]) -> None:
    bundle_dir = ROOT / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    class_name = "".join(p.capitalize() for p in bundle_id.split("_")) + "SkillPlugin"
    consts = [s[1] for s in skills]
    blocks = "\n\n".join(_manifest_block(s[1], s[0], s[2], s[3], s[4], s[5]) for s in skills)
    (bundle_dir / "manifests.py").write_text(
        f"# © Artur Czarnecki. All rights reserved.\n\n"
        f"from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier\n\n"
        f"{blocks}\n",
        encoding="utf-8",
    )
    imports = ", ".join(consts)
    reg_lines = "\n".join(f"        registry.register({c})" for c in consts)
    skill_ids = ", ".join(f"{c}.skill_id" for c in consts)
    (bundle_dir / "plugin.py").write_text(
        f"# © Artur Czarnecki. All rights reserved.\n\n"
        f"from __future__ import annotations\n\n"
        f"from intergrax.skills.core.manifest import SkillBundleManifest\n"
        f"from intergrax.skills.providers.{bundle_id}.manifests import {imports}\n"
        f"from intergrax.skills.registry.catalog import SkillBundleStatus\n"
        f"from intergrax.skills.registry.runtime import SkillRegistry\n\n\n"
        f"class {class_name}:\n"
        f"    @classmethod\n"
        f"    def skill_bundle_manifest(cls) -> SkillBundleManifest:\n"
        f"        return SkillBundleManifest(\n"
        f'            bundle_id="{bundle_id}",\n'
        f"            skill_ids=({skill_ids}),\n"
        f"            status=SkillBundleStatus.STABLE,\n"
        f'            description="{bundle_id.title()} skill packs (SK-EXP4)",\n'
        f"        )\n\n"
        f"    @classmethod\n"
        f"    def skill_manifests(cls) -> tuple:\n"
        f"        return ({imports})\n\n"
        f"    @classmethod\n"
        f"    def register_skills(cls, registry: SkillRegistry) -> None:\n"
        f"{reg_lines}\n",
        encoding="utf-8",
    )
    fn = f"register_{bundle_id}_skill_bundle"
    (bundle_dir / "bundle.py").write_text(
        f"# © Artur Czarnecki. All rights reserved.\n\n"
        f"from __future__ import annotations\n\n"
        f"from intergrax.skills.providers.{bundle_id}.plugin import {class_name}\n"
        f"from intergrax.skills.registry.plugin_register import register_skill_plugin\n\n\n"
        f"def {fn}(*, override: bool = False) -> None:\n"
        f"    register_skill_plugin({class_name}, override=override)\n",
        encoding="utf-8",
    )


def _append_to_manifests(bundle_id: str, skills: list[tuple]) -> None:
    path = ROOT / bundle_id / "manifests.py"
    text = path.read_text(encoding="utf-8")
    blocks = "\n\n".join(_manifest_block(s[1], s[0], s[2], s[3], s[4], s[5]) for s in skills)
    path.write_text(text.rstrip() + "\n\n" + blocks + "\n", encoding="utf-8")


def main() -> None:
    for bundle_id, skills in NEW_BUNDLE_SKILLS.items():
        _write_new_bundle(bundle_id, skills)
        print(f"NEW bundle: {bundle_id} ({len(skills)} skills)")

    for bundle_id, skills in EXTEND_MANIFESTS.items():
        _append_to_manifests(bundle_id, skills)
        print(f"EXTEND manifests: {bundle_id} (+{len(skills)} skills)")

    print("Done — update plugins manually or run patch_plugins step")


if __name__ == "__main__":
    main()
