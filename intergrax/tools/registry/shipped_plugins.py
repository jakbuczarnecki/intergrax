# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""First-party :class:`ToolPlugin` classes for all shipped tool bundles (lazy-loaded)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.tools.registry.plugin_factory import define_tool_plugin

if TYPE_CHECKING:
    from intergrax.tools.registry.catalog import ToolBundleStatus

_SHIPPED_TOOL_PLUGINS: tuple[type, ...] | None = None
_SHIPPED_TOOL_BUNDLE_IDS: frozenset[str] | None = None


def _load_shipped_tool_plugins() -> tuple[type, ...]:
    global _SHIPPED_TOOL_PLUGINS, _SHIPPED_TOOL_BUNDLE_IDS
    if _SHIPPED_TOOL_PLUGINS is not None:
        return _SHIPPED_TOOL_PLUGINS

    from intergrax.tools.providers.agent.bundle import AGENT_BUNDLE_ID, AGENT_TOOL_IDS, register_agent_tools
    from intergrax.tools.providers.browser.bundle import BROWSER_BUNDLE_ID, BROWSER_TOOL_IDS, register_browser_tools
    from intergrax.tools.providers.catalog.bundle import CATALOG_BUNDLE_ID, CATALOG_TOOL_IDS, register_catalog_tools
    from intergrax.tools.providers.codecraft.bundle import (
        CODECRAFT_BUNDLE_ID,
        register_codecraft_tools,
    )
    from intergrax.tools.providers.codecraft.service import CODECRAFT_TOOL_IDS
    from intergrax.tools.providers.context_tool.bundle import CONTEXT_BUNDLE_ID, CONTEXT_TOOL_IDS, register_context_tools
    from intergrax.tools.providers.braintrust.bundle import (
        BRAINTRUST_BUNDLE_ID,
        BRAINTRUST_TOOL_IDS,
        register_braintrust_tools,
    )
    from intergrax.tools.providers.billing.bundle import BILLING_BUNDLE_ID, BILLING_TOOL_IDS, register_billing_tools
    from intergrax.tools.providers.cost.bundle import COST_BUNDLE_ID, COST_TOOL_IDS, register_cost_tools
    from intergrax.tools.providers.crm.bundle import CRM_BUNDLE_ID, CRM_TOOL_IDS, register_crm_tools
    from intergrax.tools.providers.collaboration.bundle import (
        COLLABORATION_BUNDLE_ID,
        COLLABORATION_TOOL_IDS,
        register_collaboration_tools,
    )
    from intergrax.tools.providers.database.bundle import (
        DATABASE_BUNDLE_ID,
        DATABASE_TOOL_IDS,
        register_database_tools,
    )
    from intergrax.tools.providers.cache.bundle import CACHE_BUNDLE_ID, CACHE_TOOL_IDS, register_cache_tools
    from intergrax.tools.providers.cloud_platform.bundle import (
        CLOUD_PLATFORM_BUNDLE_ID,
        CLOUD_PLATFORM_TOOL_IDS,
        register_cloud_platform_tools,
    )
    from intergrax.tools.providers.filesystem.bundle import (
        FILESYSTEM_BUNDLE_ID,
        FILESYSTEM_TOOL_IDS,
        register_filesystem_tools,
    )
    from intergrax.tools.providers.confluence.bundle import (
        CONFLUENCE_BUNDLE_ID,
        CONFLUENCE_TOOL_IDS,
        register_confluence_tools,
    )
    from intergrax.tools.providers.document.bundle import DOCUMENT_BUNDLE_ID, DOCUMENT_TOOL_IDS, register_document_tools
    from intergrax.tools.providers.eval.bundle import EVAL_BUNDLE_ID, EVAL_TOOL_IDS, register_eval_tools
    from intergrax.tools.providers.graph.bundle import GRAPH_BUNDLE_ID, GRAPH_TOOL_IDS, register_graph_tools
    from intergrax.tools.providers.harness.bundle import HARNESS_BUNDLE_ID, HARNESS_TOOL_IDS, register_harness_tools
    from intergrax.tools.providers.http.bundle import HTTP_BUNDLE_ID, HTTP_TOOL_IDS, register_http_tools
    from intergrax.tools.providers.health.bundle import HEALTH_BUNDLE_ID, HEALTH_TOOL_IDS, register_health_tools
    from intergrax.tools.providers.hitl.bundle import HITL_BUNDLE_ID, HITL_TOOL_IDS, register_hitl_tools
    from intergrax.tools.providers.identity.bundle import IDENTITY_BUNDLE_ID, IDENTITY_TOOL_IDS, register_identity_tools
    from intergrax.tools.providers.interaction.bundle import (
        INTERACTION_BUNDLE_ID,
        INTERACTION_TOOL_IDS,
        register_interaction_tools,
    )
    from intergrax.tools.providers.gitlab.bundle import (
        GITLAB_BUNDLE_ID,
        GITLAB_TOOL_IDS,
        register_gitlab_tools,
    )
    from intergrax.tools.providers.issues.bundle import ISSUES_BUNDLE_ID, ISSUES_TOOL_IDS, register_issues_tools
    from intergrax.tools.providers.jira.bundle import JIRA_BUNDLE_ID, JIRA_TOOL_IDS, register_jira_tools
    from intergrax.tools.providers.knowledge.bundle import (
        KNOWLEDGE_BUNDLE_ID,
        KNOWLEDGE_TOOL_IDS,
        register_knowledge_tools,
    )
    from intergrax.tools.providers.ltm.bundle import LTM_BUNDLE_ID, LTM_TOOL_IDS, register_ltm_tools
    from intergrax.tools.providers.memory.bundle import MEMORY_BUNDLE_ID, MEMORY_TOOL_IDS, register_memory_tools
    from intergrax.tools.providers.message_bus.bundle import (
        MESSAGE_BUS_BUNDLE_ID,
        MESSAGE_BUS_TOOL_IDS,
        register_message_bus_tools,
    )
    from intergrax.tools.providers.ml.bundle import ML_BUNDLE_ID, register_ml_tools
    from intergrax.tools.providers.ml.service import (
        ML_BATCH_PREDICT_TOOL_ID,
        ML_EXPLAIN_TOOL_ID,
        ML_PREDICT_TOOL_ID,
    )
    from intergrax.tools.providers.notify.bundle import NOTIFY_BUNDLE_ID, NOTIFY_TOOL_IDS, register_notify_tools
    from intergrax.tools.providers.platform.bundle import PLATFORM_BUNDLE_ID, PLATFORM_TOOL_IDS, register_platform_tools
    from intergrax.tools.providers.openai_vector_store.bundle import (
        OPENAI_VECTOR_STORE_BUNDLE_ID,
        OPENAI_VECTOR_STORE_TOOL_IDS,
        register_openai_vector_store_tools,
    )
    from intergrax.tools.providers.observability.bundle import (
        OBSERVABILITY_BUNDLE_ID,
        OBSERVABILITY_TOOL_IDS,
        register_observability_tools,
    )
    from intergrax.tools.providers.pagerduty.bundle import (
        PAGERDUTY_BUNDLE_ID,
        PAGERDUTY_TOOL_IDS,
        register_pagerduty_tools,
    )
    from intergrax.tools.providers.rag.bundle import RAG_BUNDLE_ID, RAG_TOOL_IDS, register_rag_tools
    from intergrax.tools.providers.records.bundle import RECORDS_BUNDLE_ID, RECORDS_TOOL_IDS, register_records_tools
    from intergrax.tools.providers.sandbox.bundle import SANDBOX_BUNDLE_ID, SANDBOX_TOOL_IDS, register_sandbox_tools
    from intergrax.tools.providers.skill_tool.bundle import SKILL_BUNDLE_ID, SKILL_TOOL_IDS, register_skill_tools
    from intergrax.tools.providers.storage.bundle import STORAGE_BUNDLE_ID, STORAGE_TOOL_IDS, register_storage_tools
    from intergrax.tools.providers.security.bundle import SECURITY_BUNDLE_ID, SECURITY_TOOL_IDS, register_security_tools
    from intergrax.tools.providers.speech.bundle import SPEECH_BUNDLE_ID, register_speech_tools
    from intergrax.tools.providers.workflow.bundle import WORKFLOW_BUNDLE_ID, WORKFLOW_TOOL_IDS, register_workflow_tools
    from intergrax.tools.providers.speech.service import SPEECH_SYNTHESIZE_TOOL_ID, SPEECH_TRANSCRIBE_TOOL_ID
    from intergrax.tools.providers.vision.bundle import VISION_BUNDLE_ID, register_vision_tools
    from intergrax.tools.providers.vision.service import (
        VISION_DETECT_TOOL_ID,
        VISION_OCR_REGIONS_TOOL_ID,
        VISION_SEGMENT_TOOL_ID,
    )
    from intergrax.tools.providers.vector_store.bundle import (
        VECTOR_STORE_BUNDLE_ID,
        VECTOR_STORE_TOOL_IDS,
        register_vector_store_tools,
    )
    from intergrax.tools.providers.websearch.bundle import (
        WEBSEARCH_BUNDLE_ID,
        WEBSEARCH_TOOL_IDS,
        register_websearch_tools,
    )
    from intergrax.tools.providers.workspace.bundle import (
        WORKSPACE_BUNDLE_ID,
        WORKSPACE_TOOL_IDS,
        register_workspace_tools,
    )
    from intergrax.tools.registry.catalog import ToolBundleStatus

    plugins: tuple[type, ...] = (
        define_tool_plugin(
            bundle_id=CATALOG_BUNDLE_ID,
            tool_ids=CATALOG_TOOL_IDS,
            register_fn=register_catalog_tools,
            description="Tool catalog introspection for agent builders (list/describe tools).",
            class_name="CatalogToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=AGENT_BUNDLE_ID,
            tool_ids=AGENT_TOOL_IDS,
            register_fn=register_agent_tools,
            description="Agent registry introspection tools for multi-agent hosts.",
            class_name="AgentToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=SKILL_BUNDLE_ID,
            tool_ids=SKILL_TOOL_IDS,
            register_fn=register_skill_tools,
            description="Skill resolver introspection tool.",
            class_name="SkillToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=RAG_BUNDLE_ID,
            tool_ids=RAG_TOOL_IDS,
            register_fn=register_rag_tools,
            description="Vector retrieval tools for indexed documents (RAG).",
            class_name="RagToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=WEBSEARCH_BUNDLE_ID,
            tool_ids=WEBSEARCH_TOOL_IDS,
            register_fn=register_websearch_tools,
            description="Web research tools (live search APIs).",
            class_name="WebsearchToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=JIRA_BUNDLE_ID,
            tool_ids=JIRA_TOOL_IDS,
            register_fn=register_jira_tools,
            description="Jira issue tracker tools.",
            class_name="JiraToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=GITLAB_BUNDLE_ID,
            tool_ids=GITLAB_TOOL_IDS,
            register_fn=register_gitlab_tools,
            description="GitLab issue tracker tools.",
            class_name="GitlabToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=CONFLUENCE_BUNDLE_ID,
            tool_ids=CONFLUENCE_TOOL_IDS,
            register_fn=register_confluence_tools,
            description="Confluence wiki tools.",
            class_name="ConfluenceToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=NOTIFY_BUNDLE_ID,
            tool_ids=NOTIFY_TOOL_IDS,
            register_fn=register_notify_tools,
            description="Outbound notification tools.",
            class_name="NotifyToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=PAGERDUTY_BUNDLE_ID,
            tool_ids=PAGERDUTY_TOOL_IDS,
            register_fn=register_pagerduty_tools,
            description="PagerDuty incident tools.",
            class_name="PagerdutyToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=OBSERVABILITY_BUNDLE_ID,
            tool_ids=OBSERVABILITY_TOOL_IDS,
            register_fn=register_observability_tools,
            description="Metrics, logs, traces, and error capture tools.",
            class_name="ObservabilityToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=BRAINTRUST_BUNDLE_ID,
            tool_ids=BRAINTRUST_TOOL_IDS,
            register_fn=register_braintrust_tools,
            description="Braintrust eval logging tools.",
            class_name="BraintrustToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=SANDBOX_BUNDLE_ID,
            tool_ids=SANDBOX_TOOL_IDS,
            register_fn=register_sandbox_tools,
            description="Sandboxed code execution tools (exec, code, script, browser).",
            class_name="SandboxToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=CODECRAFT_BUNDLE_ID,
            tool_ids=CODECRAFT_TOOL_IDS,
            register_fn=register_codecraft_tools,
            description="Ephemeral Code Craft — governed generate/gate/exec loop.",
            class_name="CodeCraftToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=LTM_BUNDLE_ID,
            tool_ids=LTM_TOOL_IDS,
            register_fn=register_ltm_tools,
            description="Long-term user memory search and write tools.",
            class_name="LtmToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=CONTEXT_BUNDLE_ID,
            tool_ids=CONTEXT_TOOL_IDS,
            register_fn=register_context_tools,
            description="Context budget helpers (summarize, token estimate).",
            class_name="ContextToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=HTTP_BUNDLE_ID,
            tool_ids=HTTP_TOOL_IDS,
            register_fn=register_http_tools,
            description="Allowlisted HTTP client tool.",
            class_name="HttpToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=SECURITY_BUNDLE_ID,
            tool_ids=SECURITY_TOOL_IDS,
            register_fn=register_security_tools,
            description="Security scanner tools (image/repo scan).",
            class_name="SecurityToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=WORKFLOW_BUNDLE_ID,
            tool_ids=WORKFLOW_TOOL_IDS,
            register_fn=register_workflow_tools,
            description="Workflow orchestrator tools (trigger/poll/logs).",
            class_name="WorkflowToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=SPEECH_BUNDLE_ID,
            tool_ids=(SPEECH_SYNTHESIZE_TOOL_ID, SPEECH_TRANSCRIBE_TOOL_ID),
            register_fn=register_speech_tools,
            description="Speech synthesis and transcription tools.",
            class_name="SpeechToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=VISION_BUNDLE_ID,
            tool_ids=(VISION_DETECT_TOOL_ID, VISION_SEGMENT_TOOL_ID, VISION_OCR_REGIONS_TOOL_ID),
            register_fn=register_vision_tools,
            description="Vision detection, segmentation, and OCR tools.",
            class_name="VisionToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=VECTOR_STORE_BUNDLE_ID,
            tool_ids=VECTOR_STORE_TOOL_IDS,
            register_fn=register_vector_store_tools,
            description="Vector store backend ops tools (count, delete, collections, health).",
            class_name="VectorStoreToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=ML_BUNDLE_ID,
            tool_ids=(ML_PREDICT_TOOL_ID, ML_EXPLAIN_TOOL_ID, ML_BATCH_PREDICT_TOOL_ID),
            register_fn=register_ml_tools,
            description="Classical ML predict/explain tools.",
            class_name="MlToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=OPENAI_VECTOR_STORE_BUNDLE_ID,
            tool_ids=OPENAI_VECTOR_STORE_TOOL_IDS,
            register_fn=register_openai_vector_store_tools,
            status=ToolBundleStatus.BETA,
            description="OpenAI managed vector store + file_search tools (vendor-specific).",
            class_name="OpenaiVectorStoreToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=WORKSPACE_BUNDLE_ID,
            tool_ids=WORKSPACE_TOOL_IDS,
            register_fn=register_workspace_tools,
            description="Shadow workspace filesystem tools (runtime-bound).",
            class_name="WorkspaceToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=MEMORY_BUNDLE_ID,
            tool_ids=MEMORY_TOOL_IDS,
            register_fn=register_memory_tools,
            description="Policy-scoped task memory tools (runtime-bound).",
            class_name="MemoryToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=KNOWLEDGE_BUNDLE_ID,
            tool_ids=KNOWLEDGE_TOOL_IDS,
            register_fn=register_knowledge_tools,
            description="Provider-agnostic wiki/knowledge base tools.",
            class_name="KnowledgeToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=DOCUMENT_BUNDLE_ID,
            tool_ids=DOCUMENT_TOOL_IDS,
            register_fn=register_document_tools,
            description="Document parser tools for RAG ingestion pipelines.",
            class_name="DocumentToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=BROWSER_BUNDLE_ID,
            tool_ids=BROWSER_TOOL_IDS,
            register_fn=register_browser_tools,
            description="Headless browser automation tools.",
            class_name="BrowserToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=STORAGE_BUNDLE_ID,
            tool_ids=STORAGE_TOOL_IDS,
            register_fn=register_storage_tools,
            description="Object/blob storage tools.",
            class_name="StorageToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=ISSUES_BUNDLE_ID,
            tool_ids=ISSUES_TOOL_IDS,
            register_fn=register_issues_tools,
            description="Provider-agnostic issue tracker tools.",
            class_name="IssuesToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=PLATFORM_BUNDLE_ID,
            tool_ids=PLATFORM_TOOL_IDS,
            register_fn=register_platform_tools,
            description="Platform tools: secrets, feature flags, CI/CD status.",
            class_name="PlatformToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=MESSAGE_BUS_BUNDLE_ID,
            tool_ids=MESSAGE_BUS_TOOL_IDS,
            register_fn=register_message_bus_tools,
            description="Asynchronous task queue / message bus tools.",
            class_name="MessageBusToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=GRAPH_BUNDLE_ID,
            tool_ids=GRAPH_TOOL_IDS,
            register_fn=register_graph_tools,
            description="Property-graph query tools.",
            class_name="GraphToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=COLLABORATION_BUNDLE_ID,
            tool_ids=COLLABORATION_TOOL_IDS,
            register_fn=register_collaboration_tools,
            description="Collaboration suite tools (mail, calendar, directory).",
            class_name="CollaborationToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=DATABASE_BUNDLE_ID,
            tool_ids=DATABASE_TOOL_IDS,
            register_fn=register_database_tools,
            description="Relational store SQL tools (query/execute).",
            class_name="DatabaseToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=RECORDS_BUNDLE_ID,
            tool_ids=RECORDS_TOOL_IDS,
            register_fn=register_records_tools,
            description="Document store JSON record tools.",
            class_name="RecordsToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=CACHE_BUNDLE_ID,
            tool_ids=CACHE_TOOL_IDS,
            register_fn=register_cache_tools,
            description="Tenant-scoped key-value cache tools.",
            class_name="CacheToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=CLOUD_PLATFORM_BUNDLE_ID,
            tool_ids=CLOUD_PLATFORM_TOOL_IDS,
            register_fn=register_cloud_platform_tools,
            description="Cloud platform facade health and category default resolution tools.",
            class_name="CloudPlatformToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=IDENTITY_BUNDLE_ID,
            tool_ids=IDENTITY_TOOL_IDS,
            register_fn=register_identity_tools,
            description="Identity provider tools (token verify, user profile, tenant directory).",
            class_name="IdentityToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=HARNESS_BUNDLE_ID,
            tool_ids=HARNESS_TOOL_IDS,
            register_fn=register_harness_tools,
            description="Harness run trace read tools (persisted runs, cost, events).",
            class_name="HarnessToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=HITL_BUNDLE_ID,
            tool_ids=HITL_TOOL_IDS,
            register_fn=register_hitl_tools,
            description="Human-in-the-loop decision queue tools (read + policy-gated write).",
            class_name="HitlToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=INTERACTION_BUNDLE_ID,
            tool_ids=INTERACTION_TOOL_IDS,
            register_fn=register_interaction_tools,
            description="Interaction session read tools (list sessions, last user input).",
            class_name="InteractionToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=HEALTH_BUNDLE_ID,
            tool_ids=HEALTH_TOOL_IDS,
            register_fn=register_health_tools,
            description="Integration and profile health probe tools.",
            class_name="HealthToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=EVAL_BUNDLE_ID,
            tool_ids=EVAL_TOOL_IDS,
            register_fn=register_eval_tools,
            description="Harness online evaluation registry tools (V-EVAL).",
            class_name="EvalToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=FILESYSTEM_BUNDLE_ID,
            tool_ids=FILESYSTEM_TOOL_IDS,
            register_fn=register_filesystem_tools,
            description="Read-only allowlisted filesystem browse tools (LKW.3).",
            class_name="FilesystemToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=BILLING_BUNDLE_ID,
            tool_ids=BILLING_TOOL_IDS,
            register_fn=register_billing_tools,
            description="Billing meter usage tools (V-COST / SaaS path).",
            class_name="BillingToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=COST_BUNDLE_ID,
            tool_ids=COST_TOOL_IDS,
            register_fn=register_cost_tools,
            description="Runtime-bound cost budget and quota governance tools (V-COST).",
            class_name="CostToolPlugin",
        ),
        define_tool_plugin(
            bundle_id=CRM_BUNDLE_ID,
            tool_ids=CRM_TOOL_IDS,
            register_fn=register_crm_tools,
            description="Read-only CRM context tools for support harness agents.",
            class_name="CrmToolPlugin",
        ),
    )
    _SHIPPED_TOOL_PLUGINS = plugins
    _SHIPPED_TOOL_BUNDLE_IDS = frozenset(p.tool_bundle_manifest().bundle_id for p in plugins)
    return plugins


def shipped_tool_plugins() -> tuple[type, ...]:
    return _load_shipped_tool_plugins()


def shipped_tool_bundle_ids() -> frozenset[str]:
    _load_shipped_tool_plugins()
    assert _SHIPPED_TOOL_BUNDLE_IDS is not None
    return _SHIPPED_TOOL_BUNDLE_IDS


# Backward-compatible aliases for register modules
def __getattr__(name: str):
    _load_shipped_tool_plugins()
    if name == "SHIPPED_TOOL_PLUGINS":
        return _SHIPPED_TOOL_PLUGINS
    if name == "SHIPPED_TOOL_BUNDLE_IDS":
        return _SHIPPED_TOOL_BUNDLE_IDS
    if name.endswith("ToolPlugin"):
        for plugin in _SHIPPED_TOOL_PLUGINS or ():
            if plugin.__name__ == name:
                return plugin
        raise AttributeError(name)
    raise AttributeError(name)
