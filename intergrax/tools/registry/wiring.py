# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 dependency injection for tool handlers (Phase O.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.integrations.contracts.graph_store import GraphStore
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
from intergrax.integrations.contracts.speech_provider import SpeechProviderBackend
from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge
from intergrax.integrations.contracts.workflow_orchestrator import WorkflowOrchestratorBackend
from intergrax.tools.registry.runtime_bindings import TaskMemoryViewBinding

if TYPE_CHECKING:
    from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace


@dataclass
class ToolWiringContext:
    """
    Composed dependencies passed into catalog tool registration.

    Tier-3 applications build this from ``IntegrationProfile`` and runtime
    services (RAG manager, websearch executor, …). Tool handlers MUST NOT
    resolve integrations themselves.
    """

    issue_tracker: IssueTracker | None = None
    search_provider: SearchProvider | None = None
    wiki_knowledge: WikiKnowledge | None = None
    notification_channel: NotificationChannel | None = None
    observability_backend: ObservabilityBackend | None = None
    observability_backends: dict[str, ObservabilityBackend] = field(default_factory=dict)
    object_storage: ObjectStorage | None = None
    relational_store: RelationalStore | None = None
    document_store: DocumentStore | None = None
    browser_automation: BrowserAutomation | None = None
    document_parser: DocumentParser | None = None
    secrets_store: SecretsStore | None = None
    feature_flag_backend: FeatureFlagBackend | None = None
    ci_cd_backend: CiCdBackend | None = None
    message_bus: MessageBus | None = None
    graph_store: GraphStore | None = None
    collaboration_suite: CollaborationSuite | None = None
    key_value_cache: KeyValueCache | None = None
    shadow_workspace: ShadowWorkspace | None = None
    memory_view: TaskMemoryViewBinding | None = None
    rag_manager: Any | None = None
    vectorstore_manager: Any | None = None
    embedding_manager: Any | None = None
    retriever_manager: Any | None = None
    reranker_manager: Any | None = None
    rag_profile: Any | None = None
    retrieval_service: Any | None = None
    websearch_executor: Any | None = None
    sandbox_session: Any | None = None
    security_scanner: SecurityScannerBackend | None = None
    sandbox_host: SandboxHostBackend | None = None
    identity_provider: IdentityProviderBackend | None = None
    speech_provider: SpeechProviderBackend | None = None
    workflow_orchestrator: WorkflowOrchestratorBackend | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_integration_profile(
        cls,
        profile: Any,
        *,
        rag_manager: Any | None = None,
        vectorstore_manager: Any | None = None,
        embedding_manager: Any | None = None,
        retriever_manager: Any | None = None,
        reranker_manager: Any | None = None,
        rag_profile: Any | None = None,
        retrieval_service: Any | None = None,
        websearch_executor: Any | None = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> ToolWiringContext:
        """
        Resolve common integration contract slots from an ``IntegrationProfile``.

        Categories without a configured slug are skipped (``None``).
        """
        from intergrax.integrations.contracts.base import (
            IntegrationCategory,
            UnknownIntegrationError,
        )
        from intergrax.integrations.registry.catalog import get_entry
        from intergrax.integrations.registry.factory import resolve

        def _optional(category: IntegrationCategory) -> Any | None:
            instance = profile.instance_for_category(category)
            if instance is not None:
                return instance
            slug = profile.slug_for_category(category)
            if slug is None:
                return None
            try:
                return profile.resolve(category)
            except Exception:
                return None

        primary_obs = _optional(IntegrationCategory.OBSERVABILITY_BACKEND)
        obs_backends: dict[str, Any] = {}
        primary_binding = profile.binding_for_field("observability_backend")
        if primary_binding is not None and primary_obs is not None:
            primary_slug = primary_binding.resolved_slug()
            if primary_slug:
                obs_backends[primary_slug] = primary_obs
        for slug in profile.options:
            try:
                entry = get_entry(slug)
            except UnknownIntegrationError:
                continue
            if IntegrationCategory.OBSERVABILITY_BACKEND not in entry.categories:
                continue
            if slug in obs_backends:
                continue
            try:
                obs_backends[slug] = resolve(
                    IntegrationCategory.OBSERVABILITY_BACKEND,
                    slug=slug,
                    profile=profile,
                )
            except Exception:
                continue

        return cls(
            issue_tracker=_optional(IntegrationCategory.ISSUE_TRACKER),
            search_provider=_optional(IntegrationCategory.SEARCH_PROVIDER),
            wiki_knowledge=_optional(IntegrationCategory.WIKI_KNOWLEDGE),
            notification_channel=_optional(IntegrationCategory.NOTIFICATION_CHANNEL),
            observability_backend=primary_obs,
            observability_backends=obs_backends,
            object_storage=_optional(IntegrationCategory.OBJECT_STORAGE),
            relational_store=_optional(IntegrationCategory.RELATIONAL_STORE),
            document_store=_optional(IntegrationCategory.DOCUMENT_STORE),
            browser_automation=_optional(IntegrationCategory.BROWSER_AUTOMATION),
            document_parser=_optional(IntegrationCategory.DOCUMENT_PARSER),
            secrets_store=_optional(IntegrationCategory.SECRETS_STORE),
            feature_flag_backend=_optional(IntegrationCategory.FEATURE_FLAG),
            ci_cd_backend=_optional(IntegrationCategory.CI_CD),
            message_bus=_optional(IntegrationCategory.MESSAGE_BUS),
            graph_store=_optional(IntegrationCategory.GRAPH_STORE),
            collaboration_suite=_optional(IntegrationCategory.COLLABORATION_SUITE),
            key_value_cache=_optional(IntegrationCategory.KEY_VALUE_CACHE),
            security_scanner=_optional(IntegrationCategory.SECURITY_SCANNER),
            sandbox_host=_optional(IntegrationCategory.SANDBOX_HOST),
            identity_provider=_optional(IntegrationCategory.IDENTITY_PROVIDER),
            speech_provider=_optional(IntegrationCategory.SPEECH_PROVIDER),
            workflow_orchestrator=_optional(IntegrationCategory.WORKFLOW_ORCHESTRATOR),
            rag_manager=rag_manager,
            vectorstore_manager=vectorstore_manager,
            embedding_manager=embedding_manager,
            retriever_manager=retriever_manager,
            reranker_manager=reranker_manager,
            rag_profile=rag_profile,
            retrieval_service=retrieval_service,
            websearch_executor=websearch_executor,
            extras=dict(extras or {}),
        )
