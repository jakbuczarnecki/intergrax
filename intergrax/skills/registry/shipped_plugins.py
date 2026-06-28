# © Artur Czarnecki. All rights reserved.

"""First-party :class:`SkillPlugin` classes for all shipped skill bundles."""

from __future__ import annotations

from intergrax.skills.providers.agent.plugin import AgentSkillPlugin
from intergrax.skills.providers.billing.plugin import BillingSkillPlugin
from intergrax.skills.providers.browser.plugin import BrowserSkillPlugin
from intergrax.skills.providers.cache.plugin import CacheSkillPlugin
from intergrax.skills.providers.catalog.plugin import CatalogSkillPlugin
from intergrax.skills.providers.cloud_platform.plugin import CloudPlatformSkillPlugin
from intergrax.skills.providers.code.plugin import CodeSkillPlugin
from intergrax.skills.providers.codecraft.plugin import CodeCraftSkillPlugin
from intergrax.skills.providers.collaboration.plugin import CollaborationSkillPlugin
from intergrax.skills.providers.context.plugin import ContextSkillPlugin
from intergrax.skills.providers.cost.plugin import CostSkillPlugin
from intergrax.skills.providers.crm.plugin import CrmSkillPlugin
from intergrax.skills.providers.data.plugin import DataSkillPlugin
from intergrax.skills.providers.dev.plugin import DevSkillPlugin
from intergrax.skills.providers.eval.plugin import EvalSkillPlugin
from intergrax.skills.providers.filesystem.plugin import FilesystemSkillPlugin
from intergrax.skills.providers.gitlab.plugin import GitlabSkillPlugin
from intergrax.skills.providers.graph.plugin import GraphSkillPlugin
from intergrax.skills.providers.harness.plugin import HarnessSkillPlugin
from intergrax.skills.providers.health.plugin import HealthSkillPlugin
from intergrax.skills.providers.hitl.plugin import HitlSkillPlugin
from intergrax.skills.providers.http.plugin import HttpSkillPlugin
from intergrax.skills.providers.identity.plugin import IdentitySkillPlugin
from intergrax.skills.providers.interaction.plugin import InteractionSkillPlugin
from intergrax.skills.providers.jira.plugin import JiraSkillPlugin
from intergrax.skills.providers.knowledge.plugin import KnowledgeSkillPlugin
from intergrax.skills.providers.legal.plugin import LegalSkillPlugin
from intergrax.skills.providers.local.plugin import LocalSkillPlugin
from intergrax.skills.providers.memory.plugin import MemorySkillPlugin
from intergrax.skills.providers.message_bus.plugin import MessageBusSkillPlugin
from intergrax.skills.providers.metrics.plugin import MetricsSkillPlugin
from intergrax.skills.providers.ml.plugin import MlSkillPlugin
from intergrax.skills.providers.modality.plugin import ModalitySkillPlugin
from intergrax.skills.providers.notify.plugin import NotifySkillPlugin
from intergrax.skills.providers.openai.plugin import OpenaiSkillPlugin
from intergrax.skills.providers.ops.plugin import OpsSkillPlugin
from intergrax.skills.providers.platform.plugin import PlatformSkillPlugin
from intergrax.skills.providers.rag.plugin import RagSkillPlugin
from intergrax.skills.providers.research.plugin import ResearchSkillPlugin
from intergrax.skills.providers.sandbox.plugin import SandboxSkillPlugin
from intergrax.skills.providers.storage.plugin import StorageSkillPlugin
from intergrax.skills.providers.vector_store.plugin import VectorStoreSkillPlugin
from intergrax.skills.providers.workspace.plugin import WorkspaceSkillPlugin

SHIPPED_SKILL_PLUGINS: tuple[type, ...] = (
    AgentSkillPlugin,
    BillingSkillPlugin,
    BrowserSkillPlugin,
    CacheSkillPlugin,
    CatalogSkillPlugin,
    CloudPlatformSkillPlugin,
    CodeSkillPlugin,
    CodeCraftSkillPlugin,
    CollaborationSkillPlugin,
    ContextSkillPlugin,
    CostSkillPlugin,
    CrmSkillPlugin,
    DataSkillPlugin,
    DevSkillPlugin,
    EvalSkillPlugin,
    FilesystemSkillPlugin,
    GitlabSkillPlugin,
    GraphSkillPlugin,
    HarnessSkillPlugin,
    HealthSkillPlugin,
    HitlSkillPlugin,
    HttpSkillPlugin,
    IdentitySkillPlugin,
    InteractionSkillPlugin,
    JiraSkillPlugin,
    KnowledgeSkillPlugin,
    LegalSkillPlugin,
    LocalSkillPlugin,
    MemorySkillPlugin,
    MessageBusSkillPlugin,
    MetricsSkillPlugin,
    MlSkillPlugin,
    ModalitySkillPlugin,
    NotifySkillPlugin,
    OpenaiSkillPlugin,
    OpsSkillPlugin,
    PlatformSkillPlugin,
    RagSkillPlugin,
    ResearchSkillPlugin,
    SandboxSkillPlugin,
    StorageSkillPlugin,
    VectorStoreSkillPlugin,
    WorkspaceSkillPlugin,
)

SHIPPED_SKILL_BUNDLE_IDS: frozenset[str] = frozenset(
    p.skill_bundle_manifest().bundle_id for p in SHIPPED_SKILL_PLUGINS
)
