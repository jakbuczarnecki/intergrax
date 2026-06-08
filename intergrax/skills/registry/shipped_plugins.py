# © Artur Czarnecki. All rights reserved.

"""First-party :class:`SkillPlugin` classes for all shipped skill bundles."""

from __future__ import annotations

from intergrax.skills.providers.browser.plugin import BrowserSkillPlugin
from intergrax.skills.providers.cache.plugin import CacheSkillPlugin
from intergrax.skills.providers.collaboration.plugin import CollaborationSkillPlugin
from intergrax.skills.providers.data.plugin import DataSkillPlugin
from intergrax.skills.providers.dev.plugin import DevSkillPlugin
from intergrax.skills.providers.eval.plugin import EvalSkillPlugin
from intergrax.skills.providers.graph.plugin import GraphSkillPlugin
from intergrax.skills.providers.harness.plugin import HarnessSkillPlugin
from intergrax.skills.providers.hitl.plugin import HitlSkillPlugin
from intergrax.skills.providers.knowledge.plugin import KnowledgeSkillPlugin
from intergrax.skills.providers.legal.plugin import LegalSkillPlugin
from intergrax.skills.providers.memory.plugin import MemorySkillPlugin
from intergrax.skills.providers.message_bus.plugin import MessageBusSkillPlugin
from intergrax.skills.providers.modality.plugin import ModalitySkillPlugin
from intergrax.skills.providers.notify.plugin import NotifySkillPlugin
from intergrax.skills.providers.ops.plugin import OpsSkillPlugin
from intergrax.skills.providers.platform.plugin import PlatformSkillPlugin
from intergrax.skills.providers.rag.plugin import RagSkillPlugin
from intergrax.skills.providers.research.plugin import ResearchSkillPlugin
from intergrax.skills.providers.sandbox.plugin import SandboxSkillPlugin
from intergrax.skills.providers.storage.plugin import StorageSkillPlugin
from intergrax.skills.providers.workspace.plugin import WorkspaceSkillPlugin

SHIPPED_SKILL_PLUGINS: tuple[type, ...] = (
    HarnessSkillPlugin,
    BrowserSkillPlugin,
    CacheSkillPlugin,
    CollaborationSkillPlugin,
    DataSkillPlugin,
    DevSkillPlugin,
    EvalSkillPlugin,
    GraphSkillPlugin,
    HitlSkillPlugin,
    KnowledgeSkillPlugin,
    LegalSkillPlugin,
    MemorySkillPlugin,
    MessageBusSkillPlugin,
    ModalitySkillPlugin,
    NotifySkillPlugin,
    OpsSkillPlugin,
    PlatformSkillPlugin,
    RagSkillPlugin,
    ResearchSkillPlugin,
    SandboxSkillPlugin,
    StorageSkillPlugin,
    WorkspaceSkillPlugin,
)

SHIPPED_SKILL_BUNDLE_IDS: frozenset[str] = frozenset(
    p.skill_bundle_manifest().bundle_id for p in SHIPPED_SKILL_PLUGINS
)
