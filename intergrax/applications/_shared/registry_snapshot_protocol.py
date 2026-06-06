# © Artur Czarnecki. All rights reserved.

"""Typed registry snapshot contract (Phase REG-1)."""

from __future__ import annotations

from typing import Protocol

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.prompts.registry.prompt_registry_protocol import PromptRegistryProtocol
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry


class RegistrySnapshotProtocol(Protocol):
    """Minimal surface for harness registry resolution audits."""

    @property
    def integration_profile(self) -> IntegrationProfile | None: ...

    @property
    def tool_registry(self) -> ToolRegistry | None: ...

    @property
    def skill_registry(self) -> SkillRegistry | None: ...

    @property
    def prompt_registry(self) -> PromptRegistryProtocol | None: ...

    @property
    def policy_bundle(self) -> RuntimePolicyBundle | None: ...

    def tool_ids(self) -> tuple[str, ...]: ...

    def skill_ids(self) -> tuple[str, ...]: ...

    def prompt_ids(self) -> tuple[str, ...]: ...

    def agent_contract_ids(self) -> tuple[str, ...]: ...

    def evaluation_registry_ids(self) -> tuple[str, ...]: ...
