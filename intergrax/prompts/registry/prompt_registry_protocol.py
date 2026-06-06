# © Artur Czarnecki. All rights reserved.

"""Typed contract for YAML-backed prompt resolution (Phase PE-3)."""

from __future__ import annotations

from typing import Protocol

from intergrax.prompts.registry.pin_config import PromptPinConfig
from intergrax.prompts.schema.prompt_schema import LocalizedContent
from intergrax.prompts.storage.models import LoadedPrompt


class PromptRegistryProtocol(Protocol):
    """Minimal surface used by Nexus prompt builders."""

    def resolve(
        self,
        prompt_id: str,
        pin: PromptPinConfig | None = None,
    ) -> LoadedPrompt: ...

    def resolve_localized(
        self,
        prompt_id: str,
        *,
        locale: str = "en",
        pin: PromptPinConfig | None = None,
    ) -> LocalizedContent: ...

    def list_prompt_ids(self) -> tuple[str, ...]: ...
