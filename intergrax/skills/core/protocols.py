# © Artur Czarnecki. All rights reserved.

"""Skill provider protocol (mirrors ToolProvider pattern)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from intergrax.skills.registry.runtime import SkillRegistry


class SkillProvider(Protocol):
    """Register one or more skill manifests into a registry."""

    def register_skills(self, registry: "SkillRegistry") -> None:
        ...
