# © Artur Czarnecki. All rights reserved.

"""Read-only Skill registry surface for application contracts and host lifecycle."""

from __future__ import annotations

from typing import Any, Protocol


class SkillRegistryRead(Protocol):
    """Host-scoped skill registry read — not marketplace authority."""

    def has(self, skill_id: str) -> bool: ...

    def get(self, skill_id: str) -> Any: ...

    def binding_metadata(self, skill_id: str) -> Any | None: ...


__all__ = ["SkillRegistryRead"]
