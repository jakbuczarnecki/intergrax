# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Sequence

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry


class SkillResolutionError(ValueError):
    """Raised when skill_ids cannot be resolved against registries."""


@dataclass(frozen=True, slots=True)
class ResolvedSkillPack:
    """Output of :class:`SkillResolver` — merged skill composition for one agent run."""

    skill_ids: tuple[str, ...]
    tool_ids: frozenset[str]
    prompt_instruction_ids: frozenset[str]
    policy_fragment_ids: frozenset[str]
    risk_tier: SkillRiskTier

    def merged_allowed_tools(self, extra_allowed: Sequence[str] = ()) -> tuple[str, ...]:
        merged = set(self.tool_ids)
        merged.update(item.strip() for item in extra_allowed if item.strip())
        return tuple(sorted(merged))


class SkillResolver:
    """
    Resolves ``skill_ids`` into tool allow-lists and metadata (Phase R-Skill.3).

    No LLM calls — pure registry lookups and set merges.
    """

    def __init__(
        self,
        skill_registry: SkillRegistry,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    def resolve(self, skill_ids: Sequence[str]) -> ResolvedSkillPack:
        normalized = tuple(dict.fromkeys(sid.strip() for sid in skill_ids if sid.strip()))
        if not normalized:
            return ResolvedSkillPack(
                skill_ids=(),
                tool_ids=frozenset(),
                prompt_instruction_ids=frozenset(),
                policy_fragment_ids=frozenset(),
                risk_tier=SkillRiskTier.LOW,
            )

        tool_ids: set[str] = set()
        prompt_ids: set[str] = set()
        policy_ids: set[str] = set()
        max_risk = SkillRiskTier.LOW
        risk_order = list(SkillRiskTier)

        for skill_id in normalized:
            registered = self._skill_registry.get(skill_id)
            manifest: SkillManifest = registered.manifest
            tool_ids.update(manifest.tool_ids)
            prompt_ids.update(manifest.prompt_instruction_ids)
            if manifest.policy_fragment_id:
                policy_ids.add(manifest.policy_fragment_id)
            if risk_order.index(manifest.risk_tier) > risk_order.index(max_risk):
                max_risk = manifest.risk_tier

        if self._tool_registry is not None:
            self._validate_tools_exist(tool_ids)

        return ResolvedSkillPack(
            skill_ids=normalized,
            tool_ids=frozenset(tool_ids),
            prompt_instruction_ids=frozenset(prompt_ids),
            policy_fragment_ids=frozenset(policy_ids),
            risk_tier=max_risk,
        )

    def validate_skill_ids(self, skill_ids: AbstractSet[str] | Sequence[str]) -> None:
        for skill_id in skill_ids:
            sid = skill_id.strip()
            if sid and not self._skill_registry.has(sid):
                raise SkillResolutionError(f"Unknown skill_id: {sid}")

    def _validate_tools_exist(self, tool_ids: set[str]) -> None:
        assert self._tool_registry is not None
        missing = [tid for tid in sorted(tool_ids) if not self._tool_registry.has(tid)]
        if missing:
            raise SkillResolutionError(
                f"Skill references tool_id(s) not in ToolRegistry: {', '.join(missing)}"
            )
