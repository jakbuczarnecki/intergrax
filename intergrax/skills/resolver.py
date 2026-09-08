# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Protocol, Sequence

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.core.version_binding import (
    ResolvedSkillRef,
    ResolvedSkillRole,
    SkillVersionResolutionMode,
)
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.snapshot_digest import compute_resolved_skill_pack_digest
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.runtime import ToolRegistry


class SkillResolutionError(ValueError):
    """Raised when skill_ids cannot be resolved against registries."""


@dataclass(frozen=True, slots=True)
class ResolvedSkillPack:
    """Output of :class:`SkillResolver` — merged skill composition for one agent run."""

    resolved_skills: tuple[ResolvedSkillRef, ...]
    tool_ids: frozenset[str]
    prompt_instruction_ids: frozenset[str]
    policy_fragment_ids: frozenset[str]
    risk_tier: SkillRiskTier
    snapshot_digest: str

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(ref.skill_id for ref in self.resolved_skills)

    def merged_allowed_tools(self, extra_allowed: Sequence[str] = ()) -> tuple[str, ...]:
        merged = set(self.tool_ids)
        merged.update(item.strip() for item in extra_allowed if item.strip())
        return tuple(sorted(merged))


def _validate_resolved_skill_composition(
    pack: ResolvedSkillPack,
    observed_manifests: tuple[SkillManifest, ...],
) -> None:
    manifest_by_skill_id: dict[str, SkillManifest] = {}
    for manifest in observed_manifests:
        skill_id = manifest.skill_id
        if skill_id in manifest_by_skill_id:
            raise SkillResolutionError(
                f"duplicate observed manifest for skill_id: {skill_id}",
            )
        manifest_by_skill_id[skill_id] = manifest

    for ref in pack.resolved_skills:
        manifest = manifest_by_skill_id.get(ref.skill_id)
        if manifest is None:
            raise SkillResolutionError(
                f"missing observed manifest for resolved skill: {ref.qualified_id}",
            )
        if manifest.version != ref.version:
            raise SkillResolutionError(
                f"manifest version mismatch for {ref.qualified_id}: "
                f"observed {manifest.skill_id}@{manifest.version}",
            )


@dataclass(frozen=True, slots=True)
class ResolvedSkillComposition:
    """Coherent resolution observation: pack plus manifests observed during traversal."""

    pack: ResolvedSkillPack
    observed_manifests: tuple[SkillManifest, ...]

    def __post_init__(self) -> None:
        _validate_resolved_skill_composition(self.pack, self.observed_manifests)

    def manifest_by_skill_id(self) -> dict[str, SkillManifest]:
        return {manifest.skill_id: manifest for manifest in self.observed_manifests}


class SkillResolverProtocol(Protocol):
    """Typed contract for skill composition resolution (Phase TS-3)."""

    @property
    def skill_registry(self) -> SkillRegistry: ...

    def resolve(self, skill_ids: Sequence[str]) -> ResolvedSkillPack: ...

    def resolve_composition(self, skill_ids: Sequence[str]) -> ResolvedSkillComposition: ...

    def validate_skill_ids(self, skill_ids: AbstractSet[str] | Sequence[str]) -> None: ...

    def validate_skills(self, skills: Sequence[SkillManifest]) -> None: ...

    def resolve_skills(self, skills: Sequence[SkillManifest]) -> ResolvedSkillPack: ...

    def resolve_skills_composition(
        self,
        skills: Sequence[SkillManifest],
    ) -> ResolvedSkillComposition: ...


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

    def _materialized_manifest(self, skill_id: str) -> SkillManifest:
        if not self._skill_registry.has(skill_id):
            raise SkillResolutionError(f"Unknown skill_id: {skill_id}")
        return self._skill_registry.get(skill_id).manifest

    def _verify_manifest_version(
        self,
        manifest: SkillManifest,
        requested_version: str,
    ) -> None:
        if manifest.version != requested_version:
            raise SkillResolutionError(
                f"Skill version mismatch for '{manifest.skill_id}': "
                f"requested {requested_version}, "
                f"registry materialized {manifest.version}",
            )

    def _verify_pinned_version(self, skill_id: str, requested_version: str) -> SkillManifest:
        manifest = self._materialized_manifest(skill_id)
        self._verify_manifest_version(manifest, requested_version)
        return manifest

    def _normalize_root_manifests(
        self,
        skills: Sequence[SkillManifest],
    ) -> tuple[SkillManifest, ...]:
        seen_versions: dict[str, str] = {}
        order: list[SkillManifest] = []
        for manifest in skills:
            skill_id = manifest.skill_id.strip()
            if not skill_id:
                raise SkillResolutionError("SkillManifest.skill_id must be non-empty")
            if skill_id in seen_versions:
                if seen_versions[skill_id] != manifest.version:
                    raise SkillResolutionError(
                        f"conflicting root version requirements for skill {skill_id}",
                    )
                continue
            seen_versions[skill_id] = manifest.version
            order.append(manifest)
        return tuple(order)

    def _expand_skill_dependencies(
        self,
        root_ids: Sequence[str],
        root_pins: dict[str, str],
    ) -> tuple[tuple[str, ...], dict[str, ResolvedSkillRole], dict[str, SkillManifest]]:
        order: list[str] = []
        seen: set[str] = set()
        visiting: set[str] = set()
        roles: dict[str, ResolvedSkillRole] = {}
        observed_manifests: dict[str, SkillManifest] = {}
        root_id_set = frozenset(root_ids)

        def visit(skill_id: str) -> None:
            if skill_id in seen:
                return
            if skill_id in visiting:
                raise SkillResolutionError(f"Cyclic requires_skills involving: {skill_id}")
            visiting.add(skill_id)
            if skill_id in root_pins:
                manifest = self._verify_pinned_version(skill_id, root_pins[skill_id])
            else:
                manifest = self._materialized_manifest(skill_id)
            observed_manifests[skill_id] = manifest
            for dep in manifest.requires_skills:
                dep_id = dep.strip()
                if dep_id:
                    visit(dep_id)
            visiting.remove(skill_id)
            seen.add(skill_id)
            order.append(skill_id)
            roles[skill_id] = (
                ResolvedSkillRole.ROOT if skill_id in root_id_set else ResolvedSkillRole.TRANSITIVE
            )

        for skill_id in root_ids:
            sid = skill_id.strip()
            if sid:
                visit(sid)
        return tuple(order), roles, observed_manifests

    def _build_composition(
        self,
        normalized_ids: tuple[str, ...],
        roles: dict[str, ResolvedSkillRole],
        root_pins: dict[str, str],
        observed_manifests: dict[str, SkillManifest],
    ) -> ResolvedSkillComposition:
        if not normalized_ids:
            pack = ResolvedSkillPack(
                resolved_skills=(),
                tool_ids=frozenset(),
                prompt_instruction_ids=frozenset(),
                policy_fragment_ids=frozenset(),
                risk_tier=SkillRiskTier.LOW,
                snapshot_digest=compute_resolved_skill_pack_digest(()),
            )
            return ResolvedSkillComposition(pack=pack, observed_manifests=())

        resolved_refs: list[ResolvedSkillRef] = []
        observed_in_order: list[SkillManifest] = []
        tool_ids: set[str] = set()
        prompt_ids: set[str] = set()
        policy_ids: set[str] = set()
        max_risk = SkillRiskTier.LOW
        risk_order = list(SkillRiskTier)

        for skill_id in normalized_ids:
            manifest = observed_manifests[skill_id]
            if skill_id in root_pins:
                self._verify_manifest_version(manifest, root_pins[skill_id])
                resolution_mode = SkillVersionResolutionMode.PINNED
            else:
                resolution_mode = SkillVersionResolutionMode.MATERIALIZED
            ref = ResolvedSkillRef.from_manifest(
                manifest,
                resolution_mode=resolution_mode,
                role=roles[skill_id],
            )
            resolved_refs.append(ref)
            observed_in_order.append(manifest)
            tool_ids.update(manifest.tool_ids)
            prompt_ids.update(manifest.prompt_instruction_ids)
            if manifest.policy_fragment_id:
                policy_ids.add(manifest.policy_fragment_id)
            if risk_order.index(manifest.risk_tier) > risk_order.index(max_risk):
                max_risk = manifest.risk_tier

        if self._tool_registry is not None:
            self._validate_tools_exist(tool_ids)

        resolved_tuple = tuple(resolved_refs)
        pack = ResolvedSkillPack(
            resolved_skills=resolved_tuple,
            tool_ids=frozenset(tool_ids),
            prompt_instruction_ids=frozenset(prompt_ids),
            policy_fragment_ids=frozenset(policy_ids),
            risk_tier=max_risk,
            snapshot_digest=compute_resolved_skill_pack_digest(resolved_tuple),
        )
        return ResolvedSkillComposition(
            pack=pack,
            observed_manifests=tuple(observed_in_order),
        )

    def _resolve_composition(
        self,
        root_ids: Sequence[str],
        root_pins: dict[str, str],
    ) -> ResolvedSkillComposition:
        normalized, roles, observed_manifests = self._expand_skill_dependencies(root_ids, root_pins)
        return self._build_composition(normalized, roles, root_pins, observed_manifests)

    def resolve_composition(self, skill_ids: Sequence[str]) -> ResolvedSkillComposition:
        roots = tuple(dict.fromkeys(sid.strip() for sid in skill_ids if sid.strip()))
        return self._resolve_composition(roots, {})

    def resolve(self, skill_ids: Sequence[str]) -> ResolvedSkillPack:
        return self.resolve_composition(skill_ids).pack

    def validate_skill_ids(self, skill_ids: AbstractSet[str] | Sequence[str]) -> None:
        for skill_id in skill_ids:
            sid = skill_id.strip()
            if sid and not self._skill_registry.has(sid):
                raise SkillResolutionError(f"Unknown skill_id: {sid}")

    def validate_skills(self, skills: Sequence[SkillManifest]) -> None:
        for manifest in self._normalize_root_manifests(skills):
            self._verify_pinned_version(manifest.skill_id, manifest.version)

    def resolve_skills_composition(self, skills: Sequence[SkillManifest]) -> ResolvedSkillComposition:
        root_manifests = self._normalize_root_manifests(skills)
        root_pins = {manifest.skill_id: manifest.version for manifest in root_manifests}
        root_ids = tuple(manifest.skill_id for manifest in root_manifests)
        return self._resolve_composition(root_ids, root_pins)

    def resolve_skills(self, skills: Sequence[SkillManifest]) -> ResolvedSkillPack:
        return self.resolve_skills_composition(skills).pack

    def validate_tool_contracts(self, tools: Sequence[ToolContract]) -> None:
        if self._tool_registry is None:
            return
        missing = [
            tool.tool_id
            for tool in tools
            if tool.tool_id.strip() and not self._tool_registry.has(tool.tool_id)
        ]
        if missing:
            raise SkillResolutionError(
                f"extra_tools reference tool_id(s) not in ToolRegistry: {', '.join(sorted(missing))}"
            )

    def _validate_tools_exist(self, tool_ids: set[str]) -> None:
        assert self._tool_registry is not None
        missing = [tid for tid in sorted(tool_ids) if not self._tool_registry.has(tid)]
        if missing:
            raise SkillResolutionError(
                f"Skill references tool_id(s) not in ToolRegistry: {', '.join(missing)}"
            )
