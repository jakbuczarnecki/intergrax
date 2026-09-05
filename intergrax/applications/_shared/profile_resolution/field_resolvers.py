# © Artur Czarnecki. All rights reserved.

"""Domain-owned field resolvers for profile layering (P1.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.applications._shared.profile_resolution.redaction import encode_provenance_value
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.profile_resolution import (
    ProfileFieldUpdate,
    ProfileLayer,
    ProfileResolutionDecision,
    ProfileResolutionDecisionKind,
    ProfileResolutionError,
)
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.tools.registry.factory import enabled_tool_ids_for_profile
from intergrax.tools.registry.profile import ToolProfile


@dataclass(frozen=True, slots=True)
class ProfileFieldResolveContext:
    """Layer-resolution context for domain-owned authority semantics."""

    expressed_paths: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ProfileFieldResolveResult:
    """Outcome of resolving one delta field path."""

    profile: ApplicationEnvironmentProfile
    decisions: tuple[ProfileResolutionDecision, ...]


class ProfileFieldResolver(Protocol):
    """Bounded extension point for domain-owned field semantics."""

    @property
    def path(self) -> str:
        """Canonical dotted path owned by this resolver."""

    def resolve(
        self,
        *,
        profile: ApplicationEnvironmentProfile,
        update: ProfileFieldUpdate[object],
        source_layer: ProfileLayer,
        context: ProfileFieldResolveContext = ProfileFieldResolveContext(),
    ) -> ProfileFieldResolveResult:
        """Apply one sparse field opinion with authority semantics."""


def _decision(
    *,
    path: str,
    requested: object | None,
    source_layer: ProfileLayer,
    previous: object | None,
    kind: ProfileResolutionDecisionKind,
    effective: object | None,
    reason: str,
) -> ProfileResolutionDecision:
    return ProfileResolutionDecision(
        path=path,
        requested_value=encode_provenance_value(path, requested),
        source_layer=source_layer,
        previous_value=encode_provenance_value(path, previous),
        decision=kind,
        effective_value=encode_provenance_value(path, effective),
        reason=reason,
    )


def _tool_profile_from_allowed_ids(
    allowed_ids: frozenset[str],
    *,
    register_all_catalog_bundles: bool = False,
) -> ToolProfile:
    if register_all_catalog_bundles:
        return ToolProfile(register_all_catalog_bundles=True)
    return ToolProfile(enabled=sorted(allowed_ids))


def _effective_tool_profile_after_authority(
    *,
    upstream: ToolProfile,
    requested: ToolProfile,
    allowed_ids: frozenset[str],
    rejected_ids: frozenset[str],
) -> ToolProfile:
    if (
        upstream.register_all_catalog_bundles
        and requested.register_all_catalog_bundles
        and not rejected_ids
    ):
        return ToolProfile(register_all_catalog_bundles=True)
    if rejected_ids:
        return _tool_profile_from_allowed_ids(allowed_ids)
    return requested.model_copy(deep=True)


def _upstream_tool_authority_scope(
    tool_profile: ToolProfile,
    *,
    expressed: bool,
) -> tuple[bool, frozenset[str]]:
    """
    Return ``(unrestricted_catalog, allowed_ids)``.

    When ``unrestricted_catalog`` is True, downstream catalog selection is not
    clamped by explicit upstream ids. When the path has no expressed upstream
    opinion yet, downstream may establish authority.
    """
    if tool_profile.register_all_catalog_bundles:
        return True, frozenset()
    if tool_profile.enabled or tool_profile.enabled_bundles:
        return False, frozenset(enabled_tool_ids_for_profile(tool_profile))
    if expressed:
        return False, frozenset()
    return False, frozenset()


def _resolve_tool_profile_authority(
    *,
    upstream: ToolProfile,
    requested: ToolProfile,
    upstream_expressed: bool,
) -> tuple[ToolProfile, frozenset[str], frozenset[str]]:
    unrestricted, upstream_ids = _upstream_tool_authority_scope(
        upstream,
        expressed=upstream_expressed,
    )
    requested_ids = frozenset(enabled_tool_ids_for_profile(requested))

    if unrestricted:
        allowed_ids = requested_ids
        rejected_ids = frozenset()
    elif not upstream_expressed and not upstream_ids:
        allowed_ids = requested_ids
        rejected_ids = frozenset()
    elif requested.register_all_catalog_bundles:
        allowed_ids = upstream_ids
        rejected_ids = requested_ids.difference(upstream_ids)
    else:
        allowed_ids = requested_ids.intersection(upstream_ids)
        rejected_ids = requested_ids.difference(upstream_ids)

    effective = _effective_tool_profile_after_authority(
        upstream=upstream,
        requested=requested,
        allowed_ids=allowed_ids,
        rejected_ids=rejected_ids,
    )
    if unrestricted and requested.register_all_catalog_bundles and not rejected_ids:
        effective = ToolProfile(register_all_catalog_bundles=True)
    return effective, allowed_ids, rejected_ids


class ToolProfileFieldResolver:
    path = "capabilities.tools"

    def resolve(
        self,
        *,
        profile: ApplicationEnvironmentProfile,
        update: ProfileFieldUpdate[object],
        source_layer: ProfileLayer,
        context: ProfileFieldResolveContext = ProfileFieldResolveContext(),
    ) -> ProfileFieldResolveResult:
        upstream = profile.tool_profile
        upstream_expressed = self.path in context.expressed_paths
        if update.action == "clear":
            return ProfileFieldResolveResult(
                profile=profile.model_copy(
                    update={
                        "capabilities": profile.capabilities.model_copy(
                            update={"tools": upstream.model_copy(deep=True)},
                        ),
                    },
                ),
                decisions=(
                    _decision(
                        path=self.path,
                        requested=None,
                        source_layer=source_layer,
                        previous=upstream,
                        kind=ProfileResolutionDecisionKind.UNCHANGED,
                        effective=upstream,
                        reason="clear removes downstream opinion; upstream tool authority retained",
                    ),
                ),
            )

        requested = ToolProfile.model_validate(update.value)
        effective, allowed_ids, rejected_ids = _resolve_tool_profile_authority(
            upstream=upstream,
            requested=requested,
            upstream_expressed=upstream_expressed,
        )
        decisions: list[ProfileResolutionDecision] = []
        if rejected_ids:
            decisions.append(
                _decision(
                    path=f"{self.path}.enabled",
                    requested=sorted(rejected_ids),
                    source_layer=source_layer,
                    previous=sorted(allowed_ids),
                    kind=ProfileResolutionDecisionKind.CLAMPED,
                    effective=sorted(allowed_ids),
                    reason="upstream host authority does not grant requested tools",
                ),
            )
        if not decisions:
            kind = (
                ProfileResolutionDecisionKind.UNCHANGED
                if effective == upstream
                else ProfileResolutionDecisionKind.APPLIED
            )
            decisions.append(
                _decision(
                    path=self.path,
                    requested=requested,
                    source_layer=source_layer,
                    previous=upstream,
                    kind=kind,
                    effective=effective,
                    reason="tool authority intersection applied",
                ),
            )
        return ProfileFieldResolveResult(
            profile=profile.model_copy(
                update={
                    "capabilities": profile.capabilities.model_copy(
                        update={"tools": effective},
                    ),
                },
            ),
            decisions=tuple(decisions),
        )


class LLMProfileFieldResolver:
    path = "capabilities.llm"

    def resolve(
        self,
        *,
        profile: ApplicationEnvironmentProfile,
        update: ProfileFieldUpdate[object],
        source_layer: ProfileLayer,
        context: ProfileFieldResolveContext = ProfileFieldResolveContext(),
    ) -> ProfileFieldResolveResult:
        upstream = profile.llm_profile
        if update.action == "clear":
            if upstream is not None:
                cleared = upstream.model_copy(deep=True)
                reason = "clear removes downstream opinion; upstream llm retained"
            else:
                cleared = LLMProfile.lab()
                reason = "clear with no upstream opinion; canonical lab default applied"
            return ProfileFieldResolveResult(
                profile=profile.model_copy(
                    update={
                        "capabilities": profile.capabilities.model_copy(
                            update={"llm": cleared},
                        ),
                    },
                ),
                decisions=(
                    _decision(
                        path=self.path,
                        requested=None,
                        source_layer=source_layer,
                        previous=upstream,
                        kind=ProfileResolutionDecisionKind.UNCHANGED
                        if upstream is not None and cleared == upstream
                        else ProfileResolutionDecisionKind.APPLIED,
                        effective=cleared,
                        reason=reason,
                    ),
                ),
            )

        requested = LLMProfile.model_validate(update.value)
        if upstream is None:
            effective = requested
        else:
            effective = upstream.model_copy(
                update={
                    "provider": requested.provider,
                    "model": requested.model if requested.model is not None else upstream.model,
                },
            )
        previous = upstream if upstream is not None else LLMProfile(provider=requested.provider)
        kind = (
            ProfileResolutionDecisionKind.UNCHANGED
            if upstream is not None and effective == upstream
            else ProfileResolutionDecisionKind.APPLIED
        )
        return ProfileFieldResolveResult(
            profile=profile.model_copy(
                update={
                    "capabilities": profile.capabilities.model_copy(
                        update={"llm": effective},
                    ),
                },
            ),
            decisions=(
                _decision(
                    path=self.path,
                    requested=requested,
                    source_layer=source_layer,
                    previous=previous,
                    kind=kind,
                    effective=effective,
                    reason="scalar llm override — last allowed layer wins",
                ),
            ),
        )


class ExecutionModeFieldResolver:
    path = "meta.execution_mode"

    def resolve(
        self,
        *,
        profile: ApplicationEnvironmentProfile,
        update: ProfileFieldUpdate[object],
        source_layer: ProfileLayer,
        context: ProfileFieldResolveContext = ProfileFieldResolveContext(),
    ) -> ProfileFieldResolveResult:
        upstream = profile.execution_mode
        if update.action == "clear":
            return ProfileFieldResolveResult(
                profile=profile.model_copy(
                    update={"meta": profile.meta.model_copy(update={"execution_mode": upstream})},
                ),
                decisions=(
                    _decision(
                        path=self.path,
                        requested=None,
                        source_layer=source_layer,
                        previous=upstream,
                        kind=ProfileResolutionDecisionKind.UNCHANGED,
                        effective=upstream,
                        reason="clear removes downstream opinion; upstream execution mode retained",
                    ),
                ),
            )

        requested = ExecutionMode(update.value)
        kind = (
            ProfileResolutionDecisionKind.UNCHANGED
            if requested == upstream
            else ProfileResolutionDecisionKind.APPLIED
        )
        return ProfileFieldResolveResult(
            profile=profile.model_copy(
                update={"meta": profile.meta.model_copy(update={"execution_mode": requested})},
            ),
            decisions=(
                _decision(
                    path=self.path,
                    requested=requested,
                    source_layer=source_layer,
                    previous=upstream,
                    kind=kind,
                    effective=requested,
                    reason="scalar execution mode override — last allowed layer wins",
                ),
            ),
        )


def _narrow_optional_limit(
  upstream: int | None,
  requested: int | None,
) -> tuple[int | None, ProfileResolutionDecisionKind, str]:
    if requested is None:
        return upstream, ProfileResolutionDecisionKind.UNCHANGED, "no requested limit"
    if upstream is None:
        return requested, ProfileResolutionDecisionKind.APPLIED, "upstream had no limit"
    if requested <= upstream:
        kind = ProfileResolutionDecisionKind.APPLIED if requested != upstream else ProfileResolutionDecisionKind.UNCHANGED
        return requested, kind, "budget narrowed within upstream authority"
    return upstream, ProfileResolutionDecisionKind.CLAMPED, "budget overlay cannot widen upstream allowed limit"


_COST_AUTHORITY_LIMIT_FIELDS: tuple[str, ...] = (
    "max_total_tokens",
    "max_llm_calls",
    "max_tool_calls",
    "max_planner_iterations",
)


class CostProfileFieldResolver:
    path = "governance.cost"

    def resolve(
        self,
        *,
        profile: ApplicationEnvironmentProfile,
        update: ProfileFieldUpdate[object],
        source_layer: ProfileLayer,
        context: ProfileFieldResolveContext = ProfileFieldResolveContext(),
    ) -> ProfileFieldResolveResult:
        upstream = profile.cost_profile
        if update.action == "clear":
            retained = upstream.model_copy(deep=True)
            return ProfileFieldResolveResult(
                profile=profile.model_copy(
                    update={
                        "governance": profile.governance.model_copy(update={"cost": retained}),
                    },
                ),
                decisions=(
                    _decision(
                        path=self.path,
                        requested=None,
                        source_layer=source_layer,
                        previous=upstream,
                        kind=ProfileResolutionDecisionKind.UNCHANGED,
                        effective=retained,
                        reason="clear removes downstream opinion; upstream budget authority retained",
                    ),
                ),
            )

        requested = CostProfile.model_validate(update.value)
        decisions: list[ProfileResolutionDecision] = []
        effective_updates: dict[str, object] = {}
        for field_name in _COST_AUTHORITY_LIMIT_FIELDS:
            upstream_limit = getattr(upstream, field_name)
            requested_limit = getattr(requested, field_name)
            narrowed, kind, reason = _narrow_optional_limit(upstream_limit, requested_limit)
            effective_updates[field_name] = narrowed
            if requested_limit is not None:
                decisions.append(
                    _decision(
                        path=f"{self.path}.{field_name}",
                        requested=requested_limit,
                        source_layer=source_layer,
                        previous=upstream_limit,
                        kind=kind,
                        effective=narrowed,
                        reason=reason,
                    ),
                )
        non_limit_updates = {
            key: value
            for key, value in requested.model_dump().items()
            if key not in _COST_AUTHORITY_LIMIT_FIELDS
        }
        effective = upstream.model_copy(update={**non_limit_updates, **effective_updates})
        if not decisions:
            decisions.append(
                _decision(
                    path=self.path,
                    requested=requested,
                    source_layer=source_layer,
                    previous=upstream,
                    kind=ProfileResolutionDecisionKind.UNCHANGED,
                    effective=effective,
                    reason="no cost opinion produced a change",
                ),
            )
        return ProfileFieldResolveResult(
            profile=profile.model_copy(
                update={"governance": profile.governance.model_copy(update={"cost": effective})},
            ),
            decisions=tuple(decisions),
        )


DEFAULT_FIELD_RESOLVERS: tuple[ProfileFieldResolver, ...] = (
    ToolProfileFieldResolver(),
    LLMProfileFieldResolver(),
    ExecutionModeFieldResolver(),
    CostProfileFieldResolver(),
)


def resolver_index(
    resolvers: tuple[ProfileFieldResolver, ...],
) -> dict[str, ProfileFieldResolver]:
    indexed: dict[str, ProfileFieldResolver] = {}
    for resolver in resolvers:
        if resolver.path in indexed:
            raise ProfileResolutionError(f"duplicate field resolver for path: {resolver.path}")
        indexed[resolver.path] = resolver
    return indexed


def delta_update_for_path(delta: object, path: str) -> ProfileFieldUpdate[object] | None:
    from intergrax.applications.contracts.profile_resolution.delta import ProfileDelta

    if not isinstance(delta, ProfileDelta):
        return None
    mapping = {
        "capabilities.tools": delta.tool_profile,
        "capabilities.llm": delta.llm_profile,
        "meta.execution_mode": delta.execution_mode,
        "governance.cost": delta.cost_profile,
    }
    return mapping.get(path)
