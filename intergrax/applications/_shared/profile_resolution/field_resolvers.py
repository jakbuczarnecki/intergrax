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
from intergrax.tools.registry.profile import ToolProfile


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


def _upstream_tool_scope(tool_profile: ToolProfile) -> set[str] | None:
    if tool_profile.register_all_catalog_bundles:
        return None
    if tool_profile.enabled:
        return set(tool_profile.enabled)
    return None


class ToolProfileFieldResolver:
    path = "capabilities.tools"

    def resolve(
        self,
        *,
        profile: ApplicationEnvironmentProfile,
        update: ProfileFieldUpdate[object],
        source_layer: ProfileLayer,
    ) -> ProfileFieldResolveResult:
        upstream = profile.tool_profile
        if update.action == "clear":
            cleared = ToolProfile()
            return ProfileFieldResolveResult(
                profile=profile.model_copy(
                    update={
                        "capabilities": profile.capabilities.model_copy(
                            update={"tools": cleared},
                        ),
                    },
                ),
                decisions=(
                    _decision(
                        path=self.path,
                        requested=None,
                        source_layer=source_layer,
                        previous=upstream,
                        kind=ProfileResolutionDecisionKind.APPLIED,
                        effective=cleared,
                        reason="explicit clear",
                    ),
                ),
            )

        requested = ToolProfile.model_validate(update.value)
        upstream_scope = _upstream_tool_scope(upstream)
        decisions: list[ProfileResolutionDecision] = []
        effective_enabled = list(requested.enabled)
        if upstream_scope is not None and requested.enabled:
            allowed = sorted(set(requested.enabled).intersection(upstream_scope))
            rejected = sorted(set(requested.enabled).difference(upstream_scope))
            effective_enabled = allowed
            if rejected:
                decisions.append(
                    _decision(
                        path=f"{self.path}.enabled",
                        requested=rejected,
                        source_layer=source_layer,
                        previous=sorted(upstream_scope),
                        kind=ProfileResolutionDecisionKind.CLAMPED,
                        effective=allowed,
                        reason="upstream host authority does not grant requested tools",
                    ),
                )
        effective = requested.model_copy(update={"enabled": effective_enabled})
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
    ) -> ProfileFieldResolveResult:
        upstream = profile.llm_profile
        if update.action == "clear":
            default_provider = upstream.provider if upstream is not None else LLMProvider.OPENAI
            cleared = LLMProfile(provider=default_provider)
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
                        kind=ProfileResolutionDecisionKind.APPLIED,
                        effective=cleared,
                        reason="explicit clear",
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
    ) -> ProfileFieldResolveResult:
        upstream = profile.execution_mode
        if update.action == "clear":
            cleared = ExecutionMode.BALANCED
            return ProfileFieldResolveResult(
                profile=profile.model_copy(
                    update={"meta": profile.meta.model_copy(update={"execution_mode": cleared})},
                ),
                decisions=(
                    _decision(
                        path=self.path,
                        requested=None,
                        source_layer=source_layer,
                        previous=upstream,
                        kind=ProfileResolutionDecisionKind.APPLIED,
                        effective=cleared,
                        reason="explicit clear to balanced default",
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


class CostProfileFieldResolver:
    path = "governance.cost"

    def resolve(
        self,
        *,
        profile: ApplicationEnvironmentProfile,
        update: ProfileFieldUpdate[object],
        source_layer: ProfileLayer,
    ) -> ProfileFieldResolveResult:
        upstream = profile.cost_profile
        if update.action == "clear":
            cleared = CostProfile()
            return ProfileFieldResolveResult(
                profile=profile.model_copy(
                    update={
                        "governance": profile.governance.model_copy(update={"cost": cleared}),
                    },
                ),
                decisions=(
                    _decision(
                        path=self.path,
                        requested=None,
                        source_layer=source_layer,
                        previous=upstream,
                        kind=ProfileResolutionDecisionKind.APPLIED,
                        effective=cleared,
                        reason="explicit clear",
                    ),
                ),
            )

        requested = CostProfile.model_validate(update.value)
        decisions: list[ProfileResolutionDecision] = []
        max_tool_calls, kind, reason = _narrow_optional_limit(
            upstream.max_tool_calls,
            requested.max_tool_calls,
        )
        effective = upstream.model_copy(update={"max_tool_calls": max_tool_calls})
        if requested.max_tool_calls is not None:
            decisions.append(
                _decision(
                    path=f"{self.path}.max_tool_calls",
                    requested=requested.max_tool_calls,
                    source_layer=source_layer,
                    previous=upstream.max_tool_calls,
                    kind=kind,
                    effective=max_tool_calls,
                    reason=reason,
                ),
            )
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
