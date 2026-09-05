# © Artur Czarnecki. All rights reserved.

"""Canonical profile resolution engine (P1.1)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.applications._shared.profile_resolution.field_resolvers import (
    DEFAULT_FIELD_RESOLVERS,
    ProfileFieldResolveContext,
    ProfileFieldResolver,
    delta_update_for_path,
    resolver_index,
)
from intergrax.applications._shared.profile_resolution.fingerprint import (
    compute_effective_profile_fingerprint,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution import (
    ProfileDelta,
    ProfileFieldUpdate,
    ProfileLayer,
    ProfileLayerConflictError,
    ProfileLayerInput,
    ProfileLayerResolution,
    ProfileResolution,
    ProfileResolutionDecision,
    ProfileResolutionError,
    profile_layer_sort_key,
)


def _tracked_paths() -> tuple[str, ...]:
    return (
        "meta.execution_mode",
        "capabilities.llm",
        "capabilities.tools",
        "governance.cost",
    )


def _value_at_path(profile: ApplicationEnvironmentProfile, path: str) -> object:
    if path == "meta.execution_mode":
        return profile.execution_mode
    if path == "capabilities.llm":
        return profile.llm_profile
    if path == "capabilities.tools":
        return profile.tool_profile
    if path == "governance.cost":
        return profile.cost_profile
    raise ProfileResolutionError(f"unsupported tracked path: {path}")


def _set_value_at_path(
    profile: ApplicationEnvironmentProfile,
    path: str,
    value: object,
) -> ApplicationEnvironmentProfile:
    if path == "meta.execution_mode":
        return profile.model_copy(
            update={"meta": profile.meta.model_copy(update={"execution_mode": value})},
        )
    if path == "capabilities.llm":
        return profile.model_copy(
            update={
                "capabilities": profile.capabilities.model_copy(update={"llm": value}),
            },
        )
    if path == "capabilities.tools":
        return profile.model_copy(
            update={
                "capabilities": profile.capabilities.model_copy(update={"tools": value}),
            },
        )
    if path == "governance.cost":
        return profile.model_copy(
            update={"governance": profile.governance.model_copy(update={"cost": value})},
        )
    raise ProfileResolutionError(f"unsupported tracked path: {path}")


def _application_opinion_absent(path: str, value: object) -> bool:
    if path == "capabilities.llm":
        return value is None
    return False


def _application_layer_decisions(
    *,
    upstream: ApplicationEnvironmentProfile,
    configured: ApplicationEnvironmentProfile,
    resolvers: dict[str, ProfileFieldResolver],
    expressed_paths: frozenset[str],
) -> tuple[ApplicationEnvironmentProfile, frozenset[str], tuple[ProfileResolutionDecision, ...]]:
    """
    Apply configured application opinions through the same resolver pipeline as overlays.

    Untracked fields keep the configured application baseline. Tracked authority paths
    cannot widen stricter upstream platform/product/run policy.
    """
    effective = configured.model_copy(deep=True)
    resolution_context = upstream.model_copy(deep=True)
    decisions: list[ProfileResolutionDecision] = []
    updated_expressed_paths = set(expressed_paths)

    for path in _tracked_paths():
        opinion = _value_at_path(configured, path)
        if _application_opinion_absent(path, opinion):
            continue
        resolver = resolvers[path]
        result = resolver.resolve(
            profile=resolution_context,
            update=ProfileFieldUpdate(value=opinion),
            source_layer=ProfileLayer.APPLICATION,
            context=ProfileFieldResolveContext(expressed_paths=expressed_paths),
        )
        updated_expressed_paths.add(path)
        expressed_paths = frozenset(updated_expressed_paths)
        resolution_context = result.profile
        effective = _set_value_at_path(
            effective,
            path,
            _value_at_path(resolution_context, path),
        )
        decisions.extend(result.decisions)

    return effective, frozenset(updated_expressed_paths), tuple(decisions)


def _apply_delta(
    *,
    profile: ApplicationEnvironmentProfile,
    delta: ProfileDelta,
    source_layer: ProfileLayer,
    resolvers: dict[str, ProfileFieldResolver],
    expressed_paths: frozenset[str],
) -> tuple[ApplicationEnvironmentProfile, frozenset[str], tuple[ProfileResolutionDecision, ...]]:
    effective = profile
    decisions: list[ProfileResolutionDecision] = []
    updated_expressed_paths = set(expressed_paths)
    for path in delta.opinion_paths():
        resolver = resolvers.get(path)
        if resolver is None:
            raise ProfileResolutionError(f"no resolver registered for delta path: {path}")
        update = delta_update_for_path(delta, path)
        if update is None:
            raise ProfileResolutionError(f"delta missing opinion for declared path: {path}")
        result = resolver.resolve(
            profile=effective,
            update=update,
            source_layer=source_layer,
            context=ProfileFieldResolveContext(expressed_paths=expressed_paths),
        )
        updated_expressed_paths.add(path)
        expressed_paths = frozenset(updated_expressed_paths)
        effective = result.profile
        decisions.extend(result.decisions)
    return effective, frozenset(updated_expressed_paths), tuple(decisions)


def _normalize_layer_inputs(
    layers: Sequence[ProfileLayerInput],
) -> tuple[ProfileLayerInput, ...]:
    seen: set[ProfileLayer] = set()
    normalized: list[ProfileLayerInput] = []
    for item in layers:
        if item.layer in seen:
            raise ProfileLayerConflictError(item.layer)
        seen.add(item.layer)
        normalized.append(item)
    normalized.sort(key=lambda item: profile_layer_sort_key(item.layer))
    return tuple(normalized)


def _partition_layers(
    layers: Sequence[ProfileLayerInput],
) -> tuple[tuple[ProfileLayerInput, ...], tuple[ProfileLayerInput, ...]]:
    application_order = profile_layer_sort_key(ProfileLayer.APPLICATION)
    pre: list[ProfileLayerInput] = []
    post: list[ProfileLayerInput] = []
    for item in layers:
        order = profile_layer_sort_key(item.layer)
        if order < application_order:
            pre.append(item)
        else:
            post.append(item)
    return tuple(pre), tuple(post)


def resolve_profile(
    application_profile: ApplicationEnvironmentProfile,
    *,
    layers: Sequence[ProfileLayerInput] = (),
    field_resolvers: Sequence[ProfileFieldResolver] = DEFAULT_FIELD_RESOLVERS,
) -> ProfileResolution:
    """
    Resolve configured composition into effective profile evidence.

    ``application_profile`` remains the sole Tier-3 composition authority input.
    Overlay layers supply sparse typed deltas only.
    """
    configured_application = application_profile.model_copy(deep=True)
    resolvers = resolver_index(tuple(field_resolvers))
    normalized_layers = _normalize_layer_inputs(layers)
    pre_application_layers, post_application_layers = _partition_layers(normalized_layers)

    effective = ApplicationEnvironmentProfile()
    layer_records: list[ProfileLayerResolution] = []
    decisions: list[ProfileResolutionDecision] = []
    expressed_paths: frozenset[str] = frozenset()

    for layer_input in pre_application_layers:
        assert layer_input.delta is not None
        effective, expressed_paths, layer_decisions = _apply_delta(
            profile=effective,
            delta=layer_input.delta,
            source_layer=layer_input.layer,
            resolvers=resolvers,
            expressed_paths=expressed_paths,
        )
        layer_records.append(
            ProfileLayerResolution(
                layer=layer_input.layer,
                revision=layer_input.revision,
                delta=layer_input.delta,
            ),
        )
        decisions.extend(layer_decisions)

    effective, expressed_paths, application_decisions = _application_layer_decisions(
        upstream=effective,
        configured=configured_application,
        resolvers=resolvers,
        expressed_paths=expressed_paths,
    )
    decisions.extend(application_decisions)
    layer_records.append(
        ProfileLayerResolution(
            layer=ProfileLayer.APPLICATION,
            revision=configured_application.profile_id,
            delta=None,
        ),
    )

    for layer_input in post_application_layers:
        assert layer_input.delta is not None
        effective, expressed_paths, layer_decisions = _apply_delta(
            profile=effective,
            delta=layer_input.delta,
            source_layer=layer_input.layer,
            resolvers=resolvers,
            expressed_paths=expressed_paths,
        )
        layer_records.append(
            ProfileLayerResolution(
                layer=layer_input.layer,
                revision=layer_input.revision,
                delta=layer_input.delta,
            ),
        )
        decisions.extend(layer_decisions)

    fingerprint = compute_effective_profile_fingerprint(effective)
    return ProfileResolution(
        effective_profile=effective,
        layers=tuple(layer_records),
        decisions=tuple(decisions),
        warnings=(),
        dependency_failures=(),
        degraded_capabilities=(),
        fingerprint=fingerprint,
    )
