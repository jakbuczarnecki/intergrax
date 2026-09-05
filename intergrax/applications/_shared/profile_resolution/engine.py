# © Artur Czarnecki. All rights reserved.

"""Canonical profile resolution engine (P1.1)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.applications._shared.profile_resolution.field_resolvers import (
    DEFAULT_FIELD_RESOLVERS,
    ProfileFieldResolver,
    delta_update_for_path,
    resolver_index,
)
from intergrax.applications._shared.profile_resolution.fingerprint import (
    compute_effective_profile_fingerprint,
)
from intergrax.applications._shared.profile_resolution.redaction import encode_provenance_value
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution import (
    ProfileDelta,
    ProfileLayer,
    ProfileLayerConflictError,
    ProfileLayerInput,
    ProfileLayerResolution,
    ProfileResolution,
    ProfileResolutionDecision,
    ProfileResolutionDecisionKind,
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


def _application_layer_decisions(
    *,
    previous: ApplicationEnvironmentProfile,
    configured: ApplicationEnvironmentProfile,
) -> tuple[ApplicationEnvironmentProfile, tuple[ProfileResolutionDecision, ...]]:
    decisions: list[ProfileResolutionDecision] = []
    for path in _tracked_paths():
        before = _value_at_path(previous, path)
        after = _value_at_path(configured, path)
        if before == after:
            decisions.append(
                ProfileResolutionDecision(
                    path=path,
                    requested_value=encode_provenance_value(path, after),
                    source_layer=ProfileLayer.APPLICATION,
                    previous_value=encode_provenance_value(path, before),
                    decision=ProfileResolutionDecisionKind.UNCHANGED,
                    effective_value=encode_provenance_value(path, after),
                    reason="application configured value retained",
                ),
            )
            continue
        decisions.append(
            ProfileResolutionDecision(
                path=path,
                requested_value=encode_provenance_value(path, after),
                source_layer=ProfileLayer.APPLICATION,
                previous_value=encode_provenance_value(path, before),
                decision=ProfileResolutionDecisionKind.APPLIED,
                effective_value=encode_provenance_value(path, after),
                reason="application configured composition applied",
            ),
        )
    return configured.model_copy(deep=True), tuple(decisions)


def _apply_delta(
    *,
    profile: ApplicationEnvironmentProfile,
    delta: ProfileDelta,
    source_layer: ProfileLayer,
    resolvers: dict[str, ProfileFieldResolver],
) -> tuple[ApplicationEnvironmentProfile, tuple[ProfileResolutionDecision, ...]]:
    effective = profile
    decisions: list[ProfileResolutionDecision] = []
    for path in delta.opinion_paths():
        resolver = resolvers.get(path)
        if resolver is None:
            raise ProfileResolutionError(f"no resolver registered for delta path: {path}")
        update = delta_update_for_path(delta, path)
        if update is None:
            raise ProfileResolutionError(f"delta missing opinion for declared path: {path}")
        result = resolver.resolve(profile=effective, update=update, source_layer=source_layer)
        effective = result.profile
        decisions.extend(result.decisions)
    return effective, tuple(decisions)


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

    for layer_input in pre_application_layers:
        assert layer_input.delta is not None
        effective, layer_decisions = _apply_delta(
            profile=effective,
            delta=layer_input.delta,
            source_layer=layer_input.layer,
            resolvers=resolvers,
        )
        layer_records.append(
            ProfileLayerResolution(
                layer=layer_input.layer,
                revision=layer_input.revision,
                delta=layer_input.delta,
            ),
        )
        decisions.extend(layer_decisions)

    effective, application_decisions = _application_layer_decisions(
        previous=effective,
        configured=configured_application,
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
        effective, layer_decisions = _apply_delta(
            profile=effective,
            delta=layer_input.delta,
            source_layer=layer_input.layer,
            resolvers=resolvers,
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
