# © Artur Czarnecki. All rights reserved.

"""P1.1 — canonical profile resolution layering and provenance."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.profile_resolution import (
    ProfileFieldResolveResult,
    ProfileFieldResolver,
    resolve_profile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import HostMeta
from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications.contracts.profile_resolution import (
    ProfileDelta,
    ProfileFieldUpdate,
    ProfileLayer,
    ProfileLayerConflictError,
    ProfileLayerInput,
    ProfileResolutionDecisionKind,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _application(
    *,
    provider: LLMProvider = LLMProvider.OPENAI,
    tools: list[str] | None = None,
    execution_mode: ExecutionMode = ExecutionMode.BALANCED,
    max_tool_calls: int | None = None,
) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="resolution.test")
    updates: dict[str, object] = {
        "meta": profile.meta.model_copy(update={"execution_mode": execution_mode}),
        "capabilities": profile.capabilities.model_copy(
            update={
                "llm": LLMProfile(provider=provider, model="gpt-4o-mini"),
                "tools": ToolProfile(enabled=tools or ["search", "calculator"]),
            },
        ),
    }
    if max_tool_calls is not None:
        updates["governance"] = profile.governance.model_copy(
            update={"cost": CostProfile(max_tool_calls=max_tool_calls)},
        )
    return profile.model_copy(update=updates)


def test_resolution_order_scalar_wins() -> None:
    application = _application(provider=LLMProvider.CLAUDE)
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                revision="platform-1",
                delta=ProfileDelta(
                    llm_profile=ProfileFieldUpdate(
                        value=LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o"),
                    ),
                ),
            ),
            ProfileLayerInput(
                layer=ProfileLayer.EXECUTION,
                revision="exec-1",
                delta=ProfileDelta(
                    execution_mode=ProfileFieldUpdate(value=ExecutionMode.STRICT),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.llm_profile.provider == LLMProvider.CLAUDE
    assert resolution.effective_profile.execution_mode == ExecutionMode.STRICT
    layer_order = [item.layer for item in resolution.layers]
    assert layer_order == [
        ProfileLayer.PLATFORM,
        ProfileLayer.APPLICATION,
        ProfileLayer.EXECUTION,
    ]


def test_input_order_independence() -> None:
    application = _application(execution_mode=ExecutionMode.BALANCED)
    layers = (
        ProfileLayerInput(
            layer=ProfileLayer.EXECUTION,
            delta=ProfileDelta(
                execution_mode=ProfileFieldUpdate(value=ExecutionMode.STRICT),
            ),
        ),
        ProfileLayerInput(
            layer=ProfileLayer.PLATFORM,
            delta=ProfileDelta(
                execution_mode=ProfileFieldUpdate(value=ExecutionMode.EXPLORATORY),
            ),
        ),
    )
    first = resolve_profile(application, layers=layers)
    second = resolve_profile(application, layers=tuple(reversed(layers)))
    assert first.effective_profile == second.effective_profile
    assert first.fingerprint == second.fingerprint


def test_tool_authority_clamp_records_rejected_tools() -> None:
    application = _application(tools=["search"])
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search", "calculator"]),
                    ),
                ),
            ),
            ProfileLayerInput(
                layer=ProfileLayer.EXECUTION,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search", "shell"]),
                    ),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.tool_profile.enabled == ["search"]
    clamped = [
        item
        for item in resolution.decisions
        if item.decision == ProfileResolutionDecisionKind.CLAMPED
        and item.source_layer == ProfileLayer.EXECUTION
    ]
    assert clamped
    assert "shell" in (clamped[0].requested_value or "")


def test_provenance_winner_layer_for_llm_override() -> None:
    application = _application(provider=LLMProvider.CLAUDE)
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    llm_profile=ProfileFieldUpdate(
                        value=LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o"),
                    ),
                ),
            ),
        ),
    )
    llm_decisions = [item for item in resolution.decisions if item.path == "capabilities.llm"]
    assert any(item.source_layer == ProfileLayer.APPLICATION for item in llm_decisions)
    assert resolution.effective_profile.llm_profile.provider == LLMProvider.CLAUDE


def test_budget_overlay_cannot_widen_upstream_limit() -> None:
    application = _application(max_tool_calls=10)
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.EXECUTION,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(
                        value=CostProfile(max_tool_calls=25),
                    ),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.cost_profile.max_tool_calls == 10
    clamped = [
        item
        for item in resolution.decisions
        if item.path == "governance.cost.max_tool_calls"
        and item.decision == ProfileResolutionDecisionKind.CLAMPED
    ]
    assert clamped


def test_resolution_does_not_mutate_inputs() -> None:
    application = _application()
    application_copy = application.model_copy(deep=True)
    delta = ProfileDelta(
        execution_mode=ProfileFieldUpdate(value=ExecutionMode.STRICT),
    )
    delta_copy = delta.model_copy(deep=True)
    resolve_profile(
        application,
        layers=(
            ProfileLayerInput(layer=ProfileLayer.EXECUTION, delta=delta),
        ),
    )
    assert application == application_copy
    assert delta == delta_copy


def test_deterministic_fingerprint_for_same_semantics() -> None:
    application = ApplicationEnvironmentProfile(
        meta=HostMeta(profile_id="deterministic.profile", execution_mode=ExecutionMode.STRICT),
    )
    first = resolve_profile(application)
    second = resolve_profile(application.model_copy(deep=True))
    assert first.effective_profile == second.effective_profile
    assert first.fingerprint == second.fingerprint


def test_fingerprint_changes_when_effective_semantics_change() -> None:
    base = resolve_profile(_application(execution_mode=ExecutionMode.BALANCED))
    changed = resolve_profile(_application(execution_mode=ExecutionMode.STRICT))
    assert base.fingerprint != changed.fingerprint


def test_provenance_only_layer_revision_does_not_change_fingerprint() -> None:
    application = _application()
    first = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                revision="rev-a",
                delta=ProfileDelta(
                    execution_mode=ProfileFieldUpdate(value=ExecutionMode.BALANCED),
                ),
            ),
        ),
    )
    second = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                revision="rev-b",
                delta=ProfileDelta(
                    execution_mode=ProfileFieldUpdate(value=ExecutionMode.BALANCED),
                ),
            ),
        ),
    )
    assert first.fingerprint == second.fingerprint


def test_duplicate_layer_fails_deterministically() -> None:
    application = _application()
    layers = (
        ProfileLayerInput(
            layer=ProfileLayer.AGENT,
            delta=ProfileDelta(
                execution_mode=ProfileFieldUpdate(value=ExecutionMode.STRICT),
            ),
        ),
        ProfileLayerInput(
            layer=ProfileLayer.AGENT,
            delta=ProfileDelta(
                execution_mode=ProfileFieldUpdate(value=ExecutionMode.EXPLORATORY),
            ),
        ),
    )
    with pytest.raises(ProfileLayerConflictError):
        resolve_profile(application, layers=layers)


def test_unknown_delta_field_fails_closed() -> None:
    with pytest.raises(ValidationError):
        ProfileDelta.model_validate({"unexpected": "value"})


def test_invalid_delta_missing_value_fails_closed() -> None:
    with pytest.raises(ValidationError):
        ProfileFieldUpdate[ExecutionMode](action="set", value=None)


class _ExecutionModeEchoResolver:
    path = "meta.execution_mode"

    def resolve(
        self,
        *,
        profile: ApplicationEnvironmentProfile,
        update: ProfileFieldUpdate[object],
        source_layer: ProfileLayer,
        context: ProfileFieldResolveContext | None = None,
    ) -> ProfileFieldResolveResult:
        from intergrax.applications._shared.profile_resolution.field_resolvers import (
            ExecutionModeFieldResolver,
            ProfileFieldResolveContext,
        )

        return ExecutionModeFieldResolver().resolve(
            profile=profile,
            update=update,
            source_layer=source_layer,
            context=context or ProfileFieldResolveContext(),
        )


def test_custom_field_resolver_participates_without_core_change() -> None:
    application = _application(execution_mode=ExecutionMode.BALANCED)
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.RUN,
                delta=ProfileDelta(
                    execution_mode=ProfileFieldUpdate(value=ExecutionMode.STRICT),
                ),
            ),
        ),
        field_resolvers=(_ExecutionModeEchoResolver(),),
    )
    assert resolution.effective_profile.execution_mode == ExecutionMode.STRICT


def test_build_harness_host_runtime_uses_effective_profile_resolution() -> None:
    manifest = ApplicationManifest.lab(
        app_id="profile_resolution_host",
        name="Profile Resolution Host",
        route_prefix="/v1/profile_resolution_host",
        env_prefix="PROFILE_RESOLUTION_HOST_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    configured = ApplicationEnvironmentProfile.lab_defaults(profile_id="profile_resolution_host.lab")
    runtime = build_harness_host_runtime(
        manifest,
        configured,
        use_in_memory_trace=True,
    )
    assert runtime.profile_resolution is not None
    assert runtime.environment == runtime.profile_resolution.effective_profile
    assert configured.model_copy(deep=True) == configured


def test_application_cannot_widen_platform_tool_authority() -> None:
    application = _application(tools=["search", "shell"])
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search"]),
                    ),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.tool_profile.enabled == ["search"]
    clamped = [
        item
        for item in resolution.decisions
        if item.decision == ProfileResolutionDecisionKind.CLAMPED
        and item.source_layer == ProfileLayer.APPLICATION
    ]
    assert clamped
    assert "shell" in (clamped[0].requested_value or "")


def test_product_narrowing_survives_application_widen_attempt() -> None:
    application = _application(tools=["search", "calculator"])
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search", "calculator"]),
                    ),
                ),
            ),
            ProfileLayerInput(
                layer=ProfileLayer.PRODUCT,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search"]),
                    ),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.tool_profile.enabled == ["search"]
    clamped = [
        item
        for item in resolution.decisions
        if item.decision == ProfileResolutionDecisionKind.CLAMPED
        and item.source_layer == ProfileLayer.APPLICATION
    ]
    assert clamped


def test_execution_clear_cannot_widen_upstream_tools() -> None:
    application = _application(tools=["search"])
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search"]),
                    ),
                ),
            ),
            ProfileLayerInput(
                layer=ProfileLayer.EXECUTION,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(action="clear"),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.tool_profile.enabled == ["search"]


def test_execution_clear_cannot_widen_upstream_budget() -> None:
    application = _application(max_tool_calls=10)
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(
                        value=CostProfile(max_tool_calls=10),
                    ),
                ),
            ),
            ProfileLayerInput(
                layer=ProfileLayer.EXECUTION,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(action="clear"),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.cost_profile.max_tool_calls == 10


def test_bundle_authority_clamps_tools_outside_bundle() -> None:
    application = _application(tools=["jira.search_tasks", "shell.exec"])
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled_bundles=["jira"]),
                    ),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.tool_profile.enabled == ["jira.search_tasks"]
    clamped = [
        item
        for item in resolution.decisions
        if item.decision == ProfileResolutionDecisionKind.CLAMPED
    ]
    assert clamped
    assert "shell.exec" in (clamped[0].requested_value or "")


def test_empty_upstream_tool_profile_denies_downstream_tool() -> None:
    application = _application(tools=["shell.exec"])
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(value=ToolProfile()),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.tool_profile.enabled == []
    clamped = [
        item
        for item in resolution.decisions
        if item.decision == ProfileResolutionDecisionKind.CLAMPED
        and item.source_layer == ProfileLayer.APPLICATION
    ]
    assert clamped


def test_application_scalar_execution_mode_override_allowed() -> None:
    application = _application(execution_mode=ExecutionMode.STRICT)
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    execution_mode=ProfileFieldUpdate(value=ExecutionMode.BALANCED),
                ),
            ),
        ),
    )
    assert resolution.effective_profile.execution_mode == ExecutionMode.STRICT
    applied = [
        item
        for item in resolution.decisions
        if item.path == "meta.execution_mode"
        and item.source_layer == ProfileLayer.APPLICATION
        and item.decision == ProfileResolutionDecisionKind.APPLIED
    ]
    assert applied


def test_fingerprint_ignores_provenance_only_clamp_difference() -> None:
    application = _application(tools=["search", "shell"])
    platform_only = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search"]),
                    ),
                ),
            ),
        ),
    )
    platform_and_product = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search"]),
                    ),
                ),
            ),
            ProfileLayerInput(
                layer=ProfileLayer.PRODUCT,
                delta=ProfileDelta(
                    tool_profile=ProfileFieldUpdate(
                        value=ToolProfile(enabled=["search"]),
                    ),
                ),
            ),
        ),
    )
    assert platform_only.effective_profile == platform_and_product.effective_profile
    assert platform_only.fingerprint == platform_and_product.fingerprint
    assert platform_only.decisions != platform_and_product.decisions


def test_resolution_evidence_survives_harness_composition() -> None:
    manifest = ApplicationManifest.lab(
        app_id="profile_resolution_evidence",
        name="Profile Resolution Evidence",
        route_prefix="/v1/profile_resolution_evidence",
        env_prefix="PROFILE_RESOLUTION_EVIDENCE_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    configured = ApplicationEnvironmentProfile.lab_defaults(profile_id="profile_resolution_evidence.lab")
    runtime = build_harness_host_runtime(
        manifest,
        configured,
        use_in_memory_trace=True,
        profile_layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    execution_mode=ProfileFieldUpdate(value=ExecutionMode.STRICT),
                ),
            ),
        ),
    )
    assert runtime.profile_resolution is not None
    assert runtime.profile_resolution.fingerprint
    assert runtime.environment == runtime.profile_resolution.effective_profile
    assert any(
        item.source_layer == ProfileLayer.PLATFORM
        for item in runtime.profile_resolution.decisions
    )
