# © Artur Czarnecki. All rights reserved.

"""AGENT-CONSOLIDATION-3 harness host registry authority proofs."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agent_distribution.roster import EffectiveRosterEntry
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import (
    HarnessHostRegistryAuthorityError,
    RegistryAssemblyMode,
    resolve_registry_assembly_mode,
)
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionInputBundle,
    build_registry_projection,
)
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from tests.unit.applications.test_registry_projection_ap10 import (
    ECHO_BUILDERS,
    _APP,
    _ARTIFACT,
    _ENV,
    _ROSTER_A,
    _echo_factory,
    _entry,
    _manifest,
    _revision,
    _roster,
    _resolver_from_roster,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _strict_environment(*, profile_id: str = _ENV) -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.product_defaults(profile_id=profile_id)


def _build_projection(*, roster_entries: tuple[EffectiveRosterEntry, ...]) -> MaterializedRegistryProjection:
    roster = _roster(roster_entries)
    manifest = _manifest()
    revision = _revision(
        "rev-ac3",
        roster_revision_id=roster.effective_roster_revision_id or _ROSTER_A,
    )
    bundle = RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        factory_resolver=_resolver_from_roster(roster),
        builders=ECHO_BUILDERS,
        materialization_artifact_digest=_ARTIFACT,
    )
    return build_registry_projection(bundle)


def test_resolve_registry_assembly_mode_strict_is_revision_bound() -> None:
    env = _strict_environment()
    assert resolve_registry_assembly_mode(env) is RegistryAssemblyMode.REVISION_BOUND


def test_resolve_registry_assembly_mode_lab_is_manifest_development() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="lab.ac3")
    assert resolve_registry_assembly_mode(env) is RegistryAssemblyMode.MANIFEST_DEVELOPMENT


def test_revision_bound_host_uses_projection_registry_only() -> None:
    manifest = _manifest()
    environment = _strict_environment()
    projection = _build_projection(roster_entries=(_entry("search"),))
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        registry_projection=projection,
        use_in_memory_trace=True,
    )
    assert runtime.registry.list_agent_ids() == ["search"]
    assert runtime.registry_projection_evidence is not None
    assert runtime.registry_projection_evidence.runtime_revision_id == "rev-ac3"
    assert resolve_harness_host_nexus_loop_legacy(runtime).registry.list_agent_ids() == ["search"]


def test_manifest_extra_agent_absent_from_revision_bound_nexus_registry() -> None:
    manifest = _manifest()
    environment = _strict_environment()
    projection = _build_projection(roster_entries=(_entry("search"),))
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        registry_projection=projection,
        use_in_memory_trace=True,
    )
    assert "indexer" in {binding.contract_id for binding in manifest.enabled_agents()}
    assert "indexer" not in resolve_harness_host_nexus_loop_legacy(runtime).registry.list_agent_ids()
    assert "synthesizer" not in resolve_harness_host_nexus_loop_legacy(runtime).registry.list_agent_ids()


def test_revision_bound_without_projection_fails_closed() -> None:
    manifest = _manifest()
    environment = _strict_environment()
    with pytest.raises(HarnessHostRegistryAuthorityError, match="MaterializedRegistryProjection"):
        build_harness_host_runtime(
            manifest,
            environment,
            use_in_memory_trace=True,
        )


def test_revision_bound_without_projection_does_not_call_manifest_builder() -> None:
    manifest = _manifest()
    environment = _strict_environment()
    with patch(
        "intergrax.applications._shared.harness_registry_authority.build_manifest_development_registry",
    ) as manifest_builder:
        with pytest.raises(HarnessHostRegistryAuthorityError):
            build_harness_host_runtime(
                manifest,
                environment,
                use_in_memory_trace=True,
            )
    manifest_builder.assert_not_called()


def test_projection_scope_mismatch_application_id_fails_closed() -> None:
    manifest = _manifest(app_id="host_app")
    environment = _strict_environment(profile_id=_ENV)
    projection = _build_projection(roster_entries=(_entry("search"),))
    mismatched = replace(
        projection,
        evidence=projection.evidence.model_copy(update={"application_id": _APP}),
    )
    with pytest.raises(HarnessHostRegistryAuthorityError, match="application_id"):
        build_harness_host_runtime(
            manifest,
            environment,
            registry_projection=mismatched,
            use_in_memory_trace=True,
        )


def test_projection_scope_mismatch_environment_id_fails_closed() -> None:
    manifest = _manifest()
    environment = _strict_environment(profile_id="host-env")
    projection = _build_projection(roster_entries=(_entry("search"),))
    with pytest.raises(HarnessHostRegistryAuthorityError, match="application_environment_id"):
        build_harness_host_runtime(
            manifest,
            environment,
            registry_projection=projection,
            use_in_memory_trace=True,
        )


def test_explicit_manifest_development_mode_builds_from_manifest() -> None:
    manifest = _manifest(
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", factory=_echo_factory)],
        app_id="dev_app",
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="dev_env")
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        use_in_memory_trace=True,
    )
    assert runtime.registry.list_agent_ids() == ["echo"]
    assert runtime.registry_projection_evidence is None


def test_build_manifest_development_registry_is_explicit_non_production_path() -> None:
    manifest = ApplicationManifest.lab(
        app_id="inline_dev",
        name="Inline Dev",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", factory=_echo_factory)],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    registry = build_manifest_development_registry(manifest, ctx, builders=ECHO_BUILDERS)
    assert registry.list_agent_ids() == ["echo"]


def test_revision_bound_rejects_anonymous_registry_override() -> None:
    manifest = _manifest()
    environment = _strict_environment()
    projection = _build_projection(roster_entries=(_entry("search"),))
    injected = AgentRegistry()
    with pytest.raises(HarnessHostRegistryAuthorityError, match="anonymous AgentRegistry"):
        build_harness_host_runtime(
            manifest,
            environment,
            registry_projection=projection,
            registry=injected,
            use_in_memory_trace=True,
        )


def test_revision_bound_rejects_builders_fallback() -> None:
    manifest = _manifest()
    environment = _strict_environment()
    projection = _build_projection(roster_entries=(_entry("search"),))
    with pytest.raises(HarnessHostRegistryAuthorityError, match="builders fallback"):
        build_harness_host_runtime(
            manifest,
            environment,
            registry_projection=projection,
            builders=ECHO_BUILDERS,
            use_in_memory_trace=True,
        )


def test_strict_environment_infers_revision_bound_without_silent_manifest_fallback() -> None:
    manifest = _manifest()
    environment = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    assert environment.execution_mode is ExecutionMode.STRICT
    with pytest.raises(HarnessHostRegistryAuthorityError, match="manifest-only registry fallback"):
        build_harness_host_runtime(
            manifest,
            environment,
            builders=ECHO_BUILDERS,
            use_in_memory_trace=True,
        )


def test_strict_explicit_manifest_development_override_fails_closed() -> None:
    env = _strict_environment()
    with pytest.raises(
        HarnessHostRegistryAuthorityError,
        match="STRICT execution mode requires revision-bound registry authority",
    ):
        resolve_registry_assembly_mode(
            env,
            explicit=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
        )


def test_strict_explicit_manifest_development_host_build_fails_without_manifest_builder() -> None:
    manifest = _manifest()
    environment = _strict_environment()
    with patch(
        "intergrax.applications._shared.harness_registry_authority.build_manifest_development_registry",
    ) as manifest_builder:
        with pytest.raises(HarnessHostRegistryAuthorityError):
            build_harness_host_runtime(
                manifest,
                environment,
                registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
                use_in_memory_trace=True,
            )
    manifest_builder.assert_not_called()
