# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications._shared.agent_runtime_governance_wiring import (
    AgentRuntimeGovernanceMaterializationError,
)
from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_for_application_host,
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from echo.echo_agent import EchoAgent
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry import ToolProfile, ToolRegistry, ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_build_declarative_invoker_returns_none_when_tools_disabled() -> None:
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=False),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    assert build_declarative_invoker_from_tool_wiring(wiring) is None


def test_build_declarative_invoker_fail_closed_without_governance_in_production_mode() -> None:
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=["read_file"]),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    with pytest.raises(AgentRuntimeGovernanceMaterializationError):
        build_declarative_invoker_from_tool_wiring(wiring, production_mode=True)


def test_build_declarative_invoker_for_application_host_skips_governance_when_tools_disabled() -> None:
    manifest = ApplicationManifest.lab(
        app_id="decl_skip_gov",
        name="Decl Skip Gov",
        route_prefix="/v1/decl_skip_gov",
        env_prefix="DECL_SKIP_GOV_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    build_ctx = ApplicationBuildContext.for_manifest(manifest, policy_bundle=RuntimePolicyBundle())
    registry = build_application_registry(manifest, build_ctx)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="decl_skip_gov.strict").model_copy(
        update={"execution_mode": ExecutionMode.STRICT},
    )
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=False),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    assert (
        build_declarative_invoker_for_application_host(
            wiring,
            env,
            manifest=manifest,
            agent_registry=registry,
            tenant_id="   ",
        )
        is None
    )
