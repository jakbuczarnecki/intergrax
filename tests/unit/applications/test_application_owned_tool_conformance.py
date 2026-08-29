# © Artur Czarnecki. All rights reserved.

"""PLATFORM-5B — application-owned tool declaration conformance."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.application_owned_tool_conformance import (
    application_owned_tool_declarations,
    assert_application_owned_tool_conformance,
    merge_application_owned_tool_registry,
    validate_application_owned_tool_conformance,
)
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.registry_snapshot import resolve_registry_snapshot
from intergrax.applications.contracts.application_owned_tools import ApplicationOwnedToolDeclaration
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.application_package import ApplicationPackageClosureError
from intergrax.applications.contracts.errors import ApplicationManifestConformanceError
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.tools.core.contracts import ToolContract
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolProfile, ToolRegistry
from intergrax.tools.registry.catalog import list_catalog_tool_ids
from intergrax.tools.tool_executor import ToolHandler

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CUSTOM_TOOL_ID = "custom_app.echo.ping"


class _EchoInput(BaseModel):
    message: str = "pong"


class _EchoOutput(BaseModel):
    message: str


class _EchoHandler(ToolHandler[_EchoInput, _EchoOutput]):
    def execute(self, request: ToolExecutionRequest[_EchoInput]) -> _EchoOutput:
        return _EchoOutput(message=request.input.message)


def _custom_tool_contract() -> ToolContract:
    return ToolContract(
        tool_id=_CUSTOM_TOOL_ID,
        name=_CUSTOM_TOOL_ID,
        description="Generic application-owned echo tool for conformance tests.",
        input_schema=_EchoInput,
        output_schema=_EchoOutput,
        error_mapping={},
        side_effects=False,
    )


def _register_custom_tool(registry: ToolRegistry) -> None:
    registry.register(_custom_tool_contract(), _EchoHandler())


def _custom_manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="custom_tool_demo",
        name="Custom Tool Demo",
        route_prefix="/v1/custom_tool_demo",
        env_prefix="CUSTOM_TOOL_DEMO_",
        agents=[],
        application_owned_tools=application_owned_tool_declarations([_CUSTOM_TOOL_ID]),
    )


def _custom_env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults(
        profile_id="custom_tool_demo.scaffold",
    ).model_copy(
        update={"tool_profile": ToolProfile(enabled=[_CUSTOM_TOOL_ID])},
    )


def test_valid_custom_application_tool_passes_conformance() -> None:
    manifest = _custom_manifest()
    env = _custom_env()
    application_registry = ToolRegistry()
    _register_custom_tool(application_registry)
    wiring = wire_application_environment(
        manifest,
        env,
        tenant_id="tenant-a",
        application_tool_registry=application_registry,
        conformance_check=True,
    )
    assert _CUSTOM_TOOL_ID in wiring.tool_wiring.registry.tool_ids()


def test_undeclared_application_tool_fails_closed() -> None:
    manifest = _custom_manifest()
    env = _custom_env()
    application_registry = ToolRegistry()
    _register_custom_tool(application_registry)
    application_registry.register(
        ToolContract(
            tool_id="custom_app.secret.extra",
            name="custom_app.secret.extra",
            description="undeclared extra tool",
            input_schema=_EchoInput,
            output_schema=_EchoOutput,
            error_mapping={},
            side_effects=False,
        ),
        _EchoHandler(),
    )
    with pytest.raises(ApplicationManifestConformanceError, match="undeclared"):
        wire_application_environment(
            manifest,
            env,
            tenant_id="tenant-a",
            application_tool_registry=application_registry,
            conformance_check=True,
        )


def test_declared_but_not_registered_fails_closed() -> None:
    manifest = _custom_manifest()
    env = _custom_env()
    with pytest.raises(
        (ApplicationManifestConformanceError, ApplicationPackageClosureError),
        match="missing from wired tool registry",
    ):
        wire_application_environment(
            manifest,
            env,
            tenant_id="tenant-a",
            application_tool_registry=ToolRegistry(),
            conformance_check=True,
        )


def test_collision_with_platform_catalog_fails_closed() -> None:
    catalog_ids = list_catalog_tool_ids()
    if not catalog_ids:
        pytest.skip("catalog has no tool ids in this test environment")
    colliding_id = catalog_ids[0]
    manifest = ApplicationManifest.lab(
        app_id="collision_demo",
        name="Collision Demo",
        route_prefix="/v1/collision_demo",
        env_prefix="COLLISION_DEMO_",
        agents=[],
        application_owned_tools=[ApplicationOwnedToolDeclaration(tool_id=colliding_id)],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="collision_demo.scaffold").model_copy(
        update={"tool_profile": ToolProfile(enabled=[colliding_id])},
    )
    violations = validate_application_owned_tool_conformance(
        manifest,
        env,
        resolve_registry_snapshot(
            ApplicationBuildContext.for_manifest(
                manifest,
                tool_profile=env.tool_profile,
                tool_registry=ToolRegistry(),
            ),
        ),
        platform_tool_ids=frozenset(catalog_ids),
    )
    assert any("collide" in violation for violation in violations)


def test_profile_outside_closure_fails_closed() -> None:
    manifest = _custom_manifest()
    env = _custom_env().model_copy(
        update={"tool_profile": ToolProfile(enabled=[_CUSTOM_TOOL_ID, "custom_app.unknown.tool"])},
    )
    application_registry = ToolRegistry()
    _register_custom_tool(application_registry)
    violations = validate_application_owned_tool_conformance(
        manifest,
        env,
        resolve_registry_snapshot(
            ApplicationBuildContext.for_manifest(
                manifest,
                tool_profile=env.tool_profile,
                tool_registry=application_registry,
            ),
        ),
    )
    assert any("outside allowed closure" in violation for violation in violations)


def test_custom_application_tool_executes_through_registry() -> None:
    registry = ToolRegistry()
    _register_custom_tool(registry)
    result = registry.get(_CUSTOM_TOOL_ID).handler.execute(
        ToolExecutionRequest(
            input=_EchoInput(message="hello"),
            tool_id=_CUSTOM_TOOL_ID,
            run_id=mint_run_id(),
            step_id="step-1",
        ),
    )
    assert result.message == "hello"


def test_merge_rejects_platform_collision() -> None:
    catalog_registry = ToolRegistry()
    application_registry = ToolRegistry()
    _register_custom_tool(application_registry)
    catalog_registry.register(_custom_tool_contract(), _EchoHandler())
    with pytest.raises(ApplicationManifestConformanceError, match="collides with platform catalog"):
        merge_application_owned_tool_registry(
            catalog_registry=catalog_registry,
            application_registry=application_registry,
            declared_tool_ids=frozenset({_CUSTOM_TOOL_ID}),
        )


def test_assert_application_owned_tool_conformance_raises() -> None:
    manifest = _custom_manifest()
    env = _custom_env()
    with pytest.raises(ApplicationManifestConformanceError):
        assert_application_owned_tool_conformance(
            manifest,
            env,
            resolve_registry_snapshot(
                ApplicationBuildContext.for_manifest(
                    manifest,
                    tool_profile=env.tool_profile,
                    tool_registry=ToolRegistry(),
                ),
            ),
        )
