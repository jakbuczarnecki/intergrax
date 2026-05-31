# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import ToolBundleEntry, clear_tool_catalog, register_tool_bundle
from intergrax.tools.registry.factory import build_registry_from_profile, enabled_tool_ids_for_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    result: int


@pytest.fixture(autouse=True)
def _isolated_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def _install_demo_bundles() -> None:
    def register_alpha(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        registry.register(
            ToolContract(
                tool_id="alpha.one",
                name="alpha.one",
                description="alpha one",
                input_schema=_In,
                output_schema=_Out,
                error_mapping={},
                side_effects=False,
            ),
            type("H", (), {"execute": lambda self, req: _Out(result=req.input.value)})(),
        )

    def register_beta(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        for tool_id in ("beta.one", "beta.two"):
            registry.register(
                ToolContract(
                    tool_id=tool_id,
                    name=tool_id,
                    description=tool_id,
                    input_schema=_In,
                    output_schema=_Out,
                    error_mapping={},
                    side_effects=False,
                ),
                type("H", (), {"execute": lambda self, req: _Out(result=req.input.value)})(),
            )

    register_tool_bundle(
        ToolBundleEntry(bundle_id="alpha", tool_ids=("alpha.one",), register=register_alpha)
    )
    register_tool_bundle(
        ToolBundleEntry(
            bundle_id="beta",
            tool_ids=("beta.one", "beta.two"),
            register=register_beta,
        )
    )


def test_build_registry_empty_lab_profile() -> None:
    register_default_tools()
    registry = build_registry_from_profile(ToolProfile.lab())
    assert registry.tool_ids() == []


def test_build_registry_from_enabled_tool_ids() -> None:
    _install_demo_bundles()
    profile = ToolProfile(enabled=["beta.one"])
    registry = build_registry_from_profile(profile)
    assert registry.tool_ids() == ["beta.one"]


def test_build_registry_from_enabled_bundle() -> None:
    _install_demo_bundles()
    profile = ToolProfile(enabled_bundles=["beta"])
    registry = build_registry_from_profile(profile)
    assert registry.tool_ids() == ["beta.one", "beta.two"]


def test_build_registry_all_catalog() -> None:
    _install_demo_bundles()
    registry = build_registry_from_profile(ToolProfile.all_catalog())
    assert registry.tool_ids() == ["alpha.one", "beta.one", "beta.two"]


def test_enabled_tool_ids_for_profile() -> None:
    _install_demo_bundles()
    profile = ToolProfile(enabled=["alpha.one", "beta.two"])
    assert enabled_tool_ids_for_profile(profile) == ["alpha.one", "beta.two"]
