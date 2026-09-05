# © Artur Czarnecki. All rights reserved.

"""Tool bundle catalog adapter contract tests (Stage 2)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.capability_catalog.adapters.tool import (
    TOOL_BUILTIN_CATALOG_SOURCE_ID,
    ToolBundleCatalogSource,
    project_tool_bundle_entry,
)
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.catalog import ToolBundleEntry, clear_tool_catalog, register_tool_bundle
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


@pytest.fixture(autouse=True)
def _isolated_catalog() -> None:
    clear_tool_catalog()
    yield
    clear_tool_catalog()


def _bundle() -> ToolBundleEntry:
    def _register(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        del ctx
        handler = type("H", (), {"execute": lambda self, req: _Out(y=req.input.x)})()
        registry.register(
            ToolContract(
                tool_id="echo.ping",
                name="echo.ping",
                description="echo",
                input_schema=_In,
                output_schema=_Out,
                error_mapping={},
                side_effects=False,
            ),
            handler,
        )

    return ToolBundleEntry(
        bundle_id="echo",
        tool_ids=("echo.ping", "echo.status"),
        register=_register,
        description="Echo tools",
    )


def test_project_tool_bundle_entry_uses_tool_identity_not_bundle_identity() -> None:
    projected = project_tool_bundle_entry(_bundle(), "echo.ping")
    assert projected.identity.kind is CapabilityKind.TOOL
    assert projected.identity.logical.logical_id == "echo.ping"
    assert projected.identity.logical.logical_id != "echo"
    assert projected.provenance.package_reference == "echo"
    assert projected.identity.source.source_id == TOOL_BUILTIN_CATALOG_SOURCE_ID
    assert projected.identity.source.source_kind is CapabilitySourceKind.BUILTIN


def test_tool_bundle_catalog_source_expands_bundle_to_capability_level_entries() -> None:
    register_tool_bundle(_bundle())
    entries = ToolBundleCatalogSource().read_entries()
    logical_ids = {entry.identity.logical.logical_id for entry in entries}
    assert logical_ids == {"echo.ping", "echo.status"}
    assert all(entry.provenance.package_reference == "echo" for entry in entries)
    assert all(entry.identity.logical.logical_id != "echo" for entry in entries)
