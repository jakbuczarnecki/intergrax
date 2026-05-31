# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.catalog import (
    ToolBundleEntry,
    clear_tool_catalog,
    get_bundle,
    list_bundle_ids,
    list_catalog_tool_ids,
    register_tool_bundle,
)
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


def _register_echo_bundle() -> None:
    def _register(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
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

    register_tool_bundle(
        ToolBundleEntry(
            bundle_id="echo",
            tool_ids=("echo.ping",),
            register=_register,
            description="Lab echo bundle",
        )
    )


def test_register_and_list_bundles() -> None:
    _register_echo_bundle()
    assert list_bundle_ids() == ["echo"]
    assert list_catalog_tool_ids() == ["echo.ping"]
    assert get_bundle("echo").tool_ids == ("echo.ping",)


def test_duplicate_bundle_raises() -> None:
    _register_echo_bundle()
    with pytest.raises(ValueError, match="already registered"):
        _register_echo_bundle()
