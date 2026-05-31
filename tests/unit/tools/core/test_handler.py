# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.tools.core.handler import ServiceToolHandler, WiringContextToolHandler
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _EchoInput(BaseModel):
    text: str


class _EchoOutput(BaseModel):
    text: str


def _echo_service(ctx: ToolWiringContext, payload: _EchoInput) -> _EchoOutput:
    _ = ctx
    return _EchoOutput(text=payload.text.upper())


class _EchoHandler(ServiceToolHandler[_EchoInput, _EchoOutput]):
    _service = _echo_service


def test_service_tool_handler_delegates_to_service() -> None:
    handler = _EchoHandler(ToolWiringContext())
    result = handler.execute(
        ToolExecutionRequest(
            run_id="run-1",
            step_id="step-1",
            tool_id="echo",
            input=_EchoInput(text="hi"),
        )
    )
    assert result.text == "HI"


def test_wiring_context_tool_handler_is_abstract() -> None:
    with pytest.raises(TypeError):
        WiringContextToolHandler()  # type: ignore[misc]
