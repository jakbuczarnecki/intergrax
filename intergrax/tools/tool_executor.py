# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionRequest


InModelT = TypeVar("InModelT", bound=BaseModel)
OutModelT = TypeVar("OutModelT", bound=BaseModel)


@runtime_checkable
class ToolHandler(Protocol[InModelT, OutModelT]):
    """
    Pure tool implementation contract.

    - MUST NOT perform runtime enforcement
    - MAY raise exceptions (runtime maps them via ToolContract.error_mapping)
    """

    def execute(self, request: ToolExecutionRequest[InModelT]) -> OutModelT: ...


@runtime_checkable
class ToolExecutor(Protocol):
    """
    Runtime port used for executing tools.

    Runtime controls:
    - registry lookup
    - schema validation
    - trace start/end/error
    - error mapping

    Executor:
    - locates the tool handler by tool_id
    - calls handler.execute(...)
    - may raise (runtime owns mapping + normalization)
    """

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel: ...
