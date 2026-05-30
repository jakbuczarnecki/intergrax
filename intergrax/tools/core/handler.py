# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Base classes for catalog tool handlers (Phase O)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Generic

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.tools.tool_executor import InModelT, OutModelT

ServiceFn = Callable[[ToolWiringContext, InModelT], OutModelT]


class WiringContextToolHandler(ABC, Generic[InModelT, OutModelT]):
    """
    Abstract catalog handler — receives ``ToolWiringContext`` at registration time.

    Implements the :class:`~intergrax.tools.tool_executor.ToolHandler` protocol.
    Subclass :class:`ServiceToolHandler` when ``execute`` only delegates to a service fn.
    """

    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    @abstractmethod
    def execute(self, request: ToolExecutionRequest[InModelT]) -> OutModelT:
        """Run the tool; runtime owns validation, trace, and error mapping."""


class ServiceToolHandler(WiringContextToolHandler[InModelT, OutModelT]):
    """
    Handler that delegates to ``_service(ctx, input) -> output``.

    Example::

        class JiraGetIssueHandler(ServiceToolHandler[JiraGetIssueInput, JiraIssueOutput]):
            _service = jira_get_issue
    """

    _service: ClassVar[ServiceFn]

    def execute(self, request: ToolExecutionRequest[InModelT]) -> OutModelT:
        service = type(self)._service
        return service(self._ctx, request.input)
