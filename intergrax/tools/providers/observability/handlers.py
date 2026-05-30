# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.observability.contracts import (
    LogsSearchInput,
    LogsSearchOutput,
    MetricsQueryInstantInput,
    MetricsQueryInstantOutput,
)
from intergrax.tools.providers.observability.service import logs_search, metrics_query_instant
from intergrax.tools.registry.wiring import ToolWiringContext


class MetricsQueryInstantHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[MetricsQueryInstantInput]) -> MetricsQueryInstantOutput:
        return metrics_query_instant(self._ctx, request.input)


class LogsSearchHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[LogsSearchInput]) -> LogsSearchOutput:
        return logs_search(self._ctx, request.input)
