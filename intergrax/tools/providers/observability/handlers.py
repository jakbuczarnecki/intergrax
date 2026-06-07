# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.observability.contracts import (
    ErrorsCaptureInput,
    ErrorsCaptureOutput,
    LogsSearchInput,
    LogsSearchOutput,
    LogsTailInput,
    LogsTailOutput,
    MetricsQueryInstantInput,
    MetricsQueryInstantOutput,
    MetricsQueryRangeInput,
    MetricsQueryRangeOutput,
    TracesQueryInput,
    TracesQueryOutput,
)
from intergrax.tools.providers.observability.service import (
    errors_capture,
    logs_search,
    logs_tail,
    metrics_query_instant,
    metrics_query_range,
    traces_query,
)


class MetricsQueryInstantHandler(
    ServiceToolHandler[MetricsQueryInstantInput, MetricsQueryInstantOutput]
):
    _service = metrics_query_instant


class MetricsQueryRangeHandler(
    ServiceToolHandler[MetricsQueryRangeInput, MetricsQueryRangeOutput]
):
    _service = metrics_query_range


class LogsSearchHandler(ServiceToolHandler[LogsSearchInput, LogsSearchOutput]):
    _service = logs_search


class LogsTailHandler(ServiceToolHandler[LogsTailInput, LogsTailOutput]):
    _service = logs_tail


class TracesQueryHandler(ServiceToolHandler[TracesQueryInput, TracesQueryOutput]):
    _service = traces_query


class ErrorsCaptureHandler(ServiceToolHandler[ErrorsCaptureInput, ErrorsCaptureOutput]):
    _service = errors_capture
