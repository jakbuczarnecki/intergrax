# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.observability.contracts import (
    ErrorsCaptureInput,
    ErrorsCaptureOutput,
    LogsSearchInput,
    LogsSearchOutput,
    MetricsQueryInstantInput,
    MetricsQueryInstantOutput,
    TracesQueryInput,
    TracesQueryOutput,
)
from intergrax.tools.providers.observability.service import logs_search, metrics_query_instant, traces_query, errors_capture


class MetricsQueryInstantHandler(
    ServiceToolHandler[MetricsQueryInstantInput, MetricsQueryInstantOutput]
):
    _service = metrics_query_instant


class LogsSearchHandler(ServiceToolHandler[LogsSearchInput, LogsSearchOutput]):
    _service = logs_search


class TracesQueryHandler(ServiceToolHandler[TracesQueryInput, TracesQueryOutput]):
    _service = traces_query


class ErrorsCaptureHandler(ServiceToolHandler[ErrorsCaptureInput, ErrorsCaptureOutput]):
    _service = errors_capture
