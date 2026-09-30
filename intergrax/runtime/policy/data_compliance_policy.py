# © Artur Czarnecki. All rights reserved.

"""Neutral data compliance policy (trace/API export); Nexus consumes inward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ApiTraceExportMode = Literal["none", "redacted", "full"]


@dataclass(frozen=True, slots=True)
class DataCompliancePolicy:
    api_trace_export: ApiTraceExportMode = "redacted"
    redact_tool_calls_in_api: bool = True


__all__ = ["DataCompliancePolicy"]
