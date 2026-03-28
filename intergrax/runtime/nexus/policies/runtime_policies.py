# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ExecutionKind(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    STORAGE = "storage"


@dataclass(frozen=True)
class TimeoutPolicy:
    llm_seconds: float = 30.0
    tool_seconds: float = 30.0
    retrieval_seconds: float = 10.0
    storage_seconds: float = 5.0


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 0.0  # deterministic for tests / CI


@dataclass(frozen=True)
class FallbackPolicy:
    escalate_to_hitl: bool = True


@dataclass(frozen=True)
class HitlPolicy:
    enabled: bool = True
    stop_reason: str = "needs_user_input"


ApiTraceExportMode = Literal["none", "redacted", "full"]


@dataclass(frozen=True)
class DataCompliancePolicy:
    """
    Cross-cutting rules for what product HTTP/API surfaces may expose outside the tenant boundary.

    - ``api_trace_export``: how :class:`~intergrax.runtime.nexus.tracing.trace_models.TraceEvent`
      payloads are serialized on eg. Legal HTTP ``trace_events`` (Nexus ``redact()`` vs raw ``to_dict()``).
    - ``redact_tool_calls_in_api``: when True, strip tool ``arguments`` from API-shaped tool_calls
      (summaries and success/error may remain).
    """

    api_trace_export: ApiTraceExportMode = "redacted"
    redact_tool_calls_in_api: bool = True


@dataclass(frozen=True)
class RuntimePolicies:
    timeout: TimeoutPolicy = TimeoutPolicy()
    retry: RetryPolicy = RetryPolicy()
    fallback: FallbackPolicy = FallbackPolicy()
    hitl: HitlPolicy = HitlPolicy()
    data_compliance: DataCompliancePolicy = DataCompliancePolicy()
