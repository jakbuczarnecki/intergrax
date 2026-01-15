# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True)
class RuntimePolicies:
    timeout: TimeoutPolicy = TimeoutPolicy()
    retry: RetryPolicy = RetryPolicy()
    fallback: FallbackPolicy = FallbackPolicy()
    hitl: HitlPolicy = HitlPolicy()
