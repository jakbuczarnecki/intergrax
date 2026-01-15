# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.nexus.policies.runtime_policies import (    
    ExecutionKind,
    RuntimePolicies,
)

T = TypeVar("T")


class TransientOperationError(Exception):
    """Retryable transient failure (timeouts, network, upstream 5xx)."""


class NonRetryableOperationError(Exception):
    """Logical or validation error — no retry."""


class PolicyAbortError(Exception):
    """Raised when policy decides to abort execution."""


class PolicyEnforcer:
    def __init__(self, policies: RuntimePolicies) -> None:
        self._policies = policies

    async def execute(
        self,
        *,
        kind: ExecutionKind,
        op_name: str,
        fn: Callable[[], Awaitable[T]],
        state: RuntimeState,
    ) -> T:
        timeout = self._timeout_for(kind)
        max_attempts = self._policies.retry.max_attempts

        attempt = 0
        while True:
            attempt += 1

            state.trace_event(
                component=TraceComponent.POLICY,
                step=op_name,
                level=TraceLevel.INFO,
                message=f"Policy execute attempt {attempt}",
            )

            try:
                return await asyncio.wait_for(fn(), timeout=timeout)

            except asyncio.TimeoutError as exc:
                err = TransientOperationError("Operation timed out")

            except TransientOperationError as exc:
                err = exc

            except NonRetryableOperationError:
                raise

            if attempt >= max_attempts:
                state.trace_event(
                    component=TraceComponent.POLICY,
                    step=op_name,
                    level=TraceLevel.WARNING,
                    message=f"Retry exhausted after {attempt} attempts",
                )

                if self._policies.fallback.escalate_to_hitl and self._policies.hitl.enabled:
                    raise PolicyAbortError("Escalate to HITL")

                raise PolicyAbortError("Abort after retry exhaustion")

            await asyncio.sleep(self._policies.retry.backoff_seconds)

    def _timeout_for(self, kind: ExecutionKind) -> float:
        t = self._policies.timeout
        return {
            ExecutionKind.LLM: t.llm_seconds,
            ExecutionKind.TOOL: t.tool_seconds,
            ExecutionKind.RETRIEVAL: t.retrieval_seconds,
            ExecutionKind.STORAGE: t.storage_seconds,
        }[kind]
