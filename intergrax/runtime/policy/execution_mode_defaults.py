# © Artur Czarnecki. All rights reserved.

"""Map neutral execution mode to Nexus runtime policy defaults."""

from __future__ import annotations

from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.runtime.nexus.policies.runtime_policies import (
    FallbackPolicy,
    HitlPolicy,
    RetryPolicy,
    RuntimePolicies,
    TimeoutPolicy,
)


def runtime_policies_for_execution_mode(mode: ExecutionMode) -> RuntimePolicies:
    """Map execution mode to deterministic ``RuntimePolicies`` defaults."""
    if mode is ExecutionMode.STRICT:
        return RuntimePolicies(
            timeout=TimeoutPolicy(llm_seconds=20.0, tool_seconds=15.0),
            retry=RetryPolicy(max_attempts=1, backoff_seconds=0.0),
            fallback=FallbackPolicy(escalate_to_hitl=True),
            hitl=HitlPolicy(enabled=True),
        )
    if mode is ExecutionMode.EXPLORATORY:
        return RuntimePolicies(
            timeout=TimeoutPolicy(llm_seconds=45.0, tool_seconds=45.0),
            retry=RetryPolicy(max_attempts=5, backoff_seconds=0.0),
            fallback=FallbackPolicy(escalate_to_hitl=True),
            hitl=HitlPolicy(enabled=True),
        )
    return RuntimePolicies(
        timeout=TimeoutPolicy(),
        retry=RetryPolicy(max_attempts=3, backoff_seconds=0.0),
        fallback=FallbackPolicy(escalate_to_hitl=True),
        hitl=HitlPolicy(enabled=True),
    )
