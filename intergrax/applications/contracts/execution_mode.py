# © Artur Czarnecki. All rights reserved.

"""Tier-3 execution mode → Nexus runtime policy defaults (Phase H-APP.2.5)."""

from __future__ import annotations

from enum import Enum

from intergrax.runtime.nexus.policies.runtime_policies import (
    FallbackPolicy,
    HitlPolicy,
    RetryPolicy,
    RuntimePolicies,
    TimeoutPolicy,
)


class ExecutionMode(str, Enum):
    """Application-level execution posture (IDEAL §3.4)."""

    STRICT = "strict"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


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
