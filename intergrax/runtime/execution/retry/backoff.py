# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded backoff policy for execution-attempt retry (NPSC-5E/R1)."""

from __future__ import annotations

import random

from intergrax.contracts.execution_retry import BackoffKind, BackoffPolicyConfig
from intergrax.contracts.resilience_policy import ResiliencePolicy


def backoff_config_from_resilience_policy(policy: ResiliencePolicy) -> BackoffPolicyConfig:
    kind = BackoffKind.EXPONENTIAL
    if policy.backoff == "fixed":
        kind = BackoffKind.FIXED
    elif policy.backoff == "none":
        kind = BackoffKind.NONE
    return BackoffPolicyConfig(kind=kind)


def compute_backoff_delay(
    *,
    attempt_number: int,
    config: BackoffPolicyConfig,
    retry_after_seconds: float | None = None,
) -> float:
    if retry_after_seconds is not None:
        return min(max(retry_after_seconds, 0.0), config.max_delay_seconds)

    if config.kind is BackoffKind.NONE:
        return 0.0

    if config.kind is BackoffKind.FIXED:
        delay = config.base_delay_seconds
    else:
        exponent = max(attempt_number - 1, 0)
        delay = config.base_delay_seconds * (config.multiplier ** exponent)

    delay = min(delay, config.max_delay_seconds)

    if config.kind is BackoffKind.JITTERED and config.jitter_ratio > 0.0:
        jitter = delay * config.jitter_ratio
        delay = random.uniform(delay - jitter, delay + jitter)

    return min(max(delay, 0.0), config.max_delay_seconds)
