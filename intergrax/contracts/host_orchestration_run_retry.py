# © Artur Czarnecki. All rights reserved.

"""Neutral host orchestration run retry configuration (EE maps to Nexus RetryPolicy)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HostOrchestrationRunRetrySpec:
    max_retries: int = 0
    retry_alternate_agent: bool = True


__all__ = ["HostOrchestrationRunRetrySpec"]
