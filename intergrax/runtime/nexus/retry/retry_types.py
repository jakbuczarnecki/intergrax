# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Retry DTOs without agent-engine dependencies (Phase Q+-N.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetryRecord:
    attempt: int
    agent_id: str
    reason: str
    alternate_agent_id: Optional[str] = None


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    reason: str = ""
    alternate_agent_id: Optional[str] = None
