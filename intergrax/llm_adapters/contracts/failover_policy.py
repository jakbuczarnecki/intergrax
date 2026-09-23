# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Canonical failover progression and candidate eligibility contracts (EBH-2E-R6-R1-R4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile


class FailoverDecision(Enum):
    """Executor-facing progression decision after a candidate failure."""

    STOP = "stop"
    TRY_NEXT = "try_next"


@dataclass(frozen=True, slots=True)
class FailoverProgressionContext:
    """Input for failover progression policy after a failed candidate attempt."""

    attempt_index: int
    is_last_candidate: bool
    error: BaseException
    failover_retry_config: LLMCallConfig


@runtime_checkable
class FailoverPolicy(Protocol):
    """Pluggable policy for stop vs try-next after a candidate failure."""

    def decide_after_failure(self, context: FailoverProgressionContext) -> FailoverDecision:
        """Return whether the executor should attempt the next configured candidate."""


class FailoverCandidateEligibility(Enum):
    """Authoritative eligibility verdict for a failover candidate."""

    ALLOWED = "allowed"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class FailoverRoutingAuthorisationContext:
    """Minimal routing policy context for failover candidate authorisation."""

    allowed_profiles: tuple[LLMProfile, ...]


@dataclass(frozen=True, slots=True)
class FailoverCandidateEligibilityContext:
    """Input for candidate authorisation before execution."""

    candidate: LLMProfile
    candidate_index: int
    primary_selected: LLMProfile
    routing_authorisation: FailoverRoutingAuthorisationContext | None


@runtime_checkable
class FailoverEligibilityPolicy(Protocol):
    """Pluggable policy answering whether a failover candidate may execute."""

    def evaluate(self, context: FailoverCandidateEligibilityContext) -> FailoverCandidateEligibility:
        """Return whether the candidate is policy-authorised."""


class FailoverCandidateNotAuthorizedError(ValueError):
    """Raised when a configured failover candidate is outside canonical routing policy."""


__all__ = [
    "FailoverCandidateEligibility",
    "FailoverCandidateEligibilityContext",
    "FailoverCandidateNotAuthorizedError",
    "FailoverDecision",
    "FailoverEligibilityPolicy",
    "FailoverPolicy",
    "FailoverProgressionContext",
    "FailoverRoutingAuthorisationContext",
]
