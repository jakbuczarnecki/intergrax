# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reliability policy contract for Execution Runtime persistence failures (decision only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intergrax.contracts.execution_evidence.persistence_failure_contract import (
    EvidencePersistenceFailureCategory,
)

__all__ = [
    "KnownPersistencePortFailure",
    "PersistenceReliabilityDisposition",
    "PersistenceReliabilityPolicy",
    "PersistenceReliabilityPolicyDecision",
    "PersistenceReliabilityPolicyRequest",
    "RuntimeEvidenceDurabilityRequirement",
]


class RuntimeEvidenceDurabilityRequirement(str, Enum):
    """Durability posture at the persistence port (runtime-facing, provider-agnostic)."""

    MANDATORY = "mandatory"
    BEST_EFFORT = "best_effort"


@dataclass(frozen=True, slots=True)
class KnownPersistencePortFailure:
    """Normalized persistence problem reported by Execution Runtime to reliability policy."""

    category: EvidencePersistenceFailureCategory


class PersistenceReliabilityDisposition(str, Enum):
    """Explicit resilience reaction chosen by policy."""

    FAIL_CLOSED = "fail_closed"
    ALLOW_CONTINUE = "allow_continue"


@dataclass(frozen=True, slots=True)
class PersistenceReliabilityPolicyDecision:
    """Typed outcome of a reliability policy evaluation."""

    disposition: PersistenceReliabilityDisposition


@dataclass(frozen=True, slots=True)
class PersistenceReliabilityPolicyRequest:
    """Inputs for persistence reliability policy — no provider or storage details."""

    failure: KnownPersistencePortFailure
    durability: RuntimeEvidenceDurabilityRequirement


class PersistenceReliabilityPolicy(Protocol):
    """Pluggable policy: maps a known runtime persistence problem to a resilience decision."""

    def decide(
        self,
        request: PersistenceReliabilityPolicyRequest,
    ) -> PersistenceReliabilityPolicyDecision:
        """Return how Execution Runtime should react to the reported persistence problem."""
