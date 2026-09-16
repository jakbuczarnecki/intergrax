# © Artur Czarnecki. All rights reserved.

"""Delegated provider plane invariant probe contract (P2.1 read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.execution_identity import ExecutionId


@dataclass(frozen=True, slots=True)
class DelegatedProviderInvariantFacts:
    """Optional facts — rules return NOT_APPLICABLE when unset."""

    provider_claims_canonical_execution_identity: bool | None = None
    correlation_execution_id: ExecutionId | None = None
    correlation_binding_execution_id: ExecutionId | None = None
    provider_correlation_fallback_synthesis_active: bool | None = None
    reattach_creates_new_work: bool | None = None


class DelegatedProviderInvariantProbe(Protocol):
    """Reads delegated provider facts without mutating P2.1 runtime."""

    def read_facts(self) -> DelegatedProviderInvariantFacts:
        ...


__all__ = ["DelegatedProviderInvariantFacts", "DelegatedProviderInvariantProbe"]
