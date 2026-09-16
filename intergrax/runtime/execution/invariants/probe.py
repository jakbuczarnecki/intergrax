# © Artur Czarnecki. All rights reserved.

"""Execution domain invariant probe contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExecutionInvariantFacts:
    """Read-only execution-plane facts for invariant rules."""

    identity_authority_module: str
    lifecycle_owner_module: str
    supported_execution_bypass_active: bool


class ExecutionInvariantProbe(Protocol):
    """Reads execution facts — does not own execution lifecycle."""

    def read_facts(self) -> ExecutionInvariantFacts:
        ...


__all__ = ["ExecutionInvariantFacts", "ExecutionInvariantProbe"]
