# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical runtime inspection read port (INSPECT-01-A)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.runtime_inspection.query import RuntimeInspectionQuery
from intergrax.contracts.runtime_inspection.snapshot import RuntimeInspectionSnapshot


@runtime_checkable
class RuntimeInspectionReadPort(Protocol):
    """Read-only federated runtime inspection — never a command surface."""

    def inspect(self, query: RuntimeInspectionQuery) -> RuntimeInspectionSnapshot: ...


__all__ = ["RuntimeInspectionReadPort"]
