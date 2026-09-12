# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy-specific Problem reconciliation key port surface (DIAG-5D / HARDENING-8)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProblemReconciliationKey(Protocol):
    """Strategy-specific recurrence evidence — not opaque Problem identity."""

    @property
    def kind(self) -> str: ...

    def index_token(self) -> str: ...


__all__ = ["ProblemReconciliationKey"]
