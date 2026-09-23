# © Artur Czarnecki. All rights reserved.

"""Typed durable persistence conflicts for suspended execution operations (UCA-6C-R6)."""

from __future__ import annotations


class SuspendedOperationPersistenceConflictError(Exception):
    """Optimistic concurrency conflict while persisting suspended-operation state."""


__all__ = ["SuspendedOperationPersistenceConflictError"]
