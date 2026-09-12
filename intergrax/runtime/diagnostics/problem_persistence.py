# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Runtime re-exports — canonical Problem persistence port lives in ``intergrax.contracts.diagnostics``."""

from __future__ import annotations

from intergrax.contracts.diagnostics.problem_persistence import (
    ProblemListPage,
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
    ProblemPersistenceIntegrityReason,
)

__all__ = [
    "ProblemListPage",
    "ProblemPersistence",
    "ProblemPersistenceConflictError",
    "ProblemPersistenceIntegrityError",
    "ProblemPersistenceIntegrityReason",
]
