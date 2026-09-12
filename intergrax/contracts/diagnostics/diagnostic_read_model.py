# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Diagnostic read-side persistence models (HARDENING-8)."""

from __future__ import annotations

from intergrax.contracts.diagnostics.problem_persistence import ProblemListPage
from intergrax.contracts.diagnostics.problem_record import PersistedProblem

__all__ = ["PersistedProblem", "ProblemListPage"]
