# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Diagnostic Problem repository port alias (HARDENING-8)."""

from __future__ import annotations

from intergrax.contracts.diagnostics.problem_persistence import ProblemPersistence

DiagnosticProblemRepository = ProblemPersistence

__all__ = ["DiagnosticProblemRepository", "ProblemPersistence"]
