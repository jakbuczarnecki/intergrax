# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Functional diagnostic check status (public contract)."""

from __future__ import annotations

from enum import StrEnum


class FunctionalDiagnosticCheckStatus(StrEnum):
    """
    Deterministic check outcome — not probabilistic confidence.

    PROVEN_PASS / PROVEN_FAIL require direct supporting evidence.
    INSUFFICIENT_EVIDENCE means absence of facts cannot be upgraded to FAIL.
    NOT_EVALUATED means the check was not reached in this analysis cycle.
    BLOCKED_BY_UPSTREAM means a dependency prevented evaluation.
    """

    PROVEN_PASS = "proven_pass"
    PROVEN_FAIL = "proven_fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_EVALUATED = "not_evaluated"
    BLOCKED_BY_UPSTREAM = "blocked_by_upstream"


__all__ = [
    "FunctionalDiagnosticCheckStatus",
]
