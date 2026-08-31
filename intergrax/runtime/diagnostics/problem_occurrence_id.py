# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Stable deterministic identity for durable ProblemOccurrence records (DIAG-ENTERPRISE-2)."""

from __future__ import annotations

from typing import NewType

from intergrax.runtime.diagnostics.problem_lifecycle import ProblemOccurrence

ProblemOccurrenceId = NewType("ProblemOccurrenceId", str)


def problem_occurrence_id_for(occurrence: ProblemOccurrence) -> ProblemOccurrenceId:
    """
    Natural occurrence identity: one accepted subject per Problem.

    ``subject_ref`` is unique within ``(tenant_id, problem_id)``; the index token
  is stable across retries and idempotent append.
    """
    return ProblemOccurrenceId(occurrence.subject_ref.index_token)
