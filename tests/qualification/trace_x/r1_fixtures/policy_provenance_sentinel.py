# © Artur Czarnecki. All rights reserved.

"""Qualification-only policy provenance discovery sentinel (not production authority)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyntheticGovernanceRevisionTrace:
    """Structural policy provenance probe — name absent from P5 classification registry."""

    policy_document_id: str
    revision_id: str
    decision_id: str
