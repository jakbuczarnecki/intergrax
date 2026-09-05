# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Frozen normative capability stage vocabulary (CAPABILITY-CATALOG-1 Stage 1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityStageVocabulary(StrEnum):
    """Shared vocabulary only — not a cross-domain lifecycle engine.

    Domains may implement subsets. States describe conceptual separation between
    catalog availability, discovery output, selection, and domain-owned lifecycle.
    """

    AVAILABLE = "available"
    DISCOVERED = "discovered"
    SELECTED = "selected"
    INSTALLED = "installed"
    ENABLED = "enabled"
    MATERIALIZED = "materialized"
    ACTIVE = "active"
    EXECUTABLE = "executable"


NORMATIVE_CAPABILITY_STAGE_VOCABULARY: Final[frozenset[CapabilityStageVocabulary]] = (
    frozenset(CapabilityStageVocabulary)
)
