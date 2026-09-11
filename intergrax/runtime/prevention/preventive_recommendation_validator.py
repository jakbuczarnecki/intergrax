# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Candidate validation — evidence mandatory (PREVENTIVE R6)."""

from __future__ import annotations

from intergrax.contracts.preventive.recommendation import PreventiveRecommendationCandidate


class PreventiveRecommendationValidationError(ValueError):
    """Candidate rejected before recommendation minting."""


def validate_candidate(candidate: PreventiveRecommendationCandidate) -> None:
    if not candidate.evidence_refs:
        raise PreventiveRecommendationValidationError(
            "recommendation requires evidence_refs",
        )


__all__ = ["PreventiveRecommendationValidationError", "validate_candidate"]
