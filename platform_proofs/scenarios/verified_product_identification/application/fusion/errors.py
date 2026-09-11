"""Fusion-layer validation failures."""

from __future__ import annotations


class OfferCandidateFusionError(ValueError):
    """Raised when offer-level candidate fusion input violates domain invariants."""
