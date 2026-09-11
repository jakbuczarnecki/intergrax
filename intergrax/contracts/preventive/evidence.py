# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Evidence binding for preventive recommendations (PREVENTIVE R6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecommendationEvidenceReference:
    """One auditable evidence link — recommendations without refs are rejected."""

    source_type: str
    source_id: str
    relation: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source_type", self.source_type),
            ("source_id", self.source_id),
            ("relation", self.relation),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")


__all__ = ["RecommendationEvidenceReference"]
