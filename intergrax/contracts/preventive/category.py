# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Extensible recommendation categories — base namespace + plugin dotted ids (PREVENTIVE R6)."""

from __future__ import annotations

_BASE_CATEGORIES: frozenset[str] = frozenset(
    {
        "OBSERVE",
        "INVESTIGATE",
        "OPTIMIZE",
        "CONFIGURATION_REVIEW",
        "CAPACITY_REVIEW",
        "SECURITY_REVIEW",
    },
)


class PreventiveRecommendationCategory:
    """String namespace — plugins may register dotted categories (e.g. crm.customer_data_validation)."""

    OBSERVE = "OBSERVE"
    INVESTIGATE = "INVESTIGATE"
    OPTIMIZE = "OPTIMIZE"
    CONFIGURATION_REVIEW = "CONFIGURATION_REVIEW"
    CAPACITY_REVIEW = "CAPACITY_REVIEW"
    SECURITY_REVIEW = "SECURITY_REVIEW"

    @staticmethod
    def is_valid(value: str) -> bool:
        normalized = value.strip()
        if not normalized:
            return False
        if normalized in _BASE_CATEGORIES:
            return True
        if "." not in normalized:
            return False
        parts = normalized.split(".")
        return all(part.strip() for part in parts) and all(
            part.replace("_", "").isalnum() or part.isidentifier() for part in parts
        )

    @staticmethod
    def validate(value: str) -> str:
        normalized = value.strip()
        if not PreventiveRecommendationCategory.is_valid(normalized):
            raise ValueError(f"invalid preventive recommendation category: {value!r}")
        return normalized


__all__ = ["PreventiveRecommendationCategory"]
