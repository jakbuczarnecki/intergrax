# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Detect conflicting recommendations — never merge (PREVENTIVE R6-Q)."""

from __future__ import annotations

from intergrax.contracts.preventive.conflict import (
    CONFLICTING_RECOMMENDATIONS,
    PreventiveRecommendationConflict,
)
from intergrax.contracts.preventive.recommendation import PreventiveRecommendation

_OPPOSING_PAIRS: tuple[tuple[str, str], ...] = (
    ("increase", "reduce"),
    ("increase", "decrease"),
    ("enable", "disable"),
    ("expand", "shrink"),
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _descriptions_conflict(left: str, right: str) -> bool:
    a = _normalize(left)
    b = _normalize(right)
    if a == b:
        return False
    for token_a, token_b in _OPPOSING_PAIRS:
        if token_a in a and token_b in b:
            return True
        if token_b in a and token_a in b:
            return True
    return a != b


def resolve_preventive_recommendation_conflicts(
    recommendations: tuple[PreventiveRecommendation, ...],
    *,
    scope_subject: str,
) -> tuple[PreventiveRecommendationConflict, ...]:
    """
    Same scope + distinct analyzer outputs that do not agree → conflict marker.

    Recommendations are returned unchanged; conflicts are additive metadata.
    """
    if len(recommendations) < 2:
        return ()
    conflicts: list[PreventiveRecommendationConflict] = []
    by_scope: dict[tuple[str, str], list[PreventiveRecommendation]] = {}
    for item in recommendations:
        key = (item.tenant_id, item.risk_signal_id)
        by_scope.setdefault(key, []).append(item)

    for (tenant_id, risk_signal_id), group in by_scope.items():
        if len(group) < 2:
            continue
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if left.analyzer_id == right.analyzer_id:
                    continue
                if not _descriptions_conflict(left.description, right.description):
                    continue
                pair = tuple(sorted((left.recommendation_id, right.recommendation_id)))
                conflicts.append(
                    PreventiveRecommendationConflict(
                        marker=CONFLICTING_RECOMMENDATIONS,
                        tenant_id=tenant_id,
                        risk_signal_id=risk_signal_id,
                        scope_subject=scope_subject,
                        recommendation_ids=pair,
                        analyzer_ids=(left.analyzer_id, right.analyzer_id),
                        summaries=(left.description, right.description),
                    ),
                )
    return tuple(conflicts)


class PreventiveRecommendationConflictResolver:
    """Enterprise conflict surface — markers only, no merged recommendations."""

    @staticmethod
    def resolve(
        recommendations: tuple[PreventiveRecommendation, ...],
        *,
        scope_subject: str,
    ) -> tuple[PreventiveRecommendationConflict, ...]:
        return resolve_preventive_recommendation_conflicts(
            recommendations,
            scope_subject=scope_subject,
        )


__all__ = [
    "PreventiveRecommendationConflictResolver",
    "resolve_preventive_recommendation_conflicts",
]
