"""Merge and conflict policies for query understanding."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.query_understanding.contracts import (
    ExtractedConstraintRecord,
    ExtractedIdentifierRecord,
    ExtractedNegativeConstraintRecord,
    ExtractedSoftPreferenceRecord,
    QueryInterpretationCandidate,
    QueryUnderstandingExtractionBundle,
    QueryUnderstandingIssue,
    QueryUnderstandingIssueCode,
)


class QueryUnderstandingMergePolicy(Protocol):
    def merge(
        self,
        deterministic: QueryUnderstandingExtractionBundle,
        interpreter: QueryInterpretationCandidate | None,
    ) -> tuple[QueryUnderstandingExtractionBundle, tuple[QueryUnderstandingIssue, ...]]:
        ...


def _identifier_key(item: ExtractedIdentifierRecord) -> tuple[str, str]:
    return (item.identifier.identifier_type.value, item.identifier.value)


def _conflicting_identifiers(
    records: tuple[ExtractedIdentifierRecord, ...],
) -> bool:
    by_type: dict[str, set[str]] = {}
    for item in records:
        by_type.setdefault(item.identifier.identifier_type.value, set()).add(
            item.identifier.value.casefold()
        )
    return any(len(values) > 1 for values in by_type.values())


def _merge_constraints(
    primary: tuple[ExtractedConstraintRecord, ...],
    secondary: tuple[ExtractedConstraintRecord, ...],
) -> tuple[ExtractedConstraintRecord, ...]:
    merged: dict[str, ExtractedConstraintRecord] = {
        item.constraint.attribute_name.casefold(): item for item in primary
    }
    for item in secondary:
        key = item.constraint.attribute_name.casefold()
        if key in merged:
            if merged[key].constraint.value.casefold() != item.constraint.value.casefold():
                continue
            continue
        merged[key] = item
    return tuple(merged[k] for k in sorted(merged))


class DeterministicQueryUnderstandingMergePolicy:
    """Deterministic extraction wins; interpreter fills only non-conflicting gaps."""

    def merge(
        self,
        deterministic: QueryUnderstandingExtractionBundle,
        interpreter: QueryInterpretationCandidate | None,
    ) -> tuple[QueryUnderstandingExtractionBundle, tuple[QueryUnderstandingIssue, ...]]:
        issues: list[QueryUnderstandingIssue] = []
        identifiers = list(deterministic.identifiers)
        required = list(deterministic.required_constraints)
        negative = list(deterministic.negative_constraints)
        soft = list(deterministic.soft_preferences)
        missing = list(deterministic.missing_requirements)

        if interpreter is not None:
            det_id_keys = {_identifier_key(item) for item in identifiers}
            for item in interpreter.identifiers:
                key = _identifier_key(item)
                if key in det_id_keys:
                    continue
                if any(
                    existing.identifier.identifier_type == item.identifier.identifier_type
                    and existing.identifier.value.casefold() != item.identifier.value.casefold()
                    for existing in identifiers
                ):
                    issues.append(
                        QueryUnderstandingIssue(
                            code=QueryUnderstandingIssueCode.CONFLICTING_USER_CONSTRAINT,
                            detail="interpreter identifier conflicts with deterministic extraction",
                        )
                    )
                    continue
                identifiers.append(item)
                det_id_keys.add(key)

            required = list(
                _merge_constraints(
                    tuple(required),
                    interpreter.required_constraints,
                )
            )
            neg_keys = {
                (n.constraint.attribute_name.casefold(), n.constraint.excluded_value.casefold())
                for n in negative
            }
            for item in interpreter.negative_constraints:
                key = (
                    item.constraint.attribute_name.casefold(),
                    item.constraint.excluded_value.casefold(),
                )
                if key not in neg_keys:
                    negative.append(item)
                    neg_keys.add(key)

            soft_keys = {
                (s.preference.attribute_name.casefold(), s.preference.value.casefold()) for s in soft
            }
            for item in interpreter.soft_preferences:
                key = (item.preference.attribute_name.casefold(), item.preference.value.casefold())
                if key not in soft_keys:
                    soft.append(item)
                    soft_keys.add(key)

            missing.extend(interpreter.missing_requirements)

        identifiers_sorted = tuple(sorted(identifiers, key=lambda r: (r.source_span.start_offset, r.identifier.value)))
        if _conflicting_identifiers(identifiers_sorted):
            issues.append(
                QueryUnderstandingIssue(
                    code=QueryUnderstandingIssueCode.CONFLICTING_USER_CONSTRAINT,
                    detail="conflicting identifier values for the same type",
                )
            )

        bundle = QueryUnderstandingExtractionBundle(
            identifiers=identifiers_sorted,
            required_constraints=tuple(
                sorted(required, key=lambda r: r.constraint.attribute_name)
            ),
            negative_constraints=tuple(
                sorted(negative, key=lambda r: r.constraint.attribute_name)
            ),
            soft_preferences=tuple(sorted(soft, key=lambda r: r.preference.attribute_name)),
            missing_requirements=tuple(missing),
        )
        return bundle, tuple(issues)
