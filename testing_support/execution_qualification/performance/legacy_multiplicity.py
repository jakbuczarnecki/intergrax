# © Artur Czarnecki. All rights reserved.

"""Legacy orchestration expansion counts for performance duplicate accounting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from testing_support.execution_qualification.catalog.expansion import (
    CatalogRequiredTarget,
    expand_mandatory_subprocesses,
    unique_required_leaf_targets,
)
from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)
from testing_support.execution_qualification.semantic_parity.matrix import (
    GLOBAL_SEMANTIC_PARITY_MATRIX,
)
from testing_support.execution_qualification.semantic_parity.models import (
    QualificationSemanticParityCase,
)


@dataclass(frozen=True, slots=True)
class LegacySuiteInvocationMultiplicity:
    pytest_arguments: tuple[str, ...]
    legacy_invocation_count: int

    def __post_init__(self) -> None:
        if self.legacy_invocation_count < 1:
            raise ValueError("legacy_invocation_count must be >= 1")


@dataclass(frozen=True, slots=True)
class LegacyExecutionMultiplicity:
    """Recursive legacy expansion vs semantic unique leaf vectors."""

    legacy_logical_subprocess_count: int
    legacy_semantic_unique_leaf_count: int
    per_argument_multiplicity: tuple[LegacySuiteInvocationMultiplicity, ...]

    def __post_init__(self) -> None:
        if self.legacy_logical_subprocess_count < 1:
            raise ValueError("legacy_logical_subprocess_count must be >= 1")
        if self.legacy_semantic_unique_leaf_count < 1:
            raise ValueError("legacy_semantic_unique_leaf_count must be >= 1")
        if (
            self.legacy_semantic_unique_leaf_count
            > self.legacy_logical_subprocess_count
        ):
            raise ValueError(
                "legacy_semantic_unique_leaf_count cannot exceed logical subprocess count"
            )


def _multiplicity_rows(
    subprocesses: tuple[CatalogRequiredTarget, ...],
) -> tuple[LegacySuiteInvocationMultiplicity, ...]:
    counts = Counter(entry.pytest_arguments for entry in subprocesses)
    rows = [
        LegacySuiteInvocationMultiplicity(
            pytest_arguments=arguments,
            legacy_invocation_count=count,
        )
        for arguments, count in counts.items()
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (-row.legacy_invocation_count, row.pytest_arguments),
        )
    )


def legacy_execution_multiplicity_for_source(
    source: FrozenPytestSuiteSource,
) -> LegacyExecutionMultiplicity:
    subprocesses = expand_mandatory_subprocesses(source)
    unique = unique_required_leaf_targets(source)
    return LegacyExecutionMultiplicity(
        legacy_logical_subprocess_count=len(subprocesses),
        legacy_semantic_unique_leaf_count=len(unique),
        per_argument_multiplicity=_multiplicity_rows(subprocesses),
    )


def parity_case_for_profile(profile_id: str) -> QualificationSemanticParityCase:
    for case in GLOBAL_SEMANTIC_PARITY_MATRIX:
        if case.profile_id == profile_id:
            return case
    raise KeyError(f"no semantic parity case for profile_id={profile_id!r}")


def legacy_execution_multiplicity_for_profile(
    profile_id: str,
) -> LegacyExecutionMultiplicity:
    case = parity_case_for_profile(profile_id)
    return legacy_execution_multiplicity_for_source(case.legacy_semantic_source)
