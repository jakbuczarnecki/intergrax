# © Artur Czarnecki. All rights reserved.

"""Legacy mandatory target expansion and normalization for NPSC-5F/R1 Final."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.execution_qualification.catalog.expansion import (
    CatalogRequiredTarget,
    expand_mandatory_subprocesses,
    is_nested_orchestrator_leaf,
    unique_required_leaf_targets,
)
from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5F_R1_FINAL_MANDATORY,
)
from testing_support.execution_qualification.catalog.normalize import (
    normalize_pytest_arguments,
)
from testing_support.execution_qualification.catalog.orchestrators import (
    LEGACY_ORCHESTRATOR_EXPANSION_PATHS,
)
from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)

FrozenMandatorySource = FrozenPytestSuiteSource

_ORCHESTRATOR_SINGLE_TARGET = LEGACY_ORCHESTRATOR_EXPANSION_PATHS


@dataclass(frozen=True, slots=True)
class LegacyRequiredTarget:
    """One legacy mandatory pytest invocation (label + normalized argument vector)."""

    display_label: str
    pytest_arguments: tuple[str, ...]

    @classmethod
    def from_targets(
        cls, display_label: str, targets: list[str]
    ) -> LegacyRequiredTarget:
        return cls(
            display_label=display_label,
            pytest_arguments=normalize_pytest_arguments(targets),
        )


def normalize_required_target_set(
    targets: tuple[LegacyRequiredTarget, ...],
) -> frozenset[tuple[str, ...]]:
    return frozenset(entry.pytest_arguments for entry in targets)


def _to_legacy(target: CatalogRequiredTarget) -> LegacyRequiredTarget:
    return LegacyRequiredTarget(
        display_label=target.display_label,
        pytest_arguments=target.pytest_arguments,
    )


def expand_legacy_mandatory_subprocesses(
    source: FrozenMandatorySource,
) -> tuple[LegacyRequiredTarget, ...]:
    return tuple(_to_legacy(entry) for entry in expand_mandatory_subprocesses(source))


def legacy_r1_final_mandatory_subprocesses() -> tuple[LegacyRequiredTarget, ...]:
    return expand_legacy_mandatory_subprocesses(NPSC5F_R1_FINAL_MANDATORY)


def legacy_r1_final_required_leaf_targets() -> tuple[LegacyRequiredTarget, ...]:
    return tuple(
        _to_legacy(entry)
        for entry in unique_required_leaf_targets(NPSC5F_R1_FINAL_MANDATORY)
    )


def legacy_subprocess_count() -> int:
    return len(legacy_r1_final_mandatory_subprocesses())


def legacy_reference_mandatory_source_from_tests() -> FrozenMandatorySource:
    """Legacy parity/reference path — live test module tuple for drift detection."""
    from tests.unit.runtime.architecture import (
        test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity as npsc5f_r1_final,
    )

    return npsc5f_r1_final._MANDATORY_SUITES
