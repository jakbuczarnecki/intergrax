# © Artur Czarnecki. All rights reserved.

"""Legacy mandatory target expansion and normalization for NPSC-5F/R1 Final."""

from __future__ import annotations

from dataclasses import dataclass

from tests.unit.runtime.architecture import (
    test_npsc5e_final_recovery_plane_qualification_and_freeze as npsc5e_final,
)
from tests.unit.runtime.architecture import (
    test_npsc5e_r2_final_checkpoint_durable_resume_qualification as npsc5e_r2_final,
)
from tests.unit.runtime.architecture import (
    test_npsc5e_r3_final_child_fanout_partial_recovery_qualification as npsc5e_r3_final,
)
from tests.unit.runtime.architecture import (
    test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity as npsc5f_r1_final,
)

FrozenMandatorySource = tuple[tuple[str, list[str]], ...]

_ORCHESTRATOR_SINGLE_TARGET: frozenset[str] = frozenset(
    {
        "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py",
        "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py",
        "tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py",
    },
)


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


def normalize_pytest_arguments(targets: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(target.replace("\\", "/") for target in targets)


def normalize_required_target_set(
    targets: tuple[LegacyRequiredTarget, ...],
) -> frozenset[tuple[str, ...]]:
    return frozenset(entry.pytest_arguments for entry in targets)


def _orchestrator_expansion(targets: list[str]) -> FrozenMandatorySource | None:
    if len(targets) != 1:
        return None
    normalized = normalize_pytest_arguments(targets)[0]
    if normalized == (
        "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
    ):
        return npsc5e_final._MANDATORY_SUITES
    if normalized == (
        "tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py"
    ):
        return npsc5e_r3_final._MANDATORY_SUITES
    if normalized == (
        "tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py"
    ):
        return npsc5e_r2_final._MANDATORY_SUITES
    return None


def _expand_one_mandatory(
    display_label: str,
    targets: list[str],
) -> tuple[LegacyRequiredTarget, ...]:
    nested = _orchestrator_expansion(targets)
    if nested is None:
        return (LegacyRequiredTarget.from_targets(display_label, targets),)
    expanded: list[LegacyRequiredTarget] = []
    for child_label, child_targets in nested:
        child_path = f"{display_label}>{child_label}"
        expanded.extend(_expand_one_mandatory(child_path, child_targets))
    return tuple(expanded)


def expand_legacy_mandatory_subprocesses(
    source: FrozenMandatorySource,
) -> tuple[LegacyRequiredTarget, ...]:
    """Expand orchestrator single-file targets into nested mandatory subprocesses (legacy order)."""
    expanded: list[LegacyRequiredTarget] = []
    for display_label, targets in source:
        expanded.extend(_expand_one_mandatory(display_label, targets))
    return tuple(expanded)


def legacy_r1_final_mandatory_subprocesses() -> tuple[LegacyRequiredTarget, ...]:
    return expand_legacy_mandatory_subprocesses(npsc5f_r1_final._MANDATORY_SUITES)


def legacy_r1_final_required_leaf_targets() -> tuple[LegacyRequiredTarget, ...]:
    """Unique leaf targets (semantic coverage set) in first-seen order."""
    subprocesses = legacy_r1_final_mandatory_subprocesses()
    seen: set[tuple[str, ...]] = set()
    unique: list[LegacyRequiredTarget] = []
    for entry in subprocesses:
        if entry.pytest_arguments in seen:
            continue
        seen.add(entry.pytest_arguments)
        unique.append(entry)
    return tuple(unique)


def legacy_subprocess_count() -> int:
    return len(legacy_r1_final_mandatory_subprocesses())


def is_nested_orchestrator_leaf(pytest_arguments: tuple[str, ...]) -> bool:
    return (
        len(pytest_arguments) == 1
        and pytest_arguments[0] in _ORCHESTRATOR_SINGLE_TARGET
    )
