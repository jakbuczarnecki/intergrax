# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P5 manifest reconciliation — consumers only; pytest owns execution."""

from __future__ import annotations

from pathlib import Path

from tests.qualification.qual_x._p5_freeze_selection_catalog import (
    QUAL_X_P5_FREEZE_SELECTION_MANIFEST,
    QUAL_X_P5_FRZ_REG_EVIDENCE_LEDGER,
    QualXFreezeEvidenceClass,
    QualXFreezeEvidenceStatus,
    QualXFreezeMandatoryStatus,
    QualXFreezeQualificationDimension,
    QualXFreezeSelectionEntry,
    qual_x_p5_entry_pytest_node_ids,
    qual_x_p5_mandatory_collectable_node_ids,
)
from tests.qualification.qual_x._p4_determinism_environment_support import (
    QualXP4Outcome,
    mandatory_outcome_from_pytest_execute_summary,
    mandatory_qualification_allows_pass,
)

QUAL_X_P5_KNOWN_OWNING_STAGE_PREFIXES: tuple[str, ...] = (
    "QUAL-X-P",
    "QUAL-X-P5",
    "HARNESS-01",
    "PROD-Q",
    "STATE-X",
    "TRACE-X",
    "EBH",
)

_EXPECTED_FRZ_REG_IDS: frozenset[str] = frozenset(
    f"FRZ-REG-{index:02d}" for index in range(1, 11)
)


def validate_manifest_selection_ids(
    entries: tuple[QualXFreezeSelectionEntry, ...],
) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    errors: list[str] = []
    for entry in entries:
        seen[entry.selection_id] = seen.get(entry.selection_id, 0) + 1
    for selection_id, count in seen.items():
        if count > 1:
            errors.append(f"duplicate selection_id: {selection_id}")
    return tuple(errors)


def validate_manifest_frz_criteria(
    entries: tuple[QualXFreezeSelectionEntry, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    for entry in entries:
        if not entry.frz_criteria:
            errors.append(f"missing frz_criteria: {entry.selection_id}")
        for criterion in entry.frz_criteria:
            if not criterion.startswith("FRZ-"):
                errors.append(f"invalid frz token on {entry.selection_id}: {criterion}")
    return tuple(errors)


def validate_manifest_owning_stages(
    entries: tuple[QualXFreezeSelectionEntry, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    for entry in entries:
        if not entry.owning_stage.strip():
            errors.append(f"empty owning_stage: {entry.selection_id}")
            continue
        if not any(
            entry.owning_stage.startswith(prefix)
            for prefix in QUAL_X_P5_KNOWN_OWNING_STAGE_PREFIXES
        ):
            errors.append(f"unknown owning_stage: {entry.selection_id} -> {entry.owning_stage}")
    return tuple(errors)


def validate_manifest_dependencies(
    entries: tuple[QualXFreezeSelectionEntry, ...],
) -> tuple[str, ...]:
    known = {entry.selection_id for entry in entries}
    errors: list[str] = []
    for entry in entries:
        for dep in entry.depends_on:
            if dep not in known:
                errors.append(f"unresolved dependency {dep} on {entry.selection_id}")
    return tuple(errors)


def validate_frz_reg_ledger_complete() -> tuple[str, ...]:
    present = {row.criterion_id for row in QUAL_X_P5_FRZ_REG_EVIDENCE_LEDGER}
    missing = sorted(_EXPECTED_FRZ_REG_IDS - present)
    if missing:
        return tuple(f"missing FRZ-REG ledger row: {item}" for item in missing)
    return ()


def covered_qual_x_responsibilities(
    entries: tuple[QualXFreezeSelectionEntry, ...],
) -> frozenset[str]:
    covered: set[str] = set()
    for entry in entries:
        covered.update(entry.qual_x_responsibilities)
    return frozenset(covered)


def dimension_a_node_ids_excluding_dimension_b(
    entries: tuple[QualXFreezeSelectionEntry, ...],
) -> tuple[str, ...]:
    dim_b_nodes: set[str] = set()
    for entry in entries:
        if (
            entry.qualification_dimension
            is QualXFreezeQualificationDimension.LIVE_PROVIDER_DIMENSION_B
        ):
            dim_b_nodes.update(qual_x_p5_entry_pytest_node_ids(entry))
    all_collectable = set(qual_x_p5_mandatory_collectable_node_ids())
    return tuple(sorted(all_collectable - dim_b_nodes))


def historical_linkage_cannot_mandatory_pass(
    entry: QualXFreezeSelectionEntry,
) -> bool:
    """Historical-only rows must not be classified as unconditional current-HEAD PASS."""
    if entry.evidence_status is QualXFreezeEvidenceStatus.HISTORICAL_ONLY:
        return entry.mandatory_status is QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD
    if entry.expected_evidence_class is QualXFreezeEvidenceClass.HISTORICAL_STAGE_LINKAGE:
        return entry.mandatory_status in (
            QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
            QualXFreezeMandatoryStatus.MANDATORY_EXECUTE_AT_QUAL_X_FINAL,
        )
    return True


def dimension_b_entry_must_not_count_for_dimension_a_pass(
    entry: QualXFreezeSelectionEntry,
) -> bool:
    if (
        entry.qualification_dimension
        is not QualXFreezeQualificationDimension.LIVE_PROVIDER_DIMENSION_B
    ):
        return True
    return entry.mandatory_status is QualXFreezeMandatoryStatus.EXCLUDED_FROM_DIMENSION_A


def qualification_sha_matches_baseline(
    *,
    baseline_sha: str,
    observed_sha: str,
) -> QualXP4Outcome:
    if baseline_sha == observed_sha:
        return QualXP4Outcome.PASS
    return QualXP4Outcome.INVALID_QUALIFICATION


def missing_modules_for_manifest(
    repo_root: Path,
    entries: tuple[QualXFreezeSelectionEntry, ...] = QUAL_X_P5_FREEZE_SELECTION_MANIFEST,
) -> tuple[str, ...]:
    missing: list[str] = []
    root = Path(repo_root)
    for entry in entries:
        for rel in entry.pytest_modules:
            if not (root / rel).is_file():
                missing.append(rel)
        if entry.coordination_batch_module and not (
            root / entry.coordination_batch_module
        ).is_file():
            missing.append(entry.coordination_batch_module)
    return tuple(sorted(set(missing)))


def reconcile_manifest_integrity() -> tuple[str, ...]:
    errors: list[str] = []
    errors.extend(validate_manifest_selection_ids(QUAL_X_P5_FREEZE_SELECTION_MANIFEST))
    errors.extend(validate_manifest_frz_criteria(QUAL_X_P5_FREEZE_SELECTION_MANIFEST))
    errors.extend(validate_manifest_owning_stages(QUAL_X_P5_FREEZE_SELECTION_MANIFEST))
    errors.extend(validate_manifest_dependencies(QUAL_X_P5_FREEZE_SELECTION_MANIFEST))
    errors.extend(validate_frz_reg_ledger_complete())
    for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST:
        if not historical_linkage_cannot_mandatory_pass(entry):
            errors.append(
                f"historical-only entry misclassified as unconditional pass: {entry.selection_id}"
            )
        if not dimension_b_entry_must_not_count_for_dimension_a_pass(entry):
            errors.append(f"dimension B entry may count toward A: {entry.selection_id}")
    return tuple(errors)


__all__ = [
    "QualXP4Outcome",
    "dimension_a_node_ids_excluding_dimension_b",
    "historical_linkage_cannot_mandatory_pass",
    "mandatory_outcome_from_pytest_execute_summary",
    "mandatory_qualification_allows_pass",
    "missing_modules_for_manifest",
    "qualification_sha_matches_baseline",
    "reconcile_manifest_integrity",
    "validate_frz_reg_ledger_complete",
    "validate_manifest_dependencies",
    "validate_manifest_frz_criteria",
    "validate_manifest_owning_stages",
    "validate_manifest_selection_ids",
]
