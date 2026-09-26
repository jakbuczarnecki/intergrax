# © Artur Czarnecki. All rights reserved.

"""Evaluate public contract modules against dependency boundary policy."""

from __future__ import annotations

from pathlib import Path

from testing_support.architecture.public_contract_boundary.debt_registry import (
    PUBLIC_CONTRACT_DEPENDENCY_DEBT,
)
from testing_support.architecture.public_contract_boundary.debt_validation import (
    validate_debt_registry,
)
from testing_support.architecture.public_contract_boundary.discovery import (
    discover_public_contract_source_files,
    path_to_module_name,
    validate_supplemental_public_contract_surfaces,
)
from testing_support.architecture.public_contract_boundary.import_extraction import (
    extract_imports_from_file,
)
from testing_support.architecture.public_contract_boundary.models import (
    ContractDependencyDebtEntry,
    ContractDependencyViolation,
    PublicContractBoundaryGateResult,
)
from testing_support.architecture.public_contract_boundary.policy import (
    classify_intergrax_dependency,
    is_allowed_compat_reexport,
)
from testing_support.architecture.public_contract_boundary.supplemental_surfaces import (
    SUPPLEMENTAL_PUBLIC_CONTRACT_SURFACES,
    SupplementalPublicContractSurface,
)


def _debt_covers_violation(
    violation: ContractDependencyViolation,
    entry: ContractDependencyDebtEntry,
) -> bool:
    if violation.source_module != entry.source_module:
        return False
    if violation.rule_id != entry.rule_id:
        return False
    return violation.imported_module == entry.forbidden_import_module


def _collect_violations(
    repo_root: Path,
    *,
    supplemental_surfaces: tuple[SupplementalPublicContractSurface, ...] | None,
) -> list[ContractDependencyViolation]:
    intergrax_root = repo_root / "intergrax"
    violations: list[ContractDependencyViolation] = []
    for path in discover_public_contract_source_files(
        repo_root,
        supplemental_surfaces=supplemental_surfaces,
    ):
        if not path.is_file():
            continue
        source_module = path_to_module_name(path, intergrax_root=intergrax_root)
        rel_path = path.relative_to(repo_root).as_posix()
        for extracted in extract_imports_from_file(path, intergrax_root=intergrax_root):
            if is_allowed_compat_reexport(source_module, extracted.imported_module):
                continue
            rule = classify_intergrax_dependency(extracted.imported_module)
            if rule is None:
                continue
            violations.append(
                ContractDependencyViolation(
                    source_module=source_module,
                    imported_module=extracted.imported_module,
                    rule_id=rule,
                    source_path=rel_path,
                    line=extracted.line,
                ),
            )
    violations.sort(key=lambda item: item.sort_key())
    return violations


def _find_stale_debt(
    violations: list[ContractDependencyViolation],
    debt_entries: tuple[ContractDependencyDebtEntry, ...],
) -> tuple[ContractDependencyDebtEntry, ...]:
    stale: list[ContractDependencyDebtEntry] = []
    for entry in debt_entries:
        has_match = any(_debt_covers_violation(v, entry) for v in violations)
        if not has_match:
            stale.append(entry)
    return tuple(sorted(stale, key=lambda e: e.finding_id))


def evaluate_public_contract_dependency_boundary(
    repo_root: Path,
    *,
    debt_entries: tuple[ContractDependencyDebtEntry, ...] | None = None,
    supplemental_surfaces: tuple[SupplementalPublicContractSurface, ...] | None = None,
) -> PublicContractBoundaryGateResult:
    registry = debt_entries if debt_entries is not None else PUBLIC_CONTRACT_DEPENDENCY_DEBT
    surfaces = (
        SUPPLEMENTAL_PUBLIC_CONTRACT_SURFACES
        if supplemental_surfaces is None
        else supplemental_surfaces
    )
    registry_errors = list(validate_debt_registry(registry))
    registry_errors.extend(
        validate_supplemental_public_contract_surfaces(
            repo_root,
            supplemental_surfaces=surfaces,
        ),
    )
    if registry_errors:
        return PublicContractBoundaryGateResult(
            unregistered_violations=(),
            stale_debt_entries=(),
            registry_validation_errors=tuple(registry_errors),
        )
    violations = _collect_violations(repo_root, supplemental_surfaces=supplemental_surfaces)
    unregistered: list[ContractDependencyViolation] = []
    for violation in violations:
        if any(_debt_covers_violation(violation, entry) for entry in registry):
            continue
        unregistered.append(violation)
    stale = _find_stale_debt(violations, registry)
    return PublicContractBoundaryGateResult(
        unregistered_violations=tuple(unregistered),
        stale_debt_entries=stale,
        registry_validation_errors=(),
    )


def format_gate_failure(result: PublicContractBoundaryGateResult) -> str:
    lines: list[str] = []
    if result.registry_validation_errors:
        lines.append("Public contract boundary registry validation errors:")
        for error in result.registry_validation_errors:
            lines.append(f"  - {error}")
    if result.unregistered_violations:
        lines.append("Unregistered public contract dependency violations:")
        for violation in result.unregistered_violations:
            lines.append(f"  - {violation.as_message()}")
    if result.stale_debt_entries:
        lines.append("Stale debt registry entries (violation no longer present):")
        for entry in result.stale_debt_entries:
            lines.append(
                f"  - {entry.finding_id}: {entry.source_module} "
                f"import={entry.forbidden_import_module!r} "
                f"rule={entry.rule_id.value} stage={entry.removal_stage.value}",
            )
    return "\n".join(lines)
