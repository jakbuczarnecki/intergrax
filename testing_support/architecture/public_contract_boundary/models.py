# © Artur Czarnecki. All rights reserved.

"""Typed models for EBH-2A public contract dependency boundary gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from testing_support.architecture.public_contract_boundary.contract_surface_purity import (
        ContractSurfacePurityViolation,
    )


class DependencyRuleId(str, Enum):
    FORBIDDEN_RUNTIME_NAMESPACE = "forbidden_runtime_namespace"
    FORBIDDEN_REGISTRY_NAMESPACE = "forbidden_registry_namespace"
    FORBIDDEN_BOOTSTRAP_NAMESPACE = "forbidden_bootstrap_namespace"
    FOREIGN_DOMAIN_IMPLEMENTATION = "foreign_domain_implementation"


class RemovalStage(str, Enum):
    EBH_2B = "EBH-2B"
    EBH_2C = "EBH-2C"
    EBH_2D = "EBH-2D"
    EBH_2E = "EBH-2E"
    EBH_2F = "EBH-2F"
    EBH_2G = "EBH-2G"
    EBH_2H = "EBH-2H"
    EBH_2I = "EBH-2I"
    EBH_3 = "EBH-3"
    QUAL_X = "QUAL-X"


@dataclass(frozen=True, slots=True)
class ContractSurfacePurityDebtEntry:
    finding_id: str
    source_path: str
    line: int
    rule_id: str
    removal_stage: RemovalStage
    rationale: str


@dataclass(frozen=True, slots=True)
class ContractSurfacePurityGateResult:
    unregistered_violations: tuple[ContractSurfacePurityViolation, ...]
    stale_debt_entries: tuple[ContractSurfacePurityDebtEntry, ...]
    expired_debt_entries: tuple[ContractSurfacePurityDebtEntry, ...]
    registry_validation_errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            not self.unregistered_violations
            and not self.stale_debt_entries
            and not self.expired_debt_entries
            and not self.registry_validation_errors
        )


@dataclass(frozen=True, slots=True)
class ContractDependencyDebtEntry:
    finding_id: str
    source_module: str
    forbidden_import_module: str
    rule_id: DependencyRuleId
    removal_stage: RemovalStage


@dataclass(frozen=True, slots=True)
class ContractDependencyViolation:
    source_module: str
    imported_module: str
    rule_id: DependencyRuleId
    source_path: str
    line: int

    def sort_key(self) -> tuple[str, str, str]:
        return (self.source_module, self.imported_module, self.rule_id.value)

    def as_message(self) -> str:
        return (
            f"{self.source_path}:{self.line}: {self.source_module} "
            f"imports forbidden dependency {self.imported_module!r} "
            f"(rule={self.rule_id.value})"
        )


@dataclass(frozen=True, slots=True)
class PublicContractBoundaryGateResult:
    unregistered_violations: tuple[ContractDependencyViolation, ...]
    stale_debt_entries: tuple[ContractDependencyDebtEntry, ...]
    registry_validation_errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            not self.unregistered_violations
            and not self.stale_debt_entries
            and not self.registry_validation_errors
        )
