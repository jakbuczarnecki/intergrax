# © Artur Czarnecki. All rights reserved.

"""Typed contracts for global qualification semantic parity certification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from testing_support.execution_qualification.contracts import QualificationSuiteStatus
from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)


class SemanticParityCertificationStatus(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class QualificationSemanticParityCase:
    profile_id: str
    legacy_semantic_source: FrozenPytestSuiteSource
    canonical_profile_id: str
    expected_root_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoverageParityDiagnostics:
    missing_in_canonical: frozenset[tuple[str, ...]]
    unexpected_in_canonical: frozenset[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class LeafInjectionParityRow:
    profile_id: str
    suite_id: str
    affected_root_ids: tuple[str, ...]
    expected_root_status: QualificationSuiteStatus
    actual_root_status: tuple[tuple[str, QualificationSuiteStatus], ...]


@dataclass(frozen=True, slots=True)
class QualificationProfileParityResult:
    profile_id: str
    coverage_parity: bool
    all_pass_parity: bool
    failure_injection_parity: bool
    skip_injection_parity: bool
    collect_all_parity: bool
    receipt_parity: bool
    reachability_parity: bool
    determinism_parity: bool
    gate_semantics_parity: bool
    physical_dedup_parity: bool
    deterministic: bool
    leaf_count: int
    gate_count: int
    coverage_diagnostics: CoverageParityDiagnostics | None = None
    failure_rows: tuple[LeafInjectionParityRow, ...] = ()
    skip_rows: tuple[LeafInjectionParityRow, ...] = ()
    invariant_failures: tuple[str, ...] = ()

    @property
    def profile_pass(self) -> bool:
        return (
            self.coverage_parity
            and self.all_pass_parity
            and self.failure_injection_parity
            and self.skip_injection_parity
            and self.collect_all_parity
            and self.receipt_parity
            and self.reachability_parity
            and self.determinism_parity
            and self.gate_semantics_parity
            and self.physical_dedup_parity
            and self.deterministic
            and not self.invariant_failures
        )


@dataclass(frozen=True, slots=True)
class QualificationSemanticParityReport:
    profile_results: tuple[QualificationProfileParityResult, ...]
    overall_status: SemanticParityCertificationStatus

    @property
    def overall_pass(self) -> bool:
        return self.overall_status is SemanticParityCertificationStatus.PASS
