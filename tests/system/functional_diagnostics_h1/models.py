# © Artur Czarnecki. All rights reserved.

"""Typed models for DIAG-FUNCTIONAL-H1 test-suite health qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DiagnosticTestLayer(StrEnum):
    UNIT = "UNIT"
    CONFORMANCE = "CONFORMANCE"
    INTEGRATION = "INTEGRATION"
    SYSTEM = "SYSTEM"
    REAL_SERVICE_QUALIFICATION = "REAL_SERVICE_QUALIFICATION"
    STATIC_ARCHITECTURE = "STATIC_ARCHITECTURE"
    PERFORMANCE_STRUCTURAL = "PERFORMANCE_STRUCTURAL"
    RECOVERY = "RECOVERY"


class DeterminismClass(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    EXTERNAL_DEPENDENT = "EXTERNAL_DEPENDENT"
    STRUCTURAL_PROBE = "STRUCTURAL_PROBE"


class ExpectedOutcome(StrEnum):
    PASS = "PASS"
    BLOCKED_WHEN_UNAVAILABLE = "BLOCKED_WHEN_UNAVAILABLE"
    CONDITIONAL = "CONDITIONAL"


class QualificationFamily(StrEnum):
    CORE = "CORE"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    Q5 = "Q5"
    D1 = "D1"
    S1 = "S1"
    R1 = "R1"
    R1_R1 = "R1_R1"
    R1_R2 = "R1_R2"
    R1_R3 = "R1_R3"
    H1 = "H1"


class ExternalDependencyState(StrEnum):
    READY = "READY"
    BLOCKED_MISSING_CREDENTIAL = "BLOCKED_MISSING_CREDENTIAL"
    BLOCKED_SERVICE_UNAVAILABLE = "BLOCKED_SERVICE_UNAVAILABLE"
    FAILED_PREFLIGHT = "FAILED_PREFLIGHT"
    NOT_EXECUTED = "NOT_EXECUTED"


class SkipClassification(StrEnum):
    JUSTIFIED = "JUSTIFIED"
    STALE = "STALE"
    MASKING_FAILURE = "MASKING_FAILURE"
    NONE = "NONE"


class HealthVerdict(StrEnum):
    PASS = "PASS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"


class HealthGateId(StrEnum):
    H1_A_COLLECTION = "H1-A"
    H1_B_CORE_HEALTH = "H1-B"
    H1_C_REPEATABILITY = "H1-C"
    H1_D_INVARIANT_COVERAGE = "H1-D"
    H1_E_SKIP_XFAIL_HONESTY = "H1-E"
    H1_F_EXTERNAL_DEPENDENCY = "H1-F"
    H1_G_RUNNER_INTEGRITY = "H1-G"
    H1_H_STALE_DEAD = "H1-H"
    H1_I_SUPERSESSION = "H1-I"
    H1_J_REPORT_INTEGRITY = "H1-J"
    H1_K_LOCAL_INTEGRATION = "H1-K"


class HealthDimension(StrEnum):
    DISCOVERABILITY = "DISCOVERABILITY"
    EXECUTABILITY = "EXECUTABILITY"
    DETERMINISM = "DETERMINISM"
    COVERAGE_OF_INVARIANTS = "COVERAGE_OF_INVARIANTS"
    FAILURE_HONESTY = "FAILURE_HONESTY"
    EXTERNAL_DEPENDENCY_HANDLING = "EXTERNAL_DEPENDENCY_HANDLING"
    ARCHITECTURE_PROTECTION = "ARCHITECTURE_PROTECTION"
    REGRESSION_COHERENCE = "REGRESSION_COHERENCE"
    QUALIFICATION_TRACEABILITY = "QUALIFICATION_TRACEABILITY"
    DEAD_STALE_TEST_DETECTION = "DEAD_STALE_TEST_DETECTION"


@dataclass(frozen=True, slots=True)
class DiagnosticTestDescriptor:
    id: str
    path: str
    layer: DiagnosticTestLayer
    domain: str
    requires_external_service: bool
    required_services: tuple[str, ...]
    qualification_family: QualificationFamily
    determinism_class: DeterminismClass
    expected_outcome: ExpectedOutcome


@dataclass(frozen=True, slots=True)
class InvariantOwner:
    invariant_id: str
    description: str
    unit_owner: str | None
    conformance_owner: str | None
    system_real_owner: str | None
    normative_owner: str


@dataclass(frozen=True, slots=True)
class QualificationRepositoryState:
    head_sha: str
    origin_development_sha: str
    working_tree_clean: bool


@dataclass(frozen=True, slots=True)
class QualificationRepositoryTransition:
    start: QualificationRepositoryState
    end: QualificationRepositoryState


@dataclass(frozen=True, slots=True)
class PytestSubprocessResult:
    exit_code: int
    collected_count: int | None
    passed: int
    failed: int
    skipped: int
    xfailed: int
    xpassed: int
    errors: int
    collection_errors: int
    stdout_tail: str
    stderr_tail: str
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: HealthGateId
    verdict: HealthVerdict
    summary: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkipXfailFinding:
    path: str
    line: int
    marker: str
    classification: SkipClassification
    note: str


@dataclass(frozen=True, slots=True)
class QualificationRunnerDescriptor:
    family: QualificationFamily
    runner_path: str
    doc_path: str | None
    powershell_path: str | None


@dataclass(frozen=True, slots=True)
class DiagnosticQualificationDependencyStatus:
    family: QualificationFamily
    state: ExternalDependencyState
    required_services: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class DiagnosticTestHealthStatus:
    gate_id: HealthGateId
    verdict: HealthVerdict
    dimension: HealthDimension
    message: str


class LocalIntegrationDependencyClass(StrEnum):
    LOCAL_DETERMINISTIC = "LOCAL_DETERMINISTIC"


@dataclass(frozen=True, slots=True)
class LocalIntegrationSuiteResult:
    target: str
    collected: int | None
    passed: int
    failed: int
    skipped: int
    xfailed: int
    xpassed: int
    errors: int
    collection_errors: int
    exit_code: int
    duration_seconds: float
    verdict: HealthVerdict
    dependency_class: LocalIntegrationDependencyClass


@dataclass(frozen=True, slots=True)
class LocalIntegrationRunResult:
    run_index: int
    suite_results: tuple[LocalIntegrationSuiteResult, ...]
    verdict: HealthVerdict
    total_collected: int | None
    total_passed: int
    total_failed: int
    total_errors: int


@dataclass(frozen=True, slots=True)
class LocalIntegrationQualificationReport:
    qualification_id: str
    schema_version: str
    tested_sha: str
    start_head: str
    final_head: str
    origin_development_at_start: str
    origin_development_at_end: str
    working_tree_clean_at_start: bool
    working_tree_clean_at_end: bool
    repository_precondition: HealthVerdict
    repository_postcondition: HealthVerdict
    runs: tuple[LocalIntegrationRunResult, ...]
    repeatability_verdict: HealthVerdict
    overall_verdict: HealthVerdict
    blocking_findings: tuple[str, ...]
    timestamp: str


@dataclass(frozen=True, slots=True)
class DiagnosticTestSuiteResult:
    scope: str
    collected: int | None
    passed: int
    failed: int
    skipped: int
    xfailed: int
    verdict: HealthVerdict


@dataclass(frozen=True, slots=True)
class DiagnosticHealthReport:
    schema_version: str
    qualification_id: str
    tested_sha: str
    start_head: str
    final_head: str
    origin_development_sha: str
    origin_development_at_end: str
    working_tree_clean_at_start: bool
    working_tree_clean_at_end: bool
    repository_precondition: HealthVerdict
    repository_postcondition: HealthVerdict
    timestamp: str
    h1_semantics: str
    inventory_counts: dict[str, int]
    collection_result: GateResult
    static_results: GateResult
    unit_results: GateResult
    repeatability_results: tuple[DiagnosticTestSuiteResult, ...]
    local_system_results: GateResult
    external_preflight_results: tuple[DiagnosticQualificationDependencyStatus, ...]
    skip_xfail_inventory: tuple[SkipXfailFinding, ...]
    invariant_coverage: tuple[InvariantOwner, ...]
    dead_stale_findings: tuple[str, ...]
    gate_results: tuple[GateResult, ...]
    core_test_health: HealthVerdict
    real_service_qualification_availability: HealthVerdict
    overall_h1: HealthVerdict
    blocking_findings: tuple[str, ...]
    warnings: tuple[str, ...]


H1_SCHEMA_VERSION = "diag_functional_h1_v1"
H1_QUALIFICATION_ID = "DIAG-FUNCTIONAL-H1"
H1_R1_QUALIFICATION_ID = "DIAG-FUNCTIONAL-H1-R1"
H1_R2_QUALIFICATION_ID = "DIAG-FUNCTIONAL-H1-R2"
H1_R3_QUALIFICATION_ID = "DIAG-FUNCTIONAL-H1-R3"
H1_K_QUALIFICATION_ID = "DIAG-H1-K-QUALIFICATION-R1"
H1_K_SCHEMA_VERSION = "diag_h1_k_local_integration_v1"
H1_SEMANTICS = (
    "H1 measures diagnostic TEST-SUITE HEALTH, not live requalification of all "
    "historical real-world qualifications. External service absence yields "
    "REAL-SERVICE REQUALIFICATION = NOT REVALIDATED / BLOCKED BY ENVIRONMENT "
    "without blocking core H1 PASS when runner/preflight classification is honest."
)
