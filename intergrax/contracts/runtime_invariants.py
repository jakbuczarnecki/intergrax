# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Runtime invariant evaluation contracts (RI-01 foundation).

Platform-owned evaluation mechanics only — invariant meaning stays domain-owned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from intergrax.contracts.execution_identity import ExecutionId


class RuntimeInvariantCompositionError(ValueError):
    """Fail-closed composition: duplicate pack or rule identities."""


class RuntimeInvariantDomain(StrEnum):
    """Stable domain partition for rule packs."""

    EXECUTION = "execution"
    DELEGATED_PROVIDER = "delegated_provider"
    GOVERNANCE = "governance"


class RuntimeInvariantSeverity(StrEnum):
    """Consumer-facing severity hint — not a policy decision."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuntimeInvariantStatus(StrEnum):
    """Per-rule evaluation outcome."""

    PASS = "pass"
    VIOLATION = "violation"
    NOT_APPLICABLE = "not_applicable"
    EVALUATION_ERROR = "evaluation_error"


class RuntimeInvariantOverallStatus(StrEnum):
    """Aggregated conformance — not enforcement."""

    CONFORMANT = "conformant"
    NON_CONFORMANT = "non_conformant"
    INDETERMINATE = "indeterminate"


class RuntimeInvariantEvaluationMode(StrEnum):
    """Invocation mode metadata for consumers."""

    QUALIFICATION = "qualification"
    DIAGNOSTIC = "diagnostic"
    AD_HOC = "ad_hoc"


RuntimeInvariantId = str
RuntimeInvariantEvaluationId = str
RuntimeInvariantPackId = str


@dataclass(frozen=True, slots=True)
class RuntimeInvariantEvaluationContext:
    """Minimal shared metadata — domain evidence is probe-owned."""

    evaluation_id: RuntimeInvariantEvaluationId
    correlation_id: str
    requested_at: datetime
    mode: RuntimeInvariantEvaluationMode = RuntimeInvariantEvaluationMode.AD_HOC
    execution_id: ExecutionId | None = None


@dataclass(frozen=True, slots=True)
class RuntimeInvariantSelection:
    """Typed subset selection — ``None`` means no filter on that axis."""

    domains: frozenset[RuntimeInvariantDomain] | None = None
    rule_ids: frozenset[RuntimeInvariantId] | None = None

    @staticmethod
    def all() -> RuntimeInvariantSelection:
        return RuntimeInvariantSelection()

    def includes_rule(
        self,
        *,
        domain: RuntimeInvariantDomain,
        rule_id: RuntimeInvariantId,
    ) -> bool:
        if self.domains is not None and domain not in self.domains:
            return False
        if self.rule_ids is not None and rule_id not in self.rule_ids:
            return False
        return True


@dataclass(frozen=True, slots=True)
class RuntimeInvariantEvaluationRequest:
    """Immutable evaluation invocation."""

    selection: RuntimeInvariantSelection = field(default_factory=RuntimeInvariantSelection.all)
    correlation_id: str | None = None
    mode: RuntimeInvariantEvaluationMode = RuntimeInvariantEvaluationMode.AD_HOC
    execution_id: ExecutionId | None = None


@dataclass(frozen=True, slots=True)
class RuntimeInvariantResult:
    """Immutable per-rule outcome."""

    rule_id: RuntimeInvariantId
    domain: RuntimeInvariantDomain
    rule_version: str
    severity: RuntimeInvariantSeverity
    status: RuntimeInvariantStatus
    summary: str
    evaluation_id: RuntimeInvariantEvaluationId
    correlation_id: str
    diagnostic_code: str | None = None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeInvariantReportSummary:
    """Bounded count rollup."""

    pass_count: int
    violation_count: int
    not_applicable_count: int
    evaluation_error_count: int


@dataclass(frozen=True, slots=True)
class RuntimeInvariantReport:
    """Deterministic, immutable evaluation report."""

    evaluation_id: RuntimeInvariantEvaluationId
    correlation_id: str
    evaluated_at: datetime
    mode: RuntimeInvariantEvaluationMode
    pack_ids: tuple[RuntimeInvariantPackId, ...]
    pack_versions: tuple[str, ...]
    results: tuple[RuntimeInvariantResult, ...]
    summary: RuntimeInvariantReportSummary
    overall_status: RuntimeInvariantOverallStatus
    execution_id: ExecutionId | None = None


class RuntimeInvariantRule(Protocol):
    """Domain-owned rule — runner invokes ``evaluate`` only."""

    @property
    def rule_id(self) -> RuntimeInvariantId: ...

    @property
    def domain(self) -> RuntimeInvariantDomain: ...

    @property
    def rule_version(self) -> str: ...

    @property
    def severity(self) -> RuntimeInvariantSeverity: ...

    def evaluate(
        self,
        context: RuntimeInvariantEvaluationContext,
    ) -> RuntimeInvariantResult:
        """Read-only evaluation against domain-injected facts."""
        ...


class RuntimeInvariantRulePack(Protocol):
    """Immutable rule bundle — does not execute rules."""

    @property
    def pack_id(self) -> RuntimeInvariantPackId: ...

    @property
    def pack_version(self) -> str: ...

    @property
    def domain(self) -> RuntimeInvariantDomain: ...

    @property
    def rules(self) -> tuple[RuntimeInvariantRule, ...]: ...


class RuntimeInvariantEvaluationClock(Protocol):
    """Injected time source for deterministic tests."""

    def now(self) -> datetime:
        """Timezone-aware instant."""
        ...


class RuntimeInvariantEvaluationIdFactory(Protocol):
    """Issues opaque evaluation identities."""

    def mint_evaluation_id(self) -> RuntimeInvariantEvaluationId:
        ...


class RuntimeInvariantRunner(Protocol):
    """Executes selected rules from composed packs."""

    def evaluate(
        self,
        request: RuntimeInvariantEvaluationRequest,
    ) -> RuntimeInvariantReport:
        ...


def runtime_invariant_rule_sort_key(rule: RuntimeInvariantRule) -> tuple[str, str, str]:
    """Documented deterministic ordering: domain, rule_id, rule_version."""
    return (rule.domain.value, rule.rule_id, rule.rule_version)


def aggregate_runtime_invariant_overall_status(
    results: tuple[RuntimeInvariantResult, ...],
) -> RuntimeInvariantOverallStatus:
    """Conformance rollup — not policy."""
    if any(r.status is RuntimeInvariantStatus.VIOLATION for r in results):
        return RuntimeInvariantOverallStatus.NON_CONFORMANT
    if any(r.status is RuntimeInvariantStatus.EVALUATION_ERROR for r in results):
        return RuntimeInvariantOverallStatus.INDETERMINATE
    return RuntimeInvariantOverallStatus.CONFORMANT


def summarize_runtime_invariant_results(
    results: tuple[RuntimeInvariantResult, ...],
) -> RuntimeInvariantReportSummary:
    pass_count = 0
    violation_count = 0
    not_applicable_count = 0
    evaluation_error_count = 0
    for result in results:
        if result.status is RuntimeInvariantStatus.PASS:
            pass_count += 1
        elif result.status is RuntimeInvariantStatus.VIOLATION:
            violation_count += 1
        elif result.status is RuntimeInvariantStatus.NOT_APPLICABLE:
            not_applicable_count += 1
        elif result.status is RuntimeInvariantStatus.EVALUATION_ERROR:
            evaluation_error_count += 1
    return RuntimeInvariantReportSummary(
        pass_count=pass_count,
        violation_count=violation_count,
        not_applicable_count=not_applicable_count,
        evaluation_error_count=evaluation_error_count,
    )


__all__ = [
    "RuntimeInvariantCompositionError",
    "RuntimeInvariantDomain",
    "RuntimeInvariantEvaluationClock",
    "RuntimeInvariantEvaluationContext",
    "RuntimeInvariantEvaluationId",
    "RuntimeInvariantEvaluationIdFactory",
    "RuntimeInvariantEvaluationMode",
    "RuntimeInvariantEvaluationRequest",
    "RuntimeInvariantId",
    "RuntimeInvariantOverallStatus",
    "RuntimeInvariantPackId",
    "RuntimeInvariantReport",
    "RuntimeInvariantReportSummary",
    "RuntimeInvariantResult",
    "RuntimeInvariantRule",
    "RuntimeInvariantRulePack",
    "RuntimeInvariantRunner",
    "RuntimeInvariantSelection",
    "RuntimeInvariantSeverity",
    "RuntimeInvariantStatus",
    "aggregate_runtime_invariant_overall_status",
    "runtime_invariant_rule_sort_key",
    "summarize_runtime_invariant_results",
]
