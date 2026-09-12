# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ERL plugin SPI — capability strategies behind stable platform ports (foundation)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyLifecyclePhase,
    UncertaintyResolutionKind,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import ExternalEffectEvidence
from intergrax.contracts.enterprise_reliability.compensation_decision import CompensationDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.contracts.enterprise_reliability.reconciliation_execution import (
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)

if TYPE_CHECKING:
    from intergrax.contracts.enterprise_reliability.compensation_execution import (
        CompensationExecutionRequest,
        CompensationPluginExecutionResult,
    )


class EnterpriseReliabilityCapabilityKind(StrEnum):
    """Replaceable ERL behavior families — one plugin per capability kind + plugin_id."""

    RECONCILIATION = "reconciliation"
    RESOLUTION = "resolution"
    COMPENSATION = "compensation"
    RISK_EVALUATION = "risk_evaluation"


@dataclass(frozen=True, slots=True)
class EnterpriseReliabilityPluginDescriptor:
    """Production plugin identity — required for registry admission."""

    plugin_id: str
    version: str
    owner: str
    capability_kind: EnterpriseReliabilityCapabilityKind
    capabilities: tuple[str, ...]
    tenant_scope: frozenset[str] | None
    priority: int

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("plugin_id required")
        if not self.version.strip():
            raise ValueError("version required")
        if not self.owner.strip():
            raise ValueError("owner required")
        if not self.capabilities:
            raise ValueError("capabilities must be non-empty")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")


@dataclass(frozen=True, slots=True)
class EnterpriseReliabilityStrategyContext:
    """Readonly platform context for strategy evaluation — no provider payloads."""

    tenant_id: str
    correlation_id: str
    contract_id: str
    effect_outcome: ExternalEffectOutcome
    lifecycle_phase: UncertaintyLifecyclePhase
    evidence_verdict: ExternalEffectEvidenceVerdict | None = None
    evidence_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.correlation_id.strip():
            raise ValueError("correlation_id required")
        if not self.contract_id.strip():
            raise ValueError("contract_id required")


@dataclass(frozen=True, slots=True)
class ReconciliationStrategyAdvice:
    """Non-binding reconcile plan hook — orchestration applies bounds in later phases."""

    probe_ref: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.probe_ref.strip():
            raise ValueError("probe_ref required")


@dataclass(frozen=True, slots=True)
class ResolutionStrategyEvaluationRequest:
    """Inputs for resolution strategy evaluation — evidence-bound, contract-scoped."""

    evidence: ExternalEffectEvidence
    execution_context: EnterpriseReliabilityStrategyContext
    effect_contract: ExternalEffectContract


@dataclass(frozen=True, slots=True)
class CompensationStrategyEvaluationRequest:
    """Inputs for compensation strategy evaluation — resolution-bound, evidence-scoped."""

    resolution_decision: ResolutionDecision
    evidence: ExternalEffectEvidence
    execution_context: EnterpriseReliabilityStrategyContext
    effect_contract: ExternalEffectContract


@dataclass(frozen=True, slots=True)
class ResolutionStrategyAdvice:
    """Lifecycle closure path derived from a platform ``ResolutionDecision``."""

    resolution_kind: UncertaintyResolutionKind
    rationale: str = ""
    platform_action: ResolutionDecision | None = None


class EnterpriseReliabilityRiskLevel(StrEnum):
    LOW = "low"
    ELEVATED = "elevated"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class RiskEvaluationStrategyAdvice:
    """Risk posture for governance gating — does not authorize execution."""

    risk_level: EnterpriseReliabilityRiskLevel
    requires_human_review: bool
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class CompensationStrategyAdvice:
    """Compensation intent — Reliability queue executes in later phases."""

    compensation_operation_ref: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.compensation_operation_ref.strip():
            raise ValueError("compensation_operation_ref required")


def assert_plugin_identity_matches_descriptor(
    plugin_id: str,
    version: str,
    descriptor: EnterpriseReliabilityPluginDescriptor,
) -> None:
    if descriptor.plugin_id != plugin_id:
        raise ValueError("descriptor.plugin_id mismatch")
    if descriptor.version != version:
        raise ValueError("descriptor.version mismatch")


@runtime_checkable
class ReconciliationStrategy(Protocol):
    """Plugin contract — reconcile planning only; no I/O or mutation."""

    @property
    def plugin_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor: ...

    def evaluate(
        self,
        context: EnterpriseReliabilityStrategyContext,
    ) -> ReconciliationStrategyAdvice | None:
        """Return advice when this strategy applies; ``None`` when it abstains."""


@runtime_checkable
class ReconciliationProbeExecutor(Protocol):
    """Plugin contract — read-only probe I/O in integrations; returns platform evidence."""

    @property
    def plugin_id(self) -> str: ...

    def execute_probe(
        self,
        request: ReconciliationProbeRequest,
    ) -> ReconciliationProbeResult:
        """Perform provider read for ``request.probe_ref`` — no mutations."""


@runtime_checkable
class ResolutionStrategy(Protocol):
    """Plugin contract — resolution recommendation only."""

    @property
    def plugin_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor: ...

    def evaluate(
        self,
        request: ResolutionStrategyEvaluationRequest,
    ) -> ResolutionDecision | None: ...


@runtime_checkable
class CompensationStrategy(Protocol):
    """Plugin contract — compensation recommendation only."""

    @property
    def plugin_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor: ...

    def evaluate(
        self,
        request: CompensationStrategyEvaluationRequest,
    ) -> CompensationDecision | None: ...


@runtime_checkable
class CompensationExecutionStrategy(Protocol):
    """Plugin contract — external compensation I/O; returns platform-neutral outcomes."""

    @property
    def plugin_id(self) -> str: ...

    def execute_compensation(
        self,
        request: CompensationExecutionRequest,
    ) -> CompensationPluginExecutionResult:
        """Perform provider compensation for ``request.compensation_operation_ref``."""


@runtime_checkable
class RiskEvaluationStrategy(Protocol):
    """Plugin contract — risk evaluation only."""

    @property
    def plugin_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor: ...

    def evaluate(
        self,
        context: EnterpriseReliabilityStrategyContext,
    ) -> RiskEvaluationStrategyAdvice | None: ...


EnterpriseReliabilityPlugin = (
    ReconciliationStrategy
    | ResolutionStrategy
    | CompensationStrategy
    | RiskEvaluationStrategy
)


@runtime_checkable
class EnterpriseReliabilityPluginRegistry(Protocol):
    """Port — platform and applications depend on this, not concrete registries."""

    def register(self, plugin: EnterpriseReliabilityPlugin) -> None:
        """Register or replace a plugin by ``(capability_kind, plugin_id)``."""

    def resolve_reconciliation(self, plugin_id: str) -> ReconciliationStrategy | None: ...

    def resolve_reconciliation_probe(
        self,
        plugin_id: str,
    ) -> ReconciliationProbeExecutor | None: ...

    def resolve_resolution(self, plugin_id: str) -> ResolutionStrategy | None: ...

    def resolve_compensation(self, plugin_id: str) -> CompensationStrategy | None: ...

    def resolve_compensation_executor(
        self,
        plugin_id: str,
    ) -> CompensationExecutionStrategy | None: ...

    def resolve_risk_evaluation(self, plugin_id: str) -> RiskEvaluationStrategy | None: ...

    def list_by_capability(
        self,
        capability_kind: EnterpriseReliabilityCapabilityKind,
        *,
        tenant_id: str | None = None,
    ) -> tuple[EnterpriseReliabilityPlugin, ...]: ...


@runtime_checkable
class EnterpriseReliabilityPluginGateway(Protocol):
    """
    Invocation port — all ERL strategy behavior must be reached through this abstraction.
    """

    def evaluate_reconciliation(
        self,
        plugin_id: str,
        context: EnterpriseReliabilityStrategyContext,
    ) -> ReconciliationStrategyAdvice | None: ...

    def execute_reconciliation_probe(
        self,
        plugin_id: str,
        request: ReconciliationProbeRequest,
    ) -> ReconciliationProbeResult | None: ...

    def resolution_strategy_registered(self, plugin_id: str) -> bool: ...

    def evaluate_resolution(
        self,
        plugin_id: str,
        request: ResolutionStrategyEvaluationRequest,
    ) -> ResolutionDecision | None: ...

    def compensation_strategy_registered(self, plugin_id: str) -> bool: ...

    def evaluate_compensation(
        self,
        plugin_id: str,
        request: CompensationStrategyEvaluationRequest,
    ) -> CompensationDecision | None: ...

    def compensation_executor_registered(self, plugin_id: str) -> bool: ...

    def execute_compensation(
        self,
        plugin_id: str,
        request: CompensationExecutionRequest,
    ) -> CompensationPluginExecutionResult | None: ...

    def evaluate_risk(
        self,
        plugin_id: str,
        context: EnterpriseReliabilityStrategyContext,
    ) -> RiskEvaluationStrategyAdvice | None: ...


__all__ = [
    "CompensationDecision",
    "CompensationExecutionStrategy",
    "CompensationStrategy",
    "CompensationStrategyAdvice",
    "CompensationStrategyEvaluationRequest",
    "EnterpriseReliabilityCapabilityKind",
    "EnterpriseReliabilityPlugin",
    "EnterpriseReliabilityPluginDescriptor",
    "EnterpriseReliabilityPluginGateway",
    "EnterpriseReliabilityPluginRegistry",
    "EnterpriseReliabilityRiskLevel",
    "EnterpriseReliabilityStrategyContext",
    "ReconciliationProbeExecutor",
    "ReconciliationStrategy",
    "ReconciliationStrategyAdvice",
    "ResolutionDecision",
    "ResolutionStrategy",
    "ResolutionStrategyAdvice",
    "ResolutionStrategyEvaluationRequest",
    "RiskEvaluationStrategy",
    "RiskEvaluationStrategyAdvice",
    "assert_plugin_identity_matches_descriptor",
]
