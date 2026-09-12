# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing strategy input — readonly platform context only (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelfHealingDiagnosticInvestigation:
    """Readonly diagnostic investigation slice — not a second diagnostic engine."""

    investigation_id: str
    problem_id: str
    tenant_id: str
    evidence_refs: tuple[str, ...] = ()
    capability_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelfHealingPredictiveSignal:
    tenant_id: str
    signal_id: str
    signal_type: str
    severity_label: str = "unknown"


@dataclass(frozen=True, slots=True)
class SelfHealingHistoricalOutcome:
    tenant_id: str
    strategy_id: str
    successful_preventions: int = 0
    failed_actions: int = 0
    rollback_rate: float = 0.0
    false_positive_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class SelfHealingOperationDescriptor:
    """Qualified operation the platform may admit — strategy proposes against this catalog."""

    operation_kind: str
    target_resource: str
    provider_id: str


@dataclass(frozen=True, slots=True)
class SelfHealingPolicyConstraints:
    production_target: bool = False
    max_blast_radius: str = "tenant"
    policy_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelfHealingContext:
    """
    Pure decision input — strategies must not perform I/O or call external systems.
    """

    tenant_id: str
    diagnostic_investigation: SelfHealingDiagnosticInvestigation
    predictive_signals: tuple[SelfHealingPredictiveSignal, ...]
    historical_outcomes: tuple[SelfHealingHistoricalOutcome, ...]
    available_operations: tuple[SelfHealingOperationDescriptor, ...]
    constraints: SelfHealingPolicyConstraints

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        inv = self.diagnostic_investigation
        if inv.tenant_id != self.tenant_id:
            raise ValueError("tenant isolation violation: investigation")
        for signal in self.predictive_signals:
            if signal.tenant_id != self.tenant_id:
                raise ValueError("tenant isolation violation: predictive_signals")
        for outcome in self.historical_outcomes:
            if outcome.tenant_id != self.tenant_id:
                raise ValueError("tenant isolation violation: historical_outcomes")


__all__ = [
    "SelfHealingContext",
    "SelfHealingDiagnosticInvestigation",
    "SelfHealingHistoricalOutcome",
    "SelfHealingOperationDescriptor",
    "SelfHealingPolicyConstraints",
    "SelfHealingPredictiveSignal",
]
