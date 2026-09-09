# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Structured qualification observation signals (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.decision_system.qualification.taxonomy import DecisionFailureBoundary


@dataclass(frozen=True, slots=True)
class EnvironmentQualificationSignal:
    credential_unavailable: bool = False
    provider_configuration_invalid: bool = False
    model_configuration_invalid: bool = False
    qualification_disabled: bool = False


@dataclass(frozen=True, slots=True)
class ProviderQualificationSignal:
    rate_limit: bool = False
    timeout: bool = False
    network_failure: bool = False
    server_error: bool = False
    protocol_error: bool = False


@dataclass(frozen=True, slots=True)
class PlatformContractQualificationSignal:
    wrong_execution_route: bool = False
    trace_finalized: bool = True
    strict_tool_contract_violation: bool = False
    tool_dispatch_contract_violation: bool = False
    invalid_phase_transition: bool = False
    completion_reconciliation_contract_violation: bool = False
    terminal_acceptance_contract_violation: bool = False
    violation_boundary: DecisionFailureBoundary | None = None


@dataclass(frozen=True, slots=True)
class ModelBehaviorQualificationSignal:
    insufficient_evidence_gathering: bool = False
    tool_use_deficiency: bool = False
    epistemic_contradiction: bool = False
    unsupported_completion: bool = False
    premature_completion: bool = False
    behavior_boundary: DecisionFailureBoundary | None = None


@dataclass(frozen=True, slots=True)
class EvaluatorQualificationSignal:
    passed: bool | None = None
    false_negative: bool = False
    false_positive: bool = False
    criterion_semantics_invalid: bool = False
    contract_error: bool = False


@dataclass(frozen=True, slots=True)
class ObservabilityQualificationSignal:
    missing_required_signal: bool = False
    incomplete_trace: bool = False
    ambiguous_failure_boundary: bool = False
    critical_boundary_unknown: bool = False
