# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Validation registry → validators → aggregation (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.execution.context import SelfHealingExecutionContext
from intergrax.contracts.self_healing.observation.provider import ObservationResult
from intergrax.contracts.self_healing.validation.decision import (
    SelfHealingValidationDecision,
    ValidationDecisionStatus,
)
from intergrax.contracts.self_healing.validation.validator import (
    SelfHealingValidator,
    ValidatorCheckStatus,
)
from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
from intergrax.contracts.self_healing.workflow.errors import PLUGIN_FAILED, SelfHealingWorkflowPluginFailedError
from intergrax.contracts.self_healing.workflow.registry import SelfHealingWorkflowPluginDescriptor
from intergrax.runtime.self_healing.workflow.registries import InMemorySelfHealingValidationRegistry


@dataclass
class SelfHealingValidationPipeline:
    validation_registry: InMemorySelfHealingValidationRegistry
    validators: tuple[SelfHealingValidator, ...] = ()
    validator_descriptors: dict[str, SelfHealingWorkflowPluginDescriptor] = field(default_factory=dict)

    def register_validator(
        self,
        validator: SelfHealingValidator,
        descriptor: SelfHealingWorkflowPluginDescriptor,
    ) -> None:
        if descriptor.plugin_id != validator.validator_id:
            raise ValueError("descriptor.plugin_id must match validator.validator_id")
        self.validator_descriptors[validator.validator_id] = descriptor
        self.validators = (*self.validators, validator)

    def evaluate(
        self,
        workflow_context: SelfHealingWorkflowContext,
        *,
        execution_context: SelfHealingExecutionContext,
        observation: ObservationResult | None,
    ) -> SelfHealingValidationDecision:
        if observation is None or not observation.evidence_refs:
            raise ValueError("validation requires observation evidence")

        passed: list[str] = []
        failed: list[str] = []
        evidence: set[str] = set(observation.evidence_refs)
        explanations: list[str] = []
        min_confidence = 1.0

        for validator in self.validators:
            descriptor = self.validator_descriptors.get(validator.validator_id)
            if descriptor is not None and descriptor.timeout_seconds <= 0:
                continue
            try:
                check = validator.validate(
                    workflow_context,
                    execution_context=execution_context,
                    observation=observation,
                )
            except Exception as exc:  # noqa: BLE001
                raise SelfHealingWorkflowPluginFailedError(
                    f"{PLUGIN_FAILED}: validator {validator.validator_id} failed: {exc}",
                ) from exc
            evidence.update(check.evidence_refs)
            if check.status is ValidatorCheckStatus.PASSED:
                passed.append(check.check_id)
            elif check.status is ValidatorCheckStatus.FAILED:
                failed.append(check.check_id)
                explanations.append(check.detail)
            else:
                explanations.append(f"{check.check_id}: inconclusive")

        provider = self.validation_registry.resolve(
            workflow_context.plan.validation_policy_id,
            tenant_id=workflow_context.tenant_id,
        )
        if provider is not None:
            try:
                legacy = provider.validate(workflow_context)
            except Exception as exc:  # noqa: BLE001
                raise SelfHealingWorkflowPluginFailedError(
                    f"{PLUGIN_FAILED}: validation provider failed: {exc}",
                ) from exc
            evidence.update(legacy.evidence_refs)
            min_confidence = min(min_confidence, legacy.confidence)
            if legacy.status.value == "PASSED":
                passed.append(provider.provider_id)
            else:
                failed.append(provider.provider_id)
                explanations.append(legacy.explanation)

        if failed:
            status = ValidationDecisionStatus.FAILED
            confidence = min(min_confidence, observation.confidence)
        elif passed:
            status = ValidationDecisionStatus.PASSED
            confidence = observation.confidence
        else:
            status = ValidationDecisionStatus.INCONCLUSIVE
            confidence = observation.confidence * 0.5

        explanation = "; ".join(explanations) if explanations else "aggregated validation"
        return SelfHealingValidationDecision(
            status=status,
            confidence=confidence,
            passed_checks=tuple(passed),
            failed_checks=tuple(failed),
            evidence_refs=tuple(sorted(evidence)),
            explanation=explanation,
        )


__all__ = ["SelfHealingValidationPipeline"]
