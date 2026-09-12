# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform evidence-based validation provider (SELF-HEALING R2)."""

from __future__ import annotations

from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
from intergrax.contracts.self_healing.workflow.validation import (
    ValidationResult,
    ValidationStatus,
)


class PlatformEvidenceValidationProvider:
    provider_id = "platform.default.validation"

    def validate(self, workflow_context: SelfHealingWorkflowContext) -> ValidationResult:
        refs = workflow_context.evidence_refs
        if not refs:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                confidence=0.0,
                evidence_refs=("missing_evidence",),
                explanation="validation requires diagnostic evidence refs",
            )
        if workflow_context.failed_step_ids:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                confidence=0.2,
                evidence_refs=refs,
                explanation="one or more healing steps failed",
            )
        return ValidationResult(
            status=ValidationStatus.PASSED,
            confidence=0.85,
            evidence_refs=refs,
            explanation="evidence-backed validation passed",
        )


__all__ = ["PlatformEvidenceValidationProvider"]
