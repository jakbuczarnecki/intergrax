# © Artur Czarnecki. All rights reserved.

"""Technical consistency control providers (DS-E2E-15J-L17)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_governance_framework.contracts import (
    EvolutionGovernanceFrameworkContext,
    EvolutionGovernanceIssue,
    EvolutionGovernanceIssueSeverity,
    EvolutionGovernanceLifecycleStage,
)

_DEFAULT_CONTROL_PROVIDER_ID = "default_evolution_governance_control"
_DEFAULT_CONTROL_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEvolutionGovernanceControlProvider:
    @property
    def provider_id(self) -> str:
        return _DEFAULT_CONTROL_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_CONTROL_PROVIDER_VERSION

    def validate(
        self,
        context: EvolutionGovernanceFrameworkContext,
    ) -> tuple[EvolutionGovernanceIssue, ...]:
        issues: list[EvolutionGovernanceIssue] = []
        if not context.scope_id:
            issues.append(
                EvolutionGovernanceIssue(
                    issue_id="control-missing-scope-id",
                    severity=EvolutionGovernanceIssueSeverity.BLOCKED,
                    issue_code="missing_scope_id",
                    summary="Evolution governance context is missing scope_id.",
                    lifecycle_stage=None,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        if not context.version:
            issues.append(
                EvolutionGovernanceIssue(
                    issue_id="control-missing-scope-version",
                    severity=EvolutionGovernanceIssueSeverity.BLOCKED,
                    issue_code="missing_scope_version",
                    summary="Evolution governance context is missing version.",
                    lifecycle_stage=None,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )
        if context.process_reference is None:
            issues.append(
                EvolutionGovernanceIssue(
                    issue_id="control-missing-process-reference",
                    severity=EvolutionGovernanceIssueSeverity.REVIEW,
                    issue_code="missing_process_reference",
                    summary="Process reference is not attached to governance context.",
                    lifecycle_stage=None,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                )
            )

        for index, execution in enumerate(context.execution_results):
            if execution.version != context.version and context.version:
                issues.append(
                    EvolutionGovernanceIssue(
                        issue_id=f"control-adaptation-version-{index}",
                        severity=EvolutionGovernanceIssueSeverity.REVIEW,
                        issue_code="adaptation_version_mismatch",
                        summary=(
                            f"Adaptation {execution.adaptation_id} version "
                            f"{execution.version} does not match scope version "
                            f"{context.version}."
                        ),
                        lifecycle_stage=EvolutionGovernanceLifecycleStage.ADAPTATION,
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                    )
                )
            source = execution.source_reference
            if not source.proposal_id and not source.controlled_evolution_record_id:
                issues.append(
                    EvolutionGovernanceIssue(
                        issue_id=f"control-adaptation-source-{index}",
                        severity=EvolutionGovernanceIssueSeverity.REVIEW,
                        issue_code="missing_adaptation_source_reference",
                        summary=(
                            f"Adaptation {execution.adaptation_id} lacks proposal "
                            "or controlled evolution source reference."
                        ),
                        lifecycle_stage=EvolutionGovernanceLifecycleStage.ADAPTATION,
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                    )
                )

        for index, record in enumerate(context.operation_records):
            if record.version != context.version and context.version:
                issues.append(
                    EvolutionGovernanceIssue(
                        issue_id=f"control-operation-version-{index}",
                        severity=EvolutionGovernanceIssueSeverity.REVIEW,
                        issue_code="operation_version_mismatch",
                        summary=(
                            f"Operation record {record.record_id} version "
                            f"{record.version} does not match scope version "
                            f"{context.version}."
                        ),
                        lifecycle_stage=EvolutionGovernanceLifecycleStage.OPERATIONS,
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                    )
                )

        return tuple(issues)


def default_evolution_governance_control_providers() -> tuple[
    DefaultEvolutionGovernanceControlProvider,
]:
    return (DefaultEvolutionGovernanceControlProvider(),)


__all__ = [
    "DefaultEvolutionGovernanceControlProvider",
    "default_evolution_governance_control_providers",
]
