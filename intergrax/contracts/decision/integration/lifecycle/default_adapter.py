# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default L7 reference lifecycle → platform lifecycle mapping plugin."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.contracts.decision.integration.references import (
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    PlatformDecisionLifecycleReference,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
)
from intergrax.contracts.decision.integration.result import (
    DecisionAdapterMetadata,
    DecisionIntegrationResult,
    DecisionIntegrationStatus,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage

DEFAULT_LIFECYCLE_ADAPTER_ID = "default.decision.lifecycle.integration"
DEFAULT_LIFECYCLE_ADAPTER_VERSION = "1.0.0"
DEFAULT_LIFECYCLE_MAPPING_VERSION = "1"

_REFERENCE_TO_PLATFORM_STAGE: dict[
    ReferenceEnterpriseLifecycleState,
    tuple[DecisionLifecycleStage, int],
] = {
    ReferenceEnterpriseLifecycleState.CREATED: (
        DecisionLifecycleStage.PROPOSAL,
        0,
    ),
    ReferenceEnterpriseLifecycleState.EVALUATING: (
        DecisionLifecycleStage.DELIBERATION,
        0,
    ),
    ReferenceEnterpriseLifecycleState.APPROVED: (
        DecisionLifecycleStage.ADJUDICATION,
        0,
    ),
    ReferenceEnterpriseLifecycleState.REJECTED: (
        DecisionLifecycleStage.TERMINAL,
        0,
    ),
    ReferenceEnterpriseLifecycleState.EXECUTING: (
        DecisionLifecycleStage.FINALIZATION,
        0,
    ),
    ReferenceEnterpriseLifecycleState.COMPLETED: (
        DecisionLifecycleStage.TERMINAL,
        1,
    ),
    ReferenceEnterpriseLifecycleState.FAILED: (
        DecisionLifecycleStage.TERMINAL,
        1,
    ),
}


def _adapter_metadata(
    source: ReferenceDecisionLifecycleReference,
    integrated_at: datetime,
) -> DecisionAdapterMetadata:
    return DecisionAdapterMetadata(
        source_type=source.source_type,
        adapter_id=DEFAULT_LIFECYCLE_ADAPTER_ID,
        adapter_version=DEFAULT_LIFECYCLE_ADAPTER_VERSION,
        mapping_version=DEFAULT_LIFECYCLE_MAPPING_VERSION,
        integrated_at=integrated_at,
    )


@dataclass(frozen=True, slots=True)
class DefaultDecisionLifecycleIntegrationAdapter:
    """Maps reference enterprise lifecycle records to platform lifecycle references."""

    @property
    def adapter_id(self) -> str:
        return DEFAULT_LIFECYCLE_ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return DEFAULT_LIFECYCLE_ADAPTER_VERSION

    @property
    def mapping_version(self) -> str:
        return DEFAULT_LIFECYCLE_MAPPING_VERSION

    @property
    def source_type(self) -> str:
        return REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE

    def integrate_lifecycle(
        self,
        source: ReferenceDecisionLifecycleReference,
    ) -> DecisionIntegrationResult:
        if type(source) is not ReferenceDecisionLifecycleReference:
            raise TypeError("source must be ReferenceDecisionLifecycleReference")

        integrated_at = datetime.now(tz=UTC)
        metadata = _adapter_metadata(source, integrated_at)

        if source.source_type != REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE:
            return DecisionIntegrationResult(
                status=DecisionIntegrationStatus.FAILED,
                source=source,
                target=None,
                adapter_metadata=metadata,
                detail="unsupported source_type for default lifecycle adapter",
            )

        if source.lifecycle_state is None:
            return DecisionIntegrationResult(
                status=DecisionIntegrationStatus.FAILED,
                source=source,
                target=None,
                adapter_metadata=metadata,
                detail="reference lifecycle_state is required",
            )

        if source.created_at_iso is None or not source.created_at_iso.strip():
            return DecisionIntegrationResult(
                status=DecisionIntegrationStatus.WARNING,
                source=source,
                target=None,
                adapter_metadata=metadata,
                detail="reference created_at_iso is required for platform mapping",
            )

        mapping = _REFERENCE_TO_PLATFORM_STAGE.get(source.lifecycle_state)
        if mapping is None:
            return DecisionIntegrationResult(
                status=DecisionIntegrationStatus.FAILED,
                source=source,
                target=None,
                adapter_metadata=metadata,
                detail="no platform mapping for reference lifecycle_state",
            )

        stage, transition_index = mapping
        target = PlatformDecisionLifecycleReference(
            reference_decision_id=source.decision_id,
            stage=stage,
            transition_index=transition_index,
            mapping_version=DEFAULT_LIFECYCLE_MAPPING_VERSION,
        )
        return DecisionIntegrationResult(
            status=DecisionIntegrationStatus.SUCCESS,
            source=source,
            target=target,
            adapter_metadata=metadata,
            detail=(
                f"mapped reference lifecycle_state={source.lifecycle_state.value} "
                f"to platform stage={stage.value}"
            ),
        )


__all__ = [
    "DEFAULT_LIFECYCLE_ADAPTER_ID",
    "DEFAULT_LIFECYCLE_ADAPTER_VERSION",
    "DEFAULT_LIFECYCLE_MAPPING_VERSION",
    "DefaultDecisionLifecycleIntegrationAdapter",
]
