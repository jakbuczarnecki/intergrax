# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-DECISION-SYSTEM-INTEGRATION-BOUNDARY contract tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from intergrax.contracts.decision.integration import (
    DecisionIntegrationAuditRecord,
    DecisionIntegrationStatus,
    DecisionSystemIntegrationEngine,
    DefaultDecisionLifecycleIntegrationAdapter,
    PlatformDecisionLifecycleReference,
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
    SingleLifecycleAdapterProvider,
)
from intergrax.contracts.decision.integration.result import (
    DecisionAdapterMetadata,
    DecisionIntegrationResult,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleRecord,
    DecisionLifecycleState,
    DecisionSourceKind,
    DecisionSourceReference,
    DecisionType,
)

pytestmark = pytest.mark.unit


def _reference_from_matrix_record(
    record: DecisionLifecycleRecord,
) -> ReferenceDecisionLifecycleReference:
    return ReferenceDecisionLifecycleReference(
        source_type=REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
        decision_id=record.decision_id,
        lifecycle_state=ReferenceEnterpriseLifecycleState(record.lifecycle_state.value),
        decision_type=record.decision_type.value,
        created_at_iso=record.created_at.isoformat(),
        mapping_version="1",
    )


def _matrix_record(
    *,
    lifecycle_state: DecisionLifecycleState = DecisionLifecycleState.CREATED,
) -> DecisionLifecycleRecord:
    return DecisionLifecycleRecord(
        decision_id="matrix-decision-001",
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        lifecycle_state=lifecycle_state,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.MODEL_SELECTION,
                reference_id="sel-1",
            ),
        ),
    )


def _engine_with_default_adapter() -> DecisionSystemIntegrationEngine:
    adapter = DefaultDecisionLifecycleIntegrationAdapter()
    provider = SingleLifecycleAdapterProvider(lifecycle_adapter=adapter)
    return DecisionSystemIntegrationEngine(adapter_providers=(provider,))


def test_lifecycle_mapping_reference_record_to_platform_reference() -> None:
    record = _matrix_record(lifecycle_state=DecisionLifecycleState.CREATED)
    source = _reference_from_matrix_record(record)
    result = _engine_with_default_adapter().integrate_lifecycle(source)

    assert result.status is DecisionIntegrationStatus.SUCCESS
    assert result.target is not None
    assert result.target.reference_decision_id == record.decision_id
    assert result.target.stage is DecisionLifecycleStage.PROPOSAL
    assert result.target.transition_index == 0


@dataclass
class _RecordingAuditProvider:
    records: list[DecisionIntegrationAuditRecord] = field(default_factory=list)

    def record_integration(self, record: DecisionIntegrationAuditRecord) -> None:
        self.records.append(record)


@dataclass(frozen=True, slots=True)
class _CustomLifecycleAdapter:
    """Plugin replacement — maps every reference state to VERIFICATION."""

    @property
    def adapter_id(self) -> str:
        return "custom.lifecycle.integration"

    @property
    def adapter_version(self) -> str:
        return "9.9.9"

    @property
    def mapping_version(self) -> str:
        return "custom-1"

    @property
    def source_type(self) -> str:
        return REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE

    def integrate_lifecycle(
        self,
        source: ReferenceDecisionLifecycleReference,
    ) -> DecisionIntegrationResult:
        integrated_at = datetime.now(tz=UTC)
        target = PlatformDecisionLifecycleReference(
            reference_decision_id=source.decision_id,
            stage=DecisionLifecycleStage.VERIFICATION,
            transition_index=0,
            mapping_version=self.mapping_version,
        )
        metadata = DecisionAdapterMetadata(
            source_type=source.source_type,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            mapping_version=self.mapping_version,
            integrated_at=integrated_at,
        )
        return DecisionIntegrationResult(
            status=DecisionIntegrationStatus.SUCCESS,
            source=source,
            target=target,
            adapter_metadata=metadata,
            detail="custom adapter mapping",
        )


def test_custom_adapter_plugin_engine_unchanged() -> None:
    record = _matrix_record()
    source = _reference_from_matrix_record(record)
    custom_engine = DecisionSystemIntegrationEngine(
        adapter_providers=(
            SingleLifecycleAdapterProvider(lifecycle_adapter=_CustomLifecycleAdapter()),
        ),
    )
    result = custom_engine.integrate_lifecycle(source)

    assert result.status is DecisionIntegrationStatus.SUCCESS
    assert result.target is not None
    assert result.target.stage is DecisionLifecycleStage.VERIFICATION
    assert result.adapter_metadata.adapter_id == "custom.lifecycle.integration"


def test_missing_required_source_reports_warning_or_failed() -> None:
    record = _matrix_record()
    source = ReferenceDecisionLifecycleReference(
        source_type=REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
        decision_id=record.decision_id,
        lifecycle_state=ReferenceEnterpriseLifecycleState.CREATED,
        decision_type=record.decision_type.value,
        created_at_iso=None,
        mapping_version="1",
    )
    result = _engine_with_default_adapter().integrate_lifecycle(source)

    assert result.status in {
        DecisionIntegrationStatus.WARNING,
        DecisionIntegrationStatus.FAILED,
    }
    assert result.target is None


def test_audit_records_adapter_source_and_target() -> None:
    record = _matrix_record(lifecycle_state=DecisionLifecycleState.EVALUATING)
    source = _reference_from_matrix_record(record)
    audit = _RecordingAuditProvider()
    engine = DecisionSystemIntegrationEngine(
        adapter_providers=(
            SingleLifecycleAdapterProvider(
                lifecycle_adapter=DefaultDecisionLifecycleIntegrationAdapter(),
            ),
        ),
        audit_provider=audit,
    )
    result = engine.integrate_lifecycle(source)

    assert len(audit.records) == 1
    entry = audit.records[0]
    assert entry.adapter_metadata.adapter_id == "default.decision.lifecycle.integration"
    assert entry.adapter_metadata.adapter_version == "1.0.0"
    assert entry.source.decision_id == record.decision_id
    assert entry.target is result.target
    assert entry.target is not None
    assert entry.target.stage is DecisionLifecycleStage.DELIBERATION


def test_provider_replacement_without_engine_changes() -> None:
    record = _matrix_record()
    source = _reference_from_matrix_record(record)
    engine = DecisionSystemIntegrationEngine(
        adapter_providers=(
            SingleLifecycleAdapterProvider(lifecycle_adapter=_CustomLifecycleAdapter()),
        ),
    )
    result = engine.integrate_lifecycle(source)

    assert type(engine) is DecisionSystemIntegrationEngine
    assert result.adapter_metadata.adapter_id == "custom.lifecycle.integration"


def test_no_registered_adapter_fails() -> None:
    source = _reference_from_matrix_record(_matrix_record())
    engine = DecisionSystemIntegrationEngine(adapter_providers=())
    result = engine.integrate_lifecycle(source)

    assert result.status is DecisionIntegrationStatus.FAILED
