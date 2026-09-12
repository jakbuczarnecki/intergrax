# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J production hardening tests for Decision System integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from intergrax.contracts.decision.integration import (
    DecisionIntegrationStatus,
    DecisionSystemIntegrationFactory,
    InMemoryDecisionAuditSink,
    PluginAdmissionDecision,
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    RecordingDecisionIntegrationAuditProvider,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
    SingleLifecycleAdapterProvider,
)
from intergrax.contracts.decision.integration.admission import (
    DecisionIntegrationPluginDescriptor,
)
from intergrax.contracts.decision.integration.audit_sink import (
    DecisionAuditSink,
    DecisionIntegrationAuditEnvelope,
)
from intergrax.contracts.decision.integration.composition import (
    ConfiguredDecisionIntegrationCompositionProvider,
    DecisionIntegrationCompositionSpec,
)
from intergrax.contracts.decision.integration.result import (
    DecisionAdapterMetadata,
    DecisionIntegrationResult,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.runtime.decision_integration_composition import (
    production_decision_system_integration,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleRecord,
    DecisionLifecycleState,
    DecisionSourceKind,
    DecisionSourceReference,
    DecisionType,
)

pytestmark = pytest.mark.unit


def _reference_source() -> ReferenceDecisionLifecycleReference:
    record = DecisionLifecycleRecord(
        decision_id="hardening-decision-001",
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        lifecycle_state=DecisionLifecycleState.CREATED,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.MODEL_SELECTION,
                reference_id="sel-1",
            ),
        ),
    )
    return ReferenceDecisionLifecycleReference(
        source_type=REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
        decision_id=record.decision_id,
        lifecycle_state=ReferenceEnterpriseLifecycleState(record.lifecycle_state.value),
        decision_type=record.decision_type.value,
        created_at_iso=record.created_at.isoformat(),
        mapping_version="1",
    )


def test_production_audit_provider_records_through_sink() -> None:
    sink = InMemoryDecisionAuditSink()
    engine = production_decision_system_integration(audit_sink=sink)
    result = engine.integrate_lifecycle(_reference_source())

    assert result.status is DecisionIntegrationStatus.SUCCESS
    assert len(sink.entries) == 1
    envelope = sink.entries[0]
    assert envelope.record.target is result.target
    assert envelope.provider_metadata.provider_id == (
        "decision.integration.audit.recording"
    )


@dataclass
class _AppendOnlyAuditSink:
    envelopes: list[DecisionIntegrationAuditEnvelope] = field(default_factory=list)

    def append(self, envelope: DecisionIntegrationAuditEnvelope) -> None:
        self.envelopes.append(envelope)


def test_custom_audit_sink_swap_without_engine_changes() -> None:
    custom_sink: DecisionAuditSink = _AppendOnlyAuditSink()
    audit = RecordingDecisionIntegrationAuditProvider(sink=custom_sink)
    composition = ConfiguredDecisionIntegrationCompositionProvider(
        composition_spec=DecisionIntegrationCompositionSpec(
            active_lifecycle_source_types=frozenset(
                {REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE},
            ),
            audit_enabled=True,
        ),
        adapter_providers=(
            SingleLifecycleAdapterProvider(
                lifecycle_adapter=_AllowLifecycleAdapter(),
            ),
        ),
        audit_provider=audit,
    )
    engine = DecisionSystemIntegrationFactory.create_engine(composition)
    engine.integrate_lifecycle(_reference_source())

    assert len(custom_sink.envelopes) == 1


@dataclass(frozen=True, slots=True)
class _AllowLifecycleAdapter:
    @property
    def adapter_id(self) -> str:
        return "hardening.allow.adapter"

    @property
    def adapter_version(self) -> str:
        return "1.0.0"

    @property
    def mapping_version(self) -> str:
        return "1"

    @property
    def source_type(self) -> str:
        return REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE

    def integrate_lifecycle(
        self,
        source: ReferenceDecisionLifecycleReference,
    ) -> DecisionIntegrationResult:
        integrated_at = datetime.now(tz=UTC)
        from intergrax.contracts.decision.integration.references import (
            PlatformDecisionLifecycleReference,
        )

        metadata = DecisionAdapterMetadata(
            source_type=source.source_type,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            mapping_version=self.mapping_version,
            integrated_at=integrated_at,
        )
        target = PlatformDecisionLifecycleReference(
            reference_decision_id=source.decision_id,
            stage=DecisionLifecycleStage.PROPOSAL,
            transition_index=0,
            mapping_version=self.mapping_version,
        )
        return DecisionIntegrationResult(
            status=DecisionIntegrationStatus.SUCCESS,
            source=source,
            target=target,
            adapter_metadata=metadata,
            detail="allow adapter",
        )


@dataclass(frozen=True, slots=True)
class _StaticAdmissionProvider:
    decision: PluginAdmissionDecision

    def evaluate(
        self,
        descriptor: DecisionIntegrationPluginDescriptor,
    ) -> PluginAdmissionDecision:
        return self.decision


def test_plugin_admission_allow_flow() -> None:
    engine = production_decision_system_integration(
        plugin_admission_provider=_StaticAdmissionProvider(
            decision=PluginAdmissionDecision.ALLOW,
        ),
    )
    result = engine.integrate_lifecycle(_reference_source())
    assert result.status is DecisionIntegrationStatus.SUCCESS


def test_plugin_admission_deny_controlled_failure() -> None:
    engine = production_decision_system_integration(
        plugin_admission_provider=_StaticAdmissionProvider(
            decision=PluginAdmissionDecision.DENY,
        ),
    )
    result = engine.integrate_lifecycle(_reference_source())
    assert result.status is DecisionIntegrationStatus.FAILED
    assert "no lifecycle integration adapter" in result.detail


@dataclass(frozen=True, slots=True)
class _FailingLifecycleAdapter:
    @property
    def adapter_id(self) -> str:
        return "hardening.failing.adapter"

    @property
    def adapter_version(self) -> str:
        return "1.0.0"

    @property
    def mapping_version(self) -> str:
        return "1"

    @property
    def source_type(self) -> str:
        return REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE

    def integrate_lifecycle(
        self,
        source: ReferenceDecisionLifecycleReference,
    ) -> DecisionIntegrationResult:
        raise RuntimeError("simulated adapter fault")


def test_adapter_failure_controlled_with_audit() -> None:
    sink = InMemoryDecisionAuditSink()
    composition = ConfiguredDecisionIntegrationCompositionProvider(
        composition_spec=DecisionIntegrationCompositionSpec(
            active_lifecycle_source_types=frozenset(
                {REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE},
            ),
            audit_enabled=True,
        ),
        adapter_providers=(
            SingleLifecycleAdapterProvider(
                lifecycle_adapter=_FailingLifecycleAdapter()
            ),
        ),
        audit_provider=RecordingDecisionIntegrationAuditProvider(sink=sink),
    )
    engine = DecisionSystemIntegrationFactory.create_engine(composition)
    result = engine.integrate_lifecycle(_reference_source())

    assert result.status is DecisionIntegrationStatus.FAILED
    assert result.detail.startswith("adapter_execution_error:")
    assert len(sink.entries) == 1
    assert sink.entries[0].record.status is DecisionIntegrationStatus.FAILED


def test_regression_import_boundary_composition_tests() -> None:
    """Ensure hardening modules stay importable alongside existing DS-E2E-15J surface."""
    from intergrax.contracts.decision.integration import DecisionSystemIntegrationEngine
    from intergrax.runtime.decision_integration_composition import (
        default_decision_system_integration,
    )

    assert (
        type(default_decision_system_integration()) is DecisionSystemIntegrationEngine
    )
