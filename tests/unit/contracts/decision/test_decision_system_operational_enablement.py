# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-DECISION-SYSTEM-OPERATIONAL-ENABLEMENT acceptance tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.contracts.decision.integration import (
    DecisionIntegrationStatus,
    DecisionSystemIntegrationEngine,
    DecisionSystemIntegrationFactory,
    InMemoryDecisionAuditSink,
    PluginAdmissionDecision,
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
)
from intergrax.contracts.decision.integration.admission import (
    DecisionIntegrationPluginDescriptor,
    resolve_integration_plugin_descriptor,
)
from intergrax.contracts.decision.integration.composition import (
    ConfiguredDecisionIntegrationCompositionProvider,
    DecisionIntegrationCompositionSpec,
)
from intergrax.contracts.decision.integration.lifecycle.provider import (
    DefaultLifecycleAdapterProvider,
)
from intergrax.runtime.decision_integration_composition import (
    default_decision_system_integration,
    production_decision_integration_composition_provider,
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
        decision_id="operational-decision-001",
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        lifecycle_state=DecisionLifecycleState.CREATED,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.MODEL_SELECTION,
                reference_id="sel-operational",
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


def test_operational_production_composition_root_to_engine() -> None:
    sink = InMemoryDecisionAuditSink()
    provider = production_decision_integration_composition_provider(audit_sink=sink)
    engine = DecisionSystemIntegrationFactory.create_engine(provider)
    result = engine.integrate_lifecycle(_reference_source())

    assert result.status is DecisionIntegrationStatus.SUCCESS
    assert result.source.decision_id == "operational-decision-001"
    assert len(sink.entries) == 1


def test_operational_plugin_lifecycle_metadata() -> None:
    descriptor = resolve_integration_plugin_descriptor(
        DefaultLifecycleAdapterProvider()
    )
    assert type(descriptor) is DecisionIntegrationPluginDescriptor
    assert descriptor.plugin_id
    assert descriptor.version
    assert descriptor.source == "decision.integration.lifecycle_adapter"


@dataclass(frozen=True, slots=True)
class _DenyAdmission:
    def evaluate(
        self,
        descriptor: DecisionIntegrationPluginDescriptor,
    ) -> PluginAdmissionDecision:
        return PluginAdmissionDecision.DENY


def test_operational_failure_handling_missing_provider_and_admission() -> None:
    denied = production_decision_system_integration(
        plugin_admission_provider=_DenyAdmission(),
    )
    denied_result = denied.integrate_lifecycle(_reference_source())
    assert denied_result.status is DecisionIntegrationStatus.FAILED

    empty = DecisionSystemIntegrationFactory.create_engine(
        ConfiguredDecisionIntegrationCompositionProvider(
            composition_spec=DecisionIntegrationCompositionSpec(
                active_lifecycle_source_types=frozenset(),
                audit_enabled=False,
            ),
            adapter_providers=(),
            audit_provider=None,
        ),
    )
    empty_result = empty.integrate_lifecycle(_reference_source())
    assert empty_result.status is DecisionIntegrationStatus.FAILED


def test_operational_configuration_replacement_without_engine_changes() -> None:
    sink_a = InMemoryDecisionAuditSink()
    sink_b = InMemoryDecisionAuditSink()
    engine_a = production_decision_system_integration(audit_sink=sink_a)
    engine_b = production_decision_system_integration(audit_sink=sink_b)
    source = _reference_source()

    engine_a.integrate_lifecycle(source)
    engine_b.integrate_lifecycle(source)

    assert len(sink_a.entries) == 1
    assert len(sink_b.entries) == 1
    assert sink_a.entries is not sink_b.entries


def test_operational_regression_composition_and_hardening_imports() -> None:
    assert (
        type(default_decision_system_integration()) is DecisionSystemIntegrationEngine
    )
    assert (
        type(production_decision_system_integration())
        is DecisionSystemIntegrationEngine
    )
