# © Artur Czarnecki. All rights reserved.

"""ERL plugin architecture foundation — contract tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    EnterpriseReliabilityRiskLevel,
    EnterpriseReliabilityStrategyContext,
    ExternalEffectOutcome,
    ReconciliationStrategyAdvice,
    UncertaintyLifecyclePhase,
    assert_plugin_identity_matches_descriptor,
)

pytestmark = pytest.mark.unit


def test_plugin_descriptor_requires_capabilities() -> None:
    with pytest.raises(ValueError, match="capabilities must be non-empty"):
        EnterpriseReliabilityPluginDescriptor(
            plugin_id="p1",
            version="1.0.0",
            owner="team",
            capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
            capabilities=(),
            tenant_scope=None,
            priority=0,
        )


def test_strategy_context_validates_ids() -> None:
    with pytest.raises(ValueError, match="correlation_id required"):
        EnterpriseReliabilityStrategyContext(
            tenant_id="t1",
            correlation_id=" ",
            contract_id="c1",
            effect_outcome=ExternalEffectOutcome.UNKNOWN,
            lifecycle_phase=UncertaintyLifecyclePhase.CONTAINED,
        )


def test_assert_plugin_identity_matches_descriptor() -> None:
    descriptor = EnterpriseReliabilityPluginDescriptor(
        plugin_id="probe-default",
        version="1.0.0",
        owner="platform",
        capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
        capabilities=("reconcile",),
        tenant_scope=None,
        priority=10,
    )
    assert_plugin_identity_matches_descriptor("probe-default", "1.0.0", descriptor)
    with pytest.raises(ValueError, match="plugin_id mismatch"):
        assert_plugin_identity_matches_descriptor("other", "1.0.0", descriptor)


def test_reconciliation_advice_requires_probe_ref() -> None:
    with pytest.raises(ValueError, match="probe_ref required"):
        ReconciliationStrategyAdvice(probe_ref=" ")


def test_capability_kind_covers_future_strategy_families() -> None:
    kinds = {item.value for item in EnterpriseReliabilityCapabilityKind}
    assert kinds == {
        "reconciliation",
        "resolution",
        "compensation",
        "risk_evaluation",
    }


def test_risk_level_enum_values() -> None:
    assert EnterpriseReliabilityRiskLevel.CRITICAL.value == "critical"
