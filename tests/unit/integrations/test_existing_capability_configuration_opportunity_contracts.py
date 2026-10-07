# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P1 configuration opportunity contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    IntegrationConfigurationPayload,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ConfigurationOpportunityRef,
    ExistingCapabilityConfigurationMutationRiskPolicy,
    ExistingCapabilityConfigurationOpportunity,
    ExistingCapabilityConfigurationOpportunityFacts,
    ExistingCapabilityConfigurationOpportunityLookupError,
    ExistingCapabilityConfigurationOpportunityLookupFailureReason,
    ExistingCapabilityConfigurationOpportunityProvider,
    ExistingCapabilityConfigurationOpportunityReadPort,
    validate_configuration_opportunity_ref,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _TestConfigurationPayload:
    _configuration_type: str
    _configuration_version: str
    _configuration_fingerprint: str

    @property
    def configuration_type(self) -> str:
        return self._configuration_type

    @property
    def configuration_version(self) -> str:
        return self._configuration_version

    @property
    def configuration_fingerprint(self) -> str:
        return self._configuration_fingerprint


def _payload(**overrides: str) -> IntegrationConfigurationPayload:
    return _TestConfigurationPayload(
        _configuration_type=overrides.get("configuration_type", "test.config.v1"),
        _configuration_version=overrides.get("configuration_version", "1"),
        _configuration_fingerprint=overrides.get(
            "configuration_fingerprint", "fp-test-001"
        ),
    )


def _facts(**overrides: str) -> ExistingCapabilityConfigurationOpportunityFacts:
    fp = overrides.get("configuration_fingerprint", "fp-test-001")
    return ExistingCapabilityConfigurationOpportunityFacts(
        tenant_id=overrides.get("tenant_id", "tenant-a"),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=overrides.get("provider_id", "sqlite"),
        resource_scope=overrides.get("resource_scope", "scope-a"),
        current_revision=overrides.get("current_revision", "rev-1"),
        configuration=_payload(configuration_fingerprint=fp),
        configuration_fingerprint=fp,
    )


def _opportunity(
    risk: ControlPlaneMutationRisk = ControlPlaneMutationRisk.HIGH,
) -> ExistingCapabilityConfigurationOpportunity:
    facts = _facts()
    return ExistingCapabilityConfigurationOpportunity(
        configuration_ref=validate_configuration_opportunity_ref("opp-ref-001"),
        tenant_id=facts.tenant_id,
        integration_category=facts.integration_category,
        provider_id=facts.provider_id,
        resource_scope=facts.resource_scope,
        current_revision=facts.current_revision,
        configuration=facts.configuration,
        configuration_fingerprint=facts.configuration_fingerprint,
        risk_classification=risk,
    )


def test_valid_opportunity_facts() -> None:
    facts = _facts()
    assert facts.tenant_id == "tenant-a"
    assert facts.configuration.configuration_fingerprint == "fp-test-001"


def test_facts_reject_empty_tenant() -> None:
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as exc:
        _facts(tenant_id="")
    assert exc.value.reason == ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID


def test_facts_reject_empty_provider() -> None:
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError):
        _facts(provider_id="")


def test_facts_reject_empty_resource_scope() -> None:
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError):
        _facts(resource_scope="")


def test_facts_reject_empty_revision() -> None:
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError):
        _facts(current_revision="")


def test_facts_reject_empty_configuration_type() -> None:
    payload = _TestConfigurationPayload("", "1", "fp-x")
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError):
        ExistingCapabilityConfigurationOpportunityFacts(
            tenant_id="tenant-a",
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id="sqlite",
            resource_scope="scope-a",
            current_revision="rev-1",
            configuration=payload,
            configuration_fingerprint="fp-x",
        )


def test_facts_reject_empty_configuration_version() -> None:
    payload = _TestConfigurationPayload("t", "", "fp-y")
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError):
        ExistingCapabilityConfigurationOpportunityFacts(
            tenant_id="tenant-a",
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id="sqlite",
            resource_scope="scope-a",
            current_revision="rev-1",
            configuration=payload,
            configuration_fingerprint="fp-y",
        )


def test_facts_reject_fingerprint_mismatch() -> None:
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as exc:
        ExistingCapabilityConfigurationOpportunityFacts(
            tenant_id="tenant-a",
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id="sqlite",
            resource_scope="scope-a",
            current_revision="rev-1",
            configuration=_payload(configuration_fingerprint="fp-a"),
            configuration_fingerprint="fp-b",
        )
    assert (
        exc.value.reason
        == ExistingCapabilityConfigurationOpportunityLookupFailureReason.FINGERPRINT_MISMATCH
    )


def test_configuration_opportunity_ref_validation() -> None:
    ref = validate_configuration_opportunity_ref("opaque-ref-1")
    assert isinstance(ref, str)
    with pytest.raises(ValueError):
        validate_configuration_opportunity_ref("")
    with pytest.raises(ValueError):
        validate_configuration_opportunity_ref("  padded  ")


def test_opportunity_retains_risk_exactly() -> None:
    opp = _opportunity(ControlPlaneMutationRisk.CRITICAL)
    assert opp.risk_classification is ControlPlaneMutationRisk.CRITICAL


class _FakeReadPort:
    def read_exact(
        self,
        *,
        tenant_id: str,
        configuration_ref: ConfigurationOpportunityRef,
    ) -> ExistingCapabilityConfigurationOpportunity:
        return _opportunity()


class _FakeProvider:
    def discover_opportunity_facts(
        self,
        *,
        tenant_id: str,
    ) -> tuple[ExistingCapabilityConfigurationOpportunityFacts, ...]:
        return (_facts(),)


class _FakeRiskPolicy:
    def classify(
        self,
        facts: ExistingCapabilityConfigurationOpportunityFacts,
    ) -> ControlPlaneMutationRisk:
        return ControlPlaneMutationRisk.MEDIUM


def test_read_port_structural_conformance() -> None:
    assert isinstance(_FakeReadPort(), ExistingCapabilityConfigurationOpportunityReadPort)


def test_provider_spi_structural_conformance() -> None:
    assert isinstance(_FakeProvider(), ExistingCapabilityConfigurationOpportunityProvider)


def test_risk_policy_structural_conformance() -> None:
    assert isinstance(_FakeRiskPolicy(), ExistingCapabilityConfigurationMutationRiskPolicy)


def test_opportunity_has_no_permission_fields() -> None:
    opp = _opportunity()
    assert not hasattr(opp, "authorization_evidence")
    assert not hasattr(opp, "policy_action")


def test_facts_reject_raw_string_integration_category() -> None:
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as exc:
        ExistingCapabilityConfigurationOpportunityFacts(
            tenant_id="tenant-a",
            integration_category=cast(Any, "relational_store"),
            provider_id="sqlite",
            resource_scope="scope-a",
            current_revision="rev-1",
            configuration=_payload(),
            configuration_fingerprint="fp-test-001",
        )
    assert exc.value.reason == ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID


def test_opportunity_reject_raw_string_integration_category() -> None:
    facts = _facts()
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as exc:
        ExistingCapabilityConfigurationOpportunity(
            configuration_ref=validate_configuration_opportunity_ref("opp-ref-001"),
            tenant_id=facts.tenant_id,
            integration_category=cast(Any, "relational_store"),
            provider_id=facts.provider_id,
            resource_scope=facts.resource_scope,
            current_revision=facts.current_revision,
            configuration=facts.configuration,
            configuration_fingerprint=facts.configuration_fingerprint,
            risk_classification=ControlPlaneMutationRisk.HIGH,
        )
    assert exc.value.reason == ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID


def test_opportunity_reject_raw_string_risk_classification() -> None:
    facts = _facts()
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as exc:
        ExistingCapabilityConfigurationOpportunity(
            configuration_ref=validate_configuration_opportunity_ref("opp-ref-001"),
            tenant_id=facts.tenant_id,
            integration_category=facts.integration_category,
            provider_id=facts.provider_id,
            resource_scope=facts.resource_scope,
            current_revision=facts.current_revision,
            configuration=facts.configuration,
            configuration_fingerprint=facts.configuration_fingerprint,
            risk_classification=cast(Any, "low"),
        )
    assert exc.value.reason == ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID
