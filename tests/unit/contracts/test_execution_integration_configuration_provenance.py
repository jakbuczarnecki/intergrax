# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P1 neutral integration configuration provenance tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.contracts.execution_integration_configuration_provenance import (
    ConfiguredIntegrationProvenanceSlice,
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceMode,
    ExecutionIntegrationConfigurationProvenanceReadStatus,
    ExecutionIntegrationConfigurationProvenanceReader,
    IntegrationConfigurationSubject,
    validate_execution_integration_configuration_provenance_record,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.execution_integration_configuration import (
    EffectiveIntegrationIdentity,
    IntegrationMaterializationKind,
)

pytestmark = pytest.mark.unit

_EXEC = validate_execution_id("exec_01234567890123456789012345678901")


def _effective() -> EffectiveIntegrationIdentity:
    return EffectiveIntegrationIdentity(
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        materialization_kind=IntegrationMaterializationKind.CATALOG_FACTORY,
    )


def _configured_slice(
    *,
    tenant_id: str = "tenant-a",
    category: IntegrationCategory = IntegrationCategory.RELATIONAL_STORE,
    provider_id: str = "sqlite",
) -> ConfiguredIntegrationProvenanceSlice:
    return ConfiguredIntegrationProvenanceSlice(
        tenant_id=tenant_id,
        integration_category=category,
        provider_id=provider_id,
        resource_scope="scope-a",
        configuration_type="test.config.v1",
        configuration_version="1",
        configuration_fingerprint="fp-prov-1",
    )


def test_valid_configured_adopted_provenance() -> None:
    record = ExecutionIntegrationConfigurationProvenance(
        tenant_id="tenant-a",
        execution_id=_EXEC,
        mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED,
        effective=_effective(),
        configured=_configured_slice(),
    )
    assert record.configured is not None


def test_configured_adopted_requires_slice() -> None:
    with pytest.raises(ValueError, match="CONFIGURED_ADOPTED"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED,
            effective=_effective(),
            configured=None,
        )


def test_configured_adopted_tenant_continuity() -> None:
    with pytest.raises(ValueError, match="tenant mismatch"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED,
            effective=_effective(),
            configured=_configured_slice(tenant_id="tenant-b"),
        )


def test_configured_adopted_category_continuity() -> None:
    with pytest.raises(ValueError, match="category mismatch"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED,
            effective=_effective(),
            configured=_configured_slice(category=IntegrationCategory.VECTOR_STORE),
        )


def test_configured_adopted_provider_continuity() -> None:
    with pytest.raises(ValueError, match="provider mismatch"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED,
            effective=_effective(),
            configured=_configured_slice(provider_id="postgres"),
        )


def test_valid_effective_only() -> None:
    record = ExecutionIntegrationConfigurationProvenance(
        tenant_id="tenant-a",
        execution_id=_EXEC,
        mode=ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY,
        effective=_effective(),
        configured=None,
    )
    assert record.configured is None


def test_effective_only_rejects_configured_slice() -> None:
    with pytest.raises(ValueError, match="EFFECTIVE_ONLY"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY,
            effective=_effective(),
            configured=_configured_slice(),
        )


def test_execution_id_canonical_validation() -> None:
    with pytest.raises(ValueError):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=ExecutionId("not-canonical"),
            mode=ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY,
            effective=_effective(),
            configured=None,
        )


def test_read_status_enum_values() -> None:
    assert ExecutionIntegrationConfigurationProvenanceReadStatus.NOT_CONFIGURED.value == "not_configured"
    assert ExecutionIntegrationConfigurationProvenanceReadStatus.REQUIRED_MISSING.value == "required_missing"


class _FakeReader:
    def read_all(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
        return (
            ExecutionIntegrationConfigurationProvenance(
                tenant_id=tenant_id,
                execution_id=execution_id,
                mode=ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY,
                effective=_effective(),
                configured=None,
            ),
        )


def test_reader_protocol_structural_conformance() -> None:
    assert isinstance(_FakeReader(), ExecutionIntegrationConfigurationProvenanceReader)


def test_provenance_records_are_immutable() -> None:
    record = ExecutionIntegrationConfigurationProvenance(
        tenant_id="tenant-a",
        execution_id=_EXEC,
        mode=ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY,
        effective=_effective(),
        configured=None,
    )
    with pytest.raises(AttributeError):
        record.tenant_id = "other"  # type: ignore[misc]


def test_subject_ordering_deterministic() -> None:
    a = IntegrationConfigurationSubject(
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="a",
        resource_scope="s",
        configuration_type="t",
    )
    b = IntegrationConfigurationSubject(
        integration_category=IntegrationCategory.VECTOR_STORE,
        provider_id="b",
        resource_scope="s",
        configuration_type="t",
    )
    assert a < b


def test_subject_reject_raw_string_integration_category() -> None:
    with pytest.raises(ValueError, match="integration_category"):
        IntegrationConfigurationSubject(
            integration_category=cast(Any, "relational_store"),
            provider_id="sqlite",
            resource_scope="scope-a",
            configuration_type="test.config.v1",
        )


def test_configured_slice_reject_raw_string_integration_category() -> None:
    with pytest.raises(ValueError, match="integration_category"):
        ConfiguredIntegrationProvenanceSlice(
            tenant_id="tenant-a",
            integration_category=cast(Any, "relational_store"),
            provider_id="sqlite",
            resource_scope="scope-a",
            configuration_type="test.config.v1",
            configuration_version="1",
            configuration_fingerprint="fp-prov-1",
        )


def test_provenance_reject_raw_string_mode_configured_adopted() -> None:
    with pytest.raises(ValueError, match="unknown provenance mode"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=cast(Any, "configured_adopted"),
            effective=_effective(),
            configured=_configured_slice(),
        )


def test_provenance_reject_raw_string_mode_effective_only() -> None:
    with pytest.raises(ValueError, match="unknown provenance mode"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=cast(Any, "effective_only"),
            effective=_effective(),
            configured=None,
        )


@dataclass
class _EffectiveLookalike:
    integration_category: IntegrationCategory
    provider_id: str
    materialization_kind: IntegrationMaterializationKind


@dataclass
class _ConfiguredSliceLookalike:
    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    configuration_type: str
    configuration_version: str
    configuration_fingerprint: str


def test_provenance_reject_effective_lookalike() -> None:
    lookalike = _EffectiveLookalike(
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        materialization_kind=IntegrationMaterializationKind.CATALOG_FACTORY,
    )
    with pytest.raises(ValueError, match="effective identity invalid"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY,
            effective=cast(Any, lookalike),
            configured=None,
        )


def test_provenance_reject_configured_lookalike() -> None:
    lookalike = _ConfiguredSliceLookalike(
        tenant_id="tenant-a",
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope-a",
        configuration_type="test.config.v1",
        configuration_version="1",
        configuration_fingerprint="fp-prov-1",
    )
    with pytest.raises(ValueError, match="configured provenance slice invalid"):
        ExecutionIntegrationConfigurationProvenance(
            tenant_id="tenant-a",
            execution_id=_EXEC,
            mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED,
            effective=_effective(),
            configured=cast(Any, lookalike),
        )


def test_record_validator_rejects_tenant_mismatch() -> None:
    record = ExecutionIntegrationConfigurationProvenance(
        tenant_id="tenant-b",
        execution_id=_EXEC,
        mode=ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY,
        effective=_effective(),
        configured=None,
    )
    with pytest.raises(ValueError, match="tenant mismatch"):
        validate_execution_integration_configuration_provenance_record(
            record,
            expected_tenant_id="tenant-a",
            expected_execution_id=_EXEC,
        )
