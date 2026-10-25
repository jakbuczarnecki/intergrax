# © Artur Czarnecki. All rights reserved.

"""R4 systemic RCA — authoritative catalog→trusted effect classification (no Class B default)."""

from __future__ import annotations

import pytest

from intergrax.contracts.external_operation_redrive_effect_binding import (
    ExternalOperationRedriveEffectBindingMismatchError,
)
from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveEffectClass,
    resolve_authoritative_redrive_effect_class,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
    TOOL_LIFE_HTTP_READ_ONLY_QUAL_EFFECT,
    TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
    catalog_tool_has_sanctioned_redrive_effect_binding,
    resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    InMemoryExternalOperationStateStore,
)
from intergrax.runtime.external_operations.tool_external_operation_redrive_binding import (
    bind_sanctioned_redrive_effect_before_tool_physical_execution,
)
from intergrax.tools.providers.http.service import HTTP_REQUEST_TOOL_ID

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_unknown_catalog_tool_cannot_resolve_sanctioned_class_b_binding() -> None:
    assert not catalog_tool_has_sanctioned_redrive_effect_binding("custom.side_effect")
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool(
            "custom.side_effect",
        )


def test_custom_catalog_tool_does_not_inherit_platform_lifecycle_on_bind() -> None:
    store = InMemoryExternalOperationStateStore()
    record = store.create_if_absent("tenant-a", "op-custom")
    revision = bind_sanctioned_redrive_effect_before_tool_physical_execution(
        store,
        tenant_id="tenant-a",
        operation_id="op-custom",
        expected_revision=record.revision,
        catalog_tool_id="custom.side_effect",
        physical_idempotency_key="idem-1",
    )
    assert revision == record.revision
    loaded = store.load("tenant-a", "op-custom")
    assert loaded is not None
    assert loaded.bound_redrive_effect_contract_id is None


def test_known_http_catalog_tool_retains_authoritative_class_b_identity() -> None:
    contract_id, provider_id, tool_id = (
        resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool(
            HTTP_REQUEST_TOOL_ID,
        )
    )
    assert contract_id == TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT.contract_id
    assert provider_id == "http_rest"
    assert tool_id == "http_request"
    assert (
        resolve_authoritative_redrive_effect_class(TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT)
        is ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED
    )


def test_read_only_catalog_tool_maps_to_class_c_semantics() -> None:
    contract_id, provider_id, tool_id = (
        resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool(
            "http.read_only",
        )
    )
    assert contract_id == TOOL_LIFE_HTTP_READ_ONLY_QUAL_EFFECT.contract_id
    assert provider_id == "http_rest"
    assert tool_id == "http_read_only"
    assert (
        resolve_authoritative_redrive_effect_class(TOOL_LIFE_HTTP_READ_ONLY_QUAL_EFFECT)
        is ExternalOperationRedriveEffectClass.CLASS_C_FAIL_CLOSED
    )


def test_platform_lifecycle_binding_only_for_explicit_catalog_tool_id() -> None:
    contract_id, _, tool_id = resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool(
        "external_operation_physical_lifecycle",
    )
    assert contract_id == TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT.contract_id
    assert tool_id == "external_operation_physical_lifecycle"
    with pytest.raises(ExternalOperationRedriveEffectBindingMismatchError):
        resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool(
            "arbitrary.unknown.tool",
        )
