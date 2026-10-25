# © Artur Czarnecki. All rights reserved.

"""R4 B2-B — generic MCP tools/call remains Class C for automated redrive (G-A02)."""

from __future__ import annotations

import pytest

from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveEffectClass,
    ExternalOperationRedriveEffectClassRejectedError,
    resolve_authoritative_redrive_effect_class,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    InMemoryExternalOperationStateStore,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    transfer_redrive_authority_after_reconciliation,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_mcp_tools_call_default_trusted_contract_is_class_c() -> None:
    assert (
        resolve_authoritative_redrive_effect_class(TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT)
        is ExternalOperationRedriveEffectClass.CLASS_C_FAIL_CLOSED
    )


def test_mcp_class_c_cannot_obtain_redrive_authority_transfer() -> None:
    store = InMemoryExternalOperationStateStore()
    with pytest.raises(ExternalOperationRedriveEffectClassRejectedError):
        transfer_redrive_authority_after_reconciliation(
            store,
            tenant_id="tenant-a",
            operation_id="mcp-op-1",
            expected_revision=0,
            trusted_effect_contract=TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT,
            worker_instance_id="worker-recovery",
            worker_ownership_epoch=2,
            caller_claimed_effect_class=ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED,
        )
