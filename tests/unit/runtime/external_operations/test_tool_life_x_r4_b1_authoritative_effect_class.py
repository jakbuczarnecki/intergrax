# © Artur Czarnecki. All rights reserved.

"""R4 B1 — authoritative trusted-contract redrive effect class (G-A02)."""

from __future__ import annotations

import pytest

from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveEffectClass,
    ExternalOperationRedriveEffectClassRejectedError,
    assert_redrive_effect_class_claim_matches_authoritative,
    resolve_authoritative_redrive_effect_class,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
    TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT,
    TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT,
    TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_authoritative_class_b_from_trusted_contracts() -> None:
    assert (
        resolve_authoritative_redrive_effect_class(
            TOOL_LIFE_PLATFORM_DURABLE_PHYSICAL_LIFECYCLE_EFFECT,
        )
        is ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED
    )
    assert (
        resolve_authoritative_redrive_effect_class(TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_EFFECT)
        is ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED
    )


def test_resolve_authoritative_class_c_for_unqualified_effects() -> None:
    assert (
        resolve_authoritative_redrive_effect_class(TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT)
        is ExternalOperationRedriveEffectClass.CLASS_C_FAIL_CLOSED
    )
    assert (
        resolve_authoritative_redrive_effect_class(TOOL_LIFE_MCP_TOOLS_CALL_DEFAULT_QUAL_EFFECT)
        is ExternalOperationRedriveEffectClass.CLASS_C_FAIL_CLOSED
    )


def test_caller_cannot_elevate_class_c_claim_over_authoritative() -> None:
    authoritative = resolve_authoritative_redrive_effect_class(
        TOOL_LIFE_HTTP_BLIND_POST_QUAL_EFFECT,
    )
    assert authoritative is ExternalOperationRedriveEffectClass.CLASS_C_FAIL_CLOSED
    with pytest.raises(ExternalOperationRedriveEffectClassRejectedError):
        assert_redrive_effect_class_claim_matches_authoritative(
            caller_claimed_effect_class=ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED,
            authoritative_effect_class=authoritative,
        )
