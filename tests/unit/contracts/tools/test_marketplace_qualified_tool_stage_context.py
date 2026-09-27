# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P2 marketplace qualified tool stage context contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
)

pytestmark = pytest.mark.unit


def test_context_immutable_fields() -> None:
    ctx = MarketplaceQualifiedToolStageContext(
        handoff_id="marketplace-gap-handoff:v2:abc",
        tenant_id="tenant-1",
        acquisition_request_id="acq-1",
    )
    assert ctx.handoff_id.startswith("marketplace-gap-handoff:v2:")
    with pytest.raises(ValidationError):
        MarketplaceQualifiedToolStageContext(
            handoff_id="",
            tenant_id="tenant-1",
            acquisition_request_id="acq-1",
        )
