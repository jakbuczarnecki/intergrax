# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R: projection lifecycle contract carries trusted RequestIdentity."""

from __future__ import annotations

import inspect

import pytest

from intergrax.memory.contracts.memory_lifecycle import (
    UserProfileMemoryProjection,
    UserProfileMemoryProjectionContext,
    UserProfileMemoryReconciliationContext,
)

pytestmark = pytest.mark.gate


def test_projection_upsert_accepts_typed_context() -> None:
    upsert = UserProfileMemoryProjection.upsert_memory_entry
    signature = inspect.signature(upsert)
    params = list(signature.parameters.values())
    assert params[1].name == "context"
    assert params[1].annotation in {
        UserProfileMemoryProjectionContext,
        "UserProfileMemoryProjectionContext",
    }


def test_reconciliation_context_carries_identity() -> None:
    fields = {name for name in UserProfileMemoryReconciliationContext.__dataclass_fields__}
    assert "identity" in fields
