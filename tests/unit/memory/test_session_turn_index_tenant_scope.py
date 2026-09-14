# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.memory.memory_vector_errors import MemoryTenantScopeViolationError
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_search_turns_rejects_cross_tenant_scope_override() -> None:
    store = VectorSessionTurnIndexStore(
        embedding_manager=MagicMock(),
        vectorstore_manager=MagicMock(),
        tenant_id="tenant-bound",
    )
    with pytest.raises(MemoryTenantScopeViolationError):
        await store.search_turns(
            query="hello",
            tenant_id="other-tenant",
            session_id="sess-1",
        )
