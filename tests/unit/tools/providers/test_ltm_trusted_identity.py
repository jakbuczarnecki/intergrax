# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R2: LTM tool fail-closed trusted identity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_control import MemoryControlAccessDenied
from intergrax.tools.providers.ltm.contracts import LtmWriteFactInput
from intergrax.tools.providers.ltm.service import ltm_write_fact
from intergrax.tools.registry.runtime_bindings import UserProfileManagerBinding
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.gate


def _manager_binding() -> UserProfileManagerBinding:
    binding = MagicMock(spec=UserProfileManagerBinding)
    binding.add_memory_entry = AsyncMock(return_value=MagicMock(entry_id="e1"))
    return binding


def test_ltm_write_fact_requires_trusted_identity() -> None:
    ctx = ToolWiringContext(user_profile_manager=_manager_binding())
    with pytest.raises(MemoryControlAccessDenied, match="trusted request identity required"):
        ltm_write_fact(ctx, LtmWriteFactInput(user_id="u1", content="fact"))


def test_ltm_write_fact_rejects_user_mismatch() -> None:
    identity = RequestIdentity(tenant_id="t1", user_id="user-a")
    ctx = ToolWiringContext(
        user_profile_manager=_manager_binding(),
        extras={"request_identity": identity},
    )
    with pytest.raises(MemoryControlAccessDenied, match="conflicts"):
        ltm_write_fact(ctx, LtmWriteFactInput(user_id="user-b", content="fact"))


def test_ltm_write_fact_passes_original_identity() -> None:
    identity = RequestIdentity(tenant_id="t1", user_id="u1")
    manager = _manager_binding()
    ctx = ToolWiringContext(
        user_profile_manager=manager,
        extras={"request_identity": identity},
    )
    result = ltm_write_fact(ctx, LtmWriteFactInput(user_id="u1", content="likes tea"))
    assert result.written is True
    manager.add_memory_entry.assert_awaited_once()
    call_identity = manager.add_memory_entry.await_args.args[0]
    assert call_identity is identity
