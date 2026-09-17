# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-3: bounded security/lifecycle/resilience certification proofs."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlForgetRequest,
    MemoryControlNotFound,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from tests.unit.memory.test_mem_ent_3_memory_control_plane import (
    RecordingMemoryProjection,
    _identity,
    _plane,
)

pytestmark = pytest.mark.gate

_REPO = Path(__file__).resolve().parents[3]
_MEMORY_ROOT = _REPO / "intergrax" / "memory"


def test_memory_core_forbids_synthetic_request_identity_construction() -> None:
    """Memory core must not mint authority; identity enters from caller trust boundary."""
    offenders: list[str] = []
    for path in _MEMORY_ROOT.rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        if "RequestIdentity(" in text:
            offenders.append(str(path.relative_to(_REPO)))
    assert offenders == []


@pytest.mark.asyncio
async def test_forget_second_identical_request_raises_not_found() -> None:
    """Explicit forget retry semantics: second delete is deterministic NOT_FOUND, not duplicate mutation."""
    projection = RecordingMemoryProjection()
    plane = _plane(projection)
    identity = _identity()
    scope = user_memory_scope(identity)
    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="once"),
    )
    entry_id = remembered.entry_id or ""
    await plane.forget(identity, scope, MemoryControlForgetRequest(entry_id=entry_id))
    with pytest.raises(MemoryControlNotFound):
        await plane.forget(identity, scope, MemoryControlForgetRequest(entry_id=entry_id))
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(top_k=5),
    )
    assert all(item.entry_id != entry_id for item in recall.items)
