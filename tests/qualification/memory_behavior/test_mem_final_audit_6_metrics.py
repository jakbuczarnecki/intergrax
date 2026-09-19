# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — semantic quality metrics and zero-violation ledger."""

from __future__ import annotations

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from tests.qualification.memory_behavior.contracts import BehaviorViolationCounters
from tests.qualification.memory_behavior.fixtures import (
    build_user_control_plane,
    request_identity,
)
from tests.qualification.memory_behavior.runner import build_semantic_metrics

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_semantic_quality_deterministic_dataset() -> None:
    plane, _ = build_user_control_plane(enable_ltm_vector=True)
    identity = request_identity(user_id="metrics-user")
    scope = user_memory_scope(identity)
    lang = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="Preferred language is Polish."),
    )
    db = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="Favorite database is PostgreSQL."),
    )
    queries = ("language", "database")
    expected = (lang.entry_id or "", db.entry_id or "")
    retrieved: list[tuple[str, ...]] = []
    for query in queries:
        recall = await plane.recall(
            identity,
            scope,
            MemoryControlRecallRequest(query=query, top_k=3),
        )
        retrieved.append(tuple(item.entry_id for item in recall.items))
    metrics = build_semantic_metrics(
        expected_ids=expected,
        retrieved_ids=retrieved,
        top_k=3,
    )
    assert metrics.dataset_size == 2
    assert metrics.hit_at_1 >= 0.5
    assert metrics.recall_at_k >= 0.5


def test_zero_violation_counters_default() -> None:
    counters = BehaviorViolationCounters()
    assert counters.cross_tenant_leaks == 0
    assert counters.cross_user_leaks == 0
    assert counters.deleted_resurrections == 0
    assert counters.superseded_as_current == 0
    assert counters.projection_only_ghosts == 0
    assert counters.identity_authority_violations == 0
    assert not counters.has_hard_violation
