# © Artur Czarnecki. All rights reserved.

"""USER scope behavioral scenarios (single source for pytest + aggregate runner)."""

from __future__ import annotations

from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.memory_control import (
    MemoryControlBackendError,
    MemoryControlForgetRequest,
    MemoryControlGovernanceDenied,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_recall import MemorySupersessionIntent
from tests.qualification.memory_behavior.contracts import BehaviorEvalContext
from tests.qualification.memory_behavior.fixtures import (
    build_user_control_plane,
    deny_remember_governance,
    request_identity,
)
from tests.qualification.memory_behavior.gate_helpers import (
    assert_superseded_not_current_winner,
    assert_unresolved_conflict_evidence,
)

_USER = "audit6-user-1"


async def run_user_01_basic_remember_recall(_ctx: BehaviorEvalContext) -> None:
    plane, _manager = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="Preferred language is Polish."),
    )
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="language", top_k=5),
    )
    assert remembered.entry_id
    assert any(item.entry_id == remembered.entry_id for item in recall.items)
    assert all("secret" not in item.content.lower() for item in recall.items)


async def run_user_02_irrelevant_memory_not_displacing(_ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    lang = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="Preferred language is Polish."),
    )
    await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="Favorite database is PostgreSQL."),
    )
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="language", top_k=5),
    )
    ids = {item.entry_id for item in recall.items}
    assert lang.entry_id in ids


async def run_user_03_top_k_respected(_ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    for index in range(8):
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content=f"preference item {index}"),
        )
    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=3))
    assert len(recall.items) <= 3


async def run_user_04_empty_query_no_semantic_recall(_ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    await plane.remember(identity, scope, MemoryControlRememberRequest(content="fact"))
    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(query="", top_k=5))
    assert recall.used_semantic is False
    assert recall.reason == "profile_scan"
    assert recall.items


async def run_user_05_disabled_semantic_fallback(_ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    await plane.remember(identity, scope, MemoryControlRememberRequest(content="alpha"))
    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(query="alpha", top_k=3))
    assert recall.used_semantic is False
    assert recall.reason in {"keyword", "profile_scan"}


async def run_user_06_forget_hard_gate(ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="secret fact"),
    )
    await plane.forget(
        identity,
        scope,
        MemoryControlForgetRequest(entry_id=remembered.entry_id or ""),
    )
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="secret", top_k=5),
    )
    for item in recall.items:
        if item.entry_id == remembered.entry_id:
            ctx.ledger.record_deleted_resurrection()
            raise AssertionError("deleted memory entry resurrected in recall")


async def run_user_09_supersession_lineage(_ctx: BehaviorEvalContext) -> None:
    plane, manager = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    old = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="The user works at Company A."),
    )
    new = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="The user works at Company B."),
    )
    assert old.entry_id and new.entry_id
    await plane.apply_memory_supersession(
        identity,
        scope,
        MemorySupersessionIntent(
            superseded_memory_id=old.entry_id,
            superseding_memory_id=new.entry_id,
            reason="employment update",
        ),
    )
    profile = await manager.get_profile(_USER)
    by_id = {entry.entry_id: entry for entry in profile.memory_entries}
    assert by_id[old.entry_id].lineage.superseded_by_memory_id == new.entry_id
    assert by_id[new.entry_id].lineage.supersedes_memory_id == old.entry_id


async def run_user_10_supersession_recall_prefers_new(ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    old = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="The user works at Company A.", title="work"),
    )
    new = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="The user works at Company B.", title="work"),
    )
    assert old.entry_id and new.entry_id
    await plane.apply_memory_supersession(
        identity,
        scope,
        MemorySupersessionIntent(
            superseded_memory_id=old.entry_id,
            superseding_memory_id=new.entry_id,
            reason="employment update",
        ),
    )
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="work", top_k=5),
    )
    assert_superseded_not_current_winner(
        recall.items,
        superseded_entry_id=old.entry_id,
        superseding_entry_id=new.entry_id,
        ledger=ctx.ledger,
    )


async def run_user_11_self_supersession_rejected(_ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    entry = await plane.remember(identity, scope, MemoryControlRememberRequest(content="x"))
    assert entry.entry_id
    try:
        await plane.apply_memory_supersession(
            identity,
            scope,
            MemorySupersessionIntent(
                superseded_memory_id=entry.entry_id,
                superseding_memory_id=entry.entry_id,
                reason="invalid",
            ),
        )
    except MemoryControlBackendError:
        return
    raise AssertionError("expected MemoryControlBackendError for self-supersession")


async def run_user_12_supersession_missing_target(_ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    only = await plane.remember(identity, scope, MemoryControlRememberRequest(content="one"))
    assert only.entry_id
    try:
        await plane.apply_memory_supersession(
            identity,
            scope,
            MemorySupersessionIntent(
                superseded_memory_id=only.entry_id,
                superseding_memory_id="missing-id",
                reason="invalid",
            ),
        )
    except MemoryControlBackendError:
        return
    raise AssertionError("expected MemoryControlBackendError for missing superseding target")


async def run_user_16_provenance_preserved(_ctx: BehaviorEvalContext) -> None:
    plane, manager = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    provenance = MemoryProvenance(
        source_type=MemoryRecordSourceType.USER_EXPLICIT,
        actor_user_id=_USER,
    )
    trust = MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT)
    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(
            content="provenance fact",
            provenance=provenance,
            trust=trust,
        ),
    )
    profile = await manager.get_profile(_USER)
    stored = next(e for e in profile.memory_entries if e.entry_id == remembered.entry_id)
    assert stored.provenance.source_type is MemoryRecordSourceType.USER_EXPLICIT
    assert stored.trust.trust_class is MemoryTrustClass.USER_EXPLICIT


async def run_user_18_governance_deny_zero_side_effects(_ctx: BehaviorEvalContext) -> None:
    plane, manager = build_user_control_plane(governance=deny_remember_governance())
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    try:
        await plane.remember(identity, scope, MemoryControlRememberRequest(content="blocked"))
    except MemoryControlGovernanceDenied:
        profile = await manager.get_profile(_USER)
        assert profile.memory_entries == []
        return
    raise AssertionError("expected MemoryControlGovernanceDenied")


async def run_user_20_conflicting_facts_unresolved(_ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="User lives in Kraków.", title="city"),
    )
    await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="User lives in Warsaw.", title="city"),
    )
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="lives", top_k=5),
    )
    assert len(recall.items) >= 1
    assert_unresolved_conflict_evidence(recall.items)


async def run_user_21_supersession_resolves_conflict(ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    a = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="User lives in Kraków.", title="city"),
    )
    b = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="User lives in Warsaw.", title="city"),
    )
    assert a.entry_id and b.entry_id
    await plane.apply_memory_supersession(
        identity,
        scope,
        MemorySupersessionIntent(
            superseded_memory_id=a.entry_id,
            superseding_memory_id=b.entry_id,
            reason="move",
        ),
    )
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="lives", top_k=3),
    )
    assert_superseded_not_current_winner(
        recall.items,
        superseded_entry_id=a.entry_id,
        superseding_entry_id=b.entry_id,
        ledger=ctx.ledger,
    )


async def run_user_22_deterministic_ordering(_ctx: BehaviorEvalContext) -> None:
    plane, _ = build_user_control_plane()
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    for content in ("alpha", "beta", "gamma"):
        await plane.remember(identity, scope, MemoryControlRememberRequest(content=content))
    first = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=5))
    second = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=5))
    assert [item.entry_id for item in first.items] == [item.entry_id for item in second.items]
    assert first.reason == second.reason
