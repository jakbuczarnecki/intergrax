# © Artur Czarnecki. All rights reserved.

"""MP-6A-C1-R1 — atomic append position ownership and materialization boundary gates."""

from __future__ import annotations

import ast
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.collaborative_activity import (
    ActivityIdempotencyKey,
    CollaborativeActivity,
    CollaborativeActivityActorRef,
    CollaborativeActivityAppendIntent,
    CollaborativeActivityBuiltinSource,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityDurabilityClass,
    CollaborativeActivityOutcome,
    CollaborativeActivityOutcomeStatus,
    CollaborativeActivityPublication,
    CollaborativeActivityScope,
    WorkItemActivityTargetRef,
    mint_collaborative_activity_id,
)
from intergrax.contracts.collaborative_work import PrincipalKind

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "collaborative_activity.py"

_NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
_RECORDED = datetime(2026, 9, 18, 13, 0, 0, tzinfo=timezone.utc)


def _read_contract() -> str:
    return _CONTRACT.read_text(encoding="utf-8-sig")


def _model_field_names(class_name: str) -> set[str]:
    tree = ast.parse(_read_contract())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        names: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names.add(stmt.target.id)
        return names
    raise AssertionError(f"class not found: {class_name}")


def _append_idempotent_param() -> tuple[str, str]:
    tree = ast.parse(_read_contract())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "CollaborativeActivityAppendStore":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "append_idempotent":
                continue
            args = item.args.args
            if not args:
                raise AssertionError("append_idempotent has no parameters")
            param = args[1] if args[0].arg == "self" and len(args) > 1 else args[0]
            annotation = ast.unparse(param.annotation) if param.annotation else ""
            return param.arg, annotation
    raise AssertionError("CollaborativeActivityAppendStore.append_idempotent not found")


def _actor(tenant: str = "tenant-a") -> CollaborativeActivityActorRef:
    return CollaborativeActivityActorRef(
        tenant_id=tenant,
        principal_id="principal-1",
        principal_kind=PrincipalKind.HUMAN,
    )


def _scope(
    tenant: str = "tenant-a",
    workspace: str = "ws-a",
    work_item: str = "wi-1",
) -> CollaborativeActivityScope:
    return CollaborativeActivityScope(
        tenant_id=tenant,
        workspace_id=workspace,
        work_item_id=work_item,
    )


def _key(
    *,
    tenant: str = "tenant-a",
    workspace: str = "ws-a",
    source_stable_id: str = "stable-1",
) -> ActivityIdempotencyKey:
    return ActivityIdempotencyKey(
        tenant_id=tenant,
        workspace_id=workspace,
        source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
        source_stable_id=source_stable_id,
        activity_type=CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
    )


def _publication(
    *,
    tenant: str = "tenant-a",
    workspace: str = "ws-a",
    source_stable_id: str = "stable-1",
) -> CollaborativeActivityPublication:
    return CollaborativeActivityPublication(
        idempotency_key=_key(
            tenant=tenant,
            workspace=workspace,
            source_stable_id=source_stable_id,
        ),
        actor=_actor(tenant),
        scope=_scope(tenant, workspace),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
    )


def _intent(publication: CollaborativeActivityPublication | None = None) -> CollaborativeActivityAppendIntent:
    pub = publication or _publication()
    return CollaborativeActivityAppendIntent(
        publication=pub,
        effective_durability_class=CollaborativeActivityDurabilityClass.COLLABORATIVE,
    )


class _ContractFakeCollaborativeActivityAppendStore:
    """Test-only store proving MP-6A-C1-R1 append semantics (not a production implementation)."""

    def __init__(self, *, recorded_at: datetime = _RECORDED) -> None:
        self._recorded_at = recorded_at
        self._by_key: dict[str, CollaborativeActivity] = {}
        self._next_position: dict[tuple[str, str], int] = defaultdict(lambda: 1)

    def append_idempotent(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        publication = intent.publication
        activity_id = mint_collaborative_activity_id(idempotency_key=publication.idempotency_key)
        existing = self._by_key.get(activity_id)
        if existing is not None:
            return existing

        workspace_key = (
            publication.scope.tenant_id,
            publication.scope.workspace_id,
        )
        position = self._next_position[workspace_key]
        self._next_position[workspace_key] = position + 1

        materialized = CollaborativeActivity(
            activity_id=activity_id,
            idempotency_key=publication.idempotency_key,
            activity_type=publication.activity_type,
            actor=publication.actor,
            scope=publication.scope,
            target=publication.target,
            outcome=publication.outcome,
            occurred_at=publication.occurred_at,
            recorded_at=self._recorded_at,
            append_position=position,
            provenance_refs=publication.provenance_refs,
            correlation=publication.correlation,
            caused_by_activity_id=publication.caused_by_activity_id,
            durability_class=intent.effective_durability_class,
        )
        self._by_key[activity_id] = materialized
        return materialized


def test_mp6a_c1_r1_publication_has_no_append_position() -> None:
    fields = _model_field_names("CollaborativeActivityPublication")
    assert "append_position" not in fields


def test_mp6a_c1_r1_publication_has_no_recorded_at() -> None:
    fields = _model_field_names("CollaborativeActivityPublication")
    assert "recorded_at" not in fields


def test_mp6a_c1_r1_materialized_activity_has_append_position_and_recorded_at() -> None:
    fields = _model_field_names("CollaborativeActivity")
    assert "append_position" in fields
    assert "recorded_at" in fields


def test_mp6a_c1_r1_append_store_input_is_append_intent_not_materialized_activity() -> None:
    name, annotation = _append_idempotent_param()
    assert name == "intent"
    assert annotation == "CollaborativeActivityAppendIntent"


def test_mp6a_c1_r1_new_publication_materializes_with_position() -> None:
    store = _ContractFakeCollaborativeActivityAppendStore()
    activity = store.append_idempotent(_intent(_publication()))
    assert activity.append_position == 1
    assert activity.recorded_at == _RECORDED


def test_mp6a_c1_r1_duplicate_replay_same_identity_and_position() -> None:
    store = _ContractFakeCollaborativeActivityAppendStore()
    pub = _publication()
    first = store.append_idempotent(_intent(pub))
    second = store.append_idempotent(_intent(pub))
    assert second.activity_id == first.activity_id
    assert second.append_position == first.append_position
    assert second.recorded_at == first.recorded_at


def test_mp6a_c1_r1_distinct_keys_monotonic_positions_same_workspace() -> None:
    store = _ContractFakeCollaborativeActivityAppendStore()
    a = store.append_idempotent(_intent(_publication(source_stable_id="a")))
    b = store.append_idempotent(_intent(_publication(source_stable_id="b")))
    assert a.append_position < b.append_position
    assert a.append_position == 1
    assert b.append_position == 2


def test_mp6a_c1_r1_cross_workspace_independent_position_domains() -> None:
    store = _ContractFakeCollaborativeActivityAppendStore()
    ws_a = store.append_idempotent(_intent(_publication(tenant="t1", workspace="ws-a")))
    ws_b = store.append_idempotent(_intent(_publication(tenant="t1", workspace="ws-b")))
    assert ws_a.append_position == 1
    assert ws_b.append_position == 1
