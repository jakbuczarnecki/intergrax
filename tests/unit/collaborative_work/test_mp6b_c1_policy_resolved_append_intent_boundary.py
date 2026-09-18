# © Artur Czarnecki. All rights reserved.

"""MP-6B-C1 — policy-resolved durability and validated append intent boundary gates."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

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


def _scope(tenant: str = "tenant-a", workspace: str = "ws-a") -> CollaborativeActivityScope:
    return CollaborativeActivityScope(
        tenant_id=tenant,
        workspace_id=workspace,
        work_item_id="wi-1",
    )


def _publication(
    *,
    requested: CollaborativeActivityDurabilityClass = CollaborativeActivityDurabilityClass.COLLABORATIVE,
    source_stable_id: str = "stable-1",
) -> CollaborativeActivityPublication:
    return CollaborativeActivityPublication(
        idempotency_key=ActivityIdempotencyKey(
            tenant_id="tenant-a",
            workspace_id="ws-a",
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id=source_stable_id,
            activity_type=CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
        ),
        actor=_actor(),
        scope=_scope(),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
        requested_durability_class=requested,
    )


def _intent(
    publication: CollaborativeActivityPublication | None = None,
    *,
    effective: CollaborativeActivityDurabilityClass,
) -> CollaborativeActivityAppendIntent:
    return CollaborativeActivityAppendIntent(
        publication=publication or _publication(),
        effective_durability_class=effective,
    )


class _ContractFakeCollaborativeActivityAppendStore:
    """Test-only store — materializes effective durability from intent only."""

    def __init__(self, *, recorded_at: datetime = _RECORDED) -> None:
        self._recorded_at = recorded_at
        self._by_key: dict[str, CollaborativeActivity] = {}

    def append_idempotent(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        publication = intent.publication
        activity_id = mint_collaborative_activity_id(idempotency_key=publication.idempotency_key)
        existing = self._by_key.get(activity_id)
        if existing is not None:
            return existing

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
            append_position=1,
            provenance_refs=publication.provenance_refs,
            correlation=publication.correlation,
            caused_by_activity_id=publication.caused_by_activity_id,
            durability_class=intent.effective_durability_class,
        )
        self._by_key[activity_id] = materialized
        return materialized


def test_mp6b_c1_append_store_input_is_append_intent() -> None:
    name, annotation = _append_idempotent_param()
    assert name == "intent"
    assert annotation == "CollaborativeActivityAppendIntent"


def test_mp6b_c1_append_intent_has_no_sequencing_fields() -> None:
    fields = _model_field_names("CollaborativeActivityAppendIntent")
    assert "append_position" not in fields
    assert "recorded_at" not in fields


def test_mp6b_c1_publication_retains_requested_durability_only() -> None:
    fields = _model_field_names("CollaborativeActivityPublication")
    assert "requested_durability_class" in fields
    assert "effective_durability_class" not in fields
    assert "durability_class" not in fields


def test_mp6b_c1_policy_override_materializes_effective_durability() -> None:
    pub = _publication(requested=CollaborativeActivityDurabilityClass.INFORMATIONAL)
    intent = _intent(pub, effective=CollaborativeActivityDurabilityClass.AUDIT_CRITICAL)
    store = _ContractFakeCollaborativeActivityAppendStore()
    activity = store.append_idempotent(intent)
    assert pub.requested_durability_class == CollaborativeActivityDurabilityClass.INFORMATIONAL
    assert activity.durability_class == CollaborativeActivityDurabilityClass.AUDIT_CRITICAL


def test_mp6b_c1_replay_does_not_mutate_effective_durability() -> None:
    pub = _publication(requested=CollaborativeActivityDurabilityClass.INFORMATIONAL)
    store = _ContractFakeCollaborativeActivityAppendStore()
    first = store.append_idempotent(
        _intent(pub, effective=CollaborativeActivityDurabilityClass.COLLABORATIVE)
    )
    replay = store.append_idempotent(
        _intent(pub, effective=CollaborativeActivityDurabilityClass.AUDIT_CRITICAL)
    )
    assert replay.activity_id == first.activity_id
    assert replay.durability_class == CollaborativeActivityDurabilityClass.COLLABORATIVE
    assert replay.durability_class != CollaborativeActivityDurabilityClass.AUDIT_CRITICAL


def test_mp6b_c1_append_intent_json_roundtrip() -> None:
    intent = _intent(
        _publication(requested=CollaborativeActivityDurabilityClass.INFORMATIONAL),
        effective=CollaborativeActivityDurabilityClass.AUDIT_CRITICAL,
    )
    restored = CollaborativeActivityAppendIntent.model_validate_json(intent.model_dump_json())
    assert restored == intent
    payload = json.loads(intent.model_dump_json())
    assert payload["effective_durability_class"] == "audit_critical"
    assert (
        payload["publication"]["requested_durability_class"] == "informational"
    )


def test_mp6b_c1_append_intent_json_schema() -> None:
    CollaborativeActivityAppendIntent.model_json_schema()


def test_mp6b_c1_append_intent_extra_field_rejected() -> None:
    payload = _intent(
        _publication(),
        effective=CollaborativeActivityDurabilityClass.COLLABORATIVE,
    ).model_dump(mode="json")
    payload["policy_id"] = "future"
    with pytest.raises(ValidationError):
        CollaborativeActivityAppendIntent.model_validate(payload)


def test_mp6b_c1_append_intent_frozen_ast_gate() -> None:
    tree = ast.parse(_read_contract())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "CollaborativeActivityAppendIntent":
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "model_config":
                        assert "frozen=True" in ast.unparse(stmt.value)
                        return
    raise AssertionError("CollaborativeActivityAppendIntent frozen model_config not found")
