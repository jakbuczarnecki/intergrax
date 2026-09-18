# © Artur Czarnecki. All rights reserved.

"""MP-6B — runtime contract hardening for Collaborative Activity & Provenance."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_activity import (
    ActivityIdempotencyKey,
    CollaborativeActivityAppendIntent,
    CollaborativeActivityAppendStore,
    CollaborativeActivityDurabilityClass,
    CollaborativeActivityPublicationPort,
    CollaborativeActivityReadPort,
    CollaborativeActivityTargetRef,
    CollaborativeActivityWritePort,
    ApprovalActivityProvenanceRef,
    ApprovalActivityTargetRef,
    ArtifactVersionActivityProvenanceRef,
    AssignmentActivityTargetRef,
    CollaborativeActivity,
    CollaborativeActivityActorRef,
    CollaborativeActivityBuiltinSource,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityCorrelation,
    CollaborativeActivityOutcome,
    CollaborativeActivityOutcomeStatus,
    CollaborativeActivityPage,
    CollaborativeActivityPageCursor,
    CollaborativeActivityPublication,
    CollaborativeActivityQuery,
    CollaborativeActivityRecordTargetRef,
    CollaborativeActivityScope,
    CollaborativeActivitySourceId,
    CollaborativeActivityTypeId,
    CollaborativeDecisionBindingActivityTargetRef,
    ContextViewActivityProvenanceRef,
    ContextViewActivityTargetRef,
    DecisionActivityProvenanceRef,
    DecisionActivityTargetRef,
    ExecutionActivityProvenanceRef,
    GovernanceEvidenceActivityProvenanceRef,
    ProofReceiptActivityProvenanceRef,
    WorkArtifactActivityTargetRef,
    WorkArtifactVersionActivityTargetRef,
    WorkItemActivityTargetRef,
    _ACTIVITY_ID_HASH_SCHEME,
    _activity_id_hash_material,
    mint_collaborative_activity_id,
)
from intergrax.contracts.collaborative_work import PrincipalKind, WorkArtifactVersionRef
from intergrax.contracts.execution_provenance import (
    AttemptId,
    ExecutionId,
    ExecutionProvenanceRef,
    RunId,
    TaskId,
)
from intergrax.contracts.governed_proof import GovernanceEvidenceRef

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "collaborative_activity.py"
_MP6_TEST_DIR = Path(__file__).resolve().parent

_STALE_APPEND_STORE_SIGNATURE_EXEMPT = frozenset({"_WrongAppendStore"})

_NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
_RECORDED = datetime(2026, 9, 18, 13, 0, 0, tzinfo=timezone.utc)

_GOLDEN_KEY = ActivityIdempotencyKey(
    tenant_id="tenant-a",
    workspace_id="ws-a",
    source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
    source_stable_id="stable-1",
    activity_type=CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
)
_GOLDEN_ACTIVITY_ID = "cact_35fa0b88b3369040c0378fc68db25a2f"


def _actor(
    tenant: str = "tenant-a",
    kind: PrincipalKind = PrincipalKind.HUMAN,
    delegation_id: str | None = None,
    delegator_principal_id: str | None = None,
) -> CollaborativeActivityActorRef:
    return CollaborativeActivityActorRef(
        tenant_id=tenant,
        principal_id="principal-1",
        principal_kind=kind,
        delegation_id=delegation_id,
        delegator_principal_id=delegator_principal_id,
    )


def _scope(
    tenant: str = "tenant-a",
    workspace: str = "ws-a",
    work_item: str | None = "wi-1",
) -> CollaborativeActivityScope:
    return CollaborativeActivityScope(
        tenant_id=tenant,
        workspace_id=workspace,
        work_item_id=work_item,
    )


def _publication(
    *,
    activity_type: CollaborativeActivityTypeId | None = None,
    target: CollaborativeActivityTargetRef | None = None,
    caused_by: str | None = None,
) -> CollaborativeActivityPublication:
    at = activity_type or CollaborativeActivityBuiltinType.WORK_ITEM_CREATED
    return CollaborativeActivityPublication(
        idempotency_key=ActivityIdempotencyKey(
            tenant_id="tenant-a",
            workspace_id="ws-a",
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id="stable-1",
            activity_type=at,
        ),
        actor=_actor(),
        scope=_scope(),
        target=target or WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
        caused_by_activity_id=caused_by,
    )


def _materialized(
    pub: CollaborativeActivityPublication,
    *,
    position: int = 1,
    durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    ),
) -> CollaborativeActivity:
    activity_id = mint_collaborative_activity_id(idempotency_key=pub.idempotency_key)
    return CollaborativeActivity(
        activity_id=activity_id,
        idempotency_key=pub.idempotency_key,
        activity_type=pub.activity_type,
        actor=pub.actor,
        scope=pub.scope,
        target=pub.target,
        outcome=pub.outcome,
        occurred_at=pub.occurred_at,
        recorded_at=_RECORDED,
        append_position=position,
        provenance_refs=pub.provenance_refs,
        correlation=pub.correlation,
        caused_by_activity_id=pub.caused_by_activity_id,
        durability_class=durability_class,
    )


def _intent(
    publication: CollaborativeActivityPublication | None = None,
    *,
    effective: CollaborativeActivityDurabilityClass = CollaborativeActivityDurabilityClass.COLLABORATIVE,
) -> CollaborativeActivityAppendIntent:
    return CollaborativeActivityAppendIntent(
        publication=publication or _publication(),
        effective_durability_class=effective,
    )


def _protocol_method_param(class_name: str, method_name: str) -> tuple[str, str]:
    tree = ast.parse(_CONTRACT.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != method_name:
                continue
            args = item.args.args
            if not args:
                raise AssertionError(f"{class_name}.{method_name} has no parameters")
            param = args[1] if args[0].arg == "self" and len(args) > 1 else args[0]
            annotation = ast.unparse(param.annotation) if param.annotation else ""
            return param.arg, annotation
    raise AssertionError(f"{class_name}.{method_name} not found in contract")


def _assert_append_store_signature(store_type: type) -> None:
    """Python Protocol assignment alone does not verify parameter annotations at runtime."""
    method = store_type.append_idempotent
    signature = inspect.signature(method)
    params = list(signature.parameters.values())
    if len(params) != 2 or params[0].name != "self":
        raise AssertionError(
            f"{store_type.__name__}.append_idempotent must accept (self, intent)"
        )
    intent_param = params[1]
    if intent_param.name != "intent":
        raise AssertionError(
            f"{store_type.__name__}.append_idempotent parameter must be named intent, "
            f"got {intent_param.name!r}"
        )
    module = sys.modules[store_type.__module__]
    hints = get_type_hints(method, globalns=module.__dict__, localns=module.__dict__)
    if hints.get(intent_param.name) is not CollaborativeActivityAppendIntent:
        raise AssertionError(
            f"{store_type.__name__}.append_idempotent must accept "
            f"CollaborativeActivityAppendIntent, got {hints.get(intent_param.name)!r}"
        )


def _stale_append_store_implementations_in_mp6_tests() -> list[str]:
    offenders: list[str] = []
    for path in sorted(_MP6_TEST_DIR.glob("test_mp6*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name in _STALE_APPEND_STORE_SIGNATURE_EXEMPT:
                continue
            if "AppendStore" not in node.name:
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef) or item.name != "append_idempotent":
                    continue
                args = item.args.args
                if len(args) < 2:
                    continue
                param = args[1] if args[0].arg == "self" else args[0]
                annotation = ast.unparse(param.annotation) if param.annotation else ""
                if annotation == "CollaborativeActivityPublication":
                    offenders.append(f"{path.name}:{node.name}.{item.name}")
    return offenders


def test_mp6b_golden_activity_id() -> None:
    assert mint_collaborative_activity_id(idempotency_key=_GOLDEN_KEY) == _GOLDEN_ACTIVITY_ID


def test_mp6b_hash_material_includes_version_scheme() -> None:
    material = _activity_id_hash_material(idempotency_key=_GOLDEN_KEY)
    assert material.startswith(_ACTIVITY_ID_HASH_SCHEME)


def test_mp6b_type_and_source_roundtrip_json() -> None:
    type_id = CollaborativeActivityTypeId.for_extension("acme.corp", "signal.emitted")
    source_id = CollaborativeActivitySourceId.for_extension("acme.corp", "connector")
    assert CollaborativeActivityTypeId.model_validate_json(type_id.model_dump_json()) == type_id
    assert CollaborativeActivitySourceId.model_validate_json(source_id.model_dump_json()) == source_id


def test_mp6b_publication_json_roundtrip_without_wire_activity_type() -> None:
    pub = _publication()
    restored = CollaborativeActivityPublication.model_validate_json(pub.model_dump_json())
    assert restored.idempotency_key == pub.idempotency_key
    assert restored.activity_type == pub.activity_type
    assert "activity_type" not in json.loads(pub.model_dump_json())


@pytest.mark.parametrize(
    "kind",
    [
        PrincipalKind.HUMAN,
        PrincipalKind.AGENT,
        PrincipalKind.SERVICE,
        PrincipalKind.EXTERNAL_AGENT,
    ],
)
def test_mp6b_actor_principal_kinds(kind: PrincipalKind) -> None:
    actor = _actor(kind=kind)
    assert actor.principal_kind == kind


def test_mp6b_delegation_pairing_both_directions() -> None:
    CollaborativeActivityActorRef(
        tenant_id="t1",
        principal_id="p",
        principal_kind=PrincipalKind.AGENT,
        delegation_id="d1",
        delegator_principal_id="delegator",
    )
    with pytest.raises(ValidationError, match="delegation"):
        CollaborativeActivityActorRef(
            tenant_id="t1",
            principal_id="p",
            principal_kind=PrincipalKind.AGENT,
            delegation_id="d1",
        )


def test_mp6b_actor_tenant_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        CollaborativeActivityPublication(
            idempotency_key=_GOLDEN_KEY,
            actor=_actor(tenant="other-tenant"),
            scope=_scope(),
            target=WorkItemActivityTargetRef(work_item_id="wi-1"),
            outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
            occurred_at=_NOW,
        )


@pytest.mark.parametrize(
    "target",
    [
        WorkItemActivityTargetRef(work_item_id="wi-1"),
        AssignmentActivityTargetRef(assignment_id="asg-1", work_item_id="wi-1"),
        WorkArtifactActivityTargetRef(work_artifact_id="art-1", work_item_id="wi-1"),
        WorkArtifactVersionActivityTargetRef(
            version_ref=WorkArtifactVersionRef(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                work_item_id="wi-1",
                work_artifact_id="art-1",
                work_artifact_version_id="v1",
            )
        ),
        DecisionActivityTargetRef(decision_id="dec-1"),
        ApprovalActivityTargetRef(approval_id="apr-1"),
        ContextViewActivityTargetRef(view_id="view-1"),
        CollaborativeDecisionBindingActivityTargetRef(binding_id="bind-1"),
        CollaborativeActivityRecordTargetRef(activity_id=_GOLDEN_ACTIVITY_ID),
    ],
)
def test_mp6b_each_target_constructible(target: CollaborativeActivityTargetRef) -> None:
    _publication(target=target)


def test_mp6b_work_item_scope_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="work_item_id"):
        _publication(target=WorkItemActivityTargetRef(work_item_id="other-wi"))


def test_mp6b_artifact_version_target_scope_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        _publication(
            target=WorkArtifactVersionActivityTargetRef(
                version_ref=WorkArtifactVersionRef(
                    tenant_id="other",
                    workspace_id="ws-a",
                    work_item_id="wi-1",
                    work_artifact_id="art-1",
                    work_artifact_version_id="v1",
                )
            )
        )


def test_mp6b_provenance_constructible_and_no_payload_fields() -> None:
    refs = (
        ExecutionActivityProvenanceRef(
            execution=ExecutionProvenanceRef(
                task_id=TaskId("task_" + "a" * 32),
                run_id=RunId("run_" + "b" * 32),
                attempt_id=AttemptId("attempt_" + "c" * 32),
                execution_id=ExecutionId("exec_" + "d" * 32),
            )
        ),
        GovernanceEvidenceActivityProvenanceRef(
            evidence=GovernanceEvidenceRef(kind="hitl", evidence_id="ev-1"),
        ),
        ProofReceiptActivityProvenanceRef(proof_id="proof-1"),
        ContextViewActivityProvenanceRef(view_id="view-1"),
        DecisionActivityProvenanceRef(decision_id="dec-1"),
        ApprovalActivityProvenanceRef(approval_id="apr-1"),
        ArtifactVersionActivityProvenanceRef(
            version_ref=WorkArtifactVersionRef(
                tenant_id="tenant-a",
                workspace_id="ws-a",
                work_item_id="wi-1",
                work_artifact_id="art-1",
                work_artifact_version_id="v1",
            )
        ),
    )
    pub = CollaborativeActivityPublication(
        idempotency_key=_GOLDEN_KEY,
        actor=_actor(),
        scope=_scope(),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
        provenance_refs=refs,
    )
    assert len(pub.provenance_refs) == len(refs)


def test_mp6b_provenance_dedupe_canonical_order() -> None:
    a = ContextViewActivityProvenanceRef(view_id="view-1")
    b = DecisionActivityProvenanceRef(decision_id="dec-1")
    pub = CollaborativeActivityPublication(
        idempotency_key=_GOLDEN_KEY,
        actor=_actor(),
        scope=_scope(),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
        provenance_refs=(a, b, a),
    )
    assert len(pub.provenance_refs) == 2


def test_mp6b_provenance_artifact_scope_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="provenance artifact version"):
        CollaborativeActivityPublication(
            idempotency_key=_GOLDEN_KEY,
            actor=_actor(),
            scope=_scope(),
            target=WorkItemActivityTargetRef(work_item_id="wi-1"),
            outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
            occurred_at=_NOW,
            provenance_refs=(
                ArtifactVersionActivityProvenanceRef(
                    version_ref=WorkArtifactVersionRef(
                        tenant_id="other",
                        workspace_id="ws-a",
                        work_item_id="wi-1",
                        work_artifact_id="art-1",
                        work_artifact_version_id="v1",
                    )
                ),
            ),
        )


def test_mp6b_naive_occurred_at_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        CollaborativeActivityPublication(
            idempotency_key=_GOLDEN_KEY,
            actor=_actor(),
            scope=_scope(),
            target=WorkItemActivityTargetRef(work_item_id="wi-1"),
            outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
            occurred_at=datetime(2026, 9, 18, 12, 0, 0),
        )


def test_mp6b_publication_has_no_recorded_at_or_append_position_fields() -> None:
    fields = set(CollaborativeActivityPublication.model_fields)
    assert "recorded_at" not in fields
    assert "append_position" not in fields


def test_mp6b_activity_self_caused_by_rejected() -> None:
    pub = _publication()
    activity_id = mint_collaborative_activity_id(idempotency_key=pub.idempotency_key)
    with pytest.raises(ValidationError, match="caused_by_activity_id"):
        CollaborativeActivity(
            activity_id=activity_id,
            idempotency_key=pub.idempotency_key,
            activity_type=pub.activity_type,
            actor=pub.actor,
            scope=pub.scope,
            target=pub.target,
            outcome=pub.outcome,
            occurred_at=pub.occurred_at,
            recorded_at=_RECORDED,
            append_position=1,
            caused_by_activity_id=activity_id,
        )


def test_mp6b_correction_requires_causal_or_activity_target() -> None:
    with pytest.raises(ValidationError, match="activity.correction"):
        _publication(activity_type=CollaborativeActivityBuiltinType.ACTIVITY_CORRECTION)
    corrected = _GOLDEN_ACTIVITY_ID
    _publication(
        activity_type=CollaborativeActivityBuiltinType.ACTIVITY_CORRECTION,
        caused_by=corrected,
    )
    _publication(
        activity_type=CollaborativeActivityBuiltinType.ACTIVITY_CORRECTION,
        target=CollaborativeActivityRecordTargetRef(activity_id=corrected),
        caused_by=None,
    )


def test_mp6b_query_invalid_time_range_rejected() -> None:
    with pytest.raises(ValidationError, match="occurred_after"):
        CollaborativeActivityQuery(
            tenant_id="t1",
            workspace_id="w1",
            occurred_after=_NOW,
            occurred_before=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_mp6b_query_dedupes_activity_types() -> None:
    builtin = CollaborativeActivityBuiltinType.WORK_ITEM_CREATED
    plugin = CollaborativeActivityTypeId.for_extension("acme", "signal")
    query = CollaborativeActivityQuery(
        tenant_id="t1",
        workspace_id="w1",
        activity_types=(builtin, plugin, builtin),
    )
    assert len(query.activity_types) == 2


def test_mp6b_page_and_correlation_schema_versions() -> None:
    page = CollaborativeActivityPage()
    corr = CollaborativeActivityCorrelation(run_id="run-1")
    assert page.schema_version == "collaborative_activity_page.v1"
    assert corr.schema_version == "collaborative_activity_correlation.v1"


def test_mp6b_json_schema_publication_and_activity() -> None:
    CollaborativeActivityPublication.model_json_schema()
    CollaborativeActivity.model_json_schema()


def test_mp6b_publication_requested_durability_not_effective_field() -> None:
    assert "requested_durability_class" in CollaborativeActivityPublication.model_fields
    assert "durability_class" not in CollaborativeActivityPublication.model_fields


def test_mp6b_empty_correlation_object_rejected() -> None:
    with pytest.raises(ValidationError, match="correlation must include"):
        CollaborativeActivityCorrelation()


def test_mp6b_unknown_target_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        CollaborativeActivityPublication.model_validate(
            {
                "schema_version": "collaborative_activity_publication.v1",
                "idempotency_key": _GOLDEN_KEY.model_dump(mode="json"),
                "actor": _actor().model_dump(mode="json"),
                "scope": _scope().model_dump(mode="json"),
                "target": {"kind": "unknown_kind", "work_item_id": "wi-1"},
                "outcome": {
                    "schema_version": "collaborative_activity_outcome.v1",
                    "status": "succeeded",
                },
                "occurred_at": _NOW.isoformat(),
            }
        )


def test_mp6b_publication_extra_field_rejected() -> None:
    payload = _publication().model_dump(mode="json")
    payload["unexpected_field"] = "x"
    with pytest.raises(ValidationError):
        CollaborativeActivityPublication.model_validate(payload)


class _Mp6bFakePublicationPort:
    def publish(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
        return _materialized(publication)


class _Mp6bFakeWritePort:
    def append(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
        return _materialized(publication)


class _Mp6bFakeReadPort:
    def query(self, query: CollaborativeActivityQuery) -> CollaborativeActivityPage:
        return CollaborativeActivityPage()


class _Mp6bFakeAppendStore:
    def append_idempotent(
        self,
        intent: CollaborativeActivityAppendIntent,
    ) -> CollaborativeActivity:
        publication = intent.publication
        return _materialized(
            publication,
            durability_class=intent.effective_durability_class,
        )

    def get_by_idempotency_key(
        self,
        key: ActivityIdempotencyKey,
    ) -> CollaborativeActivity | None:
        return None


class _WrongAppendStore:
    def append_idempotent(
        self,
        publication: CollaborativeActivityPublication,
    ) -> CollaborativeActivity:
        return _materialized(publication)


def test_mp6b_c1_r1_contract_port_boundary_matrix() -> None:
    matrix = (
        ("CollaborativeActivityPublicationPort", "publish", "publication", "CollaborativeActivityPublication"),
        ("CollaborativeActivityWritePort", "append", "publication", "CollaborativeActivityPublication"),
        ("CollaborativeActivityAppendStore", "append_idempotent", "intent", "CollaborativeActivityAppendIntent"),
        ("CollaborativeActivityReadPort", "query", "query", "CollaborativeActivityQuery"),
    )
    for class_name, method_name, param_name, annotation in matrix:
        name, parsed = _protocol_method_param(class_name, method_name)
        assert name == param_name, class_name
        assert parsed == annotation, class_name


def test_mp6b_c1_r1_no_stale_append_store_publication_signatures_in_mp6_tests() -> None:
    offenders = _stale_append_store_implementations_in_mp6_tests()
    assert not offenders, f"stale append_idempotent(publication) implementations: {offenders}"


def test_mp6b_c1_r1_append_store_signature_gate_accepts_current_fake() -> None:
    _assert_append_store_signature(_Mp6bFakeAppendStore)


def test_mp6b_c1_r1_append_store_signature_gate_rejects_stale_publication_input() -> None:
    with pytest.raises(AssertionError, match="intent"):
        _assert_append_store_signature(_WrongAppendStore)


def test_mp6b_pluginability_custom_ports_satisfy_protocols() -> None:
    publication_port: CollaborativeActivityPublicationPort = _Mp6bFakePublicationPort()
    write_port: CollaborativeActivityWritePort = _Mp6bFakeWritePort()
    read_port: CollaborativeActivityReadPort = _Mp6bFakeReadPort()
    append_store: CollaborativeActivityAppendStore = _Mp6bFakeAppendStore()
    _assert_append_store_signature(_Mp6bFakeAppendStore)
    pub = _publication()
    assert publication_port.publish(pub).activity_id == mint_collaborative_activity_id(
        idempotency_key=pub.idempotency_key
    )
    assert write_port.append(pub).append_position == 1
    assert read_port.query(
        CollaborativeActivityQuery(tenant_id="tenant-a", workspace_id="ws-a")
    ).activities == ()
    pub = _publication().model_copy(
        update={"requested_durability_class": CollaborativeActivityDurabilityClass.INFORMATIONAL}
    )
    intent = _intent(
        pub,
        effective=CollaborativeActivityDurabilityClass.AUDIT_CRITICAL,
    )
    activity = append_store.append_idempotent(intent)
    assert activity.recorded_at == _RECORDED
    assert pub.requested_durability_class == CollaborativeActivityDurabilityClass.INFORMATIONAL
    assert activity.durability_class == CollaborativeActivityDurabilityClass.AUDIT_CRITICAL
    assert append_store.get_by_idempotency_key(pub.idempotency_key) is None


def test_mp6b_contract_no_typing_escape_hatches() -> None:
    text = _CONTRACT.read_text(encoding="utf-8-sig")
    assert "type: ignore" not in text
    assert "pyright: ignore" not in text
    assert "from typing import Any" not in text
    assert " cast(" not in text


def test_mp6b_core_dto_frozen_ast_gate() -> None:
    tree = ast.parse(_CONTRACT.read_text(encoding="utf-8-sig"))
    frozen_models = {
        "CollaborativeActivity",
        "CollaborativeActivityPublication",
        "CollaborativeActivityPage",
        "CollaborativeActivityOutcome",
        "CollaborativeActivityCorrelation",
        "WorkItemActivityTargetRef",
        "ExecutionActivityProvenanceRef",
    }
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in frozen_models:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "model_config":
                        if "frozen=True" in ast.unparse(stmt.value):
                            found.add(node.name)
    assert found == frozen_models
