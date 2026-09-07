# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-3A — MP-3 WorkArtifact contract and invariant tests."""

from __future__ import annotations

import ast
import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_work import (
    SCHEMA_ARTIFACT_CONTENT_REF_V1,
    SCHEMA_WORK_ARTIFACT_V1,
    SCHEMA_WORK_ARTIFACT_VERSION_V1,
    ArtifactContentRef,
    Assignment,
    CollaborativeWorkArtifactInvariantError,
    WorkArtifact,
    WorkArtifactVersion,
    WorkItem,
    validate_work_artifact_current_version,
    validate_work_artifact_version_scope,
)
from intergrax.contracts.execution_evidence.receipt import ProofReceipt
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef
from intergrax.runtime.context_lifecycle.repository import OptimizationArtifactReference
from intergrax.runtime.context_lifecycle.contracts import OptimizationArtifactType
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_state import TaskState
from intergrax.runtime.task_memory.models import TaskMemoryRecord

_UTC = timezone.utc
_WARSAW = timezone(timedelta(hours=2))
_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=_UTC)
_LATER = _NOW + timedelta(minutes=5)
_DIGEST = "sha256:" + ("a" * 64)


def _content_ref(**overrides: object) -> ArtifactContentRef:
    payload = {
        "content_ref": "content://tenant-a/workspace-a/artifact-body-1",
        "media_type": "application/json",
        "integrity_digest": _DIGEST,
    }
    payload.update(overrides)
    return ArtifactContentRef.model_validate(payload)


def _work_artifact(**overrides: object) -> WorkArtifact:
    payload = {
        "work_artifact_id": "artifact-1",
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "work_item_id": "work-item-1",
        "created_by_principal_id": "principal-creator",
        "current_version_id": "artifact-version-1",
        "revision": 0,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkArtifact.model_validate(payload)


def _work_artifact_version(**overrides: object) -> WorkArtifactVersion:
    payload = {
        "work_artifact_version_id": "artifact-version-1",
        "work_artifact_id": "artifact-1",
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "work_item_id": "work-item-1",
        "created_by_principal_id": "principal-creator",
        "published_by_principal_id": "principal-publisher",
        "content_ref": _content_ref(),
        "created_at": _NOW,
        "published_at": _LATER,
        "execution": None,
    }
    payload.update(overrides)
    return WorkArtifactVersion.model_validate(payload)


def _execution(**overrides: object) -> ExecutionProvenanceRef:
    payload = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
    }
    payload.update(overrides)
    return ExecutionProvenanceRef(**payload)


@pytest.mark.unit
def test_artifact_content_ref_schema_version_is_stable() -> None:
    content_ref = _content_ref()
    assert content_ref.schema_version == SCHEMA_ARTIFACT_CONTENT_REF_V1


@pytest.mark.unit
def test_artifact_content_ref_is_frozen_and_rejects_extra_fields() -> None:
    content_ref = _content_ref()
    with pytest.raises(ValidationError):
        content_ref.content_ref = "other"
    with pytest.raises(ValidationError):
        _content_ref(provider="s3")


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["content_ref", "media_type"])
def test_artifact_content_ref_rejects_empty_required_fields(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _content_ref(**{field_name: "   "})


@pytest.mark.unit
def test_artifact_content_ref_integrity_digest_uses_platform_convention() -> None:
    assert _content_ref(integrity_digest=_DIGEST).integrity_digest == _DIGEST
    with pytest.raises(ValidationError):
        _content_ref(integrity_digest="md5:deadbeef")


@pytest.mark.unit
def test_artifact_content_ref_size_bytes_non_negative() -> None:
    assert _content_ref(size_bytes=0).size_bytes == 0
    with pytest.raises(ValidationError):
        _content_ref(size_bytes=-1)


@pytest.mark.unit
def test_work_artifact_schema_version_is_stable() -> None:
    artifact = _work_artifact()
    assert artifact.schema_version == SCHEMA_WORK_ARTIFACT_V1


@pytest.mark.unit
def test_work_artifact_is_frozen_and_rejects_extra_fields() -> None:
    artifact = _work_artifact()
    with pytest.raises(ValidationError):
        artifact.revision = 1
    with pytest.raises(ValidationError):
        _work_artifact(status="draft")


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    [
        "work_artifact_id",
        "tenant_id",
        "workspace_id",
        "work_item_id",
        "created_by_principal_id",
        "current_version_id",
    ],
)
def test_work_artifact_rejects_empty_ids(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _work_artifact(**{field_name: "   "})


@pytest.mark.unit
def test_work_artifact_revision_and_timestamp_validation() -> None:
    assert _work_artifact(revision=3).revision == 3
    with pytest.raises(ValidationError):
        _work_artifact(revision=-1)
    with pytest.raises(ValidationError, match="timezone-aware"):
        _work_artifact(created_at=datetime(2026, 9, 7, 12, 0))
    with pytest.raises(ValidationError, match="updated_at must be greater"):
        _work_artifact(updated_at=_NOW - timedelta(seconds=1))


@pytest.mark.unit
def test_work_artifact_has_no_embedded_version_or_payload_fields() -> None:
    field_names = set(WorkArtifact.model_fields)
    forbidden = {
        "versions",
        "content",
        "payload",
        "metadata",
        "status",
        "approval_status",
        "review_state",
        "activity",
        "decision",
    }
    assert forbidden.isdisjoint(field_names)


@pytest.mark.unit
def test_work_artifact_version_schema_version_is_stable() -> None:
    version = _work_artifact_version()
    assert version.schema_version == SCHEMA_WORK_ARTIFACT_VERSION_V1


@pytest.mark.unit
def test_work_artifact_version_is_frozen_and_rejects_extra_fields() -> None:
    version = _work_artifact_version()
    with pytest.raises(ValidationError):
        version.published_by_principal_id = "other"
    with pytest.raises(ValidationError):
        _work_artifact_version(revision=1)


@pytest.mark.unit
def test_work_artifact_version_requires_principal_provenance() -> None:
    version = _work_artifact_version()
    assert version.created_by_principal_id == "principal-creator"
    assert version.published_by_principal_id == "principal-publisher"
    with pytest.raises(ValidationError):
        _work_artifact_version(created_by_principal_id="   ")


@pytest.mark.unit
def test_work_artifact_version_timestamp_validation() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _work_artifact_version(published_at=datetime(2026, 9, 7, 12, 5))
    with pytest.raises(ValidationError, match="published_at must be greater"):
        _work_artifact_version(published_at=_NOW - timedelta(seconds=1))
    assert _work_artifact_version(
        created_at=_NOW.replace(tzinfo=_WARSAW),
        published_at=_LATER.replace(tzinfo=_WARSAW),
    ).created_at.tzinfo is not None


@pytest.mark.unit
def test_work_artifact_version_has_no_mutable_status_fields() -> None:
    field_names = set(WorkArtifactVersion.model_fields)
    forbidden = {
        "revision",
        "state",
        "status",
        "updated_at",
        "approved_by",
        "review_status",
        "decision_id",
        "activity",
        "metadata",
        "payload",
    }
    assert forbidden.isdisjoint(field_names)


@pytest.mark.unit
def test_work_artifact_version_execution_optional_human_created() -> None:
    version = _work_artifact_version(execution=None)
    assert version.execution is None


@pytest.mark.unit
def test_work_artifact_version_execution_created_path() -> None:
    provenance = _execution()
    version = _work_artifact_version(execution=provenance)
    assert version.execution is provenance


@pytest.mark.unit
def test_work_artifact_version_execution_rejects_invalid_type() -> None:
    with pytest.raises(ValidationError):
        WorkArtifactVersion.model_validate(
            {
                **_work_artifact_version().model_dump(mode="python"),
                "execution": {"task_id": "bad"},
            },
        )
    with pytest.raises(ValidationError):
        WorkArtifactVersion.model_validate(
            {
                **_work_artifact_version().model_dump(mode="python"),
                "execution": "not-provenance",
            },
        )


@pytest.mark.unit
def test_validate_work_artifact_version_scope_accepts_matching_records() -> None:
    artifact = _work_artifact()
    version = _work_artifact_version()
    validate_work_artifact_version_scope(artifact=artifact, version=version)


@pytest.mark.unit
def test_validate_work_artifact_current_version_accepts_matching_pointer() -> None:
    artifact = _work_artifact()
    version = _work_artifact_version()
    validate_work_artifact_current_version(artifact=artifact, version=version)


@pytest.mark.unit
def test_validate_work_artifact_version_scope_rejects_wrong_tenant() -> None:
    artifact = _work_artifact()
    version = _work_artifact_version(tenant_id="tenant-other")
    with pytest.raises(CollaborativeWorkArtifactInvariantError, match="tenant_id"):
        validate_work_artifact_version_scope(artifact=artifact, version=version)


@pytest.mark.unit
def test_validate_work_artifact_version_scope_rejects_wrong_workspace() -> None:
    artifact = _work_artifact()
    version = _work_artifact_version(workspace_id="workspace-other")
    with pytest.raises(CollaborativeWorkArtifactInvariantError, match="workspace_id"):
        validate_work_artifact_version_scope(artifact=artifact, version=version)


@pytest.mark.unit
def test_validate_work_artifact_version_scope_rejects_wrong_work_item() -> None:
    artifact = _work_artifact()
    version = _work_artifact_version(work_item_id="work-item-other")
    with pytest.raises(CollaborativeWorkArtifactInvariantError, match="work_item_id"):
        validate_work_artifact_version_scope(artifact=artifact, version=version)


@pytest.mark.unit
def test_validate_work_artifact_version_scope_rejects_wrong_artifact_id() -> None:
    artifact = _work_artifact()
    version = _work_artifact_version(work_artifact_id="artifact-other")
    with pytest.raises(CollaborativeWorkArtifactInvariantError, match="work_artifact_id"):
        validate_work_artifact_version_scope(artifact=artifact, version=version)


@pytest.mark.unit
def test_validate_work_artifact_current_version_rejects_wrong_pointer() -> None:
    artifact = _work_artifact(current_version_id="artifact-version-other")
    version = _work_artifact_version()
    with pytest.raises(CollaborativeWorkArtifactInvariantError, match="current_version_id"):
        validate_work_artifact_current_version(artifact=artifact, version=version)


@pytest.mark.unit
def test_work_artifact_immutability_runtime() -> None:
    artifact = _work_artifact()
    with pytest.raises(ValidationError):
        artifact.current_version_id = "artifact-version-2"
    with pytest.raises(ValidationError):
        artifact.revision = 2


@pytest.mark.unit
def test_work_artifact_version_immutability_runtime() -> None:
    version = _work_artifact_version(execution=_execution())
    with pytest.raises(ValidationError):
        version.content_ref = _content_ref(content_ref="content://other")
    with pytest.raises(ValidationError):
        version.execution = _execution()


@pytest.mark.unit
def test_work_artifact_round_trip() -> None:
    artifact = _work_artifact()
    reloaded = WorkArtifact.model_validate(artifact.model_dump(mode="python"))
    assert reloaded == artifact


@pytest.mark.unit
def test_work_artifact_version_round_trip_without_execution() -> None:
    version = _work_artifact_version(execution=None)
    reloaded = WorkArtifactVersion.model_validate(version.model_dump(mode="python"))
    assert reloaded == version


@pytest.mark.unit
def test_work_artifact_version_round_trip_with_execution() -> None:
    provenance = _execution()
    version = _work_artifact_version(execution=provenance)
    dumped = version.model_dump(mode="python")
    reloaded = WorkArtifactVersion.model_validate(dumped)
    assert reloaded.execution == provenance


@pytest.mark.unit
def test_work_artifact_is_not_work_item() -> None:
    assert WorkArtifact is not WorkItem
    assert not issubclass(WorkArtifact, WorkItem)


@pytest.mark.unit
def test_work_artifact_version_is_not_work_item() -> None:
    assert WorkArtifactVersion is not WorkItem
    assert not issubclass(WorkArtifactVersion, WorkItem)


@pytest.mark.unit
def test_work_artifact_is_not_task() -> None:
    assert WorkArtifact is not Task
    assert not issubclass(WorkArtifact, Task)


@pytest.mark.unit
def test_work_artifact_version_is_not_task() -> None:
    assert WorkArtifactVersion is not Task
    assert not issubclass(WorkArtifactVersion, Task)


@pytest.mark.unit
def test_work_artifact_is_not_optimization_artifact_reference() -> None:
    assert WorkArtifact is not OptimizationArtifactReference
    assert WorkArtifactVersion is not OptimizationArtifactReference


@pytest.mark.unit
def test_work_artifact_is_not_proof_receipt() -> None:
    assert WorkArtifact is not ProofReceipt
    assert WorkArtifactVersion is not ProofReceipt


@pytest.mark.unit
def test_work_artifact_is_not_task_memory_record() -> None:
    assert WorkArtifact is not TaskMemoryRecord
    assert WorkArtifactVersion is not TaskMemoryRecord


@pytest.mark.unit
def test_work_artifact_is_not_assignment() -> None:
    assert WorkArtifact is not Assignment
    assert WorkArtifactVersion is not Assignment


@pytest.mark.unit
def test_work_artifact_has_no_execution_task_state_fields() -> None:
    artifact_fields = set(WorkArtifact.model_fields)
    version_fields = set(WorkArtifactVersion.model_fields)
    forbidden = {"task_id", "run_id", "attempt_id", "state", "owner_id", "user_id", "agent_id"}
    assert forbidden.isdisjoint(artifact_fields)
    assert forbidden.isdisjoint(version_fields)
    assert {state.value for state in TaskState}.isdisjoint(
        {name for name in artifact_fields.union(version_fields)},
    )


@pytest.mark.unit
def test_optimization_artifact_reference_shape_differs() -> None:
    reference = OptimizationArtifactReference(
        tenant_id="tenant-1",
        artifact_id="artifact-1",
        artifact_lookup_key_hash="hash-lookup",
        artifact_content_hash="hash-content",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
    )
    artifact = _work_artifact()
    assert type(reference) is not type(artifact)
    assert "work_item_id" in WorkArtifact.model_fields
    assert "work_item_id" not in OptimizationArtifactReference.__dataclass_fields__


@pytest.mark.unit
def test_collaborative_work_contract_module_has_no_storage_imports() -> None:
    module = importlib.import_module("intergrax.contracts.collaborative_work")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported)
    assert "boto" not in joined
    assert "sqlite" not in joined
    assert "collaborative_work.repository" not in joined
    assert "collaborative_work.service" not in joined
