# © Artur Czarnecki. All rights reserved.

"""PDC-D8-R002 (mandatory digest, mismatch rejected) and R004 (lineage reconstructable)."""

from __future__ import annotations

from datetime import datetime

import pytest
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactDigestMismatchError,
    ArtifactDigestRequiredError,
    ArtifactLineage,
    ArtifactLineageError,
    ArtifactParent,
    ArtifactProvenance,
    ArtifactRef,
    ArtifactRetentionClass,
    ArtifactScope,
    DurableArtifactRef,
    artifact_ref_from_payload,
    compute_artifact_digest,
    reconstruct_artifact_lineage,
    verify_artifact_digest,
)
from intergrax.runtime.nexus.artifacts.models import Artifact
from pydantic import ValidationError

from ._d8_support import T0, make_artifact

pytestmark = pytest.mark.unit

BODY = b"quarterly report body"
DIGEST = compute_artifact_digest(BODY)


def _durable(
    artifact_id: str = "a1",
    *,
    digest: str = DIGEST,
    parents: tuple[ArtifactParent, ...] = (),
    version: int = 1,
) -> DurableArtifactRef:
    return DurableArtifactRef(
        artifact_id=artifact_id,
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
        type="report",
        uri=f"artifact://tenant-a/{artifact_id}",
        content_digest=digest,
        size_bytes=len(BODY),
        version=version,
        scope=ArtifactScope(tenant_id="tenant-a", workspace_id="ws-1"),
        producer=ArtifactProvenance(created_by_agent_id="agent-1"),
        lineage=ArtifactLineage(run_id="run-1", task_id="task-1", parent_artifacts=parents),
        retention_class=ArtifactRetentionClass.DEFAULT,
        created_at=T0,
    )


def test_digest_is_mandatory_on_a_durable_ref() -> None:
    with pytest.raises(ValidationError) as exc:
        DurableArtifactRef.model_validate(
            _durable().model_dump(exclude={"content_digest"}),
        )
    assert "content_digest" in str(exc.value)


@pytest.mark.parametrize("bad", ["", "   ", "abc", "sha256:xyz", "md5:" + "a" * 32])
def test_malformed_or_empty_digest_is_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        _durable(digest=bad)


def test_digest_mismatch_is_rejected() -> None:
    ref = _durable()
    ref.verify_content(BODY)
    with pytest.raises(ArtifactDigestMismatchError):
        ref.verify_content(b"tampered body")
    with pytest.raises(ArtifactDigestMismatchError):
        verify_artifact_digest(b"tampered body", DIGEST)


def test_artifact_with_wrong_digest_cannot_be_constructed() -> None:
    with pytest.raises(ArtifactDigestMismatchError):
        Artifact(
            tenant_id="t",
            artifact_id="a",
            run_id="r",
            step_id=None,
            kind="k",
            mime_type="text/plain",
            created_at_utc=T0,
            data=b"real",
            size_bytes=4,
            content_digest=compute_artifact_digest(b"other"),
        )


def test_artifact_digest_is_derived_never_empty() -> None:
    artifact = make_artifact(data=BODY)
    assert artifact.content_digest == DIGEST


def test_durable_ref_carries_producer_scope_lineage_and_retention() -> None:
    ref = make_artifact(data=BODY).to_durable_ref()
    assert ref.content_digest == DIGEST
    assert ref.producer.created_by_agent_id == "agent-1"
    assert ref.scope.tenant_id == "tenant-a" and ref.scope.workspace_id == "ws-1"
    assert ref.lineage.run_id == "run-1" and ref.lineage.task_id == "task-1"
    assert ref.retention_class is ArtifactRetentionClass.DEFAULT


def test_legacy_ref_migrates_only_with_content_or_explicit_digest() -> None:
    legacy = artifact_ref_from_payload(
        {"artifact_id": "legacy-1", "type": "report", "uri": "file:///x.pdf"},
        run_id="run-1",
        trace_id="trace-1",
        agent_id="agent-1",
    )
    assert isinstance(legacy, ArtifactRef)
    assert legacy.schema_version == "artifact_ref.v1"
    scope = ArtifactScope(tenant_id="tenant-a")
    lineage = ArtifactLineage(run_id="run-1")
    # No content and no digest: explicit typed error, never an empty digest.
    with pytest.raises(ArtifactDigestRequiredError):
        legacy.to_durable_ref(scope=scope, lineage=lineage)
    migrated = legacy.to_durable_ref(scope=scope, lineage=lineage, content=BODY)
    assert migrated.content_digest == DIGEST
    # A digest supplied next to the content must match it.
    with pytest.raises(ArtifactDigestMismatchError):
        legacy.to_durable_ref(scope=scope, lineage=lineage, content=BODY, content_digest=compute_artifact_digest(b"x"))


def test_legacy_checksum_in_platform_format_migrates_without_content() -> None:
    legacy = artifact_ref_from_payload(
        {"artifact_id": "legacy-2", "type": "report", "uri": "file:///y.pdf", "checksum": DIGEST},
        run_id="run-1",
        trace_id="trace-1",
        agent_id="agent-1",
    )
    migrated = legacy.to_durable_ref(
        scope=ArtifactScope(tenant_id="tenant-a"),
        lineage=ArtifactLineage(run_id="run-1"),
        size_bytes=len(BODY),
    )
    assert migrated.content_digest == DIGEST


def test_lineage_is_reconstructable_from_references() -> None:
    root = _durable("root", digest=compute_artifact_digest(b"root"))
    mid = _durable(
        "mid",
        digest=compute_artifact_digest(b"mid"),
        parents=(ArtifactParent(artifact_id="root", content_digest=root.content_digest),),
    )
    leaf = _durable(
        "leaf",
        digest=compute_artifact_digest(b"leaf"),
        parents=(ArtifactParent(artifact_id="mid", content_digest=mid.content_digest),),
    )
    chain = reconstruct_artifact_lineage("leaf", [leaf, root, mid])
    assert [ref.artifact_id for ref in chain] == ["root", "mid", "leaf"]


def test_lineage_with_missing_or_altered_parent_is_rejected() -> None:
    root = _durable("root", digest=compute_artifact_digest(b"root"))
    leaf = _durable(
        "leaf",
        digest=compute_artifact_digest(b"leaf"),
        parents=(ArtifactParent(artifact_id="root", content_digest=compute_artifact_digest(b"other")),),
    )
    with pytest.raises(ArtifactLineageError):
        reconstruct_artifact_lineage("leaf", [leaf, root])  # parent digest does not match any version
    with pytest.raises(ArtifactLineageError):
        reconstruct_artifact_lineage("leaf", [leaf])  # parent absent
    with pytest.raises(ArtifactLineageError):
        reconstruct_artifact_lineage("nope", [leaf, root])


def test_created_at_default_is_timezone_aware() -> None:
    ref = DurableArtifactRef(
        artifact_id="x",
        artifact_class=ArtifactClass.TOOL_OUTPUT,
        type="t",
        uri="artifact://t/x",
        content_digest=DIGEST,
        size_bytes=1,
        scope=ArtifactScope(tenant_id="t"),
        producer=ArtifactProvenance(created_by_agent_id="a"),
        lineage=ArtifactLineage(run_id="r"),
    )
    assert isinstance(ref.created_at, datetime) and ref.created_at.tzinfo is not None
