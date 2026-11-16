# © Artur Czarnecki. All rights reserved.

"""Typed artifact references on agent run results (architecture §40.6 · ACP-PROD-6).

``ArtifactRef`` (``artifact_ref.v1``) is the loose pointer carried on agent run results.
``DurableArtifactRef`` (``durable_artifact_ref.v1``) is the shared durable reference contract
of the artifact lifecycle (ADR-PDC-08): a mandatory content digest next to producer identity,
execution lineage, tenant/workspace scope and a retention class. A v1 pointer migrates to a
durable reference only through ``ArtifactRef.to_durable_ref``, which never invents a digest.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ArtifactSensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PII = "pii"


class ArtifactClass(StrEnum):
    """The four artifact classes; each has exactly one owner (see the runtime taxonomy gate)."""

    BUSINESS = "business"
    EXECUTION_EVIDENCE = "execution_evidence"
    CONTEXT = "context"
    TOOL_OUTPUT = "tool_output"


class ArtifactRetentionClass(StrEnum):
    """Retention classes of durable artifacts.

    ``AUDIT_EVIDENCE`` is never expirable and never deletable by TTL or compaction.
    Legal hold is a separate, revocable marker held by the owning store.
    """

    EPHEMERAL = "ephemeral"
    DEFAULT = "default"
    EXTENDED = "extended"
    AUDIT_EVIDENCE = "audit_evidence"


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ArtifactDigestError(ValueError):
    """Base class of artifact digest failures."""


class ArtifactDigestRequiredError(ArtifactDigestError):
    """A durable reference was requested but no content digest is known (never defaulted)."""


class ArtifactDigestMismatchError(ArtifactDigestError):
    """Content does not match the digest recorded on its reference."""


class ArtifactLineageError(ValueError):
    """Lineage cannot be reconstructed from the available references."""


def compute_artifact_digest(content: bytes) -> str:
    """Return the platform ``sha256:<hex>`` digest of ``content``."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def is_valid_artifact_digest(value: str) -> bool:
    return bool(_DIGEST_RE.match(value))


def verify_artifact_digest(content: bytes, expected_digest: str) -> None:
    """Raise ``ArtifactDigestMismatchError`` unless ``content`` hashes to ``expected_digest``."""
    if compute_artifact_digest(content) != expected_digest:
        raise ArtifactDigestMismatchError("artifact content does not match its recorded digest")


class ArtifactProvenance(BaseModel):
    """Producer identity of an artifact."""

    model_config = ConfigDict(extra="forbid")

    created_by_agent_id: str
    created_by_tool_id: str | None = None
    source_side_effect_id: str | None = None


class ArtifactScope(BaseModel):
    """Tenant and optional workspace scope of an artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1)
    workspace_id: str | None = None


class ArtifactParent(BaseModel):
    """Content-addressed pointer to a parent artifact (identity plus digest)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(min_length=1)
    content_digest: str

    @field_validator("content_digest")
    @classmethod
    def _digest_format(cls, value: str) -> str:
        if not is_valid_artifact_digest(value):
            raise ValueError("content_digest must match sha256:<64 lowercase hex>")
        return value


class ArtifactLineage(BaseModel):
    """Execution lineage of an artifact: where it was produced and what it derives from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    task_id: str | None = None
    attempt_id: str | None = None
    execution_id: str | None = None
    trace_id: str = ""
    step_index: int | None = None
    parent_artifacts: tuple[ArtifactParent, ...] = ()


class DurableArtifactRef(BaseModel):
    """Shared durable artifact reference (ADR-PDC-08): the digest is mandatory, never defaulted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["durable_artifact_ref.v1"] = "durable_artifact_ref.v1"
    artifact_id: str = Field(min_length=1)
    artifact_class: ArtifactClass
    type: str
    uri: str
    mime_type: str | None = None
    content_digest: str
    size_bytes: int = Field(ge=0)
    version: int = Field(default=1, ge=1)
    scope: ArtifactScope
    producer: ArtifactProvenance
    lineage: ArtifactLineage
    retention_class: ArtifactRetentionClass = ArtifactRetentionClass.DEFAULT
    sensitivity: ArtifactSensitivity = ArtifactSensitivity.INTERNAL
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("content_digest")
    @classmethod
    def _digest_format(cls, value: str) -> str:
        if not is_valid_artifact_digest(value):
            raise ValueError("content_digest must match sha256:<64 lowercase hex>")
        return value

    def verify_content(self, content: bytes) -> None:
        """Reject content that does not match the recorded digest (and size)."""
        verify_artifact_digest(content, self.content_digest)
        if len(content) != self.size_bytes:
            raise ArtifactDigestMismatchError("artifact content size does not match its reference")


def reconstruct_artifact_lineage(
    artifact_id: str,
    refs: Iterable[DurableArtifactRef],
) -> tuple[DurableArtifactRef, ...]:
    """Return the ancestors of ``artifact_id`` followed by the artifact, oldest first.

    Parents are addressed by identity *and* content digest, so an altered or missing parent
    fails with ``ArtifactLineageError`` instead of silently resolving to a different version.
    """
    pool = list(refs)
    by_key: dict[tuple[str, str], DurableArtifactRef] = {(ref.artifact_id, ref.content_digest): ref for ref in pool}
    latest: dict[str, DurableArtifactRef] = {}
    for ref in pool:
        current = latest.get(ref.artifact_id)
        if current is None or ref.version > current.version:
            latest[ref.artifact_id] = ref
    start = latest.get(artifact_id)
    if start is None:
        raise ArtifactLineageError(f"artifact {artifact_id!r} is not among the references")

    ordered: list[DurableArtifactRef] = []
    visiting: set[tuple[str, str]] = set()
    done: set[tuple[str, str]] = set()

    def visit(ref: DurableArtifactRef) -> None:
        key = (ref.artifact_id, ref.content_digest)
        if key in done:
            return
        if key in visiting:
            raise ArtifactLineageError(f"lineage cycle at artifact {ref.artifact_id!r}")
        visiting.add(key)
        for parent in ref.lineage.parent_artifacts:
            parent_ref = by_key.get((parent.artifact_id, parent.content_digest))
            if parent_ref is None:
                raise ArtifactLineageError(
                    f"parent {parent.artifact_id!r} of {ref.artifact_id!r} is missing or its digest differs"
                )
            visit(parent_ref)
        visiting.discard(key)
        done.add(key)
        ordered.append(ref)

    visit(start)
    return tuple(ordered)


def _retention_class_from_str(value: str) -> ArtifactRetentionClass:
    try:
        return ArtifactRetentionClass(value)
    except ValueError:
        return ArtifactRetentionClass.DEFAULT


class ArtifactRef(BaseModel):
    """Harness-registered artifact pointer — no secrets in uri query."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["artifact_ref.v1"] = "artifact_ref.v1"
    artifact_id: str
    type: str
    uri: str
    mime_type: str | None = None
    provenance: ArtifactProvenance
    retention_class: str = "default"
    sensitivity: ArtifactSensitivity = ArtifactSensitivity.INTERNAL
    checksum: str | None = None
    size_bytes: int | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    trace_id: str = ""
    run_id: str = ""
    step_index: int | None = None

    def to_durable_ref(
        self,
        *,
        scope: ArtifactScope,
        lineage: ArtifactLineage,
        artifact_class: ArtifactClass = ArtifactClass.EXECUTION_EVIDENCE,
        content: bytes | None = None,
        content_digest: str | None = None,
        size_bytes: int | None = None,
        version: int = 1,
    ) -> DurableArtifactRef:
        """Migrate this v1 pointer to a durable reference.

        The digest is taken from ``content`` when given, from ``content_digest``, or from a
        ``checksum`` already in platform format. When several sources are present they must
        agree. With none of them a typed ``ArtifactDigestRequiredError`` is raised; an empty
        digest is never produced.
        """
        candidates: list[str] = []
        if content is not None:
            candidates.append(compute_artifact_digest(content))
        if content_digest is not None:
            candidates.append(content_digest)
        if self.checksum and is_valid_artifact_digest(self.checksum):
            candidates.append(self.checksum)
        if not candidates:
            raise ArtifactDigestRequiredError(
                f"artifact {self.artifact_id!r} has no content digest; supply the content or a digest"
            )
        if len(set(candidates)) != 1:
            raise ArtifactDigestMismatchError(
                f"artifact {self.artifact_id!r}: supplied content, digest and checksum disagree"
            )
        size = len(content) if content is not None else (size_bytes if size_bytes is not None else self.size_bytes)
        if size is None:
            raise ArtifactDigestRequiredError(f"artifact {self.artifact_id!r} has no size; supply the content")
        return DurableArtifactRef(
            artifact_id=self.artifact_id,
            artifact_class=artifact_class,
            type=self.type,
            uri=self.uri,
            mime_type=self.mime_type,
            content_digest=candidates[0],
            size_bytes=size,
            version=version,
            scope=scope,
            producer=self.provenance,
            lineage=lineage,
            retention_class=_retention_class_from_str(self.retention_class),
            sensitivity=self.sensitivity,
            created_at=self.created_at,
        )


def artifact_ref_from_payload(
    payload: dict[str, Any],
    *,
    run_id: str,
    trace_id: str,
    agent_id: str,
    step_index: int | None = None,
) -> ArtifactRef:
    """Normalize loose StepOutcome artifact dicts into ``ArtifactRef``."""
    artifact_id = str(payload.get("artifact_id") or payload.get("id") or f"art_{uuid4().hex}")
    artifact_type = str(payload.get("type") or payload.get("kind") or "structured_json")
    uri = str(payload.get("uri") or payload.get("path") or f"memory://{artifact_id}")
    tool_id = payload.get("tool_id")
    return ArtifactRef(
        artifact_id=artifact_id,
        type=artifact_type,
        uri=uri,
        mime_type=payload.get("mime_type"),
        provenance=ArtifactProvenance(
            created_by_agent_id=agent_id,
            created_by_tool_id=str(tool_id) if tool_id else None,
            source_side_effect_id=(
                str(payload["source_side_effect_id"]) if payload.get("source_side_effect_id") else None
            ),
        ),
        retention_class=str(payload.get("retention_class") or "default"),
        sensitivity=ArtifactSensitivity(str(payload.get("sensitivity") or ArtifactSensitivity.INTERNAL.value)),
        checksum=payload.get("checksum"),
        size_bytes=payload.get("size_bytes"),
        trace_id=trace_id,
        run_id=run_id,
        step_index=step_index,
    )
