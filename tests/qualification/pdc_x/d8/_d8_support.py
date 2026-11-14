# © Artur Czarnecki. All rights reserved.

"""Shared helpers for the PDC-X B-D8 (artifact lifecycle) qualification tests.

Everything here uses real production code paths (real governance pipeline, real
SQLite store, real background task registry). Only the object storage provider is
an in-memory double, because external providers are not part of this batch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    CapabilityGrant,
)
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactLineage,
    ArtifactProvenance,
    ArtifactRetentionClass,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.integrations.contracts.object_storage import StoredObject
from intergrax.runtime.agent_governance.audit import (
    GovernanceAuditRecorder,
    InMemoryGovernanceAuditSink,
)
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.agent_governance.capability_resolver import (
    InMemoryCapabilityGrantResolver,
)
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    AllowAllPolicyProvider,
)
from intergrax.runtime.nexus.artifacts.governed_access import (
    ARTIFACT_CAPABILITIES,
    ArtifactAccessContext,
    GovernedArtifactReadPort,
)
from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource

T0 = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)
TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


class MutableClock:
    """Deterministic clock injected into the sweeper, the spiller and the port."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + timedelta(**kwargs)


def make_artifact(
    *,
    tenant_id: str = TENANT_A,
    artifact_id: str = "art-1",
    run_id: str = "run-1",
    data: bytes = b"payload",
    artifact_class: ArtifactClass = ArtifactClass.EXECUTION_EVIDENCE,
    retention_class: ArtifactRetentionClass = ArtifactRetentionClass.DEFAULT,
    expires_at_utc: datetime | None = None,
    parents: tuple[tuple[str, str], ...] = (),
    created_at: datetime = T0,
    agent_id: str = "agent-1",
) -> Artifact:
    from intergrax.contracts.artifact_ref import ArtifactParent

    return Artifact(
        tenant_id=tenant_id,
        artifact_id=artifact_id,
        run_id=run_id,
        step_id="step-1",
        kind="report",
        mime_type="text/plain",
        created_at_utc=created_at,
        data=data,
        size_bytes=len(data),
        artifact_class=artifact_class,
        retention_class=retention_class,
        expires_at_utc=expires_at_utc,
        workspace_id="ws-1",
        producer=ArtifactProvenance(created_by_agent_id=agent_id, created_by_tool_id="tool.x"),
        lineage=ArtifactLineage(
            run_id=run_id,
            task_id="task-1",
            trace_id="trace-1",
            step_index=0,
            parent_artifacts=tuple(ArtifactParent(artifact_id=pid, content_digest=digest) for pid, digest in parents),
        ),
    )


def make_context(
    *,
    tenant_id: str = TENANT_A,
    agent_id: str = "agent-reader",
) -> ArtifactAccessContext:
    return ArtifactAccessContext(
        agent=AgentIdentity(agent_id=agent_id, tenant_id=tenant_id),
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


@dataclass
class GovernanceHarness:
    """Real governance pipeline (grants -> policy -> audit) with an in-memory audit sink."""

    boundary: AgentRuntimeGovernanceBoundary
    recorder: GovernanceAuditRecorder
    sink: InMemoryGovernanceAuditSink


def make_governance(
    *,
    granted: Mapping[tuple[str, str], frozenset[str]] | None = None,
) -> GovernanceHarness:
    """``granted`` maps ``(tenant_id, agent_id)`` to the capabilities the agent holds."""
    grants = tuple(
        CapabilityGrant(agent_id=agent, tenant_id=tenant, allowed_capabilities=caps)
        for (tenant, agent), caps in (granted or {}).items()
    )
    sink = InMemoryGovernanceAuditSink()
    recorder = GovernanceAuditRecorder(sink)
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver(grants),
        policy_engine=AgentRuntimePolicyEngine((AllowAllPolicyProvider(),)),
        audit_recorder=recorder,
    )
    return GovernanceHarness(
        boundary=AgentRuntimeGovernanceBoundary(pipeline),
        recorder=recorder,
        sink=sink,
    )


def reader_grants(
    *, tenant_id: str = TENANT_A, agent_id: str = "agent-reader"
) -> dict[tuple[str, str], frozenset[str]]:
    return {(tenant_id, agent_id): frozenset(ARTIFACT_CAPABILITIES)}


def make_port(
    store,
    *,
    harness: GovernanceHarness | None = None,
    clock: MutableClock | None = None,
    object_storage=None,
    max_download_ttl_seconds: int = 300,
) -> tuple[GovernedArtifactReadPort, GovernanceHarness]:
    harness = harness or make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={
            ArtifactClass.EXECUTION_EVIDENCE: ExecutionArtifactSource(store),
            ArtifactClass.TOOL_OUTPUT: ExecutionArtifactSource(store),
        },
        download_issuer=object_storage,
        max_download_ttl_seconds=max_download_ttl_seconds,
        clock=clock or MutableClock(),
    )
    return port, harness


class FakeObjectStorage:
    """External provider double (object storage): records every presigned URL request."""

    def __init__(self) -> None:
        self.objects: dict[str, StoredObject] = {}
        self.presign_calls: list[tuple[str, int, str]] = []

    def put(self, key: str, body: bytes, *, content_type: str = "application/octet-stream", metadata=None) -> None:
        self.objects[key] = StoredObject(
            key=key, body=body, content_type=content_type, metadata=dict(metadata or {}), size_bytes=len(body)
        )

    def put_if_absent(
        self, key: str, body: bytes, *, content_type: str = "application/octet-stream", metadata=None
    ) -> bool:
        if key in self.objects:
            return False
        self.put(key, body, content_type=content_type, metadata=metadata)
        return True

    def get(self, key: str) -> StoredObject | None:
        return self.objects.get(key)

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def close(self) -> None:
        return None

    def presigned_url(self, key: str, *, expires_in_seconds: int = 3600, method: str = "GET") -> str:
        self.presign_calls.append((key, expires_in_seconds, method))
        return f"https://objects.example.test/{key}?exp={expires_in_seconds}&sig=fake"


def sqlite_store(tmp_path: Path, name: str = "artifacts.sqlite3"):
    from intergrax.runtime.nexus.artifacts.sqlite_artifact_store import SQLiteArtifactStore

    return SQLiteArtifactStore(str(tmp_path / name))
