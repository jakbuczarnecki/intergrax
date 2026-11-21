# © Artur Czarnecki. All rights reserved.
"""T-IGX-051-A Part C: governed artifact read bound to producing run."""

from __future__ import annotations

import pytest
from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    GovernanceAuditEvent,
    ToolAuthorizationDecisionState,
)
from intergrax.contracts.artifact_ref import ArtifactClass
from intergrax.contracts.execution_identity import (
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_task_id,
)
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.nexus.artifacts.governed_access import (
    ARTIFACT_CAPABILITIES,
    ArtifactAccessContext,
    ArtifactAccessDenied,
    ArtifactReadBinding,
    ArtifactReadRequest,
)
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from tests.qualification.pdc_x.d8._d8_support import (
    TENANT_A,
    FakeObjectStorage,
    MutableClock,
    make_artifact,
    make_governance,
    reader_grants,
)
from tests.qualification.pdc_x.d8_repair._support import make_clocked_port, request

pytestmark = pytest.mark.unit

PRODUCING_RUN = RunId("run_" + "a" * 32)
OTHER_RUN = RunId("run_" + "b" * 32)


def _context(
    *,
    run_id: RunId,
    tenant_id: str = TENANT_A,
    agent_id: str = "agent-reader",
    read_binding: ArtifactReadBinding = ArtifactReadBinding.RUN_BOUND,
) -> ArtifactAccessContext:
    return ArtifactAccessContext(
        agent=AgentIdentity(agent_id=agent_id, tenant_id=tenant_id),
        task_id=mint_task_id(),
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        read_binding=read_binding,
    )


def _seeded_store(*, run_id: str = str(PRODUCING_RUN)) -> InMemoryArtifactStore:
    store = InMemoryArtifactStore()
    store.put(make_artifact(data=b"tenant-bound body", run_id=run_id))
    return store


def _deny_outcomes(sink, capability: str) -> list[GovernanceAuditEvent]:
    suffix = f"{capability}.outcome"
    return [
        event
        for event in sink.events
        if event.capability == suffix and event.decision is ToolAuthorizationDecisionState.DENY
    ]


class _DownloadCapableSource:
    """Minimal download-capable source double for presign tests."""

    def __init__(self, store: InMemoryArtifactStore) -> None:
        from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource

        self._inner = ExecutionArtifactSource(store)

    def read(self, read_request: ArtifactReadRequest):
        return self._inner.read(read_request)

    def download_key(self, read_request: ArtifactReadRequest) -> str | None:
        found = self._inner.read(read_request)
        if found is None:
            return None
        return f"exec/{read_request.tenant_id}/{read_request.artifact_id}/{found.ref.version}"


def _download_port(store: InMemoryArtifactStore):
    from intergrax.runtime.nexus.artifacts.governed_access import GovernedArtifactReadPort

    objects = FakeObjectStorage()
    harness = make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: _DownloadCapableSource(store)},
        download_issuer=objects,
        clock=MutableClock(),
    )
    return port, harness, objects


def test_ten_06a_read_by_other_run_refused() -> None:
    store = _seeded_store()
    port, _ = make_clocked_port(store)
    ctx = _context(run_id=OTHER_RUN)
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.read(ctx, request())
    assert exc.value.reason_code == "run_mismatch"


def test_ten_06b_read_by_producing_run_allowed() -> None:
    store = _seeded_store()
    port, _ = make_clocked_port(store)
    result = port.read(_context(run_id=PRODUCING_RUN), request())
    assert result.content == b"tenant-bound body"


def test_ten_06c_download_by_other_run_refused() -> None:
    store = _seeded_store()
    port, _, objects = _download_port(store)
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.issue_download_url(_context(run_id=OTHER_RUN), request(), ttl_seconds=60)
    assert exc.value.reason_code == "run_mismatch"
    assert objects.presign_calls == []


def test_ten_06d_denied_read_is_audited() -> None:
    store = _seeded_store()
    port, harness = make_clocked_port(store)
    with pytest.raises(ArtifactAccessDenied):
        port.read(_context(run_id=OTHER_RUN), request())
    denies = _deny_outcomes(harness.sink, "artifact.read")
    assert denies
    assert denies[-1].policy_results[0].policy_id == "artifact.port.run_mismatch"


def test_ten_06e_system_actor_paths_unchanged() -> None:
    store = _seeded_store()
    harness = make_governance(granted={(TENANT_A, "artifact_retention_sweep"): frozenset(ARTIFACT_CAPABILITIES)})
    port, _ = make_clocked_port(store, harness=harness)
    sweep_identity = BackgroundExecutionIdentity(
        tenant_id=TENANT_A,
        task_id=mint_task_id(),
        run_id=OTHER_RUN,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    ctx = ArtifactAccessContext.for_background_execution(agent_id="artifact_retention_sweep", identity=sweep_identity)
    assert ctx.read_binding is ArtifactReadBinding.TENANT_SWEEP
    port.place_legal_hold(ctx, request(), reason="retention case")
    port.release_legal_hold(ctx, request())


def test_ten_06f_other_principal_same_run_allowed() -> None:
    store = _seeded_store()
    grants = dict(reader_grants())
    grants[(TENANT_A, "other-agent")] = frozenset(ARTIFACT_CAPABILITIES)
    port, _ = make_clocked_port(store, harness=make_governance(granted=grants))
    ctx = _context(run_id=PRODUCING_RUN, agent_id="other-agent")
    result = port.read(ctx, request())
    assert result.content == b"tenant-bound body"
