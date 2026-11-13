# © Artur Czarnecki. All rights reserved.

"""PDC-D8-R006 (unauthorized denied), R016 (operations in evidence), R010 (governed download)."""

from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

import pytest
from intergrax.contracts.agent_runtime_governance import (
    GovernanceAuditEvent,
    ToolAuthorizationDecisionState,
)
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactDigestMismatchError,
)
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessDenied,
    ArtifactDownloadUnavailable,
    ArtifactNotFound,
    ArtifactReadRequest,
    GovernedArtifactReadPort,
)
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.store_base import ArtifactDeletionRefused

from ._d8_support import (
    TENANT_A,
    TENANT_B,
    FakeObjectStorage,
    MutableClock,
    make_artifact,
    make_context,
    make_governance,
    make_port,
    reader_grants,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _request(artifact_id: str = "art-1", tenant_id: str = TENANT_A, **kw) -> ArtifactReadRequest:
    return ArtifactReadRequest(
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
        tenant_id=tenant_id,
        artifact_id=artifact_id,
        **kw,
    )


def _outcomes(sink, capability: str) -> list[GovernanceAuditEvent]:
    return [e for e in sink.events if e.capability == capability]


def _seeded() -> InMemoryArtifactStore:
    store = InMemoryArtifactStore()
    store.put(make_artifact(data=b"secret body"))
    return store


# ------------------------------------------------------------------ authorized read


def test_authorized_read_returns_verified_content_and_a_durable_ref() -> None:
    port, harness = make_port(_seeded())
    result = port.read(make_context(), _request())
    assert result.content == b"secret body"
    assert result.ref.scope.tenant_id == TENANT_A
    assert result.ref.version == 1
    assert _outcomes(harness.sink, "artifact.read.outcome")[-1].decision is (ToolAuthorizationDecisionState.ALLOW)


def test_read_of_a_specific_version() -> None:
    store = _seeded()
    store.put(make_artifact(data=b"newer"))
    port, _ = make_port(store)
    assert port.read(make_context(), _request(version=1)).content == b"secret body"
    assert port.read(make_context(), _request()).content == b"newer"


def test_missing_artifact_is_not_found_and_still_audited() -> None:
    port, harness = make_port(_seeded())
    with pytest.raises(ArtifactNotFound):
        port.read(make_context(), _request("absent"))
    assert _outcomes(harness.sink, "artifact.read.outcome")


# ------------------------------------------------------------------ R006 unauthorized


def test_reader_without_the_capability_is_denied_and_the_store_is_not_touched() -> None:
    store = _seeded()
    touched: list[str] = []
    original_get = store.get
    store.get = lambda *a, **k: (touched.append("get"), original_get(*a, **k))[1]  # type: ignore[method-assign]
    harness = make_governance(granted={})  # nobody holds artifact.read
    port, _ = make_port(store, harness=harness)
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.read(make_context(), _request())
    assert exc.value.reason_code == "not_authorized"
    assert touched == []
    denied = _outcomes(harness.sink, "artifact.read.outcome")
    assert denied and denied[-1].decision is ToolAuthorizationDecisionState.DENY


def test_reader_granted_only_read_cannot_download_or_delete() -> None:
    harness = make_governance(
        granted={(TENANT_A, "agent-reader"): frozenset({"artifact.read"})},
    )
    store = _seeded()
    port, _ = make_port(store, harness=harness, object_storage=FakeObjectStorage())
    port.read(make_context(), _request())
    with pytest.raises(ArtifactAccessDenied):
        port.delete_artifact(make_context(), _request())
    with pytest.raises(ArtifactAccessDenied):
        port.place_legal_hold(make_context(), _request(), reason="r")
    assert store.get(TENANT_A, "art-1").data == b"secret body"


# ------------------------------------------------------------------ R016 evidence


def test_every_operation_appears_in_governance_evidence() -> None:
    store = _seeded()
    harness = make_governance(granted=reader_grants())
    port, _ = make_port(store, harness=harness, object_storage=FakeObjectStorage())
    ctx = make_context()
    port.read(ctx, _request())
    port.place_legal_hold(ctx, _request(), reason="case 7")
    with pytest.raises(ArtifactDeletionRefused):
        port.delete_artifact(ctx, _request())
    port.release_legal_hold(ctx, _request())
    port.delete_artifact(ctx, _request())
    with pytest.raises(ArtifactNotFound):
        port.read(ctx, _request())
    seen = {e.capability for e in harness.sink.events}
    assert {
        "artifact.read",
        "artifact.read.outcome",
        "artifact.legal_hold.outcome",
        "artifact.delete.outcome",
    } <= seen
    # evidence carries the actor, the execution identity and the artifact id
    outcome = _outcomes(harness.sink, "artifact.read.outcome")[0]
    assert outcome.agent_id == "agent-reader" and outcome.tenant_id == TENANT_A
    assert str(outcome.run_id) == str(ctx.run_id)
    assert outcome.tool_id == "artifact:art-1"
    refused = [
        e
        for e in _outcomes(harness.sink, "artifact.delete.outcome")
        if e.decision is ToolAuthorizationDecisionState.DENY
    ]
    assert refused and "legal_hold" in refused[0].policy_results[0].reason


# ------------------------------------------------------------------ integrity


def test_content_that_no_longer_matches_its_digest_is_never_returned() -> None:
    store = _seeded()
    harness = make_governance(granted=reader_grants())
    port, _ = make_port(store, harness=harness)
    artifact = store.get(TENANT_A, "art-1")
    object.__setattr__(artifact, "data", b"forged in memory")  # simulate storage tampering
    with pytest.raises(ArtifactDigestMismatchError):
        port.read(make_context(), _request())
    last = _outcomes(harness.sink, "artifact.read.outcome")[-1]
    assert last.decision is ToolAuthorizationDecisionState.DENY


# ------------------------------------------------------------------ R010 download


def test_governed_download_is_a_short_lived_url_issued_only_after_authorization() -> None:
    from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource  # noqa: F401

    objects = FakeObjectStorage()
    store = _seeded()
    clock = MutableClock()
    harness = make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: _KeyedSource(store)},
        download_issuer=objects,
        max_download_ttl_seconds=300,
        clock=clock,
    )
    grant = port.issue_download_url(make_context(), _request(), ttl_seconds=900)
    assert grant.expires_in_seconds == 300  # clamped to the port maximum
    assert grant.expires_at == clock.now + timedelta(seconds=300)
    assert grant.url.startswith("https://objects.example.test/exec/")
    assert objects.presign_calls == [("exec/tenant-a/art-1/1", 300, "GET")]
    assert _outcomes(harness.sink, "artifact.download.outcome")


def test_download_is_denied_without_the_download_capability_and_issues_no_url() -> None:
    objects = FakeObjectStorage()
    harness = make_governance(granted={(TENANT_A, "agent-reader"): frozenset({"artifact.read"})})
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: _KeyedSource(_seeded())},
        download_issuer=objects,
        clock=MutableClock(),
    )
    with pytest.raises(ArtifactAccessDenied):
        port.issue_download_url(make_context(), _request(), ttl_seconds=60)
    assert objects.presign_calls == []


def test_cross_tenant_download_is_denied_and_issues_no_url() -> None:
    objects = FakeObjectStorage()
    harness = make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: _KeyedSource(_seeded())},
        download_issuer=objects,
        clock=MutableClock(),
    )
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.issue_download_url(make_context(tenant_id=TENANT_B, agent_id="agent-reader"), _request(), ttl_seconds=60)
    assert exc.value.reason_code == "tenant_mismatch"
    assert objects.presign_calls == []


def test_download_requires_a_source_that_exposes_an_object_key() -> None:
    port, _ = make_port(_seeded(), object_storage=FakeObjectStorage())
    with pytest.raises(ArtifactDownloadUnavailable):
        port.issue_download_url(make_context(), _request(), ttl_seconds=60)


@pytest.mark.parametrize("ttl", [0, -5])
def test_non_positive_download_ttl_is_rejected(ttl: int) -> None:
    harness = make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: _KeyedSource(_seeded())},
        download_issuer=FakeObjectStorage(),
        clock=MutableClock(),
    )
    with pytest.raises(ValueError):
        port.issue_download_url(make_context(), _request(), ttl_seconds=ttl)


def _presign_callers() -> set[str]:
    allowed_prefix = "intergrax/integrations/providers/"
    allowed_files = {"intergrax/runtime/nexus/artifacts/governed_access.py"}
    offenders: set[str] = set()
    for path in (REPO_ROOT / "intergrax").rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(allowed_prefix) or rel in allowed_files:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"presigned_url", "generate_presigned_url"}
            ):
                offenders.add(rel)
    return offenders


def test_no_presigned_url_is_issued_outside_the_governed_port() -> None:
    """AST gate: ``.presigned_url(...)`` is called only by providers (delegation) and by the port."""
    assert _presign_callers() == set()


def test_storage_presigned_url_tool_fails_closed_and_signs_nothing() -> None:
    """The agent-callable storage tool must not sign an arbitrary key for an arbitrary caller."""
    from intergrax.tools.providers.storage.contracts import StoragePresignedUrlInput
    from intergrax.tools.providers.storage.service import (
        GovernedPresignRequiredError,
        storage_presigned_url,
    )
    from intergrax.tools.registry.wiring import ToolWiringContext

    objects = FakeObjectStorage()
    ctx = ToolWiringContext(object_storage=objects)
    for method in ("GET", "PUT"):
        with pytest.raises(GovernedPresignRequiredError):
            storage_presigned_url(ctx, StoragePresignedUrlInput(key="tenant-b/secret.bin", method=method))
    assert objects.presign_calls == []
    # With no storage configured the failure is the same fail-closed error, not a provider call.
    with pytest.raises(GovernedPresignRequiredError):
        storage_presigned_url(ToolWiringContext(), StoragePresignedUrlInput(key="x"))


def test_stores_contain_no_security_logic() -> None:
    """Stores stay storage-only: they never import governance or authorization."""
    forbidden = ("agent_governance", "agent_runtime_governance", "runtime_policy", "authority")
    store_files = [
        "in_memory_artifact_store.py",
        "sqlite_artifact_store.py",
        "store_base.py",
    ]
    for name in store_files:
        text = (REPO_ROOT / "intergrax/runtime/nexus/artifacts" / name).read_text(encoding="utf-8")
        imports = [line for line in text.splitlines() if line.startswith(("from ", "import "))]
        assert not [i for i in imports if any(f in i for f in forbidden)], (name, imports)


class _KeyedSource:
    """Storage-only source that also exposes an object key (download-capable)."""

    def __init__(self, store: InMemoryArtifactStore) -> None:
        from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource

        self._inner = ExecutionArtifactSource(store)

    def read(self, request):
        return self._inner.read(request)

    def download_key(self, request) -> str | None:
        artifact = self._inner.read(request)
        return None if artifact is None else f"exec/{request.tenant_id}/{request.artifact_id}/{artifact.ref.version}"
