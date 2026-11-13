# © Artur Czarnecki. All rights reserved.

"""Business class behind the governed port (R005/R006/R010/R009) and run-bundle lineage (R004)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.collaborative_work.content_storage import (
    ObjectStorageArtifactContentStore,
    StoreArtifactContentRequest,
)
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactDigestRequiredError,
    ArtifactLineage,
    ArtifactProvenance,
    compute_artifact_digest,
    reconstruct_artifact_lineage,
)
from intergrax.contracts.task_artifacts import ApplicationArtifactRef, RunArtifactBundle
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessDenied,
    ArtifactReadRequest,
    GovernedArtifactReadPort,
)
from intergrax.runtime.nexus.artifacts.sources import (
    BusinessArtifactBinding,
    CollaborativeWorkContentSource,
)

from ._d8_support import (
    TENANT_A,
    TENANT_B,
    FakeObjectStorage,
    MutableClock,
    make_context,
    make_governance,
    reader_grants,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
BODY = b"business artifact body"


def _business_port():
    objects = FakeObjectStorage()
    content_store = ObjectStorageArtifactContentStore(objects)
    content_ref = content_store.put(
        StoreArtifactContentRequest(tenant_id=TENANT_A, workspace_id="ws-1", body=BODY, media_type="text/plain")
    )
    binding = BusinessArtifactBinding(
        content_ref=content_ref,
        version=3,
        producer=ArtifactProvenance(created_by_agent_id="principal-1"),
        lineage=ArtifactLineage(run_id="work-item-1", task_id="task-1"),
    )

    def resolve(tenant_id: str, workspace_id: str, artifact_id: str, version):
        if (tenant_id, workspace_id, artifact_id) == (TENANT_A, "ws-1", "work-artifact-1"):
            return binding
        return None

    harness = make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(
        governance=harness.boundary,
        audit=harness.recorder,
        sources={ArtifactClass.BUSINESS: CollaborativeWorkContentSource(content_store, resolve)},
        download_issuer=objects,
        clock=MutableClock(),
    )
    return port, harness, objects


def _req(tenant_id: str = TENANT_A) -> ArtifactReadRequest:
    return ArtifactReadRequest(
        artifact_class=ArtifactClass.BUSINESS,
        tenant_id=tenant_id,
        artifact_id="work-artifact-1",
        workspace_id="ws-1",
    )


def test_business_artifacts_are_read_through_the_same_governed_port() -> None:
    port, harness, _ = _business_port()
    result = port.read(make_context(), _req())
    assert result.content == BODY
    assert result.ref.artifact_class is ArtifactClass.BUSINESS
    assert result.ref.content_digest == compute_artifact_digest(BODY)
    assert result.ref.version == 3
    assert harness.sink.events


def test_business_artifact_cross_tenant_read_is_denied() -> None:
    port, _, _ = _business_port()
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.read(make_context(tenant_id=TENANT_A), _req(TENANT_B))
    assert exc.value.reason_code == "tenant_mismatch"


def test_business_artifact_download_uses_the_provider_url_after_authorization() -> None:
    port, harness, objects = _business_port()
    grant = port.issue_download_url(make_context(), _req(), ttl_seconds=120)
    assert grant.expires_in_seconds == 120
    assert len(objects.presign_calls) == 1
    key, ttl, method = objects.presign_calls[0]
    assert key.startswith("intergrax/collaborative-work/artifact-content/v1/") and (ttl, method) == (120, "GET")
    assert any(e.capability == "artifact.download.outcome" for e in harness.sink.events)


def test_business_artifacts_cannot_be_held_or_deleted_through_the_port() -> None:
    """Collaborative Work defers hard delete: the port reports the operation as unsupported."""
    port, harness, _ = _business_port()
    with pytest.raises(ArtifactAccessDenied) as exc:
        port.delete_artifact(make_context(), _req())
    assert exc.value.reason_code == "operation_unsupported"
    denied = [
        e
        for e in harness.sink.events
        if e.capability == "artifact.delete.outcome" and e.decision is ToolAuthorizationDecisionState.DENY
    ]
    assert denied


def test_collaborative_work_exposes_no_hard_delete_api() -> None:
    """Gate: no delete/remove/purge method on the artifact service or its repositories."""
    offenders: list[str] = []
    for name in ("artifact_service.py", "repository.py"):
        tree = ast.parse((REPO_ROOT / "intergrax/collaborative_work" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.lstrip("_").startswith(
                ("delete", "remove", "purge", "hard_delete")
            ):
                offenders.append(f"{name}:{node.name}")
    assert offenders == []


# ------------------------------------------------------------------ R004 run bundle


def _bundle() -> RunArtifactBundle:
    digest_hex = compute_artifact_digest(b"report")[len("sha256:") :]
    return RunArtifactBundle(
        task_id="task-1",
        graph_id="graph-1",
        run_id="run-1",
        application=[
            ApplicationArtifactRef(
                artifact_id="app-1",
                kind="export",
                uri="file:///tmp/report.json",
                task_id="task-1",
                owner_app_id="research",
                tenant_id=TENANT_A,
                sha256=digest_hex,
            )
        ],
    )


def test_run_bundle_projects_to_durable_refs_with_producer_and_execution_lineage() -> None:
    (ref,) = _bundle().durable_refs()
    assert ref.content_digest == compute_artifact_digest(b"report")
    assert ref.producer.created_by_agent_id == "research"
    assert ref.lineage.run_id == "run-1" and ref.lineage.task_id == "task-1"
    assert ref.scope.tenant_id == TENANT_A
    assert [r.artifact_id for r in reconstruct_artifact_lineage("app-1", [ref])] == ["app-1"]


def test_run_bundle_entry_without_a_digest_is_a_typed_error_not_an_empty_digest() -> None:
    bundle = _bundle()
    bundle.application[0] = bundle.application[0].model_copy(update={"sha256": ""})
    with pytest.raises(ArtifactDigestRequiredError):
        bundle.durable_refs()


def test_built_bundle_carries_digests_and_the_run_id(tmp_path: Path) -> None:
    from intergrax.applications._shared.run_artifact_bundle_builder import build_run_artifact_bundle
    from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
    from intergrax.runtime.sandbox.manager import SandboxSessionManager
    from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
    from intergrax.runtime.task.task import Task
    from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskIsolationOptions
    from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
    from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY

    shadow = ShadowWorkspaceManager(root=tmp_path / "shadow")
    sandbox = SandboxSessionManager(root=tmp_path / "sandbox")
    task = Task(
        tenant_id=TENANT_A,
        user_id="user-1",
        message="bundle",
        options=TaskExecutionOptions(isolation=TaskIsolationOptions(shadow_workspace=True, sandbox=True)),
    )
    workspace = shadow.open_or_create(tenant_id=task.tenant_id, task_id=task.task_id)
    workspace.write_text("outputs/summary.md", "# Summary")
    session = sandbox.open_or_create(tenant_id=task.tenant_id, task_id=task.task_id)
    session.execute("write_file", {"path": "result.txt", "content": "ok"})
    executions = [
        AgentExecutionResult(
            agent_id="research",
            run_id="run-42",
            status=AgentExecutionStatus.COMPLETED,
            structured_data={
                SHADOW_WORKSPACE_ID_KEY: workspace.workspace_id,
                SANDBOX_SESSION_ID_KEY: session.session_id,
            },
        )
    ]
    bundle = build_run_artifact_bundle(
        task=task,
        graph_id="graph-1",
        executions=executions,
        shadow_manager=shadow,
        sandbox_manager=sandbox,
    )
    assert bundle.run_id == "run-42"
    assert bundle.sandbox and all(len(s.sha256) == 64 for s in bundle.sandbox)
    refs = bundle.durable_refs()
    assert len(refs) == len(bundle.workspace) + len(bundle.sandbox)
    assert {r.lineage.run_id for r in refs} == {"run-42"}
    sandbox_ref = next(r for r in refs if r.type == "sandbox_file")
    sandbox_file = next(p for p in (tmp_path / "sandbox").rglob("result.txt"))
    assert sandbox_ref.content_digest == compute_artifact_digest(sandbox_file.read_bytes())
