# © Artur Czarnecki. All rights reserved.

"""Source Candidate registry, intake, routing and shared folder indexing tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingError,
    WorkspaceDocumentIndexingResult,
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_ingestion import (
    KnowledgeIngestionProcessorError,
    KnowledgeIngestionProcessorRouter,
    KnowledgeIngestionService,
    LKW_KNOWLEDGE_INGESTION_TASK_NAME,
    register_knowledge_ingestion_worker_handler,
)
from local_workspace_application.workspaces.knowledge_intake import (
    KnowledgeInputResolutionError,
    KnowledgeInputSourceResolverRouter,
    KnowledgeIntakeService,
)
from local_workspace_application.workspaces.local_folder_indexing import LocalFolderIndexingService
from local_workspace_application.workspaces.managed_file_ingestion import (
    ManagedFileKnowledgeIngestionProcessor,
)
from local_workspace_application.workspaces.managed_files import ManagedFileSourceResolver
from local_workspace_application.workspaces.models import (
    KnowledgeInput,
    KnowledgeInputKind,
    KnowledgeInputStatus,
    Workspace,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.source_candidates import (
    ConfiguredSourceCandidate,
    SourceCandidateAlreadyRegistered,
    SourceCandidateIdempotencyConflict,
    SourceCandidateIntakeService,
    SourceCandidateKnowledgeIngestionProcessor,
    SourceCandidateRegistry,
    SourceCandidateRegistryError,
    SourceCandidateSourceResolver,
    SourceCandidateUnavailable,
)
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TENANT = "tenant-a"
OTHER_TENANT = "tenant-b"
WORKSPACE = "workspace-a"


def _now() -> datetime:
    return datetime.now(UTC)


def _seed_workspace(repo: ManagedWorkspaceRepository, *, tenant_id: str = TENANT) -> Workspace:
    workspace = Workspace(
        workspace_id=WORKSPACE,
        tenant_id=tenant_id,
        name="Demo",
        status=WorkspaceStatus.ACTIVE,
        created_at=_now(),
        updated_at=_now(),
    )
    return repo.put_workspace(workspace)


class _FakeExecutor:
    async def execute(self, task: object) -> object:
        _ = task
        return type(
            "R",
            (),
            {
                "metadata": {
                    "ingest_summary": {
                        "used": True,
                        "reason": "ingest_complete",
                        "num_chunks": 1,
                    }
                }
            },
        )()


class SpyIndexingService:
    def __init__(self, *, fail_paths: set[str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_paths = fail_paths or set()

    async def index_one(self, **kwargs: object) -> WorkspaceDocumentIndexingResult:
        self.calls.append(kwargs)
        physical = kwargs["physical_path"]
        assert isinstance(physical, Path)
        if str(physical) in self.fail_paths or physical.name in self.fail_paths:
            raise WorkspaceDocumentIndexingError("index_failed")
        return WorkspaceDocumentIndexingResult(
            indexed=True,
            unchanged=False,
            document_id=f"doc-{len(self.calls)}",
            documents_indexed=1,
            num_chunks=1,
            reason="ingest_complete",
        )


def _write_config(path: Path, candidates: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "lkw.source_candidates.v1",
                "candidates": candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _candidate_dict(
    *,
    folder: Path,
    candidate_id: str = "contracts",
    tenant_id: str = TENANT,
    label: str = "Contracts",
    description: str = "Current contract documents",
    recursive: bool = True,
    enabled: bool = True,
    path: str | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "tenant_id": tenant_id,
        "label": label,
        "description": description,
        "source_type": "local_folder",
        "path": path if path is not None else str(folder.resolve()),
        "recursive": recursive,
        "enabled": enabled,
    }


def _build_stack(
    tmp_path: Path,
    *,
    folder: Path | None = None,
    config_candidates: list[dict[str, object]] | None = None,
    indexing: SpyIndexingService | None = None,
    with_managed_file: bool = False,
):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_workspace(repo)
    folder = folder or (tmp_path / "docs")
    folder.mkdir(parents=True, exist_ok=True)
    if not any(folder.iterdir()):
        (folder / "note.txt").write_text("hello knowledge", encoding="utf-8")

    config_path = tmp_path / "config" / "source_candidates.json"
    if config_candidates is None:
        config_candidates = [_candidate_dict(folder=folder)]
    if config_candidates is not None:
        if config_candidates:
            _write_config(config_path, config_candidates)
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.exists():
                config_path.unlink()

    registry = (
        SourceCandidateRegistry.empty()
        if config_candidates == []
        else SourceCandidateRegistry.load(config_path)
    )
    allowlist = frozenset({str(folder.resolve())})
    if indexing is None:
        indexing = WorkspaceDocumentIndexingService(repo, _FakeExecutor())  # type: ignore[arg-type]
    folder_indexing = LocalFolderIndexingService(indexing, allowlist_roots=allowlist)  # type: ignore[arg-type]

    resolver_map: dict[KnowledgeInputKind, object] = {
        KnowledgeInputKind.SOURCE_CANDIDATE: SourceCandidateSourceResolver(
            repo,
            registry,
            allowlist_roots=allowlist,
        ),
    }
    processor_map: dict[KnowledgeInputKind, object] = {
        KnowledgeInputKind.SOURCE_CANDIDATE: SourceCandidateKnowledgeIngestionProcessor(
            folder_indexing,
        ),
    }
    if with_managed_file:
        resolver_map[KnowledgeInputKind.MANAGED_FILE] = ManagedFileSourceResolver(repo)
        processor_map[KnowledgeInputKind.MANAGED_FILE] = ManagedFileKnowledgeIngestionProcessor(
            repo,
            type("M", (), {})(),  # type: ignore[arg-type]
            indexing,  # type: ignore[arg-type]
        )

    queue = DocumentStoreTaskQueue(store)
    ctx = ToolWiringContext(message_bus=queue)
    intake = KnowledgeIntakeService(
        repo,
        KnowledgeInputSourceResolverRouter(resolver_map),  # type: ignore[arg-type]
        ctx,
    )
    ingestion = KnowledgeIngestionService(
        repo,
        KnowledgeIngestionProcessorRouter(processor_map),  # type: ignore[arg-type]
    )
    registry_tasks = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry_tasks, ingestion)
    worker = DocumentStoreTaskWorker(queue, registry_tasks)
    candidate_intake = SourceCandidateIntakeService(
        repo,
        registry,
        intake,
        allowlist_roots=allowlist,
    )
    return (
        repo,
        registry,
        candidate_intake,
        intake,
        ingestion,
        queue,
        worker,
        folder_indexing,
        indexing,
        folder,
        allowlist,
        registry_tasks,
    )


def test_missing_config_file_yields_empty_registry(tmp_path: Path) -> None:
    registry = SourceCandidateRegistry.load(tmp_path / "missing.json")
    assert registry.is_available
    assert registry.list_for_tenant(TENANT) == ()


def test_valid_config_and_deterministic_sort(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    config = tmp_path / "source_candidates.json"
    _write_config(
        config,
        [
            _candidate_dict(folder=folder, candidate_id="b", label="Zebra"),
            _candidate_dict(folder=folder, candidate_id="a", label="Apple"),
            _candidate_dict(folder=folder, candidate_id="c", label="apple"),
        ],
    )
    registry = SourceCandidateRegistry.load(config)
    ordered = registry.list_for_tenant(TENANT)
    assert [item.candidate_id for item in ordered] == ["a", "c", "b"]


def test_duplicate_identity_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    config = tmp_path / "source_candidates.json"
    _write_config(
        config,
        [
            _candidate_dict(folder=folder, candidate_id="same"),
            _candidate_dict(folder=folder, candidate_id="same", label="Other"),
        ],
    )
    registry = SourceCandidateRegistry.load(config)
    assert not registry.is_available
    with pytest.raises(SourceCandidateRegistryError):
        registry.list_for_tenant(TENANT)


def test_same_candidate_id_different_tenants_allowed(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    config = tmp_path / "source_candidates.json"
    _write_config(
        config,
        [
            _candidate_dict(folder=folder, candidate_id="shared", tenant_id=TENANT),
            _candidate_dict(folder=folder, candidate_id="shared", tenant_id=OTHER_TENANT),
        ],
    )
    registry = SourceCandidateRegistry.load(config)
    assert len(registry.list_for_tenant(TENANT)) == 1
    assert len(registry.list_for_tenant(OTHER_TENANT)) == 1


@pytest.mark.parametrize(
    "mutate,field",
    [
        (lambda d: d.__setitem__("source_type", "remote_drive"), "source_type"),
        (lambda d: d.__setitem__("label", "C:\\Windows\\System32"), "label"),
        (lambda d: d.__setitem__("description", "https://evil.example"), "description"),
        (lambda d: d.__setitem__("recursive", "true"), "recursive"),
        (lambda d: d.__setitem__("enabled", "1"), "enabled"),
        (lambda d: d.__setitem__("extra", True), "extra"),
    ],
)
def test_invalid_candidate_fields_make_registry_unavailable(
    tmp_path: Path,
    mutate,
    field: str,
) -> None:
    _ = field
    folder = tmp_path / "docs"
    folder.mkdir()
    raw = _candidate_dict(folder=folder)
    mutate(raw)
    config = tmp_path / "source_candidates.json"
    _write_config(config, [raw])
    registry = SourceCandidateRegistry.load(config)
    assert not registry.is_available


def test_malformed_json_unavailable_without_parser_details(tmp_path: Path, caplog) -> None:
    config = tmp_path / "source_candidates.json"
    config.write_text("{not-json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        registry = SourceCandidateRegistry.load(config)
    assert not registry.is_available
    joined = " ".join(record.message for record in caplog.records)
    assert "source_candidate_configuration_invalid" in joined
    assert "{not-json" not in joined
    assert str(config) not in joined


def test_fingerprint_deterministic_and_ignores_public_fields(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    base = ConfiguredSourceCandidate(
        candidate_id="contracts",
        tenant_id=TENANT,
        label="Contracts",
        description="A",
        source_type="local_folder",
        path=str(folder),
        recursive=True,
        enabled=True,
    )
    relabel = ConfiguredSourceCandidate(
        candidate_id="contracts",
        tenant_id=TENANT,
        label="Other Label",
        description="Changed description",
        source_type="local_folder",
        path=str(folder),
        recursive=True,
        enabled=True,
    )
    path_changed = ConfiguredSourceCandidate(
        candidate_id="contracts",
        tenant_id=TENANT,
        label="Contracts",
        description="A",
        source_type="local_folder",
        path=str(folder / "nested"),
        recursive=True,
        enabled=True,
    )
    recursive_changed = ConfiguredSourceCandidate(
        candidate_id="contracts",
        tenant_id=TENANT,
        label="Contracts",
        description="A",
        source_type="local_folder",
        path=str(folder),
        recursive=False,
        enabled=True,
    )
    assert base.fingerprint() == relabel.fingerprint()
    assert base.fingerprint().startswith("sha256:")
    assert len(base.fingerprint()) == len("sha256:") + 64
    assert base.fingerprint() != path_changed.fingerprint()
    assert base.fingerprint() != recursive_changed.fingerprint()


def test_public_summary_omits_path_and_fingerprint(tmp_path: Path) -> None:
    (
        repo,
        registry,
        candidate_intake,
        *_rest,
        folder,
        _allowlist,
        _tasks,
    ) = _build_stack(tmp_path)
    _ = repo, registry
    summaries = candidate_intake.list_candidates(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert len(summaries) == 1
    text = str(summaries[0])
    assert str(folder) not in text
    assert "sha256:" not in text
    assert summaries[0].available is True


def test_tenant_isolation_and_disabled_hidden(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    config = tmp_path / "source_candidates.json"
    _write_config(
        config,
        [
            _candidate_dict(folder=folder, candidate_id="visible", enabled=True),
            _candidate_dict(folder=folder, candidate_id="hidden", enabled=False),
            _candidate_dict(
                folder=folder,
                candidate_id="other",
                tenant_id=OTHER_TENANT,
                label="Other",
            ),
        ],
    )
    registry = SourceCandidateRegistry.load(config)
    ids = [item.candidate_id for item in registry.list_for_tenant(TENANT)]
    assert ids == ["visible"]
    assert registry.get(TENANT, "other") is None
    assert registry.get(OTHER_TENANT, "other") is not None


def test_unavailable_candidate_listed_as_false(tmp_path: Path) -> None:
    missing = tmp_path / "missing-folder"
    (
        _repo,
        _registry,
        candidate_intake,
        *_rest,
    ) = _build_stack(
        tmp_path,
        config_candidates=[_candidate_dict(folder=missing, path=str(missing))],
    )
    summaries = candidate_intake.list_candidates(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert summaries[0].available is False


def test_successful_registration_and_worker_processing(tmp_path: Path) -> None:
    (
        repo,
        _registry,
        candidate_intake,
        _intake,
        _ingestion,
        _queue,
        worker,
        _folder_indexing,
        _indexing,
        folder,
        _allowlist,
        _tasks,
    ) = _build_stack(tmp_path)

    accepted = candidate_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        candidate_id="contracts",
        idempotency_key="idem-1",
    )
    knowledge_input = repo.get_knowledge_input(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)[0].input_id,
    )
    assert knowledge_input is not None
    assert knowledge_input.input_kind is KnowledgeInputKind.SOURCE_CANDIDATE
    assert set(knowledge_input.submission_metadata.keys()) == {
        "candidate_id",
        "candidate_fingerprint",
    }
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=accepted.source_id,
    )
    assert source is not None
    assert source.source_type is WorkspaceSourceType.LOCAL_FOLDER
    assert Path(source.path) == folder.resolve()
    operation = repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert operation is not None
    assert operation.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
    assert operation.queue_task_id
    assert worker.drain_once() == 1
    operation = repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert operation is not None
    assert operation.status is WorkspaceOperationStatus.COMPLETED
    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=accepted.source_id,
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.READY
    refs = repo.list_document_refs(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert len(refs) == 1
    assert refs[0].source_id == accepted.source_id


def test_candidate_intake_works_without_object_storage_routing(tmp_path: Path) -> None:
    (
        _repo,
        _registry,
        candidate_intake,
        _intake,
        _ingestion,
        _queue,
        worker,
        *_rest,
    ) = _build_stack(tmp_path, with_managed_file=False)
    accepted = candidate_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        candidate_id="contracts",
        idempotency_key="no-obj",
    )
    assert accepted.source_id
    assert worker.drain_once() == 1
    op = _repo.get_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.COMPLETED


def test_exact_retry_preserves_identities(tmp_path: Path) -> None:
    (
        repo,
        _registry,
        candidate_intake,
        *_rest,
    ) = _build_stack(tmp_path)
    first = candidate_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        candidate_id="contracts",
        idempotency_key="same-key",
    )
    second = candidate_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        candidate_id="contracts",
        idempotency_key="same-key",
    )
    assert first.source_id == second.source_id
    assert first.operation_id == second.operation_id
    ops = [
        op
        for op in repo.list_operations(tenant_id=TENANT)
        if op.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
    ]
    assert len(ops) == 1
    assert ops[0].queue_task_id


def test_idempotency_conflict_different_candidate(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (
        _repo,
        _registry,
        candidate_intake,
        *_rest,
    ) = _build_stack(
        tmp_path,
        folder=folder,
        config_candidates=[
            _candidate_dict(folder=folder, candidate_id="contracts"),
            _candidate_dict(folder=folder, candidate_id="policies", label="Policies"),
        ],
    )
    candidate_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        candidate_id="contracts",
        idempotency_key="shared-key",
    )
    with pytest.raises(SourceCandidateIdempotencyConflict):
        candidate_intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            candidate_id="policies",
            idempotency_key="shared-key",
        )


def test_already_registered_same_fingerprint_different_key(tmp_path: Path) -> None:
    (
        repo,
        _registry,
        candidate_intake,
        *_rest,
    ) = _build_stack(tmp_path)
    first = candidate_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        candidate_id="contracts",
        idempotency_key="key-a",
    )
    with pytest.raises(SourceCandidateAlreadyRegistered):
        candidate_intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            candidate_id="contracts",
            idempotency_key="key-b",
        )
    sources = repo.list_sources(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert len(sources) == 1
    assert sources[0].source_id == first.source_id


@pytest.mark.parametrize(
    "setup",
    [
        "missing",
        "outside_allowlist",
        "shadow",
    ],
)
def test_unavailable_candidate_selection(tmp_path: Path, setup: str) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    if setup == "missing":
        path = tmp_path / "gone"
        allow = frozenset({str(docs.resolve())})
        shadow_roots: tuple[Path, ...] = ()
    elif setup == "outside_allowlist":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "a.txt").write_text("x", encoding="utf-8")
        path = outside
        allow = frozenset({str(docs.resolve())})
        shadow_roots = ()
    else:
        path = shadow
        allow = frozenset({str(shadow.resolve()), str(docs.resolve())})
        shadow_roots = (shadow,)

    config = tmp_path / "source_candidates.json"
    _write_config(config, [_candidate_dict(folder=path, path=str(path))])
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_workspace(repo)
    registry = SourceCandidateRegistry.load(config)
    queue = DocumentStoreTaskQueue(store)
    intake = KnowledgeIntakeService(
        repo,
        KnowledgeInputSourceResolverRouter(
            {
                KnowledgeInputKind.SOURCE_CANDIDATE: SourceCandidateSourceResolver(
                    repo,
                    registry,
                    allowlist_roots=allow,
                    shadow_roots=shadow_roots,
                )
            }
        ),
        ToolWiringContext(message_bus=queue),
    )
    candidate_intake = SourceCandidateIntakeService(
        repo,
        registry,
        intake,
        allowlist_roots=allow,
        shadow_roots=shadow_roots,
    )
    with pytest.raises(SourceCandidateUnavailable) as exc_info:
        candidate_intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            candidate_id="contracts",
            idempotency_key="bad",
        )
    assert str(path) not in str(exc_info.value)
    assert "allowlist" not in str(exc_info.value).lower()


def test_config_changed_before_resolution(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "note.txt").write_text("x", encoding="utf-8")
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_workspace(repo)
    config = tmp_path / "source_candidates.json"
    _write_config(config, [_candidate_dict(folder=folder)])
    registry = SourceCandidateRegistry.load(config)
    candidate = registry.get(TENANT, "contracts")
    assert candidate is not None
    fingerprint = candidate.fingerprint()
    input_id = "ki:pending"
    now = _now()
    knowledge_input = KnowledgeInput(
        input_id=input_id,
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.SOURCE_CANDIDATE,
        idempotency_key="pending",
        operation_id="op:pending",
        source_id=None,
        status=KnowledgeInputStatus.ACCEPTED,
        submission_metadata={
            "candidate_id": "contracts",
            "candidate_fingerprint": fingerprint,
        },
        created_at=now,
        updated_at=now,
        error_code=None,
    )
    repo.put_knowledge_input(knowledge_input)
    _write_config(
        config,
        [_candidate_dict(folder=folder, recursive=False)],
    )
    registry2 = SourceCandidateRegistry.load(config)
    resolver = SourceCandidateSourceResolver(
        repo,
        registry2,
        allowlist_roots=frozenset({str(folder.resolve())}),
    )
    with pytest.raises(KnowledgeInputResolutionError, match="source_candidate_configuration_changed"):
        resolver.resolve(knowledge_input=knowledge_input, suggested_source_id="src:1")


def test_config_removed_after_resolution_uses_durable_source(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "note.txt").write_text("x", encoding="utf-8")
    (
        repo,
        _registry,
        candidate_intake,
        intake,
        *_rest,
    ) = _build_stack(tmp_path, folder=folder)
    accepted = candidate_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        candidate_id="contracts",
        idempotency_key="keep-source",
    )
    empty_registry = SourceCandidateRegistry.empty()
    resolver = SourceCandidateSourceResolver(
        repo,
        empty_registry,
        allowlist_roots=frozenset({str(folder.resolve())}),
    )
    knowledge_input = repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)[0]
    resolved = resolver.resolve(
        knowledge_input=knowledge_input,
        suggested_source_id=accepted.source_id,
    )
    assert resolved.source_id == accepted.source_id
    assert Path(resolved.path) == folder.resolve()
    resumed = KnowledgeIntakeService(
        repo,
        resolver,
        intake._wiring_context,
    ).accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.SOURCE_CANDIDATE,
        idempotency_key="keep-source",
        submission_metadata=dict(knowledge_input.submission_metadata),
    )
    assert resumed.source.source_id == accepted.source_id


@pytest.mark.asyncio
async def test_routing_managed_file_and_candidate_and_unsupported(tmp_path: Path) -> None:
    (
        repo,
        registry,
        _candidate_intake,
        _intake,
        _ingestion,
        _queue,
        _worker,
        folder_indexing,
        indexing,
        folder,
        allowlist,
        task_registry,
    ) = _build_stack(tmp_path, with_managed_file=True)
    managed_resolver = ManagedFileSourceResolver(repo)
    candidate_resolver = SourceCandidateSourceResolver(repo, registry, allowlist_roots=allowlist)
    router = KnowledgeInputSourceResolverRouter(
        {
            KnowledgeInputKind.MANAGED_FILE: managed_resolver,
            KnowledgeInputKind.SOURCE_CANDIDATE: candidate_resolver,
        }
    )
    now = _now()
    unsupported = KnowledgeInput(
        input_id="ki:x",
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_kind=KnowledgeInputKind.WEB_URL,
        idempotency_key="x",
        operation_id="op:x",
        source_id=None,
        status=KnowledgeInputStatus.ACCEPTED,
        submission_metadata={},
        created_at=now,
        updated_at=now,
        error_code=None,
    )
    with pytest.raises(KnowledgeInputResolutionError, match="source_resolver_unavailable"):
        router.resolve(knowledge_input=unsupported, suggested_source_id="src:x")

    processor_router = KnowledgeIngestionProcessorRouter(
        {
            KnowledgeInputKind.MANAGED_FILE: ManagedFileKnowledgeIngestionProcessor(
                repo,
                type("M", (), {})(),  # type: ignore[arg-type]
                indexing,  # type: ignore[arg-type]
            ),
            KnowledgeInputKind.SOURCE_CANDIDATE: SourceCandidateKnowledgeIngestionProcessor(
                folder_indexing,
            ),
        }
    )
    with pytest.raises(
        KnowledgeIngestionProcessorError,
        match="knowledge_ingestion_processor_unavailable",
    ):
        await processor_router.process(
            knowledge_input=unsupported,
            source=WorkspaceSource(
                source_id="s",
                workspace_id=WORKSPACE,
                tenant_id=TENANT,
                source_type=WorkspaceSourceType.LOCAL_FOLDER,
                path=str(folder),
                recursive=True,
                status=WorkspaceSourceStatus.REGISTERED,
                created_at=now,
            ),
            operation=WorkspaceOperation(
                operation_id="op:x",
                tenant_id=TENANT,
                workspace_id=WORKSPACE,
                source_id="s",
                operation_type=WorkspaceOperationType.KNOWLEDGE_INGESTION,
                status=WorkspaceOperationStatus.QUEUED,
            ),
        )
    assert task_registry.get_handler(LKW_KNOWLEDGE_INGESTION_TASK_NAME) is not None


@pytest.mark.asyncio
async def test_shared_folder_indexing_used_by_sync_and_candidate(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("a", encoding="utf-8")
    nested = folder / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")
    (folder / "skip.bin").write_bytes(b"\x00\x01")
    allowlist = frozenset({str(folder.resolve())})

    (
        repo,
        _registry,
        candidate_intake,
        _intake,
        ingestion,
        _queue,
        _worker,
        folder_indexing,
        _indexing,
        _folder,
        _allowlist,
        _tasks,
    ) = _build_stack(
        tmp_path,
        folder=folder,
        config_candidates=[_candidate_dict(folder=folder, recursive=True)],
    )
    accepted = candidate_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        candidate_id="contracts",
        idempotency_key="folder-1",
    )
    await ingestion.run_operation(tenant_id=TENANT, operation_id=accepted.operation_id)
    refs = repo.list_document_refs(tenant_id=TENANT, workspace_id=WORKSPACE)
    assert len(refs) == 2

    source = repo.get_source(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        source_id=accepted.source_id,
    )
    assert source is not None
    now = _now()
    sync_op = repo.put_operation(
        WorkspaceOperation(
            operation_id="op-sync-1",
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            source_id=source.source_id,
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.QUEUED,
            created_at=now,
        )
    )
    sync = ManagedWorkspaceSyncService(
        repo,
        task_executor=_FakeExecutor(),  # type: ignore[arg-type]
        allowlist_roots=allowlist,
        folder_indexing=folder_indexing,
    )
    result = await sync.run_operation(tenant_id=TENANT, operation_id=sync_op.operation_id)
    assert result.status is WorkspaceOperationStatus.COMPLETED
    assert result.documents_unchanged == 2
    assert result.documents_indexed == 0


@pytest.mark.asyncio
async def test_empty_folder_and_all_failed_behaviors(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    allow = frozenset({str(empty.resolve())})
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_workspace(repo)
    indexing = SpyIndexingService()
    folder_indexing = LocalFolderIndexingService(indexing, allowlist_roots=allow)  # type: ignore[arg-type]
    now = _now()
    source = repo.put_source(
        WorkspaceSource(
            source_id="src-empty",
            workspace_id=WORKSPACE,
            tenant_id=TENANT,
            source_type=WorkspaceSourceType.LOCAL_FOLDER,
            path=str(empty.resolve()),
            recursive=True,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=now,
        )
    )
    op = repo.put_operation(
        WorkspaceOperation(
            operation_id="op-empty",
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            source_id=source.source_id,
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.QUEUED,
            created_at=now,
        )
    )
    sync = ManagedWorkspaceSyncService(
        repo,
        task_executor=type("E", (), {"execute": lambda *a, **k: None})(),  # type: ignore[arg-type]
        allowlist_roots=allow,
        folder_indexing=folder_indexing,
    )
    result = await sync.run_operation(tenant_id=TENANT, operation_id=op.operation_id)
    assert result.status is WorkspaceOperationStatus.COMPLETED
    assert result.files_discovered == 0

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x.txt").write_text("x", encoding="utf-8")
    allow2 = frozenset({str(bad.resolve())})
    failing = SpyIndexingService(fail_paths={"x.txt"})
    folder_indexing2 = LocalFolderIndexingService(failing, allowlist_roots=allow2)  # type: ignore[arg-type]
    source2 = repo.put_source(
        WorkspaceSource(
            source_id="src-bad",
            workspace_id=WORKSPACE,
            tenant_id=TENANT,
            source_type=WorkspaceSourceType.LOCAL_FOLDER,
            path=str(bad.resolve()),
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=now,
        )
    )
    op2 = repo.put_operation(
        WorkspaceOperation(
            operation_id="op-bad",
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            source_id=source2.source_id,
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.QUEUED,
            created_at=now,
        )
    )
    sync2 = ManagedWorkspaceSyncService(
        repo,
        task_executor=type("E", (), {"execute": lambda *a, **k: None})(),  # type: ignore[arg-type]
        allowlist_roots=allow2,
        folder_indexing=folder_indexing2,
    )
    failed = await sync2.run_operation(tenant_id=TENANT, operation_id=op2.operation_id)
    assert failed.status is WorkspaceOperationStatus.FAILED
    assert failed.error == "sync_produced_no_documents"
