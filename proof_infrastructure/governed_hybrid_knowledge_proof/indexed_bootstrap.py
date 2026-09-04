# © Artur Czarnecki. All rights reserved.

"""Real managed-workspace indexing bootstrap for the governed hybrid knowledge proof."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.registry_projection import build_registry_projection
from intergrax.applications._shared.registry_projection_input_bundle import (
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
    build_reference_registry_projection_input_bundle,
)
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.host.agent_builders import LOCAL_WORKSPACE_AGENT_BUILDERS
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.lkw_task_enricher import build_lkw_combined_task_enricher
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.execution_wiring import build_lkw_host_task_execution
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.idempotency import (
    content_hash_for_file,
    normalize_source_path,
)
from local_workspace_application.workspaces.indexed_vector_verifier import (
    ManagedWorkspaceIndexedVectorVerifier,
)
from local_workspace_application.workspaces.local_folder_indexing import (
    LocalFolderIndexingService,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.task.task import Task, TaskResult
import hashlib

import numpy as np
from testing_support.builder import FakeEmbeddingProvider, build_fake_embedding_manager
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


class _DeterministicHashEmbeddingProvider(FakeEmbeddingProvider):
  def embed(self, texts: Sequence[str]) -> np.ndarray:
    self._resolve_dim()
    if not texts:
      return np.empty((0, self._dim), dtype=np.float32)
    rows: list[np.ndarray] = []
    for text in texts:
      digest = hashlib.sha256(text.encode("utf-8")).digest()
      seed = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
      if seed.size < self._dim:
        seed = np.resize(seed, self._dim)
      rows.append(seed[: self._dim])
    return np.stack(rows, axis=0)


def _build_proof_embedding_manager() -> EmbeddingManager:
  registry = EmbeddingProviderRegistry()
  provider = _DeterministicHashEmbeddingProvider()
  registry.register(provider)
  pipeline = EmbeddingPipeline(engine=EmbeddingEngine(registry), provider_id=provider.provider_name())
  return EmbeddingManager(pipeline=pipeline)


@contextmanager
def _proof_rag_embedding_patch() -> Iterator[None]:
    from intergrax.rag.bootstrap import rag_stack_bootstrap
    from intergrax.rag.embedding.bootstrap import default_embedding_engine

    fake_embedding = _build_proof_embedding_manager()
    original_create = rag_stack_bootstrap.create_default_rag_stack

    def _create_with_fake_embedding(**kwargs: object) -> object:
        if kwargs.get("embedding_manager") is None:
            kwargs["embedding_manager"] = fake_embedding
        return original_create(**kwargs)

    with (
        patch.object(
            default_embedding_engine,
            "create_default_embedding_manager",
            return_value=fake_embedding,
        ),
        patch.object(
            rag_stack_bootstrap,
            "create_default_rag_stack",
            _create_with_fake_embedding,
        ),
    ):
        yield


from unittest.mock import patch


class TaskExecutorPort(Protocol):
    async def execute(self, task: Task) -> TaskResult: ...


@dataclass(slots=True)
class IndexedProofStack:
  """In-memory LKW stack with real index and search task execution."""

  temp_path: Path
  settings: LocalWorkspaceBackendSettings
  task_executor: LocalWorkspaceTaskExecutor
  search_task_executor: TaskExecutorPort
  repository: ManagedWorkspaceRepository
  workspace_service: ManagedWorkspaceService
  indexing_service: WorkspaceDocumentIndexingService
  vectorstore_manager: VectorstoreManager
  embedding_manager: EmbeddingManager
  tool_wiring_context: ToolWiringContext
  indexed_document_id: str
  indexed_logical_path: str
  indexed_content_hash: str

  @property
  def search_executions(self) -> int:
    inner = self.search_task_executor
    if isinstance(inner, _SearchCountingTaskExecutor):
      return inner.search_executions
    return 0


class _SearchCountingTaskExecutor:
  def __init__(self, inner: LocalWorkspaceTaskExecutor) -> None:
    self._inner = inner
    self.search_executions = 0

  async def execute(self, task: Task) -> TaskResult:
    if task.context.capability == "local.workspace.search":
      self.search_executions += 1
    return await self._inner.execute(task)


def _configure_indexed_environment(base_dir: Path, docs_root: Path) -> LocalWorkspaceBackendSettings:
  data_home = base_dir / "lkw-data"
  sqlite_dir = base_dir / "sqlite"
  shadow_dir = base_dir / "shadow"
  for path in (data_home, sqlite_dir, shadow_dir, docs_root):
    path.mkdir(parents=True, exist_ok=True)
  os.environ["LOCAL_WORKSPACE_VECTOR_STORE"] = "inmemory"
  os.environ["LOCAL_WORKSPACE_ENABLE_RAG"] = "true"
  os.environ["LOCAL_WORKSPACE_ENABLE_RAG_INGEST"] = "true"
  os.environ["INTERGRAX_RAG_CHUNKING_STRATEGY"] = "recursive"
  os.environ["INTERGRAX_ALLOWED_READ_ROOTS"] = str(docs_root.resolve())
  os.environ["LKW_DATA_HOME"] = str(data_home)
  os.environ["INTERGRAX_SQLITE_DATA_DIR"] = str(sqlite_dir)
  os.environ["INTERGRAX_SHADOW_ROOT"] = str(shadow_dir)
  os.environ.pop("INTERGRAX_MONGODB_URI", None)
  return LocalWorkspaceBackendSettings.from_env()


async def bootstrap_indexed_proof_stack(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    policy_filename: str,
    policy_content: str,
    proof_now: datetime,
) -> IndexedProofStack:
  temp_path = Path(tempfile.mkdtemp(prefix="governed-hybrid-proof-"))
  base_dir = temp_path
  docs_root = base_dir / "docs"
  policy_path = docs_root / policy_filename
  policy_path.parent.mkdir(parents=True, exist_ok=True)
  policy_path.write_text(policy_content, encoding="utf-8")

  settings = _configure_indexed_environment(base_dir, docs_root)
  env = build_local_workspace_environment_profile(settings)
  projection_input = build_reference_registry_projection_input_bundle(
    LOCAL_WORKSPACE_APPLICATION_MANIFEST,
    env,
    builders=LOCAL_WORKSPACE_AGENT_BUILDERS,
    runtime_revision_id="governed-hybrid-proof",
    settings=settings,
  )
  registry_projection = build_registry_projection(projection_input)
  with _proof_rag_embedding_patch():
    harness_runtime = build_harness_host_runtime(
      LOCAL_WORKSPACE_APPLICATION_MANIFEST,
      env,
      settings=settings,
      tenant_id=tenant_id,
      registry_projection=registry_projection,
    )
  lifecycle = LocalWorkspaceHostLifecycle()
  lifecycle.transition_to_ready()
  lifecycle.set_executor_available(True)
  task_enricher = build_lkw_combined_task_enricher(
    env,
    default_capability="local.workspace.search",
    agent_checkpoint_store=harness_runtime.agent_checkpoint_store,
    compensation_queue_store=harness_runtime.compensation_queue_store,
    idempotency_store=harness_runtime.reliability.idempotency_store,
  )
  inner_executor = LocalWorkspaceTaskExecutor(
    build_lkw_host_task_execution(resolve_harness_host_nexus_loop_legacy(harness_runtime), env),
    task_enricher=task_enricher,
    readiness=lifecycle,
  )
  search_executor = _SearchCountingTaskExecutor(inner_executor)
  store = InMemoryDocumentStore()
  repository = ManagedWorkspaceRepository(store)
  workspace_service = ManagedWorkspaceService(repository)
  vectorstore_manager = (
    harness_runtime.env_wiring.tool_wiring.wiring_context.vectorstore_manager
  )
  embedding_manager = (
    harness_runtime.env_wiring.tool_wiring.wiring_context.embedding_manager
  )
  tool_wiring_context = harness_runtime.env_wiring.tool_wiring.wiring_context
  indexing_service = WorkspaceDocumentIndexingService(
    repository,
    inner_executor,
    indexed_vector_verifier=ManagedWorkspaceIndexedVectorVerifier(
      vectorstore_manager,
    ),
  )

  repository.put_workspace(
    Workspace(
      workspace_id=workspace_id,
      tenant_id=tenant_id,
      name="ORION Proof Workspace",
      status=WorkspaceStatus.ACTIVE,
      created_at=proof_now,
      updated_at=proof_now,
    )
  )
  source = WorkspaceSource(
    source_id=source_id,
    tenant_id=tenant_id,
    workspace_id=workspace_id,
    source_type=WorkspaceSourceType.LOCAL_FOLDER,
    path=str(docs_root.resolve()),
    recursive=True,
    status=WorkspaceSourceStatus.READY,
    created_at=proof_now,
  )
  repository.put_source(source)

  folder_indexing = LocalFolderIndexingService(
    indexing_service,
    allowlist_roots=frozenset({str(docs_root.resolve())}),
  )
  index_result = await folder_indexing.index_source(
    tenant_id=tenant_id,
    workspace_id=workspace_id,
    source=source,
    operation_id="orion-proof-index",
  )
  if index_result.documents_indexed < 1:
    raise RuntimeError("deployment_policy_index_failed")

  expected_document_id = _resolve_indexed_document_id(
    tenant_id=tenant_id,
    workspace_id=workspace_id,
    source_id=source_id,
    policy_path=policy_path,
  )
  document_ref = repository.get_document_ref(
    tenant_id=tenant_id,
    workspace_id=workspace_id,
    document_id=expected_document_id,
  )
  if document_ref is None:
    refs = repository.list_document_refs(tenant_id=tenant_id, workspace_id=workspace_id)
    if not refs:
      raise RuntimeError("deployment_policy_document_missing")
    document_ref = refs[0]

  if not indexing_service._indexed_vector_verifier.has_indexed_vectors(
    tenant_id=tenant_id,
    workspace_id=workspace_id,
    source_id=source_id,
    document_id=document_ref.document_id,
  ):
    raise RuntimeError("deployment_policy_vectors_missing")

  return IndexedProofStack(
    temp_path=temp_path,
    settings=settings,
    task_executor=inner_executor,
    search_task_executor=search_executor,
    repository=repository,
    workspace_service=workspace_service,
    indexing_service=indexing_service,
    vectorstore_manager=vectorstore_manager,
    embedding_manager=embedding_manager,
    tool_wiring_context=tool_wiring_context,
    indexed_document_id=document_ref.document_id,
    indexed_logical_path=document_ref.source_path,
    indexed_content_hash=document_ref.content_hash,
  )


def _resolve_indexed_document_id(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    policy_path: Path,
) -> str:
  from local_workspace_application.workspaces.idempotency import logical_document_id

  normalized = normalize_source_path(policy_path)
  digest = content_hash_for_file(policy_path)
  return logical_document_id(
    tenant_id=tenant_id,
    workspace_id=workspace_id,
    source_id=source_id,
    normalized_source_path=normalized,
    content_hash=digest,
    materialization_scope=None,
  )

