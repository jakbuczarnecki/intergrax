# © Artur Czarnecki. All rights reserved.

"""Isolated LKW runtime construction for model runtime portability proof."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.testclient import TestClient
from intergrax.applications._shared.harness_host_runtime import (
    build_harness_host_runtime,
)
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)

from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.lkw_task_enricher import (
    build_lkw_combined_task_enricher,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.model_runtime_proof.config import (
    ModelRuntimeProofConfig,
    apply_env,
    materialize_provider_env,
)
from local_workspace_application.serving import workspace_routes
from local_workspace_application.serving.workspace_routes import (
    mount_managed_workspace_routes,
)
from local_workspace_application.workspaces.ask_service import WorkspaceAskService
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import (
    build_managed_workspace_sync_runtime,
)
from local_workspace_application.workspaces.sync_service import (
    ManagedWorkspaceSyncService,
)


@dataclass
class ProofRuntimeSession:
    provider: Literal["ollama", "vllm"] | None
    settings: LocalWorkspaceBackendSettings
    app: FastAPI
    client: TestClient
    repository: ManagedWorkspaceRepository
    task_executor: LocalWorkspaceTaskExecutor
    ask_service: WorkspaceAskService
    harness_runtime: Any
    sync_runtime: Any
    wiring_context: Any
    embedding_manager: Any
    previous_env: dict[str, str | None]

    def close(self) -> None:
        self.client.close()
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _prepare_data_home(config: ModelRuntimeProofConfig) -> Path:
    if config.data_home:
        root = Path(config.data_home)
    else:
        root = Path.cwd() / "build" / "lkw-model-runtime-proof"
    for sub in ("data", "sqlite", "shadow", "docs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def build_proof_runtime_session(
    config: ModelRuntimeProofConfig,
    *,
    provider: Literal["ollama", "vllm"] | None,
    document_store: InMemoryDocumentStore | None = None,
    data_home: Path | None = None,
) -> ProofRuntimeSession:
    tracked_keys = (
        "INTERGRAX_LLM_PROVIDER",
        "INTERGRAX_LLM_MODEL",
        "INTERGRAX_DEFAULT_OLLAMA_MODEL",
        "INTERGRAX_DEFAULT_VLLM_MODEL",
        "INTERGRAX_DEFAULT_VLLM_BASE_URL",
        "OLLAMA_HOST",
        "LOCAL_WORKSPACE_VECTOR_STORE",
        "LOCAL_WORKSPACE_ENABLE_RAG",
        "LOCAL_WORKSPACE_ENABLE_RAG_INGEST",
        "LKW_DATA_HOME",
        "INTERGRAX_SQLITE_DATA_DIR",
        "INTERGRAX_SHADOW_ROOT",
        "INTERGRAX_ALLOWED_READ_ROOTS",
    )
    previous_env = {key: os.environ.get(key) for key in tracked_keys}

    if provider is not None:
        apply_env(materialize_provider_env(provider=provider, config=config))

    root = data_home or _prepare_data_home(config)
    user_docs = root / "docs"
    sqlite_dir = root / "sqlite"
    shadow_dir = root / "shadow"

    os.environ["LOCAL_WORKSPACE_VECTOR_STORE"] = config.vector_store
    os.environ["LOCAL_WORKSPACE_ENABLE_RAG"] = "true"
    os.environ["LOCAL_WORKSPACE_ENABLE_RAG_INGEST"] = "true"
    os.environ["LKW_DATA_HOME"] = str(root)
    os.environ["INTERGRAX_SQLITE_DATA_DIR"] = str(sqlite_dir)
    os.environ["INTERGRAX_SHADOW_ROOT"] = str(shadow_dir)
    os.environ["INTERGRAX_ALLOWED_READ_ROOTS"] = str(user_docs.resolve())

    store = document_store or InMemoryDocumentStore()
    workspace_routes.resolve_managed_workspace_document_store = (
        lambda document_store=None: store
    )  # type: ignore[method-assign]

    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    harness_runtime = build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        settings=settings,
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
    task_executor = LocalWorkspaceTaskExecutor(
        harness_runtime.nexus_loop,
        task_enricher=task_enricher,
        readiness=lifecycle,
    )
    repository = ManagedWorkspaceRepository(store)
    sync = ManagedWorkspaceSyncService(repository, task_executor)
    sync_runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repository,
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=task_executor,
        settings=settings,
        repository=repository,
        sync_runtime=sync_runtime,
        llm_adapter=None,
        vectorstore_manager=harness_runtime.env_wiring.tool_wiring.wiring_context.vectorstore_manager,
        object_storage=harness_runtime.env_wiring.tool_wiring.wiring_context.object_storage,
    )
    ask_service: WorkspaceAskService = app.state.lkw_ask_service
    wiring_context = harness_runtime.env_wiring.tool_wiring.wiring_context
    embedding_manager = wiring_context.embedding_manager
    client = TestClient(app)
    return ProofRuntimeSession(
        provider=provider,
        settings=settings,
        app=app,
        client=client,
        repository=repository,
        task_executor=task_executor,
        ask_service=ask_service,
        harness_runtime=harness_runtime,
        sync_runtime=sync_runtime,
        wiring_context=wiring_context,
        embedding_manager=embedding_manager,
        previous_env=previous_env,
    )


def resolve_provider_adapter(
    *,
    provider: Literal["ollama", "vllm"],
    config: ModelRuntimeProofConfig,
):
    env = materialize_provider_env(provider=provider, config=config)
    previous = {key: os.environ.get(key) for key in env}
    apply_env(env)
    try:
        return resolve_llm_adapter(build_local_workspace_environment_profile())
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
