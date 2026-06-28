# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment profile for local_workspace_application."""

from __future__ import annotations

import os

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
    HostDeploymentProfile,
)
from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    GraphEdge,
    GraphNode,
)
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.catalog_manifests import DOCLING, INMEMORY, OTEL, QDRANT, REDIS, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.registry.profile import llm_profile_from_env
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _local_vector_store_manifest():
    raw = (os.getenv("LOCAL_WORKSPACE_VECTOR_STORE") or "qdrant").strip().lower()
    if raw == "inmemory":
        return INMEMORY
    if raw == "qdrant":
        return QDRANT
    raise ValueError("LOCAL_WORKSPACE_VECTOR_STORE must be one of: qdrant, inmemory.")


def build_local_workspace_pipeline_graph_spec() -> ApplicationGraphSpec:
    """Index → search → synthesize orchestration graph for ``local.workspace.pipeline``."""
    return ApplicationGraphSpec(
        nodes=[
            GraphNode(agent_id="local_indexer", contract_id="LocalIndexerAgent"),
            GraphNode(agent_id="local_search", contract_id="LocalSearchAgent"),
            GraphNode(agent_id="local_synthesizer", contract_id="LocalSynthesizerAgent"),
        ],
        edges=[
            GraphEdge(source_agent_id="local_indexer", target_agent_id="local_search"),
            GraphEdge(source_agent_id="local_search", target_agent_id="local_synthesizer"),
        ],
        trigger_capabilities=["local.workspace.pipeline"],
    )


def build_local_workspace_integration_profile() -> IntegrationProfile:
    """Local-first product integrations for LKW.

    Defaults are intentionally local/persistent: SQLite for relational state and
    Qdrant for the RAG vector store. Redis is opt-in until background ingest or
    queue-backed workflows are enabled.
    """

    vector_store = _local_vector_store_manifest()
    enable_redis = _env_bool("LOCAL_WORKSPACE_ENABLE_REDIS", default=False)
    options: dict[str, dict[str, object]] = {
        OTEL.slug: {},
        SQLITE.slug: {},
        vector_store.slug: {},
    }
    if enable_redis:
        options[REDIS.slug] = {}
    return IntegrationProfile(
        relational_store=SQLITE,
        vector_store=vector_store,
        key_value_cache=REDIS if enable_redis else None,
        document_parser=DOCLING,
        observability_backend=OTEL,
        options=options,
    )


def build_local_workspace_environment_profile(
    settings: LocalWorkspaceBackendSettings | None = None,
) -> ApplicationEnvironmentProfile:
    profile = (
        ApplicationEnvironmentProfile.product_defaults(
            profile_id="local_workspace.product",
            skill_bundles=["harness", "local"],
        )
        .model_copy(
            update={
                "integration_profile": build_local_workspace_integration_profile(),
                "context_profile": ContextProfile(
                    enable_rag=True if settings is None else settings.enable_rag,
                    enable_websearch=False,
                ),
                "llm_profile": llm_profile_from_env(prefix="INTERGRAX_LLM"),
            }
        )
        .with_harness_memory()
    )
    profile.observability_profile.otel_enabled = True
    profile.observability_profile.debug_surface_override = True
    profile.host_deployment_profile = HostDeploymentProfile(
        lkw_hybrid_daemon_enabled=True,
        lkw_daemon_bind_host="127.0.0.1",
        lkw_daemon_port=8020,
    )
    otel_backend = IntegrationBinding.from_manifest(OTEL)
    profile.integration_profile = profile.integration_profile.model_copy(
        update={
            "observability_backend": otel_backend,
            "options": {**profile.integration_profile.options, OTEL.slug: {}},
        },
    )
    profile.graph_spec = build_local_workspace_pipeline_graph_spec()
    return profile
