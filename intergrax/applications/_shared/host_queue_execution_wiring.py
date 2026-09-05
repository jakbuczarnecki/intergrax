# © Artur Czarnecki. All rights reserved.

"""Resolve platform queue-worker storage from Tier-3 harness host runtime (DIAG-1I-R2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.execution_terminal import ExecutionTerminalPersistenceProvider
from intergrax.distributed.contracts.kv_store import (
    DistributedKVStore,
    DistributedKVStoreProvider,
)
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.providers.document_store.mongodb.manifest import (
    MANIFEST as MONGODB_DOCUMENT_STORE_MANIFEST,
)
from intergrax.integrations.registry.catalog_manifests import REDIS
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)


@dataclass(frozen=True, slots=True)
class HostQueueExecutionDependencies:
    """Platform storage required for queue-enabled Tier-3 hosts."""

    kv_store: DistributedKVStore
    causal_evidence_persistence: CausalEvidencePersistence


def apply_queue_worker_integration_profile(
    profile: IntegrationProfile,
) -> IntegrationProfile:
    """
    Merge canonical platform queue-worker storage bindings when absent.

  Production deploys with ``include_queue_worker`` should bind Redis (identity) and a
  document store (causal evidence) through the integration profile.
    """
    redis_binding = IntegrationBinding.from_manifest(REDIS)
    document_store_binding = IntegrationBinding.from_manifest(MONGODB_DOCUMENT_STORE_MANIFEST)
    options = dict(profile.options or {})
    if profile.key_value_cache is None:
        options.setdefault(REDIS.slug, {})
    if profile.document_store is None:
        options.setdefault(MONGODB_DOCUMENT_STORE_MANIFEST.slug, {})
    return profile.model_copy(
        update={
            "key_value_cache": profile.key_value_cache or redis_binding,
            "document_store": profile.document_store or document_store_binding,
            "options": options,
        },
    )


def apply_queue_worker_environment_profile(
    environment: ApplicationEnvironmentProfile,
) -> ApplicationEnvironmentProfile:
    """Attach queue-worker integration bindings to an environment profile."""
    integration_profile = environment.integration_profile
    if integration_profile is None:
        raise ValueError(
            "queue-enabled host requires integration_profile with platform "
            "key_value_cache and document_store bindings",
        )
    return environment.model_copy(
        update={
            "integration_profile": apply_queue_worker_integration_profile(
                integration_profile,
            ),
            "reliability_profile": environment.reliability_profile.model_copy(
                update={
                    "execution_terminal_persistence_provider": (
                        environment.reliability_profile.execution_terminal_persistence_provider
                        or ExecutionTerminalPersistenceProvider.KV
                    ),
                },
            ),
        },
    )


def _resolve_distributed_kv_store(key_value_cache: object) -> DistributedKVStore:
    if isinstance(key_value_cache, DistributedKVStore):
        return key_value_cache
    if isinstance(key_value_cache, DistributedKVStoreProvider):
        kv_store = key_value_cache.kv_store
        if isinstance(kv_store, DistributedKVStore):
            return kv_store
        raise ValueError(
            "queue-enabled host requires platform key_value_cache provider exposing "
            "DistributedKVStore via kv_store",
        )
    raise ValueError(
        "queue-enabled host requires platform key_value_cache exposing "
        "DistributedKVStore (directly or via DistributedKVStoreProvider)",
    )


def resolve_host_queue_execution_dependencies(
    runtime: HarnessHostRuntime,
) -> HostQueueExecutionDependencies:
    """
    Resolve mandatory queue-worker storage from harness host runtime wiring.

    Fails closed when the platform host runtime lacks durable identity KV or
    conditional document-store causal evidence persistence.
    """
    build_context = runtime.env_wiring.build_context
    wiring_context = build_context.tool_wiring_context
    if wiring_context is None:
        raise ValueError(
            "queue-enabled host requires tool_wiring_context with platform storage "
            "capabilities for background queue execution",
        )
    if wiring_context.key_value_cache is None:
        raise ValueError(
            "queue-enabled host requires platform key_value_cache "
            "(DistributedKVStore) for Celery worker identity persistence",
        )
    if wiring_context.document_store is None:
        raise ValueError(
            "queue-enabled host requires platform document_store "
            "(ConditionalDocumentStore) for causal evidence persistence",
        )
    document_store: DocumentStore = wiring_context.document_store
    assert_conditional_document_store(document_store)
    return HostQueueExecutionDependencies(
        kv_store=_resolve_distributed_kv_store(wiring_context.key_value_cache),
        causal_evidence_persistence=wire_causal_evidence_persistence(
            document_store=document_store,
        ),
    )


__all__ = [
    "HostQueueExecutionDependencies",
    "apply_queue_worker_environment_profile",
    "apply_queue_worker_integration_profile",
    "resolve_host_queue_execution_dependencies",
]
