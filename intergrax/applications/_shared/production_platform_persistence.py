# © Artur Czarnecki. All rights reserved.

"""Canonical durable platform primitives for reference production host composition."""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from intergrax.applications._shared.profile_resolution.wiring import (
    EffectiveProfilePersistenceWiring,
    resolve_effective_profile_persistence_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.attempt_lifecycle import AttemptLifecyclePersistenceProvider
from intergrax.contracts.execution_terminal import ExecutionTerminalPersistenceProvider
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.distributed.providers.sqlite_kv_store import build_sqlite_distributed_kv_store
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore

if TYPE_CHECKING:
    from intergrax.applications._shared.production_process_composition import (
        ProductionProcessComposition,
    )


@dataclass(frozen=True, slots=True)
class ProductionPlatformPersistence:
    """Shared durable platform primitives owned by one process composition root."""

    kv_store: DistributedKVStore
    document_store: DocumentStore
    db_path: Path


def build_reference_production_platform_persistence(
    *,
    db_path: Path | None = None,
) -> ProductionPlatformPersistence:
    """Construct one durable reference-production platform persistence bundle."""
    resolved_path = db_path or (
        Path(tempfile.gettempdir())
        / "intergrax"
        / "platform-kv"
        / f"{uuid.uuid4().hex}.db"
    )
    return ProductionPlatformPersistence(
        kv_store=build_sqlite_distributed_kv_store(resolved_path),
        document_store=InMemoryDocumentStore(),
        db_path=resolved_path,
    )


def resolve_effective_profile_persistence_for_platform(
    *,
    production_mode: bool,
    platform_persistence: ProductionPlatformPersistence,
) -> EffectiveProfilePersistenceWiring:
    """Wire effective profile stores from one composition-owned platform primitive."""
    return resolve_effective_profile_persistence_wiring(
        production_mode=production_mode,
        kv_store=platform_persistence.kv_store,
    )


def resolve_reference_production_strict_host_environment(
    environment: ApplicationEnvironmentProfile,
) -> ApplicationEnvironmentProfile:
    """Disambiguate durable provider selection when KV and document stores coexist."""
    reliability = environment.reliability_profile.model_copy(
        update={
            "attempt_lifecycle_persistence_provider": (
                environment.reliability_profile.attempt_lifecycle_persistence_provider
                or AttemptLifecyclePersistenceProvider.KV
            ),
            "execution_terminal_persistence_provider": (
                environment.reliability_profile.execution_terminal_persistence_provider
                or ExecutionTerminalPersistenceProvider.KV
            ),
        },
    )
    return environment.model_copy(
        update={"governance": environment.governance.model_copy(update={"reliability": reliability})},
    )


def resolve_harness_host_profile_persistence_kwargs(
    *,
    production_mode: bool,
    platform_persistence: ProductionPlatformPersistence,
) -> dict[str, DistributedKVStore | DocumentStore]:
    """Resolve harness host kwargs for shared composition profile persistence."""
    resolve_effective_profile_persistence_for_platform(
        production_mode=production_mode,
        platform_persistence=platform_persistence,
    )
    return {
        "key_value_cache": platform_persistence.kv_store,
        "document_store": platform_persistence.document_store,
    }


def resolve_harness_host_profile_persistence_kwargs_from_composition(
    *,
    production_mode: bool,
    composition: ProductionProcessComposition,
) -> dict[str, DistributedKVStore | DocumentStore]:
    """Resolve harness host persistence kwargs from one activated process composition."""
    return resolve_harness_host_profile_persistence_kwargs(
        production_mode=production_mode,
        platform_persistence=composition.agent_platform_runtime.platform_persistence,
    )


__all__ = [
    "ProductionPlatformPersistence",
    "build_reference_production_platform_persistence",
    "resolve_effective_profile_persistence_for_platform",
    "resolve_harness_host_profile_persistence_kwargs",
    "resolve_harness_host_profile_persistence_kwargs_from_composition",
]
