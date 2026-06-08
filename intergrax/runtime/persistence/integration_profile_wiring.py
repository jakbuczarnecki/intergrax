# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Resolve runtime persistence from ``IntegrationProfile`` (Phase M.8).

Prefer ``profile.resolve(IntegrationCategory.RELATIONAL_STORE)`` or
``create_sqlite_integration()`` over direct imports of runtime SQLite classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    SQLiteIntegrationBundle,
    create_sqlite_integration,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)


def sqlite_bundle_for_profile(
    profile: IntegrationProfile,
    *,
    data_dir: Path | str | None = None,
    **config_overrides: object,
) -> Optional[SQLiteIntegrationBundle]:
    slug = profile.slug_for_category(IntegrationCategory.RELATIONAL_STORE)
    if slug != "sqlite":
        return None
    opts = dict(profile.options_for_slug("sqlite"))
    opts.update(config_overrides)
    return create_sqlite_integration(data_dir=data_dir, **opts)


def open_trace_store_from_profile(
    profile: IntegrationProfile,
    *,
    db_path: Path | None = None,
) -> Any:
    if db_path is not None:
        from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_trace_store

        return create_sqlite_trace_store(db_path=db_path)
    bundle = sqlite_bundle_for_profile(profile)
    if bundle is not None:
        return bundle.trace_store
    from intergrax.runtime.nexus.tracing.store import open_run_trace_store, resolve_trace_db_path

    path = resolve_trace_db_path(None)
    return open_run_trace_store(path)


def _validating(store: Any) -> ValidatingRuntimeEventPersistence:
    if isinstance(store, ValidatingRuntimeEventPersistence):
        return store
    return ValidatingRuntimeEventPersistence(store)


def open_runtime_event_store_from_profile(
    profile: IntegrationProfile,
    *,
    db_path: Path | None = None,
) -> Any:
    if db_path is not None:
        from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_runtime_event_store

        return _validating(create_sqlite_runtime_event_store(db_path=db_path))

    bundle = sqlite_bundle_for_profile(profile)
    if bundle is not None:
        return _validating(bundle.runtime_event_store)

    doc_slug = profile.slug_for_category(IntegrationCategory.DOCUMENT_STORE)
    if doc_slug == "cassandra":
        from intergrax.integrations.providers.document_store.cassandra.adapter import (
            CassandraDocumentStore,
        )
        from intergrax.integrations.providers.document_store.cassandra.runtime_events import (
            runtime_event_persistence_from_cassandra,
        )

        resolved = profile.resolve(IntegrationCategory.DOCUMENT_STORE)
        if isinstance(resolved, CassandraDocumentStore):
            return _validating(runtime_event_persistence_from_cassandra(resolved))

    obs_slug = profile.slug_for_category(IntegrationCategory.OBSERVABILITY_BACKEND)
    if obs_slug == "elasticsearch":
        from intergrax.integrations.providers.observability_backend.elasticsearch.adapter import (
            ElasticsearchObservabilityBackend,
        )
        from intergrax.integrations.providers.observability_backend.elasticsearch.runtime_events import (
            runtime_event_persistence_from_elasticsearch_backend,
        )

        resolved = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
        if isinstance(resolved, ElasticsearchObservabilityBackend):
            return _validating(runtime_event_persistence_from_elasticsearch_backend(resolved))

    from intergrax.runtime.events.store import resolve_runtime_event_persistence, resolve_runtime_events_db_path

    path = resolve_runtime_events_db_path(None)
    return resolve_runtime_event_persistence(db_path=path)
