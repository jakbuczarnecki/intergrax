# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Resolve runtime persistence from ``IntegrationProfile`` (Phase M.8).

Prefer ``resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)`` or
``create_sqlite_integration()`` over direct imports of runtime SQLite classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    SQLiteIntegrationBundle,
    create_sqlite_integration,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceStore


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
) -> RunTraceStore:
    if db_path is not None:
        from intergrax.integrations.providers.relational_store.sqlite.opens import open_trace_store_at

        return open_trace_store_at(db_path)
    bundle = sqlite_bundle_for_profile(profile)
    if bundle is not None:
        return bundle.trace_store
    from intergrax.runtime.nexus.tracing.store import open_run_trace_store, resolve_trace_db_path

    path = resolve_trace_db_path(None)
    return open_run_trace_store(path)


def _validating(store: RuntimeEventPersistence) -> ValidatingRuntimeEventPersistence:
    if isinstance(store, ValidatingRuntimeEventPersistence):
        return store
    return ValidatingRuntimeEventPersistence(store)


def open_runtime_event_store_from_profile(
    profile: IntegrationProfile,
    *,
    db_path: Path | None = None,
) -> RuntimeEventPersistence:
    if db_path is not None:
        from intergrax.integrations.providers.relational_store.sqlite.opens import (
            open_runtime_event_store_at,
        )

        return _validating(open_runtime_event_store_at(db_path))

    bundle = sqlite_bundle_for_profile(profile)
    if bundle is not None:
        return _validating(bundle.runtime_event_store)

    doc_slug = profile.slug_for_category(IntegrationCategory.DOCUMENT_STORE)
    if doc_slug == "cassandra":
        from intergrax.integrations.providers.document_store.cassandra.integration import (
            CassandraDocumentStoreIntegration,
        )
        from intergrax.integrations.providers.document_store.cassandra.runtime_events import (
            runtime_event_persistence_from_document_store,
        )

        resolved = resolve_from_profile(profile, IntegrationCategory.DOCUMENT_STORE)
        if isinstance(resolved, CassandraDocumentStoreIntegration):
            return _validating(
                runtime_event_persistence_from_document_store(resolved.as_document_store()),
            )

    obs_slug = profile.slug_for_category(IntegrationCategory.OBSERVABILITY_BACKEND)
    if obs_slug == "elasticsearch":
        from intergrax.integrations.providers.observability_backend.elasticsearch.integration import (
            ElasticsearchObservabilityIntegration,
        )
        from intergrax.integrations.providers.observability_backend.elasticsearch.runtime_events import (
            runtime_event_persistence_from_elasticsearch_backend,
        )

        resolved = resolve_from_profile(profile, IntegrationCategory.OBSERVABILITY_BACKEND)
        if isinstance(resolved, ElasticsearchObservabilityIntegration):
            return _validating(runtime_event_persistence_from_elasticsearch_backend(resolved))

    from intergrax.runtime.events.store import resolve_runtime_event_persistence, resolve_runtime_events_db_path

    path = resolve_runtime_events_db_path(None)
    store = resolve_runtime_event_persistence(db_path=path)
    if store is not None:
        return store
    from intergrax.runtime.events.store import open_runtime_event_store

    return _validating(open_runtime_event_store(path))
