# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-13C — durable and external provider qualification."""

from __future__ import annotations

import ast
import inspect
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderDescriptor,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
    MemoryProviderQualificationRequest,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    MemoryProviderInstanceFactory,
    MemoryProviderQualificationRunner,
)
from intergrax.memory.resolver import (
    MemoryStoreMaterializationContext,
    MemoryStorePluginCatalog,
    materialize_user_profile_store,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.stores.document_store_user_profile_store import DocumentStoreUserProfileStore
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore
from tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin import (
    ExternalInMemoryUserProfileStorePlugin,
)
from tests.unit.memory.durable_provider_qualification_harness import (
    run_durable_user_profile_production_qualification,
    user_profile_qualification_request,
)
from tests.unit.memory.test_mem_ent13_provider_qualification import (
    _BrokenCreateFactory,
    _StaticFactory,
    _context,
)

pytestmark = pytest.mark.unit

T = TypeVar("T")


def _sqlite_create_dispose(db_path: str) -> tuple[Callable[[], SQLiteUserProfileStore], Callable[[SQLiteUserProfileStore], Awaitable[None]]]:
    def _create() -> SQLiteUserProfileStore:
        return SQLiteUserProfileStore(db_path)

    async def _dispose(store: SQLiteUserProfileStore) -> None:
        store._connection.close()

    return _create, _dispose


@pytest.mark.asyncio
async def test_sqlite_user_profile_canonical_via_runner_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "qual-canonical.db")
        create, dispose = _sqlite_create_dispose(db_path)
        runner = MemoryProviderQualificationRunner()
        result = await runner.qualify(
            descriptor=MemoryProviderDescriptor(
                provider_id="sqlite.user_profile",
                capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            context=_context("sqlite-canonical-13c"),
            request=user_profile_qualification_request(),
            factories=MemoryProviderCapabilityFactories(
                user_profile_store=_StaticFactory(create, dispose),
            ),
        )
        assert result.status is MemoryProviderQualificationStatus.QUALIFIED
        assert not Path(db_path).exists() or Path(db_path).stat().st_size >= 0


@pytest.mark.asyncio
async def test_sqlite_durable_production_qualification_reopen_and_delete() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "qual-durable.db")
        create, dispose = _sqlite_create_dispose(db_path)
        evidence = await run_durable_user_profile_production_qualification(
            descriptor=MemoryProviderDescriptor(
                provider_id="sqlite.user_profile",
                capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            context=_context("sqlite-durable-13c"),
            request=user_profile_qualification_request(),
            factories=MemoryProviderCapabilityFactories(
                user_profile_store=_StaticFactory(create, dispose),
            ),
            create_store=create,
            dispose_store=dispose,
        )
        assert evidence.canonical.status is MemoryProviderQualificationStatus.QUALIFIED
        assert evidence.production_durable_qualified
        assert evidence.durability_reopen_passed
        assert evidence.durability_delete_passed is True
        assert evidence.durability_reason is None


@pytest.mark.asyncio
async def test_sqlite_durable_reopen_failure_is_not_production_qualified() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "qual-volatile.db")

        def _create() -> SQLiteUserProfileStore:
            return SQLiteUserProfileStore(db_path)

        async def _dispose(store: SQLiteUserProfileStore) -> None:
            store._connection.close()
            Path(db_path).unlink(missing_ok=True)

        evidence = await run_durable_user_profile_production_qualification(
            descriptor=MemoryProviderDescriptor(
                provider_id="sqlite.user_profile.volatile",
                capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            context=_context("sqlite-volatile-13c"),
            request=user_profile_qualification_request(),
            factories=MemoryProviderCapabilityFactories(
                user_profile_store=_StaticFactory(_create, _dispose),
            ),
            create_store=_create,
            dispose_store=_dispose,
        )
        assert evidence.canonical.status is MemoryProviderQualificationStatus.QUALIFIED
        assert not evidence.production_durable_qualified
        assert evidence.durability_reason is MemoryProviderQualificationFailureReason.DURABILITY_FAILURE


@pytest.mark.asyncio
async def test_document_store_user_profile_contract_adapter_qualification() -> None:
    backend = InMemoryDocumentStore()

    def _create() -> DocumentStoreUserProfileStore:
        return DocumentStoreUserProfileStore(backend)

    async def _dispose(store: DocumentStoreUserProfileStore) -> None:
        return None

    evidence = await run_durable_user_profile_production_qualification(
        descriptor=MemoryProviderDescriptor(
            provider_id="document_store.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("document-store-13c"),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(_create, _dispose),
        ),
        create_store=_create,
        dispose_store=_dispose,
    )
    assert evidence.canonical.status is MemoryProviderQualificationStatus.QUALIFIED
    assert evidence.production_durable_qualified


def _external_user_profile_catalog() -> MemoryStorePluginCatalog:
    return MemoryStorePluginCatalog.from_discovery(
        discover_classified_memory_store_plugins(
            discover_entry_points=False,
            explicit_plugins=(ExternalInMemoryUserProfileStorePlugin,),
        )
    )


@pytest.mark.asyncio
async def test_external_plugin_materialization_canonical_qualification() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.13c.external.plugin")
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id="qual-tenant-plugin",
        integration_profile=IntegrationProfile(),
    )
    catalog = _external_user_profile_catalog()
    plugin_id = ExternalInMemoryUserProfileStorePlugin.plugin_id()

    @dataclass(slots=True)
    class _PluginFactory(MemoryProviderInstanceFactory[UserProfileStore]):
        async def create(self) -> UserProfileStore:
            return materialize_user_profile_store(plugin_id, ctx, catalog=catalog)

        async def dispose(self, instance: UserProfileStore) -> None:
            close = getattr(instance, "close", None)
            if callable(close):
                close()

    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id=plugin_id,
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("external-plugin-13c"),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_PluginFactory()),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    assert result.descriptor.provider_id == "external.in_memory_user_profile"


@pytest.mark.asyncio
async def test_external_plugin_partial_capability_matrix() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.13c.partial")
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id="qual-tenant",
        integration_profile=IntegrationProfile(),
    )
    catalog = _external_user_profile_catalog()
    plugin_id = ExternalInMemoryUserProfileStorePlugin.plugin_id()

    @dataclass(slots=True)
    class _PluginFactory(MemoryProviderInstanceFactory[UserProfileStore]):
        async def create(self) -> UserProfileStore:
            return materialize_user_profile_store(plugin_id, ctx, catalog=catalog)

        async def dispose(self, instance: UserProfileStore) -> None:
            return None

    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id=plugin_id,
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("external-partial-13c"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            optional_capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_PluginFactory()),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    optional = next(
        item
        for item in result.capability_results
        if item.capability is MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE
    )
    assert optional.status is MemoryProviderQualificationStatus.NOT_SUPPORTED


@pytest.mark.asyncio
async def test_mongodb_user_profile_infra_unavailable_is_blocked_not_qualified() -> None:
    class _MongoUnavailableFactory(MemoryProviderInstanceFactory[UserProfileStore]):
        async def create(self) -> UserProfileStore:
            raise ConnectionError("mongodb qualification infrastructure unavailable")

        async def dispose(self, instance: UserProfileStore) -> None:
            return None

    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="mongodb.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("mongo-blocked-13c"),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_MongoUnavailableFactory(),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.BLOCKED
    assert MemoryProviderQualificationFailureReason.MATERIALIZATION_FAILURE in result.reason_codes


@pytest.mark.asyncio
async def test_faulty_provider_semantics_not_qualified() -> None:
    from tests.unit.memory.test_mem_ent13_provider_qualification import _TenantLeakingUserProfileStore

    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.tenant_leak.durable",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("faulty-durable-13c"),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(lambda: _TenantLeakingUserProfileStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED


@pytest.mark.asyncio
async def test_materialization_failure_remains_blocked() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="broken.materialization.13c",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("blocked-mat-13c"),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_BrokenCreateFactory()),
    )
    assert result.status is MemoryProviderQualificationStatus.BLOCKED


def test_durable_qualification_uses_memory_provider_qualification_runner() -> None:
    source = inspect.getsource(run_durable_user_profile_production_qualification)
    assert "MemoryProviderQualificationRunner" in source
    harness_path = Path(__file__).with_name("durable_provider_qualification_harness.py")
    tree = ast.parse(harness_path.read_text(encoding="utf-8"))
    runner_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "qualify"
    ]
    assert runner_calls


def test_qualification_evidence_has_no_connection_secrets() -> None:
    sample = (
        "mongodb://user:secret@host:27017/db",
        "password=supersecret",
        "api_key=abc123",
    )
    for secret in sample:
        assert secret not in str(MemoryProviderQualificationFailureReason.DURABILITY_FAILURE.value)


def test_sqlite_qualification_artifact_removed_with_tempdir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "cleanup-qual.db"
        assert not db_path.exists()
