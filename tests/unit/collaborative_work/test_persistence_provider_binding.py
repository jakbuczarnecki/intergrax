# © Artur Czarnecki. All rights reserved.

"""Tests for typed Collaborative Work persistence provider binding (PROVIDER-QUAL-3B-R2)."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.materialization_factory import (
    CollaborativeWorkMaterializationBinding,
    CollaborativeWorkPersistenceFactory,
)
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationEntry,
)
from intergrax.integrations.providers.relational_store.postgresql.adapter import (
    _PostgreSQLRelationalStore,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.integration import (
    PostgresqlRelationalStoreIntegration,
)
from intergrax.integrations.providers.relational_store.sqlite.adapter import (
    _SQLiteRelationalStore,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SqliteRelationalStoreIntegration,
)
from intergrax.integrations.providers.relational_store.postgresql.register import (
    register_postgresql_integration,
)
from intergrax.integrations.providers.relational_store.sqlite.register import (
    register_sqlite_integration,
)
from intergrax.integrations.registry.catalog import clear_catalog, get_entry, register_integration
from intergrax.integrations.registry.catalog_manifests import POSTGRESQL, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_FORBIDDEN_CALLS = frozenset({"getattr", "setattr", "hasattr", "vars"})
_FORBIDDEN_ATTRIBUTES = frozenset({"__dict__", "__setattr__"})
_VENDOR_LITERALS = frozenset({"postgresql", "sqlite", "oracle"})
_BINDING_FILES = (
    "intergrax/collaborative_work/persistence_provider.py",
)


class _FakeCursor:
    def fetchall(self) -> list[dict[str, Any]]:
        return []


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Sequence[Any]]] = []
        self.committed = 0
        self.closed = False

    def execute(self, sql: str, params: Sequence[Any] = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        return _FakeCursor()

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        self.closed = True


class _CountingConnectionFactory:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection
        self.invocations = 0

    def __call__(self) -> _FakeConnection:
        self.invocations += 1
        return self.connection


def _in_memory_collaborative_work_repositories() -> CollaborativeWorkRepositories:
    return CollaborativeWorkRepositories(
        membership=InMemoryWorkspaceMembershipRepository(),
        delegation=InMemoryAuthorityDelegationRepository(),
        principal_authority=InMemoryPrincipalAuthorityRepository(),
        policy=InMemoryCollaborativePolicyRepository(),
        operation_profile=InMemoryCollaborativeOperationPolicyProfileRepository(),
        store=_NoopCollaborativeWorkStore(),
    )


class _NoopCollaborativeWorkStore:
    def close(self) -> None:
        return None


class _UnsupportedRelationalStore:
    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        return None

    def fetch_all(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Sequence[Mapping[str, Any]]:
        return []

    def close(self) -> None:
        return None


class _UnsupportedRelationalStoreFactory:
    def __init__(self) -> None:
        self.generic_invocations = 0

    def __call__(self, **_kwargs: object) -> _UnsupportedRelationalStore:
        self.generic_invocations += 1
        return _UnsupportedRelationalStore()


class _FutureVendorRelationalStoreFactory:
    def __init__(self) -> None:
        self.generic_invocations = 0
        self.materialization_invocations = 0

    def __call__(self, **_kwargs: object) -> _FutureVendorRelationalStore:
        self.generic_invocations += 1
        return _FutureVendorRelationalStore()

    def materialize_collaborative_work_repositories(
        self,
        binding: CollaborativeWorkMaterializationBinding,
    ) -> CollaborativeWorkRepositories:
        del binding
        self.materialization_invocations += 1
        return _in_memory_collaborative_work_repositories()


class _TypeErrorRelationalStoreFactory:
    def __init__(self) -> None:
        self.materialization_invocations = 0

    def __call__(self, **_kwargs: object) -> _UnsupportedRelationalStore:
        return _UnsupportedRelationalStore()

    def materialize_collaborative_work_repositories(
        self,
        binding: CollaborativeWorkMaterializationBinding,
    ) -> CollaborativeWorkRepositories:
        del binding
        self.materialization_invocations += 1
        raise TypeError("internal factory bug")


class _FutureVendorRelationalStore:
    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        return None

    def fetch_all(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Sequence[Mapping[str, Any]]:
        return []

    def close(self) -> None:
        return None


def _collect_typeerror_capability_probing_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        exc_types: list[str] = []
        if isinstance(node.type, ast.Name):
            exc_types.append(node.type.id)
        elif isinstance(node.type, ast.Tuple):
            for element in node.type.elts:
                if isinstance(element, ast.Name):
                    exc_types.append(element.id)
        if "TypeError" in exc_types:
            violations.append(f"{path.name}:{node.lineno} catches TypeError")
    return violations


def _mock_psycopg_import() -> tuple[Any, Any, Any, Any]:
    fake_sql = MagicMock()
    fake_sql.SQL.return_value.format.return_value = "SET search_path TO tenant, public"
    return MagicMock(), MagicMock(), MagicMock(), fake_sql


@pytest.fixture
def _mock_psycopg() -> None:
    with patch(
        "intergrax.integrations.providers.relational_store.postgresql.session.import_psycopg",
        return_value=_mock_psycopg_import(),
    ):
        yield


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _collect_dynamic_wiring_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                violations.append(f"{path.name}:{node.lineno} calls {node.func.id}()")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "__setattr__"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "object"
            ):
                violations.append(f"{path.name}:{node.lineno} calls object.__setattr__()")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            violations.append(f"{path.name}:{node.lineno} references .{node.attr}")
    return violations


def _collect_vendor_dispatch_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            if isinstance(comparator, ast.Constant) and comparator.value in _VENDOR_LITERALS:
                violations.append(
                    f"{path.name}:{node.lineno} compares against vendor literal "
                    f"{comparator.value!r}"
                )
            if isinstance(operator, ast.In) and isinstance(comparator, (ast.List, ast.Tuple)):
                for element in comparator.elts:
                    if isinstance(element, ast.Constant) and element.value in _VENDOR_LITERALS:
                        violations.append(
                            f"{path.name}:{node.lineno} membership test against vendor literal "
                            f"{element.value!r}"
                        )
        if isinstance(node.left, ast.Constant) and node.left.value in _VENDOR_LITERALS:
            violations.append(
                f"{path.name}:{node.lineno} compares vendor literal {node.left.value!r}"
            )
    return violations


@pytest.mark.parametrize("relative_path", _BINDING_FILES)
def test_central_binding_has_no_dynamic_wiring(relative_path: str) -> None:
    path = _repo_root() / relative_path
    violations = _collect_dynamic_wiring_violations(path)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("relative_path", _BINDING_FILES)
def test_central_binding_has_no_vendor_literal_dispatch(relative_path: str) -> None:
    path = _repo_root() / relative_path
    violations = _collect_vendor_dispatch_violations(path)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("relative_path", _BINDING_FILES)
def test_central_binding_has_no_typeerror_capability_probing(relative_path: str) -> None:
    path = _repo_root() / relative_path
    violations = _collect_typeerror_capability_probing_violations(path)
    assert not violations, "\n".join(violations)


def test_sqlite_profile_materializes_collaborative_work_repositories(tmp_path: Path) -> None:
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE.slug: {"data_dir": str(tmp_path)}},
    )

    with patch(
        "intergrax.integrations.providers.relational_store.sqlite.bundle._SQLiteRelationalStore.connect",
    ) as generic_connect:
        bundle = resolve_collaborative_work_repositories(profile)

    generic_connect.assert_not_called()
    assert isinstance(bundle, CollaborativeWorkRepositories)
    assert bundle.membership.capabilities.durable is True
    assert bundle.membership.capabilities.reference_only is False
    bundle.close()


def test_postgresql_profile_materializes_collaborative_work_repositories() -> None:
    register_postgresql_integration()
    connection = _FakeConnection()
    counting_factory = _CountingConnectionFactory(connection)
    profile = IntegrationProfile(
        relational_store=POSTGRESQL,
        options={
            POSTGRESQL.slug: {
                "dsn": "postgresql://localhost/test",
                "connection_factory": counting_factory,
            }
        },
    )

    with (
        patch(
            "intergrax.integrations.providers.relational_store.postgresql.opens.open_postgresql_relational_store",
        ) as generic_open,
        patch(
            "intergrax.collaborative_work.persistence.open_postgresql_collaborative_work_repositories",
            return_value=_in_memory_collaborative_work_repositories(),
        ) as collaborative_open,
    ):
        bundle = resolve_collaborative_work_repositories(profile)

    generic_open.assert_not_called()
    collaborative_open.assert_called_once()
    assert counting_factory.invocations == 0
    assert isinstance(bundle, CollaborativeWorkRepositories)
    bundle.close()


def test_postgresql_profile_preserves_injected_connection_factory(_mock_psycopg: None) -> None:
    register_postgresql_integration()
    connection = _FakeConnection()
    counting_factory = _CountingConnectionFactory(connection)
    profile = IntegrationProfile(
        relational_store=POSTGRESQL,
        options={
            POSTGRESQL.slug: {
                "dsn": "postgresql://localhost/test",
                "connection_factory": counting_factory,
            }
        },
    )

    bundle = resolve_collaborative_work_repositories(profile)

    assert counting_factory.invocations >= 1
    assert connection.executed
    bundle.close()
    assert connection.closed is True


def test_unsupported_relational_provider_fails_before_factory_invocation() -> None:
    unsupported_factory = _UnsupportedRelationalStoreFactory()
    register_integration(
        IntegrationEntry(
            slug="unsupported-relational",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=unsupported_factory,
        )
    )
    profile = IntegrationProfile(relational_store="unsupported-relational")

    with pytest.raises(IntegrationConfigurationError, match="does not implement Collaborative Work"):
        resolve_collaborative_work_repositories(profile)

    assert unsupported_factory.generic_invocations == 0


def test_future_vendor_uses_same_bridge_without_central_dispatch_changes() -> None:
    future_factory = _FutureVendorRelationalStoreFactory()
    register_integration(
        IntegrationEntry(
            slug="future_vendor_x",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=future_factory,
        )
    )
    profile = IntegrationProfile(relational_store="future_vendor_x")

    bundle = resolve_collaborative_work_repositories(profile)

    assert future_factory.generic_invocations == 0
    assert future_factory.materialization_invocations == 1
    assert isinstance(bundle, CollaborativeWorkRepositories)
    assert bundle.membership.capabilities.durable is False
    bundle.close()


def test_provider_factory_internal_typeerror_propagates_once() -> None:
    failing_factory = _TypeErrorRelationalStoreFactory()
    register_integration(
        IntegrationEntry(
            slug="typeerror-relational",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=failing_factory,
        )
    )
    profile = IntegrationProfile(relational_store="typeerror-relational")

    with pytest.raises(TypeError, match="internal factory bug"):
        resolve_collaborative_work_repositories(profile)

    assert failing_factory.materialization_invocations == 1


def test_postgresql_factory_materialization_invoked_exactly_once() -> None:
    register_postgresql_integration()
    profile = IntegrationProfile(
        relational_store=POSTGRESQL,
        options={POSTGRESQL.slug: {"dsn": "postgresql://localhost/test"}},
    )
    factory = get_entry(POSTGRESQL.slug).factory
    assert isinstance(factory, CollaborativeWorkPersistenceFactory)

    with (
        patch.object(
            factory,
            "materialize_collaborative_work_repositories",
            wraps=factory.materialize_collaborative_work_repositories,
        ) as materialize,
        patch(
            "intergrax.collaborative_work.persistence.open_postgresql_collaborative_work_repositories",
            return_value=_in_memory_collaborative_work_repositories(),
        ),
    ):
        bundle = resolve_collaborative_work_repositories(profile)

    materialize.assert_called_once()
    bundle.close()


def test_prebuilt_postgresql_instance_transfers_connection_ownership(_mock_psycopg: None) -> None:
    connection = _FakeConnection()
    config = PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test")
    adapter = _PostgreSQLRelationalStore(config=config, connection=connection)
    integration = PostgresqlRelationalStoreIntegration.from_client(adapter, enabled=True)
    profile = IntegrationProfile(relational_store=integration)

    bundle = resolve_collaborative_work_repositories(profile)

    assert isinstance(bundle, CollaborativeWorkRepositories)
    bundle.close()
    assert connection.closed is True


def test_prebuilt_sqlite_instance_fails_closed() -> None:
    adapter = _SQLiteRelationalStore(Path("build/test_relational.db"))
    adapter.connect()
    integration = SqliteRelationalStoreIntegration.from_client(adapter, enabled=True)
    profile = IntegrationProfile(relational_store=integration)

    with pytest.raises(
        IntegrationConfigurationError,
        match="Pre-built Sqlite relational store instances do not support",
    ):
        resolve_collaborative_work_repositories(profile)
