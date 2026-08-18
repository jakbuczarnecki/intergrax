# © Artur Czarnecki. All rights reserved.

"""Tests for typed Collaborative Work persistence provider binding (PROVIDER-QUAL-3B-R1)."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationEntry,
)
from intergrax.integrations.providers.relational_store.postgresql.register import (
    register_postgresql_integration,
)
from intergrax.integrations.providers.relational_store.sqlite.register import (
    register_sqlite_integration,
)
from intergrax.integrations.registry.catalog import clear_catalog, register_integration
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

    def materialize_collaborative_work_repositories(self) -> CollaborativeWorkRepositories:
        return _in_memory_collaborative_work_repositories()


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


def test_sqlite_profile_materializes_collaborative_work_repositories(tmp_path: Path) -> None:
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE: {"data_dir": str(tmp_path)}},
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
            POSTGRESQL: {
                "dsn": "postgresql://localhost/test",
                "connection_factory": counting_factory,
            }
        },
    )

    with patch(
        "intergrax.integrations.providers.relational_store.postgresql.opens.open_postgresql_relational_store",
    ) as generic_open:
        bundle = resolve_collaborative_work_repositories(profile)

    generic_open.assert_not_called()
    assert counting_factory.invocations >= 1
    assert isinstance(bundle, CollaborativeWorkRepositories)
    bundle.close()
    assert connection.closed is True


def test_postgresql_profile_preserves_injected_connection_factory() -> None:
    register_postgresql_integration()
    connection = _FakeConnection()
    counting_factory = _CountingConnectionFactory(connection)
    profile = IntegrationProfile(
        relational_store=POSTGRESQL,
        options={
            POSTGRESQL: {
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


def test_unsupported_relational_provider_fails_explicitly() -> None:
    register_integration(
        IntegrationEntry(
            slug="unsupported-relational",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=lambda **_kwargs: _UnsupportedRelationalStore(),
        )
    )
    profile = IntegrationProfile(relational_store="unsupported-relational")

    with pytest.raises(IntegrationConfigurationError, match="does not implement Collaborative Work"):
        resolve_collaborative_work_repositories(profile)


def test_future_vendor_uses_same_bridge_without_central_dispatch_changes() -> None:
    register_integration(
        IntegrationEntry(
            slug="future_vendor_x",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=lambda **_kwargs: _FutureVendorRelationalStore(),
        )
    )
    profile = IntegrationProfile(relational_store="future_vendor_x")

    bundle = resolve_collaborative_work_repositories(profile)

    assert isinstance(bundle, CollaborativeWorkRepositories)
    assert bundle.membership.capabilities.durable is False
    bundle.close()
