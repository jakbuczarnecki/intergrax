# © Artur Czarnecki. All rights reserved.

"""Composition-root factories for Collaborative Work repository adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from intergrax.collaborative_work.postgresql_repository import (
    PostgreSQLAuthorityDelegationRepository,
    PostgreSQLCollaborativeOperationPolicyProfileRepository,
    PostgreSQLCollaborativePolicyRepository,
    PostgreSQLCollaborativeWorkStore,
    PostgreSQLPrincipalAuthorityRepository,
    PostgreSQLWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    AssignmentRepository,
    AuthorityDelegationRepository,
    CollaborativeOperationPolicyProfileRepository,
    CollaborativePolicyRepository,
    PrincipalAuthorityRepository,
    WorkItemRepository,
    WorkspaceMembershipRepository,
)
from intergrax.collaborative_work.sqlite_repository import (
    SQLiteAssignmentRepository,
    SQLiteAuthorityDelegationRepository,
    SQLiteCollaborativeOperationPolicyProfileRepository,
    SQLiteCollaborativePolicyRepository,
    SQLiteCollaborativeWorkStore,
    SQLitePrincipalAuthorityRepository,
    SQLiteWorkItemRepository,
    SQLiteWorkspaceMembershipRepository,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)


@runtime_checkable
class CollaborativeWorkStoreOwner(Protocol):
    """Lifecycle owner for durable Collaborative Work repository adapters."""

    def close(self) -> None:
        """Release persistence resources."""


@dataclass(frozen=True, slots=True)
class CollaborativeWorkSharedWorkRepositories:
    """MP-2 Shared Work repository ports materialized by a persistence backend."""

    work_item: WorkItemRepository
    assignment: AssignmentRepository


_SHARED_WORK_UNAVAILABLE = (
    "MP-2 Shared Work repositories are not available for this persistence backend"
)


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRepositories:
    """Bundle of authoritative Collaborative Work repository ports."""

    membership: WorkspaceMembershipRepository
    delegation: AuthorityDelegationRepository
    principal_authority: PrincipalAuthorityRepository
    policy: CollaborativePolicyRepository
    operation_profile: CollaborativeOperationPolicyProfileRepository
    store: CollaborativeWorkStoreOwner
    shared_work: CollaborativeWorkSharedWorkRepositories | None = None

    @property
    def work_item(self) -> WorkItemRepository:
        if self.shared_work is None:
            raise RuntimeError(_SHARED_WORK_UNAVAILABLE)
        return self.shared_work.work_item

    @property
    def assignment(self) -> AssignmentRepository:
        if self.shared_work is None:
            raise RuntimeError(_SHARED_WORK_UNAVAILABLE)
        return self.shared_work.assignment

    def close(self) -> None:
        self.store.close()


def open_sqlite_collaborative_work_repositories(db_path: str) -> CollaborativeWorkRepositories:
    """Open durable Collaborative Work repositories backed by configured SQL storage."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteCollaborativeWorkStore(str(path))
    return CollaborativeWorkRepositories(
        membership=SQLiteWorkspaceMembershipRepository(store),
        delegation=SQLiteAuthorityDelegationRepository(store),
        principal_authority=SQLitePrincipalAuthorityRepository(store),
        policy=SQLiteCollaborativePolicyRepository(store),
        operation_profile=SQLiteCollaborativeOperationPolicyProfileRepository(store),
        store=store,
        shared_work=CollaborativeWorkSharedWorkRepositories(
            work_item=SQLiteWorkItemRepository(store),
            assignment=SQLiteAssignmentRepository(store),
        ),
    )


def open_postgresql_collaborative_work_repositories(
    *,
    config: PostgreSQLIntegrationConfig | None = None,
    connection_factory: Callable[[], Any] | None = None,
    schema_name: str | None = None,
) -> CollaborativeWorkRepositories:
    """Open production-grade Collaborative Work repositories backed by PostgreSQL."""
    resolved = config or PostgreSQLIntegrationConfig.from_env()
    try:
        store = PostgreSQLCollaborativeWorkStore(
            resolved,
            connection_factory=connection_factory,
            schema_name=schema_name,
        )
    except IntegrationConfigurationError:
        raise
    except Exception as exc:
        raise IntegrationConfigurationError(
            "PostgreSQL Collaborative Work repositories could not be opened"
        ) from exc
    return CollaborativeWorkRepositories(
        membership=PostgreSQLWorkspaceMembershipRepository(store),
        delegation=PostgreSQLAuthorityDelegationRepository(store),
        principal_authority=PostgreSQLPrincipalAuthorityRepository(store),
        policy=PostgreSQLCollaborativePolicyRepository(store),
        operation_profile=PostgreSQLCollaborativeOperationPolicyProfileRepository(store),
        store=store,
    )
