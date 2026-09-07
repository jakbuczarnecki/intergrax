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
class CollaborativeWorkRepositories:
    """Bundle of authoritative Collaborative Work repository ports (MP-1 core)."""

    membership: WorkspaceMembershipRepository
    delegation: AuthorityDelegationRepository
    principal_authority: PrincipalAuthorityRepository
    policy: CollaborativePolicyRepository
    operation_profile: CollaborativeOperationPolicyProfileRepository
    store: CollaborativeWorkStoreOwner

    def close(self) -> None:
        self.store.close()


@dataclass(frozen=True, slots=True)
class CollaborativeWorkSharedWorkRepositories:
    """MP-2 Shared Work repository ports materialized by a persistence backend."""

    work_item: WorkItemRepository
    assignment: AssignmentRepository


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRepositoriesWithSharedWork:
    """Full Collaborative Work persistence bundle with MP-2 Shared Work ports."""

    core: CollaborativeWorkRepositories
    shared_work: CollaborativeWorkSharedWorkRepositories

    @property
    def membership(self) -> WorkspaceMembershipRepository:
        return self.core.membership

    @property
    def delegation(self) -> AuthorityDelegationRepository:
        return self.core.delegation

    @property
    def principal_authority(self) -> PrincipalAuthorityRepository:
        return self.core.principal_authority

    @property
    def policy(self) -> CollaborativePolicyRepository:
        return self.core.policy

    @property
    def operation_profile(self) -> CollaborativeOperationPolicyProfileRepository:
        return self.core.operation_profile

    @property
    def store(self) -> CollaborativeWorkStoreOwner:
        return self.core.store

    @property
    def work_item(self) -> WorkItemRepository:
        return self.shared_work.work_item

    @property
    def assignment(self) -> AssignmentRepository:
        return self.shared_work.assignment

    def close(self) -> None:
        self.core.close()


CollaborativeWorkMaterializedRepositories = (
    CollaborativeWorkRepositories | CollaborativeWorkRepositoriesWithSharedWork
)


def collaborative_work_core_repositories(
    bundle: CollaborativeWorkMaterializedRepositories,
) -> CollaborativeWorkRepositories:
    """Return the MP-1 core bundle from any materialized persistence composition."""
    if isinstance(bundle, CollaborativeWorkRepositoriesWithSharedWork):
        return bundle.core
    return bundle


def open_sqlite_collaborative_work_repositories(
    db_path: str,
) -> CollaborativeWorkRepositoriesWithSharedWork:
    """Open durable Collaborative Work repositories backed by configured SQL storage."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteCollaborativeWorkStore(str(path))
    core = CollaborativeWorkRepositories(
        membership=SQLiteWorkspaceMembershipRepository(store),
        delegation=SQLiteAuthorityDelegationRepository(store),
        principal_authority=SQLitePrincipalAuthorityRepository(store),
        policy=SQLiteCollaborativePolicyRepository(store),
        operation_profile=SQLiteCollaborativeOperationPolicyProfileRepository(store),
        store=store,
    )
    return CollaborativeWorkRepositoriesWithSharedWork(
        core=core,
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
