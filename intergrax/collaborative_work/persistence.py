# © Artur Czarnecki. All rights reserved.

"""Composition-root factories for Collaborative Work repository adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.collaborative_work.repository import (
    AuthorityDelegationRepository,
    CollaborativeOperationPolicyProfileRepository,
    CollaborativePolicyRepository,
    PrincipalAuthorityRepository,
    WorkspaceMembershipRepository,
)
from intergrax.collaborative_work.sqlite_repository import (
    SQLiteCollaborativeWorkStore,
    SQLiteAuthorityDelegationRepository,
    SQLiteCollaborativeOperationPolicyProfileRepository,
    SQLiteCollaborativePolicyRepository,
    SQLitePrincipalAuthorityRepository,
    SQLiteWorkspaceMembershipRepository,
)


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRepositories:
    """Bundle of authoritative Collaborative Work repository ports."""

    membership: WorkspaceMembershipRepository
    delegation: AuthorityDelegationRepository
    principal_authority: PrincipalAuthorityRepository
    policy: CollaborativePolicyRepository
    operation_profile: CollaborativeOperationPolicyProfileRepository
    store: SQLiteCollaborativeWorkStore

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
    )
