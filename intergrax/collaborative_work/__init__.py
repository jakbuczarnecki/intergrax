# © Artur Czarnecki. All rights reserved.

"""Collaborative Work platform domain — membership, delegation, and authority resolution."""

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    AuthorityDelegationAlreadyExists,
    AuthorityDelegationIdempotencyConflict,
    AuthorityDelegationNotFound,
    AuthorityDelegationRepository,
    AuthorityDelegationRevisionConflict,
    AuthorityDelegationScopeKey,
    CollaborativeWorkRepositoryCapabilities,
    CreateAuthorityDelegationCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    UpdateAuthorityDelegationCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipNotFound,
    WorkspaceMembershipRepository,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
)

__all__ = [
    "CollaborativeWorkAuthorityResolver",
    "INITIAL_RECORD_REVISION",
    "AuthorityDelegationAlreadyExists",
    "AuthorityDelegationIdempotencyConflict",
    "AuthorityDelegationNotFound",
    "AuthorityDelegationRepository",
    "AuthorityDelegationRevisionConflict",
    "AuthorityDelegationScopeKey",
    "CollaborativeWorkRepositoryCapabilities",
    "CreateAuthorityDelegationCommand",
    "CreateWorkspaceMembershipCommand",
    "InMemoryAuthorityDelegationRepository",
    "InMemoryWorkspaceMembershipRepository",
    "UpdateAuthorityDelegationCommand",
    "UpdateWorkspaceMembershipCommand",
    "WorkspaceMembershipAlreadyExists",
    "WorkspaceMembershipIdempotencyConflict",
    "WorkspaceMembershipNotFound",
    "WorkspaceMembershipRepository",
    "WorkspaceMembershipRevisionConflict",
    "WorkspaceMembershipScopeKey",
]
