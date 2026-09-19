# © Artur Czarnecki. All rights reserved.

"""Composition-root wiring for MP-6C Collaborative Activity ingestion."""

from __future__ import annotations

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.persistence import (
    sqlite_collaborative_activity_append_store,
    sqlite_collaborative_activity_read_store,
)
from intergrax.collaborative_work.collaborative_activity_read import CollaborativeActivityReadService
from intergrax.collaborative_work.collaborative_activity_read_authorization import (
    CollaborativeActivityReadAuthorizationEvaluator,
    build_default_collaborative_activity_read_authorization_policy,
)
from intergrax.contracts.collaborative_activity_read import (
    CollaborativeActivityReadAuthorizationPolicy,
)
from intergrax.collaborative_work.collaborative_activity_ingestion import (
    CollaborativeActivityIngestionService,
    DefaultCollaborativeActivityIngestionPolicy,
)
from intergrax.contracts.collaborative_activity import CollaborativeActivityAppendStore
from intergrax.contracts.collaborative_activity import CollaborativeActivityReadPort
from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityIngestionPolicy,
    DefaultCollaborativeActivityIngestionPolicyConfig,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPublisherContextResolver,
    VerifiedCollaborativeActivityPublisherIdentity,
)


def build_default_collaborative_activity_ingestion_policy(
    *,
    config: DefaultCollaborativeActivityIngestionPolicyConfig | None = None,
) -> DefaultCollaborativeActivityIngestionPolicy:
    return DefaultCollaborativeActivityIngestionPolicy(config=config)


def build_collaborative_activity_ingestion_service(
    *,
    verified_publisher_identity: VerifiedCollaborativeActivityPublisherIdentity,
    publisher_context_resolver: CollaborativeActivityPublisherContextResolver,
    append_store: CollaborativeActivityAppendStore,
    ingestion_policy: CollaborativeActivityIngestionPolicy | None = None,
) -> CollaborativeActivityIngestionService:
    """Bind ingestion to resolver-derived publisher authority — no raw context injection."""
    publisher_context = publisher_context_resolver.resolve(verified_publisher_identity)
    policy = ingestion_policy or build_default_collaborative_activity_ingestion_policy()
    return CollaborativeActivityIngestionService(
        publisher_context=publisher_context,
        ingestion_policy=policy,
        append_store=append_store,
    )


def build_sqlite_collaborative_activity_append_store(
    db_path: str,
) -> CollaborativeActivityAppendStore:
    return sqlite_collaborative_activity_append_store(db_path)


def build_collaborative_activity_read_service(
    *,
    authority_resolver: CollaborativeWorkAuthorityResolver,
    read_port: CollaborativeActivityReadPort,
    read_authorization_policy: CollaborativeActivityReadAuthorizationPolicy | None = None,
) -> CollaborativeActivityReadService:
    policy = (
        read_authorization_policy
        or build_default_collaborative_activity_read_authorization_policy()
    )
    evaluator = CollaborativeActivityReadAuthorizationEvaluator(
        authority_resolver=authority_resolver,
        read_authorization_policy=policy,
    )
    return CollaborativeActivityReadService(
        read_authorization=evaluator,
        read_port=read_port,
    )


def build_sqlite_collaborative_activity_read_store(
    db_path: str,
) -> CollaborativeActivityReadPort:
    return sqlite_collaborative_activity_read_store(db_path)


__all__ = [
    "build_collaborative_activity_ingestion_service",
    "build_collaborative_activity_read_service",
    "build_default_collaborative_activity_ingestion_policy",
    "build_sqlite_collaborative_activity_append_store",
    "build_sqlite_collaborative_activity_read_store",
]
