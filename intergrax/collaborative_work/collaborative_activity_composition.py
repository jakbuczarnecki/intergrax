# © Artur Czarnecki. All rights reserved.

"""Composition-root wiring for MP-6C Collaborative Activity ingestion."""

from __future__ import annotations

from intergrax.collaborative_work.collaborative_activity_ingestion import (
    CollaborativeActivityIngestionService,
    DefaultCollaborativeActivityIngestionPolicy,
)
from intergrax.contracts.collaborative_activity import CollaborativeActivityAppendStore
from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityIngestionPolicy,
    CollaborativeActivityPublisherContext,
    DefaultCollaborativeActivityIngestionPolicyConfig,
)


def build_default_collaborative_activity_ingestion_policy(
    *,
    config: DefaultCollaborativeActivityIngestionPolicyConfig | None = None,
) -> DefaultCollaborativeActivityIngestionPolicy:
    return DefaultCollaborativeActivityIngestionPolicy(config=config)


def build_collaborative_activity_ingestion_service(
    *,
    publisher_context: CollaborativeActivityPublisherContext,
    append_store: CollaborativeActivityAppendStore,
    ingestion_policy: CollaborativeActivityIngestionPolicy | None = None,
) -> CollaborativeActivityIngestionService:
    policy = ingestion_policy or build_default_collaborative_activity_ingestion_policy()
    return CollaborativeActivityIngestionService(
        publisher_context=publisher_context,
        ingestion_policy=policy,
        append_store=append_store,
    )


__all__ = [
    "build_collaborative_activity_ingestion_service",
    "build_default_collaborative_activity_ingestion_policy",
]
