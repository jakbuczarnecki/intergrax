# © Artur Czarnecki. All rights reserved.

"""MP-6F composition root — binds source services to activity publication ingress."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.collaborative_work.collaborative_activity_source_adapters import (
    CollaborativeActivitySourcePublicationSideEffect,
    CollaborativeDecisionBindingServiceWithActivityPublication,
    CollaborativeWorkArtifactServiceWithActivityPublication,
    CollaborativeWorkServiceWithActivityPublication,
    ContextViewComposerWithActivityPublication,
)
from intergrax.collaborative_work.collaborative_activity_source_mapping import (
    CollaborativeActivityActorPrincipalKindResolver,
    CollaborativeWorkActivitySourceMapper,
    ContextViewActivitySourceMapper,
    DefaultCollaborativeWorkActivitySourceMapper,
    DefaultContextViewActivitySourceMapper,
)
from intergrax.collaborative_work.collaborative_activity_source_ports import (
    CollaborativeDecisionBindingActivitySourcePort,
    CollaborativeWorkActivityMutationPort,
    CollaborativeWorkArtifactActivityMutationPort,
)
from intergrax.contracts.collaborative_activity import CollaborativeActivityPublicationPort
from intergrax.contracts.context_view_composition import ContextViewComposer


@dataclass(frozen=True, slots=True)
class CollaborativeActivitySourceIntegration:
    """Explicit DI bundle for MP-6F source → publication wiring."""

    collaborative_work_service: CollaborativeWorkServiceWithActivityPublication | None = None
    artifact_service: CollaborativeWorkArtifactServiceWithActivityPublication | None = None
    decision_binding_service: CollaborativeDecisionBindingServiceWithActivityPublication | None = None
    context_view_composer: ContextViewComposerWithActivityPublication | None = None


def _default_side_effect(
    *,
    publication_port: CollaborativeActivityPublicationPort,
) -> CollaborativeActivitySourcePublicationSideEffect:
    return CollaborativeActivitySourcePublicationSideEffect(
        publication_port=publication_port,
    )


def wire_collaborative_work_service_with_activity_publication(
    *,
    inner: CollaborativeWorkActivityMutationPort,
    publication_port: CollaborativeActivityPublicationPort,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    mapper: CollaborativeWorkActivitySourceMapper | None = None,
) -> CollaborativeWorkServiceWithActivityPublication:
    resolved_mapper = mapper or DefaultCollaborativeWorkActivitySourceMapper(
        principal_kind_resolver=principal_kind_resolver,
    )
    return CollaborativeWorkServiceWithActivityPublication(
        inner=inner,
        side_effect=_default_side_effect(publication_port=publication_port),
        mapper=resolved_mapper,
    )


def wire_collaborative_work_artifact_service_with_activity_publication(
    *,
    inner: CollaborativeWorkArtifactActivityMutationPort,
    publication_port: CollaborativeActivityPublicationPort,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    mapper: CollaborativeWorkActivitySourceMapper | None = None,
) -> CollaborativeWorkArtifactServiceWithActivityPublication:
    resolved_mapper = mapper or DefaultCollaborativeWorkActivitySourceMapper(
        principal_kind_resolver=principal_kind_resolver,
    )
    return CollaborativeWorkArtifactServiceWithActivityPublication(
        inner=inner,
        side_effect=_default_side_effect(publication_port=publication_port),
        mapper=resolved_mapper,
    )


def wire_collaborative_decision_binding_service_with_activity_publication(
    *,
    inner: CollaborativeDecisionBindingActivitySourcePort,
    publication_port: CollaborativeActivityPublicationPort,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    mapper: CollaborativeWorkActivitySourceMapper | None = None,
) -> CollaborativeDecisionBindingServiceWithActivityPublication:
    resolved_mapper = mapper or DefaultCollaborativeWorkActivitySourceMapper(
        principal_kind_resolver=principal_kind_resolver,
    )
    return CollaborativeDecisionBindingServiceWithActivityPublication(
        inner=inner,
        side_effect=_default_side_effect(publication_port=publication_port),
        mapper=resolved_mapper,
    )


def wire_context_view_composer_with_activity_publication(
    *,
    inner: ContextViewComposer,
    publication_port: CollaborativeActivityPublicationPort,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    mapper: ContextViewActivitySourceMapper | None = None,
) -> ContextViewComposerWithActivityPublication:
    resolved_mapper = mapper or DefaultContextViewActivitySourceMapper(
        principal_kind_resolver=principal_kind_resolver,
    )
    return ContextViewComposerWithActivityPublication(
        inner=inner,
        side_effect=_default_side_effect(publication_port=publication_port),
        mapper=resolved_mapper,
    )


__all__ = [
    "CollaborativeActivitySourceIntegration",
    "wire_collaborative_decision_binding_service_with_activity_publication",
    "wire_collaborative_work_artifact_service_with_activity_publication",
    "wire_collaborative_work_service_with_activity_publication",
    "wire_context_view_composer_with_activity_publication",
]
