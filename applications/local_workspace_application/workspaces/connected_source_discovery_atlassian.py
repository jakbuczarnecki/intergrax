# © Artur Czarnecki. All rights reserved.

"""Provider-owned exact-resource discovery for Jira projects and Confluence spaces."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from local_workspace_application.workspaces.connected_source_discovery_strategy import (
    ConnectedSourceRevalidationLimits,
    RemoteResourceStrategyPage,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceCandidateV1,
    RemoteResourceTypeV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    ConfluenceSpaceCandidatePayload,
    JiraProjectCandidatePayload,
    RemoteResourceOpaqueRefCodec,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
    JiraIssueTrackerIntegration,
)
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    validate_jira_project_key,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
    ConfluenceWikiKnowledgeIntegration,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    validate_confluence_space_id,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry


@dataclass(frozen=True, slots=True)
class JiraKnownProject:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    project_key: str
    safe_display_label: str


class JiraKnownProjectCatalog:
    """Application configuration for bounded project selection."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._projects: dict[tuple[str, str, str, str], JiraKnownProject] = {}

    def register(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        project_key: str,
        safe_display_label: str,
    ) -> JiraKnownProject:
        project = JiraKnownProject(
            tenant_id=_require_text(tenant_id, "tenant_id"),
            workspace_id=_require_text(workspace_id, "workspace_id"),
            connection_ref=_require_text(connection_ref, "connection_ref"),
            project_key=validate_jira_project_key(project_key),
            safe_display_label=_require_text(safe_display_label, "safe_display_label"),
        )
        with self._lock:
            self._projects[
                (
                    project.tenant_id,
                    project.workspace_id,
                    project.connection_ref,
                    project.project_key,
                )
            ] = project
        return project

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        project_key: str,
    ) -> JiraKnownProject | None:
        key = (
            tenant_id.strip(),
            workspace_id.strip(),
            connection_ref.strip(),
            project_key.strip(),
        )
        with self._lock:
            return self._projects.get(key)

    def for_connection(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> JiraKnownProject | None:
        with self._lock:
            return next(
                (
                    project
                    for project in self._projects.values()
                    if project.tenant_id == tenant_id
                    and project.workspace_id == workspace_id
                    and project.connection_ref == connection_ref
                ),
                None,
            )

    def list_for_connection(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> tuple[JiraKnownProject, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        project
                        for project in self._projects.values()
                        if project.tenant_id == tenant_id
                        and project.workspace_id == workspace_id
                        and project.connection_ref == connection_ref
                    ),
                    key=lambda project: project.project_key,
                )
            )


@dataclass(frozen=True, slots=True)
class ConfluenceKnownSpace:
    tenant_id: str
    workspace_id: str
    connection_ref: str
    space_id: str
    safe_display_label: str


class ConfluenceKnownSpaceCatalog:
    """Application configuration for bounded space selection."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._spaces: dict[tuple[str, str, str, str], ConfluenceKnownSpace] = {}

    def register(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        space_id: str,
        safe_display_label: str,
    ) -> ConfluenceKnownSpace:
        space = ConfluenceKnownSpace(
            tenant_id=_require_text(tenant_id, "tenant_id"),
            workspace_id=_require_text(workspace_id, "workspace_id"),
            connection_ref=_require_text(connection_ref, "connection_ref"),
            space_id=validate_confluence_space_id(space_id),
            safe_display_label=_require_text(safe_display_label, "safe_display_label"),
        )
        with self._lock:
            self._spaces[
                (
                    space.tenant_id,
                    space.workspace_id,
                    space.connection_ref,
                    space.space_id,
                )
            ] = space
        return space

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        space_id: str,
    ) -> ConfluenceKnownSpace | None:
        key = (
            tenant_id.strip(),
            workspace_id.strip(),
            connection_ref.strip(),
            space_id.strip(),
        )
        with self._lock:
            return self._spaces.get(key)

    def for_connection(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> ConfluenceKnownSpace | None:
        with self._lock:
            return next(
                (
                    space
                    for space in self._spaces.values()
                    if space.tenant_id == tenant_id
                    and space.workspace_id == workspace_id
                    and space.connection_ref == connection_ref
                ),
                None,
            )

    def list_for_connection(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> tuple[ConfluenceKnownSpace, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        space
                        for space in self._spaces.values()
                        if space.tenant_id == tenant_id
                        and space.workspace_id == workspace_id
                        and space.connection_ref == connection_ref
                    ),
                    key=lambda space: space.space_id,
                )
            )


class JiraProjectDiscoveryStrategy:
    resource_type = RemoteResourceTypeV1.JIRA_PROJECT

    def __init__(
        self,
        *,
        connection_registry: KnowledgeConnectionRegistry,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
        known_projects: JiraKnownProjectCatalog,
    ) -> None:
        self._connection_registry = connection_registry
        self._codec = opaque_ref_codec
        self._known_projects = known_projects

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        provider_cursor: str | None,
        limit: int,
    ) -> RemoteResourceStrategyPage:
        if provider_cursor is not None:
            return RemoteResourceStrategyPage(items=(), provider_cursor=None)
        projects = self._known_projects.list_for_connection(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        if not projects:
            return RemoteResourceStrategyPage(items=(), provider_cursor=None)
        items: list[RemoteResourceCandidateV1] = []
        for project in projects:
            label = await self._resolve_label(project)
            items.append(
                RemoteResourceCandidateV1(
                    opaque_candidate_ref=self._codec.encode_jira_project_candidate(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        connection_ref=connection_ref,
                        project_key=project.project_key,
                        safe_display_label=label,
                    ),
                    resource_type=self.resource_type,
                    remote_resource_id=project.project_key,
                    safe_display_label=label,
                    safe_description="Jira project",
                )
            )
        return RemoteResourceStrategyPage(
            items=tuple(items[:limit]),
            provider_cursor=None,
        )

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        limits: ConnectedSourceRevalidationLimits,
    ) -> str:
        del limits
        payload = self._codec.decode_jira_project_candidate(opaque_candidate_ref)
        self._validate_payload(
            payload,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        project = self._known_projects.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            project_key=payload.project_key,
        )
        if project is None:
            raise ConnectedSourceDiscoveryError("candidate_inaccessible")
        return await self._resolve_label(project)

    async def _resolve_label(self, project: JiraKnownProject) -> str:
        integration = self._resolve_integration(
            tenant_id=project.tenant_id,
            connection_ref=project.connection_ref,
        )
        try:
            page = integration.search_knowledge_issues(
                project_key=project.project_key,
                next_page_token=None,
                limit=1,
            )
        except Exception as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
        if page.issues and page.issues[0].project_name.strip():
            return page.issues[0].project_name.strip()
        return project.safe_display_label

    def _resolve_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> JiraIssueTrackerIntegration:
        try:
            integration = self._connection_registry.resolve(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
            )
        except Exception as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
        if not isinstance(integration, JiraIssueTrackerIntegration):
            raise ConnectedSourceDiscoveryError("connection_incompatible")
        return integration

    def _validate_payload(
        self,
        payload: JiraProjectCandidatePayload,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> None:
        if (
            payload.tenant_id != tenant_id
            or payload.workspace_id != workspace_id
            or payload.connection_ref != connection_ref
            or payload.resource_type is not self.resource_type
        ):
            raise ConnectedSourceDiscoveryError("workspace_not_found")
        validate_jira_project_key(payload.project_key)


class ConfluenceSpaceDiscoveryStrategy:
    resource_type = RemoteResourceTypeV1.CONFLUENCE_SPACE

    def __init__(
        self,
        *,
        connection_registry: KnowledgeConnectionRegistry,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
        known_spaces: ConfluenceKnownSpaceCatalog,
    ) -> None:
        self._connection_registry = connection_registry
        self._codec = opaque_ref_codec
        self._known_spaces = known_spaces

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        provider_cursor: str | None,
        limit: int,
    ) -> RemoteResourceStrategyPage:
        if provider_cursor is not None:
            return RemoteResourceStrategyPage(items=(), provider_cursor=None)
        spaces = self._known_spaces.list_for_connection(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        if not spaces:
            return RemoteResourceStrategyPage(items=(), provider_cursor=None)
        items: list[RemoteResourceCandidateV1] = []
        for space in spaces:
            await self._verify_space(space)
            items.append(
                RemoteResourceCandidateV1(
                    opaque_candidate_ref=self._codec.encode_confluence_space_candidate(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        connection_ref=connection_ref,
                        space_id=space.space_id,
                        safe_display_label=space.safe_display_label,
                    ),
                    resource_type=self.resource_type,
                    remote_resource_id=space.space_id,
                    safe_display_label=space.safe_display_label,
                    safe_description="Confluence space",
                )
            )
        return RemoteResourceStrategyPage(
            items=tuple(items[:limit]),
            provider_cursor=None,
        )

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        limits: ConnectedSourceRevalidationLimits,
    ) -> str:
        del limits
        payload = self._codec.decode_confluence_space_candidate(opaque_candidate_ref)
        self._validate_payload(
            payload,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        space = self._known_spaces.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            space_id=payload.space_id,
        )
        if space is None:
            raise ConnectedSourceDiscoveryError("candidate_inaccessible")
        await self._verify_space(space)
        return space.safe_display_label

    async def _verify_space(self, space: ConfluenceKnownSpace) -> None:
        integration = self._resolve_integration(
            tenant_id=space.tenant_id,
            connection_ref=space.connection_ref,
        )
        try:
            integration.list_knowledge_pages(
                space_id=space.space_id,
                cursor=None,
                limit=1,
            )
        except Exception as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc

    def _resolve_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> ConfluenceWikiKnowledgeIntegration:
        try:
            integration = self._connection_registry.resolve(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
                integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
            )
        except Exception as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
        if not isinstance(integration, ConfluenceWikiKnowledgeIntegration):
            raise ConnectedSourceDiscoveryError("connection_incompatible")
        return integration

    def _validate_payload(
        self,
        payload: ConfluenceSpaceCandidatePayload,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> None:
        if (
            payload.tenant_id != tenant_id
            or payload.workspace_id != workspace_id
            or payload.connection_ref != connection_ref
            or payload.resource_type is not self.resource_type
        ):
            raise ConnectedSourceDiscoveryError("workspace_not_found")
        validate_confluence_space_id(payload.space_id)


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}_must_not_be_blank")
    return value.strip()


__all__ = [
    "ConfluenceKnownSpaceCatalog",
    "ConfluenceSpaceDiscoveryStrategy",
    "JiraKnownProjectCatalog",
    "JiraProjectDiscoveryStrategy",
]
