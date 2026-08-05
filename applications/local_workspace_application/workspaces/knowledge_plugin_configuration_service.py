# © Artur Czarnecki. All rights reserved.

"""Safe, provider-neutral LKW view over registered knowledge plugins."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
    RemoteResourceAvailabilityV1,
    RemoteResourceDiscoveryPageV1,
    TenantRemoteResourceDiscoveryService,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
    TenantConnectionCapabilityReadService,
    TenantLiveCapabilityCatalogPort,
    is_bindable_read_only_capability,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)

from local_workspace_application.workspaces.conversation_context_models import (
    ConversationAudienceMode,
    ConversationExecutionContextV1,
)

_MAX_CONNECTIONS = 20
_MAX_RESOURCES = 50
_MAX_CAPABILITIES = 100
_MAX_LABEL_LENGTH = 128
_MAX_DESCRIPTION_LENGTH = 512
_MAX_SERIALIZED_SIZE = 100_000
_SECRET_TEXT_RE = re.compile(
    r"(?:authorization\s*[:=]|bearer\s+|api[_-]?key\s*[:=]|"
    r"(?:access|refresh)[_-]?token\s*[:=])",
    re.IGNORECASE,
)


def _safe_product_text(value: str) -> str:
    if _SECRET_TEXT_RE.search(value):
        raise ValueError("safe product text contains credential material")
    return value


class KnowledgeConfigurationModeV1(StrEnum):
    INDEXED_SOURCE_ELIGIBLE = "INDEXED_SOURCE_ELIGIBLE"
    LIVE_ACCESS_ELIGIBLE = "LIVE_ACCESS_ELIGIBLE"
    BOTH = "BOTH"
    INFORMATION_ONLY = "INFORMATION_ONLY"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class KnowledgePluginConfigurationError(RuntimeError):
    """Stable product error without provider details or secret-bearing text."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code)


class KnowledgeWorkspaceAuthorizationPort(Protocol):
    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> object | None: ...


class KnowledgeConnectionServiceFactory(Protocol):
    def __call__(self, tenant_id: str) -> TenantConnectionCapabilityReadService: ...


class KnowledgeCapabilityCatalogFactory(Protocol):
    def __call__(self, tenant_id: str) -> TenantLiveCapabilityCatalogPort: ...


class KnowledgeResourceDiscoveryServiceFactory(Protocol):
    def __call__(self, tenant_id: str) -> TenantRemoteResourceDiscoveryService: ...


class KnowledgeConnectionSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(min_length=1, max_length=128)
    safe_display_label: str = Field(min_length=1, max_length=_MAX_LABEL_LENGTH)
    provider_id: str = Field(min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    administrative_status: TenantConnectionAdministrativeStatus
    available_configuration_modes: tuple[KnowledgeConfigurationModeV1, ...] = ()
    available_source_kinds: tuple[str, ...] = ()

    @field_validator("safe_display_label")
    @classmethod
    def _safe_label(cls, value: str) -> str:
        return _safe_product_text(value)


class KnowledgeRemoteResourceSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(min_length=1, max_length=128)
    remote_resource_id: str = Field(min_length=1, max_length=256)
    resource_type: str = Field(min_length=1, max_length=64)
    safe_display_label: str = Field(min_length=1, max_length=_MAX_LABEL_LENGTH)
    safe_description: str = Field(default="", max_length=_MAX_DESCRIPTION_LENGTH)
    availability: RemoteResourceAvailabilityV1
    supported_capability_ids: tuple[str, ...] = ()
    source_kind: str = Field(min_length=1, max_length=64)
    snapshot_version: str = Field(min_length=1, max_length=64)
    configuration_modes: tuple[KnowledgeConfigurationModeV1, ...] = ()

    @field_validator("safe_display_label", "safe_description")
    @classmethod
    def _safe_text(cls, value: str) -> str:
        return _safe_product_text(value)


class KnowledgeCapabilitySummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(min_length=1, max_length=128)
    remote_resource_id: str | None = Field(default=None, max_length=256)
    capability_id: str = Field(min_length=1, max_length=128)
    effect: CapabilityEffectV1
    read_only: bool
    resource_scope_required: bool
    supported_resource_types: tuple[str, ...] = ()
    available: bool
    bindable_read_only: bool
    max_result_items: int | None = Field(default=None, gt=0)
    max_result_bytes: int | None = Field(default=None, gt=0)
    configuration_mode: KnowledgeConfigurationModeV1
    indexed_source_eligibility: KnowledgeConfigurationModeV1 = (
        KnowledgeConfigurationModeV1.UNKNOWN
    )

    @field_validator("max_result_items", "max_result_bytes", mode="before")
    @classmethod
    def _validate_declared_limit(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("declared result limits must be positive integers")
        return value


class KnowledgeRemoteResourcePageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resources: tuple[KnowledgeRemoteResourceSummaryV1, ...] = ()
    next_page_token: str | None = None
    snapshot_version: str = Field(min_length=1, max_length=64)


class KnowledgeConfigurationTargetV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_ref: str = Field(min_length=1, max_length=128)
    remote_resource_id: str = Field(min_length=1, max_length=256)
    capability_id: str = Field(min_length=1, max_length=128)
    source_kind: str = Field(min_length=1, max_length=64)
    snapshot_version: str = Field(min_length=1, max_length=64)


class KnowledgePluginConfigurationSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    available_connections: tuple[KnowledgeConnectionSummaryV1, ...] = ()
    available_remote_resources: tuple[KnowledgeRemoteResourceSummaryV1, ...] = ()
    available_resource_capabilities: tuple[KnowledgeCapabilitySummaryV1, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_bounds_and_serialized_size(self) -> KnowledgePluginConfigurationSnapshotV1:
        if len(self.available_connections) > _MAX_CONNECTIONS:
            raise ValueError("too many knowledge connections")
        if len(self.available_remote_resources) > _MAX_RESOURCES:
            raise ValueError("too many knowledge remote resources")
        if len(self.available_resource_capabilities) > _MAX_CAPABILITIES:
            raise ValueError("too many knowledge capabilities")
        serialized = json.dumps(self.model_dump(mode="json"), separators=(",", ":"))
        if len(serialized) > _MAX_SERIALIZED_SIZE:
            raise ValueError("knowledge configuration snapshot too large")
        return self


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _sorted_modes(modes: set[KnowledgeConfigurationModeV1]) -> tuple[KnowledgeConfigurationModeV1, ...]:
    return tuple(sorted(modes, key=lambda item: item.value))


class KnowledgePluginConfigurationService:
    """LKW-owned safe adapter over provider-neutral Vendor Knowledge services."""

    def __init__(
        self,
        *,
        connection_service_factory: KnowledgeConnectionServiceFactory,
        capability_catalog_factory: KnowledgeCapabilityCatalogFactory,
        resource_discovery_service_factory: KnowledgeResourceDiscoveryServiceFactory,
        workspace_authorization: KnowledgeWorkspaceAuthorizationPort,
    ) -> None:
        self._connection_service_factory = connection_service_factory
        self._capability_catalog_factory = capability_catalog_factory
        self._resource_discovery_service_factory = resource_discovery_service_factory
        self._workspace_authorization = workspace_authorization

    def list_connections(
        self,
        *,
        tenant_id: str,
        execution_context: ConversationExecutionContextV1,
        limit: int = _MAX_CONNECTIONS,
        include_inactive: bool = True,
    ) -> tuple[KnowledgeConnectionSummaryV1, ...]:
        self._authorize(tenant_id=tenant_id, execution_context=execution_context)
        if not 1 <= limit <= _MAX_CONNECTIONS:
            raise KnowledgePluginConfigurationError("knowledge_plugin_configuration_unavailable")
        try:
            connections = self._connection_service_factory(tenant_id).list_connections(
                limit=limit,
                administrative_status=None
                if include_inactive
                else TenantConnectionAdministrativeStatus.ACTIVE,
            )
            summaries = [
                self._connection_summary(
                    connection,
                    tenant_id=tenant_id,
                )
                for connection in connections
            ]
        except KnowledgePluginConfigurationError:
            raise
        except Exception:
            raise KnowledgePluginConfigurationError(
                "knowledge_plugin_configuration_unavailable"
            ) from None
        summaries.sort(key=lambda item: item.connection_ref)
        return tuple(summaries[:limit])

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        execution_context: ConversationExecutionContextV1,
        connection_ref: str,
        source_kind: str,
        page_token: str | None = None,
        limit: int = 50,
    ) -> KnowledgeRemoteResourcePageV1:
        self._authorize(tenant_id=tenant_id, execution_context=execution_context)
        if not 1 <= limit <= _MAX_RESOURCES:
            raise KnowledgePluginConfigurationError("knowledge_plugin_configuration_unavailable")
        connection = self._active_connection(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        discovery = self._resource_discovery_service_factory(tenant_id)
        try:
            if source_kind not in discovery.list_source_kinds(connection_ref=connection_ref):
                raise KnowledgePluginConfigurationError(
                    "knowledge_resource_discovery_unavailable"
                )
            page = await discovery.list_remote_resources(
                connection_ref=connection_ref,
                source_kind=source_kind,
                page_token=page_token,
                limit=limit,
            )
            return self._project_resource_page(
                page,
                tenant_id=tenant_id,
                execution_context=execution_context,
                connection=connection,
                source_kind=source_kind,
            )
        except KnowledgePluginConfigurationError:
            raise
        except Exception:
            raise KnowledgePluginConfigurationError(
                "knowledge_resource_discovery_unavailable",
                retryable=True,
            ) from None

    def list_resource_capabilities(
        self,
        *,
        tenant_id: str,
        execution_context: ConversationExecutionContextV1,
        connection_ref: str,
        remote_resource_id: str | None = None,
    ) -> tuple[KnowledgeCapabilitySummaryV1, ...]:
        self._authorize(tenant_id=tenant_id, execution_context=execution_context)
        self._active_connection(tenant_id=tenant_id, connection_ref=connection_ref)
        if remote_resource_id is not None and not remote_resource_id.strip():
            raise KnowledgePluginConfigurationError("knowledge_resource_not_found")
        try:
            descriptors = self._capability_catalog_factory(tenant_id).list_capabilities(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                remote_resource_id=remote_resource_id,
            )
            if not isinstance(descriptors, tuple):
                raise TypeError("catalog result must be a tuple")
            projected = [
                self._capability_summary(
                    descriptor,
                    tenant_id=tenant_id,
                    connection_ref=connection_ref,
                    remote_resource_id=remote_resource_id,
                )
                for descriptor in descriptors
            ]
        except KnowledgePluginConfigurationError:
            raise
        except Exception:
            raise KnowledgePluginConfigurationError(
                "knowledge_plugin_configuration_unavailable"
            ) from None
        projected.sort(key=lambda item: item.capability_id)
        return tuple(projected[:_MAX_CAPABILITIES])

    async def get_configuration_snapshot(
        self,
        *,
        tenant_id: str,
        execution_context: ConversationExecutionContextV1,
    ) -> KnowledgePluginConfigurationSnapshotV1:
        connections = self.list_connections(
            tenant_id=tenant_id,
            execution_context=execution_context,
        )
        resources: list[KnowledgeRemoteResourceSummaryV1] = []
        capabilities: list[KnowledgeCapabilitySummaryV1] = []
        warnings: list[str] = []
        for connection in connections:
            if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
                continue
            for source_kind in connection.available_source_kinds:
                if len(resources) >= _MAX_RESOURCES:
                    break
                try:
                    page = await self.list_remote_resources(
                        tenant_id=tenant_id,
                        execution_context=execution_context,
                        connection_ref=connection.connection_ref,
                        source_kind=source_kind,
                        limit=min(_MAX_RESOURCES - len(resources), 20),
                    )
                except KnowledgePluginConfigurationError as exc:
                    warnings.append(exc.code)
                    continue
                resources.extend(page.resources)
                for resource in page.resources:
                    if len(capabilities) >= _MAX_CAPABILITIES:
                        break
                    try:
                        capabilities.extend(
                            self.list_resource_capabilities(
                                tenant_id=tenant_id,
                                execution_context=execution_context,
                                connection_ref=resource.connection_ref,
                                remote_resource_id=resource.remote_resource_id,
                            )
                        )
                    except KnowledgePluginConfigurationError as exc:
                        warnings.append(exc.code)
        resources.sort(
            key=lambda item: (
                item.connection_ref,
                item.remote_resource_id,
                item.source_kind,
            )
        )
        capabilities.sort(
            key=lambda item: (
                item.connection_ref,
                item.remote_resource_id or "",
                item.capability_id,
            )
        )
        return KnowledgePluginConfigurationSnapshotV1(
            available_connections=connections,
            available_remote_resources=tuple(resources[:_MAX_RESOURCES]),
            available_resource_capabilities=tuple(capabilities[:_MAX_CAPABILITIES]),
            warnings=tuple(sorted(set(warnings))),
        )

    def _authorize(
        self,
        *,
        tenant_id: str,
        execution_context: ConversationExecutionContextV1,
    ) -> None:
        if execution_context.tenant_id != tenant_id:
            raise KnowledgePluginConfigurationError("conversation_execution_context_mismatch")
        if execution_context.audience_mode is not ConversationAudienceMode.PERSONAL:
            raise KnowledgePluginConfigurationError("knowledge_plugin_configuration_unavailable")
        workspace = self._workspace_authorization.get_workspace(
            tenant_id=tenant_id,
            workspace_id=execution_context.workspace_id,
        )
        if workspace is None or getattr(workspace, "tenant_id", tenant_id) != tenant_id:
            raise KnowledgePluginConfigurationError("workspace_not_found")

    def _active_connection(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> SafeTenantConnectionV1:
        try:
            connection = self._connection_service_factory(tenant_id).get_connection(connection_ref)
        except Exception:
            raise KnowledgePluginConfigurationError(
                "knowledge_connection_not_found"
            ) from None
        if connection is None:
            raise KnowledgePluginConfigurationError("knowledge_connection_not_found")
        if connection.tenant_id != tenant_id:
            raise KnowledgePluginConfigurationError("knowledge_connection_not_found")
        if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
            raise KnowledgePluginConfigurationError("knowledge_connection_not_active")
        return connection

    def _connection_summary(
        self,
        connection: SafeTenantConnectionV1,
        *,
        tenant_id: str,
    ) -> KnowledgeConnectionSummaryV1:
        if not isinstance(connection, SafeTenantConnectionV1) or connection.tenant_id != tenant_id:
            raise KnowledgePluginConfigurationError("knowledge_plugin_configuration_unavailable")
        source_kinds: tuple[str, ...] = ()
        modes: set[KnowledgeConfigurationModeV1] = set()
        if connection.administrative_status is TenantConnectionAdministrativeStatus.ACTIVE:
            discovery = self._resource_discovery_service_factory(tenant_id)
            try:
                source_kinds = discovery.list_source_kinds(
                    connection_ref=connection.connection_ref
                )
            except Exception:
                source_kinds = ()
            try:
                descriptors = self._capability_catalog_factory(tenant_id).list_capabilities(
                    tenant_id=tenant_id,
                    connection_ref=connection.connection_ref,
                    remote_resource_id=None,
                )
                if any(is_bindable_read_only_capability(item) for item in descriptors):
                    modes.add(KnowledgeConfigurationModeV1.LIVE_ACCESS_ELIGIBLE)
            except Exception:
                modes.add(KnowledgeConfigurationModeV1.UNKNOWN)
            modes.add(KnowledgeConfigurationModeV1.UNKNOWN)
        else:
            modes.add(KnowledgeConfigurationModeV1.UNAVAILABLE)
        return KnowledgeConnectionSummaryV1(
            connection_ref=connection.connection_ref,
            safe_display_label=connection.safe_display_name,
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
            administrative_status=connection.administrative_status,
            available_configuration_modes=_sorted_modes(modes),
            available_source_kinds=tuple(sorted(set(source_kinds))),
        )

    def _project_resource_page(
        self,
        page: RemoteResourceDiscoveryPageV1,
        *,
        tenant_id: str,
        execution_context: ConversationExecutionContextV1,
        connection: SafeTenantConnectionV1,
        source_kind: str,
    ) -> KnowledgeRemoteResourcePageV1:
        if not isinstance(page, RemoteResourceDiscoveryPageV1):
            raise KnowledgePluginConfigurationError("knowledge_resource_discovery_unavailable")
        projected: list[KnowledgeRemoteResourceSummaryV1] = []
        for resource in page.resources:
            if resource.connection_ref != connection.connection_ref:
                raise KnowledgePluginConfigurationError(
                    "knowledge_resource_discovery_unavailable"
                )
            capabilities = self.list_resource_capabilities(
                tenant_id=tenant_id,
                execution_context=execution_context,
                connection_ref=connection.connection_ref,
                remote_resource_id=resource.remote_resource_id,
            )
            modes = {
                item.configuration_mode
                for item in capabilities
                if item.available
            }
            if not modes:
                modes.add(KnowledgeConfigurationModeV1.INFORMATION_ONLY)
            projected.append(
                KnowledgeRemoteResourceSummaryV1(
                    connection_ref=resource.connection_ref,
                    remote_resource_id=resource.remote_resource_id,
                    resource_type=resource.resource_type,
                    safe_display_label=resource.safe_display_label,
                    safe_description=resource.safe_description,
                    availability=resource.availability,
                    supported_capability_ids=resource.supported_capability_ids,
                    source_kind=source_kind,
                    snapshot_version=resource.snapshot_version,
                    configuration_modes=_sorted_modes(modes),
                )
            )
        projected.sort(
            key=lambda item: (
                item.connection_ref,
                item.remote_resource_id,
                item.source_kind,
            )
        )
        return KnowledgeRemoteResourcePageV1(
            resources=tuple(projected),
            next_page_token=page.next_page_token,
            snapshot_version=page.snapshot_version,
        )

    def _capability_summary(
        self,
        descriptor: LiveCapabilityDescriptorV1,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> KnowledgeCapabilitySummaryV1:
        if not isinstance(descriptor, LiveCapabilityDescriptorV1):
            raise KnowledgePluginConfigurationError("knowledge_plugin_configuration_unavailable")
        if descriptor.provider_id != self._active_connection(
            tenant_id=tenant_id,
            connection_ref=connection_ref
        ).provider_id:
            raise KnowledgePluginConfigurationError("knowledge_plugin_configuration_unavailable")
        bindable = is_bindable_read_only_capability(descriptor)
        if not descriptor.available:
            mode = KnowledgeConfigurationModeV1.UNAVAILABLE
        elif bindable:
            mode = KnowledgeConfigurationModeV1.LIVE_ACCESS_ELIGIBLE
        else:
            mode = KnowledgeConfigurationModeV1.INFORMATION_ONLY
        return KnowledgeCapabilitySummaryV1(
            connection_ref=connection_ref,
            remote_resource_id=remote_resource_id,
            capability_id=descriptor.capability_id,
            effect=descriptor.effect,
            read_only=descriptor.read_only,
            resource_scope_required=descriptor.resource_scope_required,
            supported_resource_types=descriptor.supported_resource_types,
            available=descriptor.available,
            bindable_read_only=bindable,
            max_result_items=descriptor.max_result_items,
            max_result_bytes=descriptor.max_result_bytes,
            configuration_mode=mode,
        )
