# © Artur Czarnecki. All rights reserved.

"""Provider-neutral execution of validated Hybrid Ask evidence plans."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

if TYPE_CHECKING:
    from local_workspace_application.host.task_executor import (
        LocalWorkspaceTaskExecutor,
    )

from local_workspace_application.serving.workspace_schemas import WorkspaceSearchHitV1
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    HybridAskIndexedRetrievalStatusV1,
    HybridAskLiveExecutionStatusV1,
    HybridAskTruncationStateV1,
    IndexedWorkspaceEvidenceV1,
    LiveWorkspaceEvidenceV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    ExecutableLiveCallV1,
    IndexedRetrievalDirectiveV1,
    KnowledgeQueryAudienceV1,
    ValidatedEvidencePlanV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
    LiveResultRetentionV1,
    QueryPolicyModeV2,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
    is_workspace_source_product_visible,
)
from local_workspace_application.workspaces.models import Workspace
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.search_evidence import map_search_hits
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.live.contracts import (
    LiveCapabilityExecutionContextV1,
    LiveCapabilityExecutionResultV1,
    LiveCapabilityHandlerV1,
    LiveCapabilityResultItemV1,
    LiveExecutionOutcomeV1,
    LiveExecutionReceiptV1,
    content_sha256,
    evidence_id_for_call,
    result_hash_for_items,
    safe_locator_or_none,
)
from intergrax.runtime.vendor_knowledge.live.errors import LiveErrorCodeV1
from intergrax.runtime.vendor_knowledge.live.identity import (
    validate_capability_identity,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    PublishedLiveRegistrationV1,
)

_LIVE_ERROR_CODES = frozenset(
    code.value for code in LiveErrorCodeV1
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_nonblank(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name}_must_not_be_blank")
    return cleaned


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_must_be_timezone_aware")
    return value


class LiveCapabilityHandlerKeyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    capability_id: str = Field(..., min_length=1, max_length=128)
    contract_version: str = Field(..., min_length=1, max_length=32)

    _validate_ids = field_validator(
        "provider_id", "capability_id", "contract_version"
    )(
        lambda value, info: _require_nonblank(value, info.field_name)
    )


class LiveCapabilityHandlerRegistryV1:
    """Immutable exact-identity registry for provider-neutral live handlers."""

    def __init__(self, handlers: Iterable[LiveCapabilityHandlerV1] = ()) -> None:
        entries: dict[tuple[str, IntegrationCategory, str, str], LiveCapabilityHandlerV1] = {}
        for handler in handlers:
            key = self._key_for_handler(handler)
            if key in entries:
                raise ValueError("duplicate_live_handler_identity")
            entries[key] = handler
        self._handlers = MappingProxyType(entries)

    @classmethod
    def from_published_registration(
        cls,
        published: PublishedLiveRegistrationV1,
    ) -> LiveCapabilityHandlerRegistryV1:
        return cls(tuple(published.handlers.values()))

    @staticmethod
    def _key_for_handler(
        handler: LiveCapabilityHandlerV1,
    ) -> tuple[str, IntegrationCategory, str, str]:
        try:
            key = LiveCapabilityHandlerKeyV1(
                provider_id=handler.provider_id,
                integration_kind=handler.integration_kind,
                capability_id=handler.capability_id,
                contract_version=handler.contract_version,
            )
            validate_capability_identity(
                capability_id=key.capability_id,
                provider_id=key.provider_id,
                integration_kind=key.integration_kind,
                source_kind=handler.source_kind,
                contract_version=key.contract_version,
            )
            if not isinstance(handler.expected_request_model, type) or not issubclass(
                handler.expected_request_model, BaseModel
            ):
                raise ValueError("invalid_live_handler_request_model")
            if not isinstance(handler.request_schema_ref, str) or not isinstance(
                handler.result_schema_ref, str
            ):
                raise ValueError("invalid_live_handler_schema_reference")
        except (AttributeError, TypeError, ValidationError, ValueError) as exc:
            raise ValueError("invalid_live_handler_identity") from exc
        return (
            key.provider_id,
            key.integration_kind,
            key.capability_id,
            key.contract_version,
        )

    def resolve(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        capability_id: str,
        contract_version: str = "1",
    ) -> LiveCapabilityHandlerV1:
        key = (
            provider_id.strip(),
            integration_kind,
            capability_id.strip(),
            contract_version.strip(),
        )
        handler = self._handlers.get(key)
        if handler is None:
            raise LookupError("live_capability_unavailable")
        return handler


@runtime_checkable
class TenantConnectionIntegrationResolverPort(Protocol):
    def resolve(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> object:
        ...


class KnowledgeConnectionRegistryIntegrationResolverV1:
    """Narrow adapter over the existing instance-local Connection registry."""

    def __init__(self, registry: KnowledgeConnectionRegistry) -> None:
        self._registry = registry

    def resolve(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> object:
        return self._registry.resolve(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
        )


@runtime_checkable
class LiveRuntimeAuthorityPort(Protocol):
    def is_usable(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        live_access_binding_id: str,
        connection_ref: str,
        capability_id: str,
    ) -> bool:
        ...


@runtime_checkable
class IndexedEvidenceRetrieverPort(Protocol):
    async def retrieve(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        configuration_revision: int,
        question: str,
        directive: IndexedRetrievalDirectiveV1,
        audience_context: AudienceContextV1,
    ) -> tuple[IndexedWorkspaceEvidenceV1, ...]:
        ...


class _RepositoryWorkspaceLookup:
    def __init__(self, repository: ManagedWorkspaceRepository) -> None:
        self._repository = repository

    def require_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> Workspace | None:
        return self._repository.get_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )


class WorkspaceIndexedEvidenceRetrieverV1:
    """Adapter from the existing application search boundary to transient evidence."""

    def __init__(
        self,
        *,
        task_executor: LocalWorkspaceTaskExecutor,
        workspace_repository: ManagedWorkspaceRepository,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._task_executor = task_executor
        self._workspace_repository = workspace_repository
        self._clock = clock

    async def retrieve(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        configuration_revision: int,
        question: str,
        directive: IndexedRetrievalDirectiveV1,
        audience_context: AudienceContextV1,
    ) -> tuple[IndexedWorkspaceEvidenceV1, ...]:
        from local_workspace_application.serving.run_metadata import (
            attach_lkw_evidence_metadata,
        )

        from intergrax.runtime.task.task import Task, TaskContext
        from intergrax.runtime.task.task_run_bridge import new_run_id

        task = Task(
            task_id=new_run_id(),
            tenant_id=tenant_id,
            user_id="lkw.managed_workspace.hybrid_ask",
            message=question,
            context=TaskContext(capability="local.workspace.search"),
            metadata={
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "collection_id": workspace_id,
                "query": question,
                "top_k": directive.max_results,
                "requested_by": "lkw.managed_workspace.hybrid_ask",
            },
        )
        result = await self._task_executor.execute(task)
        result_metadata = dict(getattr(result, "metadata", None) or {})
        attach_lkw_evidence_metadata(
            result_metadata,
            task_result=result,
            capability="local.workspace.search",
        )
        result = result.model_copy(update={"metadata": result_metadata})
        configuration = WorkspaceKnowledgeConfigurationService(
            repository=self._workspace_repository,
            workspace_lookup=_RepositoryWorkspaceLookup(self._workspace_repository),
        ).get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if (
            configuration is None
            or configuration.tenant_id != tenant_id
            or configuration.workspace_id != workspace_id
            or configuration.configuration_revision != configuration_revision
        ):
            raise ValueError("indexed_configuration_revision_unavailable")
        hits = map_search_hits(
            repository=self._workspace_repository,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            task_result=result,
            limit=directive.max_results,
        )
        return tuple(
            self._map_hit(
                hit,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                configuration=configuration,
                audience_context=audience_context,
            )
            for hit in hits
        )

    def _map_hit(
        self,
        hit: WorkspaceSearchHitV1,
        *,
        tenant_id: str,
        workspace_id: str,
        configuration: WorkspaceKnowledgeConfigurationV1,
        audience_context: AudienceContextV1,
    ) -> IndexedWorkspaceEvidenceV1:
        self._verify_indexed_hit(
            hit,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            configuration=configuration,
            audience_context=audience_context,
        )
        metadata = hit.metadata
        chunk_id = metadata.get("chunk_id")
        chunk = chunk_id.strip() if isinstance(chunk_id, str) and chunk_id.strip() else None
        location_value = metadata.get("location")
        location = None
        if isinstance(location_value, dict) and isinstance(location_value.get("page"), int):
            from local_workspace_application.workspaces.hybrid_ask_models import (
                AskCitationLocationV1,
            )

            location = AskCitationLocationV1(page=location_value["page"])
        identity = chunk or _sha256(hit.snippet)
        return IndexedWorkspaceEvidenceV1(
            evidence_id=f"idx:{workspace_id}:{hit.document_id}:{identity}",
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            safe_display_name=hit.file_name,
            retrieved_at=self._clock(),
            content=hit.snippet,
            content_hash=_sha256(hit.snippet),
            audience=AskAudienceV1(audience_context.audience.value),
            source_id=hit.source_id,
            document_id=hit.document_id,
            chunk_id=chunk,
            location=location,
            score=hit.score,
            safe_source_label=_safe_string(metadata.get("safe_source_label")),
            indexed_source_binding_id=_safe_string(
                metadata.get("indexed_source_binding_id")
            ),
        )

    def _verify_indexed_hit(
        self,
        hit: WorkspaceSearchHitV1,
        *,
        tenant_id: str,
        workspace_id: str,
        configuration: WorkspaceKnowledgeConfigurationV1,
        audience_context: AudienceContextV1,
    ) -> None:
        if (
            configuration.tenant_id != tenant_id
            or configuration.workspace_id != workspace_id
            or not isinstance(configuration.configuration_revision, int)
        ):
            raise ValueError("indexed_configuration_identity_mismatch")

        document = self._workspace_repository.get_document_ref(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            document_id=hit.document_id,
        )
        source = self._workspace_repository.get_source(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=hit.source_id,
        )
        if (
            document is None
            or source is None
            or document.tenant_id != tenant_id
            or document.workspace_id != workspace_id
            or document.source_id != hit.source_id
            or source.tenant_id != tenant_id
            or source.workspace_id != workspace_id
            or source.source_id != hit.source_id
            or not is_workspace_source_product_visible(
                source,
                committed_configuration_revision=configuration.configuration_revision,
            )
        ):
            raise ValueError("indexed_source_ownership_unverified")

        bindings = tuple(
            binding
            for binding in getattr(configuration, "indexed_sources", ())
            if binding.source_id == hit.source_id
            and binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
        )
        if len(bindings) != 1:
            raise ValueError("indexed_source_binding_ambiguous")
        binding = bindings[0]
        if (
            binding.tenant_id != tenant_id
            or binding.workspace_id != workspace_id
            or binding.source_id != document.source_id
            or binding.effective_revision > configuration.configuration_revision
        ):
            raise ValueError("indexed_source_binding_identity_mismatch")

        metadata_binding_id = hit.metadata.get("indexed_source_binding_id")
        if (
            metadata_binding_id is not None
            and (
                not isinstance(metadata_binding_id, str)
                or not metadata_binding_id.strip()
                or metadata_binding_id.strip() != binding.indexed_source_binding_id
            )
        ):
            raise ValueError("indexed_source_binding_metadata_mismatch")

        requested_audience = audience_context.audience
        if (
            requested_audience is KnowledgeQueryAudienceV1.SHARED
            and binding.audience_eligibility
            is not KnowledgeAudienceEligibilityV1.SHARED_ALLOWED
        ):
            raise ValueError("indexed_source_shared_access_forbidden")


def _safe_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _validate_indexed_evidence_batch(
    evidence_items: tuple[IndexedWorkspaceEvidenceV1, ...],
    *,
    tenant_id: str,
    workspace_id: str,
    audience: KnowledgeQueryAudienceV1,
) -> None:
    expected_audience = audience.value
    seen_ids: set[str] = set()
    for evidence in evidence_items:
        if not isinstance(evidence, IndexedWorkspaceEvidenceV1):
            raise TypeError("indexed_evidence_structural_invalid")
        if (
            evidence.tenant_id != tenant_id
            or evidence.workspace_id != workspace_id
            or evidence.audience.value != expected_audience
            or not evidence.evidence_id.startswith("idx:")
            or evidence.evidence_id in seen_ids
            or evidence.content_hash != _sha256(evidence.content)
            or evidence.retrieved_at.tzinfo is None
            or evidence.retrieved_at.utcoffset() is None
        ):
            raise ValueError("indexed_evidence_execution_validation_failed")
        seen_ids.add(evidence.evidence_id)


class LiveCapabilityExecutorV1:
    """Execute exactly one validated live call through an exact registered handler."""

    def __init__(
        self,
        *,
        handler_registry: LiveCapabilityHandlerRegistryV1 | None = None,
        published_registration: PublishedLiveRegistrationV1 | None = None,
        integration_resolver: TenantConnectionIntegrationResolverPort,
        runtime_authority: LiveRuntimeAuthorityPort | None = None,
        clock: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        if handler_registry is not None and published_registration is not None:
            raise ValueError("live_handler_registry_sources_conflict")
        self._handler_registry = (
            LiveCapabilityHandlerRegistryV1.from_published_registration(
                published_registration
            )
            if published_registration is not None
            else handler_registry or LiveCapabilityHandlerRegistryV1()
        )
        self._integration_resolver = integration_resolver
        self._runtime_authority = runtime_authority
        self._clock = clock
        self._monotonic = monotonic
        self._id_factory = id_factory

    async def execute(
        self,
        *,
        run_id: str,
        tenant_id: str,
        workspace_id: str,
        call: ExecutableLiveCallV1,
        audience: KnowledgeQueryAudienceV1,
        retention: LiveResultRetentionV1,
        deadline_monotonic: float | None = None,
    ) -> LiveCapabilityExecutionResultV1:
        started_at = self._clock()
        started_monotonic = self._monotonic()
        deadline = deadline_monotonic
        if deadline is None:
            deadline = started_monotonic + call.effective_budget.max_total_duration_ms / 1000
        context = LiveCapabilityExecutionContextV1(
            run_id=run_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            audience=audience,
            started_at=started_at,
            deadline_monotonic=deadline,
            retention=retention,
        )

        try:
            call.assert_identity()
        except (LookupError, ValueError):
            return self._failure(
                call=call,
                started_at=started_at,
                error_code="live_capability_unavailable",
                run_id=run_id,
                retention=retention,
            )

        if self._runtime_authority is not None:
            try:
                runtime_usable = self._runtime_authority.is_usable(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    live_access_binding_id=call.live_access_binding_id,
                    connection_ref=call.connection_ref,
                    capability_id=call.capability_id,
                )
            except Exception:  # noqa: BLE001 - runtime authority must fail closed
                runtime_usable = False
            if not runtime_usable:
                return self._failure(
                    call=call,
                    started_at=started_at,
                    error_code="live_binding_unavailable",
                    run_id=run_id,
                    retention=retention,
                )

        try:
            handler = self._handler_registry.resolve(
                provider_id=call.provider_id,
                integration_kind=call.integration_kind,
                capability_id=call.capability_id,
                contract_version=call.contract_version,
            )
        except (LookupError, ValueError):
            return self._failure(
                call=call,
                started_at=started_at,
                error_code="live_capability_unavailable",
                run_id=run_id,
                retention=retention,
            )

        if not isinstance(call.validated_request, handler.expected_request_model):
            return self._failure(
                call=call,
                started_at=started_at,
                error_code="live_request_invalid",
                run_id=run_id,
                retention=retention,
            )

        try:
            integration = self._integration_resolver.resolve(
                tenant_id=context.tenant_id,
                connection_ref=call.connection_ref,
                provider_id=call.provider_id,
                integration_kind=call.integration_kind,
            )
        except Exception:
            return self._failure(
                call=call,
                started_at=started_at,
                error_code="live_execution_failed",
                run_id=run_id,
                retention=retention,
            )

        remaining = deadline - self._monotonic()
        if remaining <= 0:
            return self._failure(
                call=call,
                started_at=started_at,
                error_code="live_execution_timeout",
                run_id=run_id,
                retention=retention,
            )

        try:
            raw_result = handler.execute(
                integration=integration,
                call=call,
                context=context,
            )
            if not inspect.isawaitable(raw_result):
                raise TypeError("handler must return an awaitable result")
            handler_result = await asyncio.wait_for(raw_result, timeout=remaining)
        except TimeoutError:
            return self._failure(
                call=call,
                started_at=started_at,
                error_code="live_execution_timeout",
                run_id=run_id,
                retention=retention,
            )
        except Exception:
            return self._failure(
                call=call,
                started_at=started_at,
                error_code="live_execution_failed",
                run_id=run_id,
                retention=retention,
            )

        completed_at = self._clock()
        if self._monotonic() > deadline:
            return self._failure(
                call=call,
                started_at=started_at,
                completed_at=completed_at,
                error_code="live_execution_timeout",
                run_id=run_id,
                retention=retention,
            )

        try:
            if not isinstance(handler_result, LiveCapabilityExecutionResultV1):
                handler_result = LiveCapabilityExecutionResultV1.model_validate(handler_result)
            if handler_result.call_id != call.call_id:
                raise ValueError("live_result_call_id_mismatch")
            if handler_result.normalized_outcome is LiveExecutionOutcomeV1.FAILED:
                error_code = handler_result.error_code
                if error_code not in _LIVE_ERROR_CODES:
                    error_code = "live_execution_failed"
                return self._failure(
                    call=call,
                    started_at=started_at,
                    completed_at=completed_at,
                    error_code=error_code,
                    run_id=run_id,
                    retention=retention,
                )
            normalized_items, truncated = self._bound_items(
                handler_result.items,
                max_items=call.effective_budget.max_result_items,
                max_bytes=call.effective_budget.max_result_bytes,
                max_content_bytes_per_item=call.effective_budget.max_content_bytes_per_item,
            )
        except ValidationError:
            return self._failure(
                call=call,
                started_at=started_at,
                completed_at=completed_at,
                error_code="live_result_invalid",
                run_id=run_id,
                retention=retention,
            )
        except ValueError:
            return self._failure(
                call=call,
                started_at=started_at,
                completed_at=completed_at,
                error_code="live_result_invalid",
                run_id=run_id,
                retention=retention,
            )

        item_count = len(normalized_items)
        byte_count = sum(len(item.content.encode("utf-8")) for item in normalized_items)
        outcome = (
            LiveExecutionOutcomeV1.TRUNCATED
            if truncated or handler_result.truncated
            else LiveExecutionOutcomeV1.COMPLETED
        )
        receipt = self._receipt(
            run_id=run_id,
            call=call,
            started_at=started_at,
            completed_at=completed_at,
            item_count=item_count,
            byte_count=byte_count,
            items=normalized_items,
            truncated=outcome is LiveExecutionOutcomeV1.TRUNCATED,
            normalized_outcome=outcome,
            retention=retention,
        )
        return LiveCapabilityExecutionResultV1(
            call_id=call.call_id,
            normalized_outcome=outcome,
            items=tuple(normalized_items),
            item_count=item_count,
            byte_count=byte_count,
            started_at=started_at,
            completed_at=completed_at,
            truncated=outcome is LiveExecutionOutcomeV1.TRUNCATED,
            receipt=receipt,
            provider_id=call.provider_id,
            integration_kind=call.integration_kind,
            source_kind=call.source_kind,
            capability_id=call.capability_id,
            contract_version=call.contract_version,
            live_access_binding_id=call.live_access_binding_id,
            connection_ref=call.connection_ref,
            remote_resource_id=call.remote_resource_id,
        )

    def _bound_items(
        self,
        items: tuple[LiveCapabilityResultItemV1, ...],
        *,
        max_items: int,
        max_bytes: int,
        max_content_bytes_per_item: int,
    ) -> tuple[tuple[LiveCapabilityResultItemV1, ...], bool]:
        seen: set[str] = set()
        bounded: list[LiveCapabilityResultItemV1] = []
        used_bytes = 0
        truncated = False
        for item in items:
            if item.remote_item_id in seen:
                raise ValueError("duplicate_remote_item_id")
            seen.add(item.remote_item_id)
            if item.content_hash != content_sha256(item.content):
                raise ValueError("content_hash_mismatch")
            if len(bounded) >= max_items:
                truncated = True
                break
            content_bytes = item.content.encode("utf-8")
            if len(content_bytes) > max_content_bytes_per_item:
                content = content_bytes[:max_content_bytes_per_item].decode(
                    "utf-8", errors="ignore"
                )
                if not content:
                    truncated = True
                    break
                item = item.model_copy(
                    update={
                        "content": content,
                        "content_hash": content_sha256(content),
                        "safe_locator": safe_locator_or_none(item.safe_locator),
                        "truncated": True,
                    }
                )
                content_bytes = content.encode("utf-8")
                truncated = True
            remaining = max_bytes - used_bytes
            if len(content_bytes) > remaining:
                if remaining <= 0:
                    truncated = True
                    break
                content = content_bytes[:remaining].decode("utf-8", errors="ignore")
                if not content:
                    truncated = True
                    break
                bounded.append(
                    item.model_copy(
                        update={
                            "content": content,
                            "content_hash": content_sha256(content),
                            "safe_locator": safe_locator_or_none(item.safe_locator),
                            "truncated": True,
                        }
                    )
                )
                truncated = True
                break
            bounded.append(
                item.model_copy(update={"safe_locator": safe_locator_or_none(item.safe_locator)})
            )
            used_bytes += len(content_bytes)
            if item.truncated:
                truncated = True
        if len(bounded) < len(items):
            truncated = True
        return tuple(bounded), truncated

    def _failure(
        self,
        *,
        call: ExecutableLiveCallV1,
        started_at: datetime,
        error_code: str,
        run_id: str,
        retention: LiveResultRetentionV1,
        completed_at: datetime | None = None,
    ) -> LiveCapabilityExecutionResultV1:
        if error_code not in _LIVE_ERROR_CODES:
            error_code = "live_execution_failed"
        finished = completed_at or self._clock()
        receipt = self._receipt(
            run_id=run_id,
            call=call,
            started_at=started_at,
            completed_at=finished,
            item_count=0,
            byte_count=0,
            items=(),
            truncated=False,
            normalized_outcome=LiveExecutionOutcomeV1.FAILED,
            retention=retention,
            error_code=error_code,
        )
        return LiveCapabilityExecutionResultV1(
            call_id=call.call_id,
            normalized_outcome=LiveExecutionOutcomeV1.FAILED,
            item_count=0,
            byte_count=0,
            started_at=started_at,
            completed_at=finished,
            error_code=error_code,
            receipt=receipt,
            provider_id=call.provider_id,
            integration_kind=call.integration_kind,
            source_kind=call.source_kind,
            capability_id=call.capability_id,
            contract_version=call.contract_version,
            live_access_binding_id=call.live_access_binding_id,
            connection_ref=call.connection_ref,
            remote_resource_id=call.remote_resource_id,
        )

    def _receipt(
        self,
        *,
        run_id: str,
        call: ExecutableLiveCallV1,
        started_at: datetime,
        completed_at: datetime,
        item_count: int,
        byte_count: int,
        items: tuple[LiveCapabilityResultItemV1, ...],
        truncated: bool,
        normalized_outcome: LiveExecutionOutcomeV1,
        retention: LiveResultRetentionV1,
        error_code: str | None = None,
    ) -> LiveExecutionReceiptV1 | None:
        if retention is not LiveResultRetentionV1.RECEIPT_ONLY:
            return None
        result_hash = result_hash_for_items(
            items=items,
            normalized_outcome=normalized_outcome.value,
            error_code=error_code,
            item_count=item_count,
            byte_count=byte_count,
        )
        return LiveExecutionReceiptV1(
            receipt_id=self._id_factory(),
            run_id=run_id,
            call_id=call.call_id,
            live_access_binding_id=call.live_access_binding_id,
            provider_id=call.provider_id,
            source_kind=call.source_kind,
            capability_id=call.capability_id,
            contract_version=call.contract_version,
            started_at=started_at,
            completed_at=completed_at,
            item_count=item_count,
            byte_count=byte_count,
            result_hash=result_hash,
            truncated=truncated,
            normalized_outcome=normalized_outcome.value,
            error_code=error_code,
        )


class KnowledgeQueryExecutionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: str = Field(..., min_length=1, max_length=128)
    plan_id: str = Field(..., min_length=1, max_length=128)
    mode: QueryPolicyModeV2
    indexed_evidence: tuple[IndexedWorkspaceEvidenceV1, ...] = ()
    live_evidence: tuple[LiveWorkspaceEvidenceV1, ...] = ()
    receipts: tuple[LiveExecutionReceiptV1, ...] = ()
    indexed_retrieval_status: HybridAskIndexedRetrievalStatusV1
    live_execution_status: HybridAskLiveExecutionStatusV1
    truncation_state: HybridAskTruncationStateV1
    partial_failure: bool
    error_code: str | None = None
    started_at: datetime
    completed_at: datetime

    _validate_ids = field_validator("run_id", "plan_id")(
        lambda value, info: _require_nonblank(value, info.field_name)
    )
    _validate_timestamps = field_validator("started_at", "completed_at")(
        lambda value, info: _require_aware(value, info.field_name)
    )
    _validate_error_code = field_validator("error_code")(
        lambda value: None if value is None else _require_nonblank(value, "error_code")
    )


@runtime_checkable
class LiveEvidenceExpansionHookV1(Protocol):
    def expand(
        self,
        *,
        stage: int,
        calls: tuple[ExecutableLiveCallV1, ...],
        outcomes: tuple[LiveCapabilityExecutionResultV1, ...],
        attempted_calls: tuple[ExecutableLiveCallV1, ...],
        remaining_provider_call_budget: int,
        deadline_reached: bool,
    ) -> tuple[ExecutableLiveCallV1, ...]:
        ...


class KnowledgeQueryOrchestratorV1:
    """Coordinate indexed retrieval and required validated live calls only."""

    def __init__(
        self,
        *,
        indexed_retriever: IndexedEvidenceRetrieverPort,
        live_executor: LiveCapabilityExecutorV1,
        clock: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._indexed_retriever = indexed_retriever
        self._live_executor = live_executor
        self._clock = clock
        self._monotonic = monotonic

    async def execute(
        self,
        *,
        run_id: str,
        question: str,
        validated_plan: ValidatedEvidencePlanV1,
        retention: LiveResultRetentionV1,
        live_expansion: LiveEvidenceExpansionHookV1 | None = None,
    ) -> KnowledgeQueryExecutionResultV1:
        started_at = self._clock()
        plan = validated_plan.plan
        indexed: tuple[IndexedWorkspaceEvidenceV1, ...] = ()
        live: list[LiveWorkspaceEvidenceV1] = []
        receipts: list[LiveExecutionReceiptV1] = []
        indexed_status = HybridAskIndexedRetrievalStatusV1.SKIPPED
        live_status = HybridAskLiveExecutionStatusV1.SKIPPED
        truncation = HybridAskTruncationStateV1.NONE

        if plan.mode in (QueryPolicyModeV2.INDEXED_ONLY, QueryPolicyModeV2.HYBRID):
            directive = plan.indexed_retrieval_directive
            if directive is None:
                return self._result(
                    run_id=run_id,
                    plan=validated_plan,
                    indexed=indexed,
                    live=live,
                    receipts=receipts,
                    indexed_status=HybridAskIndexedRetrievalStatusV1.FAILED,
                    live_status=live_status,
                    truncation=truncation,
                    partial_failure=False,
                    error_code="indexed_retrieval_failed",
                    started_at=started_at,
                )
            try:
                indexed_result = self._indexed_retriever.retrieve(
                    tenant_id=plan.tenant_id,
                    workspace_id=plan.workspace_id,
                    configuration_revision=plan.configuration_revision,
                    question=question,
                    directive=directive,
                    audience_context=plan.audience_context,
                )
                retrieved_indexed = (
                    tuple(await indexed_result)
                    if inspect.isawaitable(indexed_result)
                    else tuple(indexed_result)
                )
                _validate_indexed_evidence_batch(
                    retrieved_indexed,
                    tenant_id=plan.tenant_id,
                    workspace_id=plan.workspace_id,
                    audience=plan.audience_context.audience,
                )
                indexed = retrieved_indexed
                indexed_status = HybridAskIndexedRetrievalStatusV1.COMPLETED
            except Exception:
                return self._result(
                    run_id=run_id,
                    plan=validated_plan,
                    indexed=(),
                    live=live,
                    receipts=receipts,
                    indexed_status=HybridAskIndexedRetrievalStatusV1.FAILED,
                    live_status=live_status,
                    truncation=truncation,
                    partial_failure=False,
                    error_code="indexed_retrieval_failed",
                    started_at=started_at,
                )

        if plan.mode in (QueryPolicyModeV2.LIVE_ONLY, QueryPolicyModeV2.HYBRID):
            if not validated_plan.executable_live_calls:
                return self._result(
                    run_id=run_id,
                    plan=validated_plan,
                    indexed=indexed,
                    live=live,
                    receipts=receipts,
                    indexed_status=indexed_status,
                    live_status=HybridAskLiveExecutionStatusV1.FAILED,
                    truncation=truncation,
                    partial_failure=bool(indexed),
                    error_code="live_execution_failed",
                    started_at=started_at,
                )
            live_status = HybridAskLiveExecutionStatusV1.COMPLETED
            deadline = self._monotonic() + (
                validated_plan.effective_budget.max_total_duration_ms / 1000
            )
            attempted_live_calls = 0
            first_live_error: str | None = None

            async def execute_stage(
                calls: tuple[ExecutableLiveCallV1, ...],
            ) -> tuple[
                tuple[LiveCapabilityExecutionResultV1, ...],
                tuple[ExecutableLiveCallV1, ...],
            ]:
                nonlocal attempted_live_calls
                nonlocal first_live_error
                nonlocal live_status
                nonlocal truncation
                stage_outcomes: list[LiveCapabilityExecutionResultV1] = []
                stage_attempted: list[ExecutableLiveCallV1] = []
                for call in calls:
                    if self._monotonic() >= deadline:
                        truncation = HybridAskTruncationStateV1.LIVE
                        live_status = (
                            HybridAskLiveExecutionStatusV1.PARTIAL
                            if indexed or live or stage_outcomes
                            else HybridAskLiveExecutionStatusV1.FAILED
                        )
                        break
                    outcome = await self._live_executor.execute(
                        run_id=run_id,
                        tenant_id=plan.tenant_id,
                        workspace_id=plan.workspace_id,
                        call=call,
                        audience=plan.audience_context.audience,
                        retention=retention,
                        deadline_monotonic=deadline,
                    )
                    attempted_live_calls += 1
                    stage_attempted.append(call)
                    stage_outcomes.append(outcome)
                    if outcome.receipt is not None:
                        receipts.append(outcome.receipt)
                    if outcome.normalized_outcome is LiveExecutionOutcomeV1.FAILED:
                        first_live_error = first_live_error or (
                            outcome.error_code or "live_execution_failed"
                        )
                        live_status = (
                            HybridAskLiveExecutionStatusV1.PARTIAL
                            if indexed or live
                            else HybridAskLiveExecutionStatusV1.FAILED
                        )
                        continue
                    for item in outcome.items:
                        live.append(
                            LiveWorkspaceEvidenceV1(
                                evidence_id=evidence_id_for_call(
                                    provider_id=call.provider_id,
                                    integration_kind=call.integration_kind,
                                    source_kind=call.source_kind,
                                    capability_id=call.capability_id,
                                    contract_version=call.contract_version,
                                    live_access_binding_id=call.live_access_binding_id,
                                    connection_ref=call.connection_ref,
                                    remote_resource_id=call.remote_resource_id,
                                    call_id=call.call_id,
                                    remote_item_id=item.remote_item_id,
                                ),
                                tenant_id=plan.tenant_id,
                                workspace_id=plan.workspace_id,
                                safe_display_name=item.safe_display_name,
                                retrieved_at=item.retrieved_at,
                                content=item.content,
                                content_hash=item.content_hash,
                                audience=AskAudienceV1(plan.audience_context.audience.value),
                                live_access_binding_id=call.live_access_binding_id,
                                connection_ref=call.connection_ref,
                                capability_id=call.capability_id,
                                source_kind=call.source_kind,
                                contract_version=call.contract_version,
                                remote_resource_id=call.resolved_resource_scope.remote_resource_id,
                                remote_item_id=item.remote_item_id,
                                provider_id=call.provider_id,
                                integration_kind=call.integration_kind.value,
                                call_id=call.call_id,
                                remote_updated_at=item.remote_updated_at,
                                safe_locator=item.safe_locator,
                                truncated=item.truncated or outcome.truncated,
                            )
                        )
                    if outcome.truncated:
                        truncation = HybridAskTruncationStateV1.LIVE
                return tuple(stage_outcomes), tuple(stage_attempted)

            stage_one_outcomes, stage_one_attempted = await execute_stage(
                validated_plan.executable_live_calls
            )
            stage_two_calls: tuple[ExecutableLiveCallV1, ...] = ()
            if live_expansion is not None:
                stage_two_calls = live_expansion.expand(
                    stage=1,
                    calls=validated_plan.executable_live_calls,
                    outcomes=stage_one_outcomes,
                    attempted_calls=stage_one_attempted,
                    remaining_provider_call_budget=max(
                        0,
                        validated_plan.effective_budget.max_live_calls
                        - attempted_live_calls,
                    ),
                    deadline_reached=self._monotonic() >= deadline,
                )
                remaining_calls = max(
                    0,
                    validated_plan.effective_budget.max_live_calls
                    - attempted_live_calls,
                )
                stage_two_calls = stage_two_calls[:remaining_calls]
            if stage_two_calls:
                stage_two_outcomes, stage_two_attempted = await execute_stage(
                    stage_two_calls
                )
                if live_expansion is not None:
                    live_expansion.expand(
                        stage=2,
                        calls=stage_two_calls,
                        outcomes=stage_two_outcomes,
                        attempted_calls=stage_two_attempted,
                        remaining_provider_call_budget=max(
                            0,
                            validated_plan.effective_budget.max_live_calls
                            - attempted_live_calls,
                        ),
                        deadline_reached=self._monotonic() >= deadline,
                    )

            if first_live_error is None:
                if live_status is not HybridAskLiveExecutionStatusV1.FAILED:
                    live_status = HybridAskLiveExecutionStatusV1.COMPLETED
            else:
                live_status = (
                    HybridAskLiveExecutionStatusV1.PARTIAL
                    if indexed or live
                    else HybridAskLiveExecutionStatusV1.FAILED
                )
            return self._result(
                run_id=run_id,
                plan=validated_plan,
                indexed=indexed,
                live=live,
                receipts=receipts,
                indexed_status=indexed_status,
                live_status=live_status,
                truncation=truncation,
                partial_failure=first_live_error is not None,
                error_code=first_live_error,
                started_at=started_at,
            )

        return self._result(
            run_id=run_id,
            plan=validated_plan,
            indexed=indexed,
            live=live,
            receipts=receipts,
            indexed_status=indexed_status,
            live_status=live_status,
            truncation=truncation,
            partial_failure=False,
            error_code=None,
            started_at=started_at,
        )

    def _result(
        self,
        *,
        run_id: str,
        plan: ValidatedEvidencePlanV1,
        indexed: tuple[IndexedWorkspaceEvidenceV1, ...],
        live: list[LiveWorkspaceEvidenceV1],
        receipts: list[LiveExecutionReceiptV1],
        indexed_status: HybridAskIndexedRetrievalStatusV1,
        live_status: HybridAskLiveExecutionStatusV1,
        truncation: HybridAskTruncationStateV1,
        partial_failure: bool,
        error_code: str | None,
        started_at: datetime,
    ) -> KnowledgeQueryExecutionResultV1:
        return KnowledgeQueryExecutionResultV1(
            run_id=run_id,
            plan_id=plan.plan.plan_id,
            mode=plan.plan.mode,
            indexed_evidence=indexed,
            live_evidence=tuple(live),
            receipts=tuple(receipts),
            indexed_retrieval_status=indexed_status,
            live_execution_status=live_status,
            truncation_state=truncation,
            partial_failure=partial_failure,
            error_code=error_code,
            started_at=started_at,
            completed_at=self._clock(),
        )
