# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-neutral one-page synchronization coordinator."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingService,
)
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeFacade
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemRevision,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeReconciliationCandidateInventoryRepository,
    KnowledgeReconciliationRunRepository,
    KnowledgeRemoteItemStateRepository,
    KnowledgeSourceLeaseRepository,
    KnowledgeSyncCheckpointConflict,
    KnowledgeSyncCheckpointRepository,
    KnowledgeSyncCorruptState,
    KnowledgeSyncSink,
    KnowledgeSyncSinkReceiptInspector,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationRecoveryCommand,
    KnowledgeReconciliationRunPhase,
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStatus,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncBatch,
    KnowledgeSyncCheckpoint,
    KnowledgeSyncEnvelope,
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
    VendorKnowledgeReconciliationEngine,
)

_ACTIVE_RECONCILIATION_PHASES: frozenset[KnowledgeReconciliationRunPhase] = frozenset(
    {
        KnowledgeReconciliationRunPhase.COLLECTING,
        KnowledgeReconciliationRunPhase.PAGE_PREPARED,
        KnowledgeReconciliationRunPhase.FINALIZING,
        KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
    }
)

_ACTIVE_CHANGE_KINDS: frozenset[KnowledgeChangeKind] = frozenset(
    {
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.METADATA_CHANGED,
        KnowledgeChangeKind.PERMISSIONS_CHANGED,
    }
)

_TOMBSTONE_CHANGE_KINDS: frozenset[KnowledgeChangeKind] = frozenset(
    {
        KnowledgeChangeKind.DELETED,
        KnowledgeChangeKind.REVOKED,
    }
)


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _cursor_fingerprint(cursor: KnowledgeCursor | None) -> dict[str, str | None]:
    if cursor is None:
        return {"value": None, "version": None}
    return {"value": cursor.value, "version": cursor.version}


def _revision_fingerprint(revision: KnowledgeItemRevision | None) -> dict[str, Any]:
    if revision is None:
        return {
            "version": None,
            "etag": None,
            "content_hash": None,
            "acl_hash": None,
            "updated_at": None,
        }
    return {
        "version": revision.version,
        "etag": revision.etag,
        "content_hash": revision.content_hash,
        "acl_hash": revision.acl_hash,
        "updated_at": (
            revision.updated_at.isoformat() if revision.updated_at is not None else None
        ),
    }


def _build_delivery_id(
    *,
    tenant_id: str,
    binding_id: str,
    binding_configuration_version: int,
    mode: KnowledgeSyncMode,
    input_cursor: KnowledgeCursor | None,
    page: KnowledgePage,
) -> str:
    payload = {
        "tenant_id": tenant_id,
        "binding_id": binding_id,
        "binding_configuration_version": binding_configuration_version,
        "mode": mode.value,
        "input_cursor": _cursor_fingerprint(input_cursor),
        "proposed_checkpoint": _cursor_fingerprint(page.proposed_checkpoint),
        "next_cursor": _cursor_fingerprint(page.next_cursor),
        "changes": [
            {
                "kind": change.kind.value,
                "remote_id": change.remote_id,
                "revision": _revision_fingerprint(
                    change.descriptor.revision
                    if change.descriptor is not None
                    else None
                ),
            }
            for change in page.changes
        ],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class VendorKnowledgeSyncCoordinator:
    """Process at most one bounded synchronization page per call."""

    def __init__(
        self,
        *,
        tenant_id: str,
        owner_id: str,
        binding_service: KnowledgeSourceBindingService,
        facade: VendorKnowledgeFacade,
        lease_repository: KnowledgeSourceLeaseRepository,
        checkpoint_repository: KnowledgeSyncCheckpointRepository,
        item_state_repository: KnowledgeRemoteItemStateRepository,
        sink: KnowledgeSyncSink,
        lease_ttl_seconds: int,
        reconciliation_run_repository: KnowledgeReconciliationRunRepository
        | None = None,
        candidate_inventory_repository: (
            KnowledgeReconciliationCandidateInventoryRepository | None
        ) = None,
        sink_receipt_inspector: KnowledgeSyncSinkReceiptInspector | None = None,
    ) -> None:
        self._tenant_id = _require_non_empty(tenant_id, field_name="tenant_id")
        self._owner_id = _require_non_empty(owner_id, field_name="owner_id")
        if lease_ttl_seconds < 1 or lease_ttl_seconds > 3600:
            raise ValueError("lease_ttl_seconds must be in range 1..3600")
        self._binding_service = binding_service
        self._facade = facade
        self._lease_repository = lease_repository
        self._checkpoint_repository = checkpoint_repository
        self._item_state_repository = item_state_repository
        self._sink = sink
        self._lease_ttl_seconds = int(lease_ttl_seconds)
        self._reconciliation_engine: VendorKnowledgeReconciliationEngine | None = None
        if (
            reconciliation_run_repository is not None
            and candidate_inventory_repository is not None
        ):
            resolved_inspector = sink_receipt_inspector
            if resolved_inspector is None and isinstance(
                sink, KnowledgeSyncSinkReceiptInspector
            ):
                resolved_inspector = sink
            self._reconciliation_engine = VendorKnowledgeReconciliationEngine(
                tenant_id=self._tenant_id,
                binding_service=binding_service,
                facade=facade,
                reconciliation_run_repository=reconciliation_run_repository,
                candidate_inventory_repository=candidate_inventory_repository,
                checkpoint_repository=checkpoint_repository,
                item_state_repository=item_state_repository,
                sink=sink,
                sink_receipt_inspector=resolved_inspector,
            )

    async def sync_once(
        self,
        *,
        binding_id: str,
        page_size: int = 100,
    ) -> KnowledgeSyncRunResult:
        return await self._run_once(
            binding_id=binding_id,
            page_size=page_size,
            mode=KnowledgeSyncMode.INCREMENTAL,
            restart=True,
        )

    async def reconcile_once(
        self,
        *,
        binding_id: str,
        page_size: int = 100,
        restart: bool = True,
        operation_id: str | None = None,
        trigger_delivery_id: str | None = None,
    ) -> KnowledgeSyncRunResult:
        if self._reconciliation_engine is None:
            return await self._run_once(
                binding_id=binding_id,
                page_size=page_size,
                mode=KnowledgeSyncMode.RECONCILIATION,
                restart=restart,
            )
        cleaned_operation = _require_non_empty(
            operation_id or binding_id,
            field_name="operation_id",
        )
        lease = self._acquire_lease(binding_id=binding_id)
        if lease is None:
            return KnowledgeSyncRunResult(
                status=KnowledgeSyncRunStatus.LEASE_BUSY,
                mode=KnowledgeSyncMode.RECONCILIATION,
                tenant_id=self._tenant_id,
                binding_id=binding_id,
                delivery_id=None,
                changes_count=0,
                active_count=0,
                tombstone_count=0,
                checkpoint_advanced=False,
                has_more=False,
                retryable=True,
            )
        try:
            assert self._reconciliation_engine is not None
            return await self._reconciliation_engine.reconcile_page(
                binding_id=binding_id,
                operation_id=cleaned_operation,
                page_size=page_size,
                restart=restart,
                trigger_delivery_id=trigger_delivery_id,
            )
        finally:
            self._release_lease(lease)

    def execute_reconciliation_recovery(
        self,
        command: KnowledgeReconciliationRecoveryCommand,
    ) -> None:
        if self._reconciliation_engine is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Knowledge reconciliation recovery is not configured",
                retryable=False,
            )
        self._reconciliation_engine.execute_recovery_command(command)

    async def _run_once(
        self,
        *,
        binding_id: str,
        page_size: int,
        mode: KnowledgeSyncMode,
        restart: bool,
    ) -> KnowledgeSyncRunResult:
        cleaned_binding_id = _require_non_empty(binding_id, field_name="binding_id")
        if page_size < 1 or page_size > 1000:
            raise ValueError("page_size must be in range 1..1000")
        if mode is KnowledgeSyncMode.INCREMENTAL:
            self._block_incremental_for_active_reconciliation(
                binding_id=cleaned_binding_id
            )

        lease = self._acquire_lease(binding_id=cleaned_binding_id)
        if lease is None:
            return KnowledgeSyncRunResult(
                status=KnowledgeSyncRunStatus.LEASE_BUSY,
                mode=mode,
                tenant_id=self._tenant_id,
                binding_id=cleaned_binding_id,
                delivery_id=None,
                changes_count=0,
                active_count=0,
                tombstone_count=0,
                checkpoint_advanced=False,
                has_more=False,
                retryable=True,
            )

        operation_error: BaseException | None = None
        result: KnowledgeSyncRunResult | None = None
        try:
            result = await self._process_page(
                binding_id=cleaned_binding_id,
                page_size=page_size,
                mode=mode,
                restart=restart,
            )
        except BaseException as exc:
            operation_error = exc
        finally:
            release_error: BaseException | None = None
            try:
                self._release_lease(lease)
            except BaseException as exc:
                release_error = exc

        if operation_error is not None:
            raise operation_error
        if release_error is not None:
            if isinstance(release_error, VendorKnowledgeError):
                raise release_error
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Failed to release knowledge source lease",
                retryable=True,
            ) from None
        assert result is not None
        return result

    def _block_incremental_for_active_reconciliation(self, *, binding_id: str) -> None:
        if self._reconciliation_engine is None:
            return
        if not self._reconciliation_engine.has_active_run(binding_id=binding_id):
            return
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_CURSOR,
            safe_message=(
                "Incremental knowledge sync is blocked while reconciliation is active"
            ),
            retryable=False,
        )

    def _acquire_lease(self, *, binding_id: str) -> KnowledgeSourceLeaseToken | None:
        try:
            lease = self._lease_repository.acquire(
                tenant_id=self._tenant_id,
                binding_id=binding_id,
                owner_id=self._owner_id,
                ttl_seconds=self._lease_ttl_seconds,
            )
        except VendorKnowledgeError:
            raise
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source lease state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Failed to acquire knowledge source lease",
                retryable=True,
            ) from None

        if lease is None:
            return None
        if (
            lease.tenant_id != self._tenant_id
            or lease.binding_id != binding_id
            or lease.owner_id != self._owner_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source lease identity is inconsistent",
                retryable=False,
            )
        return lease

    def _release_lease(self, lease: KnowledgeSourceLeaseToken) -> None:
        try:
            self._lease_repository.release(lease=lease)
        except VendorKnowledgeError:
            raise
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source lease state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Failed to release knowledge source lease",
                retryable=True,
            ) from None

    async def _process_page(
        self,
        *,
        binding_id: str,
        page_size: int,
        mode: KnowledgeSyncMode,
        restart: bool,
    ) -> KnowledgeSyncRunResult:
        binding, source = self._load_binding_and_source(binding_id=binding_id)
        loaded_checkpoint = self._read_checkpoint(binding_id=binding_id)
        input_cursor = self._resolve_input_cursor(
            mode=mode,
            binding=binding,
            loaded_checkpoint=loaded_checkpoint,
            restart=restart,
        )

        try:
            scope_info = await self._facade.inspect_source(source=source)
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge source inspection failed",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None

        self._enforce_read_capabilities(
            mode=mode,
            input_cursor=input_cursor,
            capabilities=scope_info.capabilities,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
        )

        try:
            page = await self._facade.read_page(
                source=source,
                cursor=input_cursor,
                limit=page_size,
            )
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge source page read failed",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None

        self._validate_page(page=page, input_cursor=input_cursor, source=source)
        envelopes = await self._materialize_envelopes(
            page=page,
            source=source,
            scope_info=scope_info,
        )
        delivery_id = _build_delivery_id(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            binding_configuration_version=binding.configuration_version,
            mode=mode,
            input_cursor=input_cursor,
            page=page,
        )
        batch = KnowledgeSyncBatch(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            binding_configuration_version=binding.configuration_version,
            source=source,
            mode=mode,
            delivery_id=delivery_id,
            envelopes=tuple(envelopes),
            has_more=page.has_more,
        )
        await self._apply_durable_batch(
            batch=batch,
            binding=binding,
            loaded_checkpoint=loaded_checkpoint,
            page=page,
        )
        active_count = sum(
            1 for envelope in envelopes if envelope.change_kind in _ACTIVE_CHANGE_KINDS
        )
        tombstone_count = sum(
            1
            for envelope in envelopes
            if envelope.change_kind in _TOMBSTONE_CHANGE_KINDS
        )
        return KnowledgeSyncRunResult(
            status=KnowledgeSyncRunStatus.COMPLETED,
            mode=mode,
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            delivery_id=delivery_id,
            changes_count=len(envelopes),
            active_count=active_count,
            tombstone_count=tombstone_count,
            checkpoint_advanced=page.proposed_checkpoint is not None,
            has_more=page.has_more,
            retryable=False,
        )

    def _load_binding_and_source(
        self, *, binding_id: str
    ) -> tuple[KnowledgeSourceBinding, KnowledgeSourceRef]:
        try:
            binding = self._binding_service.get(binding_id)
            source = self._binding_service.resolve_source(binding_id)
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge source binding lookup failed",
                retryable=True,
            ) from None

        if binding.tenant_id != self._tenant_id or source.tenant_id != self._tenant_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.TENANT_MISMATCH,
                safe_message="Knowledge source binding tenant does not match coordinator tenant",
                retryable=False,
            )
        if binding.binding_id != binding_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding identity is inconsistent",
                retryable=False,
            )
        if (
            source.provider_id != binding.provider_id
            or source.integration_kind != binding.integration_kind
            or source.source_kind != binding.source_kind
            or source.connection_ref != binding.connection_ref
            or source.scope != binding.scope
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Resolved knowledge source does not match binding",
                retryable=False,
            )
        return binding, source

    def _read_checkpoint(self, *, binding_id: str) -> KnowledgeSyncCheckpoint | None:
        try:
            checkpoint = self._checkpoint_repository.get(
                tenant_id=self._tenant_id,
                binding_id=binding_id,
            )
        except VendorKnowledgeError:
            raise
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge sync checkpoint state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Failed to read knowledge sync checkpoint",
                retryable=True,
            ) from None

        if checkpoint is not None and (
            checkpoint.tenant_id != self._tenant_id
            or checkpoint.binding_id != binding_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge sync checkpoint identity is inconsistent",
                retryable=False,
            )
        return checkpoint

    def _resolve_input_cursor(
        self,
        *,
        mode: KnowledgeSyncMode,
        binding: KnowledgeSourceBinding,
        loaded_checkpoint: KnowledgeSyncCheckpoint | None,
        restart: bool,
    ) -> KnowledgeCursor | None:
        if mode is KnowledgeSyncMode.RECONCILIATION:
            if restart:
                return None
            if loaded_checkpoint is None:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                    safe_message=(
                        "Knowledge sync continuation requires an existing checkpoint; "
                        "no continuation checkpoint is available"
                    ),
                    provider_id=binding.provider_id,
                    source_kind=binding.source_kind,
                    retryable=False,
                )
            if (
                loaded_checkpoint.binding_configuration_version
                != binding.configuration_version
            ):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                    safe_message=(
                        "Knowledge sync checkpoint configuration version is stale; "
                        "restart reconciliation is required"
                    ),
                    provider_id=binding.provider_id,
                    source_kind=binding.source_kind,
                    retryable=False,
                )
            return loaded_checkpoint.cursor
        if loaded_checkpoint is None:
            return None
        if (
            loaded_checkpoint.binding_configuration_version
            != binding.configuration_version
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message=(
                    "Knowledge sync checkpoint configuration version is stale; "
                    "reconciliation is required"
                ),
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            )
        return loaded_checkpoint.cursor

    def _enforce_read_capabilities(
        self,
        *,
        mode: KnowledgeSyncMode,
        input_cursor: KnowledgeCursor | None,
        capabilities: KnowledgeAdapterCapabilities,
        provider_id: str,
        source_kind: str,
    ) -> None:
        if mode is KnowledgeSyncMode.RECONCILIATION:
            if not capabilities.reconciliation and not capabilities.full_inventory:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                    safe_message=(
                        "Reconciliation requires reconciliation or full_inventory capability"
                    ),
                    provider_id=provider_id,
                    source_kind=source_kind,
                    retryable=False,
                )
            return
        if input_cursor is None:
            if not capabilities.full_inventory and not capabilities.incremental_changes:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                    safe_message=(
                        "Incremental sync requires full_inventory or incremental_changes "
                        "capability"
                    ),
                    provider_id=provider_id,
                    source_kind=source_kind,
                    retryable=False,
                )
            return
        if not capabilities.incremental_changes:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                safe_message="Incremental sync requires incremental_changes capability",
                provider_id=provider_id,
                source_kind=source_kind,
                retryable=False,
            )

    def _validate_page(
        self,
        *,
        page: KnowledgePage,
        input_cursor: KnowledgeCursor | None,
        source: KnowledgeSourceRef,
    ) -> None:
        seen_remote_ids: set[str] = set()
        for change in page.changes:
            if change.remote_id in seen_remote_ids:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge page contains duplicate remote item identifiers",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )
            seen_remote_ids.add(change.remote_id)
            if change.kind in _ACTIVE_CHANGE_KINDS and change.descriptor is None:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Active knowledge change is missing a descriptor",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )

        if page.has_more:
            if page.next_cursor is None or page.proposed_checkpoint is None:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message=(
                        "Knowledge page with has_more requires next_cursor and "
                        "proposed_checkpoint"
                    ),
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )
            if page.next_cursor != page.proposed_checkpoint:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message=(
                        "Knowledge page continuation and checkpoint response are "
                        "inconsistent"
                    ),
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )
            if input_cursor is not None and page.proposed_checkpoint == input_cursor:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message=(
                        "Knowledge page proposed checkpoint must advance when more "
                        "pages remain"
                    ),
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )

    async def _materialize_envelopes(
        self,
        *,
        page: KnowledgePage,
        source: KnowledgeSourceRef,
        scope_info: KnowledgeScopeInfo,
    ) -> list[KnowledgeSyncEnvelope]:
        envelopes: list[KnowledgeSyncEnvelope] = []
        capabilities = scope_info.capabilities
        for change in page.changes:
            envelopes.append(
                await self._materialize_envelope(
                    change=change,
                    source=source,
                    content_fetch=capabilities.content_fetch,
                    permissions_enabled=capabilities.permissions,
                )
            )
        return envelopes

    async def _materialize_envelope(
        self,
        *,
        change: KnowledgeChange,
        source: KnowledgeSourceRef,
        content_fetch: bool,
        permissions_enabled: bool,
    ) -> KnowledgeSyncEnvelope:
        if change.kind in _TOMBSTONE_CHANGE_KINDS:
            return KnowledgeSyncEnvelope(
                change_kind=change.kind,
                remote_id=change.remote_id,
                descriptor=change.descriptor,
                content=None,
                permissions=None,
            )

        descriptor = change.descriptor
        if descriptor is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Active knowledge change is missing a descriptor",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        content = None
        permissions = None

        if change.kind is KnowledgeChangeKind.UPSERT:
            if descriptor.content_available and content_fetch:
                content = await self._fetch_content(source=source, item=descriptor)
            if permissions_enabled:
                permissions = await self._fetch_permissions(
                    source=source, item=descriptor
                )
        elif change.kind is KnowledgeChangeKind.METADATA_CHANGED:
            pass
        elif change.kind is KnowledgeChangeKind.PERMISSIONS_CHANGED:
            if not permissions_enabled:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
                    safe_message="Permissions changes require permissions capability",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )
            permissions = await self._fetch_permissions(source=source, item=descriptor)
        else:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Unsupported knowledge change kind",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        return KnowledgeSyncEnvelope(
            change_kind=change.kind,
            remote_id=change.remote_id,
            descriptor=descriptor,
            content=content,
            permissions=permissions,
        )

    async def _fetch_content(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        try:
            return await self._facade.fetch_content(source=source, item=item)
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge content fetch failed",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None

    async def _fetch_permissions(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        try:
            return await self._facade.fetch_permissions(source=source, item=item)
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge permissions fetch failed",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None

    async def _apply_durable_batch(
        self,
        *,
        batch: KnowledgeSyncBatch,
        binding: KnowledgeSourceBinding,
        loaded_checkpoint: KnowledgeSyncCheckpoint | None,
        page: KnowledgePage,
    ) -> None:
        try:
            await self._sink.apply_batch(batch=batch)
        except VendorKnowledgeError:
            raise
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge sync sink state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge sync sink failed to accept batch",
                retryable=True,
            ) from None

        states = self._build_remote_states(batch=batch, binding=binding)
        try:
            self._item_state_repository.apply_batch(
                tenant_id=batch.tenant_id,
                binding_id=batch.binding_id,
                delivery_id=batch.delivery_id,
                states=states,
            )
        except VendorKnowledgeError:
            raise
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge remote item state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Failed to apply knowledge remote item state batch",
                retryable=True,
            ) from None

        if page.proposed_checkpoint is None:
            return

        new_checkpoint = KnowledgeSyncCheckpoint(
            tenant_id=batch.tenant_id,
            binding_id=batch.binding_id,
            binding_configuration_version=binding.configuration_version,
            cursor=page.proposed_checkpoint,
        )
        try:
            self._checkpoint_repository.commit(
                new_checkpoint,
                expected_previous=loaded_checkpoint,
            )
        except VendorKnowledgeError:
            raise
        except KnowledgeSyncCheckpointConflict:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge sync checkpoint conflict",
                retryable=True,
            ) from None
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge sync checkpoint state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Failed to commit knowledge sync checkpoint",
                retryable=True,
            ) from None

    def _build_remote_states(
        self,
        *,
        batch: KnowledgeSyncBatch,
        binding: KnowledgeSourceBinding,
    ) -> tuple[KnowledgeRemoteItemState, ...]:
        states: list[KnowledgeRemoteItemState] = []
        for envelope in batch.envelopes:
            if envelope.change_kind is KnowledgeChangeKind.DELETED:
                status = KnowledgeRemoteItemStatus.DELETED
            elif envelope.change_kind is KnowledgeChangeKind.REVOKED:
                status = KnowledgeRemoteItemStatus.REVOKED
            else:
                status = KnowledgeRemoteItemStatus.ACTIVE

            revision: KnowledgeItemRevision | None = None
            if envelope.descriptor is not None:
                revision = envelope.descriptor.revision
            elif status is KnowledgeRemoteItemStatus.ACTIVE:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Active knowledge change is missing a descriptor",
                    provider_id=binding.provider_id,
                    source_kind=binding.source_kind,
                    retryable=False,
                )

            states.append(
                KnowledgeRemoteItemState(
                    tenant_id=batch.tenant_id,
                    binding_id=batch.binding_id,
                    binding_configuration_version=binding.configuration_version,
                    provider_id=binding.provider_id,
                    source_kind=binding.source_kind,
                    remote_id=envelope.remote_id,
                    status=status,
                    revision=revision,
                    last_delivery_id=batch.delivery_id,
                )
            )
        return tuple(states)
