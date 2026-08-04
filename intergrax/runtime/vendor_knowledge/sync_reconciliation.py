# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable provider-neutral reconciliation state machine for Vendor Knowledge."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
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
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeCandidateInventoryIncomplete,
    KnowledgeReconciliationCandidateInventoryRepository,
    KnowledgeReconciliationRunConflict,
    KnowledgeReconciliationRunRepository,
    KnowledgeRemoteItemStateRepository,
    KnowledgeSyncCheckpointConflict,
    KnowledgeSyncCheckpointRepository,
    KnowledgeSyncCorruptState,
    KnowledgeSyncSink,
    KnowledgeSyncSinkReceiptInspector,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationLimitPolicy,
    KnowledgeReconciliationMutationSemantic,
    KnowledgeReconciliationPreparedStateMutationTemplate,
    KnowledgeReconciliationRecoveryCommand,
    KnowledgeReconciliationRecoveryCommandKind,
    KnowledgeReconciliationRecoveryEvidenceCollecting,
    KnowledgeReconciliationRecoveryEvidenceFinalizing,
    KnowledgeReconciliationRecoveryEvidencePagePrepared,
    KnowledgeReconciliationRun,
    KnowledgeReconciliationRunAborted,
    KnowledgeReconciliationRunCollecting,
    KnowledgeReconciliationRunCompleted,
    KnowledgeReconciliationRunFinalizing,
    KnowledgeReconciliationRunPagePrepared,
    KnowledgeReconciliationRunPhase,
    KnowledgeReconciliationRunRecoveryRequired,
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStateReceiptStatus,
    KnowledgeRemoteItemStatus,
    KnowledgeSyncBatch,
    KnowledgeSyncCheckpoint,
    KnowledgeSyncEnvelope,
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
    KnowledgeSyncSinkReceiptStatus,
    canonical_prepared_state_mutations_fingerprint,
    knowledge_cursor_fingerprint_sha256,
    knowledge_sync_checkpoint_fingerprint_sha256,
    reconciliation_delivery_id,
    reconciliation_prepared_batch_payload_fingerprint,
    reconciliation_provider_page_fingerprint,
    recovery_evidence_from_run,
    canonical_reconciliation_candidate_inventory_bytes,
)

_ACTIVE_RUN_PHASES: frozenset[KnowledgeReconciliationRunPhase] = frozenset(
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

UtcClock = Callable[[], datetime]
RunIdFactory = Callable[[str, str, str], str]


class _JobInvocationClass(StrEnum):
    CURRENT_JOB = "current_job"
    NEXT_CONTINUATION = "next_continuation"
    SAME_JOB_REPLAY = "same_job_replay"
    STALE_OR_FOREIGN = "stale_or_foreign"


@dataclass(frozen=True)
class _PagePreparedRecoveryDecision:
    item_status: KnowledgeRemoteItemStateReceiptStatus
    sink_status: KnowledgeSyncSinkReceiptStatus | None


def _default_run_id_factory(
    tenant_id: str,
    binding_id: str,
    operation_id: str,
) -> str:
    return derive_reconciliation_run_id(
        tenant_id=tenant_id,
        binding_id=binding_id,
        operation_id=operation_id,
    )


def _default_utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def derive_reconciliation_run_id(
    *,
    tenant_id: str,
    binding_id: str,
    operation_id: str,
) -> str:
    payload = {
        "tenant_id": tenant_id,
        "binding_id": binding_id,
        "operation_id": operation_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


_CANDIDATE_INVENTORY_CORRUPT_MESSAGE = (
    "Knowledge reconciliation candidate inventory is corrupt"
)
_CANDIDATE_INVENTORY_LIMIT_MESSAGE = (
    "Knowledge reconciliation candidate inventory exceeds configured limit"
)
_CANDIDATE_REMOTE_ID_LIMIT_MESSAGE = (
    "Knowledge reconciliation candidate remote ID exceeds configured limit"
)
_CANDIDATE_PAYLOAD_LIMIT_MESSAGE = (
    "Knowledge reconciliation candidate payload exceeds configured limit"
)


def _raise_corrupt_candidate_inventory(*, source: KnowledgeSourceRef) -> None:
    raise VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
        safe_message=_CANDIDATE_INVENTORY_CORRUPT_MESSAGE,
        provider_id=source.provider_id,
        source_kind=source.source_kind,
        retryable=False,
    )


def _validate_repository_candidate_inventory(
    inventory: object,
    *,
    policy: KnowledgeReconciliationLimitPolicy,
    source: KnowledgeSourceRef,
) -> tuple[str, ...]:
    if isinstance(inventory, str) or not isinstance(inventory, (tuple, list)):
        _raise_corrupt_candidate_inventory(source=source)
    ordered = tuple(inventory)
    seen: set[str] = set()
    for remote_id in ordered:
        if not isinstance(remote_id, str):
            _raise_corrupt_candidate_inventory(source=source)
        cleaned = remote_id.strip()
        if not cleaned:
            _raise_corrupt_candidate_inventory(source=source)
        if cleaned in seen:
            _raise_corrupt_candidate_inventory(source=source)
        seen.add(cleaned)
    if len(ordered) > policy.max_reconciliation_candidate_count:
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
            safe_message=_CANDIDATE_INVENTORY_LIMIT_MESSAGE,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
            retryable=False,
        )
    for remote_id in ordered:
        if len(remote_id.encode("utf-8")) > policy.max_reconciliation_remote_id_bytes:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=_CANDIDATE_REMOTE_ID_LIMIT_MESSAGE,
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
    payload = canonical_reconciliation_candidate_inventory_bytes(ordered)
    if len(payload) > policy.max_reconciliation_candidate_payload_bytes:
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
            safe_message=_CANDIDATE_PAYLOAD_LIMIT_MESSAGE,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
            retryable=False,
        )
    return ordered


def _advance_fields(
    run: KnowledgeReconciliationRun, *, updated_at: datetime
) -> dict[str, Any]:
    return {
        "record_version": run.record_version + 1,
        "updated_at": updated_at,
    }


class VendorKnowledgeReconciliationEngine:
    """One provider page per call with durable PAGE_PREPARED boundaries."""

    def __init__(
        self,
        *,
        tenant_id: str,
        binding_service: KnowledgeSourceBindingService,
        facade: VendorKnowledgeFacade,
        reconciliation_run_repository: KnowledgeReconciliationRunRepository,
        candidate_inventory_repository: KnowledgeReconciliationCandidateInventoryRepository,
        checkpoint_repository: KnowledgeSyncCheckpointRepository,
        item_state_repository: KnowledgeRemoteItemStateRepository,
        sink: KnowledgeSyncSink,
        sink_receipt_inspector: KnowledgeSyncSinkReceiptInspector | None,
        limit_policy: KnowledgeReconciliationLimitPolicy | None = None,
        utc_clock: UtcClock | None = None,
        run_id_factory: RunIdFactory | None = None,
    ) -> None:
        self._tenant_id = _require_non_empty(tenant_id, field_name="tenant_id")
        self._binding_service = binding_service
        self._facade = facade
        self._run_repository = reconciliation_run_repository
        self._candidate_inventory = candidate_inventory_repository
        self._checkpoint_repository = checkpoint_repository
        self._item_state_repository = item_state_repository
        self._sink = sink
        self._sink_receipt_inspector = sink_receipt_inspector
        self._policy = limit_policy or KnowledgeReconciliationLimitPolicy()
        self._utc_clock = utc_clock or _default_utc_clock
        self._run_id_factory = run_id_factory or _default_run_id_factory

    async def reconcile_page(
        self,
        *,
        binding_id: str,
        operation_id: str,
        page_size: int,
        restart: bool,
        trigger_delivery_id: str | None,
    ) -> KnowledgeSyncRunResult:
        if page_size < 1 or page_size > 1000:
            raise ValueError("page_size must be in range 1..1000")
        if self._sink_receipt_inspector is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Knowledge reconciliation receipt inspection is not configured",
                retryable=False,
            )
        binding, source = self._load_binding_and_source(binding_id=binding_id)
        run_id = self._run_id_factory(
            self._tenant_id,
            binding.binding_id,
            operation_id,
        )
        existing_run = self._read_run(binding_id=binding.binding_id)
        invocation = self._classify_job_invocation(
            existing_run=existing_run,
            run_id=run_id,
            restart=restart,
            trigger_delivery_id=trigger_delivery_id,
        )
        if invocation is _JobInvocationClass.STALE_OR_FOREIGN:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Knowledge reconciliation continuation is stale",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if invocation is _JobInvocationClass.SAME_JOB_REPLAY:
            assert existing_run is not None
            return self._replay_same_job_result(
                run=existing_run,
                binding=binding,
            )
        loaded_checkpoint: KnowledgeSyncCheckpoint | None = None
        if existing_run is None or (existing_run.run_id != run_id and restart):
            loaded_checkpoint = self._read_checkpoint(binding_id=binding.binding_id)
        if (
            loaded_checkpoint is not None
            and loaded_checkpoint.binding_configuration_version
            != binding.configuration_version
            and existing_run is None
            and not restart
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Knowledge reconciliation requires restart after binding configuration change",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        run = self._resolve_run_for_invocation(
            existing_run=existing_run,
            run_id=run_id,
            binding=binding,
            source=source,
            loaded_checkpoint=loaded_checkpoint,
            restart=restart,
            trigger_delivery_id=trigger_delivery_id,
            invocation=invocation,
        )
        if run.phase is KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation requires operator recovery",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if run.phase is KnowledgeReconciliationRunPhase.COLLECTING:
            assert isinstance(run, KnowledgeReconciliationRunCollecting)
            return await self._process_collecting(
                run=run,
                binding=binding,
                source=source,
                page_size=page_size,
                trigger_delivery_id=trigger_delivery_id,
            )
        if run.phase is KnowledgeReconciliationRunPhase.PAGE_PREPARED:
            assert isinstance(run, KnowledgeReconciliationRunPagePrepared)
            return await self._process_page_prepared(
                run=run,
                binding=binding,
                source=source,
                page_size=page_size,
            )
        if run.phase is KnowledgeReconciliationRunPhase.FINALIZING:
            assert isinstance(run, KnowledgeReconciliationRunFinalizing)
            return await self._process_finalizing(
                run=run,
                binding=binding,
                source=source,
            )
        if run.phase is KnowledgeReconciliationRunPhase.COMPLETED:
            assert isinstance(run, KnowledgeReconciliationRunCompleted)
            return self._completed_result(
                run=run,
                binding=binding,
                checkpoint_advanced=True,
                has_more=False,
                delivery_id=run.final_delivery_id,
                templates=(),
            )
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message="Knowledge reconciliation run state is invalid",
            provider_id=source.provider_id,
            source_kind=source.source_kind,
            retryable=False,
        )

    def execute_recovery_command(
        self,
        command: KnowledgeReconciliationRecoveryCommand,
    ) -> KnowledgeReconciliationRun:
        return self._execute_recovery_command_sync(command)

    async def execute_recovery_command_async(
        self,
        command: KnowledgeReconciliationRecoveryCommand,
    ) -> KnowledgeReconciliationRun:
        if command.kind is KnowledgeReconciliationRecoveryCommandKind.RESUME_EXACT:
            return await self._resume_exact_async(command)
        return self._execute_recovery_command_sync(command)

    def _execute_recovery_command_sync(
        self,
        command: KnowledgeReconciliationRecoveryCommand,
    ) -> KnowledgeReconciliationRun:
        run = self._read_run(binding_id=command.binding_id)
        if run is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation run was not found",
                retryable=False,
            )
        if (
            run.run_id != command.expected_run_id
            or run.record_version != command.expected_run_record_version
            or run.phase != command.expected_phase
            or run.tenant_id != command.tenant_id
            or run.binding_id != command.binding_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation recovery command is stale",
                retryable=False,
            )
        now = self._utc_clock()
        if command.kind is KnowledgeReconciliationRecoveryCommandKind.RESUME_EXACT:
            return self._resume_exact_sync(run=run, now=now)
        if (
            command.kind
            is KnowledgeReconciliationRecoveryCommandKind.FINALIZE_ALREADY_COMMITTED
        ):
            if not isinstance(run, KnowledgeReconciliationRunFinalizing):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge reconciliation finalize requires finalizing state",
                    retryable=False,
                )
            current = self._read_checkpoint(binding_id=run.binding_id)
            if current != run.intended_final_completed_checkpoint:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge reconciliation checkpoint does not match intended final state",
                    retryable=False,
                )
            completed = KnowledgeReconciliationRunCompleted(
                **run.model_dump(
                    exclude={
                        "phase",
                        "record_version",
                        "updated_at",
                        "final_delivery_id",
                        "intended_final_completed_checkpoint",
                        "intended_final_checkpoint_fingerprint",
                        "expected_previous_completed_checkpoint",
                        "prepared_batch_payload_fingerprint",
                    }
                ),
                phase=KnowledgeReconciliationRunPhase.COMPLETED,
                committed_completed_checkpoint=run.intended_final_completed_checkpoint,
                final_delivery_id=run.final_delivery_id,
                **_advance_fields(run, updated_at=now),
            )
            self._cas_recovery(expected=run, replacement=completed)
            return completed
        if command.kind is KnowledgeReconciliationRecoveryCommandKind.ABORT_PRISTINE:
            return self._abort_pristine(
                run=run, operator_reason_code=command.operator_reason_code, now=now
            )
        if command.kind is KnowledgeReconciliationRecoveryCommandKind.REPAIR_REQUIRED:
            if not isinstance(run, KnowledgeReconciliationRunRecoveryRequired):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge reconciliation repair requires recovery state",
                    retryable=False,
                )
            replacement = run.model_copy(
                update={
                    "recovery_reason_code": command.operator_reason_code,
                    "machine_recovery_reason_code": run.machine_recovery_reason_code
                    or run.recovery_reason_code,
                    **_advance_fields(run, updated_at=now),
                }
            )
            self._cas_recovery(expected=run, replacement=replacement)
            return replacement
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message="Knowledge reconciliation recovery command is unsupported",
            retryable=False,
        )

    def has_active_run(self, *, binding_id: str) -> bool:
        run = self._read_run(binding_id=binding_id)
        return run is not None and run.phase in _ACTIVE_RUN_PHASES

    async def _process_collecting(
        self,
        *,
        run: KnowledgeReconciliationRunCollecting,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
        page_size: int,
        trigger_delivery_id: str | None,
    ) -> KnowledgeSyncRunResult:
        if run.binding_configuration_version != binding.configuration_version:
            return await self._enter_recovery(
                run=run,
                reason_code="binding_configuration_mismatch",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Knowledge reconciliation binding configuration is stale",
                retryable=False,
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
        self._enforce_reconciliation_capabilities(
            input_cursor=run.current_input_cursor,
            capabilities=scope_info.capabilities,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
        )
        try:
            page = await self._facade.read_page(
                source=source,
                cursor=run.current_input_cursor,
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
        self._validate_page(
            page=page, input_cursor=run.current_input_cursor, source=source
        )
        if not page.has_more and page.proposed_checkpoint is None:
            return await self._enter_recovery(
                run=run,
                reason_code="missing_final_checkpoint",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation final page is missing checkpoint",
                retryable=False,
            )
        envelopes = await self._materialize_envelopes(
            page=page,
            source=source,
            scope_info=scope_info,
        )
        remaining_before = set(run.remaining_candidate_remote_ids) | {
            change.remote_id for change in page.changes
        }
        remaining = sorted(
            remaining_before - {change.remote_id for change in page.changes}
        )
        remaining_tuple = tuple(sorted(remaining))
        synthetic_ids: tuple[str, ...] = ()
        if not page.has_more:
            synthetic_ids = remaining_tuple
            for remote_id in synthetic_ids:
                envelopes.append(
                    KnowledgeSyncEnvelope(
                        change_kind=KnowledgeChangeKind.DELETED,
                        remote_id=remote_id,
                        reconciliation_semantic=KnowledgeReconciliationMutationSemantic.ABSENT_FROM_COMPLETED_SYNCHRONIZED_SOURCE_INVENTORY,
                    )
                )
        templates = self._build_templates(
            envelopes=envelopes,
            binding_configuration_version=binding.configuration_version,
        )
        mutations_fp = canonical_prepared_state_mutations_fingerprint(templates)
        input_fp = run.current_input_cursor_fingerprint
        proposed_fp = knowledge_cursor_fingerprint_sha256(page.proposed_checkpoint)
        next_fp = knowledge_cursor_fingerprint_sha256(page.next_cursor)
        provider_fp = reconciliation_provider_page_fingerprint(
            input_cursor_fingerprint=input_fp,
            has_more=page.has_more,
            proposed_checkpoint_fingerprint=proposed_fp,
            next_cursor_fingerprint=next_fp,
            changes=tuple(
                (
                    change.remote_id,
                    change.kind,
                    change.descriptor.revision if change.descriptor else None,
                )
                for change in page.changes
            ),
        )
        batch_fp = reconciliation_prepared_batch_payload_fingerprint(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            binding_configuration_version=binding.configuration_version,
            mode=KnowledgeSyncMode.RECONCILIATION,
            run_id=run.run_id,
            source=source,
            has_more=page.has_more,
            envelopes=tuple(envelopes),
            prepared_state_mutations_fingerprint=mutations_fp,
            provider_page_fingerprint=provider_fp,
            input_cursor_fingerprint=input_fp,
            proposed_checkpoint_fingerprint=proposed_fp,
            next_cursor_fingerprint=next_fp,
        )
        delivery_id = reconciliation_delivery_id(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            binding_configuration_version=binding.configuration_version,
            mode=KnowledgeSyncMode.RECONCILIATION,
            run_id=run.run_id,
            provider_page_fingerprint=provider_fp,
            prepared_batch_payload_fingerprint=batch_fp,
            prepared_state_mutations_fingerprint=mutations_fp,
            input_cursor_fingerprint=input_fp,
            proposed_checkpoint_fingerprint=proposed_fp,
            next_cursor_fingerprint=next_fp,
        )
        now = self._utc_clock()
        prepared = KnowledgeReconciliationRunPagePrepared(
            **run.model_dump(
                exclude={
                    "phase",
                    "current_input_cursor",
                    "current_input_cursor_fingerprint",
                    "remaining_candidate_remote_ids",
                    "candidate_inventory_continuation_token",
                    "record_version",
                    "updated_at",
                }
            ),
            phase=KnowledgeReconciliationRunPhase.PAGE_PREPARED,
            prepared_input_cursor=run.current_input_cursor,
            prepared_input_cursor_fingerprint=input_fp,
            provider_page_fingerprint=provider_fp,
            prepared_batch_payload_fingerprint=batch_fp,
            prepared_state_mutation_templates=templates,
            prepared_state_mutations_fingerprint=mutations_fp,
            prepared_proposed_checkpoint=page.proposed_checkpoint,
            prepared_proposed_checkpoint_fingerprint=proposed_fp,
            prepared_next_cursor=page.next_cursor,
            prepared_next_cursor_fingerprint=next_fp,
            prepared_page_size=page_size,
            has_more=page.has_more,
            delivery_id=delivery_id,
            prepared_parent_delivery_id=trigger_delivery_id,
            remaining_candidate_remote_ids=remaining_tuple,
            synthetic_tombstone_remote_ids=synthetic_ids,
            **_advance_fields(run, updated_at=now),
        )
        self._cas_replace(expected=run, replacement=prepared)
        return await self._process_page_prepared(
            run=prepared,
            binding=binding,
            source=source,
            page_size=page_size,
            trusted_envelopes=tuple(envelopes),
        )

    async def _process_page_prepared(
        self,
        *,
        run: KnowledgeReconciliationRunPagePrepared,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
        page_size: int,
        trusted_envelopes: tuple[KnowledgeSyncEnvelope, ...] | None = None,
    ) -> KnowledgeSyncRunResult:
        if run.binding_configuration_version != binding.configuration_version:
            return await self._enter_recovery(
                run=run,
                reason_code="binding_configuration_mismatch",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Knowledge reconciliation binding configuration is stale",
                retryable=False,
            )
        try:
            item_receipt = self._inspect_item_receipt_for_page_prepared(run=run)
        except KnowledgeSyncCorruptState:
            return await self._transition_page_prepared_receipt_corruption(
                run=run,
                source=source,
                reason_code="item_state_receipt_corrupt",
            )
        if item_receipt.status is KnowledgeRemoteItemStateReceiptStatus.COMPLETED:
            return await self._transition_applied_page(
                run=run,
                binding=binding,
                source=source,
            )
        if item_receipt.status is KnowledgeRemoteItemStateReceiptStatus.APPLYING:
            states = self._states_from_templates(
                run=run, binding=binding, source=source
            )
            self._apply_states(run=run, states=states)
            return await self._transition_applied_page(
                run=run, binding=binding, source=source
            )
        if item_receipt.status is KnowledgeRemoteItemStateReceiptStatus.CONFLICT:
            return await self._enter_recovery(
                run=run,
                reason_code="item_state_receipt_conflict",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation item state receipt conflict",
                retryable=False,
            )
        try:
            sink_receipt = self._inspect_sink_receipt_for_page_prepared(run=run)
        except KnowledgeSyncCorruptState:
            return await self._transition_page_prepared_receipt_corruption(
                run=run,
                source=source,
                reason_code="sink_receipt_corrupt",
            )
        if sink_receipt.status is KnowledgeSyncSinkReceiptStatus.UNKNOWN:
            return await self._enter_recovery(
                run=run,
                reason_code="sink_receipt_unknown",
                source=source,
                error_code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge delivery outcome requires recovery",
                retryable=False,
            )
        if sink_receipt.status is KnowledgeSyncSinkReceiptStatus.CONFLICT:
            return await self._enter_recovery(
                run=run,
                reason_code="sink_receipt_conflict",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation sink receipt conflict",
                retryable=False,
            )
        if sink_receipt.status is KnowledgeSyncSinkReceiptStatus.APPLIED:
            states = self._states_from_templates(
                run=run, binding=binding, source=source
            )
            self._apply_states(run=run, states=states)
            return await self._transition_applied_page(
                run=run, binding=binding, source=source
            )
        assert sink_receipt.status is KnowledgeSyncSinkReceiptStatus.ABSENT
        if item_receipt.status is not KnowledgeRemoteItemStateReceiptStatus.ABSENT:
            return await self._enter_recovery(
                run=run,
                reason_code="item_state_receipt_unexpected",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation item state receipt is inconsistent",
                retryable=False,
            )
        if trusted_envelopes is not None:
            envelopes = trusted_envelopes
        else:
            reproduced = await self._reproduce_prepared_page(
                run=run,
                binding=binding,
                source=source,
                page_size=run.prepared_page_size,
            )
            if reproduced is None:
                return await self._enter_recovery(
                    run=run,
                    reason_code="provider_page_mismatch",
                    source=source,
                    error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge reconciliation prepared page no longer matches provider",
                    retryable=False,
                )
            envelopes, _templates, provider_fp, batch_fp, mutations_fp = reproduced
            if (
                provider_fp != run.provider_page_fingerprint
                or batch_fp != run.prepared_batch_payload_fingerprint
                or mutations_fp != run.prepared_state_mutations_fingerprint
            ):
                return await self._enter_recovery(
                    run=run,
                    reason_code="prepared_fingerprint_mismatch",
                    source=source,
                    error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge reconciliation prepared page no longer matches provider",
                    retryable=False,
                )
        batch = KnowledgeSyncBatch(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            binding_configuration_version=binding.configuration_version,
            source=source,
            mode=KnowledgeSyncMode.RECONCILIATION,
            delivery_id=run.delivery_id,
            envelopes=envelopes,
            has_more=run.has_more,
        )
        await self._apply_sink(batch=batch, source=source)
        states = self._states_from_templates(run=run, binding=binding, source=source)
        self._apply_states(run=run, states=states)
        return await self._transition_applied_page(
            run=run, binding=binding, source=source
        )

    async def _process_finalizing(
        self,
        *,
        run: KnowledgeReconciliationRunFinalizing,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
    ) -> KnowledgeSyncRunResult:
        if run.binding_configuration_version != binding.configuration_version:
            return await self._enter_recovery(
                run=run,
                reason_code="binding_configuration_mismatch",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Knowledge reconciliation binding configuration is stale",
                retryable=False,
            )
        try:
            current = self._read_checkpoint_for_finalizing(
                binding_id=binding.binding_id
            )
        except KnowledgeSyncCorruptState:
            return await self._enter_recovery(
                run=run,
                reason_code="checkpoint_read_corrupt",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge sync checkpoint state is corrupt",
                retryable=False,
            )
        intended = run.intended_final_completed_checkpoint
        expected_previous = run.expected_previous_completed_checkpoint
        if current == expected_previous:
            try:
                self._checkpoint_repository.commit(
                    intended,
                    expected_previous=expected_previous,
                )
            except KnowledgeSyncCheckpointConflict:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                    safe_message="Knowledge sync checkpoint conflict",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=True,
                ) from None
            except KnowledgeSyncCorruptState:
                return await self._enter_recovery(
                    run=run,
                    reason_code="checkpoint_commit_corrupt",
                    source=source,
                    error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge sync checkpoint state is corrupt",
                    retryable=False,
                )
            except Exception:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                    safe_message="Knowledge sync checkpoint commit failed",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=True,
                ) from None
        elif current != intended:
            return await self._enter_recovery(
                run=run,
                reason_code="checkpoint_divergence",
                source=source,
                error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation checkpoint diverged from expected state",
                retryable=False,
            )
        now = self._utc_clock()
        completed = KnowledgeReconciliationRunCompleted(
            **run.model_dump(
                exclude={
                    "phase",
                    "record_version",
                    "updated_at",
                    "final_delivery_id",
                    "intended_final_completed_checkpoint",
                    "intended_final_checkpoint_fingerprint",
                    "expected_previous_completed_checkpoint",
                    "prepared_batch_payload_fingerprint",
                }
            ),
            phase=KnowledgeReconciliationRunPhase.COMPLETED,
            committed_completed_checkpoint=intended,
            final_delivery_id=run.final_delivery_id,
            **_advance_fields(run, updated_at=now),
        )
        self._cas_replace(expected=run, replacement=completed)
        return self._completed_result(
            run=completed,
            binding=binding,
            checkpoint_advanced=True,
            has_more=False,
            delivery_id=run.final_delivery_id,
            templates=(),
        )

    async def _transition_applied_page(
        self,
        *,
        run: KnowledgeReconciliationRunPagePrepared,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
    ) -> KnowledgeSyncRunResult:
        now = self._utc_clock()
        if run.has_more:
            collecting = KnowledgeReconciliationRunCollecting(
                **run.model_dump(
                    exclude={
                        "phase",
                        "prepared_input_cursor",
                        "prepared_input_cursor_fingerprint",
                        "provider_page_fingerprint",
                        "prepared_batch_payload_fingerprint",
                        "prepared_state_mutation_templates",
                        "prepared_state_mutations_fingerprint",
                        "prepared_proposed_checkpoint",
                        "prepared_proposed_checkpoint_fingerprint",
                        "prepared_next_cursor",
                        "prepared_next_cursor_fingerprint",
                        "prepared_page_size",
                        "delivery_id",
                        "has_more",
                        "synthetic_tombstone_remote_ids",
                        "remaining_candidate_remote_ids",
                        "record_version",
                        "updated_at",
                        "applied_page_count",
                        "last_applied_delivery_id",
                        "last_applied_parent_delivery_id",
                        "prepared_parent_delivery_id",
                    }
                ),
                phase=KnowledgeReconciliationRunPhase.COLLECTING,
                applied_page_count=run.applied_page_count + 1,
                last_applied_delivery_id=run.delivery_id,
                last_applied_parent_delivery_id=run.prepared_parent_delivery_id,
                current_input_cursor=run.prepared_next_cursor,
                current_input_cursor_fingerprint=run.prepared_next_cursor_fingerprint,
                remaining_candidate_remote_ids=run.remaining_candidate_remote_ids,
                **_advance_fields(run, updated_at=now),
            )
            self._cas_replace(expected=run, replacement=collecting)
            return self._completed_result(
                run=collecting,
                binding=binding,
                checkpoint_advanced=False,
                has_more=True,
                delivery_id=run.delivery_id,
                templates=run.prepared_state_mutation_templates,
            )
        intended_checkpoint = KnowledgeSyncCheckpoint(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            binding_configuration_version=binding.configuration_version,
            cursor=run.prepared_proposed_checkpoint,
        )
        finalizing = KnowledgeReconciliationRunFinalizing(
            **run.model_dump(
                exclude={
                    "phase",
                    "prepared_input_cursor",
                    "prepared_input_cursor_fingerprint",
                    "provider_page_fingerprint",
                    "prepared_batch_payload_fingerprint",
                    "prepared_state_mutation_templates",
                    "prepared_state_mutations_fingerprint",
                    "prepared_proposed_checkpoint",
                    "prepared_proposed_checkpoint_fingerprint",
                    "prepared_next_cursor",
                    "prepared_next_cursor_fingerprint",
                    "prepared_page_size",
                    "delivery_id",
                    "has_more",
                    "remaining_candidate_remote_ids",
                    "synthetic_tombstone_remote_ids",
                    "record_version",
                    "updated_at",
                    "applied_page_count",
                    "last_applied_delivery_id",
                    "last_applied_parent_delivery_id",
                    "prepared_parent_delivery_id",
                }
            ),
            phase=KnowledgeReconciliationRunPhase.FINALIZING,
            applied_page_count=run.applied_page_count + 1,
            last_applied_delivery_id=run.delivery_id,
            last_applied_parent_delivery_id=run.prepared_parent_delivery_id,
            intended_final_completed_checkpoint=intended_checkpoint,
            intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
                intended_checkpoint
            ),
            expected_previous_completed_checkpoint=run.expected_base_completed_checkpoint,
            final_delivery_id=run.delivery_id,
            prepared_batch_payload_fingerprint=run.prepared_batch_payload_fingerprint,
            **_advance_fields(run, updated_at=now),
        )
        self._cas_replace(expected=run, replacement=finalizing)
        return await self._process_finalizing(
            run=finalizing, binding=binding, source=source
        )

    async def _reproduce_prepared_page(
        self,
        *,
        run: KnowledgeReconciliationRunPagePrepared,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
        page_size: int,
    ) -> (
        tuple[
            tuple[KnowledgeSyncEnvelope, ...],
            tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...],
            str,
            str,
            str,
        ]
        | None
    ):
        try:
            scope_info = await self._facade.inspect_source(source=source)
            page = await self._facade.read_page(
                source=source,
                cursor=run.prepared_input_cursor,
                limit=page_size,
            )
        except VendorKnowledgeError:
            raise
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation provider state is corrupt",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation provider reproduction failed",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None
        self._validate_page(
            page=page, input_cursor=run.prepared_input_cursor, source=source
        )
        envelopes = await self._materialize_envelopes(
            page=page,
            source=source,
            scope_info=scope_info,
        )
        remaining_before = set(run.remaining_candidate_remote_ids) | {
            change.remote_id for change in page.changes
        }
        remaining = sorted(
            remaining_before - {change.remote_id for change in page.changes}
        )
        if not page.has_more:
            for remote_id in tuple(sorted(remaining)):
                envelopes.append(
                    KnowledgeSyncEnvelope(
                        change_kind=KnowledgeChangeKind.DELETED,
                        remote_id=remote_id,
                        reconciliation_semantic=KnowledgeReconciliationMutationSemantic.ABSENT_FROM_COMPLETED_SYNCHRONIZED_SOURCE_INVENTORY,
                    )
                )
        templates = self._build_templates(
            envelopes=envelopes,
            binding_configuration_version=binding.configuration_version,
        )
        mutations_fp = canonical_prepared_state_mutations_fingerprint(templates)
        input_fp = run.prepared_input_cursor_fingerprint
        proposed_fp = knowledge_cursor_fingerprint_sha256(page.proposed_checkpoint)
        next_fp = knowledge_cursor_fingerprint_sha256(page.next_cursor)
        provider_fp = reconciliation_provider_page_fingerprint(
            input_cursor_fingerprint=input_fp,
            has_more=page.has_more,
            proposed_checkpoint_fingerprint=proposed_fp,
            next_cursor_fingerprint=next_fp,
            changes=tuple(
                (
                    change.remote_id,
                    change.kind,
                    change.descriptor.revision if change.descriptor else None,
                )
                for change in page.changes
            ),
        )
        batch_fp = reconciliation_prepared_batch_payload_fingerprint(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            binding_configuration_version=binding.configuration_version,
            mode=KnowledgeSyncMode.RECONCILIATION,
            run_id=run.run_id,
            source=source,
            has_more=page.has_more,
            envelopes=tuple(envelopes),
            prepared_state_mutations_fingerprint=mutations_fp,
            provider_page_fingerprint=provider_fp,
            input_cursor_fingerprint=input_fp,
            proposed_checkpoint_fingerprint=proposed_fp,
            next_cursor_fingerprint=next_fp,
        )
        return tuple(envelopes), templates, provider_fp, batch_fp, mutations_fp

    def _resolve_run_for_invocation(
        self,
        *,
        existing_run: KnowledgeReconciliationRun | None,
        run_id: str,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
        loaded_checkpoint: KnowledgeSyncCheckpoint | None,
        restart: bool,
        trigger_delivery_id: str | None,
        invocation: _JobInvocationClass,
    ) -> KnowledgeReconciliationRun:
        if existing_run is None:
            return self._create_initial_run(
                run_id=run_id,
                binding=binding,
                source=source,
                loaded_checkpoint=loaded_checkpoint,
            )
        if existing_run.run_id != run_id:
            if not restart:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                    safe_message="Knowledge reconciliation continuation is stale",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )
            if existing_run.phase in {
                KnowledgeReconciliationRunPhase.FINALIZING,
                KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
            }:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                    safe_message="Knowledge reconciliation restart is blocked for active finalization or recovery",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )
            superseded_run: KnowledgeReconciliationRun
            if existing_run.phase in {
                KnowledgeReconciliationRunPhase.COLLECTING,
                KnowledgeReconciliationRunPhase.PAGE_PREPARED,
            }:
                superseded_run = self._abort_pristine(
                    run=existing_run,
                    operator_reason_code="restart_supersede",
                    now=self._utc_clock(),
                )
            elif existing_run.phase in {
                KnowledgeReconciliationRunPhase.COMPLETED,
                KnowledgeReconciliationRunPhase.ABORTED,
            }:
                superseded_run = existing_run
            else:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                    safe_message="Knowledge reconciliation restart is blocked",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=False,
                )
            return self._create_initial_run(
                run_id=run_id,
                binding=binding,
                source=source,
                loaded_checkpoint=loaded_checkpoint,
                superseded_run=superseded_run,
            )
        return existing_run

    def _classify_job_invocation(
        self,
        *,
        existing_run: KnowledgeReconciliationRun | None,
        run_id: str,
        restart: bool,
        trigger_delivery_id: str | None,
    ) -> _JobInvocationClass:
        is_initial = restart and trigger_delivery_id is None
        is_continuation = (not restart) and trigger_delivery_id is not None
        if not is_initial and not is_continuation:
            return _JobInvocationClass.STALE_OR_FOREIGN
        if existing_run is None:
            if is_continuation:
                return _JobInvocationClass.STALE_OR_FOREIGN
            return _JobInvocationClass.CURRENT_JOB
        if existing_run.run_id != run_id:
            if is_initial:
                return _JobInvocationClass.CURRENT_JOB
            return _JobInvocationClass.STALE_OR_FOREIGN
        if is_initial:
            return self._classify_initial_job_invocation(existing_run)
        assert trigger_delivery_id is not None
        return self._classify_continuation_job_invocation(
            existing_run,
            trigger_delivery_id=trigger_delivery_id,
        )

    def _classify_initial_job_invocation(
        self,
        run: KnowledgeReconciliationRun,
    ) -> _JobInvocationClass:
        if run.last_applied_parent_delivery_id is not None:
            return _JobInvocationClass.STALE_OR_FOREIGN
        if isinstance(run, KnowledgeReconciliationRunCollecting):
            if run.applied_page_count == 0:
                return _JobInvocationClass.CURRENT_JOB
            if run.applied_page_count == 1 and run.last_applied_delivery_id is not None:
                return _JobInvocationClass.SAME_JOB_REPLAY
            return _JobInvocationClass.STALE_OR_FOREIGN
        if isinstance(run, KnowledgeReconciliationRunPagePrepared):
            if run.applied_page_count == 0 and run.prepared_parent_delivery_id is None:
                return _JobInvocationClass.CURRENT_JOB
            return _JobInvocationClass.STALE_OR_FOREIGN
        if run.phase is KnowledgeReconciliationRunPhase.FINALIZING:
            return _JobInvocationClass.CURRENT_JOB
        if isinstance(run, KnowledgeReconciliationRunCompleted):
            return _JobInvocationClass.SAME_JOB_REPLAY
        if run.phase is KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED:
            return _JobInvocationClass.CURRENT_JOB
        return _JobInvocationClass.STALE_OR_FOREIGN

    def _classify_continuation_job_invocation(
        self,
        run: KnowledgeReconciliationRun,
        *,
        trigger_delivery_id: str,
    ) -> _JobInvocationClass:
        if isinstance(run, KnowledgeReconciliationRunPagePrepared):
            if run.prepared_parent_delivery_id == trigger_delivery_id:
                return _JobInvocationClass.CURRENT_JOB
            return _JobInvocationClass.STALE_OR_FOREIGN
        if isinstance(run, KnowledgeReconciliationRunCollecting):
            if run.last_applied_delivery_id == trigger_delivery_id:
                return _JobInvocationClass.NEXT_CONTINUATION
            if run.last_applied_parent_delivery_id == trigger_delivery_id:
                return _JobInvocationClass.SAME_JOB_REPLAY
            return _JobInvocationClass.STALE_OR_FOREIGN
        if isinstance(run, KnowledgeReconciliationRunFinalizing):
            if run.last_applied_parent_delivery_id == trigger_delivery_id:
                return _JobInvocationClass.CURRENT_JOB
            return _JobInvocationClass.STALE_OR_FOREIGN
        if isinstance(run, KnowledgeReconciliationRunCompleted):
            if run.last_applied_parent_delivery_id == trigger_delivery_id:
                return _JobInvocationClass.SAME_JOB_REPLAY
            return _JobInvocationClass.STALE_OR_FOREIGN
        return _JobInvocationClass.STALE_OR_FOREIGN

    def _replay_same_job_result(
        self,
        *,
        run: KnowledgeReconciliationRun,
        binding: KnowledgeSourceBinding,
    ) -> KnowledgeSyncRunResult:
        if isinstance(run, KnowledgeReconciliationRunCompleted):
            return self._completed_result(
                run=run,
                binding=binding,
                checkpoint_advanced=True,
                has_more=False,
                delivery_id=run.final_delivery_id,
                templates=(),
            )
        assert isinstance(run, KnowledgeReconciliationRunCollecting)
        assert run.last_applied_delivery_id is not None
        return self._completed_result(
            run=run,
            binding=binding,
            checkpoint_advanced=False,
            has_more=True,
            delivery_id=run.last_applied_delivery_id,
            templates=(),
        )

    def _create_initial_run(
        self,
        *,
        run_id: str,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
        loaded_checkpoint: KnowledgeSyncCheckpoint | None,
        superseded_run: KnowledgeReconciliationRun | None = None,
    ) -> KnowledgeReconciliationRunCollecting:
        limit = self._policy.max_reconciliation_candidate_count + 1
        try:
            inventory = self._candidate_inventory.list_active_remote_ids(
                tenant_id=self._tenant_id,
                binding_id=binding.binding_id,
                binding_configuration_version=binding.configuration_version,
                limit=limit,
            )
        except KnowledgeCandidateInventoryIncomplete:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Knowledge reconciliation candidate inventory is incomplete",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation candidate inventory is corrupt",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation candidate inventory lookup failed",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None
        inventory = _validate_repository_candidate_inventory(
            inventory,
            policy=self._policy,
            source=source,
        )
        now = self._utc_clock()
        collecting = KnowledgeReconciliationRunCollecting(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            binding_configuration_version=binding.configuration_version,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
            run_id=run_id,
            record_version=1,
            created_at=now,
            updated_at=now,
            expected_base_completed_checkpoint=loaded_checkpoint,
            superseded_run_id=superseded_run.run_id
            if superseded_run is not None
            else None,
            current_input_cursor_fingerprint=knowledge_cursor_fingerprint_sha256(None),
            remaining_candidate_remote_ids=inventory,
        )
        if superseded_run is not None:
            self._cas_supersede_terminal(
                expected=superseded_run,
                replacement=collecting,
                source=source,
            )
            return collecting
        try:
            self._run_repository.create_initial_run(collecting)
        except KnowledgeReconciliationRunConflict:
            existing = self._read_run(binding_id=binding.binding_id)
            if existing is None or existing.run_id != run_id:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                    safe_message="Knowledge reconciliation run creation conflict",
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    retryable=True,
                ) from None
            return existing  # type: ignore[return-value]
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation run state is corrupt",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation run creation failed",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None
        return collecting

    def _resume_exact_sync(
        self,
        *,
        run: KnowledgeReconciliationRun,
        now: datetime,
    ) -> KnowledgeReconciliationRun:
        if not isinstance(run, KnowledgeReconciliationRunRecoveryRequired):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation resume requires recovery state",
                retryable=False,
            )
        if isinstance(
            run.recovery_evidence, KnowledgeReconciliationRecoveryEvidencePagePrepared
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation page-prepared resume requires async recovery",
                retryable=False,
            )
        replacement = self._build_resume_replacement(run=run, now=now)
        self._cas_recovery(expected=run, replacement=replacement)
        return replacement

    async def _resume_exact_async(
        self,
        command: KnowledgeReconciliationRecoveryCommand,
    ) -> KnowledgeReconciliationRun:
        run = self._read_run(binding_id=command.binding_id)
        if run is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation run was not found",
                retryable=False,
            )
        if (
            run.run_id != command.expected_run_id
            or run.record_version != command.expected_run_record_version
            or run.phase != command.expected_phase
            or run.tenant_id != command.tenant_id
            or run.binding_id != command.binding_id
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation recovery command is stale",
                retryable=False,
            )
        if not isinstance(run, KnowledgeReconciliationRunRecoveryRequired):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation resume requires recovery state",
                retryable=False,
            )
        evidence = run.recovery_evidence
        if isinstance(evidence, KnowledgeReconciliationRecoveryEvidencePagePrepared):
            binding, source = self._load_binding_and_source(binding_id=run.binding_id)
            if run.binding_configuration_version != binding.configuration_version:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                    safe_message="Knowledge reconciliation binding configuration is stale",
                    retryable=False,
                )
            prepared = self._page_prepared_from_recovery_evidence(
                run=run, evidence=evidence
            )
            await self._decide_page_prepared_recovery(
                run=prepared,
                binding=binding,
                source=source,
            )
            now = self._utc_clock()
            replacement = self._build_page_prepared_resume_replacement(
                run=run,
                evidence=evidence,
                now=now,
            )
            self._cas_recovery(expected=run, replacement=replacement)
            return replacement
        now = self._utc_clock()
        replacement = self._build_resume_replacement(run=run, now=now)
        self._cas_recovery(expected=run, replacement=replacement)
        return replacement

    def _build_resume_replacement(
        self,
        *,
        run: KnowledgeReconciliationRunRecoveryRequired,
        now: datetime,
    ) -> KnowledgeReconciliationRun:
        evidence = run.recovery_evidence
        binding, _source = self._load_binding_and_source(binding_id=run.binding_id)
        if run.binding_configuration_version != binding.configuration_version:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Knowledge reconciliation binding configuration is stale",
                retryable=False,
            )
        if isinstance(evidence, KnowledgeReconciliationRecoveryEvidenceFinalizing):
            current_checkpoint = self._read_checkpoint(binding_id=run.binding_id)
            if current_checkpoint not in {
                evidence.expected_previous_completed_checkpoint,
                evidence.intended_final_completed_checkpoint,
            }:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge reconciliation checkpoint does not match finalizing evidence",
                    retryable=False,
                )
            return KnowledgeReconciliationRunFinalizing(
                **run.model_dump(
                    exclude={
                        "phase",
                        "recovery_reason_code",
                        "machine_recovery_reason_code",
                        "recovery_evidence",
                        "record_version",
                        "updated_at",
                    }
                ),
                phase=KnowledgeReconciliationRunPhase.FINALIZING,
                intended_final_completed_checkpoint=evidence.intended_final_completed_checkpoint,
                intended_final_checkpoint_fingerprint=evidence.intended_final_checkpoint_fingerprint,
                expected_previous_completed_checkpoint=evidence.expected_previous_completed_checkpoint,
                final_delivery_id=evidence.final_delivery_id,
                prepared_batch_payload_fingerprint=evidence.prepared_batch_payload_fingerprint,
                **_advance_fields(run, updated_at=now),
            )
        current_checkpoint = self._read_checkpoint(binding_id=run.binding_id)
        if current_checkpoint != run.expected_base_completed_checkpoint:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation checkpoint does not match recovery evidence",
                retryable=False,
            )
        if isinstance(evidence, KnowledgeReconciliationRecoveryEvidenceCollecting):
            return KnowledgeReconciliationRunCollecting(
                **run.model_dump(
                    exclude={
                        "phase",
                        "recovery_reason_code",
                        "machine_recovery_reason_code",
                        "recovery_evidence",
                        "record_version",
                        "updated_at",
                    }
                ),
                phase=KnowledgeReconciliationRunPhase.COLLECTING,
                current_input_cursor=evidence.current_input_cursor,
                current_input_cursor_fingerprint=evidence.current_input_cursor_fingerprint,
                remaining_candidate_remote_ids=evidence.remaining_candidate_remote_ids,
                **_advance_fields(run, updated_at=now),
            )
        assert isinstance(evidence, KnowledgeReconciliationRecoveryEvidencePagePrepared)
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message="Knowledge reconciliation page-prepared resume requires async recovery",
            retryable=False,
        )

    def _build_page_prepared_resume_replacement(
        self,
        *,
        run: KnowledgeReconciliationRunRecoveryRequired,
        evidence: KnowledgeReconciliationRecoveryEvidencePagePrepared,
        now: datetime,
    ) -> KnowledgeReconciliationRunPagePrepared:
        return KnowledgeReconciliationRunPagePrepared(
            **run.model_dump(
                exclude={
                    "phase",
                    "recovery_reason_code",
                    "machine_recovery_reason_code",
                    "recovery_evidence",
                    "record_version",
                    "updated_at",
                }
            ),
            phase=KnowledgeReconciliationRunPhase.PAGE_PREPARED,
            prepared_input_cursor=evidence.prepared_input_cursor,
            prepared_input_cursor_fingerprint=evidence.prepared_input_cursor_fingerprint,
            provider_page_fingerprint=evidence.provider_page_fingerprint,
            prepared_batch_payload_fingerprint=evidence.prepared_batch_payload_fingerprint,
            prepared_state_mutation_templates=evidence.prepared_state_mutation_templates,
            prepared_state_mutations_fingerprint=evidence.prepared_state_mutations_fingerprint,
            prepared_proposed_checkpoint=evidence.prepared_proposed_checkpoint,
            prepared_proposed_checkpoint_fingerprint=evidence.prepared_proposed_checkpoint_fingerprint,
            prepared_next_cursor=evidence.prepared_next_cursor,
            prepared_next_cursor_fingerprint=evidence.prepared_next_cursor_fingerprint,
            prepared_page_size=evidence.prepared_page_size,
            has_more=evidence.has_more,
            delivery_id=evidence.delivery_id,
            prepared_parent_delivery_id=evidence.prepared_parent_delivery_id,
            remaining_candidate_remote_ids=evidence.remaining_candidate_remote_ids,
            synthetic_tombstone_remote_ids=evidence.synthetic_tombstone_remote_ids,
            **_advance_fields(run, updated_at=now),
        )

    def _page_prepared_from_recovery_evidence(
        self,
        *,
        run: KnowledgeReconciliationRunRecoveryRequired,
        evidence: KnowledgeReconciliationRecoveryEvidencePagePrepared,
    ) -> KnowledgeReconciliationRunPagePrepared:
        return KnowledgeReconciliationRunPagePrepared(
            tenant_id=run.tenant_id,
            binding_id=run.binding_id,
            binding_configuration_version=run.binding_configuration_version,
            provider_id=run.provider_id,
            source_kind=run.source_kind,
            run_id=run.run_id,
            record_version=run.record_version,
            created_at=run.created_at,
            updated_at=run.updated_at,
            applied_page_count=run.applied_page_count,
            last_applied_delivery_id=run.last_applied_delivery_id,
            last_applied_parent_delivery_id=run.last_applied_parent_delivery_id,
            expected_base_completed_checkpoint=run.expected_base_completed_checkpoint,
            prepared_input_cursor=evidence.prepared_input_cursor,
            prepared_input_cursor_fingerprint=evidence.prepared_input_cursor_fingerprint,
            provider_page_fingerprint=evidence.provider_page_fingerprint,
            prepared_batch_payload_fingerprint=evidence.prepared_batch_payload_fingerprint,
            prepared_state_mutation_templates=evidence.prepared_state_mutation_templates,
            prepared_state_mutations_fingerprint=evidence.prepared_state_mutations_fingerprint,
            prepared_proposed_checkpoint=evidence.prepared_proposed_checkpoint,
            prepared_proposed_checkpoint_fingerprint=evidence.prepared_proposed_checkpoint_fingerprint,
            prepared_next_cursor=evidence.prepared_next_cursor,
            prepared_next_cursor_fingerprint=evidence.prepared_next_cursor_fingerprint,
            prepared_page_size=evidence.prepared_page_size,
            has_more=evidence.has_more,
            delivery_id=evidence.delivery_id,
            prepared_parent_delivery_id=evidence.prepared_parent_delivery_id,
            remaining_candidate_remote_ids=evidence.remaining_candidate_remote_ids,
            synthetic_tombstone_remote_ids=evidence.synthetic_tombstone_remote_ids,
        )

    async def _decide_page_prepared_recovery(
        self,
        *,
        run: KnowledgeReconciliationRunPagePrepared,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
    ) -> _PagePreparedRecoveryDecision:
        item_receipt = self._inspect_item_receipt(run=run)
        if item_receipt.status is KnowledgeRemoteItemStateReceiptStatus.CONFLICT:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation item state receipt conflict",
                retryable=False,
            )
        if item_receipt.status is KnowledgeRemoteItemStateReceiptStatus.COMPLETED:
            return _PagePreparedRecoveryDecision(
                item_status=item_receipt.status,
                sink_status=None,
            )
        if item_receipt.status is KnowledgeRemoteItemStateReceiptStatus.APPLYING:
            return _PagePreparedRecoveryDecision(
                item_status=item_receipt.status,
                sink_status=None,
            )
        sink_receipt = self._inspect_sink_receipt(run=run)
        if sink_receipt.status is KnowledgeSyncSinkReceiptStatus.CONFLICT:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation sink receipt conflict",
                retryable=False,
            )
        if sink_receipt.status is KnowledgeSyncSinkReceiptStatus.UNKNOWN:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge delivery outcome requires recovery",
                retryable=False,
            )
        if sink_receipt.status is KnowledgeSyncSinkReceiptStatus.APPLIED:
            return _PagePreparedRecoveryDecision(
                item_status=item_receipt.status,
                sink_status=sink_receipt.status,
            )
        reproduced = await self._reproduce_prepared_page(
            run=run,
            binding=binding,
            source=source,
            page_size=run.prepared_page_size,
        )
        if reproduced is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation prepared page no longer matches provider",
                retryable=False,
            )
        _, _, provider_fp, batch_fp, mutations_fp = reproduced
        if (
            provider_fp != run.provider_page_fingerprint
            or batch_fp != run.prepared_batch_payload_fingerprint
            or mutations_fp != run.prepared_state_mutations_fingerprint
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation prepared page no longer matches provider",
                retryable=False,
            )
        return _PagePreparedRecoveryDecision(
            item_status=item_receipt.status,
            sink_status=sink_receipt.status,
        )

    def _abort_pristine(
        self,
        *,
        run: KnowledgeReconciliationRun,
        operator_reason_code: str,
        now: datetime,
    ) -> KnowledgeReconciliationRunAborted:
        if run.applied_page_count > 0 or run.last_applied_delivery_id is not None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation abort requires pristine run",
                retryable=False,
            )
        current = self._read_checkpoint(binding_id=run.binding_id)
        if current != run.expected_base_completed_checkpoint:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation abort checkpoint proof failed",
                retryable=False,
            )
        if isinstance(run, KnowledgeReconciliationRunPagePrepared):
            if (
                self._inspect_sink_receipt(run=run).status
                is not KnowledgeSyncSinkReceiptStatus.ABSENT
            ):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge reconciliation abort requires absent sink receipt",
                    retryable=False,
                )
            if (
                self._inspect_item_receipt(run=run).status
                is not KnowledgeRemoteItemStateReceiptStatus.ABSENT
            ):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge reconciliation abort requires absent item receipt",
                    retryable=False,
                )
        aborted = KnowledgeReconciliationRunAborted(
            **run.model_dump(
                exclude={
                    "phase",
                    "record_version",
                    "updated_at",
                    "candidate_inventory_continuation_token",
                    "current_input_cursor",
                    "current_input_cursor_fingerprint",
                    "remaining_candidate_remote_ids",
                    "prepared_input_cursor",
                    "prepared_input_cursor_fingerprint",
                    "provider_page_fingerprint",
                    "prepared_batch_payload_fingerprint",
                    "prepared_state_mutation_templates",
                    "prepared_state_mutations_fingerprint",
                    "prepared_proposed_checkpoint",
                    "prepared_proposed_checkpoint_fingerprint",
                    "prepared_next_cursor",
                    "prepared_next_cursor_fingerprint",
                    "prepared_page_size",
                    "delivery_id",
                    "has_more",
                    "synthetic_tombstone_remote_ids",
                    "prepared_parent_delivery_id",
                    "intended_final_completed_checkpoint",
                    "intended_final_checkpoint_fingerprint",
                    "expected_previous_completed_checkpoint",
                    "final_delivery_id",
                    "recovery_reason_code",
                    "recovery_evidence",
                }
            ),
            phase=KnowledgeReconciliationRunPhase.ABORTED,
            operator_reason_code=operator_reason_code,
            **_advance_fields(run, updated_at=now),
        )
        self._cas_recovery(expected=run, replacement=aborted)
        return aborted

    async def _enter_recovery(
        self,
        *,
        run: KnowledgeReconciliationRunCollecting
        | KnowledgeReconciliationRunPagePrepared
        | KnowledgeReconciliationRunFinalizing,
        reason_code: str,
        source: KnowledgeSourceRef,
        error_code: VendorKnowledgeErrorCode,
        safe_message: str,
        retryable: bool,
    ) -> KnowledgeSyncRunResult:
        now = self._utc_clock()
        exclude_fields = {
            "phase",
            "record_version",
            "updated_at",
            "candidate_inventory_continuation_token",
            "current_input_cursor",
            "current_input_cursor_fingerprint",
            "remaining_candidate_remote_ids",
            "prepared_input_cursor",
            "prepared_input_cursor_fingerprint",
            "provider_page_fingerprint",
            "prepared_batch_payload_fingerprint",
            "prepared_state_mutation_templates",
            "prepared_state_mutations_fingerprint",
            "prepared_proposed_checkpoint",
            "prepared_proposed_checkpoint_fingerprint",
            "prepared_next_cursor",
            "prepared_next_cursor_fingerprint",
            "prepared_page_size",
            "delivery_id",
            "has_more",
            "synthetic_tombstone_remote_ids",
            "prepared_parent_delivery_id",
            "intended_final_completed_checkpoint",
            "intended_final_checkpoint_fingerprint",
            "expected_previous_completed_checkpoint",
            "final_delivery_id",
            "prepared_batch_payload_fingerprint",
        }
        recovery = KnowledgeReconciliationRunRecoveryRequired(
            **run.model_dump(exclude=exclude_fields),
            phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
            recovery_reason_code=reason_code,
            machine_recovery_reason_code=reason_code,
            recovery_evidence=recovery_evidence_from_run(run),
            **_advance_fields(run, updated_at=now),
        )
        self._cas_replace(expected=run, replacement=recovery)
        raise VendorKnowledgeError(
            code=error_code,
            safe_message=safe_message,
            provider_id=source.provider_id,
            source_kind=source.source_kind,
            retryable=retryable,
        )

    def _inspect_sink_receipt_for_page_prepared(
        self, *, run: KnowledgeReconciliationRunPagePrepared
    ):
        try:
            return self._inspect_sink_receipt_raw(run=run)
        except KnowledgeSyncCorruptState:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation receipt inspection failed",
                retryable=True,
            ) from None

    def _inspect_item_receipt_for_page_prepared(
        self, *, run: KnowledgeReconciliationRunPagePrepared
    ):
        try:
            return self._inspect_item_receipt_raw(run=run)
        except KnowledgeSyncCorruptState:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation receipt inspection failed",
                retryable=True,
            ) from None

    async def _transition_page_prepared_receipt_corruption(
        self,
        *,
        run: KnowledgeReconciliationRunPagePrepared,
        source: KnowledgeSourceRef,
        reason_code: str,
    ) -> KnowledgeSyncRunResult:
        return await self._enter_recovery(
            run=run,
            reason_code=reason_code,
            source=source,
            error_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message="Knowledge reconciliation receipt state is corrupt",
            retryable=False,
        )

    def _inspect_sink_receipt(self, *, run: KnowledgeReconciliationRunPagePrepared):
        assert self._sink_receipt_inspector is not None
        try:
            return self._inspect_sink_receipt_raw(run=run)
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation sink receipt is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation receipt inspection failed",
                retryable=True,
            ) from None

    def _inspect_sink_receipt_raw(self, *, run: KnowledgeReconciliationRunPagePrepared):
        assert self._sink_receipt_inspector is not None
        return self._sink_receipt_inspector.inspect_receipt(
            tenant_id=self._tenant_id,
            binding_id=run.binding_id,
            delivery_id=run.delivery_id,
            prepared_batch_payload_fingerprint=run.prepared_batch_payload_fingerprint,
        )

    def _inspect_item_receipt(self, *, run: KnowledgeReconciliationRunPagePrepared):
        try:
            return self._inspect_item_receipt_raw(run=run)
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation item state receipt is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation receipt inspection failed",
                retryable=True,
            ) from None

    def _inspect_item_receipt_raw(self, *, run: KnowledgeReconciliationRunPagePrepared):
        return self._item_state_repository.inspect_delivery_receipt(
            tenant_id=self._tenant_id,
            binding_id=run.binding_id,
            delivery_id=run.delivery_id,
            prepared_state_mutations_fingerprint=run.prepared_state_mutations_fingerprint,
        )

    def _states_from_templates(
        self,
        *,
        run: KnowledgeReconciliationRunPagePrepared,
        binding: KnowledgeSourceBinding,
        source: KnowledgeSourceRef,
    ) -> tuple[KnowledgeRemoteItemState, ...]:
        states: list[KnowledgeRemoteItemState] = []
        for template in run.prepared_state_mutation_templates:
            states.append(
                KnowledgeRemoteItemState(
                    tenant_id=self._tenant_id,
                    binding_id=binding.binding_id,
                    binding_configuration_version=binding.configuration_version,
                    provider_id=source.provider_id,
                    source_kind=source.source_kind,
                    remote_id=template.remote_id,
                    status=template.resulting_status,
                    revision=template.revision,
                    last_delivery_id=run.delivery_id,
                )
            )
        return tuple(states)

    def _build_templates(
        self,
        *,
        envelopes: list[KnowledgeSyncEnvelope],
        binding_configuration_version: int,
    ) -> tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...]:
        templates: list[KnowledgeReconciliationPreparedStateMutationTemplate] = []
        for envelope in envelopes:
            if envelope.change_kind in _TOMBSTONE_CHANGE_KINDS:
                status = (
                    KnowledgeRemoteItemStatus.REVOKED
                    if envelope.change_kind is KnowledgeChangeKind.REVOKED
                    else KnowledgeRemoteItemStatus.DELETED
                )
                templates.append(
                    KnowledgeReconciliationPreparedStateMutationTemplate(
                        remote_id=envelope.remote_id,
                        resulting_status=status,
                        binding_configuration_version=binding_configuration_version,
                        reconciliation_semantic=envelope.reconciliation_semantic,
                    )
                )
                continue
            revision = envelope.descriptor.revision if envelope.descriptor else None
            templates.append(
                KnowledgeReconciliationPreparedStateMutationTemplate(
                    remote_id=envelope.remote_id,
                    resulting_status=KnowledgeRemoteItemStatus.ACTIVE,
                    revision=revision,
                    binding_configuration_version=binding_configuration_version,
                )
            )
        return tuple(sorted(templates, key=lambda template: template.remote_id))

    async def _apply_sink(
        self, *, batch: KnowledgeSyncBatch, source: KnowledgeSourceRef
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
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None

    def _apply_states(
        self,
        *,
        run: KnowledgeReconciliationRunPagePrepared,
        states: tuple[KnowledgeRemoteItemState, ...],
    ) -> None:
        try:
            self._item_state_repository.apply_batch(
                tenant_id=self._tenant_id,
                binding_id=run.binding_id,
                delivery_id=run.delivery_id,
                states=states,
                prepared_state_mutations_fingerprint=run.prepared_state_mutations_fingerprint,
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

    def _completed_result(
        self,
        *,
        run: KnowledgeReconciliationRun,
        binding: KnowledgeSourceBinding,
        checkpoint_advanced: bool,
        has_more: bool,
        delivery_id: str,
        templates: tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...],
    ) -> KnowledgeSyncRunResult:
        active_count = sum(
            1
            for template in templates
            if template.resulting_status is KnowledgeRemoteItemStatus.ACTIVE
        )
        tombstone_count = len(templates) - active_count
        return KnowledgeSyncRunResult(
            status=KnowledgeSyncRunStatus.COMPLETED,
            mode=KnowledgeSyncMode.RECONCILIATION,
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
            delivery_id=delivery_id,
            changes_count=len(templates),
            active_count=active_count,
            tombstone_count=tombstone_count,
            checkpoint_advanced=checkpoint_advanced,
            has_more=has_more,
            retryable=False,
        )

    def _cas_replace(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
    ) -> None:
        try:
            self._run_repository.cas_replace(expected=expected, replacement=replacement)
        except KnowledgeReconciliationRunConflict:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation run conflict",
                retryable=True,
            ) from None
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation run state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation run update failed",
                retryable=True,
            ) from None

    def _cas_recovery(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
    ) -> None:
        try:
            self._run_repository.cas_recovery(
                expected=expected, replacement=replacement
            )
        except KnowledgeReconciliationRunConflict:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation recovery conflict",
                retryable=True,
            ) from None
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation run state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation recovery update failed",
                retryable=True,
            ) from None

    def _cas_supersede_terminal(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRunCollecting,
        source: KnowledgeSourceRef,
    ) -> None:
        try:
            self._run_repository.cas_supersede_terminal(
                expected=expected,
                replacement=replacement,
            )
        except KnowledgeReconciliationRunConflict:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation run supersession conflict",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation run state is corrupt",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation run supersession failed",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None

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

    def _read_checkpoint_for_finalizing(
        self, *, binding_id: str
    ) -> KnowledgeSyncCheckpoint | None:
        try:
            checkpoint = self._checkpoint_repository.get(
                tenant_id=self._tenant_id,
                binding_id=binding_id,
            )
        except KnowledgeSyncCorruptState:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge sync checkpoint lookup failed",
                retryable=True,
            ) from None
        if checkpoint is not None and (
            checkpoint.tenant_id != self._tenant_id
            or checkpoint.binding_id != binding_id
        ):
            raise KnowledgeSyncCorruptState(
                "Knowledge sync checkpoint identity is inconsistent"
            )
        return checkpoint

    def _read_checkpoint(self, *, binding_id: str) -> KnowledgeSyncCheckpoint | None:
        try:
            checkpoint = self._checkpoint_repository.get(
                tenant_id=self._tenant_id,
                binding_id=binding_id,
            )
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge sync checkpoint state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge sync checkpoint lookup failed",
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

    def _read_run(self, *, binding_id: str) -> KnowledgeReconciliationRun | None:
        try:
            return self._run_repository.get(
                tenant_id=self._tenant_id,
                binding_id=binding_id,
            )
        except KnowledgeSyncCorruptState:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge reconciliation run state is corrupt",
                retryable=False,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Knowledge reconciliation run lookup failed",
                retryable=True,
            ) from None

    def _enforce_reconciliation_capabilities(
        self,
        *,
        input_cursor: KnowledgeCursor | None,
        capabilities: KnowledgeAdapterCapabilities,
        provider_id: str,
        source_kind: str,
    ) -> None:
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
                        "Knowledge page continuation and checkpoint response are inconsistent"
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
