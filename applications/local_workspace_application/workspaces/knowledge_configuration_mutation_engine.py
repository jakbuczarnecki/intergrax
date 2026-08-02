# © Artur Czarnecki. All rights reserved.

"""Durable Workspace Knowledge Configuration mutation engine (LKW-KNOWLEDGE-ACCESS-1B-4)."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationOutcomeV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.models import Workspace
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_MUTATION_ENGINE_MAX_CAS_ATTEMPTS = 3


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_mutation_id() -> str:
    return str(uuid.uuid4())


class WorkspaceKnowledgeConfigurationMutationError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class WorkspaceKnowledgeExistingResult:
    result_entity_type: str
    result_entity_id: str

    def __post_init__(self) -> None:
        if not self.result_entity_type or len(self.result_entity_type) > 64:
            raise ValueError("result_entity_type_invalid")
        if not self.result_entity_id or len(self.result_entity_id) > 128:
            raise ValueError("result_entity_id_invalid")


@dataclass(frozen=True, slots=True)
class WorkspaceKnowledgeStagedResult:
    result_entity_type: str
    result_entity_id: str

    def __post_init__(self) -> None:
        if not self.result_entity_type or len(self.result_entity_type) > 64:
            raise ValueError("result_entity_type_invalid")
        if not self.result_entity_id or len(self.result_entity_id) > 128:
            raise ValueError("result_entity_id_invalid")


class WorkspaceKnowledgeStageStateV1(StrEnum):
    ABSENT = "absent"
    COMPLETE_VALID = "complete_valid"
    INCOMPLETE_OWNED = "incomplete_owned"
    OWNERSHIP_CONFLICT = "ownership_conflict"


@dataclass(frozen=True, slots=True)
class WorkspaceKnowledgeStageInspection:
    state: WorkspaceKnowledgeStageStateV1
    result_entity_type: str | None = None
    result_entity_id: str | None = None

    def __post_init__(self) -> None:
        has_type = self.result_entity_type is not None
        has_id = self.result_entity_id is not None
        if has_type != has_id:
            raise ValueError("result_reference_fields_mismatched")
        if self.state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            if not has_type:
                raise ValueError("complete_valid_requires_result_reference")


class WorkspaceKnowledgeMutationExecutionDispositionV1(StrEnum):
    APPLIED = "applied"
    EXISTING_RESULT = "existing_result"
    COMMITTED_REPLAY = "committed_replay"


@dataclass(frozen=True, slots=True)
class WorkspaceKnowledgeMutationExecutionResult:
    disposition: WorkspaceKnowledgeMutationExecutionDispositionV1
    mutation: WorkspaceKnowledgeMutationRecord
    configuration_revision: int
    result_entity_type: str
    result_entity_id: str


class WorkspaceKnowledgeMutationRecoveryDispositionV1(StrEnum):
    NOTHING_TO_RECOVER = "nothing_to_recover"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class WorkspaceKnowledgeMutationRecoveryResult:
    disposition: WorkspaceKnowledgeMutationRecoveryDispositionV1
    mutation: WorkspaceKnowledgeMutationRecord | None = None


class WorkspaceKnowledgeMutationWorkspaceLookupPort(Protocol):
    def require_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> Workspace | None:
        ...


class WorkspaceKnowledgeMutationHandler(Protocol):
    operation: WorkspaceKnowledgeMutationOperationV1

    def find_existing_result(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        intent: object,
    ) -> WorkspaceKnowledgeExistingResult | None:
        ...

    def stage(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        target_revision: int,
        intent: object,
        now: datetime,
    ) -> WorkspaceKnowledgeStagedResult:
        ...

    def inspect_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeStageInspection:
        ...

    def cleanup_staged(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        mutation: WorkspaceKnowledgeMutationRecord,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> bool:
        ...


class WorkspaceKnowledgeConfigurationMutationEngine:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        workspace_lookup: WorkspaceKnowledgeMutationWorkspaceLookupPort,
        configuration_reader: WorkspaceKnowledgeConfigurationService,
        handlers: Mapping[
            WorkspaceKnowledgeMutationOperationV1,
            WorkspaceKnowledgeMutationHandler,
        ],
        *,
        clock: Callable[[], datetime] = _utc_now,
        mutation_id_factory: Callable[[], str] = _new_mutation_id,
    ) -> None:
        self._repository = repository
        self._workspace_lookup = workspace_lookup
        self._configuration_reader = configuration_reader
        self._handlers = dict(handlers)
        self._clock = clock
        self._mutation_id_factory = mutation_id_factory

    def execute(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        expected_revision: int,
        idempotency_key_hash: str,
        normalized_request_hash: str,
        semantic_identity_hash: str | None,
        intent: object,
    ) -> WorkspaceKnowledgeMutationExecutionResult:
        self._validate_execute_inputs(
            expected_revision=expected_revision,
            idempotency_key_hash=idempotency_key_hash,
            normalized_request_hash=normalized_request_hash,
            semantic_identity_hash=semantic_identity_hash,
        )
        self._require_workspace(tenant_id=tenant_id, workspace_id=workspace_id)
        handler = self._require_handler(operation)

        mutation = self._reserve_or_classify_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=operation,
            idempotency_key_hash=idempotency_key_hash,
            normalized_request_hash=normalized_request_hash,
            semantic_identity_hash=semantic_identity_hash,
            handler=handler,
        )
        if isinstance(mutation, WorkspaceKnowledgeMutationExecutionResult):
            return mutation

        return self._continue_mutation_execution(
            mutation=mutation,
            handler=handler,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision,
            intent=intent,
        )

    def recover_workspace_knowledge_mutation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeMutationRecoveryResult:
        self._require_workspace(tenant_id=tenant_id, workspace_id=workspace_id)
        head = self._repository.get_knowledge_configuration_head(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

        if head is not None and head.pending_mutation_id is not None:
            return self._recover_pending_head_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                head=head,
            )

        return self._recover_post_publication_mutations(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            head=head,
        )

    def _continue_mutation_execution(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        handler: WorkspaceKnowledgeMutationHandler,
        tenant_id: str,
        workspace_id: str,
        expected_revision: int,
        intent: object,
    ) -> WorkspaceKnowledgeMutationExecutionResult:
        now = self._clock()
        status = mutation.status

        if status is WorkspaceKnowledgeMutationStatusV1.RESERVED and mutation.target_revision is None:
            no_op = self._try_semantic_no_op(
                mutation=mutation,
                handler=handler,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                intent=intent,
                now=now,
            )
            if no_op is not None:
                return no_op

            head, target_revision = self._acquire_writer_slot_for_expected_revision(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                expected_revision=expected_revision,
                now=now,
            )
            mutation = self._assign_target_revision(
                mutation=mutation,
                target_revision=target_revision,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                now=now,
            )
        elif status is WorkspaceKnowledgeMutationStatusV1.RESERVED:
            head = self._require_pending_head_for_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                mutation=mutation,
            )
            target_revision = mutation.target_revision
            assert target_revision is not None
        elif status is WorkspaceKnowledgeMutationStatusV1.PREPARED:
            head = self._require_pending_head_for_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                mutation=mutation,
            )
            target_revision = mutation.target_revision
            assert target_revision is not None
        else:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_state_conflict"
            )

        if status is not WorkspaceKnowledgeMutationStatusV1.PREPARED:
            try:
                mutation = self._stage_and_prepare(
                    mutation=mutation,
                    handler=handler,
                    target_revision=target_revision,
                    intent=intent,
                    now=now,
                )
            except WorkspaceKnowledgeConfigurationMutationError as exc:
                self._handle_pre_publication_failure(
                    mutation=mutation,
                    handler=handler,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    error_code=exc.error_code,
                )
                raise
            except Exception:
                self._handle_pre_publication_failure(
                    mutation=mutation,
                    handler=handler,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    error_code="configuration_mutation_stage_failed",
                )
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_stage_failed"
                )

        head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
        if head is None or head.pending_mutation_id != mutation.mutation_id:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_state_conflict"
            )

        try:
            published_head = self._publish_head(
                head=head,
                mutation=mutation,
                now=now,
            )
        except WorkspaceKnowledgeConfigurationMutationError:
            raise
        except Exception:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_publication_unstable"
            )

        try:
            return self._finalize_mutation(
                mutation=mutation,
                target_revision=target_revision,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                now=now,
            )
        except WorkspaceKnowledgeConfigurationMutationError:
            repaired = self._finalize_published_mutation(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                head=published_head,
                now=now,
            )
            return WorkspaceKnowledgeMutationExecutionResult(
                disposition=WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED,
                mutation=repaired,
                configuration_revision=target_revision,
                result_entity_type=repaired.result_entity_type or "",
                result_entity_id=repaired.result_entity_id or "",
            )

    def _reserve_or_classify_mutation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
        normalized_request_hash: str,
        semantic_identity_hash: str | None,
        handler: WorkspaceKnowledgeMutationHandler,
    ) -> WorkspaceKnowledgeMutationRecord | WorkspaceKnowledgeMutationExecutionResult:
        now = self._clock()
        new_mutation = WorkspaceKnowledgeMutationRecord(
            mutation_id=self._mutation_id_factory(),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=operation,
            idempotency_key_hash=idempotency_key_hash,
            normalized_request_hash=normalized_request_hash,
            semantic_identity_hash=semantic_identity_hash,
            status=WorkspaceKnowledgeMutationStatusV1.RESERVED,
            created_at=now,
            updated_at=now,
        )

        for attempt in range(_MUTATION_ENGINE_MAX_CAS_ATTEMPTS):
            if self._repository.put_knowledge_configuration_mutation_if_absent(new_mutation):
                return new_mutation

            existing = self._reload_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=operation,
                idempotency_key_hash=idempotency_key_hash,
            )
            if existing is None:
                if attempt + 1 >= _MUTATION_ENGINE_MAX_CAS_ATTEMPTS:
                    raise WorkspaceKnowledgeConfigurationMutationError(
                        "configuration_mutation_reservation_unstable"
                    )
                continue

            classified = self._classify_existing_mutation(
                existing=existing,
                normalized_request_hash=normalized_request_hash,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                handler=handler,
            )
            if isinstance(classified, WorkspaceKnowledgeMutationExecutionResult):
                return classified
            return classified

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_mutation_reservation_unstable"
        )

    def _classify_existing_mutation(
        self,
        *,
        existing: WorkspaceKnowledgeMutationRecord,
        normalized_request_hash: str,
        tenant_id: str,
        workspace_id: str,
        handler: WorkspaceKnowledgeMutationHandler,
    ) -> WorkspaceKnowledgeMutationRecord | WorkspaceKnowledgeMutationExecutionResult:
        if existing.normalized_request_hash != normalized_request_hash:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_idempotency_conflict"
            )

        status = existing.status
        if status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
            return self._committed_replay_result(existing)

        if status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )

        if status is WorkspaceKnowledgeMutationStatusV1.ABORTED:
            restarted = self._restart_aborted_mutation(existing)
            return restarted

        if status in (
            WorkspaceKnowledgeMutationStatusV1.RESERVED,
            WorkspaceKnowledgeMutationStatusV1.PREPARED,
        ):
            self.recover_workspace_knowledge_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            return self._reload_and_classify_exact_mutation_after_recovery(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=existing.operation,
                idempotency_key_hash=existing.idempotency_key_hash,
                normalized_request_hash=normalized_request_hash,
                handler=handler,
            )

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_mutation_state_conflict"
        )

    def _restart_aborted_mutation(
        self,
        aborted: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeMutationRecord | WorkspaceKnowledgeMutationExecutionResult:
        now = self._clock()
        replacement = WorkspaceKnowledgeMutationRecord(
            mutation_id=self._mutation_id_factory(),
            tenant_id=aborted.tenant_id,
            workspace_id=aborted.workspace_id,
            operation=aborted.operation,
            idempotency_key_hash=aborted.idempotency_key_hash,
            normalized_request_hash=aborted.normalized_request_hash,
            semantic_identity_hash=aborted.semantic_identity_hash,
            status=WorkspaceKnowledgeMutationStatusV1.RESERVED,
            created_at=now,
            updated_at=now,
        )

        for attempt in range(_MUTATION_ENGINE_MAX_CAS_ATTEMPTS):
            current = self._reload_mutation(
                tenant_id=aborted.tenant_id,
                workspace_id=aborted.workspace_id,
                operation=aborted.operation,
                idempotency_key_hash=aborted.idempotency_key_hash,
            )
            if current is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_reservation_unstable"
                )
            if current.normalized_request_hash != aborted.normalized_request_hash:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_idempotency_conflict"
                )
            if current.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
                return self._committed_replay_result(current)
            if current.status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            if current.status in (
                WorkspaceKnowledgeMutationStatusV1.RESERVED,
                WorkspaceKnowledgeMutationStatusV1.PREPARED,
            ):
                return self._classify_exact_pending_mutation(
                    existing=current,
                    normalized_request_hash=aborted.normalized_request_hash,
                    tenant_id=aborted.tenant_id,
                    workspace_id=aborted.workspace_id,
                )
            if current.status is not WorkspaceKnowledgeMutationStatusV1.ABORTED:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_state_conflict"
                )

            if self._repository.replace_knowledge_configuration_mutation_if_match(
                expected=current,
                replacement=replacement,
            ):
                return replacement

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_mutation_reservation_unstable"
        )

    def _try_semantic_no_op(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        handler: WorkspaceKnowledgeMutationHandler,
        tenant_id: str,
        workspace_id: str,
        intent: object,
        now: datetime,
    ) -> WorkspaceKnowledgeMutationExecutionResult | None:
        for attempt in range(2):
            configuration = self._configuration_reader.get_configuration(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if configuration is None:
                return None

            revision_before = configuration.configuration_revision
            existing = handler.find_existing_result(
                configuration=configuration,
                intent=intent,
            )
            if existing is None:
                return None

            head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
            committed = 0 if head is None else head.committed_revision
            if committed != revision_before:
                if attempt == 1:
                    raise WorkspaceKnowledgeConfigurationMutationError(
                        "configuration_projection_unstable"
                    )
                continue

            committed_mutation = mutation.model_copy(
                update={
                    "status": WorkspaceKnowledgeMutationStatusV1.COMMITTED,
                    "outcome": WorkspaceKnowledgeMutationOutcomeV1.EXISTING_RESULT,
                    "target_revision": None,
                    "committed_revision": revision_before,
                    "result_entity_type": existing.result_entity_type,
                    "result_entity_id": existing.result_entity_id,
                    "committed_at": now,
                    "updated_at": now,
                }
            )
            if not self._repository.replace_knowledge_configuration_mutation_if_match(
                expected=mutation,
                replacement=committed_mutation,
            ):
                reloaded = self._reload_mutation(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    operation=mutation.operation,
                    idempotency_key_hash=mutation.idempotency_key_hash,
                )
                if reloaded is None:
                    raise WorkspaceKnowledgeConfigurationMutationError(
                        "configuration_mutation_reservation_unstable"
                    )
                if reloaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
                    return self._committed_replay_result(reloaded)
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_state_conflict"
                )

            return WorkspaceKnowledgeMutationExecutionResult(
                disposition=WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT,
                mutation=committed_mutation,
                configuration_revision=revision_before,
                result_entity_type=existing.result_entity_type,
                result_entity_id=existing.result_entity_id,
            )

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_projection_unstable"
        )

    def _reload_and_classify_exact_mutation_after_recovery(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
        normalized_request_hash: str,
        handler: WorkspaceKnowledgeMutationHandler,
    ) -> WorkspaceKnowledgeMutationRecord | WorkspaceKnowledgeMutationExecutionResult:
        reloaded = self._reload_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=operation,
            idempotency_key_hash=idempotency_key_hash,
        )
        if reloaded is None:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_reservation_unstable"
            )
        if reloaded.normalized_request_hash != normalized_request_hash:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_idempotency_conflict"
            )

        status = reloaded.status
        if status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
            return self._committed_replay_result(reloaded)
        if status is WorkspaceKnowledgeMutationStatusV1.ABORTED:
            return self._restart_aborted_mutation(reloaded)
        if status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )
        if status in (
            WorkspaceKnowledgeMutationStatusV1.RESERVED,
            WorkspaceKnowledgeMutationStatusV1.PREPARED,
        ):
            return self._classify_exact_pending_mutation(
                existing=reloaded,
                normalized_request_hash=normalized_request_hash,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_mutation_state_conflict"
        )

    def _classify_exact_pending_mutation(
        self,
        *,
        existing: WorkspaceKnowledgeMutationRecord,
        normalized_request_hash: str,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeMutationRecord:
        if existing.normalized_request_hash != normalized_request_hash:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_idempotency_conflict"
            )

        head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
        if existing.target_revision is not None:
            if head is None or head.pending_mutation_id != existing.mutation_id:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_state_conflict"
                )
            if head.pending_revision != existing.target_revision:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_state_conflict"
                )
            return existing

        if head is not None and head.pending_mutation_id is not None:
            if head.pending_mutation_id != existing.mutation_id:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            return existing

        return existing

    def _abort_mutation_for_revision_conflict(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
        now: datetime,
    ) -> None:
        if mutation.target_revision is not None:
            return

        current = self._reload_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=operation,
            idempotency_key_hash=idempotency_key_hash,
        )
        if current is None or current.status is not WorkspaceKnowledgeMutationStatusV1.RESERVED:
            return
        if current.target_revision is not None:
            return

        aborted = current.model_copy(
            update={
                "status": WorkspaceKnowledgeMutationStatusV1.ABORTED,
                "error_code": "configuration_revision_conflict",
                "updated_at": now,
            }
        )
        self._repository.replace_knowledge_configuration_mutation_if_match(
            expected=current,
            replacement=aborted,
        )

    def _acquire_writer_slot_for_expected_revision(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        tenant_id: str,
        workspace_id: str,
        expected_revision: int,
        now: datetime,
    ) -> tuple[WorkspaceKnowledgeConfigurationHead, int]:
        for attempt in range(_MUTATION_ENGINE_MAX_CAS_ATTEMPTS):
            head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
            if head is None:
                idle_head = WorkspaceKnowledgeConfigurationHead(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    committed_revision=0,
                    updated_at=now,
                )
                self._repository.put_knowledge_configuration_head_if_absent(idle_head)
                head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
                if head is None:
                    if attempt + 1 >= _MUTATION_ENGINE_MAX_CAS_ATTEMPTS:
                        raise WorkspaceKnowledgeConfigurationMutationError(
                            "configuration_mutation_reservation_unstable"
                        )
                    continue

            if head.committed_revision != expected_revision:
                self._abort_mutation_for_revision_conflict(
                    mutation=mutation,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    operation=mutation.operation,
                    idempotency_key_hash=mutation.idempotency_key_hash,
                    now=now,
                )
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_revision_conflict"
                )

            if head.pending_mutation_id is not None:
                if head.pending_mutation_id == mutation.mutation_id:
                    target_revision = head.pending_revision
                    assert target_revision is not None
                    if target_revision != expected_revision + 1:
                        raise WorkspaceKnowledgeConfigurationMutationError(
                            "configuration_mutation_state_conflict"
                        )
                    return head, target_revision
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

            target_revision = expected_revision + 1
            pending_head = head.model_copy(
                update={
                    "pending_revision": target_revision,
                    "pending_mutation_id": mutation.mutation_id,
                    "updated_at": now,
                }
            )
            if self._repository.replace_knowledge_configuration_head_if_match(
                expected=head,
                replacement=pending_head,
            ):
                return pending_head, target_revision

            refreshed = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
            if refreshed is None:
                continue
            if refreshed.committed_revision != expected_revision:
                self._abort_mutation_for_revision_conflict(
                    mutation=mutation,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    operation=mutation.operation,
                    idempotency_key_hash=mutation.idempotency_key_hash,
                    now=now,
                )
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_revision_conflict"
                )
            if refreshed.pending_mutation_id is not None:
                if (
                    refreshed.pending_mutation_id == mutation.mutation_id
                    and refreshed.pending_revision == expected_revision + 1
                ):
                    return refreshed, expected_revision + 1
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_mutation_reservation_unstable"
        )

    def _assign_target_revision(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        target_revision: int,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
        now: datetime,
    ) -> WorkspaceKnowledgeMutationRecord:
        if mutation.target_revision == target_revision:
            return mutation

        assigned = mutation.model_copy(
            update={
                "target_revision": target_revision,
                "updated_at": now,
            }
        )
        if self._repository.replace_knowledge_configuration_mutation_if_match(
            expected=mutation,
            replacement=assigned,
        ):
            return assigned

        reloaded = self._reload_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=operation,
            idempotency_key_hash=idempotency_key_hash,
        )
        if reloaded is None:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_reservation_unstable"
            )
        if reloaded.target_revision == target_revision:
            return reloaded
        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_recovery_required"
        )

    def _stage_and_prepare(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        handler: WorkspaceKnowledgeMutationHandler,
        target_revision: int,
        intent: object,
        now: datetime,
    ) -> WorkspaceKnowledgeMutationRecord:
        staged = handler.stage(
            repository=self._repository,
            mutation=mutation,
            target_revision=target_revision,
            intent=intent,
            now=now,
        )
        inspection = handler.inspect_staged(
            repository=self._repository,
            mutation=mutation,
        )
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_stage_failed"
            )
        if (
            inspection.result_entity_type != staged.result_entity_type
            or inspection.result_entity_id != staged.result_entity_id
        ):
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_stage_failed"
            )

        prepared = mutation.model_copy(
            update={
                "status": WorkspaceKnowledgeMutationStatusV1.PREPARED,
                "result_entity_type": staged.result_entity_type,
                "result_entity_id": staged.result_entity_id,
                "updated_at": now,
            }
        )
        if not self._repository.replace_knowledge_configuration_mutation_if_match(
            expected=mutation,
            replacement=prepared,
        ):
            reloaded = self._reload_mutation(
                tenant_id=mutation.tenant_id,
                workspace_id=mutation.workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
            )
            if reloaded is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_reservation_unstable"
                )
            if reloaded.status is WorkspaceKnowledgeMutationStatusV1.PREPARED:
                return reloaded
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_state_conflict"
            )
        return prepared

    def _publish_head(
        self,
        *,
        head: WorkspaceKnowledgeConfigurationHead,
        mutation: WorkspaceKnowledgeMutationRecord,
        now: datetime,
    ) -> WorkspaceKnowledgeConfigurationHead:
        for attempt in range(_MUTATION_ENGINE_MAX_CAS_ATTEMPTS):
            if head.pending_mutation_id != mutation.mutation_id:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_state_conflict"
                )
            target_revision = head.pending_revision
            assert target_revision is not None
            published = head.model_copy(
                update={
                    "committed_revision": target_revision,
                    "pending_revision": None,
                    "pending_mutation_id": None,
                    "last_committed_mutation_id": mutation.mutation_id,
                    "updated_at": now,
                }
            )
            if self._repository.replace_knowledge_configuration_head_if_match(
                expected=head,
                replacement=published,
            ):
                return published

            head = self._reload_head(
                tenant_id=head.tenant_id,
                workspace_id=head.workspace_id,
            )
            if head is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_mutation_publication_unstable"
                )
            if (
                head.committed_revision >= target_revision
                and head.pending_mutation_id is None
            ):
                return head

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_mutation_publication_unstable"
        )

    def _finalize_mutation(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        target_revision: int,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
        now: datetime,
    ) -> WorkspaceKnowledgeMutationExecutionResult:
        committed = mutation.model_copy(
            update={
                "status": WorkspaceKnowledgeMutationStatusV1.COMMITTED,
                "outcome": WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
                "committed_revision": target_revision,
                "committed_at": now,
                "updated_at": now,
            }
        )
        if not self._repository.replace_knowledge_configuration_mutation_if_match(
            expected=mutation,
            replacement=committed,
        ):
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_state_conflict"
            )
        return WorkspaceKnowledgeMutationExecutionResult(
            disposition=WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED,
            mutation=committed,
            configuration_revision=target_revision,
            result_entity_type=committed.result_entity_type or "",
            result_entity_id=committed.result_entity_id or "",
        )

    def _finalize_published_mutation(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        tenant_id: str,
        workspace_id: str,
        head: WorkspaceKnowledgeConfigurationHead,
        now: datetime,
    ) -> WorkspaceKnowledgeMutationRecord:
        target_revision = mutation.target_revision
        if target_revision is None:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )
        if head.committed_revision < target_revision:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )
        handler = self._require_handler(mutation.operation)
        inspection = handler.inspect_staged(
            repository=self._repository,
            mutation=mutation,
        )
        if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )
        committed = mutation.model_copy(
            update={
                "status": WorkspaceKnowledgeMutationStatusV1.COMMITTED,
                "outcome": WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
                "committed_revision": target_revision,
                "result_entity_type": inspection.result_entity_type,
                "result_entity_id": inspection.result_entity_id,
                "committed_at": now,
                "updated_at": now,
            }
        )
        if not self._repository.replace_knowledge_configuration_mutation_if_match(
            expected=mutation,
            replacement=committed,
        ):
            reloaded = self._reload_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
            )
            if reloaded is None or reloaded.status is not WorkspaceKnowledgeMutationStatusV1.COMMITTED:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            return reloaded
        return committed

    def _handle_pre_publication_failure(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        handler: WorkspaceKnowledgeMutationHandler,
        tenant_id: str,
        workspace_id: str,
        error_code: str,
    ) -> None:
        now = self._clock()
        inspection = handler.inspect_staged(
            repository=self._repository,
            mutation=mutation,
        )

        if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            self._mark_recovery_required(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code="configuration_recovery_required",
                now=now,
            )
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )

        if not handler.cleanup_staged(
            repository=self._repository,
            mutation=mutation,
            inspection=inspection,
        ):
            self._mark_recovery_required(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code="configuration_mutation_cleanup_failed",
                now=now,
            )
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_cleanup_failed"
            )

        post_cleanup = handler.inspect_staged(
            repository=self._repository,
            mutation=mutation,
        )
        if post_cleanup.state is not WorkspaceKnowledgeStageStateV1.ABSENT:
            self._mark_recovery_required(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code="configuration_mutation_cleanup_failed",
                now=now,
            )
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_cleanup_failed"
            )

        head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
        if head is not None and head.pending_mutation_id == mutation.mutation_id:
            try:
                self._release_pending_head_for_mutation(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    mutation=mutation,
                    now=now,
                )
            except WorkspaceKnowledgeConfigurationMutationError:
                self._mark_recovery_required(
                    mutation=mutation,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    operation=mutation.operation,
                    idempotency_key_hash=mutation.idempotency_key_hash,
                    error_code="configuration_mutation_cleanup_failed",
                    now=now,
                )
                raise

        try:
            self._confirm_mutation_aborted(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code=error_code,
                now=now,
            )
        except WorkspaceKnowledgeConfigurationMutationError:
            self._mark_recovery_required(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code="configuration_mutation_cleanup_failed",
                now=now,
            )
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_cleanup_failed"
            )

    def _recover_pending_head_mutation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        head: WorkspaceKnowledgeConfigurationHead,
    ) -> WorkspaceKnowledgeMutationRecoveryResult:
        pending_mutation_id = head.pending_mutation_id
        assert pending_mutation_id is not None
        pending_revision = head.pending_revision
        assert pending_revision is not None

        mutation = self._repository.find_knowledge_configuration_mutation_by_id(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            mutation_id=pending_mutation_id,
        )
        if mutation is None:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )
        if (
            mutation.target_revision != pending_revision
            or mutation.tenant_id != tenant_id
            or mutation.workspace_id != workspace_id
        ):
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )

        handler = self._require_handler(mutation.operation)
        inspection = handler.inspect_staged(
            repository=self._repository,
            mutation=mutation,
        )

        if inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT:
            self._mark_recovery_required(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code="configuration_recovery_required",
                now=self._clock(),
            )
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )

        if inspection.state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
            now = self._clock()
            working = mutation
            if working.status is WorkspaceKnowledgeMutationStatusV1.RESERVED:
                prepared = working.model_copy(
                    update={
                        "status": WorkspaceKnowledgeMutationStatusV1.PREPARED,
                        "result_entity_type": inspection.result_entity_type,
                        "result_entity_id": inspection.result_entity_id,
                        "updated_at": now,
                    }
                )
                if not self._repository.replace_knowledge_configuration_mutation_if_match(
                    expected=working,
                    replacement=prepared,
                ):
                    reloaded = self._reload_mutation(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        operation=mutation.operation,
                        idempotency_key_hash=mutation.idempotency_key_hash,
                    )
                    if reloaded is None or reloaded.status is not WorkspaceKnowledgeMutationStatusV1.PREPARED:
                        raise WorkspaceKnowledgeConfigurationMutationError(
                            "configuration_recovery_required"
                        )
                    working = reloaded
                else:
                    working = prepared

            current_head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
            if current_head is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            published_head = self._publish_head(
                head=current_head,
                mutation=working,
                now=now,
            )
            finalized = self._finalize_published_mutation(
                mutation=working,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                head=published_head,
                now=now,
            )
            return WorkspaceKnowledgeMutationRecoveryResult(
                disposition=WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED,
                mutation=finalized,
            )

        if inspection.state in (
            WorkspaceKnowledgeStageStateV1.ABSENT,
            WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED,
        ):
            return self._abort_incomplete_pending_mutation(
                mutation=mutation,
                handler=handler,
                head=head,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                inspection=inspection,
            )

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_recovery_required"
        )

    def _abort_incomplete_pending_mutation(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        handler: WorkspaceKnowledgeMutationHandler,
        head: WorkspaceKnowledgeConfigurationHead,
        tenant_id: str,
        workspace_id: str,
        inspection: WorkspaceKnowledgeStageInspection,
    ) -> WorkspaceKnowledgeMutationRecoveryResult:
        now = self._clock()
        if inspection.state is not WorkspaceKnowledgeStageStateV1.ABSENT:
            if not handler.cleanup_staged(
                repository=self._repository,
                mutation=mutation,
                inspection=inspection,
            ):
                self._mark_recovery_required(
                    mutation=mutation,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    operation=mutation.operation,
                    idempotency_key_hash=mutation.idempotency_key_hash,
                    error_code="configuration_mutation_cleanup_failed",
                    now=now,
                )
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            post_cleanup = handler.inspect_staged(
                repository=self._repository,
                mutation=mutation,
            )
            if post_cleanup.state is not WorkspaceKnowledgeStageStateV1.ABSENT:
                self._mark_recovery_required(
                    mutation=mutation,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    operation=mutation.operation,
                    idempotency_key_hash=mutation.idempotency_key_hash,
                    error_code="configuration_mutation_cleanup_failed",
                    now=now,
                )
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

        try:
            self._release_pending_head_for_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                mutation=mutation,
                now=now,
            )
        except WorkspaceKnowledgeConfigurationMutationError as exc:
            self._mark_recovery_required(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code=exc.error_code,
                now=now,
            )
            raise

        try:
            aborted = self._confirm_mutation_aborted(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code="configuration_mutation_stage_failed",
                now=now,
            )
        except WorkspaceKnowledgeConfigurationMutationError:
            self._mark_recovery_required(
                mutation=mutation,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=mutation.operation,
                idempotency_key_hash=mutation.idempotency_key_hash,
                error_code="configuration_mutation_cleanup_failed",
                now=now,
            )
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_recovery_required"
            )
        return WorkspaceKnowledgeMutationRecoveryResult(
            disposition=WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED,
            mutation=aborted,
        )

    def _recover_post_publication_mutations(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        head: WorkspaceKnowledgeConfigurationHead | None,
    ) -> WorkspaceKnowledgeMutationRecoveryResult:
        if head is None:
            return WorkspaceKnowledgeMutationRecoveryResult(
                disposition=WorkspaceKnowledgeMutationRecoveryDispositionV1.NOTHING_TO_RECOVER,
            )

        committed_revision = head.committed_revision
        candidates: list[WorkspaceKnowledgeMutationRecord] = []
        for mutation in self._repository.list_knowledge_configuration_mutations(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            if mutation.status not in (
                WorkspaceKnowledgeMutationStatusV1.RESERVED,
                WorkspaceKnowledgeMutationStatusV1.PREPARED,
                WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
            ):
                continue
            if mutation.target_revision is None:
                continue
            if mutation.target_revision > committed_revision:
                continue
            candidates.append(mutation)

        if not candidates:
            return WorkspaceKnowledgeMutationRecoveryResult(
                disposition=WorkspaceKnowledgeMutationRecoveryDispositionV1.NOTHING_TO_RECOVER,
            )

        by_revision: dict[int, list[WorkspaceKnowledgeMutationRecord]] = {}
        for candidate in candidates:
            assert candidate.target_revision is not None
            by_revision.setdefault(candidate.target_revision, []).append(candidate)

        for target_revision, group in by_revision.items():
            if len(group) > 1:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

        now = self._clock()
        finalized: WorkspaceKnowledgeMutationRecord | None = None
        for candidate in candidates:
            handler = self._require_handler(candidate.operation)
            inspection = handler.inspect_staged(
                repository=self._repository,
                mutation=candidate,
            )
            if inspection.state is not WorkspaceKnowledgeStageStateV1.COMPLETE_VALID:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            assert candidate.target_revision is not None
            repaired = self._finalize_published_mutation(
                mutation=candidate,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                head=head,
                now=now,
            )
            finalized = repaired

        if finalized is None:
            return WorkspaceKnowledgeMutationRecoveryResult(
                disposition=WorkspaceKnowledgeMutationRecoveryDispositionV1.NOTHING_TO_RECOVER,
            )
        return WorkspaceKnowledgeMutationRecoveryResult(
            disposition=WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED,
            mutation=finalized,
        )

    def _release_pending_head_for_mutation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        mutation: WorkspaceKnowledgeMutationRecord,
        now: datetime,
    ) -> WorkspaceKnowledgeConfigurationHead:
        target_revision = mutation.target_revision
        for attempt in range(_MUTATION_ENGINE_MAX_CAS_ATTEMPTS):
            head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
            if head is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

            if head.pending_mutation_id is None:
                if (
                    target_revision is not None
                    and head.committed_revision >= target_revision
                ):
                    raise WorkspaceKnowledgeConfigurationMutationError(
                        "configuration_recovery_required"
                    )
                return head

            if head.pending_mutation_id != mutation.mutation_id:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

            if (
                target_revision is not None
                and head.committed_revision >= target_revision
            ):
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

            idle_head = head.model_copy(
                update={
                    "pending_revision": None,
                    "pending_mutation_id": None,
                    "updated_at": now,
                }
            )
            if self._repository.replace_knowledge_configuration_head_if_match(
                expected=head,
                replacement=idle_head,
            ):
                return idle_head

            refreshed = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
            if refreshed is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            if refreshed.pending_mutation_id is None:
                if (
                    target_revision is not None
                    and refreshed.committed_revision >= target_revision
                ):
                    raise WorkspaceKnowledgeConfigurationMutationError(
                        "configuration_recovery_required"
                    )
                return refreshed
            if refreshed.pending_mutation_id != mutation.mutation_id:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            if (
                target_revision is not None
                and refreshed.committed_revision >= target_revision
            ):
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_mutation_cleanup_failed"
        )

    def _confirm_mutation_aborted(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
        error_code: str,
        now: datetime,
    ) -> WorkspaceKnowledgeMutationRecord:
        for attempt in range(_MUTATION_ENGINE_MAX_CAS_ATTEMPTS):
            current = self._reload_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=operation,
                idempotency_key_hash=idempotency_key_hash,
            )
            if current is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            if current.status is WorkspaceKnowledgeMutationStatusV1.ABORTED:
                return current
            if current.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

            aborted = current.model_copy(
                update={
                    "status": WorkspaceKnowledgeMutationStatusV1.ABORTED,
                    "error_code": error_code,
                    "updated_at": now,
                }
            )
            if self._repository.replace_knowledge_configuration_mutation_if_match(
                expected=current,
                replacement=aborted,
            ):
                return aborted

            reloaded = self._reload_mutation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation=operation,
                idempotency_key_hash=idempotency_key_hash,
            )
            if reloaded is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )
            if reloaded.status is WorkspaceKnowledgeMutationStatusV1.ABORTED:
                return reloaded
            if reloaded.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "configuration_recovery_required"
                )

        raise WorkspaceKnowledgeConfigurationMutationError(
            "configuration_recovery_required"
        )

    def _mark_recovery_required(
        self,
        *,
        mutation: WorkspaceKnowledgeMutationRecord,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
        error_code: str,
        now: datetime,
    ) -> None:
        current = self._reload_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=operation,
            idempotency_key_hash=idempotency_key_hash,
        )
        if current is None:
            return
        if current.status in (
            WorkspaceKnowledgeMutationStatusV1.COMMITTED,
            WorkspaceKnowledgeMutationStatusV1.ABORTED,
        ):
            return
        required = current.model_copy(
            update={
                "status": WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED,
                "error_code": error_code,
                "updated_at": now,
            }
        )
        self._repository.replace_knowledge_configuration_mutation_if_match(
            expected=current,
            replacement=required,
        )

    def _committed_replay_result(
        self,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeMutationExecutionResult:
        return WorkspaceKnowledgeMutationExecutionResult(
            disposition=WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY,
            mutation=mutation,
            configuration_revision=mutation.committed_revision or 0,
            result_entity_type=mutation.result_entity_type or "",
            result_entity_id=mutation.result_entity_id or "",
        )

    def _require_pending_head_for_mutation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> WorkspaceKnowledgeConfigurationHead:
        head = self._reload_head(tenant_id=tenant_id, workspace_id=workspace_id)
        if head is None or head.pending_mutation_id != mutation.mutation_id:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "configuration_mutation_state_conflict"
            )
        return head

    def _reload_head(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationHead | None:
        return self._repository.get_knowledge_configuration_head(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

    def _reload_mutation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
    ) -> WorkspaceKnowledgeMutationRecord | None:
        return self._repository.get_knowledge_configuration_mutation(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            operation=operation,
            idempotency_key_hash=idempotency_key_hash,
        )

    def _require_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace:
        workspace = self._workspace_lookup.require_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise WorkspaceKnowledgeConfigurationMutationError("workspace_not_found")
        return workspace

    def _require_handler(
        self,
        operation: WorkspaceKnowledgeMutationOperationV1,
    ) -> WorkspaceKnowledgeMutationHandler:
        handler = self._handlers.get(operation)
        if handler is None:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "knowledge_configuration_handler_not_registered"
            )
        return handler

    def _validate_execute_inputs(
        self,
        *,
        expected_revision: int,
        idempotency_key_hash: str,
        normalized_request_hash: str,
        semantic_identity_hash: str | None,
    ) -> None:
        if not isinstance(expected_revision, int) or expected_revision < 0:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "knowledge_configuration_expected_revision_invalid"
            )
        if _SHA256_HEX_RE.fullmatch(idempotency_key_hash) is None:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "knowledge_configuration_idempotency_hash_invalid"
            )
        if _SHA256_HEX_RE.fullmatch(normalized_request_hash) is None:
            raise WorkspaceKnowledgeConfigurationMutationError(
                "knowledge_configuration_request_hash_invalid"
            )
        if semantic_identity_hash is not None:
            if _SHA256_HEX_RE.fullmatch(semantic_identity_hash) is None:
                raise WorkspaceKnowledgeConfigurationMutationError(
                    "knowledge_configuration_semantic_identity_hash_invalid"
                )
