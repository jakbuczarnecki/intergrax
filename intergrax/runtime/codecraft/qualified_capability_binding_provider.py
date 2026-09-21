# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production CodeCraft qualified capability binding (UCA-6C-R)."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock

from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingReasonCode,
    QualifiedCapabilityBindingRequest,
    QualifiedCapabilityBindingResult,
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    QualifiedCapabilitySubjectKind,
)
from intergrax.runtime.codecraft.artifact_reference import (
    execution_target_reference_for_craft,
    parse_codecraft_artifact_reference,
)
from intergrax.runtime.codecraft.ephemeral_registry import get_ephemeral_registry_store
from intergrax.runtime.codecraft.ownership import (
    CodeCraftOwnershipError,
    CodeCraftSessionOwnership,
    matches_session_ownership,
)
from intergrax.runtime.codecraft.session_manager import (
    CodeCraftSessionManager,
    get_session_manager,
)
from intergrax.tools.registry.wiring import ToolWiringContext

CODECRAFT_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID = (
    "codecraft.qualified_capability_binding.v1"
)


class CodeCraftQualifiedCapabilityBindingProvider:
    """Resolve canonical CodeCraft artifacts to opaque execution targets — no execution."""

    def __init__(
        self,
        wiring_context: ToolWiringContext,
        *,
        session_manager: CodeCraftSessionManager | None = None,
    ) -> None:
        self._ctx = wiring_context
        self._sessions = session_manager or get_session_manager(wiring_context)
        self._cache: dict[str, QualifiedCapabilityBindingResult] = {}
        self._lock = RLock()

    @property
    def provider_id(self) -> str:
        return CODECRAFT_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID

    def supports(self, request: QualifiedCapabilityBindingRequest) -> bool:
        subject = request.qualified_subject
        if (
            subject.subject_kind
            is not QualifiedCapabilitySubjectKind.ARTIFACT_REFERENCE
        ):
            return False
        return parse_codecraft_artifact_reference(subject.subject_reference) is not None

    def bind(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingResult:
        started_at = datetime.now(tz=UTC)
        with self._lock:
            cached = self._cache.get(request.binding_operation_id)
        if cached is not None:
            return cached

        craft_id = parse_codecraft_artifact_reference(
            request.qualified_subject.subject_reference,
        )
        if craft_id is None:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.NOT_SUPPORTED,
                reason_code=QualifiedCapabilityBindingReasonCode.SUBJECT_NOT_SUPPORTED,
                started_at=started_at,
            )

        ownership = CodeCraftSessionOwnership(
            tenant_id=request.tenant_id,
            task_id=str(request.task_id),
        )
        try:
            session = self._sessions.get_owned(craft_id, ownership)
        except CodeCraftOwnershipError as exc:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.BLOCKED,
                reason_code=QualifiedCapabilityBindingReasonCode.POLICY_BLOCKED,
                started_at=started_at,
                reason_detail=exc.code,
            )

        if session is None or session.disposed:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.UNAVAILABLE,
                reason_code=QualifiedCapabilityBindingReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
                reason_detail="codecraft_artifact_unavailable",
            )

        if not matches_session_ownership(
            session.tenant_id,
            session.task_id,
            session.run_id,
            ownership,
        ):
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.BLOCKED,
                reason_code=QualifiedCapabilityBindingReasonCode.POLICY_BLOCKED,
                started_at=started_at,
                reason_detail="craft_session_ownership_mismatch",
            )

        tools = get_ephemeral_registry_store(self._ctx).for_craft(craft_id).list_tools()
        if not tools:
            return _terminal(
                request=request,
                outcome=QualifiedCapabilityBindingOutcome.UNAVAILABLE,
                reason_code=QualifiedCapabilityBindingReasonCode.PROVIDER_UNAVAILABLE,
                started_at=started_at,
                reason_detail="codecraft_ephemeral_artifact_empty",
            )

        completed_at = datetime.now(tz=UTC)
        target_ref = execution_target_reference_for_craft(craft_id)
        target = QualifiedCapabilityExecutionTarget(
            execution_target_reference=target_ref,
            binding_provider_id=self.provider_id,
            qualified_subject_reference=request.qualified_subject.qualified_subject_reference,
        )
        result = QualifiedCapabilityBindingResult(
            binding_operation_id=request.binding_operation_id,
            outcome=QualifiedCapabilityBindingOutcome.BOUND,
            reason_code=QualifiedCapabilityBindingReasonCode.NONE,
            provider_id=self.provider_id,
            execution_target=target,
            started_at=started_at,
            completed_at=completed_at,
        )
        with self._lock:
            self._cache[request.binding_operation_id] = result
        return result


def _terminal(
    *,
    request: QualifiedCapabilityBindingRequest,
    outcome: QualifiedCapabilityBindingOutcome,
    reason_code: QualifiedCapabilityBindingReasonCode,
    started_at: datetime,
    reason_detail: str = "",
) -> QualifiedCapabilityBindingResult:
    completed_at = datetime.now(tz=UTC)
    return QualifiedCapabilityBindingResult(
        binding_operation_id=request.binding_operation_id,
        outcome=outcome,
        reason_code=reason_code,
        started_at=started_at,
        completed_at=completed_at,
        reason_detail=reason_detail,
    )


__all__ = [
    "CODECRAFT_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID",
    "CodeCraftQualifiedCapabilityBindingProvider",
]
