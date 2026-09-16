# © Artur Czarnecki. All rights reserved.

"""Default Memory Control Plane — routes to injected capabilities (MEM-ENT-3)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_control import (
    EpisodicMemoryCapability,
    MemoryControlAccessDenied,
    MemoryControlGovernanceDenied,
    MemoryControlBackendError,
    MemoryControlForgetRequest,
    MemoryControlForgetResult,
    MemoryControlNotFound,
    MemoryControlPartialLifecycleError,
    MemoryControlPlaneScope,
    MemoryControlRecallItem,
    MemoryControlRecallRequest,
    MemoryControlRecallResult,
    MemoryControlReconcileRequest,
    MemoryControlReconcileResult,
    MemoryControlRememberRequest,
    MemoryControlRememberResult,
    MemoryControlScopeRef,
    MemoryControlSupersessionApplyResult,
    MemoryControlUnsupportedScope,
    TaskMemoryCapability,
    UserMemoryForgetCapabilityResult,
    UserMemoryRecallCapabilityResult,
    UserMemoryRememberCapabilityResult,
    UserProfileMemoryCapability,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryReconciliationDisposition,
    MemoryReconciliationOutcome,
)
from intergrax.memory.memory_temporal import is_memory_entry_active
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.memory.recall.pipeline import run_recall_decision_pipeline
from intergrax.memory.recall.retrieval import (
    UserMemoryRecallRetrievalConfig,
    candidates_from_profile_scan,
    candidates_from_semantic_search,
    semantic_retrieval_top_k,
)
from intergrax.memory.recall_strategy_bundle import (
    MemoryRecallStrategySet,
    build_default_memory_recall_strategies,
)
from intergrax.memory.strategies.errors import MemoryStrategyError
from intergrax.memory.contracts.memory_recall import MemoryRecallReasonCode, MemorySupersessionIntent
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory_lifecycle import UserProfileMemoryLifecyclePartialError
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceRecordSnapshot,
    MemoryGovernanceTarget,
    MemorySecurityContext,
)
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticFailureClass,
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
)
from intergrax.memory.memory_diagnostic_emitter import (
    MemoryDiagnosticEmitter,
    MemoryOperationTimer,
    default_memory_diagnostic_emitter,
)
from intergrax.memory.memory_observability_support import emit_control_plane_terminal
from intergrax.utils.time_provider import SystemTimeProvider, TimeProvider

__all__ = ["DefaultMemoryControlPlane", "UserProfileManagerMemoryCapability"]


def _governance_service_or_default(
    service: MemorySecurityGovernanceService | None,
) -> MemorySecurityGovernanceService:
    return service if service is not None else build_default_memory_security_governance_service()


def _security_context(
    identity: RequestIdentity,
    scope: MemoryControlScopeRef,
    operation: MemoryGovernanceOperation,
) -> MemorySecurityContext:
    return MemorySecurityContext(
        identity=identity,
        scope=scope,
        operation=operation,
        reference_time=None,
    )


def _enforce_governance(
    service: MemorySecurityGovernanceService | None,
    request: MemoryGovernanceEvaluationRequest,
    *,
    diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
    control_plane_operation: MemoryDiagnosticOperation | None = None,
    identity: RequestIdentity | None = None,
    scope: MemoryControlScopeRef | None = None,
) -> None:
    decision = _governance_service_or_default(service).evaluate(request)
    if not (
        decision.permits_mutation()
        if request.context.operation in _GOVERNANCE_MUTATION_OPERATIONS
        else decision.permits_disclosure()
    ):
        if (
            diagnostic_emitter is not None
            and control_plane_operation is not None
            and identity is not None
            and scope is not None
        ):
            emit_control_plane_terminal(
                diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=control_plane_operation,
                outcome=MemoryDiagnosticOutcome.DENIED,
                memory_id=decision.subject_memory_id,
                failure_class=MemoryDiagnosticFailureClass.POLICY,
            )
        raise MemoryControlGovernanceDenied(
            f"memory governance denied: {decision.reason_code.value}",
            decision=decision,
        )


_GOVERNANCE_MUTATION_OPERATIONS = frozenset(
    {
        MemoryGovernanceOperation.REMEMBER,
        MemoryGovernanceOperation.PROMOTE,
        MemoryGovernanceOperation.SUPERSEDE,
        MemoryGovernanceOperation.DELETE,
        MemoryGovernanceOperation.COMPACT,
        MemoryGovernanceOperation.PROJECT,
        MemoryGovernanceOperation.UPDATE,
    }
)


def _assert_scope_authorized(
    identity: RequestIdentity,
    scope: MemoryControlScopeRef,
) -> None:
    if scope.tenant_id != identity.tenant_id:
        raise MemoryControlAccessDenied("scope tenant_id conflicts with canonical identity")
    if scope.kind is MemoryControlPlaneScope.USER:
        canonical_user = (identity.user_id or "").strip()
        scope_user = (scope.user_id or "").strip()
        if not scope_user or scope_user != canonical_user:
            raise MemoryControlAccessDenied("user memory scope conflicts with canonical user_id")
    if scope.kind is MemoryControlPlaneScope.SESSION:
        if not (scope.session_id or "").strip():
            raise MemoryControlAccessDenied("session scope requires session_id")


def _map_capability_mutation_error(exc: BaseException) -> BaseException:
    if isinstance(exc, MemoryControlPartialLifecycleError):
        return exc
    if isinstance(exc, UserProfileMemoryLifecyclePartialError):
        return MemoryControlPartialLifecycleError(exc.outcome, cause=exc)
    return MemoryControlBackendError(str(exc))


def adapt_manager_search_result(raw: dict[str, Any]) -> UserMemoryRecallCapabilityResult:
    """Translate legacy manager search dict into a typed capability result."""
    if not isinstance(raw, dict):
        raise MemoryControlBackendError("unexpected search result shape")
    hits_raw = raw.get("hits")
    scores_raw = raw.get("scores")
    debug_raw = raw.get("debug")
    if hits_raw is None or scores_raw is None:
        raise MemoryControlBackendError("search result missing hits or scores")
    if not isinstance(hits_raw, list) or not isinstance(scores_raw, list):
        raise MemoryControlBackendError("search result hits/scores must be lists")
    entries: list[UserProfileMemoryEntry] = []
    for hit in hits_raw:
        if not isinstance(hit, UserProfileMemoryEntry):
            raise MemoryControlBackendError("search hit is not a memory entry")
        entries.append(hit)
    if len(entries) != len(scores_raw):
        raise MemoryControlBackendError("search hits and scores length mismatch")
    scores: list[float | None] = []
    for score in scores_raw:
        if score is None:
            scores.append(None)
        else:
            scores.append(float(score))
    used = False
    reason = "semantic"
    if isinstance(debug_raw, dict):
        used = bool(debug_raw.get("used"))
        reason = str(debug_raw.get("reason") or reason)
    return UserMemoryRecallCapabilityResult(
        entries=tuple(entries),
        scores=tuple(scores),
        used_semantic=used,
        reason=reason,
    )


@dataclass(slots=True)
class UserProfileManagerMemoryCapability:
    """Adapter from ``UserProfileManager`` to ``UserProfileMemoryCapability``."""

    _manager: UserProfileManager

    async def add_memory_entry(
        self,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> UserMemoryRememberCapabilityResult:
        try:
            mutation = await self._manager.add_memory_entry_with_lifecycle(user_id, entry)
        except Exception as exc:
            raise _map_capability_mutation_error(exc) from exc
        if mutation.entry is None:
            raise MemoryControlBackendError("remember produced no entry")
        if mutation.lifecycle.requires_reconciliation:
            raise MemoryControlPartialLifecycleError(mutation.lifecycle)
        return UserMemoryRememberCapabilityResult(
            entry=mutation.entry,
            lifecycle=mutation.lifecycle,
        )

    async def remove_memory_entry(
        self,
        user_id: str,
        entry_id: str,
    ) -> UserMemoryForgetCapabilityResult:
        try:
            mutation = await self._manager.remove_memory_entry_with_lifecycle(user_id, entry_id)
        except Exception as exc:
            raise _map_capability_mutation_error(exc) from exc
        if not mutation.lifecycle.primary_applied:
            raise MemoryControlNotFound(f"memory entry not active: {entry_id}")
        if mutation.lifecycle.requires_reconciliation:
            raise MemoryControlPartialLifecycleError(mutation.lifecycle)
        return UserMemoryForgetCapabilityResult(
            entry_id=entry_id,
            lifecycle=mutation.lifecycle,
        )

    async def list_active_memory_entries(
        self,
        user_id: str,
    ) -> tuple[UserProfileMemoryEntry, ...]:
        profile = await self._manager.get_profile(user_id)
        return tuple(
            entry for entry in profile.memory_entries if is_memory_entry_active(entry)
        )

    def is_longterm_rag_enabled(self) -> bool:
        return self._manager.is_longterm_rag_enabled()

    async def search_longterm_memory(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> UserMemoryRecallCapabilityResult:
        try:
            raw = await self._manager.search_longterm_memory(
                user_id,
                query,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            raise MemoryControlBackendError(str(exc)) from exc
        if not isinstance(raw, dict):
            raise MemoryControlBackendError("unexpected search result shape")
        return adapt_manager_search_result(raw)

    async def reconcile_memory_projections(self, user_id: str) -> MemoryReconciliationOutcome:
        return await self._manager.reconcile_memory_projections(user_id)

    async def apply_memory_supersession(
        self,
        user_id: str,
        intent: MemorySupersessionIntent,
    ) -> MemoryControlSupersessionApplyResult:
        try:
            mutation = await self._manager.apply_memory_supersession_with_lifecycle(
                user_id,
                superseded_memory_id=intent.superseded_memory_id,
                superseding_memory_id=intent.superseding_memory_id,
            )
        except Exception as exc:
            raise _map_capability_mutation_error(exc) from exc
        if mutation.lifecycle.requires_reconciliation:
            raise MemoryControlPartialLifecycleError(mutation.lifecycle)
        return MemoryControlSupersessionApplyResult(
            scope=MemoryControlPlaneScope.USER,
            superseded_memory_id=intent.superseded_memory_id,
            superseding_memory_id=intent.superseding_memory_id,
            lifecycle=mutation.lifecycle,
        )


def _recall_strategies_or_default(
    recall: MemoryRecallStrategySet | None,
) -> MemoryRecallStrategySet:
    return recall if recall is not None else build_default_memory_recall_strategies()


def _pipeline_to_recall_result(
    pipeline_result: object,
    *,
    scope: MemoryControlPlaneScope,
    used_semantic: bool,
    reason: str,
) -> MemoryControlRecallResult:
    from intergrax.memory.recall.pipeline import MemoryRecallPipelineResult

    if not isinstance(pipeline_result, MemoryRecallPipelineResult):
        raise MemoryControlBackendError("invalid recall pipeline result")
    unresolved = pipeline_result.unresolved_conflict_entry_ids
    items: list[MemoryControlRecallItem] = []
    for ranked in pipeline_result.ranked:
        record = ranked.candidate.record
        reason_codes = list(ranked.reason_codes)
        if record.entry_id in unresolved:
            reason_codes.append(MemoryRecallReasonCode.CONFLICT_UNRESOLVED)
        items.append(
            MemoryControlRecallItem(
                entry_id=record.entry_id,
                content=record.content,
                kind=record.kind,
                score=ranked.score.total,
                score_breakdown=ranked.score,
                reason_codes=tuple(reason_codes),
                conflict_unresolved=record.entry_id in unresolved,
            )
        )
    return MemoryControlRecallResult(
        scope=scope,
        items=tuple(items),
        used_semantic=used_semantic,
        reason=reason,
    )


@dataclass(slots=True)
class DefaultMemoryControlPlane:
    user_profile: UserProfileMemoryCapability | None = None
    task_memory: TaskMemoryCapability | None = None
    episodic: EpisodicMemoryCapability | None = None
    recall_strategies: MemoryRecallStrategySet | None = None
    recall_retrieval_config: UserMemoryRecallRetrievalConfig | None = None
    time_provider: type[TimeProvider] = SystemTimeProvider
    security_governance: MemorySecurityGovernanceService | None = None
    diagnostic_emitter: MemoryDiagnosticEmitter = field(
        default_factory=default_memory_diagnostic_emitter
    )

    async def remember(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> MemoryControlRememberResult:
        _assert_scope_authorized(identity, scope)
        if scope.kind is MemoryControlPlaneScope.USER:
            return await self._remember_user(identity, scope, request)
        if scope.kind is MemoryControlPlaneScope.TASK:
            return await self._remember_task(scope, request)
        raise MemoryControlUnsupportedScope(f"unsupported scope for remember: {scope.kind.value}")

    async def recall(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> MemoryControlRecallResult:
        _assert_scope_authorized(identity, scope)
        if scope.kind is MemoryControlPlaneScope.USER:
            return await self._recall_user(identity, scope, request)
        if scope.kind is MemoryControlPlaneScope.SESSION:
            return await self._recall_session(scope, request)
        raise MemoryControlUnsupportedScope(f"unsupported scope for recall: {scope.kind.value}")

    async def apply_memory_supersession(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        intent: MemorySupersessionIntent,
    ) -> MemoryControlSupersessionApplyResult:
        _assert_scope_authorized(identity, scope)
        if scope.kind is not MemoryControlPlaneScope.USER:
            raise MemoryControlUnsupportedScope(
                f"unsupported scope for supersession apply: {scope.kind.value}"
            )
        if self.user_profile is None:
            raise MemoryControlUnsupportedScope("user profile memory capability not configured")
        user_id = scope.user_id or ""
        active_entries = await self.user_profile.list_active_memory_entries(user_id)
        by_id = {entry.entry_id: entry for entry in active_entries}
        superseded = by_id.get(intent.superseded_memory_id)
        superseding = by_id.get(intent.superseding_memory_id)
        if superseded is not None:
            _enforce_governance(
                self.security_governance,
                MemoryGovernanceEvaluationRequest(
                    context=_security_context(
                        identity, scope, MemoryGovernanceOperation.SUPERSEDE
                    ),
                    target=MemoryGovernanceTarget(memory_id=intent.superseded_memory_id),
                    existing_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(
                        superseded
                    ),
                    proposed_record=(
                        MemoryGovernanceRecordSnapshot.from_user_profile_entry(superseding)
                        if superseding is not None
                        else None
                    ),
                ),
            )
        try:
            return await self.user_profile.apply_memory_supersession(user_id, intent)
        except MemoryControlPartialLifecycleError:
            raise
        except MemoryControlBackendError:
            raise
        except Exception as exc:
            raise MemoryControlBackendError(str(exc)) from exc

    async def forget(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlForgetRequest,
    ) -> MemoryControlForgetResult:
        _assert_scope_authorized(identity, scope)
        if scope.kind is MemoryControlPlaneScope.USER:
            return await self._forget_user(identity, scope, request)
        if scope.kind is MemoryControlPlaneScope.TASK:
            return await self._forget_task(scope, request)
        raise MemoryControlUnsupportedScope(f"unsupported scope for forget: {scope.kind.value}")

    async def reconcile(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlReconcileRequest,
    ) -> MemoryControlReconcileResult:
        _assert_scope_authorized(identity, scope)
        if scope.kind is not MemoryControlPlaneScope.USER:
            raise MemoryControlUnsupportedScope(
                f"reconcile supported only for USER scope, got {scope.kind.value}"
            )
        if self.user_profile is None:
            raise MemoryControlUnsupportedScope("user profile memory capability not configured")
        user_id = scope.user_id or ""
        timer = MemoryOperationTimer()
        try:
            outcome = await self.user_profile.reconcile_memory_projections(user_id)
        except MemoryControlPartialLifecycleError:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.RECONCILE,
                outcome=MemoryDiagnosticOutcome.FAILED,
                duration_seconds=timer.elapsed_seconds(),
                failure_class=MemoryDiagnosticFailureClass.RECONCILIATION,
            )
            raise
        except Exception as exc:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.RECONCILE,
                outcome=MemoryDiagnosticOutcome.FAILED,
                duration_seconds=timer.elapsed_seconds(),
                failure_class=MemoryDiagnosticFailureClass.RECONCILIATION,
            )
            raise MemoryControlBackendError(str(exc)) from exc
        reconcile_outcome = (
            MemoryDiagnosticOutcome.FAILED
            if outcome.disposition is MemoryReconciliationDisposition.FAILED
            else MemoryDiagnosticOutcome.SUCCESS
        )
        emit_control_plane_terminal(
            self.diagnostic_emitter,
            identity=identity,
            scope=scope,
            operation=MemoryDiagnosticOperation.RECONCILE,
            outcome=reconcile_outcome,
            duration_seconds=timer.elapsed_seconds(),
        )
        return MemoryControlReconcileResult(
            scope=MemoryControlPlaneScope.USER,
            reconciliation=outcome,
        )

    async def _remember_user(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> MemoryControlRememberResult:
        timer = MemoryOperationTimer()
        if self.user_profile is None:
            raise MemoryControlUnsupportedScope("user profile memory capability not configured")
        user_id = scope.user_id or ""
        if request.entry is not None:
            entry = request.entry
        else:
            content = request.content.strip()
            if not content:
                raise ValueError("remember requires content or entry")
            provenance = request.provenance or MemoryProvenance(
                source_type=MemoryRecordSourceType.USER_EXPLICIT,
            )
            trust = request.trust or MemoryRecordTrust(
                trust_class=MemoryTrustClass.USER_EXPLICIT,
            )
            entry = UserProfileMemoryEntry(
                content=content,
                kind=request.kind,
                title=request.title,
                provenance=provenance,
                trust=trust,
                governance=request.governance or MemoryRecordGovernance(),
            )
        _enforce_governance(
            self.security_governance,
            MemoryGovernanceEvaluationRequest(
                context=_security_context(identity, scope, MemoryGovernanceOperation.REMEMBER),
                proposed_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(entry),
            ),
            diagnostic_emitter=self.diagnostic_emitter,
            control_plane_operation=MemoryDiagnosticOperation.REMEMBER,
            identity=identity,
            scope=scope,
        )
        try:
            capability_result = await self.user_profile.add_memory_entry(user_id, entry)
        except MemoryControlPartialLifecycleError:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.REMEMBER,
                outcome=MemoryDiagnosticOutcome.PARTIAL,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise
        except MemoryControlBackendError:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.REMEMBER,
                outcome=MemoryDiagnosticOutcome.FAILED,
                failure_class=MemoryDiagnosticFailureClass.STORE,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise
        except Exception as exc:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.REMEMBER,
                outcome=MemoryDiagnosticOutcome.FAILED,
                failure_class=MemoryDiagnosticFailureClass.INTERNAL,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise MemoryControlBackendError(str(exc)) from exc
        emit_control_plane_terminal(
            self.diagnostic_emitter,
            identity=identity,
            scope=scope,
            operation=MemoryDiagnosticOperation.REMEMBER,
            outcome=MemoryDiagnosticOutcome.SUCCESS,
            memory_id=capability_result.entry.entry_id,
            revision=capability_result.entry.revision,
            duration_seconds=timer.elapsed_seconds(),
        )
        return MemoryControlRememberResult(
            scope=MemoryControlPlaneScope.USER,
            entry_id=capability_result.entry.entry_id,
            lifecycle=capability_result.lifecycle,
        )

    async def _recall_user(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> MemoryControlRecallResult:
        timer = MemoryOperationTimer()
        if self.user_profile is None:
            raise MemoryControlUnsupportedScope("user profile memory capability not configured")
        user_id = scope.user_id or ""
        query = request.query.strip()
        strategies = _recall_strategies_or_default(self.recall_strategies)
        retrieval_config = self.recall_retrieval_config or UserMemoryRecallRetrievalConfig()
        as_of_iso = self.time_provider.utc_now().isoformat()
        governance_service = _governance_service_or_default(self.security_governance)
        recall_governance_request = MemoryGovernanceEvaluationRequest(
            context=_security_context(identity, scope, MemoryGovernanceOperation.RECALL),
        )
        try:
            if query and self.user_profile.is_longterm_rag_enabled():
                retrieval_k = semantic_retrieval_top_k(request.top_k, retrieval_config)
                search_result = await self.user_profile.search_longterm_memory(
                    user_id,
                    query,
                    top_k=retrieval_k,
                    score_threshold=request.score_threshold,
                )
                candidates = candidates_from_semantic_search(search_result)
                candidates = governance_service.filter_recall_candidates(
                    recall_governance_request,
                    candidates,
                )
                pipeline_result = run_recall_decision_pipeline(
                    candidates=candidates,
                    query=query,
                    top_k=request.top_k,
                    ranking=strategies.ranking,
                    conflict_detection=strategies.conflict_detection,
                    conflict_resolution=strategies.conflict_resolution,
                    as_of_iso=as_of_iso,
                )
                result = _pipeline_to_recall_result(
                    pipeline_result,
                    scope=MemoryControlPlaneScope.USER,
                    used_semantic=search_result.used_semantic,
                    reason=search_result.reason,
                )
                emit_control_plane_terminal(
                    self.diagnostic_emitter,
                    identity=identity,
                    scope=scope,
                    operation=MemoryDiagnosticOperation.RECALL,
                    outcome=MemoryDiagnosticOutcome.SUCCESS,
                    duration_seconds=timer.elapsed_seconds(),
                )
                return result
            entries = await self.user_profile.list_active_memory_entries(user_id)
            candidate_limit = max(
                request.top_k,
                request.top_k * retrieval_config.retrieval_candidate_multiplier,
            )
            candidates = candidates_from_profile_scan(
                entries,
                query=query,
                candidate_limit=candidate_limit,
            )
            candidates = governance_service.filter_recall_candidates(
                recall_governance_request,
                candidates,
            )
            pipeline_result = run_recall_decision_pipeline(
                candidates=candidates,
                query=query,
                top_k=request.top_k,
                ranking=strategies.ranking,
                conflict_detection=strategies.conflict_detection,
                conflict_resolution=strategies.conflict_resolution,
                as_of_iso=as_of_iso,
            )
            reason = "keyword" if query else "profile_scan"
            result = _pipeline_to_recall_result(
                pipeline_result,
                scope=MemoryControlPlaneScope.USER,
                used_semantic=False,
                reason=reason,
            )
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.RECALL,
                outcome=MemoryDiagnosticOutcome.SUCCESS,
                duration_seconds=timer.elapsed_seconds(),
            )
            return result
        except MemoryStrategyError as exc:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.RECALL,
                outcome=MemoryDiagnosticOutcome.FAILED,
                failure_class=MemoryDiagnosticFailureClass.INTERNAL,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise MemoryControlBackendError(str(exc)) from exc
        except MemoryControlBackendError:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.RECALL,
                outcome=MemoryDiagnosticOutcome.FAILED,
                failure_class=MemoryDiagnosticFailureClass.STORE,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise
        except Exception as exc:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.RECALL,
                outcome=MemoryDiagnosticOutcome.FAILED,
                failure_class=MemoryDiagnosticFailureClass.INTERNAL,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise MemoryControlBackendError(str(exc)) from exc

    async def _forget_user(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlForgetRequest,
    ) -> MemoryControlForgetResult:
        timer = MemoryOperationTimer()
        if self.user_profile is None:
            raise MemoryControlUnsupportedScope("user profile memory capability not configured")
        entry_id = request.entry_id.strip()
        if not entry_id:
            raise ValueError("forget requires entry_id for USER scope")
        user_id = scope.user_id or ""
        active_entries = await self.user_profile.list_active_memory_entries(user_id)
        active_ids = {entry.entry_id for entry in active_entries}
        if entry_id not in active_ids:
            raise MemoryControlNotFound(f"memory entry not active: {entry_id}")
        entry = next(e for e in active_entries if e.entry_id == entry_id)
        _enforce_governance(
            self.security_governance,
            MemoryGovernanceEvaluationRequest(
                context=_security_context(identity, scope, MemoryGovernanceOperation.DELETE),
                target=MemoryGovernanceTarget(memory_id=entry_id),
                existing_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(entry),
            ),
            diagnostic_emitter=self.diagnostic_emitter,
            control_plane_operation=MemoryDiagnosticOperation.FORGET,
            identity=identity,
            scope=scope,
        )
        try:
            capability_result = await self.user_profile.remove_memory_entry(user_id, entry_id)
        except MemoryControlPartialLifecycleError:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.FORGET,
                outcome=MemoryDiagnosticOutcome.PARTIAL,
                memory_id=entry_id,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise
        except MemoryControlNotFound:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.FORGET,
                outcome=MemoryDiagnosticOutcome.NOT_FOUND,
                memory_id=entry_id,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise
        except MemoryControlBackendError:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.FORGET,
                outcome=MemoryDiagnosticOutcome.FAILED,
                memory_id=entry_id,
                failure_class=MemoryDiagnosticFailureClass.STORE,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise
        except Exception as exc:
            emit_control_plane_terminal(
                self.diagnostic_emitter,
                identity=identity,
                scope=scope,
                operation=MemoryDiagnosticOperation.FORGET,
                outcome=MemoryDiagnosticOutcome.FAILED,
                memory_id=entry_id,
                failure_class=MemoryDiagnosticFailureClass.INTERNAL,
                duration_seconds=timer.elapsed_seconds(),
            )
            raise MemoryControlBackendError(str(exc)) from exc
        entries_after = await self.user_profile.list_active_memory_entries(user_id)
        for entry in entries_after:
            if entry.entry_id == entry_id:
                raise MemoryControlBackendError("forget left entry active in primary store")
        emit_control_plane_terminal(
            self.diagnostic_emitter,
            identity=identity,
            scope=scope,
            operation=MemoryDiagnosticOperation.FORGET,
            outcome=MemoryDiagnosticOutcome.SUCCESS,
            memory_id=entry_id,
            duration_seconds=timer.elapsed_seconds(),
        )
        return MemoryControlForgetResult(
            scope=MemoryControlPlaneScope.USER,
            entry_id=entry_id,
            lifecycle=capability_result.lifecycle,
        )

    async def _remember_task(
        self,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> MemoryControlRememberResult:
        if self.task_memory is None:
            raise MemoryControlUnsupportedScope("task memory capability not configured")
        namespace = (scope.task_namespace or "").strip()
        key = (scope.task_key or "").strip()
        if not namespace or not key:
            raise MemoryControlAccessDenied("task remember requires task_namespace and task_key on scope")
        payload = request.task_value_json or request.content
        if not payload.strip():
            raise ValueError("task remember requires task_value_json or content")
        try:
            parsed: dict[str, object] = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {"value": payload}
        await self.task_memory.write(namespace, key, parsed)
        return MemoryControlRememberResult(
            scope=MemoryControlPlaneScope.TASK,
            task_written=True,
        )

    async def _forget_task(
        self,
        scope: MemoryControlScopeRef,
        request: MemoryControlForgetRequest,
    ) -> MemoryControlForgetResult:
        if self.task_memory is None:
            raise MemoryControlUnsupportedScope("task memory capability not configured")
        namespace = (request.task_namespace or scope.task_namespace or "").strip()
        key = (request.task_key or scope.task_key or "").strip()
        if not namespace or not key:
            raise MemoryControlAccessDenied("task forget requires namespace and key")
        deleted = await self.task_memory.delete(namespace, key)
        if not deleted:
            raise MemoryControlNotFound(f"task memory key not found: {namespace}/{key}")
        return MemoryControlForgetResult(
            scope=MemoryControlPlaneScope.TASK,
            task_deleted=True,
        )

    async def _recall_session(
        self,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> MemoryControlRecallResult:
        if self.episodic is None:
            raise MemoryControlUnsupportedScope("episodic memory capability not configured")
        session_id = scope.session_id or ""
        items = await self.episodic.recall_session_turns(
            tenant_id=scope.tenant_id,
            session_id=session_id,
            query=request.query,
            top_k=request.top_k,
        )
        return MemoryControlRecallResult(
            scope=MemoryControlPlaneScope.SESSION,
            items=items,
            used_semantic=bool(request.query.strip()),
            reason="session_episodic",
        )
