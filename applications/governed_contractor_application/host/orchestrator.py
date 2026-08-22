# © Artur Czarnecki. All rights reserved.

"""GovernedExternalWorkOrchestrator — host lifecycle (PC-5…PC-7).

Owns lifecycle orchestration, persistence, attestation, and attestation-only
recovery. Does not own Tier-2 mapping rules or provider transport.

Invariants:
- continuation ≠ execution
- human/payment evidence ≠ authorization
- receipt ≠ authorization
- verification ≠ authorization
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4

from external_contractor_adapter.external_work_adapter import (
    META_CORRELATION_ID,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_WORKSPACE_REF,
    ExternalWorkAdapter,
)
from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.execution_evidence.attestation import HostAttestor
from intergrax.contracts.execution_evidence.boundary_event import ExecutionBoundaryEvent
from intergrax.contracts.execution_evidence.receipt import ProofReceipt
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)
from intergrax.contracts.governed_execution_result import GovernedExecutionResult
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from intergrax.runtime.attestation.canonical_json import stable_payload_hash
from intergrax.runtime.execution_evidence.compose import (
    AttestationOutcome,
    attest_governed_execution_result,
    compose_execution_boundary_event_from_result,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)
from governed_contractor_application.host.lifecycle_states import (
    GovernedExternalWorkHostState,
    map_provider_status_to_host_state,
)
from governed_contractor_application.host.stores import (
    ContinuationStateStore,
    GovernedExecutionStore,
    PolicyBundleArtifactStore,
    ProofReceiptStore,
)

_ACTION_TO_OPERATION: Mapping[str, str] = {
    ACTION_CREATE_EXTERNAL_WORK: "create_work",
    ACTION_ACCEPT_QUOTE: "submit_quote_acceptance",
    ACTION_CANCEL_EXTERNAL_WORK: "cancel_work",
}

META_PROVIDER_INVOCATION_ID = "provider_invocation_id"


def _workspace_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    raw = metadata.get(META_WORKSPACE_REF)
    if raw is None:
        return None
    normalized = str(raw).strip()
    return normalized or None


@dataclass(frozen=True, slots=True)
class OrchestratorStepResult:
    """Outcome of one host lifecycle step."""

    state: GovernedExternalWorkHostState
    execution_id: str | None
    adapter_result: ExternalWorkAdapterResult | None
    governed_result: GovernedExecutionResult | None
    attestation: AttestationOutcome | None
    receipt: ProofReceipt | None
    reason: str


class GovernedExternalWorkOrchestrator:
    """First-class host service for governed external-work lifecycle."""

    def __init__(
        self,
        *,
        adapter: ExternalWorkAdapter,
        policy: RuntimePolicyBundleEvaluator,
        bundle: ImmutableRuntimePolicyBundle,
        attestor: HostAttestor | None,
        capabilities: ExternalWorkProviderCapabilities,
        execution_store: GovernedExecutionStore,
        receipt_store: ProofReceiptStore,
        bundle_store: PolicyBundleArtifactStore,
        continuation_store: ContinuationStateStore,
        clock: Callable[[], datetime] | None = None,
        actor: str = "governed_contractor_host",
    ) -> None:
        self._adapter = adapter
        self._policy = policy
        self._bundle = bundle
        self._attestor = attestor
        self._capabilities = capabilities
        self._execution_store = execution_store
        self._receipt_store = receipt_store
        self._bundle_store = bundle_store
        self._continuation_store = continuation_store
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._actor = actor
        self._bundle_store.put_bundle(bundle)

    @property
    def capabilities(self) -> ExternalWorkProviderCapabilities:
        return self._capabilities

    def create(
        self,
        *,
        task_id: str,
        run_id: str,
        principal_id: str,
        tenant_id: str | None,
        metadata: Mapping[str, Any],
        execution_id: str | None = None,
        event_id: str | None = None,
        receipt_id: str | None = None,
    ) -> OrchestratorStepResult:
        if not self._capabilities.supports_create:
            raise ValueError("provider_capability_missing:supports_create")
        exec_id = execution_id or f"exec-{uuid4().hex}"
        self._execution_store.put_state(
            exec_id, GovernedExternalWorkHostState.REQUESTED
        )
        started = self._clock()
        invocation = self._new_invocation(
            action=ACTION_CREATE_EXTERNAL_WORK,
            task_id=task_id,
            run_id=run_id,
            metadata=metadata,
            started_at=started,
        )
        meta = dict(metadata)
        meta[META_PROVIDER_INVOCATION_ID] = invocation.invocation_id
        self._execution_store.put_state(
            exec_id, GovernedExternalWorkHostState.CREATE_IN_PROGRESS
        )
        request = self._adapter.build_create_request(
            task_id=task_id,
            run_id=run_id,
            metadata=meta,
        )
        adapter_result = self._adapter.create_and_map(
            request,
            principal_id=principal_id,
            tenant_id=tenant_id,
        )
        adapter_result = adapter_result.model_copy(
            update={
                "metadata": {
                    **dict(adapter_result.metadata),
                    META_PROVIDER_INVOCATION_ID: invocation.invocation_id,
                    "provider_operation": "create_work",
                }
            }
        )
        return self._finalize_side_effect(
            execution_id=exec_id,
            action=ACTION_CREATE_EXTERNAL_WORK,
            invocation=invocation,
            started_at=started,
            adapter_result=adapter_result,
            principal_id=principal_id,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            metadata=meta,
            deny_state=GovernedExternalWorkHostState.CREATE_POLICY_DENIED,
            event_id=event_id,
            receipt_id=receipt_id,
            after_create=True,
        )

    def surface_continuation(
        self,
        *,
        execution_id: str,
        adapter_result: ExternalWorkAdapterResult,
        run_id: str,
    ) -> OrchestratorStepResult:
        """Surface quote continuation — zero provider calls."""
        surfaced = self._adapter.with_continuation_surface(
            adapter_result, run_id=run_id
        )
        if surfaced.continuation is not None and surfaced.snapshot is not None:
            self._continuation_store.put_continuation(
                surfaced.snapshot.correlation.task_id,
                surfaced.continuation,
            )
            self._execution_store.put_state(
                execution_id, GovernedExternalWorkHostState.AWAITING_HUMAN
            )
            return OrchestratorStepResult(
                state=GovernedExternalWorkHostState.AWAITING_HUMAN,
                execution_id=execution_id,
                adapter_result=surfaced,
                governed_result=None,
                attestation=None,
                receipt=None,
                reason="awaiting_human_continuation",
            )
        mapped = map_provider_status_to_host_state(surfaced.status, after_create=True)
        state = mapped or GovernedExternalWorkHostState.QUOTE_RECEIVED
        self._execution_store.put_state(execution_id, state)
        return OrchestratorStepResult(
            state=state,
            execution_id=execution_id,
            adapter_result=surfaced,
            governed_result=None,
            attestation=None,
            receipt=None,
            reason="continuation_not_required",
        )

    def accept(
        self,
        *,
        execution_id: str,
        create_result: ExternalWorkAdapterResult,
        acceptance: QuoteAcceptanceEvidence,
        idempotency_key: str,
        principal_id: str,
        tenant_id: str | None,
        metadata: Mapping[str, Any] | None = None,
        event_id: str | None = None,
        receipt_id: str | None = None,
    ) -> OrchestratorStepResult:
        if not self._capabilities.supports_accept:
            raise ValueError("provider_capability_missing:supports_accept")
        if create_result.snapshot is None:
            raise ValueError("accept_requires_create_snapshot")
        # Human evidence ≠ authorization — ACCEPT requires a fresh policy eval
        # inside the adapter via the injected bundle evaluator.
        started = self._clock()
        meta = dict(metadata or {})
        meta.setdefault(
            META_PROVIDER_ID,
            create_result.snapshot.correlation.provider_id
            if hasattr(create_result.snapshot.correlation, "provider_id")
            else meta.get(META_PROVIDER_ID, ""),
        )
        # Correlation may not have provider_id — use snapshot / prior meta.
        provider_id = str(
            meta.get(META_PROVIDER_ID)
            or (
                create_result.provider.provider_id
                if create_result.provider is not None
                else "unknown"
            )
        )
        meta[META_PROVIDER_ID] = provider_id
        meta[META_IDEMPOTENCY_KEY] = idempotency_key
        if create_result.snapshot.correlation.correlation_id:
            meta[META_CORRELATION_ID] = (
                create_result.snapshot.correlation.correlation_id
            )
        accept_run_id = create_result.snapshot.correlation.run_id
        if not accept_run_id or not str(accept_run_id).strip():
            raise ValueError("accept_requires_run_id")
        invocation = self._new_invocation(
            action=ACTION_ACCEPT_QUOTE,
            task_id=create_result.snapshot.correlation.task_id,
            run_id=str(accept_run_id),
            metadata=meta,
            started_at=started,
            external_task_id=create_result.snapshot.correlation.external_task_id,
        )
        meta[META_PROVIDER_INVOCATION_ID] = invocation.invocation_id
        self._execution_store.put_state(
            execution_id, GovernedExternalWorkHostState.EXECUTION_IN_PROGRESS
        )
        adapter_result = self._adapter.forward_quote_acceptance(
            create_result.snapshot.correlation,
            acceptance,
            idempotency_key=idempotency_key,
            principal_id=principal_id,
            tenant_id=tenant_id,
            workspace_id=_workspace_from_metadata(meta),
        )
        # Attach invocation id for GER / legacy compose paths.
        adapter_result = adapter_result.model_copy(
            update={
                "metadata": {
                    **dict(adapter_result.metadata),
                    META_PROVIDER_INVOCATION_ID: invocation.invocation_id,
                    "provider_operation": "submit_quote_acceptance",
                }
            }
        )
        return self._finalize_side_effect(
            execution_id=execution_id,
            action=ACTION_ACCEPT_QUOTE,
            invocation=invocation,
            started_at=started,
            adapter_result=adapter_result,
            principal_id=principal_id,
            tenant_id=tenant_id,
            task_id=create_result.snapshot.correlation.task_id,
            run_id=str(accept_run_id),
            metadata=meta,
            deny_state=GovernedExternalWorkHostState.ACCEPT_POLICY_DENIED,
            event_id=event_id,
            receipt_id=receipt_id,
            after_create=False,
        )

    def cancel(
        self,
        *,
        execution_id: str,
        create_result: ExternalWorkAdapterResult,
        principal_id: str,
        tenant_id: str | None,
        idempotency_key: str,
        metadata: Mapping[str, Any] | None = None,
        event_id: str | None = None,
        receipt_id: str | None = None,
    ) -> OrchestratorStepResult:
        if not self._capabilities.supports_cancel:
            raise ValueError("provider_capability_missing:supports_cancel")
        if create_result.snapshot is None:
            raise ValueError("cancel_requires_create_snapshot")
        started = self._clock()
        meta = dict(metadata or {})
        meta[META_IDEMPOTENCY_KEY] = idempotency_key
        provider_id = str(
            meta.get(META_PROVIDER_ID)
            or (
                create_result.provider.provider_id
                if create_result.provider is not None
                else "unknown"
            )
        )
        meta[META_PROVIDER_ID] = provider_id
        cancel_run_id = create_result.snapshot.correlation.run_id
        if not cancel_run_id or not str(cancel_run_id).strip():
            raise ValueError("cancel_requires_run_id")
        invocation = self._new_invocation(
            action=ACTION_CANCEL_EXTERNAL_WORK,
            task_id=create_result.snapshot.correlation.task_id,
            run_id=str(cancel_run_id),
            metadata=meta,
            started_at=started,
            external_task_id=create_result.snapshot.correlation.external_task_id,
        )
        meta[META_PROVIDER_INVOCATION_ID] = invocation.invocation_id
        adapter_result = self._adapter.cancel_and_map(
            create_result.snapshot.correlation,
            principal_id=principal_id,
            tenant_id=tenant_id,
            workspace_id=_workspace_from_metadata(meta),
            idempotency_key=idempotency_key,
        )
        adapter_result = adapter_result.model_copy(
            update={
                "metadata": {
                    **dict(adapter_result.metadata),
                    META_PROVIDER_INVOCATION_ID: invocation.invocation_id,
                    "provider_operation": "cancel_work",
                }
            }
        )
        step = self._finalize_side_effect(
            execution_id=execution_id,
            action=ACTION_CANCEL_EXTERNAL_WORK,
            invocation=invocation,
            started_at=started,
            adapter_result=adapter_result,
            principal_id=principal_id,
            tenant_id=tenant_id,
            task_id=create_result.snapshot.correlation.task_id,
            run_id=str(cancel_run_id),
            metadata=meta,
            deny_state=GovernedExternalWorkHostState.EXECUTION_FAILED,
            event_id=event_id,
            receipt_id=receipt_id,
            after_create=False,
        )
        if step.attestation and step.attestation.attestation_succeeded:
            self._execution_store.put_state(
                execution_id, GovernedExternalWorkHostState.CANCELLED
            )
            return OrchestratorStepResult(
                state=GovernedExternalWorkHostState.CANCELLED,
                execution_id=step.execution_id,
                adapter_result=step.adapter_result,
                governed_result=step.governed_result,
                attestation=step.attestation,
                receipt=step.receipt,
                reason=step.reason,
            )
        return step

    def retry_attestation(
        self,
        execution_id: str,
        *,
        event_id: str | None = None,
        receipt_id: str | None = None,
    ) -> OrchestratorStepResult:
        """Attestation-only recovery — never repeats provider side effects (PC-7)."""
        existing = self._receipt_store.get_receipt(execution_id)
        if existing is not None:
            self._execution_store.put_state(
                execution_id,
                GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED,
            )
            return OrchestratorStepResult(
                state=GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED,
                execution_id=execution_id,
                adapter_result=None,
                governed_result=self._execution_store.get_result(execution_id),
                attestation=AttestationOutcome(
                    execution_succeeded=True,
                    attestation_succeeded=True,
                    receipt=existing,
                    event=existing.execution_boundary_event,
                    reason="attested_idempotent",
                    provider_invoked=False,
                ),
                receipt=existing,
                reason="attested_idempotent",
            )
        result = self._execution_store.get_result(execution_id)
        if result is None:
            raise ValueError(f"execution_result_missing:{execution_id}")
        state = self._execution_store.get_state(execution_id)
        if state is GovernedExternalWorkHostState.EXECUTION_FAILED:
            raise ValueError("cannot_attest_failed_execution")
        # Prefer persisted event bytes for deterministic retry.
        event: ExecutionBoundaryEvent | None = None
        stored_event = self._execution_store.get_event_json(execution_id)
        if stored_event:
            event = ExecutionBoundaryEvent.model_validate_json(stored_event)
        outcome = attest_governed_execution_result(
            result,
            attestor=self._attestor,
            policy_bundle_artifact=self._bundle,
            attestation_required=True,
            actor=self._actor,
            event_id=event_id or (event.event_id if event is not None else None),
            receipt_id=receipt_id,
            occurred_at=result.execution_completed_at,
            require_first_class_invocation=True,
        )
        if outcome.attestation_succeeded and outcome.receipt is not None:
            self._receipt_store.put_receipt(execution_id, outcome.receipt)
            if outcome.event is not None:
                self._execution_store.put_event_json(
                    execution_id, outcome.event.model_dump_json()
                )
            self._execution_store.put_state(
                execution_id,
                GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED,
            )
            return OrchestratorStepResult(
                state=GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED,
                execution_id=execution_id,
                adapter_result=None,
                governed_result=result,
                attestation=outcome,
                receipt=outcome.receipt,
                reason="attested",
            )
        self._execution_store.put_state(
            execution_id,
            GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTATION_FAILED,
        )
        return OrchestratorStepResult(
            state=GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTATION_FAILED,
            execution_id=execution_id,
            adapter_result=None,
            governed_result=result,
            attestation=outcome,
            receipt=None,
            reason=outcome.reason,
        )

    def get_state(self, execution_id: str) -> GovernedExternalWorkHostState | None:
        return self._execution_store.get_state(execution_id)

    def get_result(self, execution_id: str) -> GovernedExecutionResult | None:
        return self._execution_store.get_result(execution_id)

    def get_receipt(self, execution_id: str) -> ProofReceipt | None:
        return self._receipt_store.get_receipt(execution_id)

    def _finalize_side_effect(
        self,
        *,
        execution_id: str,
        action: str,
        invocation: ProviderInvocation,
        started_at: datetime,
        adapter_result: ExternalWorkAdapterResult,
        principal_id: str,
        tenant_id: str | None,
        task_id: str,
        run_id: str,
        metadata: Mapping[str, Any],
        deny_state: GovernedExternalWorkHostState,
        event_id: str | None,
        receipt_id: str | None,
        after_create: bool,
    ) -> OrchestratorStepResult:
        completed = self._clock()
        decision = adapter_result.policy_decision
        if decision is not None and decision.action is not PolicyAction.ALLOW:
            self._execution_store.put_state(execution_id, deny_state)
            return OrchestratorStepResult(
                state=deny_state,
                execution_id=execution_id,
                adapter_result=adapter_result,
                governed_result=None,
                attestation=None,
                receipt=None,
                reason="policy_denied",
            )
        if (
            not adapter_result.used
            or adapter_result.proof is None
            or decision is None
            or decision.action is not PolicyAction.ALLOW
        ):
            failed = GovernedExternalWorkHostState.EXECUTION_FAILED
            self._execution_store.put_state(execution_id, failed)
            return OrchestratorStepResult(
                state=failed,
                execution_id=execution_id,
                adapter_result=adapter_result,
                governed_result=None,
                attestation=None,
                receipt=None,
                reason=adapter_result.reason or "execution_failed",
            )

        evaluated = self._policy.last_evaluation
        if evaluated is None or evaluated.decision.decision_id != decision.decision_id:
            # Reconstruct from decision + bound pack (same evaluation identity).
            from intergrax.contracts.evaluated_policy_decision import (
                EvaluatedPolicyDecision,
            )

            req_digest = str(
                decision.audit_payload.get("request_digest")
                or stable_payload_hash({"action": action, "task_id": task_id})
            )
            evaluated_at_raw = decision.audit_payload.get("evaluated_at")
            evaluated_at = (
                datetime.fromisoformat(str(evaluated_at_raw))
                if evaluated_at_raw
                else completed
            )
            evaluated = EvaluatedPolicyDecision(
                decision=decision,
                bundle_id=decision.policy_bundle_id,
                bundle_version=decision.policy_bundle_version,
                bundle_digest=decision.policy_bundle_digest,
                matched_rule_id=decision.policy_rule_id,
                evaluated_at=evaluated_at,
                request_digest=req_digest,
            )
        evaluated.assert_consistent_with_bundle(self._bundle)

        corr = None
        idem = None
        external_task_id = invocation.external_task_id
        if adapter_result.snapshot is not None:
            corr = adapter_result.snapshot.correlation.correlation_id
            external_task_id = (
                external_task_id
                or adapter_result.snapshot.correlation.external_task_id
            )
        if adapter_result.proof is not None:
            idem = adapter_result.proof.idempotency_key
            corr = corr or adapter_result.proof.correlation_id

        inv = invocation.model_copy(
            update={
                "external_task_id": external_task_id,
                "correlation_id": corr or invocation.correlation_id,
                "idempotency_key": idem or invocation.idempotency_key,
            }
        )
        outcome = ProviderInvocationOutcome(
            invocation_id=inv.invocation_id,
            status=ProviderInvocationStatus.SUCCEEDED,
            completed_at=completed,
            response_digest=stable_payload_hash(
                {
                    "execution_id": execution_id,
                    "action": action,
                    "status": (
                        adapter_result.status.value
                        if adapter_result.status is not None
                        else "unknown"
                    ),
                }
            ),
            external_status=(
                adapter_result.status.value
                if adapter_result.status is not None
                else None
            ),
        )
        ger = GovernedExecutionResult(
            execution_id=execution_id,
            task_id=task_id,
            run_id=run_id,
            principal_id=principal_id,
            tenant_id=tenant_id,
            correlation_id=corr,
            idempotency_key=idem,
            action=action,
            evaluated_policy_decision=evaluated,
            provider_invocation=inv,
            provider_outcome=outcome,
            proof=adapter_result.proof,
            execution_started_at=started_at,
            execution_completed_at=completed,
        )
        self._execution_store.put_result(ger)
        self._execution_store.put_state(
            execution_id,
            GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTATION_PENDING,
        )
        # Persist deterministic EBE before signing so retry can reuse it.
        try:
            event = compose_execution_boundary_event_from_result(
                ger,
                event_id=event_id,
                occurred_at=completed,
                actor=self._actor,
            )
            self._execution_store.put_event_json(
                execution_id, event.model_dump_json()
            )
        except ValueError:
            event = None

        attestation = attest_governed_execution_result(
            ger,
            attestor=self._attestor,
            policy_bundle_artifact=self._bundle,
            attestation_required=True,
            actor=self._actor,
            event_id=event_id or (event.event_id if event is not None else None),
            receipt_id=receipt_id,
            occurred_at=completed,
            require_first_class_invocation=True,
        )
        if attestation.attestation_succeeded and attestation.receipt is not None:
            self._receipt_store.put_receipt(execution_id, attestation.receipt)
            self._execution_store.put_state(
                execution_id,
                GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED,
            )
            host_state = GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTED
            # After CREATE, prefer quote/awaiting mapping for UX while attested.
            if after_create and action == ACTION_CREATE_EXTERNAL_WORK:
                mapped = map_provider_status_to_host_state(
                    adapter_result.status, after_create=True
                )
                if mapped in {
                    GovernedExternalWorkHostState.QUOTE_RECEIVED,
                    GovernedExternalWorkHostState.AWAITING_HUMAN,
                }:
                    # Keep attested result; surface_continuation sets AWAITING_HUMAN.
                    pass
            return OrchestratorStepResult(
                state=host_state,
                execution_id=execution_id,
                adapter_result=adapter_result,
                governed_result=ger,
                attestation=attestation,
                receipt=attestation.receipt,
                reason="attested",
            )
        self._execution_store.put_state(
            execution_id,
            GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTATION_FAILED,
        )
        return OrchestratorStepResult(
            state=GovernedExternalWorkHostState.EXECUTION_SUCCEEDED_ATTESTATION_FAILED,
            execution_id=execution_id,
            adapter_result=adapter_result,
            governed_result=ger,
            attestation=attestation,
            receipt=None,
            reason=attestation.reason,
        )

    def _new_invocation(
        self,
        *,
        action: str,
        task_id: str,
        run_id: str,
        metadata: Mapping[str, Any],
        started_at: datetime,
        external_task_id: str | None = None,
    ) -> ProviderInvocation:
        operation = _ACTION_TO_OPERATION[action]
        provider_id = str(metadata.get(META_PROVIDER_ID) or "").strip()
        if not provider_id:
            raise ValueError("provider_id_required_for_invocation")
        corr = metadata.get(META_CORRELATION_ID)
        idem = metadata.get(META_IDEMPOTENCY_KEY)
        request_digest = stable_payload_hash(
            {
                "action": action,
                "operation": operation,
                "task_id": task_id,
                "run_id": run_id,
                "provider_id": provider_id,
                "idempotency_key": str(idem) if idem else None,
                "correlation_id": str(corr) if corr else None,
            }
        )
        return ProviderInvocation(
            invocation_id=f"inv-{uuid4().hex}",
            provider_id=provider_id,
            operation=operation,
            task_id=task_id,
            run_id=run_id,
            external_task_id=external_task_id,
            correlation_id=str(corr).strip() if isinstance(corr, str) and corr.strip() else None,
            idempotency_key=str(idem).strip() if isinstance(idem, str) and idem.strip() else None,
            request_digest=request_digest,
            started_at=started_at,
        )
