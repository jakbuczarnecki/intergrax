# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Tier-2 external-work adapter (GEC-3…GEC-6).

Owns mapping / correlation / normalization, continuation-evidence forwarding,
composition with meaningful side-effect policy evaluation, and composition of
descriptive ``GovernedProofProfile`` metadata. Does not own governance rules,
HITL decisions, payments, receipts, signing, persistence, polling, resume
engines, or transport. Tier-2 never decides accept/reject — it only evaluates
via an injected policy boundary before provider-bound side effects.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from intergrax.contracts.external_work import (
    CommercialQuote,
    ExternalDeliverableRef,
    ExternalProviderEvidenceRef,
    ExternalTaskCorrelation,
    ExternalWorkCapability,
    ExternalWorkCreateRequest,
    ExternalWorkErrorCode,
    ExternalWorkProviderDescriptor,
    ExternalWorkSnapshot,
    ExternalWorkStatus,
    ExternalWorkTimelineEvent,
    QuoteAcceptanceEvidence,
)
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
)
from intergrax.contracts.governed_proof import (
    GovernanceEvidenceRef,
    GovernedProofProfile,
    compose_governed_proof_profile,
    governance_evidence_ref_from_quote_acceptance,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.integrations.contracts.external_work import (
    ExternalWorkError,
    ExternalWorkIntegration,
)
from intergrax.runtime.policy.meaningful_side_effect import MeaningfulSideEffectEvaluator
from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)

# Statuses that surface a QUOTE continuation blocker (no acceptance evidence yet).
_QUOTE_CONTINUATION_STATUSES: frozenset[ExternalWorkStatus] = frozenset(
    {
        ExternalWorkStatus.QUOTE_AVAILABLE,
        ExternalWorkStatus.WAITING_FOR_ACCEPTANCE,
    }
)

# Metadata keys consumed from AgentRunRequest / AgentStepContext (merged).
META_PROVIDER_ID = "external_work.provider_id"
META_SCOPE_DESCRIPTION = "external_work.scope_description"
META_SCOPE_DIGEST = "external_work.scope_digest"
META_IDEMPOTENCY_KEY = "external_work.idempotency_key"
META_CORRELATION_ID = "external_work.correlation_id"
META_WORKSPACE_REF = "external_work.workspace_ref"
META_BUDGET_LIMIT = "external_work.budget_limit"
META_REQUESTED_CAPABILITY = "external_work.requested_capability"
META_QUOTE_ACCEPTANCE = "external_work.quote_acceptance_evidence"
META_ACCEPTANCE_IDEMPOTENCY_KEY = "external_work.acceptance_idempotency_key"
META_SKIP_ENRICHMENT = "external_work.skip_enrichment"
META_PRINCIPAL_ID = "external_work.principal_id"
META_TENANT_ID = "external_work.tenant_id"

_DEFAULT_CAPABILITY = "external_contractor.adapt"

# Provider-bound methods that mutate external state / create commitments.
_MEANINGFUL_PROVIDER_METHODS: frozenset[str] = frozenset(
    {
        "create_work",
        "submit_quote_acceptance",
        "cancel_work",
    }
)
_OBSERVATIONAL_PROVIDER_METHODS: frozenset[str] = frozenset(
    {
        "discover",
        "get_work",
        "get_quote",
        "get_timeline",
        "get_deliverables",
        "get_evidence",
    }
)

_PROOF_INVARIANT_MESSAGE = (
    "A governed side effect completed without the metadata required to "
    "compose its proof profile."
)


class _AuthorizedSideEffect(NamedTuple):
    """Validated identities + ALLOW decision shared by policy, provider, proof."""

    decision: PolicyDecision
    task_id: str
    run_id: str
    principal_id: str

# Documented classification for ExternalWorkIntegration methods (GEC-5).
PROVIDER_METHOD_SIDE_EFFECT_CLASS: dict[str, str] = {
    **{name: "meaningful_side_effect" for name in _MEANINGFUL_PROVIDER_METHODS},
    **{name: "observational" for name in _OBSERVATIONAL_PROVIDER_METHODS},
}


class ExternalWorkAdapter:
    """Translator: Intergrax intent → ExternalWorkIntegration → canonical view."""

    def __init__(
        self,
        integration: ExternalWorkIntegration,
        *,
        side_effect_policy: MeaningfulSideEffectEvaluator | None = None,
    ) -> None:
        self._integration = integration
        # Host/tests inject; missing evaluator fails closed for meaningful actions.
        self._side_effect_policy = side_effect_policy

    @property
    def integration(self) -> ExternalWorkIntegration:
        return self._integration

    @property
    def side_effect_policy(self) -> MeaningfulSideEffectEvaluator | None:
        return self._side_effect_policy

    def discover(self) -> ExternalWorkProviderDescriptor:
        return self._integration.discover()

    def build_create_request(
        self,
        *,
        task_id: str,
        run_id: str | None,
        metadata: Mapping[str, Any],
        message: str = "",
    ) -> ExternalWorkCreateRequest:
        """Map Intergrax identity + metadata into a platform create request.

        Forwards existing idempotency material; never invents retry policy or new ids.
        """
        provider_id = _require_meta_str(metadata, META_PROVIDER_ID, field="provider_id")
        scope_description = str(
            metadata.get(META_SCOPE_DESCRIPTION) or message or "external work"
        ).strip()
        scope_digest = _require_meta_str(metadata, META_SCOPE_DIGEST, field="scope_digest")
        idempotency_key = _require_meta_str(
            metadata, META_IDEMPOTENCY_KEY, field="idempotency_key"
        )
        requested = str(
            metadata.get(META_REQUESTED_CAPABILITY) or _DEFAULT_CAPABILITY
        ).strip()
        budget_raw = metadata.get(META_BUDGET_LIMIT)
        budget_limit: MoneyAmount | None = None
        if budget_raw is not None:
            budget_limit = (
                budget_raw
                if isinstance(budget_raw, MoneyAmount)
                else MoneyAmount.model_validate(budget_raw)
            )
        return ExternalWorkCreateRequest(
            provider_id=provider_id,
            task_id=task_id,
            run_id=run_id or None,
            correlation_id=_optional_meta_str(metadata, META_CORRELATION_ID),
            requested_capability=requested,
            scope_description=scope_description,
            scope_digest=scope_digest,
            idempotency_key=idempotency_key,
            workspace_ref=_optional_meta_str(metadata, META_WORKSPACE_REF),
            budget_limit=budget_limit,
            metadata={
                k: v
                for k, v in metadata.items()
                if isinstance(k, str) and k.startswith("external_work.ext.")
            },
        )

    def create_and_map(
        self,
        request: ExternalWorkCreateRequest,
        *,
        acceptance: QuoteAcceptanceEvidence | None = None,
        acceptance_idempotency_key: str | None = None,
        enrich: bool = True,
        principal_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ExternalWorkAdapterResult:
        """Synchronous create/correlate + optional enrich; no poll/retry loops.

        ``CREATE_EXTERNAL_WORK`` (and ``ACCEPT_QUOTE`` when evidence is supplied)
        are meaningful side effects — policy is evaluated before provider calls.
        Quote retrieval during enrich remains observational.
        """
        try:
            provider = self._integration.discover()
            create_gate = self._evaluate_side_effect(
                action=ACTION_CREATE_EXTERNAL_WORK,
                kinds=(MeaningfulSideEffectKind.MUTATION,),
                side_effect_scope_id=request.idempotency_key,
                task_id=request.task_id,
                run_id=request.run_id,
                principal_id=principal_id,
                tenant_id=tenant_id,
                resource=request.scope_digest,
                external_target=request.provider_id,
                correlation={
                    "task_id": request.task_id,
                    "run_id": request.run_id,
                    "idempotency_key": request.idempotency_key,
                    "correlation_id": request.correlation_id,
                    "scope_digest": request.scope_digest,
                },
                context={"requested_capability": request.requested_capability},
                continuation_reason=ContinuationReason.PROCUREMENT,
            )
            if isinstance(create_gate, ExternalWorkAdapterResult):
                return create_gate.model_copy(update={"provider": provider})
            authorized = create_gate

            snapshot = self._integration.create_work(request)
            if acceptance is not None:
                key = (acceptance_idempotency_key or "").strip()
                if not key:
                    return ExternalWorkAdapterResult(
                        used=False,
                        reason="acceptance_idempotency_key_required",
                        error_code=ExternalWorkErrorCode.INVALID_REQUEST,
                        error_message=(
                            "quote acceptance forward requires acceptance_idempotency_key"
                        ),
                        error_retryable=False,
                        provider=provider,
                        snapshot=snapshot,
                        status=snapshot.status,
                        quote=snapshot.quote,
                    )
                accept_principal = principal_id or acceptance.actor.actor_id
                accept_tenant = tenant_id or acceptance.actor.tenant_id
                # Evidence ≠ authorization — policy must still ALLOW before forward.
                forwarded = self.forward_quote_acceptance(
                    snapshot.correlation,
                    acceptance,
                    idempotency_key=key,
                    principal_id=accept_principal,
                    tenant_id=accept_tenant,
                    enrich=enrich,
                )
                if forwarded.provider is None:
                    return forwarded.model_copy(update={"provider": provider})
                return forwarded
            if not enrich:
                mapped = ExternalWorkAdapterResult(
                    used=True,
                    reason="mapped",
                    status=snapshot.status,
                    snapshot=snapshot,
                    quote=snapshot.quote,
                    provider=provider,
                )
            else:
                mapped = self._enrich(snapshot, provider=provider)
            # Proof uses the same validated identities as policy (pre-provider).
            return self._with_proof(
                mapped,
                decision=authorized.decision,
                action=ACTION_CREATE_EXTERNAL_WORK,
                principal_id=authorized.principal_id,
                tenant_id=tenant_id,
                task_id=authorized.task_id,
                run_id=authorized.run_id,
                provider_id=request.provider_id,
                resource=request.scope_digest,
                idempotency_key=request.idempotency_key,
                correlation_id=request.correlation_id
                or snapshot.correlation.correlation_id,
            )
        except ExternalWorkError as exc:
            return _error_result(exc, provider=None)

    def map_existing(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        enrich: bool = True,
    ) -> ExternalWorkAdapterResult:
        """Fetch and normalize an already-correlated external task."""
        try:
            provider = self._integration.discover()
            snapshot = self._integration.get_work(correlation)
            if not enrich:
                return ExternalWorkAdapterResult(
                    used=True,
                    reason="mapped",
                    status=snapshot.status,
                    snapshot=snapshot,
                    quote=snapshot.quote,
                    provider=provider,
                )
            return self._enrich(snapshot, provider=provider)
        except ExternalWorkError as exc:
            return _error_result(exc, provider=None)

    def forward_quote_acceptance(
        self,
        correlation: ExternalTaskCorrelation,
        acceptance: QuoteAcceptanceEvidence,
        *,
        idempotency_key: str,
        principal_id: str | None = None,
        tenant_id: str | None = None,
        enrich: bool = True,
    ) -> ExternalWorkAdapterResult:
        """Forward acceptance evidence after meaningful side-effect policy ALLOW.

        Does not decide accept/reject. Presence of evidence is not an allow.
        """
        try:
            provider = self._integration.discover()
            resolved_principal = (
                (principal_id or "").strip() or acceptance.actor.actor_id
            )
            resolved_tenant = (
                (tenant_id or "").strip() or acceptance.actor.tenant_id
            )
            gate = self._evaluate_side_effect(
                action=ACTION_ACCEPT_QUOTE,
                kinds=(
                    MeaningfulSideEffectKind.COMMITMENT,
                    MeaningfulSideEffectKind.MUTATION,
                ),
                side_effect_scope_id=idempotency_key,
                task_id=correlation.task_id,
                run_id=correlation.run_id,
                principal_id=resolved_principal,
                tenant_id=resolved_tenant,
                resource=acceptance.scope_digest,
                external_target=correlation.provider_id,
                correlation={
                    "task_id": correlation.task_id,
                    "run_id": correlation.run_id,
                    "external_task_id": correlation.external_task_id,
                    "provider_id": correlation.provider_id,
                    "idempotency_key": idempotency_key,
                    "correlation_id": correlation.correlation_id,
                    "quote_id": acceptance.quote_id,
                    "scope_digest": acceptance.scope_digest,
                },
                context={
                    "quote_id": acceptance.quote_id,
                    "quote_version": acceptance.quote_version,
                    "scope_digest": acceptance.scope_digest,
                    "hitl_decision_id": acceptance.hitl_decision_id,
                    "policy_decision_ref": acceptance.policy_decision_ref,
                },
                continuation_reason=ContinuationReason.QUOTE,
            )
            if isinstance(gate, ExternalWorkAdapterResult):
                return gate.model_copy(update={"provider": provider})
            authorized = gate

            # Forward unchanged idempotency key — policy must not mint a new one.
            snapshot = self._integration.submit_quote_acceptance(
                correlation,
                acceptance,
                idempotency_key=idempotency_key,
            )
            if not enrich:
                mapped = ExternalWorkAdapterResult(
                    used=True,
                    reason="mapped",
                    status=snapshot.status,
                    snapshot=snapshot,
                    quote=snapshot.quote,
                    provider=provider,
                )
            else:
                mapped = self._enrich(snapshot, provider=provider)
            return self._with_proof(
                mapped,
                decision=authorized.decision,
                action=ACTION_ACCEPT_QUOTE,
                principal_id=authorized.principal_id,
                tenant_id=resolved_tenant,
                task_id=authorized.task_id,
                run_id=authorized.run_id,
                provider_id=correlation.provider_id,
                resource=acceptance.scope_digest,
                idempotency_key=idempotency_key,
                correlation_id=correlation.correlation_id,
                governance_evidence=governance_evidence_ref_from_quote_acceptance(
                    acceptance_id=acceptance.acceptance_id,
                    hitl_decision_id=acceptance.hitl_decision_id,
                    interrupt_id=acceptance.interrupt_id,
                    policy_decision_ref=acceptance.policy_decision_ref,
                ),
                continuation_reason=ContinuationReason.QUOTE,
            )
        except ExternalWorkError as exc:
            return _error_result(exc, provider=None)

    def cancel_and_map(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        idempotency_key: str,
        reason: str = "",
        principal_id: str | None = None,
        tenant_id: str | None = None,
        enrich: bool = True,
    ) -> ExternalWorkAdapterResult:
        """Cancel correlated work after meaningful side-effect policy ALLOW."""
        try:
            provider = self._integration.discover()
            gate = self._evaluate_side_effect(
                action=ACTION_CANCEL_EXTERNAL_WORK,
                kinds=(MeaningfulSideEffectKind.MUTATION,),
                side_effect_scope_id=idempotency_key,
                task_id=correlation.task_id,
                run_id=correlation.run_id,
                principal_id=principal_id,
                tenant_id=tenant_id,
                resource=correlation.external_task_id,
                external_target=correlation.provider_id,
                correlation={
                    "task_id": correlation.task_id,
                    "run_id": correlation.run_id,
                    "external_task_id": correlation.external_task_id,
                    "provider_id": correlation.provider_id,
                    "idempotency_key": idempotency_key,
                    "correlation_id": correlation.correlation_id,
                },
                context={"cancel_reason": reason},
                continuation_reason=ContinuationReason.PROCUREMENT,
            )
            if isinstance(gate, ExternalWorkAdapterResult):
                return gate.model_copy(update={"provider": provider})
            authorized = gate

            snapshot = self._integration.cancel_work(
                correlation,
                idempotency_key=idempotency_key,
                reason=reason,
            )
            if not enrich:
                mapped = ExternalWorkAdapterResult(
                    used=True,
                    reason="mapped",
                    status=snapshot.status,
                    snapshot=snapshot,
                    quote=snapshot.quote,
                    provider=provider,
                )
            else:
                mapped = self._enrich(snapshot, provider=provider)
            return self._with_proof(
                mapped,
                decision=authorized.decision,
                action=ACTION_CANCEL_EXTERNAL_WORK,
                principal_id=authorized.principal_id,
                tenant_id=tenant_id,
                task_id=authorized.task_id,
                run_id=authorized.run_id,
                provider_id=correlation.provider_id,
                resource=correlation.external_task_id,
                idempotency_key=idempotency_key,
                correlation_id=correlation.correlation_id,
            )
        except ExternalWorkError as exc:
            return _error_result(exc, provider=None)

    def forward_continuation_evidence(
        self,
        correlation: ExternalTaskCorrelation,
        *,
        reason: ContinuationReason,
        evidence: QuoteAcceptanceEvidence,
        idempotency_key: str,
        principal_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ExternalWorkAdapterResult:
        """Forward continuation evidence after side-effect policy ALLOW.

        QUOTE is the first specialization — evidence is ``QuoteAcceptanceEvidence``.
        Other reasons are rejected as unsupported at this adapter (no domain map).
        Does not evaluate accept/reject or resume Nexus.
        """
        if reason is not ContinuationReason.QUOTE:
            return ExternalWorkAdapterResult(
                used=False,
                reason="continuation_reason_unsupported",
                error_code=ExternalWorkErrorCode.OPERATION_NOT_SUPPORTED,
                error_message=(
                    f"external work adapter forwards only {ContinuationReason.QUOTE.value} "
                    f"continuation evidence (got {reason.value})"
                ),
                error_retryable=False,
                metadata={"continuation_reason": reason.value},
            )
        return self.forward_quote_acceptance(
            correlation,
            evidence,
            idempotency_key=idempotency_key,
            principal_id=principal_id,
            tenant_id=tenant_id,
        )

    def surface_continuation_blocker(
        self,
        result: ExternalWorkAdapterResult,
        *,
        run_id: str,
        source_agent_id: str = "external_contractor_adapter",
        source_step_id: str | None = None,
    ) -> GovernedContinuationRequest | None:
        """Detect a QUOTE continuation blocker from a mapped result.

        Surfaces a ``GovernedContinuationRequest`` for Nexus interrupt composition.
        Requires a real Nexus ``run_id`` — never synthesizes one from ``task_id``.
        Does not create interrupts, call policy, or resume execution.
        """
        if not result.used or result.snapshot is None:
            return None
        if result.status not in _QUOTE_CONTINUATION_STATUSES:
            return None
        if result.quote is None:
            return None
        resolved_run_id = (run_id or "").strip()
        if not resolved_run_id:
            return None
        correlation = result.snapshot.correlation
        return GovernedContinuationRequest(
            reason=ContinuationReason.QUOTE,
            task_id=correlation.task_id,
            run_id=resolved_run_id,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
            prompt="External work quote requires governed continuation before side effects",
            operation_id=ACTION_ACCEPT_QUOTE,
            resource_scope=correlation.external_task_id,
            correlation={
                "task_id": correlation.task_id,
                "run_id": correlation.run_id,
                "external_task_id": correlation.external_task_id,
                "provider_id": correlation.provider_id,
                "idempotency_key": correlation.idempotency_key,
                "correlation_id": correlation.correlation_id,
            },
            context={
                "quote_id": result.quote.quote_id,
                "quote_version": result.quote.version,
                "scope_digest": result.quote.scope_digest,
                "external_work_status": (
                    result.status.value if result.status is not None else None
                ),
            },
        )

    def with_continuation_surface(
        self,
        result: ExternalWorkAdapterResult,
        *,
        run_id: str | None,
        source_agent_id: str = "external_contractor_adapter",
        source_step_id: str | None = None,
    ) -> ExternalWorkAdapterResult:
        """Attach continuation blocker when quote awaits governance evidence.

        Fail-closed when a blocker is required but no real Nexus run identity
        is available — never substitutes ``task_id`` for ``run_id``.
        """
        if not _needs_quote_continuation(result):
            return result
        resolved_run_id = (run_id or "").strip()
        if not resolved_run_id:
            return result.model_copy(
                update={
                    "used": False,
                    "reason": "continuation_correlation_failed",
                    "continuation": None,
                    "error_code": ExternalWorkErrorCode.INVALID_REQUEST,
                    "error_message": (
                        "A governed continuation blocker requires a real "
                        "Nexus run identity."
                    ),
                    "error_retryable": False,
                }
            )
        blocker = self.surface_continuation_blocker(
            result,
            run_id=resolved_run_id,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
        )
        if blocker is None:
            return result
        return result.model_copy(
            update={
                "reason": "continuation_blocked",
                "continuation": blocker,
            }
        )

    def _with_proof(
        self,
        result: ExternalWorkAdapterResult,
        *,
        decision: PolicyDecision,
        action: str,
        principal_id: str,
        tenant_id: str | None,
        task_id: str,
        run_id: str,
        provider_id: str,
        resource: str | None,
        idempotency_key: str | None,
        correlation_id: str | None,
        governance_evidence: GovernanceEvidenceRef | None = None,
        continuation_reason: ContinuationReason | None = None,
    ) -> ExternalWorkAdapterResult:
        """Attach a descriptive proof profile after a successful side effect.

        Proof composition is mandatory after ALLOW + successful provider execution —
        never best-effort. Composes metadata only — no persistence, signing, or
        receipt generation.
        """
        if (
            not result.used
            or not principal_id.strip()
            or not run_id.strip()
            or not task_id.strip()
        ):
            return self._proof_invariant_failure(result, decision=decision)
        try:
            proof: GovernedProofProfile = compose_governed_proof_profile(
                principal_id=principal_id,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                action=action,
                resource=resource,
                provider_id=provider_id,
                policy_action=decision.action,
                policy_rule_id=decision.policy_rule_id,
                policy_reason=decision.reason,
                governance_evidence=governance_evidence,
                continuation_reason=continuation_reason,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                execution_ref=run_id,
            )
        except Exception:  # noqa: BLE001 — never suppress into success-without-proof
            return self._proof_invariant_failure(result, decision=decision)
        return result.model_copy(
            update={"proof": proof, "policy_decision": decision}
        )

    @staticmethod
    def _proof_invariant_failure(
        result: ExternalWorkAdapterResult,
        *,
        decision: PolicyDecision,
    ) -> ExternalWorkAdapterResult:
        """Last-resort structured error when proof cannot be composed after success."""
        return ExternalWorkAdapterResult(
            used=False,
            reason="proof_composition_invariant_failed",
            error_code=ExternalWorkErrorCode.INVALID_REQUEST,
            error_message=_PROOF_INVARIANT_MESSAGE,
            error_retryable=False,
            policy_decision=decision,
            snapshot=result.snapshot,
            status=result.status,
            quote=result.quote,
            provider=result.provider,
            timeline=result.timeline,
            deliverables=result.deliverables,
            evidence=result.evidence,
            metadata={
                **dict(result.metadata),
                "proof_invariant": "missing_required_identity",
            },
        )

    def _evaluate_side_effect(
        self,
        *,
        action: str,
        kinds: tuple[MeaningfulSideEffectKind, ...],
        side_effect_scope_id: str,
        task_id: str,
        run_id: str | None,
        principal_id: str | None,
        tenant_id: str | None,
        resource: str | None,
        external_target: str | None,
        correlation: Mapping[str, Any],
        context: Mapping[str, Any],
        continuation_reason: ContinuationReason,
    ) -> _AuthorizedSideEffect | ExternalWorkAdapterResult:
        """Evaluate policy before a meaningful provider call.

        On ALLOW, returns validated identities + decision for reuse by provider
        correlation and mandatory proof composition. Otherwise returns a structured
        deny / governance / fail-closed result with no provider call.

        ``MeaningfulSideEffectRequest.run_id`` is mandatory and non-empty — missing
        execution identity fails closed before request construction (never ``""``).
        Proof-required fields are therefore guaranteed before the provider call.
        """
        if self._side_effect_policy is None:
            return ExternalWorkAdapterResult(
                used=False,
                reason="side_effect_policy_missing",
                error_code=ExternalWorkErrorCode.INVALID_REQUEST,
                error_message=(
                    "meaningful side effect requires an injected side-effect policy evaluator"
                ),
                error_retryable=False,
                policy_decision=PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="side_effect_policy_missing",
                    policy_rule_id="adapter.side_effect_policy_missing",
                ),
                metadata={"side_effect_action": action},
            )

        resolved_task = task_id.strip() if task_id else ""
        # Preserve platform optional semantics: absence is None, not "".
        resolved_run = run_id.strip() if run_id is not None else None
        if resolved_run is not None:
            resolved_run = resolved_run or None
        resolved_principal = (principal_id or "").strip()
        if not resolved_task or resolved_run is None:
            return ExternalWorkAdapterResult(
                used=False,
                reason="side_effect_identity_missing",
                error_code=ExternalWorkErrorCode.INVALID_REQUEST,
                error_message="meaningful side effect requires task_id and real Nexus run_id",
                error_retryable=False,
                policy_decision=PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="meaningful_side_effect_identity_missing",
                    policy_rule_id="adapter.side_effect_identity",
                ),
                metadata={"side_effect_action": action},
            )
        if not resolved_principal:
            return ExternalWorkAdapterResult(
                used=False,
                reason="side_effect_principal_missing",
                error_code=ExternalWorkErrorCode.INVALID_REQUEST,
                error_message="meaningful side effect requires principal identity",
                error_retryable=False,
                policy_decision=PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="meaningful_side_effect_principal_missing",
                    policy_rule_id="adapter.side_effect_principal",
                ),
                metadata={"side_effect_action": action},
            )

        resolved_scope_id = side_effect_scope_id.strip()
        if not resolved_scope_id:
            return ExternalWorkAdapterResult(
                used=False,
                reason="side_effect_scope_id_missing",
                error_code=ExternalWorkErrorCode.INVALID_REQUEST,
                error_message="meaningful side effect requires side_effect_scope_id",
                error_retryable=False,
                policy_decision=PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="meaningful_side_effect_scope_id_missing",
                    policy_rule_id="adapter.side_effect_scope_id",
                ),
                metadata={"side_effect_action": action},
            )

        try:
            request = MeaningfulSideEffectRequest(
                action=action,
                kinds=kinds,
                side_effect_scope_id=resolved_scope_id,
                task_id=resolved_task,
                run_id=resolved_run,
                principal_id=resolved_principal,
                tenant_id=tenant_id,
                resource=resource,
                external_target=external_target,
                correlation=dict(correlation),
                context=dict(context),
            )
            decision = self._side_effect_policy.evaluate_meaningful_side_effect(request)
        except Exception as exc:  # noqa: BLE001 — fail closed on evaluator faults
            return ExternalWorkAdapterResult(
                used=False,
                reason="side_effect_policy_evaluation_failed",
                error_code=ExternalWorkErrorCode.INVALID_REQUEST,
                error_message=f"side-effect policy evaluation failed: {exc}",
                error_retryable=False,
                policy_decision=PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="side_effect_policy_evaluation_failed",
                    policy_rule_id="adapter.side_effect_policy_fault",
                ),
                metadata={"side_effect_action": action},
            )

        if decision.action is PolicyAction.ALLOW:
            return _AuthorizedSideEffect(
                decision=decision,
                task_id=resolved_task,
                run_id=resolved_run,
                principal_id=resolved_principal,
            )

        if decision.action in (PolicyAction.REQUIRE_HUMAN, PolicyAction.ESCALATE):
            blocker = GovernedContinuationRequest(
                reason=continuation_reason,
                task_id=resolved_task,
                run_id=resolved_run,
                source_agent_id="external_contractor_adapter",
                prompt=(
                    f"Meaningful side effect {action} requires governed continuation "
                    f"before provider execution ({decision.reason})"
                ),
                operation_id=action,
                policy_rule_id=decision.policy_rule_id,
                resource_scope=resource,
                policy_action=decision.action,
                correlation=dict(correlation),
                context={
                    "side_effect_action": action,
                    "policy_rule_id": decision.policy_rule_id,
                    "policy_reason": decision.reason,
                    **dict(context),
                },
            )
            return ExternalWorkAdapterResult(
                used=False,
                reason="side_effect_governance_required",
                continuation=blocker,
                policy_decision=decision,
                metadata={"side_effect_action": action},
            )

        # DENY, MODIFY (unsupported), and any other non-ALLOW → fail closed.
        return ExternalWorkAdapterResult(
            used=False,
            reason="side_effect_denied",
            error_code=ExternalWorkErrorCode.INVALID_REQUEST,
            error_message=decision.reason or f"side effect {action} denied by policy",
            error_retryable=False,
            policy_decision=decision,
            metadata={"side_effect_action": action},
        )

    def _enrich(
        self,
        snapshot: ExternalWorkSnapshot,
        *,
        provider: ExternalWorkProviderDescriptor,
    ) -> ExternalWorkAdapterResult:
        correlation = snapshot.correlation
        unsupported: list[ExternalWorkCapability] = []
        quote = snapshot.quote
        timeline: tuple[ExternalWorkTimelineEvent, ...] = ()
        deliverables: tuple[ExternalDeliverableRef, ...] = ()
        evidence: tuple[ExternalProviderEvidenceRef, ...] = ()

        if provider.supports(ExternalWorkCapability.QUOTE_FIRST):
            if quote is None:
                try:
                    quote = self._integration.get_quote(correlation)
                except ExternalWorkError as exc:
                    if exc.code != ExternalWorkErrorCode.QUOTE_UNAVAILABLE:
                        return _error_result(exc, provider=provider)
        else:
            unsupported.append(ExternalWorkCapability.QUOTE_FIRST)

        if provider.supports(ExternalWorkCapability.TIMELINE):
            timeline = tuple(self._integration.get_timeline(correlation))
        else:
            unsupported.append(ExternalWorkCapability.TIMELINE)

        if provider.supports(ExternalWorkCapability.DELIVERABLES):
            deliverables = tuple(self._integration.get_deliverables(correlation))
        else:
            unsupported.append(ExternalWorkCapability.DELIVERABLES)

        if provider.supports(ExternalWorkCapability.EVIDENCE_REFS):
            evidence = tuple(self._integration.get_evidence(correlation))
        else:
            unsupported.append(ExternalWorkCapability.EVIDENCE_REFS)

        return ExternalWorkAdapterResult(
            used=True,
            reason="mapped",
            status=snapshot.status,
            snapshot=snapshot,
            quote=quote,
            timeline=timeline,
            deliverables=deliverables,
            evidence=evidence,
            provider=provider,
            unsupported_capabilities=tuple(unsupported),
            metadata={
                "external_task_id": correlation.external_task_id,
                "provider_id": correlation.provider_id,
                "idempotency_key": correlation.idempotency_key,
            },
        )


def adapt_from_step_metadata(
    integration: ExternalWorkIntegration | None,
    *,
    task_id: str,
    run_id: str | None,
    message: str,
    metadata: Mapping[str, Any],
    side_effect_policy: MeaningfulSideEffectEvaluator | None = None,
) -> ExternalWorkAdapterResult:
    """Entry used by the reflex domain step — injection required, no construction."""
    if integration is None:
        return ExternalWorkAdapterResult(
            used=False,
            reason="external_work_integration_missing",
            metadata={"capability": _DEFAULT_CAPABILITY},
        )

    adapter = ExternalWorkAdapter(integration, side_effect_policy=side_effect_policy)
    try:
        request = adapter.build_create_request(
            task_id=task_id,
            run_id=run_id,
            metadata=metadata,
            message=message,
        )
    except ValueError as exc:
        return ExternalWorkAdapterResult(
            used=False,
            reason="invalid_adapt_intent",
            error_code=ExternalWorkErrorCode.INVALID_REQUEST,
            error_message=str(exc),
            error_retryable=False,
        )

    acceptance = _parse_acceptance(metadata.get(META_QUOTE_ACCEPTANCE))
    acceptance_key = _optional_meta_str(metadata, META_ACCEPTANCE_IDEMPOTENCY_KEY)
    enrich = not bool(metadata.get(META_SKIP_ENRICHMENT))
    principal_id = _optional_meta_str(metadata, META_PRINCIPAL_ID)
    tenant_id = _optional_meta_str(metadata, META_TENANT_ID)
    if acceptance is not None:
        principal_id = principal_id or acceptance.actor.actor_id
        tenant_id = tenant_id or acceptance.actor.tenant_id
    mapped = adapter.create_and_map(
        request,
        acceptance=acceptance,
        acceptance_idempotency_key=acceptance_key,
        enrich=enrich,
        principal_id=principal_id,
        tenant_id=tenant_id,
    )
    # When no continuation evidence was supplied, surface the blocker for Nexus.
    # Forward the real Nexus run_id from execution context — never invent one.
    # Quote availability is observational — not a meaningful side-effect gate.
    if acceptance is None and mapped.used:
        return adapter.with_continuation_surface(mapped, run_id=run_id)
    return mapped


def _needs_quote_continuation(result: ExternalWorkAdapterResult) -> bool:
    """True when mapped state requires a QUOTE governed-continuation blocker."""
    return (
        result.used
        and result.snapshot is not None
        and result.quote is not None
        and result.status in _QUOTE_CONTINUATION_STATUSES
    )


def _error_result(
    exc: ExternalWorkError,
    *,
    provider: ExternalWorkProviderDescriptor | None,
) -> ExternalWorkAdapterResult:
    return ExternalWorkAdapterResult(
        used=False,
        reason="external_work_error",
        error_code=exc.code,
        error_message=str(exc),
        error_retryable=exc.retryable,
        provider=provider,
    )


def _require_meta_str(metadata: Mapping[str, Any], key: str, *, field: str) -> str:
    raw = metadata.get(key)
    if raw is None or not str(raw).strip():
        raise ValueError(f"missing required metadata {key!r} ({field})")
    return str(raw).strip()


def _optional_meta_str(metadata: Mapping[str, Any], key: str) -> str | None:
    raw = metadata.get(key)
    if raw is None:
        return None
    normalized = str(raw).strip()
    return normalized or None


def _parse_acceptance(raw: object) -> QuoteAcceptanceEvidence | None:
    if raw is None:
        return None
    if isinstance(raw, QuoteAcceptanceEvidence):
        return raw
    return QuoteAcceptanceEvidence.model_validate(raw)
