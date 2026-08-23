# © Artur Czarnecki. All rights reserved.

"""Post-apply verification loop for L4-V (Phase W-ADAPT-5.1–5.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.adaptive.contracts import ProfileVersionStatus
from intergrax.runtime.adaptive.control_plane_governance import AhiGovernanceBlockedError
from intergrax.runtime.adaptive.loop_apply_block_store import LoopApplyBlockStore
from intergrax.runtime.adaptive.profile_pointer_store import (
    ProfileActivePointerConflictError,
    ProfileActivePointerStore,
)
from intergrax.runtime.adaptive.profile_version_store import ProfileVersionStore
from intergrax.runtime.adaptive.signal_store import SignalStore
from intergrax.runtime.adaptive.verification_checks import (
    HarnessSecurityAdversarialBaselineChecker,
    SecurityAdversarialBaselineChecker,
    check_cost_budget,
    check_eval_registry_trend,
    check_regression_rate,
    check_security_adversarial,
    check_utility_trend,
    split_candidate_baseline_signals,
)
from intergrax.runtime.adaptive.verification_models import (
    VerificationContext,
    VerificationReport,
    VerificationResult,
    VerificationTarget,
)
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge


class VerificationLoop:
    """
    Compare candidate vs baseline utility trends and enforce L4-V gates.

    On failure optionally invokes AdaptationExecutor.rollback and blocks loop kind.
    """

    def __init__(
        self,
        *,
        signal_store: SignalStore,
        profile_store: ProfileVersionStore,
        executor: AdaptationExecutor | None = None,
        governance_bridge: RuntimeArchitectureGovernanceBridge | None = None,
        pointer_store: ProfileActivePointerStore | None = None,
        block_store: LoopApplyBlockStore | None = None,
        security_checker: SecurityAdversarialBaselineChecker | None = None,
    ) -> None:
        self._signal_store = signal_store
        self._profile_store = profile_store
        self._executor = executor
        self._governance_bridge = governance_bridge
        self._pointer_store = pointer_store
        self._block_store = block_store
        self._security_checker = security_checker or HarnessSecurityAdversarialBaselineChecker()

    def verify_target(
        self,
        target: VerificationTarget,
        *,
        context: VerificationContext,
    ) -> VerificationResult:
        since = datetime.now(UTC) - timedelta(days=context.window_days)
        signals = self._signal_store.list_signals(
            tenant_id=target.tenant_id,
            since=since,
            limit=2000,
        )
        scoped = [item for item in signals if item.task_class == target.task_class]
        candidate_signals, baseline_signals = split_candidate_baseline_signals(scoped)

        checks = [
            check_utility_trend(
                candidate_signals=candidate_signals,
                baseline_signals=baseline_signals,
                context=context,
            ),
            check_eval_registry_trend(
                evaluation_trend=context.evaluation_trend,
                context=context,
            ),
            check_regression_rate(
                candidate_signals=candidate_signals,
                baseline_signals=baseline_signals,
                context=context,
            ),
            check_cost_budget(
                candidate_signals=candidate_signals,
                budget_envelopes=context.budget_envelopes,
                context=context,
            ),
            check_security_adversarial(self._security_checker),
        ]
        passed = all(item.passed for item in checks)
        failure_reasons = [item.detail for item in checks if not item.passed]
        result = VerificationResult(
            target=target,
            passed=passed,
            checks=checks,
            failure_reasons=failure_reasons,
        )
        if passed or not context.auto_rollback_enabled:
            return result

        return self._remediate_failure(result, context=context)

    def verify_active_profiles(
        self,
        *,
        context: VerificationContext,
        tenant_id: str | None = None,
    ) -> VerificationReport:
        """Verify all canary and active profile versions (W-ADAPT-5.12 entry point)."""
        targets = self._collect_verification_targets(tenant_id=tenant_id)
        results = [self.verify_target(target, context=context) for target in targets]
        rollback_count = sum(1 for item in results if item.rolled_back)
        blocked_kinds = sorted(
            {
                item.blocked_loop_kind.value
                for item in results
                if item.blocked_loop_kind is not None
            }
        )
        return VerificationReport(
            results=results,
            passed=all(item.passed for item in results) if results else True,
            rollback_count=rollback_count,
            blocked_loop_kinds=blocked_kinds,
        )

    def _collect_verification_targets(
        self,
        *,
        tenant_id: str | None,
    ) -> list[VerificationTarget]:
        targets: list[VerificationTarget] = []
        for status in (ProfileVersionStatus.CANARY, ProfileVersionStatus.ACTIVE):
            versions = self._profile_store.list_versions(
                tenant_id=tenant_id,
                status=status,
                limit=200,
            )
            for record in versions:
                targets.append(
                    VerificationTarget(
                        tenant_id=record.tenant_id,
                        task_class=record.task_class,
                        artifact_type=record.artifact_type,
                        candidate_version_id=record.version_id,
                        loop_id=record.created_by if record.created_by.startswith("prop_") else None,
                    )
                )
        return targets

    def _remediate_failure(
        self,
        result: VerificationResult,
        *,
        context: VerificationContext,
    ) -> VerificationResult:
        target = result.target
        rolled_back = False
        blocked_kind = target.loop_kind

        if self._governance_bridge is not None and self._executor is not None and self._pointer_store is not None:
            service_principal = context.auto_rollback_service_principal
            rollback_mutation_id = context.auto_rollback_mutation_id
            if service_principal is None or rollback_mutation_id is None:
                rolled_back = False
            else:
                try:
                    self._governance_bridge.rollback_profile(
                        executor=self._executor,
                        pointer_store=self._pointer_store,
                        principal=service_principal,
                        mutation_id=rollback_mutation_id,
                        tenant_id=target.tenant_id,
                        task_class=target.task_class,
                        artifact_type=target.artifact_type,
                    )
                    rolled_back = True
                except (ValueError, AhiGovernanceBlockedError, ProfileActivePointerConflictError):
                    rolled_back = False

        if blocked_kind is not None and self._block_store is not None:
            self._block_store.block(
                blocked_kind,
                reason="; ".join(result.failure_reasons) or "verification_failed",
                tenant_id=target.tenant_id,
            )

        return result.model_copy(
            update={
                "rolled_back": rolled_back,
                "blocked_loop_kind": blocked_kind,
            }
        )
