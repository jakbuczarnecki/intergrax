"""Orchestrates dataset → provisioning → application → ERL → proof result."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from intergrax.contracts.enterprise_reliability import ReconciliationExecutionDisposition

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.root import (
    ScenarioApplicationCompositionRoot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.scenario_context import (
    ScenarioExecutionContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.workflow import (
    BusinessWorkflowPhase,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.dependencies import (
    EnterprisePaymentExecutionDependencies,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.erl_reliability_phase import (
    run_enterprise_reliability_phase,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.persistence_lookup_bridge import (
    PersistedExternalRealityLookup,
    build_payment_evidence_lookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.request import (
    EnterprisePaymentScenarioExecutionRequest,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.result import (
    ScenarioExecutionProofResult,
    derive_lifecycle_outcome,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.application.references import (
    LabBusinessReferences,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.context import (
    ProvisioningContext,
    ProvisioningExecutionContext,
    ScenarioIdentity,
    ScenarioVariantSelection,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.lifecycle import (
    ProvisioningLifecycleCoordinator,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_evidence_lookup import (
    InMemoryPaymentReconciliationEvidenceLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_governance_context_lookup import (
    InMemoryPaymentGovernanceBusinessContextLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_governance_context import (
    PaymentGovernanceBusinessContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.domain.lifecycle import (
    ExternalPaymentLifecycleState,
)


class EnterprisePaymentScenarioExecutor:
    """Explicit, testable composition root for one enterprise payment proof run."""

    def __init__(
        self,
        *,
        application_root: ScenarioApplicationCompositionRoot,
        dependencies: EnterprisePaymentExecutionDependencies,
        business_references: LabBusinessReferences,
    ) -> None:
        self._application_root = application_root
        self._deps = dependencies
        self._references = business_references

    def execute(
        self,
        request: EnterprisePaymentScenarioExecutionRequest,
    ) -> ScenarioExecutionProofResult:
        scenario_id = "ERL-QUAL-004"
        correlation_id = self._references.payment_intent_reference
        provisioning_context = ProvisioningContext(
            identity=ScenarioIdentity(
                qualification_id=scenario_id,
                scenario_slug="enterprise_payment_uncertainty_recovery",
            ),
            variant=ScenarioVariantSelection(variant_id=request.variant_id),
            execution=ProvisioningExecutionContext(
                run_id=request.run_id,
                dataset_package_root=self._deps.dataset_package_root,
            ),
        )
        coordinator = ProvisioningLifecycleCoordinator()
        provisioning_run = coordinator.run(
            self._deps.provisioning_port,
            provisioning_context,
            invoke_cleanup=False,
        )
        if provisioning_run.status is not ProvisioningOutcomeStatus.SUCCEEDED:
            return ScenarioExecutionProofResult(
                scenario_id=scenario_id,
                variant_id=request.variant_id,
                correlation_id=correlation_id,
                lifecycle_outcome=derive_lifecycle_outcome(
                    provisioning_status=provisioning_run.status,
                    application_phase=None,
                    recovery=None,
                ),
                provisioning_status=provisioning_run.status,
                application_phase=None,
                evidence_ref=None,
                evidence_evaluation_outcome=None,
                reconciliation_probe_verdict=None,
                resolution_result=None,
                governance_result=None,
                recovery_result=None,
            )

        execution_context = ScenarioExecutionContext(
            scenario_id=scenario_id,
            scenario_slug="enterprise_payment_uncertainty_recovery",
            variant_id=request.variant_id,
            execution_reference=f"exec-{request.run_id}-{uuid4().hex[:8]}",
            correlation_ids={
                "order_logical_id": self._references.logical_order_id,
                "payment_correlation_id": correlation_id,
            },
        )
        application_run = self._application_root.run(execution_context)
        application_phase = application_run.workflow_outcome.phase

        latest = self._deps.external_reality_store.latest()
        if latest is None or latest.bundle.lifecycle_state is not ExternalPaymentLifecycleState.UNKNOWN:
            return ScenarioExecutionProofResult(
                scenario_id=scenario_id,
                variant_id=request.variant_id,
                correlation_id=correlation_id,
                lifecycle_outcome=derive_lifecycle_outcome(
                    provisioning_status=provisioning_run.status,
                    application_phase=application_phase,
                    recovery=None,
                ),
                provisioning_status=provisioning_run.status,
                application_phase=application_phase,
                evidence_ref=None,
                evidence_evaluation_outcome=None,
                reconciliation_probe_verdict=None,
                resolution_result=None,
                governance_result=None,
                recovery_result=None,
            )

        reality_lookup = self._deps.reality_lookup
        if not isinstance(reality_lookup, PersistedExternalRealityLookup):
            raise TypeError("execution stack requires PersistedExternalRealityLookup")
        snapshot = reality_lookup.refresh_from_persistence(correlation_id)

        payment_lookup = self._deps.payment_evidence_lookup
        if not isinstance(payment_lookup, InMemoryPaymentReconciliationEvidenceLookup):
            raise TypeError("execution stack requires InMemoryPaymentReconciliationEvidenceLookup")
        seeded = build_payment_evidence_lookup(
            dataset_package_root=self._deps.dataset_package_root,
            variant_id=request.variant_id,
            snapshot=snapshot,
            observed_at=datetime.now(tz=UTC),
        )
        payment_lookup.seed(seeded.lookup_by_correlation_id(correlation_id))

        governance_lookup = self._deps.payment_governance_lookup
        if isinstance(governance_lookup, InMemoryPaymentGovernanceBusinessContextLookup):
            governance_lookup.seed(
                PaymentGovernanceBusinessContext(
                    correlation_id=correlation_id,
                    payment_amount=application_run.workflow_outcome.order.amount,
                    currency=application_run.workflow_outcome.order.currency,
                ),
            )

        erl_result = run_enterprise_reliability_phase(
            correlation_id=correlation_id,
            tenant_id=request.tenant_id,
            gateway=self._deps.erl_gateway,
            payment_evidence_lookup=payment_lookup,
        )
        reconciliation_run = erl_result.reconciliation_run
        probe_verdict = None
        if (
            reconciliation_run.execution.disposition
            is ReconciliationExecutionDisposition.PROBE_EXECUTED
            and reconciliation_run.execution.probe_result is not None
        ):
            probe_verdict = reconciliation_run.execution.probe_result.verdict

        evidence_ref = reconciliation_run.evidence.evidence_ref if reconciliation_run.evidence else None

        return ScenarioExecutionProofResult(
            scenario_id=scenario_id,
            variant_id=request.variant_id,
            correlation_id=correlation_id,
            lifecycle_outcome=derive_lifecycle_outcome(
                provisioning_status=provisioning_run.status,
                application_phase=BusinessWorkflowPhase.PAYMENT_REQUESTED,
                recovery=erl_result.recovery_result,
            ),
            provisioning_status=provisioning_run.status,
            application_phase=application_phase,
            evidence_ref=evidence_ref,
            evidence_evaluation_outcome=erl_result.evidence_evaluation_outcome,
            reconciliation_probe_verdict=probe_verdict,
            resolution_result=erl_result.resolution_result,
            governance_result=erl_result.governance_result,
            recovery_result=erl_result.recovery_result,
        )
