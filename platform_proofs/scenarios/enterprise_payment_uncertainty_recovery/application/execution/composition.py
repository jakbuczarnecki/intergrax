"""Lab execution stack — wires provisioning, application, external payment, and ERL."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.dependencies import (
    ApplicationDependencies,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.root import (
    ScenarioApplicationCompositionRoot,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.dependencies import (
    EnterprisePaymentExecutionDependencies,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.persistence_lookup_bridge import (
    PersistedExternalRealityLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution.runner import (
    EnterprisePaymentScenarioExecutor,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.observability import (
    RecordingScenarioApplicationObservability,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.application.references import (
    LabBusinessReferences,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_governance_context import (
    PaymentEnterpriseGovernancePolicy,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_governance_context_lookup import (
    InMemoryPaymentGovernanceBusinessContextLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_evidence_lookup import (
    InMemoryPaymentReconciliationEvidenceLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_recovery_action import (
    InMemoryPaymentRecoveryActionPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.wiring import (
    register_scenario_reconciliation_plugins,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.scenario_payment_workflow import (
    ScenarioExternalPaymentWorkflow,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.in_memory_persistence import (
    InMemoryExternalRealityStore,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.services.capture_service import (
    ExternalPaymentCaptureService,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.in_memory import (
    InMemoryReferenceProvisioner,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.adapters.lab_reference_order_access import (
    LabReferenceOrderAccess,
)

_SCENARIO_ROOT = Path(__file__).resolve().parents[2]
_DATASET_ROOT = _SCENARIO_ROOT / "dataset"
_DEFAULT_GOVERNANCE_POLICY = PaymentEnterpriseGovernancePolicy(
    human_approval_threshold_amount=Decimal("10000.00"),
    currency="EUR",
)


def build_lab_execution_composition(
    *,
    references: LabBusinessReferences | None = None,
    dataset_package_root: Path | None = None,
) -> EnterprisePaymentScenarioExecutor:
    """Construct a replaceable in-memory enterprise execution stack for proof runs."""
    refs = references or LabBusinessReferences()
    dataset_root = dataset_package_root or _DATASET_ROOT
    store = InMemoryExternalRealityStore()
    capture = ExternalPaymentCaptureService(store)
    payment_workflow = ScenarioExternalPaymentWorkflow(capture)
    observability = RecordingScenarioApplicationObservability()
    application_deps = ApplicationDependencies(
        order_access=LabReferenceOrderAccess(refs),
        payment_workflow=payment_workflow,
        observability=observability,
    )
    application_root = ScenarioApplicationCompositionRoot(application_deps)

    reality_lookup = PersistedExternalRealityLookup(store)
    payment_evidence_lookup = InMemoryPaymentReconciliationEvidenceLookup()
    governance_lookup = InMemoryPaymentGovernanceBusinessContextLookup()
    recovery_action_port = InMemoryPaymentRecoveryActionPort()

    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    register_scenario_reconciliation_plugins(
        registry,
        reality_lookup,
        payment_evidence_lookup=payment_evidence_lookup,
        payment_governance_lookup=governance_lookup,
        payment_governance_policy=_DEFAULT_GOVERNANCE_POLICY,
        payment_recovery_action_port=recovery_action_port,
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)

    deps = EnterprisePaymentExecutionDependencies(
        dataset_package_root=dataset_root,
        provisioning_port=InMemoryReferenceProvisioner(),
        application=application_deps,
        external_reality_store=store,
        reality_lookup=reality_lookup,
        payment_evidence_lookup=payment_evidence_lookup,
        payment_governance_lookup=governance_lookup,
        payment_recovery_action_port=recovery_action_port,
        erl_gateway=gateway,
    )
    return EnterprisePaymentScenarioExecutor(
        application_root=application_root,
        dependencies=deps,
        business_references=refs,
    )
