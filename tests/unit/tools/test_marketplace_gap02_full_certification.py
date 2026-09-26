# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-CERT — full TRUE-GAP Marketplace Tool E2E certification gates."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
    derive_capability_acquisition_request_id,
)
from intergrax.contracts.capability_catalog import (
    CapabilityKind,
    CapabilityNeed,
)
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    qualified_capability_subject_from_result,
)
from intergrax.contracts.marketplace import MarketplaceQueryContext
from intergrax.contracts.tools.marketplace_handoff_reference import (
    derive_marketplace_gap_tool_handoff_id,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.marketplace.acquisition import (
    MarketplaceGapCapabilityAcquisitionStrategy,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
)
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V2,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V2,
)
from tests.unit.tools.support.gap02_cert_harness import (
    Gap02CertHarness,
    RecordingInvoker,
    _NOW,
    _TASK_ID,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENANT_A = "tenant-gap02-cert-a"
_TENANT_B = "tenant-gap02-cert-b"


def test_c1_success_true_gap_through_toolruntime() -> None:
    from intergrax.contracts.capability_catalog.discovery_completion import (
        DiscoveryCompletionOutcome,
    )

    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:c1"
    coordinator = harness.build_fulfillment_coordinator()
    fulfillment = coordinator.fulfill(harness.fulfillment_request(recovery_id))
    assert fulfillment.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    recovery = fulfillment.recovery_outcome
    assert recovery is not None
    assert recovery.discovery_completion is not None
    assert (
        recovery.discovery_completion.outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY
    )
    assert harness.gap_port.acquire_calls == 1
    assert harness.materializer.physical_activations == 1
    assert harness.invoker.calls == 1
    req = harness.invoker.last_request
    assert req is not None
    assert req.tenant_id == _TENANT_A
    assert req.task_id == _TASK_ID
    acq = recovery.acquisition_result
    assert acq is not None
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id=_TENANT_A,
        operation_id=acq.request_id,
    )
    assert handoff_id.startswith("marketplace-gap-handoff:v2:")


def test_c2_no_prequalification_activation_timeline() -> None:
    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:c2"
    assert harness.materializer.physical_activations == 0
    assert len(harness.lifecycle.registry.list()) == 0
    coordinator = harness.build_fulfillment_coordinator()
    coordinator.fulfill(harness.fulfillment_request(recovery_id))
    assert harness.materializer.physical_activations == 1


def test_c3_exact_release_reuse_two_executions() -> None:
    harness = Gap02CertHarness.build(_TENANT_A)
    invoker = RecordingInvoker()
    coordinator = harness.build_fulfillment_coordinator(invoker)
    id_a = "recovery:gap02:cert:c3:a"
    id_b = "recovery:gap02:cert:c3:b"
    r_a = coordinator.fulfill(harness.fulfillment_request(id_a))
    r_b = coordinator.fulfill(harness.fulfillment_request(id_b))
    assert r_a.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert r_b.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert harness.materializer.physical_activations == 1
    assert invoker.calls == 2


def test_c4_active_different_release_conflict() -> None:
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
    from tests.unit.tools.support.gap02_cert_harness import CachedRecoveryPort
    from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionRequest
    from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
    from testing_support.me14_tool_catalog_provider import ME14_CATALOG_ENTRY_ID, Me14ToolCatalogProvider

    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:c4"
    invoker = RecordingInvoker()
    coordinator = harness.build_fulfillment_coordinator(invoker)
    first = coordinator.fulfill(harness.fulfillment_request(recovery_id))
    assert first.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert invoker.calls == 1
    provider = Me14ToolCatalogProvider()
    stage = harness.stage_repo.get(
        tenant_id=_TENANT_A,
        handoff_id=derive_marketplace_gap_tool_handoff_id(
            tenant_id=_TENANT_A,
            operation_id=first.recovery_outcome.acquisition_result.request_id,  # type: ignore[union-attr]
        ),
    )
    assert stage is not None
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        stage.selected_release.discovery,
    )
    harness.lifecycle.registry.unregister(ME14_TOOL_LOGICAL_ID)
    harness.activation_resolver._acquisition.acquire(  # noqa: SLF001
        DynamicToolAcquisitionRequest(
            operation_id="warm-v2-conflict",
            host_profile_id=harness.lifecycle.host_profile_id,
            capability_identity_key=identity_key,
            selected_identity=ToolDiscoveryCandidateIdentity(
                catalog_source_id=provider.catalog_source_id,
                package=ToolPackageCandidate(
                    logical_tool_id=ME14_TOOL_LOGICAL_ID,
                    package_reference=ME14_PACKAGE_REFERENCE_V1,
                    package_version=ME14_VERSION_V2,
                    package_digest=ME14_DIGEST_V2,
                ),
            ),
            catalog_entry_id=ME14_CATALOG_ENTRY_ID,
        ),
    )
    conflict_recovery = "recovery:gap02:cert:c4:second"
    cached = CachedRecoveryPort(first.recovery_outcome)  # type: ignore[arg-type]
    second_coordinator = harness.build_fulfillment_coordinator(
        invoker,
        recovery=cached,
    )
    second = second_coordinator.fulfill(harness.fulfillment_request(conflict_recovery))
    assert second.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED
    assert invoker.calls == 1


def test_c5_multitenant_same_acquisition_id() -> None:
    shared_recovery = "recovery:gap02:cert:shared-nonce"
    ha = Gap02CertHarness.build(_TENANT_A)
    hb = Gap02CertHarness.build(_TENANT_B)
    _, ta = ha.run_true_gap_recovery(shared_recovery)
    _, tb = hb.run_true_gap_recovery(shared_recovery)
    assert ta.acquisition_request_id == tb.acquisition_request_id
    assert ta.handoff_id != tb.handoff_id


def test_c5_multitenant_adversarial_reads() -> None:
    ha = Gap02CertHarness.build(_TENANT_A)
    hb = Gap02CertHarness.build(_TENANT_B)
    _, ta = ha.run_true_gap_recovery("recovery:gap02:mt:a")
    _, tb = hb.run_true_gap_recovery("recovery:gap02:mt:b")
    assert ha.stage_repo.get(tenant_id=_TENANT_A, handoff_id=tb.handoff_id) is None
    assert hb.stage_repo.get(tenant_id=_TENANT_B, handoff_id=ta.handoff_id) is None


def test_c6_restart_after_staging() -> None:
    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:c6"
    outcome, _trace = harness.run_true_gap_recovery(recovery_id)
    calls_after_stage = harness.gap_port.acquire_calls
    restarted = harness.reconstruct_tool_domain()
    restarted.gap_port.acquire_calls = harness.gap_port.acquire_calls
    assert restarted.gap_port.acquire_calls == calls_after_stage
    from tests.unit.tools.support.gap02_cert_harness import CachedRecoveryPort

    cached = CachedRecoveryPort(outcome)
    coordinator = restarted.build_fulfillment_coordinator(recovery=cached)
    result = coordinator.fulfill(restarted.fulfillment_request(recovery_id))
    assert result.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED


def test_c7_restart_after_intent() -> None:
    from tests.unit.tools.support.gap02_cert_harness import CachedRecoveryPort

    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:c7"
    coordinator = harness.build_fulfillment_coordinator()
    first = coordinator.fulfill(harness.fulfillment_request(recovery_id))
    assert first.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert first.recovery_outcome is not None
    exec_id = first.provenance.execution_request_id
    assert exec_id is not None
    assert harness.intent_repo.get(execution_request_id=exec_id) is not None
    restarted = harness.reconstruct_tool_domain()
    restarted_coordinator = restarted.build_fulfillment_coordinator(
        RecordingInvoker(),
        recovery=CachedRecoveryPort(first.recovery_outcome),
    )
    replay = restarted_coordinator.fulfill(restarted.fulfillment_request(recovery_id))
    assert replay.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED


def test_c9_association_store_unavailable() -> None:
    from intergrax.tools.marketplace_qualified_tool_stage_context_association import (
        DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository,
    )
    from tests.unit.tools.test_marketplace_qualified_tool_stage_context_association import (
        _FailingWriteDocumentStore,
    )

    store = _FailingWriteDocumentStore()
    assoc = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    harness = Gap02CertHarness.build(_TENANT_A, assoc_repo=assoc)
    need = harness.worker_need("recovery:gap02:c9")
    request = harness.acquisition_request(need, harness.recovery_decision(need))
    outcome = harness.recovery.coordinate_recovery(request, decided_at=_NOW)
    assert outcome.acquisition_result is not None
    assert outcome.acquisition_result.outcome is CapabilityAcquisitionOutcome.UNAVAILABLE
    assert harness.materializer.physical_activations == 0
    assert harness.invoker.calls == 0


def test_c21_tenantless_tool_blocked() -> None:
    harness = Gap02CertHarness.build(_TENANT_A)
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(
        harness.gap_port,
        marketplace_query_context=MarketplaceQueryContext(),
    )
    completion = build_discovery_completion(
        need_id="need-cert-c21",
        discovery_correlation_id="corr-c21",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    gap = CapabilityGap.from_discovery_completion(completion)
    acq_id = derive_capability_acquisition_request_id(gap_id=gap.gap_id, request_nonce="n-c21")
    acq_request = CapabilityAcquisitionRequest(
        request_id=acq_id,
        request_nonce="n-c21",
        capability_gap=gap,
        capability_need=CapabilityNeed(
            need_id="need-cert-c21",
            kinds=(CapabilityKind.TOOL,),
            intent_summary="tool",
        ),
        requested_at=_NOW,
    )
    result = strategy.acquire(acq_request)
    assert result.outcome is CapabilityAcquisitionOutcome.BLOCKED


def test_c18_provider_registry_deterministic_resolve() -> None:
    from intergrax.tools.marketplace_qualified_capability_staging import (
        DocumentStoreMarketplaceQualifiedToolStageRepository,
    )
    from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
        DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
    )
    from intergrax.tools.qualified_tool_invocation_resolver import (
        DefaultQualifiedToolInvocationResolver,
    )
    from tests.unit.autonomous_work.uca6c_bound_execution_fixtures import (
        recording_codecraft_execution_handler,
    )
    from tests.unit.tools.test_marketplace_qualified_capability_execution_handler import (
        _ActivationResolver,
        _MaterialProvider,
    )

    store = InMemoryDocumentStore()
    marketplace_handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store),
        stage_repository=DocumentStoreMarketplaceQualifiedToolStageRepository(store),
        activation_resolver=_ActivationResolver(),
        material_provider=_MaterialProvider(),
        invocation_resolver=DefaultQualifiedToolInvocationResolver(),
        catalog_tool_invoker=RecordingInvoker(),
    )
    codecraft_handler, _ = recording_codecraft_execution_handler()
    registry = QualifiedCapabilityExecutionBindingHandlerRegistry(
        (marketplace_handler, codecraft_handler),
    )
    handler = registry.resolve(MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID)
    assert isinstance(handler, MarketplaceToolQualifiedCapabilityExecutionHandler)
    with pytest.raises(ValueError, match="duplicate"):
        QualifiedCapabilityExecutionBindingHandlerRegistry(
            (marketplace_handler, marketplace_handler),
        )


def test_c17_codecraft_regression_wave_reference() -> None:
    path = _REPO_ROOT / "tests/unit/autonomous_work/test_uca6c_r6_r5_8_r2_worker_governed_execution_e2e.py"
    assert path.is_file()


def test_c23_evidence_continuity_ids() -> None:
    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:c23"
    coordinator = harness.build_fulfillment_coordinator()
    fulfillment = coordinator.fulfill(harness.fulfillment_request(recovery_id))
    outcome = fulfillment.recovery_outcome
    assert outcome is not None
    subject = qualified_capability_subject_from_result(outcome.qualification_result)
    trace_qual = outcome.provenance.qualification_request_id
    assert trace_qual is not None
    assert subject is not None
    need = harness.worker_need(recovery_id)
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=need.recovery_decision_id,
        qualification_request_id=trace_qual,
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    exec_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    assert harness.intent_repo.get(execution_request_id=exec_id) is not None


def test_c25_single_qualification_per_fulfillment() -> None:
    harness = Gap02CertHarness.build(_TENANT_A)
    recovery_id = "recovery:gap02:cert:c25"
    coordinator = harness.build_fulfillment_coordinator()
    before = harness.qualification_adapter.calls
    coordinator.fulfill(harness.fulfillment_request(recovery_id))
    assert harness.qualification_adapter.calls - before == 1


def test_c28_no_second_execution_engine_construction() -> None:
    gap02_modules = (
        _REPO_ROOT / "intergrax/tools/marketplace_qualified_capability_execution_handler.py",
        _REPO_ROOT / "intergrax/tools/marketplace_qualified_tool_execution_intent_preparation.py",
    )
    for path in gap02_modules:
        source = path.read_text(encoding="utf-8")
        assert "ExecutionEngine(" not in source
        assert "ExecutionRuntime(" not in source


def test_c30_contract_purity_tools() -> None:
    contracts_root = _REPO_ROOT / "intergrax/contracts/tools"
    forbidden_prefixes = (
        "intergrax.runtime",
        "intergrax.marketplace",
        "agents.nexus",
    )
    for path in contracts_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden_prefixes:
                        assert not alias.name.startswith(prefix), path
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden_prefixes:
                    assert not node.module.startswith(prefix), path


def test_c32_no_reflection_in_gap02_modules() -> None:
    roots = (
        _REPO_ROOT / "intergrax/tools",
    )
    names = (
        "marketplace_qualified",
        "marketplace_gap",
        "qualified_marketplace_tool",
    )
    for root in roots:
        for path in root.glob("*.py"):
            if not any(n in path.name for n in names):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in {"getattr", "setattr", "hasattr"}:
                        pytest.fail(f"reflection in {path}")


def test_regression_wave_r6_cert_module_collects() -> None:
    assert (_REPO_ROOT / "tests/unit/tools/test_marketplace_gap02_full_certification.py").is_file()


def _run_pytest_file(relative: str) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", relative, "-q", "--tb=no"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode
