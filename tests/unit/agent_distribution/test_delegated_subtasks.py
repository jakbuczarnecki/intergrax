# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.agent_distribution.activation import (
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.admin_models import SetAgentEnablementRequest
from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryContractError,
    AgentDiscoveryRequest,
    AgentDiscoveryResult,
    AgentDiscoveryStrategy,
    AgentDiscoveryStrategyId,
    StaticAgentDiscoveryStrategy,
    project_package_contract_capabilities,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
)
from intergrax.agent_distribution.agent_selection import (
    DeterministicIdentitySelectionStrategy,
)
from intergrax.agent_distribution.capability_matching import CapabilityMatcher
from intergrax.agent_distribution.catalog import (
    AgentDiscoveryCandidateIdentity,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskAcquisitionError,
    DelegatedSubtaskCleanupError,
    DelegatedSubtaskDelegate,
    DelegatedSubtaskExecutionAndReleaseError,
    DelegatedSubtaskInvocation,
    DelegatedSubtaskInvocationError,
    DelegatedSubtaskLifecyclePlan,
    DelegatedSubtaskNoEligibleAgent,
    DelegatedSubtaskReleaseContext,
    DelegatedSubtaskRequest,
    DelegatedSubtaskResolutionError,
    DelegatedSubtaskService,
    DelegationId,
    SpecialistInvocationPort,
)
from intergrax.agent_distribution.dynamic_acquisition import (
    DynamicAgentAcquisitionResult,
)
from intergrax.agent_distribution.federated_discovery import (
    FederatedAgentDiscoveryStrategy,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate
from intergrax.agent_distribution.task_capability_resolution import (
    build_deterministic_task_capability_resolver,
    build_task_capability_resolution_request,
    build_task_capability_rule,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentLease,
    TaskScopedAgentLeaseId,
    TaskScopedAgentLeaseState,
    TaskScopedAgentReleaseOutcome,
    TaskScopedAgentReleaseRequest,
    TaskScopedAgentService,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    RunId,
    AttemptId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    peek_active_parent_execution_id,
    require_active_execution_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.delegated_subtask_child_port import (
    as_child_execution_port,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from tests.unit.agent_distribution.test_task_scoped_agents import (
    _APP,
    _BINDING_ID,
    _DIGEST,
    _ENV,
    _PACKAGE_ID,
    _activate_request,
    _bind_request,
    _binding_revision,
    _build_request,
    _install_request,
    _release_request,
    _task_acquire_request,
    admin_test_principal,
    build_task_scoped_harness,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SOURCE_ID = "builtin-1"
_CATALOG_ENTRY_ID = "cat-researcher"
_OCR_PACKAGE = _PACKAGE_ID
_LEGAL_PACKAGE = "legal-agent"
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


@dataclass(frozen=True)
class OcrRequest:
    document_ref: str


@dataclass(frozen=True)
class OcrResult:
    text: str


def _source() -> CatalogSourceIdentity:
    return CatalogSourceIdentity(
        catalog_source_id=_SOURCE_ID,
        provider_kind=CatalogProviderKind.BUILTIN,
    )


def _candidate(package_id: str) -> AgentPackageCandidate:
    return AgentPackageCandidate(
        distribution_package_id=package_id,
        package_version="1.0.0",
        package_digest=_DIGEST,
    )


def _identity(package_id: str) -> AgentDiscoveryCandidateIdentity:
    return AgentDiscoveryCandidateIdentity(
        source=_source(), package=_candidate(package_id)
    )


def _discovery_candidate(
    package_id: str,
    *,
    capability_ids: tuple[str, ...],
) -> AgentDiscoveryCandidate:
    return AgentDiscoveryCandidate(
        identity=_identity(package_id),
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=capability_ids,
            ),
        ),
        catalog_entry_id=_CATALOG_ENTRY_ID,
        artifact_locator=f"catalog://artifact/{package_id}",
    )


def _baseline_resolver():
    return build_deterministic_task_capability_resolver(
        rules=(
            build_task_capability_rule(
                rule_id="rule.document.ocr.v1",
                task_kind="document.ocr",
                required=("document.ocr",),
            ),
            build_task_capability_rule(
                rule_id="rule.document.legal_compare.v1",
                task_kind="document.legal_compare",
                required=("document.read", "legal.analysis", "document.compare"),
                optional=("citation.generate",),
            ),
        ),
    )


def _federated_discovery(
    *candidates: AgentDiscoveryCandidate,
) -> FederatedAgentDiscoveryStrategy:
    return FederatedAgentDiscoveryStrategy(
        strategies=(
            StaticAgentDiscoveryStrategy(
                strategy_id=AgentDiscoveryStrategyId(value="static.test"),
                candidates=candidates,
            ),
        ),
    )


class _TestAcquisitionPlanFactory:
    def __init__(self, *, revision_id: str, **kwargs: object) -> None:
        self._revision_id = revision_id
        self._kwargs = kwargs
        self._harness: DelegatedHarness | None = None

    def bind_harness(self, harness: DelegatedHarness) -> None:
        self._harness = harness

    def build_acquisition_plan(
        self,
        *,
        delegation_id: DelegationId,
        task_scope_id,
        application_id: str,
        application_environment_id: str,
        lease_id: TaskScopedAgentLeaseId,
        selected_identity: AgentDiscoveryCandidateIdentity,
    ) -> DelegatedSubtaskLifecyclePlan:
        del delegation_id, application_id, application_environment_id
        prior_revision_id = None
        pointer_revision = 0
        if self._harness is not None:
            serving = self._harness.stack.service.inspect_serving(
                application_id=_APP,
                application_environment_id=_ENV,
            )
            prior_revision_id = serving.traffic_serving_revision_id
            pointer_revision = serving.serving_pointer_revision
        return DelegatedSubtaskLifecyclePlan(
            acquisition_request=_task_acquire_request(
                str(lease_id),
                task_scope_id,
                self._revision_id,
                identity=selected_identity,
                prior_revision_id=prior_revision_id,
                pointer_revision=pointer_revision,
                **self._kwargs,
            ),
        )


class _TestReleasePlanFactory:
    def __init__(self, harness) -> None:
        self._harness = harness

    def build_release_request(
        self,
        *,
        context: DelegatedSubtaskReleaseContext,
    ) -> TaskScopedAgentReleaseRequest:
        serving = self._harness.stack.service.inspect_serving(
            application_id=_APP,
            application_environment_id=_ENV,
        )
        prior_revision_id = serving.traffic_serving_revision_id
        revision_id = f"rev-release-{context.lease.lease_id}"
        return _release_request(
            str(context.lease.lease_id),
            context.task_scope_id,
            revision_id,
            disable_revision=_binding_revision(self._harness.stack),
            prior_revision_id=prior_revision_id,
            pointer_revision=serving.serving_pointer_revision,
        )


@dataclass
class _StaticSpecialistInvocation(SpecialistInvocationPort[OcrRequest, OcrResult]):
    delegate: DelegatedSubtaskDelegate[OcrRequest, OcrResult]

    def resolve_delegate(
        self,
        *,
        lease: TaskScopedAgentLease,
        acquisition_result: DynamicAgentAcquisitionResult,
    ) -> DelegatedSubtaskDelegate[OcrRequest, OcrResult]:
        del lease, acquisition_result
        return self.delegate


@dataclass
class DelegatedHarness:
    task_scoped: TaskScopedAgentService
    service: DelegatedSubtaskService[OcrRequest, OcrResult]
    stack: object
    lease_store: object


def build_delegated_harness(
    *,
    candidates: tuple[AgentDiscoveryCandidate, ...],
    revision_id: str = "rev-delegate-1",
    specialist_delegate: DelegatedSubtaskDelegate[OcrRequest, OcrResult] | None = None,
    acquisition_kwargs: dict | None = None,
) -> DelegatedHarness:
    harness = build_task_scoped_harness()
    delegate = specialist_delegate or _EchoOcrDelegate()
    acquisition_factory = _TestAcquisitionPlanFactory(
        revision_id=revision_id,
        **(acquisition_kwargs or {}),
    )
    service = DelegatedSubtaskService(
        capability_resolver=_baseline_resolver(),
        discovery=_federated_discovery(*candidates),
        matcher=CapabilityMatcher(),
        selector=DeterministicIdentitySelectionStrategy(),
        task_scoped_agents=harness.service,
        acquisition_plan_factory=acquisition_factory,
        release_plan_factory=_TestReleasePlanFactory(harness),
        specialist_invocation=_StaticSpecialistInvocation(delegate=delegate),
        child_execution=as_child_execution_port(
            ChildExecutionRunner[OcrRequest, OcrResult](ledger=_UNLIMITED_LEDGER),
        ),
    )
    delegated = DelegatedHarness(
        task_scoped=harness.service,
        service=service,
        stack=harness.stack,
        lease_store=harness.lease_store,
    )
    acquisition_factory.bind_harness(delegated)
    return delegated


class _EchoOcrDelegate:
    async def execute(self, request: OcrRequest) -> OcrResult:
        return OcrResult(text=f"ocr:{request.document_ref}")


def _delegated_request(
    *,
    task_scope,
    delegation_id: str = "delegation-1",
    lease_id: str = "lease-delegate-1",
    task_kind: str = "document.ocr",
) -> DelegatedSubtaskRequest:
    return DelegatedSubtaskRequest(
        delegation_id=DelegationId(delegation_id),
        task_scope_id=task_scope,
        application_id=_APP,
        application_environment_id=_ENV,
        lease_id=TaskScopedAgentLeaseId(lease_id),
        capability_resolution_request=build_task_capability_resolution_request(
            task_kind=task_kind,
        ),
    )


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


async def _run_delegation(
    harness: DelegatedHarness,
    *,
    task_scope,
    delegate: DelegatedSubtaskDelegate[OcrRequest, OcrResult] | None = None,
    document_ref: str = "doc-1",
    authority: ParentExecutionAuthority | None = None,
    delegation_id: str = "delegation-1",
    lease_id: str = "lease-delegate-1",
):
    if delegate is not None:
        harness.service._specialist_invocation = _StaticSpecialistInvocation(
            delegate=delegate,
        )
    root = _root_identity()
    captured: list[object] = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            result = await harness.service.execute(
                _delegated_request(
                    task_scope=task_scope,
                    delegation_id=delegation_id,
                    lease_id=lease_id,
                ),
                invocation=DelegatedSubtaskInvocation(payload=request),
                principal=admin_test_principal(),
            )
            captured.append(result)
            return result.result

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=authority or ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref=document_ref))
    assert captured
    return captured[0]


@pytest.mark.asyncio
async def test_delegated_subtask_happy_path_document_ocr() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(
                _LEGAL_PACKAGE,
                capability_ids=("legal.analysis", "document.read"),
            ),
        ),
    )
    task_scope = mint_task_id()
    result = await _run_delegation(harness, task_scope=task_scope)
    assert result.result.text == "ocr:doc-1"
    assert result.selected_identity.package.distribution_package_id == _OCR_PACKAGE
    assert result.release_result.outcome in {
        TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RUNTIME_UPDATED,
        TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RETAINED_BINDING,
    }
    lease = harness.lease_store.get(TaskScopedAgentLeaseId("lease-delegate-1"))
    assert lease is not None
    assert lease.lease_state is TaskScopedAgentLeaseState.RELEASED


@pytest.mark.asyncio
async def test_delegated_subtask_execution_lineage() -> None:
    parent_execution_id: ExecutionId | None = None
    parent_run_id: RunId | None = None
    parent_attempt_id: AttemptId | None = None
    child_execution_id: ExecutionId | None = None
    child_parent_execution_id: ExecutionId | None = None
    child_run_id: RunId | None = None
    child_attempt_id: AttemptId | None = None

    class LineageDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            nonlocal child_execution_id, child_parent_execution_id
            nonlocal child_run_id, child_attempt_id
            child_execution_id = require_active_execution_id()
            child_parent_execution_id = peek_active_parent_execution_id()
            identity = peek_active_execution_identity()
            assert identity is not None
            child_run_id, child_attempt_id = identity
            return OcrResult(text=request.document_ref)

    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=LineageDelegate(),
    )
    root = _root_identity()
    parent_execution_id = root.execution_id
    parent_run_id = root.run_id
    parent_attempt_id = root.attempt_id

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            result = await harness.service.execute(
                _delegated_request(task_scope=mint_task_id()),
                invocation=DelegatedSubtaskInvocation(payload=request),
                principal=admin_test_principal(),
            )
            return result.result

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="lineage"))

    assert child_execution_id is not None
    assert parent_execution_id is not None
    assert child_execution_id != parent_execution_id
    assert child_parent_execution_id == parent_execution_id
    assert child_run_id == parent_run_id
    assert child_attempt_id == parent_attempt_id


@pytest.mark.asyncio
async def test_delegated_subtask_no_eligible_agent() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(
                _LEGAL_PACKAGE,
                capability_ids=("legal.analysis",),
            ),
        ),
    )
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskNoEligibleAgent):
                await harness.service.execute(
                    _delegated_request(task_scope=mint_task_id()),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="none"))
    assert harness.lease_store.list_active_by_binding(_BINDING_ID) == ()


class _FailingDiscovery(AgentDiscoveryStrategy):
    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        return AgentDiscoveryStrategyId(value="failing.test")

    def discover(self, request: AgentDiscoveryRequest) -> AgentDiscoveryResult:
        del request
        raise AgentDiscoveryContractError("discovery failed")


@pytest.mark.asyncio
async def test_delegated_subtask_discovery_failure() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    harness.service._discovery = _FailingDiscovery()
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskResolutionError, match="discovery"):
                await harness.service.execute(
                    _delegated_request(task_scope=mint_task_id()),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="discover"))


class _FailingAcquisitionPort:
    def acquire(self, request, *, principal):
        del request, principal
        from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentError

        raise TaskScopedAgentError("acquisition failed")


@pytest.mark.asyncio
async def test_delegated_subtask_acquisition_failure() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    harness.service._task_scoped_agents._acquisition_service._acquisition = (
        _FailingAcquisitionPort()
    )
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskAcquisitionError):
                await harness.service.execute(
                    _delegated_request(task_scope=mint_task_id()),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="acquire"))


class _FailingSpecialistDelegate:
    async def execute(self, request: OcrRequest) -> OcrResult:
        del request
        raise RuntimeError("specialist failed")


@pytest.mark.asyncio
async def test_delegated_subtask_child_execution_failure_releases_lease() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_FailingSpecialistDelegate(),
    )
    task_scope = mint_task_id()
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskInvocationError):
                await harness.service.execute(
                    _delegated_request(task_scope=task_scope),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="failed")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="child-fail"))
    lease = harness.lease_store.get(TaskScopedAgentLeaseId("lease-delegate-1"))
    assert lease is not None
    assert lease.lease_state is TaskScopedAgentLeaseState.RELEASED


@pytest.mark.asyncio
async def test_delegated_subtask_release_failure_after_success() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    coordinator = FakeRuntimeServingProjectionCoordinator()
    coordinator.fail_prepare("rev-release-lease-delegate-1")
    harness.stack.service._activation_service._projection_coordinator = coordinator
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskCleanupError) as exc_info:
                await harness.service.execute(
                    _delegated_request(task_scope=mint_task_id()),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            assert exc_info.value.result.text == "ocr:doc-1"
            return OcrResult(text="cleanup-failed")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="doc-1"))


@pytest.mark.asyncio
async def test_delegated_subtask_child_and_release_failure() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_FailingSpecialistDelegate(),
    )
    coordinator = FakeRuntimeServingProjectionCoordinator()
    coordinator.fail_prepare("rev-release-lease-delegate-1")
    harness.stack.service._activation_service._projection_coordinator = coordinator
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskExecutionAndReleaseError) as exc_info:
                await harness.service.execute(
                    _delegated_request(task_scope=mint_task_id()),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            assert isinstance(exc_info.value.execution_cause, RuntimeError)
            assert exc_info.value.release_cause is not None
            return OcrResult(text="both-failed")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="both"))


@pytest.mark.asyncio
async def test_delegated_subtask_persistent_specialist_release_retains_binding() -> (
    None
):
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        revision_id="rev-persistent",
    )
    principal = admin_test_principal()
    harness.stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
        principal=principal,
    )
    harness.stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(),
        principal=principal,
    )
    harness.stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id=_BINDING_ID,
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable-persistent",
            expected_revision=0,
        ),
        principal=principal,
    )
    harness.stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-persistent"),
        principal=principal,
    )
    harness.stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_activate_request("rev-persistent"),
        principal=principal,
    )
    await _run_delegation(
        harness,
        task_scope=mint_task_id(),
        document_ref="persistent",
    )
    binding = harness.stack.service._binding_store.get_binding(_BINDING_ID)
    assert binding is not None
    assert binding.enablement is True


@pytest.mark.asyncio
async def test_delegated_subtask_shared_specialist_two_leases() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        revision_id="rev-shared",
    )
    task_a = mint_task_id()
    task_b = mint_task_id()
    await _run_delegation(
        harness,
        task_scope=task_a,
        document_ref="shared-a",
    )
    harness.service._acquisition_plan_factory = _TestAcquisitionPlanFactory(
        revision_id="rev-shared-2",
        install_mutation_id="mut-install-b",
        bind_mutation_id="mut-bind-b",
    )
    harness.service._acquisition_plan_factory.bind_harness(harness)
    await _run_delegation(
        harness,
        task_scope=task_b,
        document_ref="shared-b",
        delegate=_EchoOcrDelegate(),
        lease_id="lease-delegate-2",
        delegation_id="delegation-2",
    )
    assert len(harness.lease_store.list_active_by_binding(_BINDING_ID)) == 0


@pytest.mark.asyncio
async def test_delegated_subtask_recursive_delegation() -> None:
    inner_harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        revision_id="rev-inner",
    )

    class OuterDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            inner = await inner_harness.service.execute(
                _delegated_request(
                    task_scope=mint_task_id(),
                    delegation_id="delegation-inner",
                    lease_id="lease-inner",
                ),
                invocation=DelegatedSubtaskInvocation(payload=request),
                principal=admin_test_principal(),
            )
            return OcrResult(text=f"outer:{inner.result.text}")

    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        revision_id="rev-outer",
        specialist_delegate=OuterDelegate(),
    )
    result = await _run_delegation(
        harness, task_scope=mint_task_id(), document_ref="recursive"
    )
    assert result.result.text == "outer:ocr:recursive"


@pytest.mark.asyncio
async def test_delegated_subtask_authority_escalation_rejected() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskInvocationError) as exc_info:
                await harness.service.execute(
                    _delegated_request(task_scope=mint_task_id()),
                    invocation=DelegatedSubtaskInvocation(
                        payload=request,
                        requested_permission_scopes=("read", "delete"),
                    ),
                    principal=admin_test_principal(),
                )
            from intergrax.contracts.delegation_authority import (
                DelegationAuthorityError,
            )

            assert isinstance(exc_info.value.__cause__, DelegationAuthorityError)
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.scoped(("read", "write")),
    ).execute(OcrRequest(document_ref="authority"))


@pytest.mark.asyncio
async def test_delegated_subtask_budget_escalation_rejected() -> None:
    from intergrax.runtime.execution.budget.models import (
        ExecutionBudgetReservationError,
    )

    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=50))
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    harness.service._child_execution = as_child_execution_port(
        ChildExecutionRunner[OcrRequest, OcrResult](ledger=ledger),
    )
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=ledger,
            )
            try:
                with pytest.raises(DelegatedSubtaskInvocationError) as exc_info:
                    await harness.service.execute(
                        _delegated_request(task_scope=mint_task_id()),
                        invocation=DelegatedSubtaskInvocation(
                            payload=request,
                            requested_budget=RunBudget(max_tool_calls=70),
                        ),
                        principal=admin_test_principal(),
                    )
            finally:
                reset_active_execution_budget(budget_token)
            assert isinstance(
                exc_info.value.__cause__,
                ExecutionBudgetReservationError,
            )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="budget"))


def test_delegated_subtask_module_has_no_forbidden_imports() -> None:
    import intergrax.agent_distribution.delegated_subtasks as module

    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    forbidden = (
        "intergrax.runtime.nexus",
        "intergrax.runtime.execution.child",
        "intergrax.harness",
        "openai",
        "anthropic",
    )
    for module_name in imported_modules:
        for prefix in forbidden:
            assert not module_name.startswith(prefix), module_name
