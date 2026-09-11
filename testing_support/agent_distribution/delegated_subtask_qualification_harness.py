# © Artur Czarnecki. All rights reserved.

"""Reusable delegated-subtask qualification harness (OCR contract)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
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
    DelegatedSubtaskAcquisitionPlanFactory,
    DelegatedSubtaskDelegate,
    DelegatedSubtaskLifecyclePlan,
    DelegatedSubtaskReleaseContext,
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
    TaskCapabilityResolver,
    build_deterministic_task_capability_resolver,
    build_task_capability_rule,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentLease,
    TaskScopedAgentLeaseId,
    TaskScopedAgentReleaseRequest,
    TaskScopedAgentService,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_task_id,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.delegated_subtask_child_port import (
    as_child_execution_port,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from testing_support.agent_distribution.coordination_governance import (
    allowing_physical_delegation_governance,
)
from testing_support.agent_distribution.task_scoped_agent_qualification_harness import (
    QUALIFICATION_APPLICATION_ID,
    QUALIFICATION_CATALOG_ENTRY_ID,
    QUALIFICATION_CATALOG_SOURCE_ID,
    QUALIFICATION_ENVIRONMENT_ID,
    QUALIFICATION_PACKAGE_DIGEST,
    QUALIFICATION_PACKAGE_ID,
    TaskScopedAgentQualificationHarness,
    build_task_scoped_agent_qualification_harness,
    qualification_binding_revision,
    qualification_task_acquire_request,
    qualification_task_release_request,
)

OCR_QUALIFICATION_PACKAGE_ID = QUALIFICATION_PACKAGE_ID
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


@dataclass(frozen=True)
class OcrQualificationRequest:
    document_ref: str


@dataclass(frozen=True)
class OcrQualificationResult:
    text: str


@dataclass
class FixedTaskScopeAuthority:
    task_scope_id: TaskId
    resolve_count: int = 0

    def resolve_current_task_scope(
        self,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> TaskId:
        del run_id, attempt_id, execution_id
        self.resolve_count += 1
        return self.task_scope_id


def build_ocr_qualification_discovery_candidate(
    package_id: str,
    *,
    capability_ids: tuple[str, ...],
) -> AgentDiscoveryCandidate:
    source = CatalogSourceIdentity(
        catalog_source_id=QUALIFICATION_CATALOG_SOURCE_ID,
        provider_kind=CatalogProviderKind.BUILTIN,
    )
    package = AgentPackageCandidate(
        distribution_package_id=package_id,
        package_version="1.0.0",
        package_digest=QUALIFICATION_PACKAGE_DIGEST,
    )
    identity = AgentDiscoveryCandidateIdentity(source=source, package=package)
    return AgentDiscoveryCandidate(
        identity=identity,
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=capability_ids,
            ),
        ),
        catalog_entry_id=QUALIFICATION_CATALOG_ENTRY_ID,
        artifact_locator=f"catalog://artifact/{package_id}",
    )


def _baseline_resolver() -> TaskCapabilityResolver:
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


class QualificationAcquisitionPlanFactory:
    def __init__(self, *, revision_id: str, **kwargs: object) -> None:
        self._revision_id = revision_id
        self._kwargs = kwargs
        self._harness: DelegatedSubtaskQualificationHarness | None = None

    def bind_harness(self, harness: DelegatedSubtaskQualificationHarness) -> None:
        self._harness = harness

    def build_acquisition_plan(
        self,
        *,
        delegation_id: DelegationId,
        task_scope_id: TaskId,
        application_id: str,
        application_environment_id: str,
        lease_id: TaskScopedAgentLeaseId,
        selected_identity: AgentDiscoveryCandidateIdentity,
    ) -> DelegatedSubtaskLifecyclePlan:
        del delegation_id, application_id, application_environment_id
        prior_revision_id = None
        pointer_revision = 0
        if self._harness is not None:
            serving = self._harness.stack.stack.service.inspect_serving(
                application_id=QUALIFICATION_APPLICATION_ID,
                application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
            )
            prior_revision_id = serving.traffic_serving_revision_id
            pointer_revision = serving.serving_pointer_revision
        return DelegatedSubtaskLifecyclePlan(
            acquisition_request=qualification_task_acquire_request(
                str(lease_id),
                task_scope_id,
                self._revision_id,
                identity=selected_identity,
                prior_revision_id=prior_revision_id,
                pointer_revision=pointer_revision,
                **self._kwargs,
            ),
        )


class QualificationReleasePlanFactory:
    def __init__(self, harness: TaskScopedAgentQualificationHarness) -> None:
        self._harness = harness

    def build_release_request(
        self,
        *,
        context: DelegatedSubtaskReleaseContext,
    ) -> TaskScopedAgentReleaseRequest:
        serving = self._harness.stack.service.inspect_serving(
            application_id=QUALIFICATION_APPLICATION_ID,
            application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
        )
        prior_revision_id = serving.traffic_serving_revision_id
        revision_id = f"rev-release-{context.lease.lease_id}"
        return qualification_task_release_request(
            str(context.lease.lease_id),
            context.task_scope_id,
            revision_id,
            disable_revision=qualification_binding_revision(self._harness.stack),
            prior_revision_id=prior_revision_id,
            pointer_revision=serving.serving_pointer_revision,
        )


@dataclass
class StaticSpecialistInvocation(
    SpecialistInvocationPort[OcrQualificationRequest, OcrQualificationResult],
):
    delegate: DelegatedSubtaskDelegate[OcrQualificationRequest, OcrQualificationResult]

    def resolve_delegate(
        self,
        *,
        lease: TaskScopedAgentLease,
        acquisition_result: DynamicAgentAcquisitionResult,
    ) -> DelegatedSubtaskDelegate[OcrQualificationRequest, OcrQualificationResult]:
        del lease, acquisition_result
        return self.delegate


class EchoOcrQualificationDelegate:
    async def execute(self, request: OcrQualificationRequest) -> OcrQualificationResult:
        return OcrQualificationResult(text=f"ocr:{request.document_ref}")


@dataclass
class DelegatedSubtaskQualificationHarness:
    task_scoped: TaskScopedAgentService
    service: DelegatedSubtaskService[OcrQualificationRequest, OcrQualificationResult]
    stack: TaskScopedAgentQualificationHarness
    lease_store: object
    task_scope_authority: FixedTaskScopeAuthority


def build_delegated_subtask_qualification_harness(
    *,
    candidates: tuple[AgentDiscoveryCandidate, ...],
    revision_id: str = "rev-delegate-1",
    specialist_delegate: (
        DelegatedSubtaskDelegate[OcrQualificationRequest, OcrQualificationResult] | None
    ) = None,
    acquisition_kwargs: dict[str, object] | None = None,
    acquisition_plan_factory: DelegatedSubtaskAcquisitionPlanFactory | None = None,
    task_scope: TaskId | None = None,
    task_scope_authority: FixedTaskScopeAuthority | None = None,
    capability_resolver: TaskCapabilityResolver | None = None,
    physical_delegation_governance: object | None = None,
) -> DelegatedSubtaskQualificationHarness:
    harness = build_task_scoped_agent_qualification_harness()
    delegate = specialist_delegate or EchoOcrQualificationDelegate()
    resolved_task_scope = task_scope or mint_task_id()
    authority = task_scope_authority or FixedTaskScopeAuthority(resolved_task_scope)
    acquisition_factory = (
        acquisition_plan_factory
        or QualificationAcquisitionPlanFactory(
            revision_id=revision_id,
            **(acquisition_kwargs or {}),
        )
    )
    release_factory = QualificationReleasePlanFactory(harness)
    service = DelegatedSubtaskService(
        capability_resolver=capability_resolver or _baseline_resolver(),
        discovery=_federated_discovery(*candidates),
        matcher=CapabilityMatcher(),
        selector=DeterministicIdentitySelectionStrategy(),
        task_scoped_agents=harness.service,
        task_scope_authority=authority,
        acquisition_plan_factory=acquisition_factory,
        release_plan_factory=release_factory,
        specialist_invocation=StaticSpecialistInvocation(delegate=delegate),
        child_execution=as_child_execution_port(
            ChildExecutionRunner[OcrQualificationRequest, OcrQualificationResult](
                ledger=_UNLIMITED_LEDGER,
            ),
        ),
        physical_delegation_governance=(
            physical_delegation_governance or allowing_physical_delegation_governance()
        ),
    )
    delegated = DelegatedSubtaskQualificationHarness(
        task_scoped=harness.service,
        service=service,
        stack=harness,
        lease_store=harness.lease_store,
        task_scope_authority=authority,
    )
    if isinstance(acquisition_factory, QualificationAcquisitionPlanFactory):
        acquisition_factory.bind_harness(delegated)
    return delegated
