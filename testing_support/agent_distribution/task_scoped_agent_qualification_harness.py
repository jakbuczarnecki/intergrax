# © Artur Czarnecki. All rights reserved.

"""Reusable task-scoped agent harness for agent_distribution qualification."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.admin_models import (
    BindAgentRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidateIdentity,
)
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    AgentPackageCandidate,
    CatalogPackageResolution,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.dynamic_acquisition import (
    CatalogSourceProviderRegistry,
    DynamicAgentAcquisitionInstallIntent,
    DynamicAgentAcquisitionRequest,
    DynamicAgentAcquisitionService,
)
from intergrax.agent_distribution.task_scoped_agents import (
    InMemoryTaskScopedAgentLeaseStore,
    TaskScopedAgentAcquisitionRequest,
    TaskScopedAgentLeaseId,
    TaskScopedAgentReleaseRequest,
    TaskScopedAgentService,
)
from intergrax.contracts.execution_identity import TaskId
from testing_support.agent_distribution.agent_platform_admin_qualification_harness import (
    QUALIFICATION_APPLICATION_ID,
    QUALIFICATION_ENVIRONMENT_ID,
    QUALIFICATION_METADATA_REF,
    QUALIFICATION_PACKAGE_DIGEST,
    QUALIFICATION_PACKAGE_ID,
    AgentPlatformAdminQualificationStack,
    build_agent_platform_admin_qualification_stack,
    qualification_activate_request,
    qualification_build_request,
    qualification_trust_record,
)
from testing_support.agent_platform_admin_harness import admin_test_principal

QUALIFICATION_CATALOG_SOURCE_ID = "builtin-1"
QUALIFICATION_CATALOG_ENTRY_ID = "cat-researcher"
QUALIFICATION_BINDING_ID = "bind-search"
QUALIFICATION_INSTALL_ID = "inst-1"
QUALIFICATION_SLOT_ID = "slot-search"
QUALIFICATION_LOGICAL_AGENT_ID = "researcher"


def _source() -> CatalogSourceIdentity:
    return CatalogSourceIdentity(
        catalog_source_id=QUALIFICATION_CATALOG_SOURCE_ID,
        provider_kind=CatalogProviderKind.BUILTIN,
    )


def _candidate() -> AgentPackageCandidate:
    return AgentPackageCandidate(
        distribution_package_id=QUALIFICATION_PACKAGE_ID,
        package_version="1.0.0",
        package_digest=QUALIFICATION_PACKAGE_DIGEST,
    )


def _identity() -> AgentDiscoveryCandidateIdentity:
    return AgentDiscoveryCandidateIdentity(source=_source(), package=_candidate())


class _ExactQualificationCatalog:
    def __init__(self) -> None:
        self._entry = AgentCatalogEntry(
            catalog_entry_id=QUALIFICATION_CATALOG_ENTRY_ID,
            catalog_source=_source(),
            display_name="Researcher",
            package_id_line=QUALIFICATION_PACKAGE_ID,
        )
        self._resolution = CatalogPackageResolution(
            entry=self._entry,
            package_candidate=_candidate(),
            artifact_locator="catalog://artifact/researcher",
        )

    @property
    def catalog_source_id(self) -> str:
        return self._entry.catalog_source.catalog_source_id

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return [self._entry]

    def resolve_package(
        self,
        entry: AgentCatalogEntry,
        *,
        version_selector: str,
    ) -> CatalogPackageResolution:
        del entry, version_selector
        return self._resolution

    def health(self) -> None:
        return None


def qualification_task_acquire_request(
    lease_id: str,
    task_scope_id: TaskId,
    revision_id: str,
    **kwargs: object,
) -> TaskScopedAgentAcquisitionRequest:
    identity = kwargs.pop("identity", None)
    install_mutation_id = str(kwargs.pop("install_mutation_id", "mut-install"))
    bind_mutation_id = str(kwargs.pop("bind_mutation_id", "mut-bind"))
    pointer_revision = int(kwargs.pop("pointer_revision", 0))
    prior_revision_id = kwargs.pop("prior_revision_id", None)
    binding_id = str(kwargs.pop("binding_id", QUALIFICATION_BINDING_ID))
    install_id = str(kwargs.pop("install_id", QUALIFICATION_INSTALL_ID))
    slot_id = str(kwargs.pop("slot_id", QUALIFICATION_SLOT_ID))
    return TaskScopedAgentAcquisitionRequest(
        lease_id=TaskScopedAgentLeaseId(lease_id),
        task_scope_id=task_scope_id,
        acquisition_request=DynamicAgentAcquisitionRequest(
            selected_identity=identity if identity is not None else _identity(),
            application_id=QUALIFICATION_APPLICATION_ID,
            application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
            catalog_entry_id=QUALIFICATION_CATALOG_ENTRY_ID,
            install=DynamicAgentAcquisitionInstallIntent(
                mutation_id=install_mutation_id,
                installation_id=install_id,
                installation_slot_id=slot_id,
                artifact_store_ref="store://artifacts/inst-1",
                trust_record=qualification_trust_record(),
                agent_project_metadata_ref=QUALIFICATION_METADATA_REF,
            ),
            bind=BindAgentRequest(
                mutation_id=bind_mutation_id,
                application_binding_id=binding_id,
                logical_agent_id=QUALIFICATION_LOGICAL_AGENT_ID,
                installation_slot_id=slot_id,
                enablement=True,
            ),
            build=qualification_build_request(
                revision_id,
                mutation_id=f"{install_mutation_id}-build",
            ),
            activate=qualification_activate_request(
                revision_id,
                pointer_revision=pointer_revision,
                prior_revision_id=prior_revision_id,
                mutation_id=f"{install_mutation_id}-activate",
            ),
        ),
    )


def qualification_task_release_request(
    lease_id: str,
    task_scope_id: TaskId,
    revision_id: str,
    *,
    disable_revision: int = 1,
    pointer_revision: int = 1,
    prior_revision_id: str | None = None,
    disable_mutation_id: str = "mut-disable",
) -> TaskScopedAgentReleaseRequest:
    return TaskScopedAgentReleaseRequest(
        lease_id=TaskScopedAgentLeaseId(lease_id),
        task_scope_id=task_scope_id,
        application_id=QUALIFICATION_APPLICATION_ID,
        application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
        disable=SetAgentEnablementRequest(
            mutation_id=disable_mutation_id,
            expected_revision=disable_revision,
        ),
        build=qualification_build_request(
            revision_id,
            mutation_id=f"{disable_mutation_id}-build",
        ),
        activate=qualification_activate_request(
            revision_id,
            pointer_revision=pointer_revision,
            prior_revision_id=prior_revision_id,
            mutation_id=f"{disable_mutation_id}-activate",
        ),
    )


def qualification_binding_revision(
    stack: AgentPlatformAdminQualificationStack,
    binding_id: str = QUALIFICATION_BINDING_ID,
) -> int:
    binding = stack.service._binding_store.get_binding(binding_id)
    assert binding is not None
    return binding.binding_revision


@dataclass
class TaskScopedAgentQualificationHarness:
    stack: AgentPlatformAdminQualificationStack
    service: TaskScopedAgentService
    lease_store: InMemoryTaskScopedAgentLeaseStore


def build_task_scoped_agent_qualification_harness() -> (
    TaskScopedAgentQualificationHarness
):
    stack = build_agent_platform_admin_qualification_stack(with_catalog=False)
    catalog = _ExactQualificationCatalog()
    registry = CatalogSourceProviderRegistry({QUALIFICATION_CATALOG_SOURCE_ID: catalog})
    acquisition = DynamicAgentAcquisitionService(
        catalog_registry=registry,
        lifecycle=stack.service,
    )
    lease_store = InMemoryTaskScopedAgentLeaseStore()
    service = TaskScopedAgentService(
        acquisition=acquisition,
        lifecycle=stack.service,
        lease_store=lease_store,
    )
    return TaskScopedAgentQualificationHarness(
        stack=stack,
        service=service,
        lease_store=lease_store,
    )


__all__ = [
    "QUALIFICATION_APPLICATION_ID",
    "QUALIFICATION_ENVIRONMENT_ID",
    "QUALIFICATION_PACKAGE_ID",
    "QUALIFICATION_PACKAGE_DIGEST",
    "TaskScopedAgentQualificationHarness",
    "admin_test_principal",
    "build_task_scoped_agent_qualification_harness",
    "qualification_binding_revision",
    "qualification_task_acquire_request",
    "qualification_task_release_request",
]
