# © Artur Czarnecki. All rights reserved.

"""Real Workspace Ask V2 harness wiring for the governed hybrid knowledge proof."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from unittest.mock import patch

from intergrax.contracts.execution_identity import mint_task_id

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
    PROJECT_STATUS_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.live.project_status.project import (
    PROJECT_STATUS_READ_CAPABILITY_ID,
    ProjectStatusReadLiveRequestV1,
)
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
    RepositoryTenantConnectionPort,
    TenantLiveCapabilityCatalogPort,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionService,
)
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeConnectionRegistryIntegrationResolverV1,
    KnowledgeQueryOrchestratorV1,
    LiveCapabilityExecutorV1,
    WorkspaceIndexedEvidenceRetrieverV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    IndexedWorkspaceEvidenceV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    KnowledgeQueryAudienceV1,
    LiveCallProposalV1,
    LiveEvidenceRequirementV1,
    ProviderEvidencePlanV1,
)
from local_workspace_application.workspaces.hybrid_ask_service import (
    BindingResourceScopeValidator,
    SafeCapabilityRequestEnvelopeValidator,
    WorkspaceAskCommandV2,
    WorkspaceAskServiceV2,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
    CreateIndexedSourceMutationHandler,
    DisableIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_connection_detachment_handler import (
    DetachConnectionMutationHandler,
)
from local_workspace_application.workspaces.knowledge_query_policy_handlers import (
    UpdateQueryPolicyMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV2,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicyV2,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.knowledge_live_access_handlers import (
    CreateLiveAccessBindingMutationHandler,
    DisableLiveAccessBindingMutationHandler,
    DetachLiveAccessBindingMutationHandler,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    DisableWorkspaceLiveAccessBindingCommand,
    LiveAccessLifecycleService,
    WorkspaceLiveAccessBindingService,
    WorkspaceLiveAccessRuntimeAuthority,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from proof_infrastructure.controlled_project_status_service.lifecycle import (
    ControlledProjectStatusServer,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_PROJECT_ID,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.fixtures import (
    DEPLOYMENT_POLICY_CONTENT,
    DEPLOYMENT_POLICY_FILENAME,
    ORION_DEPLOYMENT_QUESTION,
    PROOF_BINDING_ID,
    PROOF_CONNECTION_REF,
    PROOF_CREDENTIAL_REF,
    PROOF_DISABLE_IDEMPOTENCY_HASH,
    PROOF_INDEXED_BINDING_ID,
    PROOF_INDEXED_SOURCE_ID,
    PROOF_LIVE_CALL_ID,
    PROOF_NOW,
    PROOF_TENANT_ID,
    PROOF_WORKSPACE_ID,
    orion_provider_request,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.indexed_bootstrap import (
    IndexedProofStack,
    bootstrap_indexed_proof_stack,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.llm import (
    DeploymentReadinessDeterministicLLM,
)


class _OrionDeploymentProviderStrategy:
    def build_plan(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        request: object,
    ) -> ProviderEvidencePlanV1:
        del configuration
        if isinstance(request, ProjectStatusReadLiveRequestV1):
            project_id = request.project_id
        else:
            project_id = ORION_FIXTURE_PROJECT_ID
        return ProviderEvidencePlanV1(
            ordered_live_call_proposals=(
                LiveCallProposalV1(
                    call_id=PROOF_LIVE_CALL_ID,
                    live_access_binding_id=PROOF_BINDING_ID,
                    capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
                    typed_capability_request={"project_id": project_id},
                ),
            ),
            required_evidence_obligations=(
                LiveEvidenceRequirementV1(
                    requirement_id="provider:orion:live-status",
                    semantic_role="Authoritative ORION project status",
                    call_id=PROOF_LIVE_CALL_ID,
                ),
            ),
        )

    def build_expansion(self, **_: object) -> None:
        return None

    def coverage(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        request: object,
    ) -> dict[str, str]:
        del configuration
        project_id = (
            request.project_id
            if isinstance(request, ProjectStatusReadLiveRequestV1)
            else ORION_FIXTURE_PROJECT_ID
        )
        return {"provider": PROJECT_STATUS_PROVIDER_ID, "project_id": project_id}


class _WorkspaceAuthority:
    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        if tenant_id == PROOF_TENANT_ID and workspace_id == PROOF_WORKSPACE_ID:
            return Workspace(
                workspace_id=PROOF_WORKSPACE_ID,
                tenant_id=PROOF_TENANT_ID,
                name="ORION Proof Workspace",
                status=WorkspaceStatus.ACTIVE,
                created_at=PROOF_NOW,
                updated_at=PROOF_NOW,
            )
        return None


class _RecordingSecretsStore:
    def __init__(self, *, secret: str = "{}") -> None:
        self.secret = secret
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.calls.append(path)
        return self.secret

    def put_secret(self, path: str, value: str) -> None:
        return None

    def delete_secret(self, path: str) -> None:
        return None


class _ProjectStatusCatalog:
    def list_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        del tenant_id, connection_ref, remote_resource_id
        return (
            LiveCapabilityDescriptorV1(
                capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
                provider_id=PROJECT_STATUS_PROVIDER_ID,
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
                source_kind=PROJECT_STATUS_SOURCE_KIND,
                contract_version="1",
                effect=CapabilityEffectV1.READ,
                read_only=True,
                resource_scope_required=False,
                request_schema_ref=(
                    "schema://vendor-knowledge/live/project_status/project/read/request/v1"
                ),
                result_schema_ref=(
                    "schema://vendor-knowledge/live/project_status/project/read/result/v1"
                ),
                max_result_items=1,
                max_result_bytes=65_536,
            ),
        )


class _ScopedTaskIdIndexedRetriever:
    """Proof-local scope: WorkspaceIndexedEvidenceRetrieverV1 uses new_run_id for task_id."""

    def __init__(self, inner: WorkspaceIndexedEvidenceRetrieverV1) -> None:
        self._inner = inner

    async def retrieve(self, **kwargs: object) -> tuple[IndexedWorkspaceEvidenceV1, ...]:
        with patch(
            "intergrax.runtime.task.task_run_bridge.new_run_id",
            mint_task_id,
        ):
            return await self._inner.retrieve(**kwargs)  # type: ignore[arg-type]


class _RevokeAfterIndexedRetriever:
    def __init__(
        self,
        *,
        inner: _ScopedTaskIdIndexedRetriever,
        lifecycle: LiveAccessLifecycleService,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        revoke_after_indexed: bool,
    ) -> None:
        self._inner = inner
        self._lifecycle = lifecycle
        self._configuration_service = configuration_service
        self._revoke_after_indexed = revoke_after_indexed
        self.configuration_revision_before_disable: int | None = None
        self.configuration_revision_after_disable: int | None = None
        self.calls = 0

    async def retrieve(self, **kwargs: object) -> tuple[IndexedWorkspaceEvidenceV1, ...]:
        self.calls += 1
        evidence = await self._inner.retrieve(**kwargs)  # type: ignore[arg-type]
        if self._revoke_after_indexed:
            configuration = self._configuration_service.get_configuration(
                tenant_id=PROOF_TENANT_ID,
                workspace_id=PROOF_WORKSPACE_ID,
            )
            if configuration is None:
                raise RuntimeError("configuration_missing_before_disable")
            self.configuration_revision_before_disable = configuration.configuration_revision
            disabled = self._lifecycle.disable(
                DisableWorkspaceLiveAccessBindingCommand(
                    tenant_id=PROOF_TENANT_ID,
                    workspace_id=PROOF_WORKSPACE_ID,
                    live_access_binding_id=PROOF_BINDING_ID,
                    expected_revision=configuration.configuration_revision,
                    idempotency_key_hash=PROOF_DISABLE_IDEMPOTENCY_HASH,
                )
            )
            self.configuration_revision_after_disable = disabled.configuration_revision
        return evidence


def _seed_knowledge_configuration(repository: ManagedWorkspaceRepository) -> None:
    attachment = WorkspaceConnectionAttachment(
        attachment_id="attachment-orion",
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
        connection_ref=PROOF_CONNECTION_REF,
        safe_display_label="ORION Project Status",
        status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
        mutation_id="mutation-orion",
        effective_revision=1,
        created_at=PROOF_NOW,
        updated_at=PROOF_NOW,
    )
    indexed_binding = WorkspaceIndexedSourceBinding(
        indexed_source_binding_id=PROOF_INDEXED_BINDING_ID,
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
        knowledge_source_binding_ref="knowledge-source-deployment-policy",
        source_id=PROOF_INDEXED_SOURCE_ID,
        status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
        audience_eligibility=KnowledgeAudienceEligibilityV1.PERSONAL_ONLY,
        mutation_id="mutation-indexed",
        effective_revision=1,
        semantic_identity_hash=sha256(PROOF_INDEXED_BINDING_ID.encode()).hexdigest(),
        created_at=PROOF_NOW,
        updated_at=PROOF_NOW,
        cached_safe_display_label="Deployment Policy",
    )
    live_binding = WorkspaceLiveAccessBinding(
        live_access_binding_id=PROOF_BINDING_ID,
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
        connection_ref=PROOF_CONNECTION_REF,
        allowed_capability_ids=(PROJECT_STATUS_READ_CAPABILITY_ID,),
        derived_provider_id=PROJECT_STATUS_PROVIDER_ID,
        derived_integration_kind=IntegrationCategory.ISSUE_TRACKER,
        derived_safe_display_label="ORION Project Status",
        status=LiveAccessBindingStatusV1.ACTIVE,
        mutation_id="mutation-live",
        effective_revision=1,
        semantic_identity_hash=sha256(PROOF_BINDING_ID.encode()).hexdigest(),
        created_at=PROOF_NOW,
        updated_at=PROOF_NOW,
    )
    query_policy = WorkspaceQueryPolicyV2(
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
        mode=QueryPolicyModeV2.HYBRID,
        allowed_connection_refs=(PROOF_CONNECTION_REF,),
        allowed_capability_ids=(PROJECT_STATUS_READ_CAPABILITY_ID,),
        max_live_calls=1,
        max_total_duration_ms=30_000,
        max_result_items=10,
        max_result_bytes=1_048_576,
        live_result_retention=LiveResultRetentionV1.EPHEMERAL,
        mutation_id="mutation-policy",
        effective_revision=1,
        updated_at=PROOF_NOW,
    )
    repository.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=PROOF_TENANT_ID,
            workspace_id=PROOF_WORKSPACE_ID,
            committed_revision=1,
            updated_at=PROOF_NOW,
        )
    )
    repository.put_knowledge_connection_attachment_version_if_absent(attachment)
    repository.put_knowledge_indexed_source_version_if_absent(indexed_binding)
    repository.put_knowledge_live_access_version_if_absent(live_binding)
    repository.put_knowledge_query_policy_version_if_absent(query_policy)


def _build_live_access_stack(
    *,
    repository: ManagedWorkspaceRepository,
    workspace_service: ManagedWorkspaceService,
    connection_repository: DocumentStoreTenantConnectionRepository,
    catalog: TenantLiveCapabilityCatalogPort,
) -> tuple[
    WorkspaceKnowledgeConfigurationService,
    WorkspaceLiveAccessBindingService,
    LiveAccessLifecycleService,
]:
    configuration_service = WorkspaceKnowledgeConfigurationService(
        repository,
        workspace_service,
    )
    mutation_engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repository,
        workspace_service,
        configuration_service,
        {
            WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: (
                CreateIndexedSourceMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE: (
                DisableIndexedSourceMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: (
                AttachConnectionMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION: (
                DetachConnectionMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING: (
                CreateLiveAccessBindingMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING: (
                DisableLiveAccessBindingMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.DETACH_LIVE_ACCESS_BINDING: (
                DetachLiveAccessBindingMutationHandler()
            ),
            WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY: (
                UpdateQueryPolicyMutationHandler()
            ),
        },
        clock=lambda: PROOF_NOW,
    )
    tenant_connection_port = RepositoryTenantConnectionPort(connection_repository)
    attachment_service = WorkspaceConnectionAttachmentService(
        connection_port=tenant_connection_port,
        configuration_service=configuration_service,
        mutation_engine=mutation_engine,
    )
    live_access_service = WorkspaceLiveAccessBindingService(
        repository=repository,
        configuration_service=configuration_service,
        mutation_engine=mutation_engine,
        tenant_connection_port=tenant_connection_port,
        capability_catalog=catalog,
        remote_resource_lookup_port=None,
    )
    lifecycle = LiveAccessLifecycleService(
        configuration_service=configuration_service,
        live_access_binding_service=live_access_service,
        connection_attachment_service=attachment_service,
        tenant_connection_port=tenant_connection_port,
        capability_catalog=catalog,
    )
    return configuration_service, live_access_service, lifecycle


@dataclass(slots=True)
class GovernedHybridKnowledgeHarness:
    server: ControlledProjectStatusServer
    service: WorkspaceAskServiceV2
    ask_repository: WorkspaceAskRepository
    llm: DeploymentReadinessDeterministicLLM
    configuration_service: WorkspaceKnowledgeConfigurationService
    live_access_lifecycle: LiveAccessLifecycleService
    indexed_stack: IndexedProofStack
    indexed_retriever: _RevokeAfterIndexedRetriever
    connection_registry: KnowledgeConnectionRegistry
    tenant_connection_repository: DocumentStoreTenantConnectionRepository
    rehydration_status: str
    run_id_counter: int = 0

    def next_run_id(self) -> str:
        self.run_id_counter += 1
        return f"orion-proof-run-{self.run_id_counter}"

    def build_command(self, *, run_id: str | None = None) -> WorkspaceAskCommandV2:
        return WorkspaceAskCommandV2(
            tenant_id=PROOF_TENANT_ID,
            workspace_id=PROOF_WORKSPACE_ID,
            question=ORION_DEPLOYMENT_QUESTION,
            requested_mode=QueryPolicyModeV2.HYBRID,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            provider_request=orion_provider_request(),
            indexed_max_results=5,
            run_id=run_id or self.next_run_id(),
            request_id=f"request-{self.run_id_counter}",
        )

    def reset_http_counter(self) -> None:
        self.server.store.reset_read_request_count()

    def http_read_count(self) -> int:
        return self.server.store.read_request_count()


async def build_harness(
    *,
    server: ControlledProjectStatusServer,
    revoke_after_indexed: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> GovernedHybridKnowledgeHarness:
    indexed_stack = await bootstrap_indexed_proof_stack(
        tenant_id=PROOF_TENANT_ID,
        workspace_id=PROOF_WORKSPACE_ID,
        source_id=PROOF_INDEXED_SOURCE_ID,
        policy_filename=DEPLOYMENT_POLICY_FILENAME,
        policy_content=DEPLOYMENT_POLICY_CONTENT,
        proof_now=PROOF_NOW,
    )
    repository = indexed_stack.repository
    _seed_knowledge_configuration(repository)

    connection_repository = DocumentStoreTenantConnectionRepository(
        repository.document_store
    )
    TenantConnectionService(
        tenant_id=PROOF_TENANT_ID,
        repository=connection_repository,
    ).create(
        TenantConnection(
            connection_ref=PROOF_CONNECTION_REF,
            tenant_id=PROOF_TENANT_ID,
            provider_id=PROJECT_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            safe_display_name="ORION Project Status",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref=PROOF_CREDENTIAL_REF,
            validated_secret_free_config={
                "base_url": server.base_url,
                "timeout_seconds": 2.0,
            },
            configuration_version=1,
            created_at=PROOF_NOW,
            updated_at=PROOF_NOW,
        )
    )

    catalog = _ProjectStatusCatalog()
    configuration_service, _, live_access_lifecycle = _build_live_access_stack(
        repository=repository,
        workspace_service=indexed_stack.workspace_service,
        connection_repository=connection_repository,
        catalog=catalog,
    )

    connection_registry = KnowledgeConnectionRegistry()
    secrets = _RecordingSecretsStore()
    rehydration = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=secrets,
        integration_factory=build_default_vendor_knowledge_connection_factory_registry(),
        connection_registry=connection_registry,
    ).rehydrate_connection(
        tenant_id=PROOF_TENANT_ID,
        connection_ref=PROOF_CONNECTION_REF,
    )

    inner_retriever = _ScopedTaskIdIndexedRetriever(
        WorkspaceIndexedEvidenceRetrieverV1(
            task_executor=indexed_stack.search_task_executor,  # type: ignore[arg-type]
            workspace_repository=repository,
            clock=clock or (lambda: PROOF_NOW),
        )
    )
    indexed_retriever = _RevokeAfterIndexedRetriever(
        inner=inner_retriever,
        lifecycle=live_access_lifecycle,
        configuration_service=configuration_service,
        revoke_after_indexed=revoke_after_indexed,
    )

    llm = DeploymentReadinessDeterministicLLM()
    ask_repository = WorkspaceAskRepository(repository.document_store)
    published = build_vendor_knowledge_live_registration_registry().publish()
    tenant_connection_port = RepositoryTenantConnectionPort(connection_repository)
    runtime_authority = WorkspaceLiveAccessRuntimeAuthority(
        configuration_service=configuration_service,
        tenant_connection_port=tenant_connection_port,
        capability_catalog=catalog,
    )
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed_retriever,
        live_executor=LiveCapabilityExecutorV1(
            published_registration=published,
            integration_resolver=KnowledgeConnectionRegistryIntegrationResolverV1(
                connection_registry
            ),
            runtime_authority=runtime_authority,
            clock=clock or (lambda: PROOF_NOW),
            monotonic=lambda: 100.0,
        ),
        clock=clock or (lambda: PROOF_NOW),
        monotonic=lambda: 100.0,
    )
    plan_counter = {"value": 0}

    def _proof_plan_id_factory() -> str:
        plan_counter["value"] += 1
        return f"orion-plan-{plan_counter['value']}"

    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=repository,
        ask_repository=ask_repository,
        configuration_service=configuration_service,
        capability_catalog=catalog,
        request_envelope_validator=SafeCapabilityRequestEnvelopeValidator(
            schema_registry=published.schemas
        ),
        resource_scope_validator=BindingResourceScopeValidator(),
        orchestrator=orchestrator,
        llm_adapter=llm,
        clock=clock or (lambda: PROOF_NOW),
        run_id_factory=lambda: "orion-proof-placeholder",
        plan_id_factory=_proof_plan_id_factory,
        provider_strategy=_OrionDeploymentProviderStrategy(),
    )
    return GovernedHybridKnowledgeHarness(
        server=server,
        service=service,
        ask_repository=ask_repository,
        llm=llm,
        configuration_service=configuration_service,
        live_access_lifecycle=live_access_lifecycle,
        indexed_stack=indexed_stack,
        indexed_retriever=indexed_retriever,
        connection_registry=connection_registry,
        tenant_connection_repository=connection_repository,
        rehydration_status=rehydration.status.value,
    )
