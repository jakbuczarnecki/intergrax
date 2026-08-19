# © Artur Czarnecki. All rights reserved.

"""Real Workspace Ask V2 harness wiring for the governed hybrid knowledge proof."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.project_status.bundle import (
    create_project_status_integration,
)
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
    PROJECT_STATUS_SOURCE_KIND,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.live.project_status.project import (
    PROJECT_STATUS_READ_CAPABILITY_ID,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
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
    ResolvedLiveResourceScopeV1,
)
from local_workspace_application.workspaces.hybrid_ask_service import (
    BindingResourceScopeValidator,
    SafeCapabilityRequestEnvelopeValidator,
    WorkspaceAskCommandV2,
    WorkspaceAskServiceV2,
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
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicyV2,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    WorkspaceLiveAccessRuntimeAuthority,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceSource,
)
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
    PROOF_DOCUMENT_ID,
    PROOF_INDEXED_BINDING_ID,
    PROOF_INDEXED_SOURCE_ID,
    PROOF_LIVE_CALL_ID,
    PROOF_NOW,
    PROOF_TENANT_ID,
    PROOF_WORKSPACE_ID,
    orion_provider_request,
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
        project_id = ORION_FIXTURE_PROJECT_ID
        if isinstance(request, dict):
            project_id = str(request.get("project_id", project_id))
        elif hasattr(request, "project_id"):
            project_id = str(getattr(request, "project_id"))
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

    def coverage(self, **_: object) -> dict[str, str]:
        return {"provider": PROJECT_STATUS_PROVIDER_ID, "project_id": ORION_FIXTURE_PROJECT_ID}


class _WorkspaceAuthority:
    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        if tenant_id == PROOF_TENANT_ID and workspace_id == PROOF_WORKSPACE_ID:
            return Workspace(
                workspace_id=PROOF_WORKSPACE_ID,
                tenant_id=PROOF_TENANT_ID,
                name="ORION Proof Workspace",
                created_at=PROOF_NOW,
                updated_at=PROOF_NOW,
            )
        return None


class _ProofWorkspaceRepository:
    def __init__(self, indexed_binding: WorkspaceIndexedSourceBinding) -> None:
        self.indexed_binding = indexed_binding
        self.workspace = Workspace(
            workspace_id=PROOF_WORKSPACE_ID,
            tenant_id=PROOF_TENANT_ID,
            name="ORION Proof Workspace",
            created_at=PROOF_NOW,
            updated_at=PROOF_NOW,
        )
        self.document = WorkspaceDocumentReference(
            document_id=PROOF_DOCUMENT_ID,
            tenant_id=PROOF_TENANT_ID,
            workspace_id=PROOF_WORKSPACE_ID,
            source_id=indexed_binding.source_id,
            source_path=f"docs/{DEPLOYMENT_POLICY_FILENAME}",
            file_name=DEPLOYMENT_POLICY_FILENAME,
            content_hash="sha256:" + sha256(DEPLOYMENT_POLICY_CONTENT.encode()).hexdigest(),
            indexed_at=PROOF_NOW,
        )
        self.source = WorkspaceSource(
            source_id=indexed_binding.source_id,
            tenant_id=PROOF_TENANT_ID,
            workspace_id=PROOF_WORKSPACE_ID,
            path="docs",
            created_at=PROOF_NOW,
        )

    def get_workspace(self, **_: object) -> Workspace | None:
        return self.workspace

    def get_knowledge_configuration_head(
        self, **_: object
    ) -> WorkspaceKnowledgeConfigurationHead:
        return WorkspaceKnowledgeConfigurationHead(
            tenant_id=PROOF_TENANT_ID,
            workspace_id=PROOF_WORKSPACE_ID,
            committed_revision=1,
            updated_at=PROOF_NOW,
        )

    def list_knowledge_connection_attachment_versions(self, **_: object) -> list[object]:
        return []

    def list_knowledge_indexed_source_versions(
        self, **_: object
    ) -> list[WorkspaceIndexedSourceBinding]:
        return [self.indexed_binding]

    def list_knowledge_live_access_versions(self, **_: object) -> list[object]:
        return []

    def list_knowledge_query_policy_versions(self, **_: object) -> list[object]:
        return []

    def get_document_ref(self, **_: object) -> WorkspaceDocumentReference | None:
        return self.document

    def get_source(self, **_: object) -> WorkspaceSource | None:
        return self.source


class _MutableProofConfiguration:
    def __init__(self, *, revoke_after_indexed: bool = False) -> None:
        self._revoke_after_indexed = revoke_after_indexed
        self.value = self._build_configuration(
            binding_status=LiveAccessBindingStatusV1.ACTIVE
        )

    def get_configuration(
        self, *, tenant_id: str, workspace_id: str
    ) -> WorkspaceKnowledgeConfigurationV1 | None:
        if tenant_id != PROOF_TENANT_ID or workspace_id != PROOF_WORKSPACE_ID:
            return None
        return self.value

    def disable_binding(self) -> None:
        binding = self.value.live_access_bindings[0]
        disabled = binding.model_copy(
            update={"status": LiveAccessBindingStatusV1.DISABLED}
        )
        self.value = self.value.model_copy(
            update={
                "live_access_bindings": (disabled,),
                "configuration_revision": self.value.configuration_revision + 1,
            }
        )

    def _build_configuration(
        self,
        *,
        binding_status: LiveAccessBindingStatusV1,
    ) -> WorkspaceKnowledgeConfigurationV1:
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
        binding = WorkspaceLiveAccessBinding(
            live_access_binding_id=PROOF_BINDING_ID,
            tenant_id=PROOF_TENANT_ID,
            workspace_id=PROOF_WORKSPACE_ID,
            connection_ref=PROOF_CONNECTION_REF,
            allowed_capability_ids=(PROJECT_STATUS_READ_CAPABILITY_ID,),
            derived_provider_id=PROJECT_STATUS_PROVIDER_ID,
            derived_integration_kind=IntegrationCategory.ISSUE_TRACKER,
            derived_safe_display_label="ORION Project Status",
            status=binding_status,
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
        return WorkspaceKnowledgeConfigurationV1(
            tenant_id=PROOF_TENANT_ID,
            workspace_id=PROOF_WORKSPACE_ID,
            configuration_revision=1,
            connection_attachments=(attachment,),
            indexed_sources=(indexed_binding,),
            live_access_bindings=(binding,),
            query_policy=query_policy,
            updated_at=PROOF_NOW,
        )


class _SearchExecution:
    def __init__(self, evidence: list[dict[str, object]]) -> None:
        self.structured_data = {"search_summary": {"evidence": evidence}}


class _SearchTaskResult:
    def __init__(self, evidence: list[dict[str, object]]) -> None:
        self.agent_id = "agent-proof"
        self.run_id = "run-proof"
        self.task_id = "task-proof"
        self.metadata: dict[str, object] = {}
        self.execution_result = _SearchExecution(evidence)

    def model_copy(self, *, update: dict[str, object]) -> _SearchTaskResult:
        self.metadata = dict(update["metadata"])  # type: ignore[arg-type]
        return self


class _SearchTaskExecutor:
    def __init__(self, result: _SearchTaskResult) -> None:
        self.result = result

    async def execute(self, task: object) -> _SearchTaskResult:
        del task
        return self.result


class _IndexedRetriever:
    def __init__(
        self,
        *,
        repository: _ProofWorkspaceRepository,
        configuration: _MutableProofConfiguration,
        revoke_after_indexed: bool,
    ) -> None:
        search_result = _SearchTaskResult(
            [
                {
                    "document_id": PROOF_DOCUMENT_ID,
                    "source_id": repository.indexed_binding.source_id,
                    "workspace_id": PROOF_WORKSPACE_ID,
                    "source_path": f"docs/{DEPLOYMENT_POLICY_FILENAME}",
                    "file_name": DEPLOYMENT_POLICY_FILENAME,
                    "score": 0.99,
                    "snippet": DEPLOYMENT_POLICY_CONTENT,
                    "metadata": {
                        "indexed_source_binding_id": PROOF_INDEXED_BINDING_ID,
                    },
                }
            ]
        )
        self._inner = WorkspaceIndexedEvidenceRetrieverV1(
            task_executor=_SearchTaskExecutor(search_result),  # type: ignore[arg-type]
            workspace_repository=repository,  # type: ignore[arg-type]
            clock=lambda: PROOF_NOW,
        )
        self._configuration = configuration
        self._revoke_after_indexed = revoke_after_indexed
        self.calls = 0

    async def retrieve(self, **kwargs: object) -> tuple[IndexedWorkspaceEvidenceV1, ...]:
        self.calls += 1
        evidence = await self._inner.retrieve(**kwargs)  # type: ignore[arg-type]
        if self._revoke_after_indexed:
            self._configuration.disable_binding()
        return evidence


class _TenantConnectionPort:
    def get_connection(self, *, tenant_id: str, connection_ref: str) -> object:
        return SafeTenantConnectionV1(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=PROJECT_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            safe_display_name="ORION Project Status",
            configuration_version=1,
            connected_principal_ref="principal-orion",
            created_at=PROOF_NOW,
            updated_at=PROOF_NOW,
        )


class _ProjectStatusCatalog:
    def list_capabilities(self, **_: object) -> tuple[LiveCapabilityDescriptorV1, ...]:
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


@dataclass(slots=True)
class GovernedHybridKnowledgeHarness:
    server: ControlledProjectStatusServer
    service: WorkspaceAskServiceV2
    ask_repository: WorkspaceAskRepository
    llm: DeploymentReadinessDeterministicLLM
    configuration: _MutableProofConfiguration
    indexed_retriever: _IndexedRetriever
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


def build_harness(
    *,
    server: ControlledProjectStatusServer,
    revoke_after_indexed: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> GovernedHybridKnowledgeHarness:
    from intergrax.runtime.task import task_run_bridge

    task_counter = {"value": 0}
    plan_counter = {"value": 0}

    def _proof_task_id_factory() -> str:
        task_counter["value"] += 1
        return f"task_{task_counter['value']:032x}"

    def _proof_plan_id_factory() -> str:
        plan_counter["value"] += 1
        return f"orion-plan-{plan_counter['value']}"

    task_run_bridge.new_run_id = _proof_task_id_factory  # type: ignore[method-assign]

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
    repository = _ProofWorkspaceRepository(indexed_binding)
    configuration = _MutableProofConfiguration(revoke_after_indexed=revoke_after_indexed)
    indexed_retriever = _IndexedRetriever(
        repository=repository,
        configuration=configuration,
        revoke_after_indexed=revoke_after_indexed,
    )
    llm = DeploymentReadinessDeterministicLLM()
    store = InMemoryDocumentStore()
    ask_repository = WorkspaceAskRepository(store)
    integration = create_project_status_integration(base_url=server.base_url)
    from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry

    connection_registry = KnowledgeConnectionRegistry()
    connection_registry.register(
        tenant_id=PROOF_TENANT_ID,
        connection_ref=PROOF_CONNECTION_REF,
        provider_id=PROJECT_STATUS_PROVIDER_ID,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        integration=integration,
    )
    published = build_vendor_knowledge_live_registration_registry().publish()
    runtime_authority = WorkspaceLiveAccessRuntimeAuthority(
        configuration_service=configuration,  # type: ignore[arg-type]
        tenant_connection_port=_TenantConnectionPort(),  # type: ignore[arg-type]
        capability_catalog=_ProjectStatusCatalog(),  # type: ignore[arg-type]
    )
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=indexed_retriever,  # type: ignore[arg-type]
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
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=repository,  # type: ignore[arg-type]
        ask_repository=ask_repository,
        configuration_service=configuration,  # type: ignore[arg-type]
        capability_catalog=_ProjectStatusCatalog(),  # type: ignore[arg-type]
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
        configuration=configuration,
        indexed_retriever=indexed_retriever,
    )
