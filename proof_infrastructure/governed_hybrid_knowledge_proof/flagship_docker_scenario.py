# © Artur Czarnecki. All rights reserved.

"""Four-provider Docker flagship scenario for COMM-5 F3-F."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.change_approval.knowledge_read import (
    CHANGE_APPROVAL_PROVIDER_ID,
)
from intergrax.integrations.providers.governance_approval.knowledge_read import (
    GOVERNANCE_APPROVAL_PROVIDER_ID,
)
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
)
from intergrax.integrations.providers.security_status.knowledge_read import (
    SECURITY_STATUS_PROVIDER_ID,
)
from intergrax.runtime.evidence.obligation_derivation import (
    DeterministicEvidenceObligationDerivation,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import ResolvedPolicyRuleV1
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.live.change_approval import (
    CHANGE_APPROVAL_READ_CAPABILITY_ID,
    build_change_approval_read_descriptor,
)
from intergrax.runtime.vendor_knowledge.live.governance_approval import (
    GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
    build_governance_approval_read_descriptor,
)
from intergrax.runtime.vendor_knowledge.live.project_status import (
    PROJECT_STATUS_READ_CAPABILITY_ID,
    build_project_status_read_descriptor,
)
from intergrax.runtime.vendor_knowledge.live.security_status import (
    SECURITY_STATUS_READ_CAPABILITY_ID,
    build_security_status_read_descriptor,
)
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    LiveCapabilityDescriptorV1,
    RepositoryTenantConnectionPort,
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
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeConnectionRegistryIntegrationResolverV1,
    KnowledgeQueryOrchestratorV1,
    LiveCapabilityExecutorV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityStatusV1,
    RequirementAdmissibilityReasonCodeV1,
    RequirementEvaluationStatusV1,
    WorkspaceAskRunV2,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    KnowledgeQueryAudienceV1,
)
from local_workspace_application.workspaces.hybrid_ask_service import (
    BindingResourceScopeValidator,
    SafeCapabilityRequestEnvelopeValidator,
    WorkspaceAskCommandV2,
    WorkspaceAskServiceV2,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV2,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicyV2,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    WorkspaceLiveAccessRuntimeAuthority,
)
from local_workspace_application.workspaces.models import Workspace
from proof_infrastructure.controlled_security_status_service.models import (
    SecurityStatusReadBehaviorV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_admin_ports import (
    FlagshipVendorAdminFacadeV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_docker_environment import (
    AdvancedFlagshipDockerEnvironmentV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_policy import (
    FLAGSHIP_BINDING_CHANGE,
    FLAGSHIP_BINDING_GOVERNANCE,
    FLAGSHIP_BINDING_READINESS,
    FLAGSHIP_BINDING_SECURITY,
    FLAGSHIP_CONN_CHANGE,
    FLAGSHIP_CONN_GOVERNANCE,
    FLAGSHIP_CONN_READINESS,
    FLAGSHIP_CONN_SECURITY,
    FLAGSHIP_POLICY_REV_17,
    FLAGSHIP_POLICY_REV_18,
    FLAGSHIP_QUESTION,
    FLAGSHIP_TENANT_ID,
    FLAGSHIP_WORKSPACE_ID,
    build_flagship_deployment_policy_rules,
)

_FLAGSHIP_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
_STALE_SECURITY_UPDATED_AT = _FLAGSHIP_NOW - timedelta(hours=2)
_FRESH_SECURITY_UPDATED_AT = _FLAGSHIP_NOW - timedelta(minutes=30)
_GOVERNANCE_VALID_FROM = _FLAGSHIP_NOW - timedelta(days=1)
_GOVERNANCE_VALID_UNTIL = _FLAGSHIP_NOW + timedelta(days=1)

_CONNECTION_DESCRIPTORS: dict[str, tuple[LiveCapabilityDescriptorV1, ...]] = {
    FLAGSHIP_CONN_READINESS: (build_project_status_read_descriptor(),),
    FLAGSHIP_CONN_SECURITY: (build_security_status_read_descriptor(),),
    FLAGSHIP_CONN_CHANGE: (build_change_approval_read_descriptor(),),
    FLAGSHIP_CONN_GOVERNANCE: (build_governance_approval_read_descriptor(),),
}


class FlagshipPolicyRulesPort:
    def __init__(self, *, policy_revision: str) -> None:
        self._policy_revision = policy_revision

    def resolve_policy_rules(self, **_: object) -> tuple[ResolvedPolicyRuleV1, ...]:
        return build_flagship_deployment_policy_rules(policy_revision=self._policy_revision)


class _RecordingSecretsStore:
    def get_secret(self, path: str, *, version: str | None = None) -> str:
        del version
        return f"credential-for-{path}"

    def put_secret(self, path: str, value: str) -> None:
        return None

    def delete_secret(self, path: str) -> None:
        return None


class _WorkspaceAuthority:
    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        if tenant_id == FLAGSHIP_TENANT_ID and workspace_id == FLAGSHIP_WORKSPACE_ID:
            return Workspace(
                workspace_id=FLAGSHIP_WORKSPACE_ID,
                tenant_id=FLAGSHIP_TENANT_ID,
                name="Flagship Proof Workspace",
                created_at=_FLAGSHIP_NOW,
                updated_at=_FLAGSHIP_NOW,
            )
        return None


class _Repository:
    def __init__(self, configuration: WorkspaceKnowledgeConfigurationV1) -> None:
        self.configuration = configuration

    def get_knowledge_configuration(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationV1 | None:
        if tenant_id == FLAGSHIP_TENANT_ID and workspace_id == FLAGSHIP_WORKSPACE_ID:
            return self.configuration
        return None


class _MutableConfigurationService:
    def __init__(self, configuration: WorkspaceKnowledgeConfigurationV1) -> None:
        self._configuration = configuration

    def get_configuration(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationV1 | None:
        if tenant_id == FLAGSHIP_TENANT_ID and workspace_id == FLAGSHIP_WORKSPACE_ID:
            return self._configuration
        return None

    def disable_binding(self, binding_id: str) -> None:
        updated_bindings = tuple(
            binding.model_copy(update={"status": LiveAccessBindingStatusV1.DISABLED})
            if binding.live_access_binding_id == binding_id
            else binding
            for binding in self._configuration.live_access_bindings
        )
        self._configuration = self._configuration.model_copy(
            update={"live_access_bindings": updated_bindings},
        )


class _Catalog:
    def list_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        del tenant_id, remote_resource_id
        return _CONNECTION_DESCRIPTORS.get(connection_ref, ())


class _RecordingLLM(LLMAdapter):
    provider = "proof"
    model = "flagship-deterministic"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        del temperature, max_tokens, run_id
        self.calls += 1
        payload = json.loads(messages[-1].content)
        evidence = payload.get("evidence", [])
        used_ids = [str(item["evidence_id"]) for item in evidence]
        return build_adapter_response(
            content=json.dumps(
                {
                    "status": "completed",
                    "answer": "YES — all mandatory evidence is structurally admissible.",
                    "used_evidence_ids": used_ids,
                },
                ensure_ascii=False,
            )
        )


class _EmptyIndexedRetriever:
    async def retrieve(self, **_: object) -> tuple[()]:
        return ()


class _RevokingOrchestrator:
    def __init__(
        self,
        *,
        inner: KnowledgeQueryOrchestratorV1,
        configuration_service: _MutableConfigurationService,
        binding_id: str,
    ) -> None:
        self._inner = inner
        self._configuration_service = configuration_service
        self._binding_id = binding_id

    async def execute(self, **kwargs: object):
        self._configuration_service.disable_binding(self._binding_id)
        return await self._inner.execute(**kwargs)  # type: ignore[arg-type]


def _binding(
    *,
    binding_id: str,
    connection_ref: str,
    capability_id: str,
    provider_id: str,
    integration_kind: IntegrationCategory,
    status: LiveAccessBindingStatusV1 = LiveAccessBindingStatusV1.ACTIVE,
) -> WorkspaceLiveAccessBinding:
    return WorkspaceLiveAccessBinding(
        live_access_binding_id=binding_id,
        tenant_id=FLAGSHIP_TENANT_ID,
        workspace_id=FLAGSHIP_WORKSPACE_ID,
        connection_ref=connection_ref,
        allowed_capability_ids=(capability_id,),
        derived_provider_id=provider_id,
        derived_integration_kind=integration_kind,
        derived_safe_display_label=f"Binding {binding_id}",
        status=status,
        mutation_id=f"mutation-{binding_id}",
        effective_revision=1,
        semantic_identity_hash=sha256(binding_id.encode()).hexdigest(),
        created_at=_FLAGSHIP_NOW,
        updated_at=_FLAGSHIP_NOW,
    )


def _all_bindings() -> tuple[WorkspaceLiveAccessBinding, ...]:
    return (
        _binding(
            binding_id=FLAGSHIP_BINDING_READINESS,
            connection_ref=FLAGSHIP_CONN_READINESS,
            capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
            provider_id=PROJECT_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        ),
        _binding(
            binding_id=FLAGSHIP_BINDING_SECURITY,
            connection_ref=FLAGSHIP_CONN_SECURITY,
            capability_id=SECURITY_STATUS_READ_CAPABILITY_ID,
            provider_id=SECURITY_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.SECURITY_SCANNER,
        ),
        _binding(
            binding_id=FLAGSHIP_BINDING_CHANGE,
            connection_ref=FLAGSHIP_CONN_CHANGE,
            capability_id=CHANGE_APPROVAL_READ_CAPABILITY_ID,
            provider_id=CHANGE_APPROVAL_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        ),
        _binding(
            binding_id=FLAGSHIP_BINDING_GOVERNANCE,
            connection_ref=FLAGSHIP_CONN_GOVERNANCE,
            capability_id=GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
            provider_id=GOVERNANCE_APPROVAL_PROVIDER_ID,
            integration_kind=IntegrationCategory.WORKFLOW_ORCHESTRATOR,
        ),
    )


def _attachment(connection_ref: str) -> WorkspaceConnectionAttachment:
    return WorkspaceConnectionAttachment(
        attachment_id=f"attachment-{connection_ref}",
        tenant_id=FLAGSHIP_TENANT_ID,
        workspace_id=FLAGSHIP_WORKSPACE_ID,
        connection_ref=connection_ref,
        safe_display_label=f"Attachment {connection_ref}",
        status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
        mutation_id=f"mutation-{connection_ref}",
        effective_revision=1,
        created_at=_FLAGSHIP_NOW,
        updated_at=_FLAGSHIP_NOW,
    )


def _configuration(
    bindings: tuple[WorkspaceLiveAccessBinding, ...],
) -> WorkspaceKnowledgeConfigurationV1:
    connection_refs = tuple({binding.connection_ref for binding in bindings})
    capability_ids = tuple(
        capability
        for binding in bindings
        for capability in binding.allowed_capability_ids
    )
    return WorkspaceKnowledgeConfigurationV1(
        tenant_id=FLAGSHIP_TENANT_ID,
        workspace_id=FLAGSHIP_WORKSPACE_ID,
        configuration_revision=1,
        connection_attachments=tuple(_attachment(ref) for ref in connection_refs),
        indexed_sources=(),
        live_access_bindings=bindings,
        query_policy=WorkspaceQueryPolicyV2(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
            mode=QueryPolicyModeV2.LIVE_ONLY,
            allowed_connection_refs=connection_refs,
            allowed_capability_ids=capability_ids,
            max_live_calls=len(bindings),
            max_total_duration_ms=30_000,
            max_result_items=10,
            max_result_bytes=1_048_576,
            live_result_retention=LiveResultRetentionV1.EPHEMERAL,
            mutation_id="mutation-flagship-policy",
            effective_revision=1,
            updated_at=_FLAGSHIP_NOW,
        ),
        updated_at=_FLAGSHIP_NOW,
    )


def _register_connections(
    repository: DocumentStoreTenantConnectionRepository,
    environment: AdvancedFlagshipDockerEnvironmentV1,
) -> tuple[TenantConnection, ...]:
    connections = (
        TenantConnection(
            connection_ref=FLAGSHIP_CONN_READINESS,
            tenant_id=FLAGSHIP_TENANT_ID,
            provider_id=PROJECT_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            safe_display_name="Flagship Project Status",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secret.flagship.project-status",
            validated_secret_free_config={
                "base_url": environment.project_vendor_base_url,
                "timeout_seconds": 5.0,
            },
            configuration_version=1,
            created_at=_FLAGSHIP_NOW,
            updated_at=_FLAGSHIP_NOW,
        ),
        TenantConnection(
            connection_ref=FLAGSHIP_CONN_SECURITY,
            tenant_id=FLAGSHIP_TENANT_ID,
            provider_id=SECURITY_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.SECURITY_SCANNER,
            safe_display_name="Flagship Security Status",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secret.flagship.security-status",
            validated_secret_free_config={
                "base_url": environment.security_vendor_base_url,
                "timeout_seconds": 5.0,
            },
            configuration_version=1,
            created_at=_FLAGSHIP_NOW,
            updated_at=_FLAGSHIP_NOW,
        ),
        TenantConnection(
            connection_ref=FLAGSHIP_CONN_CHANGE,
            tenant_id=FLAGSHIP_TENANT_ID,
            provider_id=CHANGE_APPROVAL_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            safe_display_name="Flagship Change Approval",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secret.flagship.change-approval",
            validated_secret_free_config={
                "base_url": environment.change_vendor_base_url,
                "timeout_seconds": 5.0,
            },
            configuration_version=1,
            created_at=_FLAGSHIP_NOW,
            updated_at=_FLAGSHIP_NOW,
        ),
        TenantConnection(
            connection_ref=FLAGSHIP_CONN_GOVERNANCE,
            tenant_id=FLAGSHIP_TENANT_ID,
            provider_id=GOVERNANCE_APPROVAL_PROVIDER_ID,
            integration_kind=IntegrationCategory.WORKFLOW_ORCHESTRATOR,
            safe_display_name="Flagship Governance Approval",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secret.flagship.governance-approval",
            validated_secret_free_config={
                "base_url": environment.governance_vendor_base_url,
                "timeout_seconds": 5.0,
            },
            configuration_version=1,
            created_at=_FLAGSHIP_NOW,
            updated_at=_FLAGSHIP_NOW,
        ),
    )
    for connection in connections:
        repository.create(connection)
    return connections


async def build_flagship_docker_scenario(
    environment: AdvancedFlagshipDockerEnvironmentV1,
) -> FlagshipDockerScenarioV1:
    bindings = _all_bindings()
    configuration = _configuration(bindings)
    document_store = InMemoryDocumentStore()
    connection_repository = DocumentStoreTenantConnectionRepository(document_store)
    _register_connections(connection_repository, environment)
    connection_registry = KnowledgeConnectionRegistry()
    rehydrator = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=_RecordingSecretsStore(),
        integration_factory=build_default_vendor_knowledge_connection_factory_registry(),
        connection_registry=connection_registry,
    )
    for connection_ref in (
        FLAGSHIP_CONN_READINESS,
        FLAGSHIP_CONN_SECURITY,
        FLAGSHIP_CONN_CHANGE,
        FLAGSHIP_CONN_GOVERNANCE,
    ):
        rehydrator.rehydrate_connection(
            tenant_id=FLAGSHIP_TENANT_ID,
            connection_ref=connection_ref,
        )

    configuration_service = _MutableConfigurationService(configuration)
    authority = WorkspaceLiveAccessRuntimeAuthority(
        configuration_service=configuration_service,  # type: ignore[arg-type]
        tenant_connection_port=RepositoryTenantConnectionPort(connection_repository),
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
    )
    published = build_vendor_knowledge_live_registration_registry().publish()
    executor = LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=KnowledgeConnectionRegistryIntegrationResolverV1(
            connection_registry
        ),
        runtime_authority=authority,
        clock=lambda: _FLAGSHIP_NOW,
    )
    inner_orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=_EmptyIndexedRetriever(),  # type: ignore[arg-type]
        live_executor=executor,
        clock=lambda: _FLAGSHIP_NOW,
    )
    revoking_orchestrator = _RevokingOrchestrator(
        inner=inner_orchestrator,
        configuration_service=configuration_service,
        binding_id=FLAGSHIP_BINDING_GOVERNANCE,
    )
    llm = _RecordingLLM()
    ask_repository = WorkspaceAskRepository(document_store)
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=_Repository(configuration),  # type: ignore[arg-type]
        ask_repository=ask_repository,
        configuration_service=configuration_service,  # type: ignore[arg-type]
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
        request_envelope_validator=SafeCapabilityRequestEnvelopeValidator(
            schema_registry=published.schemas,
        ),
        resource_scope_validator=BindingResourceScopeValidator(),
        orchestrator=inner_orchestrator,
        llm_adapter=llm,
        clock=lambda: _FLAGSHIP_NOW,
        run_id_factory=lambda: "flagship-placeholder",
        plan_id_factory=lambda: "flagship-plan",
        evidence_obligation_derivation_port=DeterministicEvidenceObligationDerivation(),
        resolved_policy_rules_port=FlagshipPolicyRulesPort(
            policy_revision=FLAGSHIP_POLICY_REV_17,
        ),
        schema_registry=published.schemas,
    )
    return FlagshipDockerScenarioV1(
        environment=environment,
        admin=environment.admin,
        service=service,
        llm=llm,
        inner_orchestrator=inner_orchestrator,
        revoking_orchestrator=revoking_orchestrator,
        configuration_service=configuration_service,
    )


@dataclass(slots=True)
class FlagshipDockerScenarioV1:
    environment: AdvancedFlagshipDockerEnvironmentV1
    admin: FlagshipVendorAdminFacadeV1
    service: WorkspaceAskServiceV2
    llm: _RecordingLLM
    inner_orchestrator: KnowledgeQueryOrchestratorV1
    revoking_orchestrator: _RevokingOrchestrator
    configuration_service: _MutableConfigurationService

    def seed_valid_baseline(
        self,
        *,
        security_updated_at: datetime,
    ) -> None:
        self.admin.security.set_read_behavior(SecurityStatusReadBehaviorV1.NORMAL)
        self.admin.security.reset_read_request_count()
        self.admin.project.reset_read_request_count()
        self.admin.change.reset_read_request_count()
        self.admin.governance.reset_read_request_count()
        self.admin.project.seed_project_status()
        self.admin.project.close_readiness_blocker()
        self.admin.security.refresh_security_status(updated_at=security_updated_at)
        self.admin.change.seed_change_approval()
        self.admin.governance.seed_governance_approval(
            valid_from=_GOVERNANCE_VALID_FROM,
            valid_until=_GOVERNANCE_VALID_UNTIL,
        )

    def reset_vendor_read_counts(self) -> None:
        self.admin.project.reset_read_request_count()
        self.admin.security.reset_read_request_count()
        self.admin.change.reset_read_request_count()
        self.admin.governance.reset_read_request_count()

    def restore_active_bindings(self) -> None:
        restored = tuple(
            binding.model_copy(update={"status": LiveAccessBindingStatusV1.ACTIVE})
            for binding in self.configuration_service._configuration.live_access_bindings
        )
        self.configuration_service._configuration = (
            self.configuration_service._configuration.model_copy(
                update={"live_access_bindings": restored},
            )
        )

    def set_policy_revision(self, *, policy_revision: str) -> None:
        self.service._resolved_policy_rules_port = FlagshipPolicyRulesPort(  # noqa: SLF001
            policy_revision=policy_revision,
        )

    def use_revoking_orchestrator(self) -> None:
        self.service._orchestrator = self.revoking_orchestrator  # noqa: SLF001

    def use_inner_orchestrator(self) -> None:
        self.service._orchestrator = self.inner_orchestrator  # noqa: SLF001

    async def ask(
        self,
        *,
        run_id: str,
        request_id: str,
    ) -> WorkspaceAskRunV2:
        self.llm.calls = 0
        return await self.service.ask(
            WorkspaceAskCommandV2(
                tenant_id=FLAGSHIP_TENANT_ID,
                workspace_id=FLAGSHIP_WORKSPACE_ID,
                question=FLAGSHIP_QUESTION,
                requested_mode=QueryPolicyModeV2.LIVE_ONLY,
                audience_context=AudienceContextV1(
                    audience=KnowledgeQueryAudienceV1.PERSONAL
                ),
                run_id=run_id,
                request_id=request_id,
            )
        )

    def reload_run(self, *, run_id: str) -> WorkspaceAskRunV2:
        return self.service.get_run(tenant_id=FLAGSHIP_TENANT_ID, run_id=run_id)

    def evaluation_for_suffix(self, run: WorkspaceAskRunV2, suffix: str):
        if run.evidence_admissibility is None:
            raise RuntimeError("evidence_admissibility_missing")
        return next(
            item
            for item in run.evidence_admissibility.requirement_evaluations
            if item.requirement_id.endswith(f":{suffix}")
        )

    def vendor_read_counts(self) -> dict[str, int]:
        return {
            "project": self.admin.project.read_request_count(),
            "security": self.admin.security.read_request_count(),
            "change": self.admin.change.read_request_count(),
            "governance": self.admin.governance.read_request_count(),
        }

    def fail_security_provider(self) -> None:
        self.admin.security.set_read_behavior(SecurityStatusReadBehaviorV1.HTTP_503)
        self.admin.security.reset_read_request_count()

    def recover_security_provider(self) -> None:
        self.admin.security.set_read_behavior(SecurityStatusReadBehaviorV1.NORMAL)
        self.admin.security.reset_read_request_count()

    def malformed_security_provider(self) -> None:
        self.admin.security.set_read_behavior(SecurityStatusReadBehaviorV1.MALFORMED_JSON)
        self.admin.security.reset_read_request_count()

    def refresh_security_evidence(self, *, updated_at: datetime) -> None:
        self.admin.security.refresh_security_status(updated_at=updated_at)

    @staticmethod
    def is_satisfied(run: WorkspaceAskRunV2) -> bool:
        return (
            run.status is AskRunStatus.COMPLETED
            and run.evidence_admissibility is not None
            and run.evidence_admissibility.overall_status
            is EvidenceAdmissibilityStatusV1.SATISFIED
        )

    @staticmethod
    def is_unsatisfied(run: WorkspaceAskRunV2) -> bool:
        return (
            run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
            and run.evidence_admissibility is not None
            and run.evidence_admissibility.overall_status
            is EvidenceAdmissibilityStatusV1.UNSATISFIED
        )
