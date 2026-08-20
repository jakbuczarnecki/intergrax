# © Artur Czarnecki. All rights reserved.

"""Multi-provider Hybrid Ask integration tests (COMM-5F3-C / F3-C-R1)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.change_approval.knowledge_read import (
    CHANGE_APPROVAL_PROVIDER_ID,
    CHANGE_APPROVAL_SOURCE_KIND,
)
from intergrax.integrations.providers.governance_approval.knowledge_read import (
    GOVERNANCE_APPROVAL_PROVIDER_ID,
    GOVERNANCE_APPROVAL_SOURCE_KIND,
)
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
    PROJECT_STATUS_SOURCE_KIND,
)
from intergrax.integrations.providers.security_status.knowledge_read import (
    SECURITY_STATUS_PROVIDER_ID,
    SECURITY_STATUS_SOURCE_KIND,
)
from intergrax.runtime.evidence.obligation_derivation import (
    DeterministicEvidenceObligationDerivation,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    MaxAgeTemporalConstraintV1,
    RequireLiveEvidencePolicyRuleV1,
    RequireLiveEvidenceRuleParametersV1,
    ResolvedPolicyRuleV1,
    TypedCapabilityRequestEntryV1,
)
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
from intergrax.runtime.vendor_knowledge.live.contracts import evidence_id_for_call
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
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
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    LiveCapabilityDescriptorV1,
    RepositoryTenantConnectionPort,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.hybrid_ask_admissibility import (
    evaluate_evidence_admissibility,
)
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeConnectionRegistryIntegrationResolverV1,
    KnowledgeQueryOrchestratorV1,
    LiveCapabilityExecutorV1,
    WorkspaceIndexedEvidenceRetrieverV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    AskAudienceV1,
    EvidenceAdmissibilityStatusV1,
    EvidenceTypeV1,
    LiveWorkspaceEvidenceV1,
    RequirementAdmissibilityReasonCodeV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    KnowledgeQueryAudienceV1,
    LiveEvidenceRequirementV1,
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
from proof_infrastructure.controlled_change_approval_service.lifecycle import (
    ControlledChangeApprovalServer,
)
from proof_infrastructure.controlled_change_approval_service.seed import (
    ORION_FIXTURE_CHANGE_ID,
)
from proof_infrastructure.controlled_governance_approval_service.lifecycle import (
    ControlledGovernanceApprovalServer,
)
from proof_infrastructure.controlled_governance_approval_service.seed import (
    ORION_FIXTURE_SUBJECT_ID,
)
from proof_infrastructure.controlled_project_status_service.lifecycle import (
    ControlledProjectStatusServer,
)
from proof_infrastructure.controlled_security_status_service.lifecycle import (
    ControlledSecurityStatusServer,
)
from proof_infrastructure.controlled_security_status_service.models import (
    SecurityStatusReadBehaviorV1,
)
from proof_infrastructure.controlled_project_status_service.models import (
    ProjectBlockerStatusV1,
)
from proof_infrastructure.controlled_security_status_service.seed import (
    seed_orion_security_fixture,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_BLOCKER_ID,
    ORION_FIXTURE_PROJECT_ID,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
_TENANT = "tenant-multi-provider"
_WORKSPACE = "workspace-multi-provider"
_POLICY_REV = "17"

_CONN_READINESS = "conn.project-status"
_CONN_SECURITY = "conn.security-status"
_CONN_CHANGE = "conn.change-approval"
_CONN_GOVERNANCE = "conn.governance-approval"

_BINDING_READINESS = "binding-readiness"
_BINDING_SECURITY = "binding-security"
_BINDING_CHANGE = "binding-change"
_BINDING_GOVERNANCE = "binding-governance"

_CONNECTION_DESCRIPTORS: dict[str, tuple[LiveCapabilityDescriptorV1, ...]] = {
    _CONN_READINESS: (build_project_status_read_descriptor(),),
    _CONN_SECURITY: (build_security_status_read_descriptor(),),
    _CONN_CHANGE: (build_change_approval_read_descriptor(),),
    _CONN_GOVERNANCE: (build_governance_approval_read_descriptor(),),
}


def _live_rule(
    *,
    policy_document_id: str,
    rule_id: str,
    requirement_key: str,
    semantic_role: str,
    capability_id: str,
    live_access_binding_id: str,
    live_call_descriptor_ref: str,
    typed_request: tuple[TypedCapabilityRequestEntryV1, ...],
) -> RequireLiveEvidencePolicyRuleV1:
    return RequireLiveEvidencePolicyRuleV1(
        policy_document_id=policy_document_id,
        revision_id=_POLICY_REV,
        rule_id=rule_id,
        parameters=RequireLiveEvidenceRuleParametersV1(
            semantic_role=semantic_role,
            requirement_key=requirement_key,
            capability_id=capability_id,
            live_access_binding_id=live_access_binding_id,
            live_call_descriptor_ref=live_call_descriptor_ref,
            typed_capability_request=typed_request,
        ),
    )


def _deployment_policy_rules(
    *,
    security_max_age_seconds: int | None = None,
) -> tuple[ResolvedPolicyRuleV1, ...]:
    security_parameters = RequireLiveEvidenceRuleParametersV1(
        semantic_role="Security blocker status",
        requirement_key="security",
        capability_id=SECURITY_STATUS_READ_CAPABILITY_ID,
        live_access_binding_id=_BINDING_SECURITY,
        live_call_descriptor_ref="security-read",
        typed_capability_request=(
            TypedCapabilityRequestEntryV1(
                key="project_id",
                value=ORION_FIXTURE_PROJECT_ID,
            ),
        ),
        temporal_constraint=(
            None
            if security_max_age_seconds is None
            else MaxAgeTemporalConstraintV1(max_age_seconds=security_max_age_seconds)
        ),
    )
    return (
        _live_rule(
            policy_document_id="deployment-policy",
            rule_id="RULE-READINESS",
            requirement_key="readiness",
            semantic_role="Project readiness status",
            capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
            live_access_binding_id=_BINDING_READINESS,
            live_call_descriptor_ref="readiness-read",
            typed_request=(
                TypedCapabilityRequestEntryV1(
                    key="project_id",
                    value=ORION_FIXTURE_PROJECT_ID,
                ),
            ),
        ),
        RequireLiveEvidencePolicyRuleV1(
            policy_document_id="security-policy",
            revision_id=_POLICY_REV,
            rule_id="RULE-SECURITY",
            parameters=security_parameters,
        ),
        _live_rule(
            policy_document_id="change-policy",
            rule_id="RULE-CHANGE",
            requirement_key="change",
            semantic_role="Change approval status",
            capability_id=CHANGE_APPROVAL_READ_CAPABILITY_ID,
            live_access_binding_id=_BINDING_CHANGE,
            live_call_descriptor_ref="change-read",
            typed_request=(
                TypedCapabilityRequestEntryV1(
                    key="change_id",
                    value=ORION_FIXTURE_CHANGE_ID,
                ),
            ),
        ),
        _live_rule(
            policy_document_id="architecture-policy",
            rule_id="RULE-ARCH",
            requirement_key="architecture",
            semantic_role="Architecture approval status",
            capability_id=GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
            live_access_binding_id=_BINDING_GOVERNANCE,
            live_call_descriptor_ref="architecture-read",
            typed_request=(
                TypedCapabilityRequestEntryV1(
                    key="subject_id",
                    value=ORION_FIXTURE_SUBJECT_ID,
                ),
            ),
        ),
    )


class _FixedPolicyRulesPort:
    def resolve_policy_rules(self, **_: object) -> tuple[ResolvedPolicyRuleV1, ...]:
        return _deployment_policy_rules()


class _TemporalPolicyRulesPort:
    def __init__(self, *, security_max_age_seconds: int) -> None:
        self._security_max_age_seconds = security_max_age_seconds

    def resolve_policy_rules(self, **_: object) -> tuple[ResolvedPolicyRuleV1, ...]:
        return _deployment_policy_rules(
            security_max_age_seconds=self._security_max_age_seconds,
        )


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
        if tenant_id == _TENANT and workspace_id == _WORKSPACE:
            return Workspace(
                workspace_id=_WORKSPACE,
                tenant_id=_TENANT,
                name="Multi Provider Workspace",
                created_at=_NOW,
                updated_at=_NOW,
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
        if tenant_id == _TENANT and workspace_id == _WORKSPACE:
            return self.configuration
        return None


class _ConfigurationService:
    def __init__(self, configuration: WorkspaceKnowledgeConfigurationV1) -> None:
        self._configuration = configuration

    def get_configuration(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationV1 | None:
        if tenant_id == _TENANT and workspace_id == _WORKSPACE:
            return self._configuration
        return None


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


class _TenantConnectionPort:
    def __init__(self, connections: tuple[TenantConnection, ...]) -> None:
        self._connections = connections

    def get_connection(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> TenantConnection | None:
        for connection in self._connections:
            if (
                connection.tenant_id == tenant_id
                and connection.connection_ref == connection_ref
            ):
                return connection
        return None

    def list_connections(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[TenantConnection, ...]:
        del limit, administrative_status
        return tuple(
            connection
            for connection in self._connections
            if connection.tenant_id == tenant_id
        )


class _RecordingLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.used_ids: list[str] = []

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
        self.used_ids = used_ids
        return build_adapter_response(
            content=json.dumps(
                {
                    "status": "completed",
                    "answer": "Multi-provider evidence is structurally admissible.",
                    "used_evidence_ids": used_ids,
                }
            )
        )


class _EmptyIndexedRetriever:
    async def retrieve(self, **_: object) -> tuple[()]:
        return ()


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
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=connection_ref,
        allowed_capability_ids=(capability_id,),
        derived_provider_id=provider_id,
        derived_integration_kind=integration_kind,
        derived_safe_display_label=f"Binding {binding_id}",
        status=status,
        mutation_id=f"mutation-{binding_id}",
        effective_revision=1,
        semantic_identity_hash=sha256(binding_id.encode()).hexdigest(),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _attachment(connection_ref: str) -> WorkspaceConnectionAttachment:
    return WorkspaceConnectionAttachment(
        attachment_id=f"attachment-{connection_ref}",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=connection_ref,
        safe_display_label=f"Attachment {connection_ref}",
        status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
        mutation_id=f"mutation-{connection_ref}",
        effective_revision=1,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _configuration(
    *,
    bindings: tuple[WorkspaceLiveAccessBinding, ...],
) -> WorkspaceKnowledgeConfigurationV1:
    connection_refs = tuple({binding.connection_ref for binding in bindings})
    capability_ids = tuple(
        capability
        for binding in bindings
        for capability in binding.allowed_capability_ids
    )
    return WorkspaceKnowledgeConfigurationV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        connection_attachments=tuple(_attachment(ref) for ref in connection_refs),
        indexed_sources=(),
        live_access_bindings=bindings,
        query_policy=WorkspaceQueryPolicyV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            mode=QueryPolicyModeV2.LIVE_ONLY,
            allowed_connection_refs=connection_refs,
            allowed_capability_ids=capability_ids,
            max_live_calls=len(bindings),
            max_total_duration_ms=30_000,
            max_result_items=10,
            max_result_bytes=1_048_576,
            live_result_retention=LiveResultRetentionV1.EPHEMERAL,
            mutation_id="mutation-policy",
            effective_revision=1,
            updated_at=_NOW,
        ),
        updated_at=_NOW,
    )


def _register_connections(
    repository: DocumentStoreTenantConnectionRepository,
    *,
    project_status_url: str,
    security_status_url: str,
    change_approval_url: str,
    governance_approval_url: str,
) -> tuple[TenantConnection, ...]:
    connections = (
        TenantConnection(
            connection_ref=_CONN_READINESS,
            tenant_id=_TENANT,
            provider_id=PROJECT_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            safe_display_name="Project Status",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secret.project-status",
            validated_secret_free_config={
                "base_url": project_status_url,
                "timeout_seconds": 2.0,
            },
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        ),
        TenantConnection(
            connection_ref=_CONN_SECURITY,
            tenant_id=_TENANT,
            provider_id=SECURITY_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.SECURITY_SCANNER,
            safe_display_name="Security Status",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secret.security-status",
            validated_secret_free_config={
                "base_url": security_status_url,
                "timeout_seconds": 2.0,
            },
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        ),
        TenantConnection(
            connection_ref=_CONN_CHANGE,
            tenant_id=_TENANT,
            provider_id=CHANGE_APPROVAL_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            safe_display_name="Change Approval",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secret.change-approval",
            validated_secret_free_config={
                "base_url": change_approval_url,
                "timeout_seconds": 2.0,
            },
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        ),
        TenantConnection(
            connection_ref=_CONN_GOVERNANCE,
            tenant_id=_TENANT,
            provider_id=GOVERNANCE_APPROVAL_PROVIDER_ID,
            integration_kind=IntegrationCategory.WORKFLOW_ORCHESTRATOR,
            safe_display_name="Governance Approval",
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref="secret.governance-approval",
            validated_secret_free_config={
                "base_url": governance_approval_url,
                "timeout_seconds": 2.0,
            },
            configuration_version=1,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )
    for connection in connections:
        repository.create(connection)
    return connections


async def _build_service(
    *,
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
    bindings: tuple[WorkspaceLiveAccessBinding, ...],
    orchestrator: KnowledgeQueryOrchestratorV1 | None = None,
    configuration_service: _ConfigurationService | None = None,
    resolved_policy_rules_port: _FixedPolicyRulesPort | _TemporalPolicyRulesPort | None = None,
) -> tuple[
    WorkspaceAskServiceV2,
    _RecordingLLM,
    DocumentStoreTenantConnectionRepository,
    tuple[TenantConnection, ...],
]:
    configuration = _configuration(bindings=bindings)
    document_store = InMemoryDocumentStore()
    connection_repository = DocumentStoreTenantConnectionRepository(document_store)
    connections = _register_connections(
        connection_repository,
        project_status_url=project_status_server.base_url,
        security_status_url=security_status_server.base_url,
        change_approval_url=change_approval_server.base_url,
        governance_approval_url=governance_approval_server.base_url,
    )
    connection_registry = KnowledgeConnectionRegistry()
    rehydrator = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=_RecordingSecretsStore(),
        integration_factory=build_default_vendor_knowledge_connection_factory_registry(),
        connection_registry=connection_registry,
    )
    for connection in connections:
        rehydrator.rehydrate_connection(
            tenant_id=_TENANT,
            connection_ref=connection.connection_ref,
        )

    resolved_configuration_service = configuration_service or _ConfigurationService(
        configuration
    )
    authority = WorkspaceLiveAccessRuntimeAuthority(
        configuration_service=resolved_configuration_service,  # type: ignore[arg-type]
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
        clock=lambda: _NOW,
    )
    resolved_orchestrator = orchestrator or KnowledgeQueryOrchestratorV1(
        indexed_retriever=_EmptyIndexedRetriever(),  # type: ignore[arg-type]
        live_executor=executor,
        clock=lambda: _NOW,
    )
    llm = _RecordingLLM()
    ask_repository = WorkspaceAskRepository(document_store)
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=_Repository(configuration),  # type: ignore[arg-type]
        ask_repository=ask_repository,
        configuration_service=resolved_configuration_service,  # type: ignore[arg-type]
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
        request_envelope_validator=SafeCapabilityRequestEnvelopeValidator(
            schema_registry=published.schemas,
        ),
        resource_scope_validator=BindingResourceScopeValidator(),
        orchestrator=resolved_orchestrator,
        llm_adapter=llm,
        clock=lambda: _NOW,
        run_id_factory=lambda: "run-multi-provider",
        plan_id_factory=lambda: "plan-multi-provider",
        evidence_obligation_derivation_port=DeterministicEvidenceObligationDerivation(),
        resolved_policy_rules_port=resolved_policy_rules_port or _FixedPolicyRulesPort(),
        schema_registry=published.schemas,
    )
    return service, llm, connection_repository, connections


class _MutableConfigurationService(_ConfigurationService):
    def disable_binding(self, binding_id: str) -> None:
        updated_bindings = tuple(
            binding.model_copy(
                update={"status": LiveAccessBindingStatusV1.DISABLED},
            )
            if binding.live_access_binding_id == binding_id
            else binding
            for binding in self._configuration.live_access_bindings
        )
        self._configuration = self._configuration.model_copy(
            update={"live_access_bindings": updated_bindings},
        )


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


def _all_success_bindings() -> tuple[WorkspaceLiveAccessBinding, ...]:
    return (
        _binding(
            binding_id=_BINDING_READINESS,
            connection_ref=_CONN_READINESS,
            capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
            provider_id=PROJECT_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        ),
        _binding(
            binding_id=_BINDING_SECURITY,
            connection_ref=_CONN_SECURITY,
            capability_id=SECURITY_STATUS_READ_CAPABILITY_ID,
            provider_id=SECURITY_STATUS_PROVIDER_ID,
            integration_kind=IntegrationCategory.SECURITY_SCANNER,
        ),
        _binding(
            binding_id=_BINDING_CHANGE,
            connection_ref=_CONN_CHANGE,
            capability_id=CHANGE_APPROVAL_READ_CAPABILITY_ID,
            provider_id=CHANGE_APPROVAL_PROVIDER_ID,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        ),
        _binding(
            binding_id=_BINDING_GOVERNANCE,
            connection_ref=_CONN_GOVERNANCE,
            capability_id=GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
            provider_id=GOVERNANCE_APPROVAL_PROVIDER_ID,
            integration_kind=IntegrationCategory.WORKFLOW_ORCHESTRATOR,
        ),
    )


@pytest.fixture
def project_status_server() -> ControlledProjectStatusServer:
    server = ControlledProjectStatusServer.start()
    yield server
    server.stop()


@pytest.fixture
def security_status_server() -> ControlledSecurityStatusServer:
    server = ControlledSecurityStatusServer.start()
    yield server
    server.stop()


@pytest.fixture
def change_approval_server() -> ControlledChangeApprovalServer:
    server = ControlledChangeApprovalServer.start()
    yield server
    server.stop()


@pytest.fixture
def governance_approval_server() -> ControlledGovernanceApprovalServer:
    server = ControlledGovernanceApprovalServer.start()
    yield server
    server.stop()


@pytest.mark.asyncio
async def test_multi_provider_derived_plan_executes_four_distinct_providers(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    httpx.put(
        f"{project_status_server.base_url}/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {
                    "id": ORION_FIXTURE_BLOCKER_ID,
                    "status": ProjectBlockerStatusV1.CLOSED.value,
                }
            ]
        },
        timeout=2.0,
    )
    project_status_server.store.reset_read_request_count()
    security_status_server.store.reset_read_request_count()
    change_approval_server.store.reset_read_request_count()
    governance_approval_server.store.reset_read_request_count()

    service, llm, _, connections = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
    )
    command = WorkspaceAskCommandV2(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        question="Can the deployment proceed under current governance policy?",
        requested_mode=QueryPolicyModeV2.LIVE_ONLY,
        audience_context=AudienceContextV1(
            audience=KnowledgeQueryAudienceV1.PERSONAL
        ),
        run_id="run-multi-provider-success",
        request_id="request-multi-provider-success",
    )

    run = await service.ask(command)

    assert run.status is AskRunStatus.COMPLETED
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
    )
    assert llm.calls == 1
    assert run.policy_basis is not None
    assert len(run.required_evidence_obligations) == 4
    call_ids = {
        obligation.call_id
        for obligation in run.required_evidence_obligations
        if isinstance(obligation, LiveEvidenceRequirementV1)
    }
    assert len(call_ids) == 4
    live_evidence = [
        item for item in run.persisted_evidence if item.evidence_type is EvidenceTypeV1.LIVE
    ]
    connection_refs = {item.connection_ref for item in live_evidence}
    capability_ids = {item.capability_id for item in live_evidence}
    provider_ids = {item.provider_id for item in live_evidence}
    assert len(connection_refs) == 4
    assert len(capability_ids) == 4
    assert len(provider_ids) == 4
    upstream_base_urls = {
        connection.validated_secret_free_config["base_url"] for connection in connections
    }
    assert len(upstream_base_urls) == 4
    assert upstream_base_urls == {
        project_status_server.base_url,
        security_status_server.base_url,
        change_approval_server.base_url,
        governance_approval_server.base_url,
    }
    assert project_status_server.store.read_request_count() == 1
    assert security_status_server.store.read_request_count() == 1
    assert change_approval_server.store.read_request_count() == 1
    assert governance_approval_server.store.read_request_count() == 1
    for obligation in run.required_evidence_obligations:
        if isinstance(obligation, LiveEvidenceRequirementV1):
            assert obligation.policy_origin is not None


@pytest.mark.asyncio
async def test_multi_provider_security_upstream_failure_blocks_synthesis(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    httpx.put(
        f"{project_status_server.base_url}/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {
                    "id": ORION_FIXTURE_BLOCKER_ID,
                    "status": ProjectBlockerStatusV1.CLOSED.value,
                }
            ]
        },
        timeout=2.0,
    )
    httpx.put(
        f"{security_status_server.base_url}/control/read-behavior",
        json={"behavior": SecurityStatusReadBehaviorV1.HTTP_503.value},
        timeout=2.0,
    )
    project_status_server.store.reset_read_request_count()
    security_status_server.store.reset_read_request_count()
    change_approval_server.store.reset_read_request_count()
    governance_approval_server.store.reset_read_request_count()

    service, llm, _, _ = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
    )
    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-multi-provider-security-failure",
            request_id="request-multi-provider-security-failure",
        )
    )

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is not EvidenceAdmissibilityStatusV1.SATISFIED
    )
    assert project_status_server.store.read_request_count() == 1
    assert security_status_server.store.read_request_count() == 1
    assert change_approval_server.store.read_request_count() == 1
    assert governance_approval_server.store.read_request_count() == 1


@pytest.mark.asyncio
async def test_multi_provider_revoked_binding_blocks_synthesis_without_provider_http(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    bindings = _all_success_bindings()
    configuration_service = _MutableConfigurationService(_configuration(bindings=bindings))
    document_store = InMemoryDocumentStore()
    connection_repository = DocumentStoreTenantConnectionRepository(document_store)
    connections = _register_connections(
        connection_repository,
        project_status_url=project_status_server.base_url,
        security_status_url=security_status_server.base_url,
        change_approval_url=change_approval_server.base_url,
        governance_approval_url=governance_approval_server.base_url,
    )
    connection_registry = KnowledgeConnectionRegistry()
    rehydrator = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=_RecordingSecretsStore(),
        integration_factory=build_default_vendor_knowledge_connection_factory_registry(),
        connection_registry=connection_registry,
    )
    for connection in connections:
        rehydrator.rehydrate_connection(
            tenant_id=_TENANT,
            connection_ref=connection.connection_ref,
        )
    published = build_vendor_knowledge_live_registration_registry().publish()
    authority = WorkspaceLiveAccessRuntimeAuthority(
        configuration_service=configuration_service,  # type: ignore[arg-type]
        tenant_connection_port=RepositoryTenantConnectionPort(connection_repository),
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
    )
    executor = LiveCapabilityExecutorV1(
        published_registration=published,
        integration_resolver=KnowledgeConnectionRegistryIntegrationResolverV1(
            connection_registry
        ),
        runtime_authority=authority,
        clock=lambda: _NOW,
    )
    inner_orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=_EmptyIndexedRetriever(),  # type: ignore[arg-type]
        live_executor=executor,
        clock=lambda: _NOW,
    )
    orchestrator = _RevokingOrchestrator(
        inner=inner_orchestrator,
        configuration_service=configuration_service,
        binding_id=_BINDING_GOVERNANCE,
    )
    project_status_server.store.reset_read_request_count()
    security_status_server.store.reset_read_request_count()
    change_approval_server.store.reset_read_request_count()
    governance_approval_server.store.reset_read_request_count()
    llm = _RecordingLLM()
    ask_repository = WorkspaceAskRepository(document_store)
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=_Repository(configuration_service._configuration),  # type: ignore[arg-type]
        ask_repository=ask_repository,
        configuration_service=configuration_service,  # type: ignore[arg-type]
        capability_catalog=_Catalog(),  # type: ignore[arg-type]
        request_envelope_validator=SafeCapabilityRequestEnvelopeValidator(
            schema_registry=published.schemas,
        ),
        resource_scope_validator=BindingResourceScopeValidator(),
        orchestrator=orchestrator,  # type: ignore[arg-type]
        llm_adapter=llm,
        clock=lambda: _NOW,
        run_id_factory=lambda: "run-multi-provider-revoked",
        plan_id_factory=lambda: "plan-multi-provider-revoked",
        evidence_obligation_derivation_port=DeterministicEvidenceObligationDerivation(),
        resolved_policy_rules_port=_FixedPolicyRulesPort(),
        schema_registry=published.schemas,
    )

    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-multi-provider-revoked",
            request_id="request-multi-provider-revoked",
        )
    )

    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert governance_approval_server.store.read_request_count() == 0
    assert project_status_server.store.read_request_count() >= 1
    assert security_status_server.store.read_request_count() >= 1
    assert change_approval_server.store.read_request_count() >= 1


def test_multi_provider_wrong_call_id_cannot_cross_satisfy_obligation() -> None:
    readiness = LiveEvidenceRequirementV1(
        requirement_id="policy:deployment-policy:RULE-READINESS:readiness",
        semantic_role="Project readiness status",
        call_id="policy-call:deployment-policy:RULE-READINESS:readiness-read",
    )
    security = LiveEvidenceRequirementV1(
        requirement_id="policy:security-policy:RULE-SECURITY:security",
        semantic_role="Security blocker status",
        call_id="policy-call:security-policy:RULE-SECURITY:security-read",
    )
    security_only = LiveWorkspaceEvidenceV1(
        evidence_id="live:security-only",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        safe_display_name="Security evidence",
        retrieved_at=_NOW,
        content='{"status": "clear"}',
        content_hash=sha256(b"security").hexdigest(),
        audience=AskAudienceV1.PERSONAL,
        call_id=security.call_id,
        live_access_binding_id=_BINDING_SECURITY,
        connection_ref=_CONN_SECURITY,
        provider_id=SECURITY_STATUS_PROVIDER_ID,
        integration_kind=IntegrationCategory.SECURITY_SCANNER,
        capability_id=SECURITY_STATUS_READ_CAPABILITY_ID,
        contract_version="1",
        source_kind=SECURITY_STATUS_SOURCE_KIND,
        remote_item_id=f"security:{ORION_FIXTURE_PROJECT_ID}:status",
    )
    result = evaluate_evidence_admissibility(
        obligations=(readiness, security),
        indexed_evidence=(),
        live_evidence=(security_only,),
        evaluated_at=_NOW,
    )
    readiness_eval = next(
        item
        for item in result.requirement_evaluations
        if item.requirement_id == readiness.requirement_id
    )
    assert readiness_eval.status.value == "unsatisfied"
    assert (
        readiness_eval.reason_code
        is RequirementAdmissibilityReasonCodeV1.LIVE_CALL_MISMATCH
    )


async def _seed_fresh_provider_snapshots(
    *,
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
    security_updated_at: datetime,
) -> None:
    httpx.put(
        f"{project_status_server.base_url}/control/projects/{ORION_FIXTURE_PROJECT_ID}/status",
        json={
            "blockers": [
                {
                    "id": ORION_FIXTURE_BLOCKER_ID,
                    "status": ProjectBlockerStatusV1.CLOSED.value,
                }
            ]
        },
        timeout=2.0,
    )
    seed_orion_security_fixture(
        security_status_server.store,
        updated_at=security_updated_at,
    )


@pytest.mark.asyncio
async def test_multi_provider_all_http_success_can_be_temporally_unsatisfied(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    await _seed_fresh_provider_snapshots(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        security_updated_at=_NOW - timedelta(hours=2),
    )
    service, llm, _, _ = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
        resolved_policy_rules_port=_TemporalPolicyRulesPort(
            security_max_age_seconds=3_600,
        ),
    )
    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-multi-provider-temporal-fail",
            request_id="request-multi-provider-temporal-fail",
        )
    )
    assert run.status is AskRunStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == 0
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.UNSATISFIED
    )
    security_eval = next(
        item
        for item in run.evidence_admissibility.requirement_evaluations
        if item.requirement_id.endswith(":security")
    )
    assert (
        security_eval.reason_code
        is RequirementAdmissibilityReasonCodeV1.EVIDENCE_TEMPORALLY_INVALID
    )
    assert project_status_server.store.read_request_count() >= 1
    assert security_status_server.store.read_request_count() >= 1
    assert change_approval_server.store.read_request_count() >= 1
    assert governance_approval_server.store.read_request_count() >= 1


@pytest.mark.asyncio
async def test_multi_provider_all_temporally_valid_reaches_llm(
    project_status_server: ControlledProjectStatusServer,
    security_status_server: ControlledSecurityStatusServer,
    change_approval_server: ControlledChangeApprovalServer,
    governance_approval_server: ControlledGovernanceApprovalServer,
) -> None:
    await _seed_fresh_provider_snapshots(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        security_updated_at=_NOW - timedelta(minutes=30),
    )
    service, llm, _, _ = await _build_service(
        project_status_server=project_status_server,
        security_status_server=security_status_server,
        change_approval_server=change_approval_server,
        governance_approval_server=governance_approval_server,
        bindings=_all_success_bindings(),
        resolved_policy_rules_port=_TemporalPolicyRulesPort(
            security_max_age_seconds=3_600,
        ),
    )
    run = await service.ask(
        WorkspaceAskCommandV2(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            question="Can the deployment proceed under current governance policy?",
            requested_mode=QueryPolicyModeV2.LIVE_ONLY,
            audience_context=AudienceContextV1(
                audience=KnowledgeQueryAudienceV1.PERSONAL
            ),
            run_id="run-multi-provider-temporal-pass",
            request_id="request-multi-provider-temporal-pass",
        )
    )
    assert run.status is AskRunStatus.COMPLETED
    assert llm.calls == 1
    assert run.evidence_admissibility is not None
    assert (
        run.evidence_admissibility.overall_status
        is EvidenceAdmissibilityStatusV1.SATISFIED
    )
