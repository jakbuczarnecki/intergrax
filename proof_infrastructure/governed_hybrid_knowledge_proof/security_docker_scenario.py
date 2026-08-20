# © Artur Czarnecki. All rights reserved.

"""Workspace Ask V2 security-only Docker scenario for COMM-5 F3-E-R1."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.security_status.knowledge_read import (
    SECURITY_STATUS_PROVIDER_ID,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.evidence.obligation_derivation import (
    DeterministicEvidenceObligationDerivation,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    RequireLiveEvidencePolicyRuleV1,
    RequireLiveEvidenceRuleParametersV1,
    ResolvedPolicyRuleV1,
    TypedCapabilityRequestEntryV1,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
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
from proof_infrastructure.controlled_project_status_service.seed import ORION_FIXTURE_PROJECT_ID
from proof_infrastructure.controlled_security_status_service.models import (
    SecurityStatusReadBehaviorV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.admin_port import (
    ControlledSecurityStatusAdminPort,
    SecurityStatusFixtureIdentityV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.docker_environment import (
    GovernedHybridDockerEnvironmentV1,
)

_DOCKER_TENANT_ID = "docker-security-proof"
_DOCKER_WORKSPACE_ID = "docker-security-workspace"
_DOCKER_CONNECTION_REF = "conn.docker.security-status"
_DOCKER_BINDING_ID = "binding-docker-security"
_DOCKER_POLICY_REV = "docker-security-proof-rev-1"
_DOCKER_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
_DOCKER_QUESTION = "Can the deployment proceed under current governance policy?"


class _SecurityOnlyPolicyRulesPort:
    def resolve_policy_rules(self, **_: object) -> tuple[ResolvedPolicyRuleV1, ...]:
        return (
            RequireLiveEvidencePolicyRuleV1(
                policy_document_id="security-policy",
                revision_id=_DOCKER_POLICY_REV,
                rule_id="RULE-SECURITY",
                parameters=RequireLiveEvidenceRuleParametersV1(
                    semantic_role="Security blocker status",
                    requirement_key="security",
                    capability_id=SECURITY_STATUS_READ_CAPABILITY_ID,
                    live_access_binding_id=_DOCKER_BINDING_ID,
                    live_call_descriptor_ref="security-read",
                    typed_capability_request=(
                        TypedCapabilityRequestEntryV1(
                            key="project_id",
                            value=ORION_FIXTURE_PROJECT_ID,
                        ),
                    ),
                ),
            ),
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
        if tenant_id == _DOCKER_TENANT_ID and workspace_id == _DOCKER_WORKSPACE_ID:
            return Workspace(
                workspace_id=_DOCKER_WORKSPACE_ID,
                tenant_id=_DOCKER_TENANT_ID,
                name="Docker Security Proof Workspace",
                created_at=_DOCKER_NOW,
                updated_at=_DOCKER_NOW,
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
        if tenant_id == _DOCKER_TENANT_ID and workspace_id == _DOCKER_WORKSPACE_ID:
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
        if tenant_id == _DOCKER_TENANT_ID and workspace_id == _DOCKER_WORKSPACE_ID:
            return self._configuration
        return None


class _SecurityCatalog:
    def list_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        del tenant_id, remote_resource_id
        if connection_ref != _DOCKER_CONNECTION_REF:
            return ()
        return (build_security_status_read_descriptor(),)


class _EmptyIndexedRetriever:
    async def retrieve(self, **_: object) -> tuple[()]:
        return ()


class _RecordingLLM(LLMAdapter):
    provider = "proof"
    model = "docker-security-deterministic"

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
                    "answer": "YES — security evidence satisfies deployment policy.",
                    "used_evidence_ids": used_ids,
                },
                ensure_ascii=False,
            )
        )


def _security_binding() -> WorkspaceLiveAccessBinding:
    return WorkspaceLiveAccessBinding(
        live_access_binding_id=_DOCKER_BINDING_ID,
        tenant_id=_DOCKER_TENANT_ID,
        workspace_id=_DOCKER_WORKSPACE_ID,
        connection_ref=_DOCKER_CONNECTION_REF,
        allowed_capability_ids=(SECURITY_STATUS_READ_CAPABILITY_ID,),
        derived_provider_id=SECURITY_STATUS_PROVIDER_ID,
        derived_integration_kind=IntegrationCategory.SECURITY_SCANNER,
        derived_safe_display_label="Docker Security Status",
        status=LiveAccessBindingStatusV1.ACTIVE,
        mutation_id="mutation-docker-security",
        effective_revision=1,
        semantic_identity_hash=sha256(_DOCKER_BINDING_ID.encode()).hexdigest(),
        created_at=_DOCKER_NOW,
        updated_at=_DOCKER_NOW,
    )


def _configuration(*, vendor_base_url: str) -> WorkspaceKnowledgeConfigurationV1:
    binding = _security_binding()
    return WorkspaceKnowledgeConfigurationV1(
        tenant_id=_DOCKER_TENANT_ID,
        workspace_id=_DOCKER_WORKSPACE_ID,
        configuration_revision=1,
        connection_attachments=(
            WorkspaceConnectionAttachment(
                attachment_id="attachment-docker-security",
                tenant_id=_DOCKER_TENANT_ID,
                workspace_id=_DOCKER_WORKSPACE_ID,
                connection_ref=_DOCKER_CONNECTION_REF,
                safe_display_label="Docker Security Status",
                status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
                mutation_id="mutation-attachment-docker-security",
                effective_revision=1,
                created_at=_DOCKER_NOW,
                updated_at=_DOCKER_NOW,
            ),
        ),
        indexed_sources=(),
        live_access_bindings=(binding,),
        query_policy=WorkspaceQueryPolicyV2(
            tenant_id=_DOCKER_TENANT_ID,
            workspace_id=_DOCKER_WORKSPACE_ID,
            mode=QueryPolicyModeV2.LIVE_ONLY,
            allowed_connection_refs=(_DOCKER_CONNECTION_REF,),
            allowed_capability_ids=(SECURITY_STATUS_READ_CAPABILITY_ID,),
            max_live_calls=1,
            max_total_duration_ms=30_000,
            max_result_items=10,
            max_result_bytes=1_048_576,
            live_result_retention=LiveResultRetentionV1.EPHEMERAL,
            mutation_id="mutation-policy-docker-security",
            effective_revision=1,
            updated_at=_DOCKER_NOW,
        ),
        updated_at=_DOCKER_NOW,
    )


async def build_governed_security_docker_scenario(
    environment: GovernedHybridDockerEnvironmentV1,
) -> GovernedSecurityDockerScenarioV1:
    configuration = _configuration(vendor_base_url=environment.vendor_base_url)
    document_store = InMemoryDocumentStore()
    connection_repository = DocumentStoreTenantConnectionRepository(document_store)
    connection = TenantConnection(
        connection_ref=_DOCKER_CONNECTION_REF,
        tenant_id=_DOCKER_TENANT_ID,
        provider_id=SECURITY_STATUS_PROVIDER_ID,
        integration_kind=IntegrationCategory.SECURITY_SCANNER,
        safe_display_name="Docker Security Status",
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        credential_ref="secret.docker.security-status",
        validated_secret_free_config={
            "base_url": environment.vendor_base_url,
            "timeout_seconds": 5.0,
        },
        configuration_version=1,
        created_at=_DOCKER_NOW,
        updated_at=_DOCKER_NOW,
    )
    connection_repository.create(connection)

    connection_registry = KnowledgeConnectionRegistry()
    TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=_RecordingSecretsStore(),
        integration_factory=build_default_vendor_knowledge_connection_factory_registry(),
        connection_registry=connection_registry,
    ).rehydrate_connection(
        tenant_id=_DOCKER_TENANT_ID,
        connection_ref=_DOCKER_CONNECTION_REF,
    )

    configuration_service = _ConfigurationService(configuration)
    authority = WorkspaceLiveAccessRuntimeAuthority(
        configuration_service=configuration_service,  # type: ignore[arg-type]
        tenant_connection_port=RepositoryTenantConnectionPort(connection_repository),
        capability_catalog=_SecurityCatalog(),  # type: ignore[arg-type]
    )
    published = build_vendor_knowledge_live_registration_registry().publish()
    orchestrator = KnowledgeQueryOrchestratorV1(
        indexed_retriever=_EmptyIndexedRetriever(),  # type: ignore[arg-type]
        live_executor=LiveCapabilityExecutorV1(
            published_registration=published,
            integration_resolver=KnowledgeConnectionRegistryIntegrationResolverV1(
                connection_registry
            ),
            runtime_authority=authority,
            clock=lambda: _DOCKER_NOW,
        ),
        clock=lambda: _DOCKER_NOW,
    )
    llm = _RecordingLLM()
    ask_repository = WorkspaceAskRepository(document_store)
    service = WorkspaceAskServiceV2(
        workspace_service=_WorkspaceAuthority(),  # type: ignore[arg-type]
        workspace_repository=_Repository(configuration),  # type: ignore[arg-type]
        ask_repository=ask_repository,
        configuration_service=configuration_service,  # type: ignore[arg-type]
        capability_catalog=_SecurityCatalog(),  # type: ignore[arg-type]
        request_envelope_validator=SafeCapabilityRequestEnvelopeValidator(
            schema_registry=published.schemas,
        ),
        resource_scope_validator=BindingResourceScopeValidator(),
        orchestrator=orchestrator,
        llm_adapter=llm,
        clock=lambda: _DOCKER_NOW,
        run_id_factory=lambda: "docker-security-proof-placeholder",
        plan_id_factory=lambda: "docker-security-plan",
        evidence_obligation_derivation_port=DeterministicEvidenceObligationDerivation(),
        resolved_policy_rules_port=_SecurityOnlyPolicyRulesPort(),
        schema_registry=published.schemas,
    )
    return GovernedSecurityDockerScenarioV1(
        environment=environment,
        admin=environment.admin,
        service=service,
        llm=llm,
        seeded_fixture=None,
    )


@dataclass(slots=True)
class GovernedSecurityDockerScenarioV1:
    environment: GovernedHybridDockerEnvironmentV1
    admin: ControlledSecurityStatusAdminPort
    service: WorkspaceAskServiceV2
    llm: _RecordingLLM
    seeded_fixture: SecurityStatusFixtureIdentityV1 | None

    def seed_baseline(self) -> SecurityStatusFixtureIdentityV1:
        self.admin.set_read_behavior(SecurityStatusReadBehaviorV1.NORMAL)
        self.admin.reset_read_request_count()
        fixture = self.admin.seed_security_status()
        self.seeded_fixture = fixture
        return fixture

    def fail_provider(self) -> None:
        self.admin.set_read_behavior(SecurityStatusReadBehaviorV1.HTTP_503)
        self.admin.reset_read_request_count()

    def recover_provider(self) -> None:
        self.admin.set_read_behavior(SecurityStatusReadBehaviorV1.NORMAL)
        self.admin.reset_read_request_count()

    async def ask(self, *, run_id: str, request_id: str) -> WorkspaceAskRunV2:
        self.llm.calls = 0
        return await self.service.ask(
            WorkspaceAskCommandV2(
                tenant_id=_DOCKER_TENANT_ID,
                workspace_id=_DOCKER_WORKSPACE_ID,
                question=_DOCKER_QUESTION,
                requested_mode=QueryPolicyModeV2.LIVE_ONLY,
                audience_context=AudienceContextV1(
                    audience=KnowledgeQueryAudienceV1.PERSONAL
                ),
                run_id=run_id,
                request_id=request_id,
            )
        )

    def reload_run(self, *, run_id: str) -> WorkspaceAskRunV2:
        reloaded = self.service.get_run(tenant_id=_DOCKER_TENANT_ID, run_id=run_id)
        if reloaded is None:
            raise RuntimeError(f"run_not_found: {run_id}")
        return reloaded

    def security_evaluation(self, run: WorkspaceAskRunV2):
        if run.evidence_admissibility is None:
            raise RuntimeError("evidence_admissibility_missing")
        return next(
            item
            for item in run.evidence_admissibility.requirement_evaluations
            if item.requirement_id.endswith(":security")
        )

    def fixture_identity(self) -> SecurityStatusFixtureIdentityV1:
        if self.seeded_fixture is None:
            raise RuntimeError("seeded_fixture_missing")
        return self.admin.read_safe_fixture_identity(
            project_id=self.seeded_fixture.project_id,
        )

    def vendor_read_count(self) -> int:
        return self.admin.read_request_count()

    def assert_same_fixture(self, run: WorkspaceAskRunV2) -> bool:
        fixture = self.fixture_identity()
        if self.seeded_fixture is None:
            return False
        if fixture.status != self.seeded_fixture.status:
            return False
        if fixture.updated_at != self.seeded_fixture.updated_at:
            return False
        security_eval = self.security_evaluation(run)
        return security_eval.status is RequirementEvaluationStatusV1.SATISFIED
