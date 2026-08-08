# © Artur Czarnecki. All rights reserved.

"""Application integration for the provider-neutral Workspace Ask V2 flow."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from local_workspace_application.workspaces.ask_models import (
    AskAnswerAssemblyError,
    AskAnswerAssemblyStatus,
    AskError,
    AskRunStatus,
)
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.hybrid_ask_answer_assembler import (
    HybridAskAnswerAssemblerV2,
)
from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeQueryExecutionResultV1,
    KnowledgeQueryOrchestratorV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    IndexedWorkspaceCitationV1,
    IndexedWorkspaceEvidenceV1,
    LiveWorkspaceCitationV1,
    LiveWorkspaceEvidenceV1,
    PersistedIndexedEvidenceV2,
    PersistedLiveEvidenceProvenanceV2,
    WorkspaceAskRunV2,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    AudienceContextV1,
    CapabilityRequestEnvelopeValidationPort,
    EffectiveLiveCallBudgetV1,
    EvidencePlanV1,
    HybridAskPolicyError,
    LiveCallProposalV1,
    LiveResourceScopeValidationPort,
    ResolvedLiveResourceScopeV1,
    ValidatedEvidencePlanV1,
    resolve_effective_query_policy,
    validate_evidence_plan,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveResultRetentionV1,
    QueryPolicyModeV2,
    WorkspaceLiveAccessBinding,
    WorkspaceKnowledgeConfigurationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.vendor_knowledge.live.schemas import SchemaRegistryV1
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    LiveCapabilityDescriptorV1,
    TenantLiveCapabilityCatalogPort,
)


class WorkspaceAskV2Error(RuntimeError):
    """Safe application error with a stable public code."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class WorkspaceAskV2LookupError(WorkspaceAskV2Error):
    """Workspace or run is not visible to the current tenant."""


class WorkspaceAskV2PersistenceError(RuntimeError):
    """A V2 run could not be durably persisted."""


class WorkspaceAskProviderStrategy(Protocol):
    """Optional provider-owned planning/expansion strategy for generic Ask."""

    def build_plan(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        request: Any,
    ) -> Any:
        ...

    def build_expansion(
        self,
        *,
        command: WorkspaceAskCommandV2,
        configuration: WorkspaceKnowledgeConfigurationV1,
        effective_policy: Any,
        validated_plan: ValidatedEvidencePlanV1,
    ) -> Any | None:
        ...

    def coverage(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        request: Any,
    ) -> Any:
        ...


class UnavailableTenantLiveCapabilityCatalog:
    """Empty normal-composition catalog; live requests fail closed."""

    def list_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        del tenant_id, connection_ref, remote_resource_id
        return ()


class SafeCapabilityRequestEnvelopeValidator:
    """Narrow default validator for a typed, already bounded logical envelope."""

    def __init__(self, schema_registry: SchemaRegistryV1 | None = None) -> None:
        self._schema_registry = schema_registry

    def validate_request_envelope(
        self,
        *,
        descriptor: LiveCapabilityDescriptorV1,
        typed_request: dict[str, Any],
    ) -> BaseModel:
        if self._schema_registry is None or not isinstance(typed_request, dict):
            raise HybridAskPolicyError("live_request_invalid")
        try:
            request_model = self._schema_registry.resolve_request(
                descriptor.request_schema_ref,
                descriptor.contract_version,
            )
            return request_model.model_validate(typed_request)
        except (ValidationError, LookupError, ValueError):
            raise HybridAskPolicyError("live_request_invalid") from None


class BindingResourceScopeValidator:
    """Resolve only the resource scope already committed on the binding."""

    def validate_resource_scope(
        self,
        *,
        binding: WorkspaceLiveAccessBinding,
        capability_id: str,
        validated_request: BaseModel,
    ) -> ResolvedLiveResourceScopeV1:
        del capability_id, validated_request
        return ResolvedLiveResourceScopeV1(
            remote_resource_id=binding.remote_resource_id,
            scope_token=None,
        )


class WorkspaceAskCommandV2(BaseModel):
    """Frontend-neutral command assembled before HTTP or conversation adapters."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., min_length=1)
    requested_mode: QueryPolicyModeV2
    audience_context: AudienceContextV1
    indexed_max_results: int | None = Field(default=None, ge=1, le=500)
    ordered_live_call_proposals: tuple[LiveCallProposalV1, ...] = ()
    provider_request: Any | None = None
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    run_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _question_not_blank(self) -> WorkspaceAskCommandV2:
        if not self.question.strip():
            raise ValueError("question_must_not_be_blank")
        if self.provider_request is not None and self.ordered_live_call_proposals:
            raise ValueError("provider_request_proposals_conflict")
        if (
            self.provider_request is not None
            and self.requested_mode is QueryPolicyModeV2.INDEXED_ONLY
        ):
            raise ValueError("provider_request_requires_live_mode")
        return self


class WorkspaceAskServiceV2:
    """Authorize, plan, execute, synthesize and retain one Ask V2 run."""

    def __init__(
        self,
        *,
        workspace_service: ManagedWorkspaceService,
        workspace_repository: ManagedWorkspaceRepository,
        ask_repository: WorkspaceAskRepository,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        capability_catalog: TenantLiveCapabilityCatalogPort,
        request_envelope_validator: CapabilityRequestEnvelopeValidationPort,
        resource_scope_validator: LiveResourceScopeValidationPort,
        orchestrator: KnowledgeQueryOrchestratorV1,
        schema_registry: SchemaRegistryV1 | None = None,
        llm_adapter: LLMAdapter | None = None,
        llm_adapter_factory: Callable[[], LLMAdapter] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        run_id_factory: Callable[[], str] = lambda: str(uuid4()),
        plan_id_factory: Callable[[], str] = lambda: str(uuid4()),
        provider_strategy: WorkspaceAskProviderStrategy | None = None,
    ) -> None:
        self._workspaces = workspace_service
        self._workspace_repository = workspace_repository
        self._ask_repository = ask_repository
        self._configuration_service = configuration_service
        self._capability_catalog = capability_catalog
        self._request_envelope_validator = request_envelope_validator
        self._resource_scope_validator = resource_scope_validator
        self._schema_registry = schema_registry
        self._orchestrator = orchestrator
        self._llm = llm_adapter
        self._llm_factory = llm_adapter_factory
        self._clock = clock
        self._run_id_factory = run_id_factory
        self._plan_id_factory = plan_id_factory
        self._provider_strategy = provider_strategy

    @property
    def llm_adapter(self) -> LLMAdapter:
        if self._llm is None:
            if self._llm_factory is not None:
                self._llm = self._llm_factory()
            else:
                from intergrax.runtime.wiring.llm_resolver import resolve_llm_adapter

                self._llm = resolve_llm_adapter(None)
        return self._llm

    @llm_adapter.setter
    def llm_adapter(self, adapter: LLMAdapter) -> None:
        self._llm = adapter

    async def ask(self, command: WorkspaceAskCommandV2) -> WorkspaceAskRunV2:
        workspace = self._workspaces.get_workspace(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
        )
        if workspace is None:
            raise WorkspaceAskV2LookupError("workspace_not_found")
        configuration = self._configuration_service.get_configuration(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
        )
        if configuration is None:
            raise WorkspaceAskV2LookupError("workspace_not_found")
        effective_policy = resolve_effective_query_policy(
            requested_mode=command.requested_mode,
            configuration=configuration,
            configuration_revision=configuration.configuration_revision,
        )
        plan = self._build_plan(command, configuration, effective_policy)
        validated_plan = validate_evidence_plan(
            plan=plan,
            configuration=configuration,
            effective_policy=effective_policy,
            capability_catalog=self._capability_catalog,
            request_envelope_validator=self._request_envelope_validator,
            resource_scope_validator=self._resource_scope_validator,
            schema_registry=self._schema_registry,
        )
        provider_expansion = self._build_provider_expansion(
            command=command,
            configuration=configuration,
            effective_policy=effective_policy,
            validated_plan=validated_plan,
        )
        provider_coverage = (
            self._provider_strategy.coverage(
                configuration=configuration,
                request=command.provider_request,
            )
            if command.provider_request is not None and self._provider_strategy is not None
            else None
        )
        run_id = command.run_id or self._run_id_factory()
        initial = self._initial_run(run_id, command, configuration, validated_plan)
        self._persist(initial)

        try:
            execution = await self._orchestrator.execute(
                run_id=run_id,
                question=command.question,
                validated_plan=validated_plan,
                retention=effective_policy.live_result_retention,
                live_expansion=provider_expansion,
            )
        except Exception:  # noqa: BLE001
            return self._finalize_failure(
                initial,
                code="live_execution_failed",
                execution=None,
                provider_coverage=provider_coverage,
            )

        if execution.error_code is not None:
            return self._finalize_failure(
                initial,
                code=execution.error_code,
                execution=execution,
                provider_coverage=provider_coverage,
            )

        evidence = tuple(execution.indexed_evidence) + tuple(execution.live_evidence)
        if provider_expansion is not None:
            include_evidence = getattr(provider_expansion, "include_evidence", None)
            if callable(include_evidence):
                evidence = include_evidence(evidence)
        try:
            assembler = HybridAskAnswerAssemblerV2(self.llm_adapter)
            assembly = assembler.assemble(question=command.question, evidence=evidence)
            if assembly.status is AskAnswerAssemblyStatus.INSUFFICIENT_EVIDENCE:
                return self._finalize_success(
                    initial,
                    configuration=configuration,
                    execution=execution,
                    answer=None,
                    citations=[],
                    status=AskRunStatus.INSUFFICIENT_EVIDENCE,
                    provider_coverage=provider_coverage,
                )
            citations = self._project_citations(
                assembly.used_evidence_ids,
                evidence=evidence,
                configuration=configuration,
                receipts=execution.receipts,
            )
            if command.requested_mode is QueryPolicyModeV2.HYBRID:
                types = {item.evidence_type for item in citations}
                if len(types) != 2:
                    raise WorkspaceAskV2Error("citation_validation_failed")
            if not citations:
                raise WorkspaceAskV2Error("citation_validation_failed")
            return self._finalize_success(
                initial,
                configuration=configuration,
                execution=execution,
                answer=assembly.answer,
                citations=citations,
                status=AskRunStatus.COMPLETED,
                provider_coverage=provider_coverage,
            )
        except WorkspaceAskV2Error as exc:
            return self._finalize_failure(
                initial,
                code=exc.error_code,
                execution=execution,
                provider_coverage=provider_coverage,
            )
        except AskAnswerAssemblyError as exc:
            return self._finalize_failure(
                initial,
                code=exc.code,
                execution=execution,
                provider_coverage=provider_coverage,
            )
        except Exception:  # noqa: BLE001
            return self._finalize_failure(
                initial,
                code="assembly_failed",
                execution=execution,
                provider_coverage=provider_coverage,
            )

    def get_run(self, *, tenant_id: str, run_id: str) -> WorkspaceAskRunV2:
        run = self._ask_repository.get_run_v2(tenant_id=tenant_id, run_id=run_id)
        if run is None:
            raise WorkspaceAskV2LookupError("ask_run_not_found")
        return run

    def _build_plan(
        self,
        command: WorkspaceAskCommandV2,
        configuration: WorkspaceKnowledgeConfigurationV1,
        effective_policy: Any,
    ) -> EvidencePlanV1:
        maximum = effective_policy.max_result_items
        requested = command.indexed_max_results or min(10, maximum)
        indexed_directive = None
        if command.requested_mode in (
            QueryPolicyModeV2.INDEXED_ONLY,
            QueryPolicyModeV2.HYBRID,
        ):
            indexed_directive = self._indexed_directive(min(requested, maximum))
        ordered_live_call_proposals = tuple(command.ordered_live_call_proposals)
        if command.provider_request is not None:
            if self._provider_strategy is None:
                raise HybridAskPolicyError("provider_strategy_unavailable")
            provider_plan = self._provider_strategy.build_plan(
                configuration=configuration,
                request=command.provider_request,
            )
            ordered_live_call_proposals = tuple(
                provider_plan.ordered_live_call_proposals
            )
        plan = EvidencePlanV1(
            plan_id=self._plan_id_factory(),
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            configuration_revision=configuration.configuration_revision,
            mode=command.requested_mode,
            indexed_retrieval_directive=indexed_directive,
            ordered_live_call_proposals=ordered_live_call_proposals,
            budget_snapshot=EffectiveLiveCallBudgetV1(
                max_live_calls=effective_policy.max_live_calls,
                max_total_duration_ms=effective_policy.max_total_duration_ms,
                max_result_items=effective_policy.max_result_items,
                max_result_bytes=effective_policy.max_result_bytes,
            ),
            audience_context=command.audience_context,
        )
        return plan

    def _build_provider_expansion(
        self,
        *,
        command: WorkspaceAskCommandV2,
        configuration: WorkspaceKnowledgeConfigurationV1,
        effective_policy: Any,
        validated_plan: ValidatedEvidencePlanV1,
    ) -> Any | None:
        if command.provider_request is None:
            return None
        if self._provider_strategy is None:
            raise HybridAskPolicyError("provider_strategy_unavailable")
        return self._provider_strategy.build_expansion(
            command=command,
            configuration=configuration,
            effective_policy=effective_policy,
            validated_plan=validated_plan,
        )

    @staticmethod
    def _indexed_directive(max_results: int) -> Any:
        from local_workspace_application.workspaces.hybrid_ask_policy import (
            IndexedRetrievalDirectiveV1,
        )

        return IndexedRetrievalDirectiveV1(max_results=max_results)

    def _initial_run(
        self,
        run_id: str,
        command: WorkspaceAskCommandV2,
        configuration: WorkspaceKnowledgeConfigurationV1,
        validated_plan: ValidatedEvidencePlanV1,
    ) -> WorkspaceAskRunV2:
        from local_workspace_application.workspaces.hybrid_ask_models import (
            HybridAskIndexedRetrievalStatusV1,
            HybridAskLiveExecutionStatusV1,
        )

        indexed_requested = validated_plan.plan.indexed_retrieval_directive is not None
        live_requested = bool(validated_plan.executable_live_calls)
        return WorkspaceAskRunV2(
            run_id=run_id,
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            question=command.question,
            status=AskRunStatus.FAILED,
            query_mode=command.requested_mode,
            configuration_revision=configuration.configuration_revision,
            plan_id=validated_plan.plan.plan_id,
            live_result_retention=self._live_retention(configuration),
            provider_coverage=(
                self._provider_strategy.coverage(
                    configuration=configuration,
                    request=command.provider_request,
                )
                if command.provider_request is not None
                and self._provider_strategy is not None
                else None
            ),
            created_at=self._clock(),
            indexed_retrieval_status=(
                HybridAskIndexedRetrievalStatusV1.FAILED
                if indexed_requested
                else HybridAskIndexedRetrievalStatusV1.SKIPPED
            ),
            live_execution_status=(
                HybridAskLiveExecutionStatusV1.FAILED
                if live_requested
                else HybridAskLiveExecutionStatusV1.SKIPPED
            ),
        )

    @staticmethod
    def _live_retention(
        configuration: WorkspaceKnowledgeConfigurationV1,
    ) -> LiveResultRetentionV1:
        if configuration.query_policy is None:
            return LiveResultRetentionV1.EPHEMERAL
        return configuration.query_policy.live_result_retention

    def _finalize_success(
        self,
        initial: WorkspaceAskRunV2,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        execution: KnowledgeQueryExecutionResultV1,
        answer: str | None,
        citations: list[Any],
        status: AskRunStatus,
        provider_coverage: Any = None,
    ) -> WorkspaceAskRunV2:
        return self._finalize_run_model(
            initial,
            update={
                "status": status,
                "answer": answer,
                "citations": citations,
                "persisted_evidence": self._project_persisted_evidence(execution),
                "execution_receipts": self._project_receipts(execution),
                "indexed_retrieval_status": execution.indexed_retrieval_status,
                "live_execution_status": execution.live_execution_status,
                "truncation_state": execution.truncation_state,
                "partial_failure": execution.partial_failure,
                "provider_coverage": provider_coverage,
                "completed_at": self._clock(),
                "error": None,
            },
        )

    def _finalize_failure(
        self,
        initial: WorkspaceAskRunV2,
        *,
        code: str,
        execution: KnowledgeQueryExecutionResultV1 | None,
        provider_coverage: Any = None,
    ) -> WorkspaceAskRunV2:
        if execution is None:
            update: dict[str, Any] = {
                "status": AskRunStatus.FAILED,
                "answer": None,
                "citations": [],
                "error": AskError(code=code, message=self._safe_message(code)),
                "completed_at": self._clock(),
                "provider_coverage": provider_coverage,
            }
        else:
            update = {
                "status": AskRunStatus.FAILED,
                "answer": None,
                "citations": [],
                "persisted_evidence": self._project_persisted_evidence(execution),
                "execution_receipts": self._project_receipts(execution),
                "indexed_retrieval_status": execution.indexed_retrieval_status,
                "live_execution_status": execution.live_execution_status,
                "truncation_state": execution.truncation_state,
                "partial_failure": execution.partial_failure,
                "provider_coverage": provider_coverage,
                "error": AskError(code=code, message=self._safe_message(code)),
                "completed_at": self._clock(),
            }
        return self._finalize_run_model(initial, update=update)

    def _finalize_run_model(
        self,
        initial: WorkspaceAskRunV2,
        *,
        update: dict[str, Any],
    ) -> WorkspaceAskRunV2:
        payload = initial.model_dump(mode="python")
        payload.update(update)
        try:
            finalized = WorkspaceAskRunV2.model_validate(payload)
        except ValidationError:
            safe_payload = initial.model_dump(mode="python")
            safe_payload.update(
                {
                    "status": AskRunStatus.FAILED,
                    "answer": None,
                    "citations": [],
                    "persisted_evidence": [],
                    "execution_receipts": [],
                    "completed_at": self._clock(),
                    "error": AskError(
                        code="citation_validation_failed",
                        message=self._safe_message("citation_validation_failed"),
                    ),
                }
            )
            finalized = WorkspaceAskRunV2.model_validate(safe_payload)
        self._persist(finalized)
        return finalized

    def _project_citations(
        self,
        used_ids: list[str],
        *,
        evidence: tuple[Any, ...],
        configuration: WorkspaceKnowledgeConfigurationV1,
        receipts: tuple[Any, ...],
    ) -> list[Any]:
        by_id = {item.evidence_id: item for item in evidence}
        receipt_by_call = {item.call_id: item for item in receipts}
        citations: list[Any] = []
        for evidence_id in used_ids:
            item = by_id.get(evidence_id)
            if item is None:
                raise WorkspaceAskV2Error("unknown_evidence_id")
            if isinstance(item, IndexedWorkspaceEvidenceV1):
                document = self._workspace_repository.get_document_ref(
                    tenant_id=item.tenant_id,
                    workspace_id=item.workspace_id,
                    document_id=item.document_id,
                )
                source = self._workspace_repository.get_source(
                    tenant_id=item.tenant_id,
                    workspace_id=item.workspace_id,
                    source_id=item.source_id,
                )
                if (
                    document is None
                    or source is None
                    or document.tenant_id != item.tenant_id
                    or document.workspace_id != item.workspace_id
                    or document.source_id != item.source_id
                    or source.tenant_id != item.tenant_id
                    or source.workspace_id != item.workspace_id
                    or source.source_id != item.source_id
                ):
                    raise WorkspaceAskV2Error("citation_validation_failed")
                citations.append(
                    IndexedWorkspaceCitationV1(
                        evidence_id=item.evidence_id,
                        safe_display_name=item.safe_display_name,
                        excerpt=item.content,
                        retrieved_at=item.retrieved_at,
                        document_id=item.document_id,
                        source_id=item.source_id,
                        workspace_id=item.workspace_id,
                        source_path=document.source_path,
                        file_name=document.file_name,
                        chunk_id=item.chunk_id,
                        score=item.score,
                        location=item.location,
                    )
                )
            elif isinstance(item, LiveWorkspaceEvidenceV1):
                binding = next(
                    (
                        value
                        for value in configuration.live_access_bindings
                        if value.live_access_binding_id == item.live_access_binding_id
                    ),
                    None,
                )
                attachment = next(
                    (
                        value
                        for value in configuration.connection_attachments
                        if value.connection_ref == item.connection_ref
                    ),
                    None,
                )
                if binding is None or attachment is None:
                    raise WorkspaceAskV2Error("citation_validation_failed")
                citations.append(
                    LiveWorkspaceCitationV1(
                        evidence_id=item.evidence_id,
                        safe_display_name=item.safe_display_name,
                        retrieved_at=item.retrieved_at,
                        provider_id=item.provider_id,
                        connection_safe_label=attachment.safe_display_label,
                        capability_id=item.capability_id,
                        remote_resource_id=item.remote_resource_id,
                        remote_item_id=item.remote_item_id,
                        remote_updated_at=item.remote_updated_at,
                        call_id=item.call_id,
                        receipt_id=(
                            receipt_by_call[item.call_id].receipt_id
                            if item.call_id in receipt_by_call
                            else None
                        ),
                    )
                )
            else:
                raise WorkspaceAskV2Error("citation_validation_failed")
        return citations

    @staticmethod
    def _project_persisted_evidence(
        execution: KnowledgeQueryExecutionResultV1,
    ) -> list[Any]:
        indexed = [
            PersistedIndexedEvidenceV2(
                evidence_id=item.evidence_id,
                safe_display_name=item.safe_display_name,
                retrieved_at=item.retrieved_at,
                content_hash=item.content_hash,
                audience=item.audience,
                source_id=item.source_id,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                location=item.location,
                score=item.score,
                safe_source_label=item.safe_source_label,
                indexed_source_binding_id=item.indexed_source_binding_id,
            )
            for item in execution.indexed_evidence
        ]
        live = [
            PersistedLiveEvidenceProvenanceV2(
                evidence_id=item.evidence_id,
                safe_display_name=item.safe_display_name,
                retrieved_at=item.retrieved_at,
                content_hash=item.content_hash,
                audience=item.audience,
                provider_id=item.provider_id,
                live_access_binding_id=item.live_access_binding_id,
                connection_ref=item.connection_ref,
                capability_id=item.capability_id,
                remote_resource_id=item.remote_resource_id,
                remote_item_id=item.remote_item_id,
                remote_updated_at=item.remote_updated_at,
                truncated=item.truncated,
                call_id=item.call_id,
            )
            for item in execution.live_evidence
        ]
        return indexed + live

    @staticmethod
    def _project_receipts(
        execution: KnowledgeQueryExecutionResultV1,
    ) -> list[Any]:
        live_call_ids = {item.call_id for item in execution.live_evidence}
        return [
            receipt
            for receipt in execution.receipts
            if receipt.call_id in live_call_ids
        ]

    def _persist(self, run: WorkspaceAskRunV2) -> None:
        try:
            self._ask_repository.put_run_v2(run)
        except Exception as exc:
            raise WorkspaceAskV2PersistenceError(
                "ask_run_persistence_failed"
            ) from exc

    @staticmethod
    def _safe_message(code: str) -> str:
        return {
            "indexed_retrieval_failed": "Indexed retrieval failed.",
            "live_execution_timeout": "Live execution timed out.",
            "live_capability_unavailable": "Live capability is unavailable.",
            "live_binding_unavailable": "Live binding is unavailable.",
            "unknown_evidence_id": "The answer referenced unknown evidence.",
            "citation_validation_failed": "Evidence citations could not be verified.",
            "assembly_failed": "Answer synthesis failed.",
        }.get(code, "Workspace Ask execution failed.")


__all__ = [
    "BindingResourceScopeValidator",
    "SafeCapabilityRequestEnvelopeValidator",
    "UnavailableTenantLiveCapabilityCatalog",
    "WorkspaceAskCommandV2",
    "WorkspaceAskProviderStrategy",
    "WorkspaceAskServiceV2",
    "WorkspaceAskV2Error",
    "WorkspaceAskV2LookupError",
    "WorkspaceAskV2PersistenceError",
]
