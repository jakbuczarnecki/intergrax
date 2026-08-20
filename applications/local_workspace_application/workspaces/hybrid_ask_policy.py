# © Artur Czarnecki. All rights reserved.

"""Provider-neutral Query Policy V2 resolution and Evidence Plan validation."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    QueryPolicyModeV2,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicy,
    WorkspaceQueryPolicyV2,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.contracts import (
    HARD_MAX_CONTENT_BYTES_PER_ITEM,
    HARD_MAX_PROVIDER_PAGE_SIZE,
    HARD_MAX_PROVIDER_PAGES,
    HARD_MAX_PROVIDER_REQUESTS,
    HARD_MAX_UPSTREAM_ITEMS,
    EffectiveLiveCallBudgetV1,
    KnowledgeQueryAudienceV1,
    ValidatedLiveCapabilityCallV1,
)
from intergrax.runtime.vendor_knowledge.live.identity import (
    validate_capability_identity,
)
from intergrax.runtime.vendor_knowledge.live.schemas import SchemaRegistryV1
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    PolicyEvidenceBasisV1,
    RequirementOriginV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    LiveCapabilityDescriptorV1,
    TenantLiveCapabilityCatalogPort,
    is_bindable_read_only_capability,
)

_FORBIDDEN_MODEL_CONTROLLED_FIELDS = frozenset(
    {
        "connection_ref",
        "provider_id",
        "integration_kind",
        "credentials",
        "credential_ref",
        "provider_endpoint",
        "endpoint",
        "raw_url",
        "url",
        "http_method",
        "headers",
        "jql",
        "sql",
        "dax",
        "provider_client",
    }
)


def _contains_forbidden_model_controlled_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in _FORBIDDEN_MODEL_CONTROLLED_FIELDS:
                return True
            if _contains_forbidden_model_controlled_field(nested):
                return True
    elif isinstance(value, list):
        for item in value:
            if _contains_forbidden_model_controlled_field(item):
                return True
    return False


def _reject_forbidden_model_controlled_fields(data: Any) -> Any:
    if isinstance(data, dict) and _contains_forbidden_model_controlled_field(data):
        raise ValueError("model_controlled_field_forbidden")
    return data


class HybridAskPolicyError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class EffectiveWorkspaceQueryPolicyV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: QueryPolicyModeV2
    allowed_connection_refs: tuple[str, ...] = ()
    allowed_capability_ids: tuple[str, ...] = ()
    max_live_calls: int = Field(default=0, ge=0, le=50)
    max_total_duration_ms: int = Field(default=30_000, ge=1, le=300_000)
    max_result_items: int = Field(default=50, ge=1, le=500)
    max_result_bytes: int = Field(default=1_048_576, ge=1, le=16_777_216)
    live_result_retention: LiveResultRetentionV1 = LiveResultRetentionV1.EPHEMERAL
    source_policy_schema_version: int | None = None


def _policy_mode_v1_to_v2(mode: QueryPolicyModeV1) -> QueryPolicyModeV2:
    if mode is QueryPolicyModeV1.INDEXED_ONLY:
        return QueryPolicyModeV2.INDEXED_ONLY
    return QueryPolicyModeV2.LIVE_ONLY


def _effective_from_v1(policy: WorkspaceQueryPolicy) -> EffectiveWorkspaceQueryPolicyV2:
    return EffectiveWorkspaceQueryPolicyV2(
        mode=_policy_mode_v1_to_v2(policy.mode),
        allowed_connection_refs=policy.allowed_connection_refs,
        allowed_capability_ids=policy.allowed_capability_ids,
        max_live_calls=policy.max_live_calls,
        max_total_duration_ms=policy.max_total_duration_ms,
        max_result_items=policy.max_result_items,
        max_result_bytes=policy.max_result_bytes,
        live_result_retention=policy.live_result_retention,
        source_policy_schema_version=None,
    )


def _effective_from_v2(policy: WorkspaceQueryPolicyV2) -> EffectiveWorkspaceQueryPolicyV2:
    return EffectiveWorkspaceQueryPolicyV2(
        mode=policy.mode,
        allowed_connection_refs=policy.allowed_connection_refs,
        allowed_capability_ids=policy.allowed_capability_ids,
        max_live_calls=policy.max_live_calls,
        max_total_duration_ms=policy.max_total_duration_ms,
        max_result_items=policy.max_result_items,
        max_result_bytes=policy.max_result_bytes,
        live_result_retention=policy.live_result_retention,
        source_policy_schema_version=2,
    )


def resolve_effective_query_policy(
    *,
    requested_mode: QueryPolicyModeV2,
    configuration: WorkspaceKnowledgeConfigurationV1,
    configuration_revision: int,
) -> EffectiveWorkspaceQueryPolicyV2:
    if configuration.configuration_revision != configuration_revision:
        raise HybridAskPolicyError("configuration_revision_mismatch")

    policy = configuration.query_policy
    if policy is None:
        if requested_mode is QueryPolicyModeV2.INDEXED_ONLY:
            return EffectiveWorkspaceQueryPolicyV2(mode=QueryPolicyModeV2.INDEXED_ONLY)
        raise HybridAskPolicyError("query_policy_required")

    if isinstance(policy, WorkspaceQueryPolicyV2):
        effective = _effective_from_v2(policy)
    else:
        effective = _effective_from_v1(policy)
        if requested_mode is QueryPolicyModeV2.HYBRID:
            raise HybridAskPolicyError("query_mode_not_allowed")

    if requested_mode is not effective.mode:
        raise HybridAskPolicyError("query_mode_not_allowed")

    return effective


class AudienceContextV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audience: KnowledgeQueryAudienceV1


class IndexedRetrievalDirectiveV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_results: int = Field(..., ge=1, le=500)


class LiveCallProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(..., min_length=1, max_length=128)
    live_access_binding_id: str = Field(..., min_length=1, max_length=128)
    capability_id: str = Field(..., min_length=1, max_length=128)
    typed_capability_request: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_forbidden_fields(cls, data: Any) -> Any:
        return _reject_forbidden_model_controlled_fields(data)


class IndexedEvidenceRequirementV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_type: Literal["indexed"] = "indexed"
    requirement_id: str = Field(..., min_length=1, max_length=128)
    semantic_role: str = Field(..., min_length=1, max_length=256)
    indexed_source_binding_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )
    policy_origin: RequirementOriginV1 | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_forbidden_fields(cls, data: Any) -> Any:
        return _reject_forbidden_model_controlled_fields(data)


class LiveEvidenceRequirementV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_type: Literal["live"] = "live"
    requirement_id: str = Field(..., min_length=1, max_length=128)
    semantic_role: str = Field(..., min_length=1, max_length=256)
    call_id: str = Field(..., min_length=1, max_length=128)
    policy_origin: RequirementOriginV1 | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_forbidden_fields(cls, data: Any) -> Any:
        return _reject_forbidden_model_controlled_fields(data)


RequiredEvidenceObligationV1 = Annotated[
    IndexedEvidenceRequirementV1 | LiveEvidenceRequirementV1,
    Field(discriminator="requirement_type"),
]


class ProviderEvidencePlanV1(BaseModel):
    """Provider-owned live proposals and authoritative evidence obligations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordered_live_call_proposals: tuple[LiveCallProposalV1, ...] = ()
    required_evidence_obligations: tuple[RequiredEvidenceObligationV1, ...] = ()


def derive_product_evidence_obligations(
    *,
    mode: QueryPolicyModeV2,
    include_indexed_retrieval: bool,
) -> tuple[RequiredEvidenceObligationV1, ...]:
    """Product-owned indexed admissibility obligation for generic HYBRID Workspace Ask.

    Planned live calls belong to the execution plan; mandatory per-call live evidence
    must be supplied explicitly by product/provider planning authority.
    """
    if mode is not QueryPolicyModeV2.HYBRID or not include_indexed_retrieval:
        return ()
    return (
        IndexedEvidenceRequirementV1(
            requirement_id="product:hybrid:indexed",
            semantic_role="Indexed corpus grounding",
        ),
    )


def compose_evidence_obligations(
    *,
    authoritative: tuple[RequiredEvidenceObligationV1, ...],
    additional: tuple[RequiredEvidenceObligationV1, ...],
) -> tuple[RequiredEvidenceObligationV1, ...]:
    """Additive strengthening; duplicate requirement_id fails closed."""
    seen: set[str] = set()
    composed: list[RequiredEvidenceObligationV1] = []
    for obligation in (*authoritative, *additional):
        if obligation.requirement_id in seen:
            raise HybridAskPolicyError("duplicate_requirement_id")
        seen.add(obligation.requirement_id)
        composed.append(obligation)
    return tuple(composed)


class EvidencePlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    configuration_revision: int = Field(..., ge=0)
    mode: QueryPolicyModeV2
    indexed_retrieval_directive: IndexedRetrievalDirectiveV1 | None = None
    ordered_live_call_proposals: tuple[LiveCallProposalV1, ...] = ()
    required_evidence_obligations: tuple[RequiredEvidenceObligationV1, ...] = ()
    policy_basis: PolicyEvidenceBasisV1 | None = None
    budget_snapshot: EffectiveLiveCallBudgetV1
    audience_context: AudienceContextV1

    @model_validator(mode="before")
    @classmethod
    def _reject_forbidden_fields(cls, data: Any) -> Any:
        return _reject_forbidden_model_controlled_fields(data)


class ResolvedLiveResourceScopeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_resource_id: str | None = None
    scope_token: str | None = None


class ExecutableLiveCallV1(ValidatedLiveCapabilityCallV1):
    resolved_resource_scope: ResolvedLiveResourceScopeV1

    @model_validator(mode="after")
    def _validate_call_identity(self) -> ExecutableLiveCallV1:
        self.assert_identity()
        if self.remote_resource_id != self.resolved_resource_scope.remote_resource_id:
            raise ValueError("live_resource_scope_identity_mismatch")
        return self


class ValidatedEvidencePlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: EvidencePlanV1
    executable_live_calls: tuple[ExecutableLiveCallV1, ...]
    effective_budget: EffectiveLiveCallBudgetV1


@runtime_checkable
class CapabilityRequestEnvelopeValidationPort(Protocol):
    def validate_request_envelope(
        self,
        *,
        descriptor: LiveCapabilityDescriptorV1,
        typed_request: dict[str, Any],
    ) -> BaseModel:
        ...


@runtime_checkable
class LiveResourceScopeValidationPort(Protocol):
    def validate_resource_scope(
        self,
        *,
        binding: WorkspaceLiveAccessBinding,
        capability_id: str,
        validated_request: BaseModel,
    ) -> ResolvedLiveResourceScopeV1:
        ...


def _audience_matches_binding(
  audience: KnowledgeQueryAudienceV1,
  eligibility: KnowledgeAudienceEligibilityV1,
) -> bool:
    if audience is KnowledgeQueryAudienceV1.SHARED:
        return eligibility is KnowledgeAudienceEligibilityV1.SHARED_ALLOWED
    return True


def _run_effective_budget(
    *,
    policy: EffectiveWorkspaceQueryPolicyV2,
    proposal_budget: EffectiveLiveCallBudgetV1,
) -> EffectiveLiveCallBudgetV1:
    return EffectiveLiveCallBudgetV1(
        max_live_calls=min(policy.max_live_calls, proposal_budget.max_live_calls),
        max_total_duration_ms=min(
            policy.max_total_duration_ms,
            proposal_budget.max_total_duration_ms,
        ),
        max_result_items=min(policy.max_result_items, proposal_budget.max_result_items),
        max_result_bytes=min(policy.max_result_bytes, proposal_budget.max_result_bytes),
        max_provider_pages=min(proposal_budget.max_provider_pages, HARD_MAX_PROVIDER_PAGES),
        max_provider_requests=min(
            proposal_budget.max_provider_requests, HARD_MAX_PROVIDER_REQUESTS
        ),
        max_upstream_items=min(proposal_budget.max_upstream_items, HARD_MAX_UPSTREAM_ITEMS),
        max_provider_page_size=min(
            proposal_budget.max_provider_page_size, HARD_MAX_PROVIDER_PAGE_SIZE
        ),
        max_content_bytes_per_item=min(
            proposal_budget.max_content_bytes_per_item,
            HARD_MAX_CONTENT_BYTES_PER_ITEM,
        ),
    )


def _per_call_effective_budget(
    *,
    run_budget: EffectiveLiveCallBudgetV1,
    descriptor: LiveCapabilityDescriptorV1,
) -> EffectiveLiveCallBudgetV1:
    return EffectiveLiveCallBudgetV1(
        max_live_calls=run_budget.max_live_calls,
        max_total_duration_ms=run_budget.max_total_duration_ms,
        max_result_items=min(
            run_budget.max_result_items,
            descriptor.max_result_items or run_budget.max_result_items,
        ),
        max_result_bytes=min(
            run_budget.max_result_bytes,
            descriptor.max_result_bytes,
        ),
        max_provider_pages=min(run_budget.max_provider_pages, descriptor.max_provider_pages),
        max_provider_requests=min(
            run_budget.max_provider_requests, descriptor.max_provider_requests
        ),
        max_upstream_items=min(run_budget.max_upstream_items, descriptor.max_upstream_items),
        max_provider_page_size=min(
            run_budget.max_provider_page_size, descriptor.max_provider_page_size
        ),
        max_content_bytes_per_item=min(
            run_budget.max_content_bytes_per_item,
            descriptor.max_content_bytes_per_item,
        ),
    )


def _resolve_live_capability_descriptor(
    *,
    capability_catalog: TenantLiveCapabilityCatalogPort,
    tenant_id: str,
    connection_ref: str,
    remote_resource_id: str | None,
    capability_id: str,
    provider_id: str,
    integration_kind: IntegrationCategory,
) -> LiveCapabilityDescriptorV1:
    descriptors = capability_catalog.list_capabilities(
        tenant_id=tenant_id,
        connection_ref=connection_ref,
        remote_resource_id=remote_resource_id,
    )
    matches = [
        descriptor
        for descriptor in descriptors
        if descriptor.capability_id == capability_id
        and descriptor.provider_id == provider_id
        and descriptor.integration_kind is integration_kind
    ]
    if not matches:
        raise HybridAskPolicyError("live_capability_unavailable")
    if len(matches) > 1:
        raise HybridAskPolicyError("live_capability_unavailable")
    descriptor = matches[0]
    if not is_bindable_read_only_capability(descriptor):
        raise HybridAskPolicyError("live_capability_unavailable")
    try:
        validate_capability_identity(
            capability_id=descriptor.capability_id,
            provider_id=descriptor.provider_id,
            integration_kind=descriptor.integration_kind,
            source_kind=descriptor.source_kind,
            contract_version=descriptor.contract_version,
        )
    except ValueError:
        raise HybridAskPolicyError("live_capability_unavailable") from None
    return descriptor


def _find_binding(
    configuration: WorkspaceKnowledgeConfigurationV1,
    live_access_binding_id: str,
) -> WorkspaceLiveAccessBinding | None:
    for binding in configuration.live_access_bindings:
        if binding.live_access_binding_id == live_access_binding_id:
            return binding
    return None


def _find_active_attachment(
    configuration: WorkspaceKnowledgeConfigurationV1,
    connection_ref: str,
) -> WorkspaceConnectionAttachment | None:
    for attachment in configuration.connection_attachments:
        if (
            attachment.connection_ref == connection_ref
            and attachment.status is WorkspaceConnectionAttachmentStatusV1.ATTACHED
        ):
            return attachment
    return None


def collect_policy_origins(
    obligations: tuple[RequiredEvidenceObligationV1, ...],
) -> tuple[RequirementOriginV1, ...]:
    origins: list[RequirementOriginV1] = []
    for obligation in obligations:
        if isinstance(obligation, IndexedEvidenceRequirementV1):
            if obligation.policy_origin is not None:
                origins.append(obligation.policy_origin)
        elif isinstance(obligation, LiveEvidenceRequirementV1):
            if obligation.policy_origin is not None:
                origins.append(obligation.policy_origin)
    return tuple(origins)


def validate_policy_basis_consistency(
    *,
    policy_basis: PolicyEvidenceBasisV1 | None,
    obligations: tuple[RequiredEvidenceObligationV1, ...],
) -> None:
    policy_origins = collect_policy_origins(obligations)
    if not policy_origins:
        return
    if policy_basis is None:
        raise HybridAskPolicyError("policy_basis_missing")
    revision_by_document = {
        reference.policy_document_id: reference.revision_id
        for reference in policy_basis.policy_revisions
    }
    for origin in policy_origins:
        basis_revision = revision_by_document.get(origin.policy_document_id)
        if basis_revision is None:
            raise HybridAskPolicyError("policy_origin_not_in_basis")
        if basis_revision != origin.revision_id:
            raise HybridAskPolicyError("policy_origin_revision_mismatch")


def _validate_required_evidence_obligations(
    *,
    plan: EvidencePlanV1,
    configuration: WorkspaceKnowledgeConfigurationV1,
    planned_call_ids: set[str],
) -> None:
    seen_requirement_ids: set[str] = set()
    indexed_binding_ids = {
        binding.indexed_source_binding_id for binding in configuration.indexed_sources
    }
    for obligation in plan.required_evidence_obligations:
        if obligation.requirement_id in seen_requirement_ids:
            raise HybridAskPolicyError("duplicate_requirement_id")
        seen_requirement_ids.add(obligation.requirement_id)
        if isinstance(obligation, IndexedEvidenceRequirementV1):
            if plan.mode is QueryPolicyModeV2.LIVE_ONLY:
                raise HybridAskPolicyError("required_evidence_mode_mismatch")
            if (
                obligation.indexed_source_binding_id is not None
                and obligation.indexed_source_binding_id not in indexed_binding_ids
            ):
                raise HybridAskPolicyError("indexed_binding_not_found")
        elif isinstance(obligation, LiveEvidenceRequirementV1):
            if plan.mode is QueryPolicyModeV2.INDEXED_ONLY:
                raise HybridAskPolicyError("required_evidence_mode_mismatch")
            if obligation.call_id not in planned_call_ids:
                raise HybridAskPolicyError("unknown_live_call_reference")


def validate_evidence_plan(
    *,
    plan: EvidencePlanV1,
    configuration: WorkspaceKnowledgeConfigurationV1,
    effective_policy: EffectiveWorkspaceQueryPolicyV2,
    capability_catalog: TenantLiveCapabilityCatalogPort,
    request_envelope_validator: CapabilityRequestEnvelopeValidationPort,
    resource_scope_validator: LiveResourceScopeValidationPort,
    schema_registry: SchemaRegistryV1 | None = None,
) -> ValidatedEvidencePlanV1:
    if plan.tenant_id != configuration.tenant_id:
        raise HybridAskPolicyError("tenant_workspace_mismatch")
    if plan.workspace_id != configuration.workspace_id:
        raise HybridAskPolicyError("tenant_workspace_mismatch")
    if plan.configuration_revision != configuration.configuration_revision:
        raise HybridAskPolicyError("configuration_revision_mismatch")

    if plan.mode is not effective_policy.mode:
        raise HybridAskPolicyError("query_mode_not_allowed")

    if plan.mode is QueryPolicyModeV2.INDEXED_ONLY:
        if plan.indexed_retrieval_directive is None:
            raise HybridAskPolicyError("indexed_directive_required")
        if plan.ordered_live_call_proposals:
            raise HybridAskPolicyError("live_proposals_forbidden")
    elif plan.mode is QueryPolicyModeV2.LIVE_ONLY:
        if plan.indexed_retrieval_directive is not None:
            raise HybridAskPolicyError("indexed_directive_forbidden")
        if not plan.ordered_live_call_proposals:
            raise HybridAskPolicyError("live_proposal_required")
    elif plan.mode is QueryPolicyModeV2.HYBRID:
        if plan.indexed_retrieval_directive is None:
            raise HybridAskPolicyError("indexed_directive_required")
        if not plan.ordered_live_call_proposals:
            raise HybridAskPolicyError("live_proposal_required")

    if plan.mode is not QueryPolicyModeV2.INDEXED_ONLY:
        run_effective_budget = _run_effective_budget(
            policy=effective_policy,
            proposal_budget=plan.budget_snapshot,
        )
        if len(plan.ordered_live_call_proposals) > run_effective_budget.max_live_calls:
            raise HybridAskPolicyError("live_call_budget_exceeded")

    executable_calls: list[ExecutableLiveCallV1] = []
    seen_call_ids: set[str] = set()
    if plan.mode is not QueryPolicyModeV2.INDEXED_ONLY:
        run_effective_budget = _run_effective_budget(
            policy=effective_policy,
            proposal_budget=plan.budget_snapshot,
        )
    else:
        run_effective_budget = EffectiveLiveCallBudgetV1(
            max_live_calls=0,
            max_total_duration_ms=min(
                plan.budget_snapshot.max_total_duration_ms,
                effective_policy.max_total_duration_ms,
            ),
            max_result_items=min(
                plan.budget_snapshot.max_result_items,
                effective_policy.max_result_items,
            ),
            max_result_bytes=min(
                plan.budget_snapshot.max_result_bytes,
                effective_policy.max_result_bytes,
            ),
            max_provider_pages=HARD_MAX_PROVIDER_PAGES,
            max_provider_requests=HARD_MAX_PROVIDER_REQUESTS,
            max_upstream_items=HARD_MAX_UPSTREAM_ITEMS,
            max_provider_page_size=HARD_MAX_PROVIDER_PAGE_SIZE,
            max_content_bytes_per_item=HARD_MAX_CONTENT_BYTES_PER_ITEM,
        )

    for proposal in plan.ordered_live_call_proposals:
        if proposal.call_id in seen_call_ids:
            raise HybridAskPolicyError("duplicate_call_id")
        seen_call_ids.add(proposal.call_id)

        binding = _find_binding(configuration, proposal.live_access_binding_id)
        if binding is None:
            raise HybridAskPolicyError("live_binding_not_found")
        if binding.status is not LiveAccessBindingStatusV1.ACTIVE:
            raise HybridAskPolicyError("live_binding_unavailable")

        if not _audience_matches_binding(
            plan.audience_context.audience,
            binding.audience_eligibility,
        ):
            raise HybridAskPolicyError("audience_mismatch")

        if proposal.capability_id not in binding.allowed_capability_ids:
            raise HybridAskPolicyError("live_capability_not_allowed")

        if proposal.capability_id not in effective_policy.allowed_capability_ids:
            raise HybridAskPolicyError("live_capability_not_allowed")

        attachment = _find_active_attachment(configuration, binding.connection_ref)
        if attachment is None:
            raise HybridAskPolicyError("live_binding_unavailable")

        if binding.connection_ref not in effective_policy.allowed_connection_refs:
            raise HybridAskPolicyError("live_capability_not_allowed")

        descriptor = _resolve_live_capability_descriptor(
            capability_catalog=capability_catalog,
            tenant_id=plan.tenant_id,
            connection_ref=binding.connection_ref,
            remote_resource_id=binding.remote_resource_id,
            capability_id=proposal.capability_id,
            provider_id=binding.derived_provider_id,
            integration_kind=binding.derived_integration_kind,
        )
        try:
            if schema_registry is not None:
                request_model = schema_registry.resolve_request(
                    descriptor.request_schema_ref, descriptor.contract_version
                )
                request_model.model_validate(proposal.typed_capability_request)
            validated_request = request_envelope_validator.validate_request_envelope(
                descriptor=descriptor,
                typed_request=proposal.typed_capability_request,
            )
            if not isinstance(validated_request, BaseModel):
                raise TypeError("live_request_model_required")
        except (ValidationError, TypeError, ValueError, LookupError):
            raise HybridAskPolicyError("live_request_invalid") from None
        try:
            resolved_scope = resource_scope_validator.validate_resource_scope(
                binding=binding,
                capability_id=proposal.capability_id,
                validated_request=validated_request,
            )
        except HybridAskPolicyError:
            raise
        except Exception:
            raise HybridAskPolicyError("live_resource_scope_invalid") from None
        call_effective_budget = _per_call_effective_budget(
            run_budget=run_effective_budget,
            descriptor=descriptor,
        )

        executable_calls.append(
            ExecutableLiveCallV1(
                call_id=proposal.call_id,
                live_access_binding_id=binding.live_access_binding_id,
                connection_ref=binding.connection_ref,
                provider_id=binding.derived_provider_id,
                integration_kind=binding.derived_integration_kind,
                capability_id=proposal.capability_id,
                contract_version=descriptor.contract_version,
                source_kind=descriptor.source_kind,
                remote_resource_id=binding.remote_resource_id,
                audience_context_ref=plan.audience_context.audience.value,
                validated_request=validated_request,
                resolved_resource_scope=resolved_scope,
                effective_budget=call_effective_budget,
            )
        )

    if executable_calls:
        effective_budget = run_effective_budget
    else:
        effective_budget = EffectiveLiveCallBudgetV1(
            max_live_calls=0,
            max_total_duration_ms=min(
                plan.budget_snapshot.max_total_duration_ms,
                effective_policy.max_total_duration_ms,
            ),
            max_result_items=min(
                plan.budget_snapshot.max_result_items,
                effective_policy.max_result_items,
            ),
            max_result_bytes=min(
                plan.budget_snapshot.max_result_bytes,
                effective_policy.max_result_bytes,
            ),
            max_provider_pages=HARD_MAX_PROVIDER_PAGES,
            max_provider_requests=HARD_MAX_PROVIDER_REQUESTS,
            max_upstream_items=HARD_MAX_UPSTREAM_ITEMS,
            max_provider_page_size=HARD_MAX_PROVIDER_PAGE_SIZE,
            max_content_bytes_per_item=HARD_MAX_CONTENT_BYTES_PER_ITEM,
        )

    _validate_required_evidence_obligations(
        plan=plan,
        configuration=configuration,
        planned_call_ids=seen_call_ids,
    )
    validate_policy_basis_consistency(
        policy_basis=plan.policy_basis,
        obligations=plan.required_evidence_obligations,
    )

    return ValidatedEvidencePlanV1(
        plan=plan,
        executable_live_calls=tuple(executable_calls),
        effective_budget=effective_budget,
    )


class KnowledgeQueryCommandV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(..., min_length=1, max_length=128)
    principal_id: str = Field(..., min_length=1, max_length=128)
    workspace_id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., min_length=1)
    audience_context: AudienceContextV1
    requested_evidence_mode: QueryPolicyModeV2
    configuration_revision: int = Field(..., ge=0)
    request_id: str = Field(..., min_length=1, max_length=128)

    @model_validator(mode="before")
    @classmethod
    def _reject_forbidden_fields(cls, data: Any) -> Any:
        return _reject_forbidden_model_controlled_fields(data)
