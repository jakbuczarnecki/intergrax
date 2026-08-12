# © Artur Czarnecki. All rights reserved.

"""Provider-neutral LLM planner and deterministic plan validation for LKW conversation."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Sequence

from pydantic import ValidationError

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

from local_workspace_application.conversation.interaction_draft_models import ConversationInteractionDraft
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    ExtractedObject,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddSourcesPlannedAction,
    KnowledgeCapabilitiesListPlannedAction,
    KnowledgeConnectionsListPlannedAction,
    KnowledgeResourcesListPlannedAction,
    TenantConnectionBeginAuthorizationPlannedAction,
    TenantConnectionCompleteManualAuthorizationPlannedAction,
    TenantConnectionConnectionsListPlannedAction,
    TenantConnectionInspectPlannedAction,
    TenantConnectionProvidersListPlannedAction,
    TenantConnectionReconnectPlannedAction,
    TenantConnectionRevokePlannedAction,
    WorkspaceReferenceKind,
    collect_user_text_context,
    request_attachment_ids,
)
from local_workspace_application.conversation.interaction_plan_compiler import (
    ConversationDraftCompilationError,
    ConversationDraftCompilationErrorCode,
    compile_interaction_draft,
)
from local_workspace_application.conversation.interaction_prompt import (
    RepairCategory,
    build_planning_messages,
)

logger = logging.getLogger(__name__)

_STRUCTURED_MAX_TOKENS = 8_192
_ALLOWED_SOURCE_OBJECT_TYPES = frozenset({"web_url", "local_file_reference"})


class ConversationPlanningErrorCode(str, Enum):
    conversation_planner_structured_output_unsupported = (
        "conversation_planner_structured_output_unsupported"
    )
    conversation_planner_invalid_output = "conversation_planner_invalid_output"
    conversation_planner_provider_failed = "conversation_planner_provider_failed"


class ConversationPlanningError(Exception):
    """Stable, safe planning failure without user content or provider details."""

    def __init__(
        self,
        code: ConversationPlanningErrorCode,
        *,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code.value)


class PlanRequestValidationError(ValueError):
    """Deterministic validation failure when a plan does not match the request."""


def _repair_category_for_error(exc: Exception) -> RepairCategory:
    if isinstance(exc, ValidationError):
        return RepairCategory.draft_contract

    if isinstance(exc, ConversationDraftCompilationError):
        mapping = {
            ConversationDraftCompilationErrorCode.source_value_not_found: (
                RepairCategory.source_value_not_grounded
            ),
            ConversationDraftCompilationErrorCode.source_occurrence_required: (
                RepairCategory.source_occurrence_required
            ),
            ConversationDraftCompilationErrorCode.source_occurrence_out_of_range: (
                RepairCategory.source_occurrence_required
            ),
            ConversationDraftCompilationErrorCode.invalid_action_reference: (
                RepairCategory.invalid_action_reference
            ),
            ConversationDraftCompilationErrorCode.self_action_reference: (
                RepairCategory.invalid_action_reference
            ),
            ConversationDraftCompilationErrorCode.invalid_created_workspace_reference: (
                RepairCategory.invalid_created_workspace_reference
            ),
            ConversationDraftCompilationErrorCode.ambiguous_created_workspace_reference: (
                RepairCategory.invalid_created_workspace_reference
            ),
            ConversationDraftCompilationErrorCode.conflicting_source_declaration: (
                RepairCategory.draft_contract
            ),
        }
        return mapping.get(exc.code, RepairCategory.draft_contract)

    if isinstance(exc, PlanRequestValidationError):
        return RepairCategory.canonical_request_grounding

    return RepairCategory.draft_contract


def validate_plan_against_request(
    plan: ConversationInteractionPlan,
    request: ConversationPlanningRequest,
) -> None:
    """Fail-closed validation of planner output against the safe planning request."""
    allowed_attachments = request_attachment_ids(request)
    user_contexts = collect_user_text_context(request)
    message_text = request.message_text
    object_map = {obj.object_id: obj for obj in plan.objects}

    for obj in plan.objects:
        _validate_extracted_object_evidence(obj, message_text)

    referenced_object_ids: set[str] = set()
    for action in plan.actions:
        for attachment_id in action.evidence_attachment_ids:
            if attachment_id not in allowed_attachments:
                raise PlanRequestValidationError("unknown evidence attachment ID")

        for quote in action.evidence_quotes:
            if not _evidence_quote_in_context(quote, user_contexts):
                raise PlanRequestValidationError("evidence quote not found in user context")

        if isinstance(action, KnowledgeAddAttachmentsPlannedAction):
            for attachment_id in action.attachment_ids:
                if attachment_id not in allowed_attachments:
                    raise PlanRequestValidationError("unknown attachment ID in action")

        if isinstance(action, KnowledgeAddSourcesPlannedAction):
            for source_id in action.source_object_ids:
                if source_id not in object_map:
                    raise PlanRequestValidationError("unknown source object ID in action")
                obj = object_map[source_id]
                if obj.object_type not in _ALLOWED_SOURCE_OBJECT_TYPES:
                    raise PlanRequestValidationError("invalid object type for knowledge.add_sources")
                referenced_object_ids.add(source_id)

        configuration = request.knowledge_plugin_configuration
        if isinstance(
            action,
            (
                KnowledgeConnectionsListPlannedAction,
                KnowledgeResourcesListPlannedAction,
                KnowledgeCapabilitiesListPlannedAction,
            ),
        ):
            if configuration is None:
                raise PlanRequestValidationError(
                    "knowledge configuration snapshot unavailable"
                )
            assert configuration is not None
        if isinstance(action, KnowledgeResourcesListPlannedAction):
            if configuration is None:
                raise PlanRequestValidationError(
                    "knowledge configuration snapshot unavailable"
                )
            connection = next(
                (
                    item
                    for item in configuration.available_connections
                    if item.connection_ref == action.connection_ref
                ),
                None,
            )
            if connection is None or connection.administrative_status.value != "active":
                raise PlanRequestValidationError("unknown knowledge connection reference")
            if action.source_kind not in connection.available_source_kinds:
                raise PlanRequestValidationError("unknown knowledge discovery selector")
            if action.page_token is not None:
                raise PlanRequestValidationError("unapproved knowledge pagination token")
        if isinstance(action, KnowledgeCapabilitiesListPlannedAction):
            if configuration is None:
                raise PlanRequestValidationError(
                    "knowledge configuration snapshot unavailable"
                )
            connection = next(
                (
                    item
                    for item in configuration.available_connections
                    if item.connection_ref == action.connection_ref
                ),
                None,
            )
            if connection is None or connection.administrative_status.value != "active":
                raise PlanRequestValidationError("unknown knowledge connection reference")
            if action.remote_resource_id is not None and not any(
                item.connection_ref == action.connection_ref
                and item.remote_resource_id == action.remote_resource_id
                for item in configuration.available_remote_resources
            ):
                raise PlanRequestValidationError("unknown knowledge resource reference")

        tenant_inventory = request.tenant_connection_inventory
        if isinstance(
            action,
            (
                TenantConnectionProvidersListPlannedAction,
                TenantConnectionConnectionsListPlannedAction,
                TenantConnectionInspectPlannedAction,
                TenantConnectionBeginAuthorizationPlannedAction,
                TenantConnectionCompleteManualAuthorizationPlannedAction,
                TenantConnectionReconnectPlannedAction,
                TenantConnectionRevokePlannedAction,
            ),
        ):
            if tenant_inventory is None:
                raise PlanRequestValidationError("tenant connection inventory unavailable")
        if isinstance(action, TenantConnectionBeginAuthorizationPlannedAction):
            if tenant_inventory is None:
                raise PlanRequestValidationError("tenant connection inventory unavailable")
            if not any(
                item.provider_id == action.provider_id for item in tenant_inventory.providers
            ):
                raise PlanRequestValidationError("unknown tenant connection provider reference")
        if isinstance(
            action,
            (
                TenantConnectionInspectPlannedAction,
                TenantConnectionReconnectPlannedAction,
                TenantConnectionRevokePlannedAction,
            ),
        ):
            if tenant_inventory is None:
                raise PlanRequestValidationError("tenant connection inventory unavailable")
            if not any(
                item.connection_ref == action.connection_ref
                for item in tenant_inventory.connections
            ):
                raise PlanRequestValidationError("unknown tenant connection reference")
        if isinstance(action, TenantConnectionCompleteManualAuthorizationPlannedAction):
            if tenant_inventory is None or tenant_inventory.pending_manual_authorization is None:
                raise PlanRequestValidationError("tenant connection manual authorization unavailable")

        workspace = getattr(action, "workspace", None)
        if workspace is not None and workspace.kind == WorkspaceReferenceKind.name:
            if workspace.value and not _evidence_quote_in_context(workspace.value, user_contexts):
                if not any(
                    workspace.value.lower() in context.lower() for context in user_contexts
                ):
                    raise PlanRequestValidationError("workspace name reference not in user context")

    for obj in plan.objects:
        if obj.object_id not in referenced_object_ids:
            raise PlanRequestValidationError("unused extracted object")


def _validate_extracted_object_evidence(obj: ExtractedObject, message_text: str) -> None:
    evidence = obj.evidence
    if evidence.source != "message_text":
        raise PlanRequestValidationError("evidence source must be message_text")
    if evidence.start < 0:
        raise PlanRequestValidationError("evidence start must be >= 0")
    if evidence.end <= evidence.start:
        raise PlanRequestValidationError("evidence end must be > start")
    if evidence.end > len(message_text):
        raise PlanRequestValidationError("evidence span out of range")
    if message_text[evidence.start : evidence.end] != evidence.text:
        raise PlanRequestValidationError("evidence text does not match message slice")
    if obj.value != evidence.text:
        raise PlanRequestValidationError("object value does not match evidence text")


def _evidence_quote_in_context(quote: str, contexts: Sequence[str]) -> bool:
    if not quote or not quote.strip():
        return False
    normalized_quote = quote.strip()
    for context in contexts:
        if normalized_quote in context:
            return True
    return False


class ConversationInteractionPlanner:
    """Provider-neutral planner: natural language → validated structured interaction plan."""

    def __init__(self, llm_adapter: LLMAdapter) -> None:
        self._llm_adapter = llm_adapter

    async def plan(
        self,
        request: ConversationPlanningRequest,
        *,
        run_id: str | None = None,
    ) -> ConversationInteractionPlan:
        if not self._llm_adapter.supports_structured_output():
            raise ConversationPlanningError(
                ConversationPlanningErrorCode.conversation_planner_structured_output_unsupported,
                retryable=False,
            )

        messages = build_planning_messages(request)
        try:
            return await self._plan_with_single_repair(messages, request, run_id=run_id)
        except ConversationPlanningError:
            raise
        except Exception:
            provider, model = self._adapter_identity()
            logger.warning(
                "conversation planner provider failure",
                extra={
                    "error_code": ConversationPlanningErrorCode.conversation_planner_provider_failed.value,
                    "adapter_provider": provider,
                    "adapter_model": model,
                },
            )
            raise ConversationPlanningError(
                ConversationPlanningErrorCode.conversation_planner_provider_failed,
                retryable=False,
            ) from None

    async def _plan_with_single_repair(
        self,
        messages: list[ChatMessage],
        request: ConversationPlanningRequest,
        *,
        run_id: str | None = None,
    ) -> ConversationInteractionPlan:
        last_validation_error: Exception | None = None
        current_messages = messages

        for attempt in range(2):
            try:
                draft = await self._generate_draft(current_messages, run_id=run_id)
                plan = _compile_draft(draft, request)
                validate_plan_against_request(plan, request)
                return plan
            except (ValidationError, ConversationDraftCompilationError, PlanRequestValidationError, TypeError, ValueError) as exc:
                last_validation_error = exc
                if attempt == 0:
                    repair_category = _repair_category_for_error(exc)
                    current_messages = build_planning_messages(
                        request,
                        include_repair_hint=True,
                        repair_category=repair_category,
                    )
                    continue
                break

        provider, model = self._adapter_identity()
        action_count = 0
        clarification_count = 0
        if isinstance(last_validation_error, ValidationError):
            pass
        logger.info(
            "conversation planner invalid output",
            extra={
                "error_code": ConversationPlanningErrorCode.conversation_planner_invalid_output.value,
                "action_count": action_count,
                "clarification_count": clarification_count,
                "adapter_provider": provider,
                "adapter_model": model,
            },
        )
        raise ConversationPlanningError(
            ConversationPlanningErrorCode.conversation_planner_invalid_output,
            retryable=False,
        ) from None

    async def _generate_draft(
        self,
        messages: list[ChatMessage],
        *,
        run_id: str | None,
    ) -> ConversationInteractionDraft:
        result = await asyncio.to_thread(
            self._llm_adapter.generate_structured,
            messages,
            ConversationInteractionDraft,
            temperature=0,
            max_tokens=_STRUCTURED_MAX_TOKENS,
            run_id=run_id,
        )
        parsed = result.parsed
        if not isinstance(parsed, ConversationInteractionDraft):
            raise TypeError("structured output is not ConversationInteractionDraft")
        return parsed

    def _adapter_identity(self) -> tuple[str, str]:
        provider = getattr(self._llm_adapter, "provider", "unknown")
        if hasattr(provider, "value"):
            provider = provider.value  # type: ignore[union-attr]
        model = getattr(self._llm_adapter, "model", "unknown")
        return str(provider), str(model)


def _compile_draft(
    draft: ConversationInteractionDraft,
    request: ConversationPlanningRequest,
) -> ConversationInteractionPlan:
    return compile_interaction_draft(draft, request)
