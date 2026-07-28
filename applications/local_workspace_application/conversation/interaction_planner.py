# © Artur Czarnecki. All rights reserved.

"""Provider-neutral LLM planner and deterministic plan validation for LKW conversation."""

from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum
from typing import Sequence

from pydantic import ValidationError

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddLocalReferencesPlannedAction,
    KnowledgeAddWebUrlsPlannedAction,
    WorkspaceReferenceKind,
    collect_user_text_context,
    request_attachment_ids,
)
from local_workspace_application.conversation.interaction_prompt import build_planning_messages

logger = logging.getLogger(__name__)

_STRUCTURED_MAX_TOKENS = 8_192
_TRAILING_PUNCTUATION = ".,;:!?)\"']"


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


def validate_plan_against_request(
    plan: ConversationInteractionPlan,
    request: ConversationPlanningRequest,
) -> None:
    """Fail-closed validation of planner output against the safe planning request."""
    allowed_attachments = request_attachment_ids(request)
    user_contexts = collect_user_text_context(request)

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

        if isinstance(action, KnowledgeAddWebUrlsPlannedAction):
            for url in action.urls:
                if not _url_in_context(url, user_contexts):
                    raise PlanRequestValidationError("URL not found in user context")

        if isinstance(action, KnowledgeAddLocalReferencesPlannedAction):
            for reference in action.references:
                if not _local_reference_in_context(reference.value, user_contexts):
                    raise PlanRequestValidationError("local reference not found in user context")

        workspace = getattr(action, "workspace", None)
        if workspace is not None and workspace.kind == WorkspaceReferenceKind.name:
            if workspace.value and not _evidence_quote_in_context(workspace.value, user_contexts):
                # Name references should reflect user wording; allow fuzzy via substring check.
                if not any(
                    workspace.value.lower() in context.lower() for context in user_contexts
                ):
                    raise PlanRequestValidationError("workspace name reference not in user context")


def _evidence_quote_in_context(quote: str, contexts: Sequence[str]) -> bool:
    if not quote or not quote.strip():
        return False
    normalized_quote = quote.strip()
    for context in contexts:
        if normalized_quote in context:
            return True
    return False


def _url_in_context(url: str, contexts: Sequence[str]) -> bool:
    if not url:
        return False
    for context in contexts:
        if url in context:
            return True
        pattern = re.escape(url) + rf"[{_re_escape_punctuation_class()}]?"
        if re.search(pattern, context):
            return True
    return False


def _re_escape_punctuation_class() -> str:
    return re.escape(_TRAILING_PUNCTUATION)


def _looks_like_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:\\", value)) or "\\" in value


def _local_reference_in_context(reference: str, contexts: Sequence[str]) -> bool:
    if not reference:
        return False
    for context in contexts:
        if reference in context:
            return True
        if _looks_like_windows_path(reference) and reference.lower() in context.lower():
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
                plan = await self._generate_structured(current_messages, run_id=run_id)
                validate_plan_against_request(plan, request)
                return plan
            except (ValidationError, PlanRequestValidationError, TypeError, ValueError) as exc:
                last_validation_error = exc
                if attempt == 0:
                    current_messages = build_planning_messages(request, include_repair_hint=True)
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

    async def _generate_structured(
        self,
        messages: list[ChatMessage],
        *,
        run_id: str | None,
    ) -> ConversationInteractionPlan:
        result = await asyncio.to_thread(
            self._llm_adapter.generate_structured,
            messages,
            ConversationInteractionPlan,
            temperature=0,
            max_tokens=_STRUCTURED_MAX_TOKENS,
            run_id=run_id,
        )
        parsed = result.parsed
        if not isinstance(parsed, ConversationInteractionPlan):
            raise TypeError("structured output is not ConversationInteractionPlan")
        return parsed

    def _adapter_identity(self) -> tuple[str, str]:
        provider = getattr(self._llm_adapter, "provider", "unknown")
        if hasattr(provider, "value"):
            provider = provider.value  # type: ignore[union-attr]
        model = getattr(self._llm_adapter, "model", "unknown")
        return str(provider), str(model)
