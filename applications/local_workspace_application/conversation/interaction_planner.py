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
_TRAILING_SENTENCE_PUNCTUATION = ".,;!?"
_URL_SCHEME_PATTERN = re.compile(r"https?://", re.IGNORECASE)
_PATH_START_BOUNDARY_CHARS = frozenset(" \t\n\r\"'(:[")
_PATH_END_BOUNDARY_CHARS = frozenset(" \t\n\r\"',;)")
_PATH_CONTINUATION_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\\/-_.:"
)


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
    allowed_urls = extract_user_url_candidates(user_contexts)

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
                if url not in allowed_urls:
                    raise PlanRequestValidationError("URL not found in user context")

        if isinstance(action, KnowledgeAddLocalReferencesPlannedAction):
            for reference in action.references:
                if not _local_reference_in_context(reference.value, user_contexts):
                    raise PlanRequestValidationError("local reference not found in user context")

        workspace = getattr(action, "workspace", None)
        if workspace is not None and workspace.kind == WorkspaceReferenceKind.name:
            if workspace.value and not _evidence_quote_in_context(workspace.value, user_contexts):
                if not any(
                    workspace.value.lower() in context.lower() for context in user_contexts
                ):
                    raise PlanRequestValidationError("workspace name reference not in user context")


def extract_user_url_candidates(contexts: Sequence[str]) -> frozenset[str]:
    """Build a deterministic set of exact user-provided URLs from planning context."""
    candidates: set[str] = set()
    for context in contexts:
        for match in _URL_SCHEME_PATTERN.finditer(context):
            end = match.end()
            while end < len(context) and context[end] not in " \t\n\r<>\"'":
                end += 1
            cleaned = _strip_trailing_url_punctuation(context[match.start() : end])
            if cleaned:
                candidates.add(cleaned)
    return frozenset(candidates)


def _strip_trailing_url_punctuation(url: str) -> str:
    result = url
    while result and result[-1] in _TRAILING_SENTENCE_PUNCTUATION:
        result = result[:-1]
    for close, open_char in ((")", "("), ("]", "["), ("}", "{")):
        while result.endswith(close) and result.count(open_char) < result.count(close):
            result = result[:-1]
    return result


def _evidence_quote_in_context(quote: str, contexts: Sequence[str]) -> bool:
    if not quote or not quote.strip():
        return False
    normalized_quote = quote.strip()
    for context in contexts:
        if normalized_quote in context:
            return True
    return False


def _looks_like_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:\\", value)) or value.startswith("\\\\")


def _local_reference_in_context(reference: str, contexts: Sequence[str]) -> bool:
    if not reference:
        return False
    for context in contexts:
        if _bounded_local_reference_match(reference, context):
            return True
    return False


def _bounded_local_reference_match(reference: str, context: str) -> bool:
    windows_insensitive = _looks_like_windows_path(reference)
    search_reference = reference.lower() if windows_insensitive else reference
    search_context = context.lower() if windows_insensitive else context
    start = 0
    while True:
        index = search_context.find(search_reference, start)
        if index == -1:
            return False
        end = index + len(reference)
        if _has_valid_path_start_boundary(context, index) and _has_valid_path_end_boundary(
            context, end, reference
        ):
            return True
        start = index + 1


def _has_valid_path_start_boundary(context: str, start: int) -> bool:
    if start == 0:
        return True
    return context[start - 1] in _PATH_START_BOUNDARY_CHARS


def _has_valid_path_end_boundary(context: str, end: int, reference: str) -> bool:
    if end >= len(context):
        return True
    next_char = context[end]
    if next_char in _PATH_END_BOUNDARY_CHARS:
        return True
    if next_char == ".":
        return _is_sentence_ending_period(context, end, reference)
    if next_char in _PATH_CONTINUATION_CHARS:
        return False
    return False


def _is_sentence_ending_period(context: str, period_index: int, reference: str) -> bool:
    if period_index + 1 < len(context) and context[period_index + 1] in _PATH_CONTINUATION_CHARS:
        return False
    if reference.endswith("."):
        return False
    return True


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
