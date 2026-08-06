from __future__ import annotations

import hashlib
from decimal import Decimal
from enum import StrEnum
from typing import Any

from local_workspace_application.workspaces.hybrid_ask_policy import LiveCallProposalV1
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    validate_slack_timestamp,
)
from intergrax.runtime.vendor_knowledge.live.slack import (
    SLACK_CONVERSATION_LIST_CAPABILITY_ID,
    SLACK_CONVERSATION_READ_CAPABILITY_ID,
    SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
)

SLACK_PROVIDER_ID = "slack"
SLACK_SOURCE_KIND = "slack_conversation"
MAX_SLACK_ASK_CHANNELS = 5
MAX_SLACK_ASK_ROOTS_PER_CHANNEL = 15
MAX_SLACK_ASK_THREAD_EXPANSIONS = 3
MAX_SLACK_ASK_REPLIES_PER_THREAD = 15


class SlackAskIntentV1(StrEnum):
    RECENT_CHANNEL_ACTIVITY = "recent_channel_activity"
    RECENT_MULTI_CHANNEL_ACTIVITY = "recent_multi_channel_activity"
    BOUNDED_RECENT_SEARCH = "bounded_recent_search"
    THREAD_SUMMARY = "thread_summary"
    EXACT_MESSAGE = "exact_message"


class SlackAskPlanningError(RuntimeError):
    def __init__(self, error_code: str, *, matches: tuple[str, ...] = ()) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.matches = matches


class SlackAskRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    intent: SlackAskIntentV1
    question: str = Field(..., min_length=1, max_length=4096)
    binding_references: tuple[str, ...] = ()
    oldest: str | None = None
    latest: str | None = None
    thread_root_timestamps: tuple[str, ...] = ()
    message_ts: str | None = None
    root_thread_ts: str | None = None

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question_must_not_be_blank")
        return value

    @field_validator("binding_references")
    @classmethod
    def _references_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("binding_reference_blank")
        return value

    @field_validator("oldest", "latest", "message_ts", "root_thread_ts")
    @classmethod
    def _timestamps_valid(cls, value: str | None) -> str | None:
        return None if value is None else validate_slack_timestamp(value)

    @field_validator("thread_root_timestamps")
    @classmethod
    def _thread_timestamps_valid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_slack_timestamp(item) for item in value)

    @model_validator(mode="after")
    def _intent_shape(self) -> SlackAskRequestV1:
        if (
            self.intent is SlackAskIntentV1.THREAD_SUMMARY
            and self.root_thread_ts is None
        ):
            raise ValueError("thread_root_ts_required")
        if self.intent is SlackAskIntentV1.EXACT_MESSAGE and self.message_ts is None:
            raise ValueError("message_ts_required")
        if (
            self.intent is SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY
            and len(self.binding_references) != 1
        ):
            raise ValueError("one_binding_required")
        if len(self.thread_root_timestamps) > MAX_SLACK_ASK_THREAD_EXPANSIONS:
            raise ValueError("thread_limit")
        return self


class SlackAskCoverageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_bindings: tuple[str, ...] = ()
    resolved_bindings: tuple[str, ...] = ()
    queried_bindings: tuple[str, ...] = ()
    skipped_bindings: tuple[str, ...] = ()
    effective_oldest: str | None = None
    effective_latest: str | None = None
    root_messages_inspected: int = Field(default=0, ge=0)
    threads_expanded: int = Field(default=0, ge=0)
    replies_inspected: int = Field(default=0, ge=0)
    truncated: bool = False
    provider_call_count: int = Field(default=0, ge=0)
    partial_result_reasons: tuple[str, ...] = ()


class SlackAskRootCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    message_ts: str
    text: str
    reply_count: int = Field(default=0, ge=0)
    explicit_reference: bool = False

    @field_validator("message_ts")
    @classmethod
    def _message_timestamp_valid(cls, value: str) -> str:
        return validate_slack_timestamp(value)


class SlackAskPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: SlackAskIntentV1
    search_semantics: str
    ordered_live_call_proposals: tuple[LiveCallProposalV1, ...]
    coverage: SlackAskCoverageV1
    maximum_channels: int = MAX_SLACK_ASK_CHANNELS
    maximum_roots_per_channel: int = MAX_SLACK_ASK_ROOTS_PER_CHANNEL
    maximum_thread_expansions: int = MAX_SLACK_ASK_THREAD_EXPANSIONS
    maximum_replies_per_thread: int = MAX_SLACK_ASK_REPLIES_PER_THREAD


def _normalize_channel_name(value: str) -> str:
    normalized = value.strip().casefold()
    return normalized.removeprefix("#")


def _is_slack_conversation_binding(binding: WorkspaceLiveAccessBinding) -> bool:
    return binding.derived_provider_id == SLACK_PROVIDER_ID and any(
        capability.startswith("vendor.slack.slack_conversation.")
        for capability in binding.allowed_capability_ids
    )


class SlackAskBindingResolverV1:
    """Resolve only committed, active Slack conversation bindings."""

    def resolve_bindings(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        references: tuple[str, ...],
    ) -> tuple[WorkspaceLiveAccessBinding, ...]:
        if not references:
            raise SlackAskPlanningError("slack_binding_required")
        candidates = tuple(
            binding
            for binding in configuration.live_access_bindings
            if (
                binding.tenant_id == configuration.tenant_id
                and binding.workspace_id == configuration.workspace_id
                and binding.status is LiveAccessBindingStatusV1.ACTIVE
                and _is_slack_conversation_binding(binding)
            )
        )
        resolved: list[WorkspaceLiveAccessBinding] = []
        for reference in references:
            resolved.append(self._resolve_one(candidates, reference))
        if len({binding.live_access_binding_id for binding in resolved}) != len(
            resolved
        ):
            raise SlackAskPlanningError("ambiguous_binding")
        return tuple(resolved)

    def _resolve_one(
        self,
        candidates: tuple[WorkspaceLiveAccessBinding, ...],
        reference: str,
    ) -> WorkspaceLiveAccessBinding:
        exact_ids = tuple(
            binding
            for binding in candidates
            if binding.live_access_binding_id == reference
        )
        if exact_ids:
            return exact_ids[0]
        exact_names = tuple(
            binding
            for binding in candidates
            if binding.derived_safe_display_label == reference
        )
        if len(exact_names) == 1:
            return exact_names[0]
        if len(exact_names) > 1:
            raise SlackAskPlanningError(
                "ambiguous_binding",
                matches=tuple(
                    sorted(item.live_access_binding_id for item in exact_names)
                ),
            )
        normalized = _normalize_channel_name(reference)
        normalized_matches = tuple(
            binding
            for binding in candidates
            if _normalize_channel_name(binding.derived_safe_display_label) == normalized
        )
        if len(normalized_matches) == 1:
            return normalized_matches[0]
        if len(normalized_matches) > 1:
            raise SlackAskPlanningError(
                "ambiguous_binding",
                matches=tuple(
                    sorted(item.live_access_binding_id for item in normalized_matches)
                ),
            )
        raise SlackAskPlanningError("binding_not_found")


class SlackAskPlannerV1:
    """Build bounded Slack proposals; provider inventory is never consulted."""

    def __init__(self, resolver: SlackAskBindingResolverV1 | None = None) -> None:
        self._resolver = resolver or SlackAskBindingResolverV1()

    def build_plan(
        self,
        *,
        configuration: WorkspaceKnowledgeConfigurationV1,
        request: SlackAskRequestV1,
    ) -> SlackAskPlanV1:
        bindings = self._resolver.resolve_bindings(
            configuration=configuration,
            references=request.binding_references,
        )
        if (
            request.intent is SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY
            and len(bindings) != 1
        ):
            raise SlackAskPlanningError("one_binding_required")
        if (
            request.intent is not SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY
            and len(bindings) > MAX_SLACK_ASK_CHANNELS
        ):
            raise SlackAskPlanningError("channel_limit")
        proposals: list[LiveCallProposalV1] = []
        if request.intent in {
            SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY,
            SlackAskIntentV1.RECENT_MULTI_CHANNEL_ACTIVITY,
            SlackAskIntentV1.BOUNDED_RECENT_SEARCH,
        }:
            for index, binding in enumerate(bindings):
                proposals.append(
                    self._list_proposal(
                        binding,
                        request=request,
                        index=index,
                    )
                )
            for index, root_ts in enumerate(request.thread_root_timestamps):
                proposals.append(
                    self._thread_proposal(
                        bindings[index % len(bindings)],
                        root_ts=root_ts,
                        index=index,
                    )
                )
        elif request.intent is SlackAskIntentV1.THREAD_SUMMARY:
            proposals.append(
                self._thread_proposal(
                    bindings[0],
                    root_ts=request.root_thread_ts,
                    index=0,
                )
            )
        else:
            proposals.append(
                self._exact_proposal(
                    bindings[0],
                    request=request,
                )
            )
        coverage = SlackAskCoverageV1(
            requested_bindings=request.binding_references,
            resolved_bindings=tuple(
                binding.live_access_binding_id for binding in bindings
            ),
            queried_bindings=tuple(
                binding.live_access_binding_id
                for binding in bindings
                if request.intent
                in {
                    SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY,
                    SlackAskIntentV1.RECENT_MULTI_CHANNEL_ACTIVITY,
                    SlackAskIntentV1.BOUNDED_RECENT_SEARCH,
                }
            ),
            effective_oldest=request.oldest,
            effective_latest=request.latest,
        )
        return SlackAskPlanV1(
            intent=request.intent,
            search_semantics=(
                "bounded_recent_search"
                if request.intent is SlackAskIntentV1.BOUNDED_RECENT_SEARCH
                else request.intent.value
            ),
            ordered_live_call_proposals=tuple(proposals),
            coverage=coverage,
        )

    @staticmethod
    def _call_id(kind: str, binding: WorkspaceLiveAccessBinding, index: int = 0) -> str:
        suffix = hashlib.sha256(
            binding.live_access_binding_id.encode("utf-8")
        ).hexdigest()[:12]
        return f"slack-{kind}-{index}-{suffix}"

    def _list_proposal(
        self,
        binding: WorkspaceLiveAccessBinding,
        *,
        request: SlackAskRequestV1,
        index: int,
    ) -> LiveCallProposalV1:
        typed_request: dict[str, Any] = {"page_size": MAX_SLACK_ASK_ROOTS_PER_CHANNEL}
        if request.oldest is not None:
            typed_request["oldest"] = request.oldest
        if request.latest is not None:
            typed_request["latest"] = request.latest
        return LiveCallProposalV1(
            call_id=self._call_id("list", binding, index),
            live_access_binding_id=binding.live_access_binding_id,
            capability_id=SLACK_CONVERSATION_LIST_CAPABILITY_ID,
            typed_capability_request=typed_request,
        )

    def _thread_proposal(
        self,
        binding: WorkspaceLiveAccessBinding,
        *,
        root_ts: str | None,
        index: int,
    ) -> LiveCallProposalV1:
        if root_ts is None:
            raise SlackAskPlanningError("thread_root_ts_required")
        return LiveCallProposalV1(
            call_id=self._call_id("thread", binding, index),
            live_access_binding_id=binding.live_access_binding_id,
            capability_id=SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
            typed_capability_request={
                "root_message_ts": root_ts,
                "page_size": MAX_SLACK_ASK_REPLIES_PER_THREAD,
            },
        )

    def _exact_proposal(
        self,
        binding: WorkspaceLiveAccessBinding,
        *,
        request: SlackAskRequestV1,
    ) -> LiveCallProposalV1:
        if request.message_ts is None:
            raise SlackAskPlanningError("message_ts_required")
        typed_request: dict[str, Any] = {"message_ts": request.message_ts}
        if request.root_thread_ts is not None:
            typed_request["root_thread_ts"] = request.root_thread_ts
        return LiveCallProposalV1(
            call_id=self._call_id("read", binding),
            live_access_binding_id=binding.live_access_binding_id,
            capability_id=SLACK_CONVERSATION_READ_CAPABILITY_ID,
            typed_capability_request=typed_request,
        )

    @staticmethod
    def rank_thread_candidates(
        *,
        query: str,
        candidates: tuple[SlackAskRootCandidateV1, ...],
        remaining_provider_call_budget: int,
        maximum: int = MAX_SLACK_ASK_THREAD_EXPANSIONS,
    ) -> tuple[SlackAskRootCandidateV1, ...]:
        if remaining_provider_call_budget < 1 or maximum < 1:
            return ()
        query_terms = {term.casefold() for term in query.split() if term.strip()}

        def score(candidate: SlackAskRootCandidateV1) -> tuple[int, int, Decimal, int]:
            lexical = sum(term in candidate.text.casefold() for term in query_terms)
            return (
                -lexical,
                -candidate.reply_count,
                -Decimal(candidate.message_ts),
                -int(candidate.explicit_reference),
            )

        return tuple(
            sorted(candidates, key=score)[
                : min(maximum, remaining_provider_call_budget)
            ]
        )

    @staticmethod
    def filter_and_rank_recent_evidence(
        *,
        query: str,
        candidates: tuple[SlackAskRootCandidateV1, ...],
        remaining_provider_call_budget: int,
        maximum: int = MAX_SLACK_ASK_THREAD_EXPANSIONS,
    ) -> tuple[SlackAskRootCandidateV1, ...]:
        terms = tuple(term.casefold() for term in query.split() if term.strip())
        filtered = tuple(
            candidate
            for candidate in candidates
            if not terms or any(term in candidate.text.casefold() for term in terms)
        )
        return SlackAskPlannerV1.rank_thread_candidates(
            query=query,
            candidates=filtered,
            remaining_provider_call_budget=remaining_provider_call_budget,
            maximum=maximum,
        )


__all__ = [
    "MAX_SLACK_ASK_CHANNELS",
    "MAX_SLACK_ASK_REPLIES_PER_THREAD",
    "MAX_SLACK_ASK_ROOTS_PER_CHANNEL",
    "MAX_SLACK_ASK_THREAD_EXPANSIONS",
    "SlackAskBindingResolverV1",
    "SlackAskCoverageV1",
    "SlackAskIntentV1",
    "SlackAskPlanV1",
    "SlackAskPlannerV1",
    "SlackAskPlanningError",
    "SlackAskRequestV1",
    "SlackAskRootCandidateV1",
]
