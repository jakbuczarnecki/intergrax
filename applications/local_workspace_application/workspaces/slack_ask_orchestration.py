from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from local_workspace_application.workspaces.hybrid_ask_policy import (
    ExecutableLiveCallV1,
    LiveCallProposalV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    validate_slack_timestamp,
)
from intergrax.runtime.vendor_knowledge.live.contracts import (
    LiveCapabilityExecutionResultV1,
    LiveExecutionOutcomeV1,
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

    binding_id: str
    source_call_id: str = ""
    message_ts: str
    text: str
    reply_count: int = Field(default=0, ge=0)
    retrieved_at: datetime
    content_hash: str
    safe_locator: str | None = None
    explicit_reference: bool = False

    @field_validator("binding_id", "content_hash")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("root_candidate_field_blank")
        return value

    @field_validator("message_ts")
    @classmethod
    def _message_timestamp_valid(cls, value: str) -> str:
        return validate_slack_timestamp(value)

    @field_validator("retrieved_at")
    @classmethod
    def _retrieved_at_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("root_candidate_timestamp_must_be_timezone_aware")
        return value

    @field_validator("text")
    @classmethod
    def _text_bounded(cls, value: str) -> str:
        return value[:16_384]

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
                -int(candidate.explicit_reference),
                -lexical,
                -candidate.reply_count,
                -Decimal(candidate.message_ts),
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


_PARTIAL_REASON_ORDER = (
    "message_limit",
    "thread_limit",
    "channel_limit",
    "provider_rate_limit",
    "provider_truncation",
    "deadline",
    "call_budget",
    "provider_failure",
    "ambiguous_binding",
)


class SlackAskStagedExecutionV1:
    """Discover Slack roots after list execution and expand bounded threads."""

    def __init__(
        self,
        *,
        planner: SlackAskPlannerV1,
        request: SlackAskRequestV1,
        initial_coverage: SlackAskCoverageV1,
        resolved_bindings: tuple[WorkspaceLiveAccessBinding, ...],
        proposal_validator: Callable[
            [tuple[LiveCallProposalV1, ...]],
            tuple[ExecutableLiveCallV1, ...],
        ],
    ) -> None:
        self._planner = planner
        self._request = request
        self._coverage = initial_coverage
        self._bindings_by_id = {
            binding.live_access_binding_id: binding for binding in resolved_bindings
        }
        self._proposal_validator = proposal_validator
        self._root_evidence_keys: set[tuple[str, str]] = set()
        self._thread_call_ids: set[str] = set()
        self._all_root_candidates: list[SlackAskRootCandidateV1] = []
        self._stage_one_call_ids: set[str] = set()

    @property
    def coverage(self) -> SlackAskCoverageV1:
        return self._coverage

    def expand(
        self,
        *,
        stage: int,
        calls: tuple[ExecutableLiveCallV1, ...],
        outcomes: tuple[LiveCapabilityExecutionResultV1, ...],
        attempted_calls: tuple[ExecutableLiveCallV1, ...],
        remaining_provider_call_budget: int,
        deadline_reached: bool,
    ) -> tuple[ExecutableLiveCallV1, ...]:
        self._observe_common(
            calls=calls,
            outcomes=outcomes,
            attempted_calls=attempted_calls,
            deadline_reached=deadline_reached,
        )
        if stage == 2:
            self._observe_threads(calls=calls, outcomes=outcomes)
            return ()
        if (
            stage == 1
            and self._request.intent is SlackAskIntentV1.THREAD_SUMMARY
        ):
            self._observe_threads(calls=calls, outcomes=outcomes)
            return ()
        if stage != 1 or self._request.intent not in {
            SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY,
            SlackAskIntentV1.RECENT_MULTI_CHANNEL_ACTIVITY,
            SlackAskIntentV1.BOUNDED_RECENT_SEARCH,
        }:
            return ()

        self._stage_one_call_ids = {call.call_id for call in calls}
        self._all_root_candidates = self._extract_roots(
            calls=calls,
            outcomes=outcomes,
        )
        self._coverage = self._coverage.model_copy(
            update={
                "root_messages_inspected": len(self._all_root_candidates),
            }
        )
        for candidate in self._all_root_candidates:
            self._root_evidence_keys.add(
                (candidate.source_call_id, candidate.message_ts)
            )

        if self._request.intent is SlackAskIntentV1.BOUNDED_RECENT_SEARCH:
            selected_roots = self._planner.filter_and_rank_recent_evidence(
                query=self._request.question,
                candidates=tuple(self._all_root_candidates),
                remaining_provider_call_budget=max(len(self._all_root_candidates), 1),
                maximum=len(self._all_root_candidates),
            )
            matched_keys = {
                (item.source_call_id, item.message_ts)
                for item in selected_roots
            }
            self._root_evidence_keys.intersection_update(matched_keys)
        else:
            selected_roots = tuple(self._all_root_candidates)

        eligible = tuple(item for item in selected_roots if item.reply_count > 0)
        expansion_candidates = self._planner.rank_thread_candidates(
            query=self._request.question,
            candidates=eligible,
            remaining_provider_call_budget=remaining_provider_call_budget,
        )
        if len(eligible) > MAX_SLACK_ASK_THREAD_EXPANSIONS:
            self._add_reason("thread_limit")
        if (
            remaining_provider_call_budget < MAX_SLACK_ASK_THREAD_EXPANSIONS
            and len(eligible) > remaining_provider_call_budget
        ):
            self._add_reason("call_budget")
        if deadline_reached or remaining_provider_call_budget < 1:
            if eligible:
                self._add_reason("deadline" if deadline_reached else "call_budget")
            return ()

        proposals = tuple(
            self._planner._thread_proposal(
                self._binding_for_candidate(candidate),
                root_ts=candidate.message_ts,
                index=index,
            )
            for index, candidate in enumerate(expansion_candidates)
        )
        validated = self._proposal_validator(proposals)
        self._thread_call_ids.update(call.call_id for call in validated)
        return validated

    def include_evidence(self, evidence: tuple[Any, ...]) -> tuple[Any, ...]:
        """Keep matched roots plus replies from actually selected threads."""
        if self._request.intent is not SlackAskIntentV1.BOUNDED_RECENT_SEARCH:
            return evidence
        return tuple(
            item
            for item in evidence
            if not (
                getattr(item, "call_id", None) in self._stage_one_call_ids
                and (
                    getattr(item, "call_id", None),
                    getattr(item, "remote_item_id", None),
                )
                not in self._root_evidence_keys
            )
        )

    def _observe_common(
        self,
        *,
        calls: tuple[ExecutableLiveCallV1, ...],
        outcomes: tuple[LiveCapabilityExecutionResultV1, ...],
        attempted_calls: tuple[ExecutableLiveCallV1, ...],
        deadline_reached: bool,
    ) -> None:
        attempted_ids = {call.call_id for call in attempted_calls}
        outcome_by_id = {outcome.call_id: outcome for outcome in outcomes}
        queried = {
            call.live_access_binding_id
            for call in attempted_calls
            if call.call_id in outcome_by_id
        }
        skipped = {
            call.live_access_binding_id
            for call in calls
            if call.call_id not in attempted_ids
        } - set(self._coverage.queried_bindings)
        self._coverage = self._coverage.model_copy(
            update={
                "queried_bindings": self._ordered_bindings(
                    self._coverage.queried_bindings + tuple(queried)
                ),
                "skipped_bindings": self._ordered_bindings(
                    self._coverage.skipped_bindings + tuple(skipped)
                ),
                "provider_call_count": self._coverage.provider_call_count
                + len(outcomes),
                "truncated": self._coverage.truncated or deadline_reached,
            }
        )
        for outcome in outcomes:
            if outcome.truncated:
                self._add_reason("provider_truncation")
            if outcome.normalized_outcome is LiveExecutionOutcomeV1.FAILED:
                self._add_reason(
                    "provider_rate_limit"
                    if outcome.error_code == "live_provider_throttled"
                    else (
                        "deadline"
                        if outcome.error_code == "live_execution_timeout"
                        else "provider_failure"
                    )
                )
        if skipped:
            self._add_reason("deadline" if deadline_reached else "call_budget")

    def _extract_roots(
        self,
        *,
        calls: tuple[ExecutableLiveCallV1, ...],
        outcomes: tuple[LiveCapabilityExecutionResultV1, ...],
    ) -> list[SlackAskRootCandidateV1]:
        calls_by_id = {call.call_id: call for call in calls}
        candidates: list[SlackAskRootCandidateV1] = []
        for outcome in outcomes:
            call = calls_by_id.get(outcome.call_id)
            if (
                call is None
                or call.capability_id != SLACK_CONVERSATION_LIST_CAPABILITY_ID
                or outcome.normalized_outcome is LiveExecutionOutcomeV1.FAILED
                or outcome.provider_id != SLACK_PROVIDER_ID
                or outcome.source_kind != SLACK_SOURCE_KIND
            ):
                continue
            for item in outcome.items:
                candidate = self._parse_root_item(call, item)
                if candidate is not None:
                    candidates.append(candidate)
            if len(outcome.items) >= MAX_SLACK_ASK_ROOTS_PER_CHANNEL:
                self._add_reason("message_limit")
        return self._deduplicate_candidates(candidates)

    def _observe_threads(
        self,
        *,
        calls: tuple[ExecutableLiveCallV1, ...],
        outcomes: tuple[LiveCapabilityExecutionResultV1, ...],
    ) -> None:
        calls_by_id = {call.call_id: call for call in calls}
        expanded = len(
            {
                outcome.call_id
                for outcome in outcomes
                if outcome.call_id in calls_by_id
                and calls_by_id[outcome.call_id].capability_id
                == SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID
            }
        )
        replies = 0
        for outcome in outcomes:
            call = calls_by_id.get(outcome.call_id)
            if (
                call is None
                or call.capability_id != SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID
                or outcome.normalized_outcome is LiveExecutionOutcomeV1.FAILED
            ):
                continue
            for item in outcome.items:
                if self._parse_reply_item(call, item):
                    replies += 1
            if len(outcome.items) >= MAX_SLACK_ASK_REPLIES_PER_THREAD:
                self._add_reason("thread_limit")
        self._coverage = self._coverage.model_copy(
            update={
                "threads_expanded": self._coverage.threads_expanded + expanded,
                "replies_inspected": self._coverage.replies_inspected + replies,
            }
        )

    def _parse_root_item(
        self,
        call: ExecutableLiveCallV1,
        item: Any,
    ) -> SlackAskRootCandidateV1 | None:
        payload = self._normalized_payload(item)
        if payload is None:
            self._add_reason("provider_failure")
            return None
        message_ts = payload.get("message_ts")
        text = payload.get("text")
        reply_count = payload.get("reply_count")
        if (
            not isinstance(message_ts, str)
            or message_ts != item.remote_item_id
            or payload.get("thread_root_ts") is not None
            or not isinstance(text, str)
            or (reply_count is not None and (not isinstance(reply_count, int) or isinstance(reply_count, bool)))
        ):
            self._add_reason("provider_failure")
            return None
        try:
            explicit = message_ts in self._request.thread_root_timestamps
            return SlackAskRootCandidateV1(
                binding_id=call.live_access_binding_id,
                source_call_id=call.call_id,
                message_ts=message_ts,
                text=text,
                reply_count=reply_count or 0,
                retrieved_at=item.retrieved_at,
                content_hash=item.content_hash,
                safe_locator=item.safe_locator,
                explicit_reference=explicit,
            )
        except (TypeError, ValueError):
            self._add_reason("provider_failure")
            return None

    def _parse_reply_item(
        self,
        call: ExecutableLiveCallV1,
        item: Any,
    ) -> bool:
        payload = self._normalized_payload(item)
        root_ts = getattr(call.validated_request, "root_message_ts", None)
        if (
            payload is None
            or not isinstance(root_ts, str)
            or payload.get("thread_root_ts") != root_ts
            or item.remote_item_id == root_ts
        ):
            self._add_reason("provider_failure")
            return False
        try:
            validate_slack_timestamp(item.remote_item_id)
            return True
        except (TypeError, ValueError):
            self._add_reason("provider_failure")
            return False

    @staticmethod
    def _normalized_payload(item: Any) -> dict[str, Any] | None:
        try:
            payload = json.loads(item.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("item_type") != "slack_conversation_message"
            or payload.get("content_available") is not True
            or payload.get("content_mode") != "structured_record"
        ):
            return None
        return payload

    def _binding_for_candidate(
        self,
        candidate: SlackAskRootCandidateV1,
    ) -> WorkspaceLiveAccessBinding:
        binding = self._bindings_by_id.get(candidate.binding_id)
        if binding is not None:
            return binding
        raise SlackAskPlanningError("ambiguous_binding")

    def _deduplicate_candidates(
        self,
        candidates: list[SlackAskRootCandidateV1],
    ) -> list[SlackAskRootCandidateV1]:
        seen: set[tuple[str, str]] = set()
        result: list[SlackAskRootCandidateV1] = []
        for candidate in candidates:
            key = (candidate.binding_id, candidate.message_ts)
            if key not in seen:
                seen.add(key)
                result.append(candidate)
        return result

    def _ordered_bindings(self, values: tuple[str, ...]) -> tuple[str, ...]:
        known = set(values)
        return tuple(
            binding_id
            for binding_id in self._coverage.resolved_bindings
            if binding_id in known
        )

    def _add_reason(self, reason: str) -> None:
        reasons = set(self._coverage.partial_result_reasons)
        reasons.add(reason)
        self._coverage = self._coverage.model_copy(
            update={
                "partial_result_reasons": tuple(
                    item for item in _PARTIAL_REASON_ORDER if item in reasons
                ),
                "truncated": self._coverage.truncated
                or reason
                in {
                    "message_limit",
                    "thread_limit",
                    "provider_truncation",
                    "deadline",
                    "call_budget",
                },
            }
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
    "SlackAskStagedExecutionV1",
]
