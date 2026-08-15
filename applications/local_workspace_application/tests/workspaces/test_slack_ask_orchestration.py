from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace

import pytest
from local_workspace_application.workspaces.hybrid_ask_policy import (
    ExecutableLiveCallV1,
    LiveCallProposalV1,
    ResolvedLiveResourceScopeV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
)
from local_workspace_application.workspaces.slack_ask_orchestration import (
    SlackAskIntentV1,
    SlackAskPlannerV1,
    SlackAskPlanningError,
    SlackAskRequestV1,
    SlackAskRootCandidateV1,
    SlackAskStagedExecutionV1,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.contracts import (
    EffectiveLiveCallBudgetV1,
    LiveCapabilityExecutionResultV1,
    LiveCapabilityResultItemV1,
    LiveExecutionOutcomeV1,
)
from intergrax.runtime.vendor_knowledge.live.slack import (
    SLACK_CONVERSATION_LIST_CAPABILITY_ID,
    SLACK_CONVERSATION_READ_CAPABILITY_ID,
    SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
)
from intergrax.runtime.vendor_knowledge.live.slack.conversation import (
    SlackConversationListLiveRequestV1,
    SlackConversationThreadReadLiveRequestV1,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 6, 4, 0, tzinfo=UTC)
_TENANT = "tenant-1"
_WORKSPACE = "workspace-1"
_ROOT = "1704153600.000001"


def _binding(
    binding_id: str,
    label: str,
    *,
    status: LiveAccessBindingStatusV1 = LiveAccessBindingStatusV1.ACTIVE,
    tenant_id: str = _TENANT,
    workspace_id: str = _WORKSPACE,
) -> WorkspaceLiveAccessBinding:
    return WorkspaceLiveAccessBinding(
        live_access_binding_id=binding_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        connection_ref=f"connection-{binding_id}",
        remote_resource_id=f"scope-{binding_id}",
        allowed_capability_ids=(
            SLACK_CONVERSATION_LIST_CAPABILITY_ID,
            SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
            SLACK_CONVERSATION_READ_CAPABILITY_ID,
        ),
        derived_provider_id="slack",
        derived_integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        derived_resource_type="slack_conversation",
        derived_safe_display_label=label,
        status=status,
        mutation_id=f"mutation-{binding_id}",
        effective_revision=1,
        semantic_identity_hash="a" * 64,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _configuration(
    bindings: tuple[WorkspaceLiveAccessBinding, ...],
) -> WorkspaceKnowledgeConfigurationV1:
    return WorkspaceKnowledgeConfigurationV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=1,
        live_access_bindings=bindings,
        updated_at=_NOW,
    )


def _slack_call(
    *,
    call_id: str,
    binding_id: str,
    capability_id: str,
    request: object,
    max_live_calls: int = 4,
) -> ExecutableLiveCallV1:
    return ExecutableLiveCallV1(
        call_id=call_id,
        live_access_binding_id=binding_id,
        connection_ref=f"connection-{binding_id}",
        provider_id="slack",
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        capability_id=capability_id,
        contract_version="1",
        source_kind="slack_conversation",
        validated_request=request,
        remote_resource_id=f"scope-{binding_id}",
        audience_context_ref="personal",
        resolved_resource_scope=ResolvedLiveResourceScopeV1(
            remote_resource_id=f"scope-{binding_id}",
            scope_token=None,
        ),
        effective_budget=EffectiveLiveCallBudgetV1(
            max_live_calls=max_live_calls,
            max_total_duration_ms=30_000,
            max_result_items=15,
            max_result_bytes=131_072,
        ),
    )


def _slack_outcome(
    call: ExecutableLiveCallV1,
    *,
    items: tuple[LiveCapabilityResultItemV1, ...],
) -> LiveCapabilityExecutionResultV1:
    return LiveCapabilityExecutionResultV1(
        call_id=call.call_id,
        normalized_outcome=LiveExecutionOutcomeV1.COMPLETED,
        items=items,
        item_count=len(items),
        byte_count=sum(len(item.content.encode()) for item in items),
        started_at=_NOW,
        completed_at=_NOW,
        provider_id="slack",
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind="slack_conversation",
        capability_id=call.capability_id,
        contract_version="1",
        live_access_binding_id=call.live_access_binding_id,
        connection_ref=call.connection_ref,
        remote_resource_id=call.remote_resource_id,
    )


def _slack_item(remote_item_id: str, payload: dict[str, object]) -> LiveCapabilityResultItemV1:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return LiveCapabilityResultItemV1(
        remote_item_id=remote_item_id,
        safe_display_name="Slack message",
        content=content,
        content_hash=sha256(content.encode()).hexdigest(),
        retrieved_at=_NOW,
    )


@pytest.mark.parametrize(
    ("question", "intent", "reference"),
    [
        (
            "Co działo się dziś na #engineering?",
            SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY,
            "#engineering",
        ),
        (
            "Znajdź rozmowy o błędzie płatności.",
            SlackAskIntentV1.BOUNDED_RECENT_SEARCH,
            "payments",
        ),
        (
            "Jaką decyzję podjęto w tym wątku?",
            SlackAskIntentV1.THREAD_SUMMARY,
            "engineering",
        ),
        (
            "Czy ktoś zgłaszał dziś awarię?",
            SlackAskIntentV1.BOUNDED_RECENT_SEARCH,
            "engineering",
        ),
        (
            "Podsumuj najważniejsze rzeczy z moich kluczowych kanałów.",
            SlackAskIntentV1.RECENT_MULTI_CHANNEL_ACTIVITY,
            "engineering",
        ),
    ],
)
def test_polish_acceptance_fixtures_produce_bounded_plans(
    question: str,
    intent: SlackAskIntentV1,
    reference: str,
) -> None:
    request = SlackAskRequestV1(
        intent=intent,
        question=question,
        binding_references=(
            ("engineering", "payments")
            if intent is SlackAskIntentV1.RECENT_MULTI_CHANNEL_ACTIVITY
            else (reference,)
        ),
        root_thread_ts=_ROOT if intent is SlackAskIntentV1.THREAD_SUMMARY else None,
    )
    plan = SlackAskPlannerV1().build_plan(
        configuration=_configuration(
            (
                _binding("engineering", "Engineering"),
                _binding("payments", "#payments"),
            )
        ),
        request=request,
    )

    assert plan.search_semantics != "workspace_search"
    assert plan.search_semantics != "complete_search"
    assert plan.coverage.resolved_bindings
    assert all(
        proposal.live_access_binding_id in plan.coverage.resolved_bindings
        for proposal in plan.ordered_live_call_proposals
    )
    if intent is SlackAskIntentV1.BOUNDED_RECENT_SEARCH:
        assert plan.search_semantics == "bounded_recent_search"
    if intent is SlackAskIntentV1.THREAD_SUMMARY:
        assert plan.ordered_live_call_proposals[0].capability_id == (
            SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID
        )


def test_binding_resolution_is_active_tenant_scoped_and_ambiguous_names_fail() -> None:
    configuration = _configuration(
        (
            _binding("active", "#engineering"),
            _binding(
                "inactive", "#disabled", status=LiveAccessBindingStatusV1.DISABLED
            ),
        )
    )
    planner = SlackAskPlannerV1()

    resolved = planner.build_plan(
        configuration=configuration,
        request=SlackAskRequestV1(
            intent=SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY,
            question="recent",
            binding_references=("engineering",),
        ),
    )
    assert resolved.coverage.resolved_bindings == ("active",)

    with pytest.raises(SlackAskPlanningError, match="binding_not_found"):
        planner.build_plan(
            configuration=configuration,
            request=SlackAskRequestV1(
                intent=SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY,
                question="unknown",
                binding_references=("#missing",),
            ),
        )

    ambiguous = _configuration(
        (_binding("one", "Engineering"), _binding("two", "#engineering"))
    )
    with pytest.raises(SlackAskPlanningError, match="ambiguous_binding"):
        planner.build_plan(
            configuration=ambiguous,
            request=SlackAskRequestV1(
                intent=SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY,
                question="ambiguous",
                binding_references=("engineering",),
            ),
        )


def test_hard_limits_and_exact_paths_are_explicit() -> None:
    configuration = _configuration(
        tuple(_binding(f"channel-{index}", f"channel-{index}") for index in range(6))
    )
    planner = SlackAskPlannerV1()
    with pytest.raises(SlackAskPlanningError, match="channel_limit"):
        planner.build_plan(
            configuration=configuration,
            request=SlackAskRequestV1(
                intent=SlackAskIntentV1.RECENT_MULTI_CHANNEL_ACTIVITY,
                question="many",
                binding_references=tuple(f"channel-{index}" for index in range(6)),
            ),
        )

    exact = planner.build_plan(
        configuration=_configuration((_binding("engineering", "Engineering"),)),
        request=SlackAskRequestV1(
            intent=SlackAskIntentV1.EXACT_MESSAGE,
            question="exact",
            binding_references=("engineering",),
            message_ts=_ROOT,
        ),
    )
    assert len(exact.ordered_live_call_proposals) == 1
    assert exact.ordered_live_call_proposals[0].capability_id == (
        SLACK_CONVERSATION_READ_CAPABILITY_ID
    )
    assert exact.coverage.provider_call_count == 0


def test_recent_search_plans_list_stage_before_thread_expansion() -> None:
    plan = SlackAskPlannerV1().build_plan(
        configuration=_configuration((_binding("engineering", "Engineering"),)),
        request=SlackAskRequestV1(
            intent=SlackAskIntentV1.BOUNDED_RECENT_SEARCH,
            question="payment outage",
            binding_references=("engineering",),
        ),
    )
    assert len(plan.ordered_live_call_proposals) == 1
    assert plan.ordered_live_call_proposals[0].capability_id == (
        SLACK_CONVERSATION_LIST_CAPABILITY_ID
    )
    assert plan.maximum_thread_expansions == 3
    assert plan.maximum_replies_per_thread == 15


def test_thread_selection_is_deterministic_and_budget_bounded() -> None:
    candidates = (
        SlackAskRootCandidateV1(
            binding_id="engineering",
            message_ts="1704153601.000001",
            text="payment incident",
            reply_count=1,
            retrieved_at=_NOW,
            content_hash="a" * 64,
        ),
        SlackAskRootCandidateV1(
            binding_id="engineering",
            message_ts="1704153602.000001",
            text="payment incident decision",
            reply_count=4,
            retrieved_at=_NOW,
            content_hash="a" * 64,
            explicit_reference=True,
        ),
        SlackAskRootCandidateV1(
            binding_id="engineering",
            message_ts="1704153603.000001",
            text="unrelated",
            reply_count=99,
            retrieved_at=_NOW,
            content_hash="a" * 64,
        ),
    )
    selected = SlackAskPlannerV1.rank_thread_candidates(
        query="payment",
        candidates=candidates,
        remaining_provider_call_budget=2,
    )
    assert tuple(item.message_ts for item in selected) == (
        "1704153602.000001",
        "1704153601.000001",
    )


def test_staged_execution_discovers_root_and_expands_its_origin_binding() -> None:
    bindings = (_binding("engineering", "Engineering"),)
    configuration = _configuration(bindings)
    request = SlackAskRequestV1(
        intent=SlackAskIntentV1.RECENT_CHANNEL_ACTIVITY,
        question="payment incident",
        binding_references=("engineering",),
    )
    plan = SlackAskPlannerV1().build_plan(
        configuration=configuration,
        request=request,
    )
    proposals: list[LiveCallProposalV1] = []

    def validate(
        values: tuple[LiveCallProposalV1, ...],
    ) -> tuple[ExecutableLiveCallV1, ...]:
        proposals.extend(values)
        return tuple(
            _slack_call(
                call_id=value.call_id,
                binding_id=value.live_access_binding_id,
                capability_id=value.capability_id,
                request=SlackConversationThreadReadLiveRequestV1(
                    **value.typed_capability_request
                ),
            )
            for value in values
        )

    hook = SlackAskStagedExecutionV1(
        planner=SlackAskPlannerV1(),
        request=request,
        initial_coverage=plan.coverage,
        resolved_bindings=bindings,
        proposal_validator=validate,
    )
    list_call = _slack_call(
        call_id="list-engineering",
        binding_id="engineering",
        capability_id=SLACK_CONVERSATION_LIST_CAPABILITY_ID,
        request=SlackConversationListLiveRequestV1(page_size=15),
    )
    root = _slack_item(
        _ROOT,
        {
            "change_kind": "upsert",
            "content_available": True,
            "content_mode": "structured_record",
            "item_type": "slack_conversation_message",
            "message_ts": _ROOT,
            "reply_count": 1,
            "text": "payment incident",
            "thread_root_ts": None,
        },
    )
    thread_calls = hook.expand(
        stage=1,
        calls=(list_call,),
        outcomes=(_slack_outcome(list_call, items=(root,)),),
        attempted_calls=(list_call,),
        remaining_provider_call_budget=3,
        deadline_reached=False,
    )

    assert len(proposals) == 1
    assert len(thread_calls) == 1
    assert thread_calls[0].live_access_binding_id == "engineering"
    assert thread_calls[0].capability_id == (
        SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID
    )
    reply_call = thread_calls[0]
    reply = _slack_item(
        "1704153601.000001",
        {
            "change_kind": "upsert",
            "content_available": True,
            "content_mode": "structured_record",
            "item_type": "slack_conversation_message",
            "thread_root_ts": _ROOT,
        },
    )
    hook.expand(
        stage=2,
        calls=(reply_call,),
        outcomes=(_slack_outcome(reply_call, items=(reply,)),),
        attempted_calls=(reply_call,),
        remaining_provider_call_budget=2,
        deadline_reached=False,
    )
    assert hook.coverage.root_messages_inspected == 1
    assert hook.coverage.threads_expanded == 1
    assert hook.coverage.replies_inspected == 1
    assert hook.coverage.provider_call_count == 2


def test_staged_multi_channel_expansion_is_global_and_keeps_binding_ownership() -> None:
    bindings = (
        _binding("engineering", "Engineering"),
        _binding("payments", "Payments"),
    )
    configuration = _configuration(bindings)
    request = SlackAskRequestV1(
        intent=SlackAskIntentV1.RECENT_MULTI_CHANNEL_ACTIVITY,
        question="incident",
        binding_references=("engineering", "payments"),
    )
    plan = SlackAskPlannerV1().build_plan(
        configuration=configuration,
        request=request,
    )

    def validate(
        values: tuple[LiveCallProposalV1, ...],
    ) -> tuple[ExecutableLiveCallV1, ...]:
        return tuple(
            _slack_call(
                call_id=value.call_id,
                binding_id=value.live_access_binding_id,
                capability_id=value.capability_id,
                request=SlackConversationThreadReadLiveRequestV1(
                    **value.typed_capability_request
                ),
            )
            for value in values
        )

    hook = SlackAskStagedExecutionV1(
        planner=SlackAskPlannerV1(),
        request=request,
        initial_coverage=plan.coverage,
        resolved_bindings=bindings,
        proposal_validator=validate,
    )
    list_calls = tuple(
        _slack_call(
            call_id=f"list-{binding.live_access_binding_id}",
            binding_id=binding.live_access_binding_id,
            capability_id=SLACK_CONVERSATION_LIST_CAPABILITY_ID,
            request=SlackConversationListLiveRequestV1(page_size=15),
            max_live_calls=5,
        )
        for binding in bindings
    )
    outcomes = tuple(
        _slack_outcome(
            call,
            items=tuple(
                _slack_item(
                    f"170415360{index}.000001",
                    {
                        "change_kind": "upsert",
                        "content_available": True,
                        "content_mode": "structured_record",
                        "item_type": "slack_conversation_message",
                        "message_ts": f"170415360{index}.000001",
                        "reply_count": 1,
                        "text": "incident",
                        "thread_root_ts": None,
                    },
                )
                for index in (
                    1 + 2 * list_calls.index(call),
                    2 + 2 * list_calls.index(call),
                )
            ),
        )
        for call in list_calls
    )
    thread_calls = hook.expand(
        stage=1,
        calls=list_calls,
        outcomes=outcomes,
        attempted_calls=list_calls,
        remaining_provider_call_budget=3,
        deadline_reached=False,
    )

    assert len(thread_calls) == 3
    assert {call.live_access_binding_id for call in thread_calls} == {
        "engineering",
        "payments",
    }
    assert hook.coverage.root_messages_inspected == 4
    assert hook.coverage.partial_result_reasons == ("thread_limit",)


def test_bounded_search_filters_fetched_roots_before_answer_evidence() -> None:
    binding = _binding("engineering", "Engineering")
    configuration = _configuration((binding,))
    request = SlackAskRequestV1(
        intent=SlackAskIntentV1.BOUNDED_RECENT_SEARCH,
        question="payment",
        binding_references=("engineering",),
    )
    plan = SlackAskPlannerV1().build_plan(
        configuration=configuration,
        request=request,
    )
    hook = SlackAskStagedExecutionV1(
        planner=SlackAskPlannerV1(),
        request=request,
        initial_coverage=plan.coverage,
        resolved_bindings=(binding,),
        proposal_validator=lambda values: (),
    )
    list_call = _slack_call(
        call_id="list-engineering",
        binding_id="engineering",
        capability_id=SLACK_CONVERSATION_LIST_CAPABILITY_ID,
        request=SlackConversationListLiveRequestV1(page_size=15),
    )
    roots = tuple(
        _slack_item(
            timestamp,
            {
                "change_kind": "upsert",
                "content_available": True,
                "content_mode": "structured_record",
                "item_type": "slack_conversation_message",
                "message_ts": timestamp,
                "reply_count": 1 if "3601" in timestamp else 0,
                "text": text,
                "thread_root_ts": None,
            },
        )
        for timestamp, text in (
            ("1704153601.000001", "payment incident"),
            ("1704153602.000001", "unrelated update"),
        )
    )
    hook.expand(
        stage=1,
        calls=(list_call,),
        outcomes=(_slack_outcome(list_call, items=roots),),
        attempted_calls=(list_call,),
        remaining_provider_call_budget=2,
        deadline_reached=False,
    )
    evidence = hook.include_evidence(
        tuple(
            SimpleNamespace(call_id=list_call.call_id, remote_item_id=item.remote_item_id)
            for item in roots
        )
    )
    assert [item.remote_item_id for item in evidence] == ["1704153601.000001"]
    assert hook.coverage.threads_expanded == 0
