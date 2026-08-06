from __future__ import annotations

from datetime import UTC, datetime

import pytest
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
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.slack import (
    SLACK_CONVERSATION_LIST_CAPABILITY_ID,
    SLACK_CONVERSATION_READ_CAPABILITY_ID,
    SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID,
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
            else (reference,),
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
                binding_references=("#engineering",),
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


def test_recent_search_can_bound_thread_expansion_without_provider_discovery() -> None:
    plan = SlackAskPlannerV1().build_plan(
        configuration=_configuration((_binding("engineering", "Engineering"),)),
        request=SlackAskRequestV1(
            intent=SlackAskIntentV1.BOUNDED_RECENT_SEARCH,
            question="payment outage",
            binding_references=("engineering",),
            thread_root_timestamps=(_ROOT, "1704153601.000001", "1704153602.000001"),
        ),
    )
    assert len(plan.ordered_live_call_proposals) == 4
    assert plan.ordered_live_call_proposals[0].capability_id == (
        SLACK_CONVERSATION_LIST_CAPABILITY_ID
    )
    assert all(
        proposal.capability_id == SLACK_CONVERSATION_THREAD_READ_CAPABILITY_ID
        for proposal in plan.ordered_live_call_proposals[1:]
    )
    assert plan.maximum_thread_expansions == 3
    assert plan.maximum_replies_per_thread == 15


def test_thread_selection_is_deterministic_and_budget_bounded() -> None:
    candidates = (
        SlackAskRootCandidateV1(
            message_ts="1704153601.000001",
            text="payment incident",
            reply_count=1,
        ),
        SlackAskRootCandidateV1(
            message_ts="1704153602.000001",
            text="payment incident decision",
            reply_count=4,
            explicit_reference=True,
        ),
        SlackAskRootCandidateV1(
            message_ts="1704153603.000001",
            text="unrelated",
            reply_count=99,
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
