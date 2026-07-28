# © Artur Czarnecki. All rights reserved.

"""Live E2E proof for LKW conversational interaction planner (plan v2)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from intergrax.applications._shared.llm_resolver import resolve_llm_adapter

from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningRequest,
    ConversationPlanningWorkspace,
    ExtractedObject,
    KnowledgeAddSourcesPlannedAction,
    PlannedAction,
    WorkspaceActivatePlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
    validate_plan_against_request,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FLAG = "INTERGRAX_LKW_PLANNER_E2E"
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

MIXED_ROUTING_MESSAGE = (
    "ten adres https://cenniki.pl wrzuć do workspace numer 1, "
    r"a pliki C:\cenniki\hurt.xlsx i C:\cenniki\detal.xlsx "
    "dodaj do workspace numer 2"
)


def _e2e_enabled() -> bool:
    return os.environ.get(_E2E_FLAG, "").strip() == "1"


def _load_lkw_env() -> None:
    if not _ENV_FILE.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_FILE, override=False)


def _require_e2e_config() -> tuple[str, str]:
    _load_lkw_env()
    provider = os.environ.get("INTERGRAX_LLM_PROVIDER", "").strip()
    model = os.environ.get("INTERGRAX_LLM_MODEL", "").strip()
    if not provider:
        pytest.fail("INTERGRAX_LLM_PROVIDER is required when INTERGRAX_LKW_PLANNER_E2E=1")
    if not model:
        pytest.fail("INTERGRAX_LLM_MODEL is required when INTERGRAX_LKW_PLANNER_E2E=1")
    return provider, model


@pytest.fixture(scope="module")
def live_planner_adapter():
    if not _e2e_enabled():
        pytest.skip(f"{_E2E_FLAG} is not set")
    provider, model = _require_e2e_config()
    adapter = resolve_llm_adapter(None)
    if not adapter.supports_structured_output():
        pytest.fail(
            f"configured adapter does not support structured output "
            f"(provider={provider}, model={model})"
        )
    return adapter


def _object_map(plan: ConversationInteractionPlan) -> dict[str, ExtractedObject]:
    return {obj.object_id: obj for obj in plan.objects}


def _objects_for_action(
    plan: ConversationInteractionPlan,
    action: KnowledgeAddSourcesPlannedAction,
) -> tuple[ExtractedObject, ...]:
    mapping = _object_map(plan)
    return tuple(mapping[oid] for oid in action.source_object_ids)


def _find_object_by_type_and_value(
    plan: ConversationInteractionPlan,
    *,
    object_type: str,
    value: str,
) -> ExtractedObject:
    for obj in plan.objects:
        if obj.object_type == object_type and obj.value == value:
            return obj
    raise AssertionError(f"object not found: type={object_type} value={value!r}")


def _workspace_ref_matches(
    ref: WorkspaceReference,
    *,
    kind: WorkspaceReferenceKind,
    value: str | None = None,
) -> bool:
    if ref.kind != kind:
        return False
    if value is None:
        return True
    return ref.value == value


def _assert_evidence_grounded(message_text: str, obj: ExtractedObject) -> None:
    evidence = obj.evidence
    assert message_text[evidence.start : evidence.end] == evidence.text
    assert obj.value == evidence.text


def _plan_failure_summary(
    plan: ConversationInteractionPlan,
    *,
    provider: str,
    model: str,
    error: str | None = None,
) -> str:
    object_types = [obj.object_type for obj in plan.objects]
    object_values = [obj.value for obj in plan.objects]
    action_types = [action.action_type for action in plan.actions]
    workspace_refs = [
        f"{getattr(action, 'workspace', None).kind}:{getattr(action, 'workspace', None).value}"  # type: ignore[union-attr]
        for action in plan.actions
        if hasattr(action, "workspace")
    ]
    parts = [
        f"provider={provider}",
        f"model={model}",
        f"plan_version={plan.plan_version}",
        f"object_types={object_types}",
        f"object_values={object_values}",
        f"action_types={action_types}",
        f"workspace_refs={workspace_refs}",
    ]
    if error:
        parts.append(f"validation_error={error}")
    return "; ".join(parts)


def _action_types(plan: ConversationInteractionPlan) -> set[str]:
    return {action.action_type for action in plan.actions}


def _add_sources_actions(plan: ConversationInteractionPlan) -> list[KnowledgeAddSourcesPlannedAction]:
    return [
        action
        for action in plan.actions
        if isinstance(action, KnowledgeAddSourcesPlannedAction)
    ]


def _referenced_object_ids(plan: ConversationInteractionPlan) -> set[str]:
    ids: set[str] = set()
    for action in _add_sources_actions(plan):
        ids.update(action.source_object_ids)
    return ids


@pytest.mark.asyncio
async def test_e2e_mixed_source_routing(live_planner_adapter) -> None:
    provider, model = _require_e2e_config()
    request = ConversationPlanningRequest(
        message_text=MIXED_ROUTING_MESSAGE,
        available_workspaces=(
            ConversationPlanningWorkspace(workspace_id="ws-1", name="finanse", is_active=True),
            ConversationPlanningWorkspace(workspace_id="ws-2", name="magazyn", is_active=False),
        ),
        active_workspace_id="ws-1",
    )
    planner = ConversationInteractionPlanner(live_planner_adapter)
    plan = await planner.plan(request, run_id="lkw-planner-e2e-mixed-routing")

    try:
        assert not plan.clarifications, _plan_failure_summary(plan, provider=provider, model=model)
        assert len(plan.objects) == 3, _plan_failure_summary(plan, provider=provider, model=model)
        web_urls = [o for o in plan.objects if o.object_type == "web_url"]
        local_refs = [o for o in plan.objects if o.object_type == "local_file_reference"]
        assert len(web_urls) == 1, _plan_failure_summary(plan, provider=provider, model=model)
        assert len(local_refs) == 2, _plan_failure_summary(plan, provider=provider, model=model)

        expected_values = {
            "https://cenniki.pl",
            r"C:\cenniki\hurt.xlsx",
            r"C:\cenniki\detal.xlsx",
        }
        assert {o.value for o in plan.objects} == expected_values

        for obj in plan.objects:
            _assert_evidence_grounded(request.message_text, obj)

        add_sources = _add_sources_actions(plan)
        assert len(add_sources) == 2, _plan_failure_summary(plan, provider=provider, model=model)

        url_action = next(
            (
                action
                for action in add_sources
                if _workspace_ref_matches(
                    action.workspace,
                    kind=WorkspaceReferenceKind.ordinal,
                    value="1",
                )
            ),
            None,
        )
        assert url_action is not None, _plan_failure_summary(plan, provider=provider, model=model)
        url_objects = _objects_for_action(plan, url_action)
        assert len(url_objects) == 1
        assert url_objects[0].object_type == "web_url"
        assert url_objects[0].value == "https://cenniki.pl"

        local_action = next(
            (
                action
                for action in add_sources
                if _workspace_ref_matches(
                    action.workspace,
                    kind=WorkspaceReferenceKind.ordinal,
                    value="2",
                )
            ),
            None,
        )
        assert local_action is not None, _plan_failure_summary(plan, provider=provider, model=model)
        local_objects = _objects_for_action(plan, local_action)
        assert len(local_objects) == 2
        assert {o.value for o in local_objects} == {
            r"C:\cenniki\hurt.xlsx",
            r"C:\cenniki\detal.xlsx",
        }

        assert _referenced_object_ids(plan) == {o.object_id for o in plan.objects}
        assert "workspace.activate" not in _action_types(plan)
        validate_plan_against_request(plan, request)
    except AssertionError as exc:
        raise AssertionError(
            f"{exc}\n{_plan_failure_summary(plan, provider=provider, model=model)}"
        ) from exc


@pytest.mark.asyncio
async def test_e2e_workspace_target_without_activation(live_planner_adapter) -> None:
    provider, model = _require_e2e_config()
    message = "dodaj https://example.com/docs do workspace magazyn"
    request = ConversationPlanningRequest(
        message_text=message,
        available_workspaces=(
            ConversationPlanningWorkspace(workspace_id="ws-1", name="magazyn", is_active=True),
        ),
        active_workspace_id="ws-1",
    )
    planner = ConversationInteractionPlanner(live_planner_adapter)
    plan = await planner.plan(request, run_id="lkw-planner-e2e-target-workspace")

    try:
        web_urls = [o for o in plan.objects if o.object_type == "web_url"]
        assert len(web_urls) == 1, _plan_failure_summary(plan, provider=provider, model=model)
        assert web_urls[0].value == "https://example.com/docs"
        _assert_evidence_grounded(message, web_urls[0])

        add_sources = _add_sources_actions(plan)
        assert len(add_sources) == 1, _plan_failure_summary(plan, provider=provider, model=model)
        assert _workspace_ref_matches(
            add_sources[0].workspace,
            kind=WorkspaceReferenceKind.name,
            value="magazyn",
        )
        assert "workspace.activate" not in _action_types(plan)
        validate_plan_against_request(plan, request)
    except AssertionError as exc:
        raise AssertionError(
            f"{exc}\n{_plan_failure_summary(plan, provider=provider, model=model)}"
        ) from exc


@pytest.mark.asyncio
async def test_e2e_explicit_workspace_activation(live_planner_adapter) -> None:
    provider, model = _require_e2e_config()
    message = "przełącz mnie na workspace magazyn"
    request = ConversationPlanningRequest(
        message_text=message,
        available_workspaces=(
            ConversationPlanningWorkspace(workspace_id="ws-1", name="magazyn", is_active=False),
        ),
    )
    planner = ConversationInteractionPlanner(live_planner_adapter)
    plan = await planner.plan(request, run_id="lkw-planner-e2e-explicit-activation")

    try:
        activate_actions = [
            action
            for action in plan.actions
            if isinstance(action, WorkspaceActivatePlannedAction)
        ]
        assert len(activate_actions) == 1, _plan_failure_summary(plan, provider=provider, model=model)
        assert _workspace_ref_matches(
            activate_actions[0].workspace,
            kind=WorkspaceReferenceKind.name,
            value="magazyn",
        )
        assert "knowledge.add_sources" not in _action_types(plan)
        assert not plan.objects, _plan_failure_summary(plan, provider=provider, model=model)
        validate_plan_against_request(plan, request)
    except AssertionError as exc:
        raise AssertionError(
            f"{exc}\n{_plan_failure_summary(plan, provider=provider, model=model)}"
        ) from exc
