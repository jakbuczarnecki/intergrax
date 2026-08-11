# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from local_workspace_application.conversation.conversation_setup_onboarding import (
    ConversationSetupOnboardingPresenter,
)
from local_workspace_application.workspaces.workspace_setup_snapshot_service import (
    SetupAttentionV1,
    SetupKnowledgeSummaryV1,
    SetupNextActionV1,
    SetupPhaseV1,
    SetupRecentOperationV1,
    WorkspaceSetupSnapshotService,
    WorkspaceSetupSnapshotV1,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 11, 8, 0, tzinfo=UTC)
_PRESENTER = ConversationSetupOnboardingPresenter()


def _snapshot(
    *,
    phase: SetupPhaseV1,
    next_action: SetupNextActionV1,
    can_ask: bool = False,
    suggested_question: str | None = None,
    attention: SetupAttentionV1 | None = None,
    sync_in_progress: bool = False,
) -> WorkspaceSetupSnapshotV1:
    return WorkspaceSetupSnapshotV1(
        workspace_id="ws-1",
        host_ready=True,
        phase=phase,
        can_ask=can_ask,
        has_usable_knowledge=phase is SetupPhaseV1.READY and can_ask,
        sync_in_progress=sync_in_progress,
        attention_required=phase is SetupPhaseV1.ATTENTION_REQUIRED,
        knowledge_summary=SetupKnowledgeSummaryV1(
            total=1,
            indexed=1,
            live=0,
            active=1,
            disabled=0,
            attention_required=0,
            usable=1 if can_ask else 0,
        ),
        recent_operation=SetupRecentOperationV1(
            operation_id="op-1",
            operation_type="sync",
            status="running",
        ) if sync_in_progress else None,
        attention=attention,
        next_action=next_action,
        suggested_question=suggested_question,
        updated_at=_NOW,
    )


def test_welcome_first_dm_no_workspace() -> None:
    text = _PRESENTER.render_welcome([])
    assert "Welcome to LKW" in text
    assert "workspace" in text.casefold()
    assert "tenant_id" not in text
    assert "workspace_id" not in text


def test_no_knowledge_guidance() -> None:
    snapshot = _snapshot(
        phase=SetupPhaseV1.NO_KNOWLEDGE,
        next_action=SetupNextActionV1.ADD_SOURCE,
    )
    text = _PRESENTER.render_snapshot_guidance(snapshot)
    assert "attachment" in text.casefold()
    assert "%" not in text


def test_syncing_preparation_state() -> None:
    snapshot = _snapshot(
        phase=SetupPhaseV1.SYNCING,
        next_action=SetupNextActionV1.WAIT_FOR_SYNC,
        sync_in_progress=True,
    )
    text = _PRESENTER.render_snapshot_guidance(snapshot)
    assert "being prepared" in text.casefold()
    assert "%" not in text
    assert "ETA" not in text


def test_configuring_state() -> None:
    snapshot = _snapshot(
        phase=SetupPhaseV1.CONFIGURING,
        next_action=SetupNextActionV1.WAIT_FOR_SYNC,
    )
    text = _PRESENTER.render_snapshot_guidance(snapshot)
    assert "prepared" in text.casefold() or "configured" in text.casefold()


def test_attention_required_safe_rendering() -> None:
    snapshot = _snapshot(
        phase=SetupPhaseV1.ATTENTION_REQUIRED,
        next_action=SetupNextActionV1.RETRY_OR_FIX_SOURCE,
        attention=SetupAttentionV1(
            knowledge_item_id="item-1",
            error_code="sync_failed",
            available_actions=("retry_sync",),
        ),
    )
    text = _PRESENTER.render_snapshot_guidance(snapshot)
    assert "attention" in text.casefold()
    assert "sync_failed" not in text
    assert "traceback" not in text.casefold()
    assert "syncing" in text.casefold()


def test_ready_can_ask_suggested_question() -> None:
    snapshot = _snapshot(
        phase=SetupPhaseV1.READY,
        next_action=SetupNextActionV1.ASK_QUESTION,
        can_ask=True,
        suggested_question="What information is available in Docs?",
    )
    text = _PRESENTER.render_snapshot_guidance(snapshot)
    assert "ready" in text.casefold()
    assert "What information is available in Docs?" in text
    assert not _PRESENTER.should_gate_question(snapshot)


def test_ready_cannot_ask_blocks_question() -> None:
    snapshot = _snapshot(
        phase=SetupPhaseV1.READY,
        next_action=SetupNextActionV1.NONE,
        can_ask=False,
    )
    text = _PRESENTER.render_snapshot_guidance(snapshot)
    assert "temporarily unavailable" in text.casefold()
    assert _PRESENTER.should_gate_question(snapshot)


def test_ask_blocked_uses_snapshot_not_hardcoded_phase_rules() -> None:
    snapshot = _snapshot(
        phase=SetupPhaseV1.SYNCING,
        next_action=SetupNextActionV1.WAIT_FOR_SYNC,
        sync_in_progress=True,
    )
    blocked = _PRESENTER.render_ask_blocked(snapshot)
    assert blocked.startswith("I cannot answer that yet.")
    assert _PRESENTER.should_gate_question(snapshot)


def test_architecture_presenter_does_not_duplicate_snapshot_derivation() -> None:
    module = inspect.getmodule(ConversationSetupOnboardingPresenter)
    source = inspect.getsource(module)
    forbidden = (
        "_item_usable",
        "_item_needs_attention",
        "_next_action_for_phase",
        "_derive_from_state",
        "derive_snapshot",
    )
    for name in forbidden:
        assert name not in source

    service_source = inspect.getsource(WorkspaceSetupSnapshotService)
    assert "_next_action_for_phase" in service_source
    assert "_item_usable" in service_source
