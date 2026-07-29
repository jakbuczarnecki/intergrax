# © Artur Czarnecki. All rights reserved.

"""Fixed versioned qualification corpus for LKW conversational planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from local_workspace_application.benchmarks.local_model_qualification.contracts import CORPUS_VERSION
from local_workspace_application.conversation.interaction_models import (
    ConversationPlanningAttachment,
    ConversationPlanningRequest,
    ConversationPlanningSourceCandidate,
    ConversationPlanningWorkspace,
    WorkspaceReferenceKind,
)

# Cases 1–3 mirror live E2E scenarios in
# tests/e2e/test_conversation_interaction_planner_live.py:
# test_e2e_mixed_source_routing,
# test_e2e_workspace_target_without_activation,
# test_e2e_explicit_workspace_activation.


@dataclass(frozen=True, slots=True)
class ExpectedWorkspaceRef:
    kind: WorkspaceReferenceKind
    value: str | None


@dataclass(frozen=True, slots=True)
class ExpectedSourceGroup:
    workspace: ExpectedWorkspaceRef
    object_types: frozenset[str]
    object_count: int
    values: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class ExpectedCaseOutcome:
    action_type_counts: Mapping[str, int]
    workspace_refs_by_action: tuple[ExpectedWorkspaceRef, ...] = ()
    object_count: int | None = None
    object_types: frozenset[str] | None = None
    source_groups: tuple[ExpectedSourceGroup, ...] = ()
    attachment_ids: frozenset[str] | None = None
    candidate_reference_kind: str | None = None
    candidate_reference: str | None = None
    min_clarifications: int = 0
    max_clarifications: int = 0
    forbidden_action_types: frozenset[str] = frozenset()
    allowed_state_changing: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class QualificationCase:
    case_id: str
    description: str
    request: ConversationPlanningRequest
    expected: ExpectedCaseOutcome


_STATE_CHANGING = frozenset(
    {
        "workspace.create",
        "workspace.activate",
        "workspace.delete",
        "knowledge.add_attachments",
        "knowledge.add_sources",
        "source_candidate.attach",
    }
)

_STANDARD_WORKSPACES = (
    ConversationPlanningWorkspace(workspace_id="ws-1", name="finanse", is_active=True),
    ConversationPlanningWorkspace(workspace_id="ws-2", name="magazyn", is_active=False),
)

_CASES: tuple[QualificationCase, ...] = (
    QualificationCase(
        case_id="planner.mixed_source_ordinal_routing",
        description="Mixed URL and local paths routed to different workspace ordinals",
        request=ConversationPlanningRequest(
            message_text=(
                "ten adres https://cenniki.pl wrzuć do workspace numer 1, "
                r"a pliki C:\cenniki\hurt.xlsx i C:\cenniki\detal.xlsx "
                "dodaj do workspace numer 2"
            ),
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"knowledge.add_sources": 2},
            object_count=3,
            object_types=frozenset({"web_url", "local_file_reference"}),
            source_groups=(
                ExpectedSourceGroup(
                    workspace=ExpectedWorkspaceRef(WorkspaceReferenceKind.ordinal, "1"),
                    object_types=frozenset({"web_url"}),
                    object_count=1,
                    values=frozenset({"https://cenniki.pl"}),
                ),
                ExpectedSourceGroup(
                    workspace=ExpectedWorkspaceRef(WorkspaceReferenceKind.ordinal, "2"),
                    object_types=frozenset({"local_file_reference"}),
                    object_count=2,
                    values=frozenset({r"C:\cenniki\hurt.xlsx", r"C:\cenniki\detal.xlsx"}),
                ),
            ),
            forbidden_action_types=frozenset({"workspace.activate"}),
            allowed_state_changing=frozenset({"knowledge.add_sources"}),
        ),
    ),
    QualificationCase(
        case_id="planner.target_workspace_without_activation",
        description="Target workspace by name without switching active workspace",
        request=ConversationPlanningRequest(
            message_text="dodaj https://example.com/docs do workspace magazyn",
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"knowledge.add_sources": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.name, "magazyn"),),
            object_count=1,
            object_types=frozenset({"web_url"}),
            forbidden_action_types=frozenset({"workspace.activate"}),
            allowed_state_changing=frozenset({"knowledge.add_sources"}),
        ),
    ),
    QualificationCase(
        case_id="planner.explicit_workspace_activation",
        description="Explicit workspace activation request",
        request=ConversationPlanningRequest(
            message_text="przełącz mnie na workspace magazyn",
            available_workspaces=(
                ConversationPlanningWorkspace(workspace_id="ws-1", name="magazyn", is_active=False),
            ),
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"workspace.activate": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.name, "magazyn"),),
            object_count=0,
            forbidden_action_types=frozenset({"knowledge.add_sources"}),
            allowed_state_changing=frozenset({"workspace.activate"}),
        ),
    ),
    QualificationCase(
        case_id="planner.active_workspace_source_add",
        description="Add source to active workspace",
        request=ConversationPlanningRequest(
            message_text="dodaj https://example.com/guide do aktywnego workspace",
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"knowledge.add_sources": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.active, None),),
            object_count=1,
            forbidden_action_types=frozenset({"workspace.activate"}),
            allowed_state_changing=frozenset({"knowledge.add_sources"}),
        ),
    ),
    QualificationCase(
        case_id="planner.workspace_list",
        description="List available workspaces",
        request=ConversationPlanningRequest(
            message_text="pokaż listę moich workspace",
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"workspace.list": 1},
            object_count=0,
            allowed_state_changing=frozenset(),
        ),
    ),
    QualificationCase(
        case_id="planner.source_list_named_workspace",
        description="List sources in named workspace",
        request=ConversationPlanningRequest(
            message_text="pokaż źródła w workspace magazyn",
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"source.list": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.name, "magazyn"),),
            forbidden_action_types=frozenset({"workspace.activate"}),
            allowed_state_changing=frozenset(),
        ),
    ),
    QualificationCase(
        case_id="planner.url_question_not_ingestion",
        description="URL inside question should not trigger ingestion",
        request=ConversationPlanningRequest(
            message_text="co sądzisz o https://reviews.example w workspace finanse?",
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"workspace.ask": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.name, "finanse"),),
            object_count=0,
            forbidden_action_types=frozenset({"knowledge.add_sources", "workspace.activate"}),
            allowed_state_changing=frozenset(),
        ),
    ),
    QualificationCase(
        case_id="planner.attachment_ingestion",
        description="Attach uploaded files to named workspace",
        request=ConversationPlanningRequest(
            message_text="dodaj oba załączone pliki do workspace magazyn",
            attachments=(
                ConversationPlanningAttachment(
                    attachment_id="att-1",
                    file_name="raport.pdf",
                    content_type="application/pdf",
                ),
                ConversationPlanningAttachment(
                    attachment_id="att-2",
                    file_name="dane.xlsx",
                    content_type=(
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ),
                ),
            ),
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"knowledge.add_attachments": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.name, "magazyn"),),
            attachment_ids=frozenset({"att-1", "att-2"}),
            forbidden_action_types=frozenset({"workspace.activate"}),
            allowed_state_changing=frozenset({"knowledge.add_attachments"}),
        ),
    ),
    QualificationCase(
        case_id="planner.ambiguous_missing_workspace_target",
        description="Missing workspace target should trigger clarification",
        request=ConversationPlanningRequest(
            message_text="dodaj https://example.com/docs do workspace",
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={},
            min_clarifications=1,
            max_clarifications=20,
            forbidden_action_types=frozenset(
                {"knowledge.add_sources", "workspace.activate", *_STATE_CHANGING}
            ),
            allowed_state_changing=frozenset(),
        ),
    ),
    QualificationCase(
        case_id="planner.explicit_workspace_delete",
        description="Explicit workspace deletion",
        request=ConversationPlanningRequest(
            message_text="usuń workspace archiwum",
            available_workspaces=(
                ConversationPlanningWorkspace(workspace_id="ws-1", name="archiwum", is_active=False),
            ),
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"workspace.delete": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.name, "archiwum"),),
            allowed_state_changing=frozenset({"workspace.delete"}),
        ),
    ),
    QualificationCase(
        case_id="planner.source_candidate_list",
        description="List available source candidates for workspace",
        request=ConversationPlanningRequest(
            message_text="pokaż dostępne źródła dla workspace magazyn",
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"source_candidate.list": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.name, "magazyn"),),
            forbidden_action_types=frozenset({"workspace.activate"}),
            allowed_state_changing=frozenset(),
        ),
    ),
    QualificationCase(
        case_id="planner.source_candidate_attach_ordinal",
        description="Attach source candidate by ordinal reference",
        request=ConversationPlanningRequest(
            message_text="dołącz drugie znalezione źródło do workspace magazyn",
            available_workspaces=_STANDARD_WORKSPACES,
            active_workspace_id="ws-1",
            available_source_candidates=(
                ConversationPlanningSourceCandidate(
                    candidate_id="candidate-1",
                    label="Dokument pierwszy",
                    source_type="web_url",
                    available=True,
                ),
                ConversationPlanningSourceCandidate(
                    candidate_id="candidate-2",
                    label="Dokument drugi",
                    source_type="web_url",
                    available=True,
                ),
            ),
        ),
        expected=ExpectedCaseOutcome(
            action_type_counts={"source_candidate.attach": 1},
            workspace_refs_by_action=(ExpectedWorkspaceRef(WorkspaceReferenceKind.name, "magazyn"),),
            candidate_reference_kind="ordinal",
            candidate_reference="2",
            forbidden_action_types=frozenset({"workspace.activate"}),
            allowed_state_changing=frozenset({"source_candidate.attach"}),
        ),
    ),
)


def corpus_version() -> str:
    return CORPUS_VERSION


def qualification_cases() -> tuple[QualificationCase, ...]:
    return _CASES


def case_by_id(case_id: str) -> QualificationCase:
    for case in _CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def state_changing_action_types() -> frozenset[str]:
    return _STATE_CHANGING
