# © Artur Czarnecki. All rights reserved.

"""Deterministic semantic evaluation for qualification plans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    SemanticFailureCategory,
    WorkspaceReferenceSummary,
)
from local_workspace_application.benchmarks.local_model_qualification.corpus import (
    ExpectedCaseOutcome,
    ExpectedSourceGroup,
    QualificationCase,
    state_changing_action_types,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    KnowledgeAddAttachmentsPlannedAction,
    KnowledgeAddSourcesPlannedAction,
    SourceCandidateAttachPlannedAction,
    WorkspaceReference,
    WorkspaceReferenceKind,
)


@dataclass(frozen=True, slots=True)
class SemanticEvaluation:
    passed: bool
    failure_categories: tuple[str, ...]
    primary_failure_category: str | None
    action_types: tuple[str, ...]
    object_types: tuple[str, ...]
    workspace_reference_summaries: tuple[WorkspaceReferenceSummary, ...]
    clarification_count: int
    unsafe_state_change_count: int


def _workspace_summary(ref: WorkspaceReference) -> WorkspaceReferenceSummary:
    return WorkspaceReferenceSummary(kind=ref.kind.value, value=ref.value)


def _action_workspace_ref(action: object) -> WorkspaceReference | None:
    workspace = getattr(action, "workspace", None)
    if isinstance(workspace, WorkspaceReference):
        return workspace
    return None


def _count_action_types(plan: ConversationInteractionPlan) -> Counter[str]:
    return Counter(action.action_type for action in plan.actions)


def _refs_for_workspace_actions(plan: ConversationInteractionPlan) -> list[WorkspaceReference]:
    refs: list[WorkspaceReference] = []
    for action in plan.actions:
        ref = _action_workspace_ref(action)
        if ref is not None:
            refs.append(ref)
    return refs


def _objects_for_action(
    plan: ConversationInteractionPlan,
    action: KnowledgeAddSourcesPlannedAction,
) -> list[object]:
    object_map = {obj.object_id: obj for obj in plan.objects}
    return [object_map[source_id] for source_id in action.source_object_ids]


def _workspace_ref_matches(ref: WorkspaceReference, expected_kind: WorkspaceReferenceKind, expected_value: str | None) -> bool:
    if ref.kind != expected_kind:
        return False
    if expected_kind == WorkspaceReferenceKind.active:
        return ref.value is None
    return ref.value == expected_value


def _evaluate_source_groups(
    plan: ConversationInteractionPlan,
    expected_groups: tuple[ExpectedSourceGroup, ...],
    failures: list[str],
) -> None:
    add_sources = [
        action
        for action in plan.actions
        if isinstance(action, KnowledgeAddSourcesPlannedAction)
    ]
    if len(add_sources) != len(expected_groups):
        failures.append(SemanticFailureCategory.WRONG_SOURCE_GROUPING.value)
        return
    matched = set()
    for group in expected_groups:
        found = False
        for index, action in enumerate(add_sources):
            if index in matched:
                continue
            if not _workspace_ref_matches(action.workspace, group.workspace.kind, group.workspace.value):
                continue
            objects = _objects_for_action(plan, action)
            object_types = {obj.object_type for obj in objects}
            if object_types != group.object_types or len(objects) != group.object_count:
                continue
            if group.values is not None and {obj.value for obj in objects} != group.values:
                continue
            matched.add(index)
            found = True
            break
        if not found:
            failures.append(SemanticFailureCategory.WRONG_SOURCE_GROUPING.value)


def _pick_primary(failures: list[str]) -> str | None:
    if not failures:
        return None
    precedence = [
        SemanticFailureCategory.UNEXPECTED_STATE_CHANGE.value,
        SemanticFailureCategory.UNNECESSARY_WORKSPACE_ACTIVATE.value,
        SemanticFailureCategory.MISSING_REQUIRED_ACTION.value,
        SemanticFailureCategory.WRONG_ACTION_TYPE.value,
        SemanticFailureCategory.WRONG_ACTION_COUNT.value,
        SemanticFailureCategory.WRONG_WORKSPACE_REFERENCE.value,
        SemanticFailureCategory.WRONG_SOURCE_EXTRACTION.value,
        SemanticFailureCategory.WRONG_SOURCE_GROUPING.value,
        SemanticFailureCategory.WRONG_ATTACHMENT_SELECTION.value,
        SemanticFailureCategory.WRONG_CANDIDATE_REFERENCE.value,
        SemanticFailureCategory.MISSING_CLARIFICATION.value,
        SemanticFailureCategory.UNNECESSARY_CLARIFICATION.value,
        SemanticFailureCategory.SEMANTIC_MISMATCH.value,
    ]
    failure_set = set(failures)
    for category in precedence:
        if category in failure_set:
            return category
    return failures[0]


def evaluate_semantics(plan: ConversationInteractionPlan, case: QualificationCase) -> SemanticEvaluation:
    expected = case.expected
    failures: list[str] = []
    action_counts = _count_action_types(plan)
    action_types = tuple(sorted(action_counts.keys()))
    object_types = tuple(sorted({obj.object_type for obj in plan.objects}))
    workspace_refs = tuple(
        _workspace_summary(ref) for ref in _refs_for_workspace_actions(plan)
    )
    clarification_count = len(plan.clarifications)

    unsafe_count = 0
    for action_type, count in action_counts.items():
        if action_type in state_changing_action_types():
            allowed = expected.action_type_counts.get(action_type, 0)
            extra = count - allowed
            if extra > 0:
                unsafe_count += extra
                if action_type == "workspace.activate":
                    failures.append(SemanticFailureCategory.UNNECESSARY_WORKSPACE_ACTIVATE.value)
                else:
                    failures.append(SemanticFailureCategory.UNEXPECTED_STATE_CHANGE.value)

    if action_counts != Counter(expected.action_type_counts):
        if sum(action_counts.values()) != sum(expected.action_type_counts.values()):
            failures.append(SemanticFailureCategory.WRONG_ACTION_COUNT.value)
        else:
            failures.append(SemanticFailureCategory.WRONG_ACTION_TYPE.value)
        if not action_counts and expected.action_type_counts:
            failures.append(SemanticFailureCategory.MISSING_REQUIRED_ACTION.value)

    if expected.workspace_refs_by_action:
        actual_refs = _refs_for_workspace_actions(plan)
        expected_refs = list(expected.workspace_refs_by_action)
        if len(actual_refs) != len(expected_refs):
            failures.append(SemanticFailureCategory.WRONG_WORKSPACE_REFERENCE.value)
        else:
            for actual, exp in zip(actual_refs, expected_refs, strict=True):
                if not _workspace_ref_matches(actual, exp.kind, exp.value):
                    failures.append(SemanticFailureCategory.WRONG_WORKSPACE_REFERENCE.value)
                    break

    if expected.object_count is not None and len(plan.objects) != expected.object_count:
        failures.append(SemanticFailureCategory.WRONG_SOURCE_EXTRACTION.value)
    if expected.object_types is not None:
        actual_object_types = {obj.object_type for obj in plan.objects}
        if actual_object_types != expected.object_types:
            failures.append(SemanticFailureCategory.WRONG_SOURCE_EXTRACTION.value)

    if expected.source_groups:
        _evaluate_source_groups(plan, expected.source_groups, failures)

    if expected.attachment_ids is not None:
        attachment_actions = [
            action
            for action in plan.actions
            if isinstance(action, KnowledgeAddAttachmentsPlannedAction)
        ]
        if len(attachment_actions) != 1:
            failures.append(SemanticFailureCategory.WRONG_ATTACHMENT_SELECTION.value)
        else:
            actual_ids = frozenset(attachment_actions[0].attachment_ids)
            if actual_ids != expected.attachment_ids:
                failures.append(SemanticFailureCategory.WRONG_ATTACHMENT_SELECTION.value)

    if expected.candidate_reference_kind is not None:
        attach_actions = [
            action
            for action in plan.actions
            if isinstance(action, SourceCandidateAttachPlannedAction)
        ]
        if len(attach_actions) != 1:
            failures.append(SemanticFailureCategory.WRONG_CANDIDATE_REFERENCE.value)
        else:
            action = attach_actions[0]
            if (
                action.candidate_reference_kind != expected.candidate_reference_kind
                or action.candidate_reference != expected.candidate_reference
            ):
                failures.append(SemanticFailureCategory.WRONG_CANDIDATE_REFERENCE.value)

    for forbidden in expected.forbidden_action_types:
        if action_counts.get(forbidden, 0) > 0:
            if forbidden == "workspace.activate":
                failures.append(SemanticFailureCategory.UNNECESSARY_WORKSPACE_ACTIVATE.value)
            elif forbidden in state_changing_action_types():
                failures.append(SemanticFailureCategory.UNEXPECTED_STATE_CHANGE.value)
            else:
                failures.append(SemanticFailureCategory.WRONG_ACTION_TYPE.value)

    if clarification_count < expected.min_clarifications:
        failures.append(SemanticFailureCategory.MISSING_CLARIFICATION.value)
    if clarification_count > expected.max_clarifications:
        failures.append(SemanticFailureCategory.UNNECESSARY_CLARIFICATION.value)

    unique_failures = tuple(dict.fromkeys(failures))
    primary = _pick_primary(list(unique_failures))
    return SemanticEvaluation(
        passed=not unique_failures,
        failure_categories=unique_failures,
        primary_failure_category=primary,
        action_types=action_types,
        object_types=object_types,
        workspace_reference_summaries=workspace_refs,
        clarification_count=clarification_count,
        unsafe_state_change_count=unsafe_count,
    )
