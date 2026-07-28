# © Artur Czarnecki. All rights reserved.

"""System prompt and safe planning context for the conversational interaction planner."""

from __future__ import annotations

import json
from enum import Enum

from intergrax.llm.messages import ChatMessage

from local_workspace_application.conversation.interaction_models import ConversationPlanningRequest

_SYSTEM_PROMPT = """You are an interaction planner for a knowledge workspace product. You do NOT execute actions.

Return semantic intent only as a structured draft. Do not invent technical IDs or character offsets.

Rules:
1. Return ONLY a structured semantic draft matching the supplied schema.
2. Analyze the ENTIRE user message, not just the beginning.
3. The user may write in any language, with typos.
4. One message may contain multiple actions and multiple source types (files, URLs, local paths, workspace targets).
5. Determine the direction of every action explicitly.
6. Every workspace-dependent action must include an explicit workspace reference.
7. A workspace used as the TARGET of an operation does NOT change the active workspace.
8. Return workspace.activate ONLY when the user explicitly asks to switch or set the active workspace
   (e.g. "przełącz mnie na workspace archiwum", "ustaw archiwum jako aktywny", "od teraz pracujmy w archiwum").
9. Do NOT invent attachment IDs, URLs, local paths, workspace names, or other objects — use only what appears in the context.
10. A URL in a plain question does NOT automatically mean knowledge ingestion
    (e.g. "co sądzisz o https://reviews.example?" → workspace.ask, not knowledge.add_sources).
11. When ambiguous, return clarifications instead of guessing.
12. Never claim that an action was already executed.
13. Word order in the user message is NOT execution order — use depends_on_action_numbers for logical ordering.
14. For workspace.create followed by other actions on that workspace, use created_by_action with the exact workspace name
    from the workspace.create action in the same draft.
15. Do not resolve workspace names to workspace_id — use workspace kind name with the exact user text.
16. Do not return candidate_id for source candidates — use candidate_reference_kind name or ordinal only.
17. For knowledge.add_sources, place source declarations directly in the action sources field.
18. Copy each source value exactly from decoded message_text — character for character, without trimming or normalization.
19. Use occurrence only to distinguish repeated exact values in message_text (one-based: 1 = first occurrence).
20. Use one-based action numbers in depends_on_action_numbers and blocks_action_numbers
    (1 = first action in the returned actions sequence).

Example — different workspace targets:
User message:
ten adres https://api.vendor.io wrzuć do workspace numer 1,
a pliki C:\\data\\report.xlsx i C:\\data\\summary.xlsx
dodaj do workspace numer 2
Expected meaning:
- web_url https://api.vendor.io → knowledge.add_sources → workspace ordinal 1
- local_file_reference paths → knowledge.add_sources → workspace ordinal 2
- workspace.activate → MUST NOT appear

Example — shared workspace target:
User message:
dołącz https://portal.vendor.io oraz C:\\backup\\notes.txt do workspace "projekty"
Expected meaning:
- both sources grouped in one knowledge.add_sources → workspace name "projekty"
- workspace.activate → MUST NOT appear
"""

_REPAIR_MESSAGES: dict[str, str] = {
    "draft_contract": (
        "The previous semantic draft did not match the supplied output contract. Return one complete "
        "valid semantic draft using only the available action and source variants."
    ),
    "source_value_not_grounded": (
        "The previous draft contained a source value that was not copied exactly from decoded "
        "message_text. Copy every source value character-for-character from message_text without "
        "trimming or normalization."
    ),
    "source_occurrence_required": (
        "The previous draft used a source value that occurs more than once. Set occurrence to the "
        "one-based occurrence intended by the user."
    ),
    "invalid_action_reference": (
        "The previous draft used an invalid action number. Use only positive one-based action "
        "numbers that exist in the returned actions sequence."
    ),
    "invalid_created_workspace_reference": (
        "The previous draft referenced a created workspace incorrectly. For created_by_action, use "
        "the exact name from one workspace.create action in the same draft."
    ),
    "canonical_request_grounding": (
        "The previous draft compiled but did not satisfy request grounding. Use only attachment IDs, "
        "workspace names, candidate references, and evidence quotes that appear in the supplied context."
    ),
}

_DEFAULT_REPAIR_CATEGORY = "draft_contract"


class RepairCategory(str, Enum):
    draft_contract = "draft_contract"
    source_value_not_grounded = "source_value_not_grounded"
    source_occurrence_required = "source_occurrence_required"
    invalid_action_reference = "invalid_action_reference"
    invalid_created_workspace_reference = "invalid_created_workspace_reference"
    canonical_request_grounding = "canonical_request_grounding"


def repair_message_for_category(category: RepairCategory) -> str:
    return _REPAIR_MESSAGES.get(category.value, _REPAIR_MESSAGES[_DEFAULT_REPAIR_CATEGORY])


def build_safe_planning_context(request: ConversationPlanningRequest) -> dict[str, object]:
    """Build a safe JSON-serializable context dict for the planner prompt."""
    return {
        "message_text": request.message_text,
        "attachments": [
            {
                "attachment_id": attachment.attachment_id,
                "file_name": attachment.file_name,
                "content_type": attachment.content_type,
                "size_bytes": attachment.size_bytes,
            }
            for attachment in request.attachments
        ],
        "available_workspaces": [
            {
                "workspace_id": workspace.workspace_id,
                "name": workspace.name,
                "is_active": workspace.is_active,
            }
            for workspace in request.available_workspaces
        ],
        "active_workspace_id": request.active_workspace_id,
        "available_source_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "source_type": candidate.source_type,
                "available": candidate.available,
            }
            for candidate in request.available_source_candidates
        ],
        "recent_turns": [
            {"role": turn.role, "text": turn.text} for turn in request.recent_turns
        ],
    }


def build_planning_messages(
    request: ConversationPlanningRequest,
    *,
    include_repair_hint: bool = False,
    repair_category: RepairCategory | None = None,
) -> list[ChatMessage]:
    context_json = json.dumps(build_safe_planning_context(request), ensure_ascii=False, indent=2)
    user_content = (
        "Plan the user's intended actions from this safe context:\n\n```json\n"
        f"{context_json}\n```"
    )
    if include_repair_hint:
        category = repair_category or RepairCategory.draft_contract
        repair_text = repair_message_for_category(category)
        user_content = f"{repair_text}\n\n{user_content}"
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


def system_prompt_contains_required_rules() -> bool:
    """Expose for tests verifying prompt contract coverage."""
    required_fragments = (
        "interaction planner",
        "NOT execute",
        "ENTIRE user message",
        "semantic intent",
        "Do not invent technical IDs",
        "depends_on_action_numbers",
        "occurrence",
        "decoded message_text",
        "workspace.activate",
        "clarification",
        "knowledge.add_sources",
        "created_by_action",
    )
    return all(fragment in _SYSTEM_PROMPT for fragment in required_fragments)
