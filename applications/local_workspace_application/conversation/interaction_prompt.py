# © Artur Czarnecki. All rights reserved.

"""System prompt and safe planning context for the conversational interaction planner."""

from __future__ import annotations

import json

from intergrax.llm.messages import ChatMessage

from local_workspace_application.conversation.interaction_models import ConversationPlanningRequest

_SYSTEM_PROMPT = """You are an interaction planner for a knowledge workspace product. You do NOT execute actions.

Rules:
1. Return ONLY a structured plan matching the provided schema.
2. Analyze the ENTIRE user message, not just the beginning.
3. The user may write in any language, with typos.
4. One message may contain multiple actions and multiple source types (files, URLs, local paths, workspace targets).
5. Determine the direction of every action explicitly.
6. Every workspace-dependent action must include an explicit WorkspaceReference.
7. A workspace used as the TARGET of an operation does NOT change the active workspace.
8. Return workspace.activate ONLY when the user explicitly asks to switch or set the active workspace
   (e.g. "przełącz mnie na workspace magazyn", "ustaw magazyn jako aktywny", "od teraz pracujmy w magazynie").
9. Do NOT invent attachment IDs, URLs, local paths, or other objects — use only what appears in the context.
10. A URL in a plain question does NOT automatically mean knowledge ingestion
    (e.g. "co sądzisz o https://example.com?" → workspace.ask, not knowledge.add_web_urls).
11. When ambiguous, return clarifications instead of guessing.
12. Never claim that an action was already executed.
13. Word order in the user message is NOT execution order — use depends_on for logical ordering.
14. For multiple actions, produce one aggregated plan with response_mode "aggregate".
15. For workspace.create followed by other actions on that workspace, use created_by_action references
    with depends_on pointing to the create action.
16. Do not resolve workspace names to workspace_id — use WorkspaceReference(kind="name", value=<user text>).
17. Do not return candidate_id for source candidates — use candidate_reference_kind name or ordinal only.

Example (Polish):
User message:
dołącz informacje o cennikach ze strony https://www.cenniki.pl
oraz dorzuć moją kopię lokalną cenników z
c:\\moje dokumenty\\cenniki.xls
a to wszystko do workspace "magazyn"

Expected plan meaning:
- knowledge.add_web_urls → URL https://www.cenniki.pl → target workspace name "magazyn"
- knowledge.add_local_references → c:\\moje dokumenty\\cenniki.xls → target workspace name "magazyn"
- workspace.activate → MUST NOT appear
Both intake actions share the same workspace target and need not depend on each other.
"""

_REPAIR_USER_MESSAGE = (
    "Your previous response was invalid or did not match the required schema and contract. "
    "Return a corrected plan that strictly follows the schema and all planning rules."
)


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
) -> list[ChatMessage]:
    context_json = json.dumps(build_safe_planning_context(request), ensure_ascii=False, indent=2)
    user_content = (
        "Plan the user's intended actions from this safe context:\n\n```json\n"
        f"{context_json}\n```"
    )
    if include_repair_hint:
        user_content = f"{_REPAIR_USER_MESSAGE}\n\n{user_content}"
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
        "multiple actions",
        "TARGET",
        "active workspace",
        "workspace.activate",
        "Do NOT invent",
        "clarification",
        "cenniki.pl",
        "magazyn",
    )
    return all(fragment in _SYSTEM_PROMPT for fragment in required_fragments)
