# © Artur Czarnecki. All rights reserved.

"""System prompt and safe planning context for the conversational interaction planner."""

from __future__ import annotations

import json

from intergrax.llm.messages import ChatMessage

from local_workspace_application.conversation.interaction_models import ConversationPlanningRequest

_SYSTEM_PROMPT = """You are an interaction planner for a knowledge workspace product. You do NOT execute actions.

Return plan_version "2" with separate objects and actions.

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
9. Do NOT invent attachment IDs, URLs, local paths, workspace names, or other objects — use only what appears in the context.
10. A URL in a plain question does NOT automatically mean knowledge ingestion
    (e.g. "co sądzisz o https://example.com?" → workspace.ask, not knowledge.add_sources).
11. When ambiguous, return clarifications instead of guessing.
12. Never claim that an action was already executed.
13. Word order in the user message is NOT execution order — use depends_on for logical ordering.
14. For multiple actions, produce one aggregated plan with response_mode "aggregate".
15. For workspace.create followed by other actions on that workspace, use created_by_action references
    with depends_on pointing to the create action.
16. Do not resolve workspace names to workspace_id — use WorkspaceReference(kind="name", value=<user text>).
17. Do not return candidate_id for source candidates — use candidate_reference_kind name or ordinal only.

Object extraction (plan v2):
18. Put URLs and local file/folder references ONLY in objects — never directly in actions.
19. Each extracted object needs a stable object_id within the plan, object_type, value, and evidence span.
20. evidence.source must be "message_text".
21. evidence.start is zero-based; evidence.end is exclusive.
22. Compute offsets against the decoded message_text from context, not JSON-escaped representation.
23. evidence.text must exactly equal message_text[start:end] and object.value must exactly equal evidence.text.
24. Do not normalize, trim, lowercase, or rewrite extracted values.
25. Route objects to actions via knowledge.add_sources with source_object_ids.
26. Group objects that share the same target workspace into one knowledge.add_sources action when appropriate.
27. Create separate knowledge.add_sources actions for different workspace targets.
28. Do not emit unused objects — every object must be referenced by at least one knowledge.add_sources action.

Example (Polish) — different workspace targets:
User message:
ten adres https://cenniki.pl wrzuć do workspace numer 1,
a pliki C:\\cenniki\\hurt.xlsx i C:\\cenniki\\detal.xlsx
dodaj do workspace numer 2

Expected plan meaning:
- web_url https://cenniki.pl → knowledge.add_sources → workspace ordinal 1
- local_file_reference C:\\cenniki\\hurt.xlsx → knowledge.add_sources → workspace ordinal 2
- local_file_reference C:\\cenniki\\detal.xlsx → same knowledge.add_sources as hurt.xlsx → workspace ordinal 2
- workspace.activate → MUST NOT appear

Example (Polish) — shared workspace target:
User message:
dołącz informacje o cennikach ze strony https://www.cenniki.pl
oraz dorzuć moją kopię lokalną cenników z
c:\\moje dokumenty\\cenniki.xls
a to wszystko do workspace "magazyn"

Expected plan meaning:
- web_url and local_file_reference objects with exact evidence spans
- knowledge.add_sources grouping both objects → workspace name "magazyn"
- workspace.activate → MUST NOT appear
"""

_REPAIR_USER_MESSAGE = (
    "Your previous response was invalid or did not match the required schema and contract. "
    "Return a corrected plan_version 2 plan that strictly follows the schema and all planning rules."
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
        "objects",
        "source_object_ids",
        "evidence",
        "zero-based",
        "end is exclusive",
        "decoded message_text",
        "different workspace targets",
        "workspace.activate",
        "Do NOT invent",
        "clarification",
        "cenniki.pl",
        "magazyn",
    )
    return all(fragment in _SYSTEM_PROMPT for fragment in required_fragments)
