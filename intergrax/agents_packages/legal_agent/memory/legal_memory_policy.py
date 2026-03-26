# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Tier-2 conversation memory policy for Legal Agent.

Defines how much of the session transcript is surfaced to LLM calls that only need a short
multi-turn hint (pipeline routing, tool-intent decision, replanner). Full chat persistence and
long-term memory remain governed by Nexus session storage and RuntimeConfig; this module is the
product-level contract for those trimmed snippets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.domain.legal_workspace_session_snapshot import (
    LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

if TYPE_CHECKING:
    from intergrax.agents_packages.legal_agent.domain.legal_agent_state import LegalAgentState

NO_PRIOR_TURNS_PLACEHOLDER = "(no prior turns in context)"


class LegalMemoryPolicy(BaseModel):
    """
    Tunable limits for legal-tier use of recent conversation in routing / tool-plan LLM prompts.

    Session store retention and ``RuntimeConfig.enable_user_longterm_memory`` are orthogonal;
    adjust those at deployment; use this model for per-SKU control of snippet shape.
    """

    persist_workspace_snapshot_to_session: bool = Field(
        default=True,
        description=(
            "After finalize, write LegalWorkspaceSessionSnapshotV1 into session metadata "
            f"({LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY}) for the next turn's router."
        ),
    )
    hydrate_workspace_snapshot_from_session: bool = Field(
        default=True,
        description=(
            "When LegalDynamicPipeline starts, load snapshot from session metadata into agent state "
            "for workspace metrics (cross-turn routing hints)."
        ),
    )
    ignore_workspace_snapshot_when_request_has_attachments: bool = Field(
        default=True,
        description=(
            "If the current request carries new attachments, drop the hydrated snapshot for this run "
            "(avoid routing on stale counts from a prior document)."
        ),
    )

    conversation_tail_message_limit: int = Field(
        default=12,
        ge=1,
        le=256,
        description="Max recent chat messages included in routing/tool-decision conversation snippets.",
    )
    conversation_snippet_max_chars_per_message: int = Field(
        default=500,
        ge=1,
        le=32_768,
        description="Per-message cap when formatting snippets (limits contract prose in router context).",
    )


def build_legal_conversation_snippet(state: RuntimeState, *, policy: LegalMemoryPolicy) -> str:
    """
    Build a newline-separated ``role: excerpt`` block from ``state`` for legal routing/tool prompts.

    Uses ``messages_for_llm`` when set, otherwise ``built_history_messages`` (same precedence as
    the previous inline helper).
    """
    lines: List[str] = []
    msgs: Sequence[ChatMessage] = state.messages_for_llm or state.built_history_messages or []
    tail_n = policy.conversation_tail_message_limit
    cap = policy.conversation_snippet_max_chars_per_message
    tail = list(msgs)[-tail_n:]
    for m in tail:
        role: str = m.role
        text: str = m.content
        excerpt = text if len(text) <= cap else text[:cap]
        lines.append(f"{role}: {excerpt}")
    return "\n".join(lines) if lines else NO_PRIOR_TURNS_PLACEHOLDER


async def persist_legal_workspace_session_snapshot(
    *,
    state: RuntimeState,
    agent_state: LegalAgentState,
    policy: LegalMemoryPolicy,
) -> None:
    """Persist workspace snapshot into session metadata when policy and session allow."""
    if not policy.persist_workspace_snapshot_to_session:
        return
    sess = state.session
    if sess is None:
        return
    snap = agent_state.to_workspace_session_snapshot_v1()
    md = dict(sess.metadata or {})
    md[LEGAL_WORKSPACE_SESSION_SNAPSHOT_METADATA_KEY] = snap.model_dump()
    sess.metadata = md
    await state.context.session_manager.save_session(sess)
