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

from typing import TYPE_CHECKING, List, Optional, Sequence

from pydantic import BaseModel, Field

from legal.domain.legal_workspace_session_snapshot import (
    LegalWorkspaceSessionContract,
    LegalWorkspaceSessionSnapshotV1,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession

if TYPE_CHECKING:
    from legal.domain.legal_agent_state import LegalAgentState


class LegalMemoryContextDefaults:
    """Literal strings surfaced in Tier-2 routing/tool conversation snippets."""

    NO_PRIOR_TURNS_PLACEHOLDER = "(no prior turns in context)"


def resolve_session_prior_workspace_snapshot(
    *,
    session: Optional[ChatSession],
    request: RuntimeRequest,
    policy: LegalMemoryPolicy,
) -> LegalWorkspaceSessionSnapshotV1 | None:
    """
    Load the prior-run workspace snapshot from session metadata when policy allows.

    Used by :class:`~legal.pipeline.legal_dynamic_pipeline.LegalDynamicPipeline`
    before ``run_legal_dynamic_execution_loop``; kept as a pure helper for tests and host reuse.
    """
    if not policy.hydrate_workspace_snapshot_from_session or session is None:
        return None
    if (
        policy.ignore_workspace_snapshot_when_request_has_attachments
        and bool(request.attachments)
    ):
        return None
    return LegalWorkspaceSessionContract.try_load(session.metadata)


class LegalMemoryPolicy(BaseModel):
    """
    Tunable Tier-2 memory behaviour: routing/tool snippets and optional session workspace snapshot.

    Construct directly for full control, or start from a preset (see :class:`LegalMemoryPolicyPresets`)
    and override fields with
    :meth:`~pydantic.BaseModel.model_copy` if needed.

    Session store TTL and ``RuntimeConfig.enable_user_longterm_memory`` remain configured at
    Nexus / host level; this model does not replace those switches.
    """

    persist_workspace_snapshot_to_session: bool = Field(
        default=True,
        description=(
            "After finalize, write LegalWorkspaceSessionSnapshotV1 into session metadata "
            f"({LegalWorkspaceSessionContract.METADATA_KEY}) for the next turn's router. "
            "Hosts may call LegalWorkspaceSessionContract.clear_persisted at session close if policy "
            "requires dropping these hints."
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


class LegalMemoryPolicyPresets:
    """Named :class:`LegalMemoryPolicy` bundles for SKUs and host wiring."""

    @staticmethod
    def default() -> LegalMemoryPolicy:
        """Balanced defaults — same values as :class:`LegalMemoryPolicy` field defaults."""
        return LegalMemoryPolicy()

    @staticmethod
    def minimal_exposure() -> LegalMemoryPolicy:
        """
        Shorter routing/tool history excerpts and no cross-turn workspace snapshot in session metadata.

        Use when hosts want to minimise what Tier-2 stores in ``ChatSession.metadata`` and how much
        prior chat appears in router prompts (follow-up routing may be less informed).
        """
        return LegalMemoryPolicy(
            persist_workspace_snapshot_to_session=False,
            hydrate_workspace_snapshot_from_session=False,
            conversation_tail_message_limit=6,
            conversation_snippet_max_chars_per_message=320,
        )

    @staticmethod
    def strict_legal_workspace() -> LegalMemoryPolicy:
        """
        Tighter snippets but keeps snapshot persist/hydrate for multi-turn contract workflows.

        Intended for careful legal pipelines that still rely on ``session_prior_legal_run`` hints
        between turns without widening conversational context in router prompts.
        """
        return LegalMemoryPolicy(
            persist_workspace_snapshot_to_session=True,
            hydrate_workspace_snapshot_from_session=True,
            ignore_workspace_snapshot_when_request_has_attachments=True,
            conversation_tail_message_limit=8,
            conversation_snippet_max_chars_per_message=400,
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
    return "\n".join(lines) if lines else LegalMemoryContextDefaults.NO_PRIOR_TURNS_PLACEHOLDER


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
    md[LegalWorkspaceSessionContract.METADATA_KEY] = snap.model_dump()
    sess.metadata = md
    await state.context.session_manager.save_session(sess)


# ---------------------------------------------------------------------------
# Module aliases (backward compat — prefer :class:`LegalMemoryPolicyPresets` in new code)
# ---------------------------------------------------------------------------


def default_legal_memory_policy() -> LegalMemoryPolicy:
    return LegalMemoryPolicyPresets.default()


def minimal_exposure_legal_memory_policy() -> LegalMemoryPolicy:
    return LegalMemoryPolicyPresets.minimal_exposure()


def strict_legal_workspace_legal_memory_policy() -> LegalMemoryPolicy:
    return LegalMemoryPolicyPresets.strict_legal_workspace()
