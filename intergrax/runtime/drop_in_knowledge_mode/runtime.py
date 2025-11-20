# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Drop-In Knowledge Mode runtime - MVP skeleton

Provides a minimal DropInKnowledgeRuntime with async ask() implementing
the Simple Mode flow: store incoming message, build a tiny context (last
messages), call the LLM adapter and return the answer. This file contains
an in-memory session store used by the MVP tests.
"""
from __future__ import annotations

from typing import Optional, List

from intergrax.llm_adapters.base import LLMAdapter
from intergrax.llm.conversational_memory import IntergraxConversationalMemory, ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig


class InMemorySessionStore:
    """Very small session store for MVP: keeps IntergraxConversationalMemory per session."""

    def __init__(self):
        self._store: dict[str, IntergraxConversationalMemory] = {}

    def get_or_create(self, session_id: str) -> IntergraxConversationalMemory:
        if session_id not in self._store:
            self._store[session_id] = IntergraxConversationalMemory(session_id=session_id)
        return self._store[session_id]


class DropInKnowledgeRuntime:
    """Minimal Drop-In Knowledge Runtime (Simple Mode only).

    This MVP focuses on developer ergonomics: it exposes `ask()` which
    app code can call. Internally it stores messages in a session store
    and delegates to the configured LLM adapter.
    """

    def __init__(self, config: RuntimeConfig, session_store: Optional[InMemorySessionStore] = None):
        self.config = config
        self.llm: LLMAdapter = config.llm_adapter
        self.session_store = session_store or InMemorySessionStore()

    async def ask(self, user_id: str, session_id: str, message: str) -> dict:
        """Simple Mode: append message to session, call LLM with recent history.

        Returns a dict with keys: answer (str), sources (list), metadata (dict).
        """
        if not user_id:
            raise ValueError("user_id is required")
        if not session_id:
            raise ValueError("session_id is required")

        mem = self.session_store.get_or_create(session_id)
        # store incoming user message
        mem.add_message("user", message)

        # build tiny context: last N messages
        recent = mem.get_recent(self.config.max_history_messages or 10)

        # call LLM adapter (synchronous call wrapped here; adapters may be blocking)
        try:
            # LLMAdapter.generate_messages expects Sequence[ChatMessage]
            answer = self.llm.generate_messages(recent)
        except Exception as e:
            # return structured error info; runtime could implement retries/fallback
            return {"answer": "", "sources": [], "metadata": {"error": str(e)}}

        # store assistant message
        mem.add_message("assistant", answer)

        return {"answer": answer, "sources": [], "metadata": {"session_id": session_id}}
