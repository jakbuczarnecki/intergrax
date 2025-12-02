# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import threading
import uuid
from typing import List, Optional, Sequence

from intergrax.llm.messages import ChatMessage, MessageRole


class ConversationalMemory:
    """
    Universal in-memory conversation history component.

    Responsibilities:
      - keep messages in RAM,
      - provide a simple API to add / extend / read / clear messages,
      - optionally enforce a max_messages limit,
      - prepare messages for different model backends (get_for_model).

    Important:
      - this class does NOT know anything about persistence
        (no files, no SQLite, no external storage).
      - persistence is delegated to dedicated memory store providers.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        max_messages: Optional[int] = None,
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self._messages: List[ChatMessage] = []
        self._max_messages = max_messages
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Core mutation API
    # ------------------------------------------------------------------

    def add_message(self, role: MessageRole, content: str) -> None:
        """
        Append a new message to the history and enforce max_messages limit.
        """
        self.add_message_item(ChatMessage(role=role, content=str(content)))        

    def add_message_item(self, message: ChatMessage) -> None:
        """
        Append a new message to the history and enforce max_messages limit.
        """
        with self._lock:
            self._messages.append(message)
            self._trim_if_needed()

    def extend(self, messages: Sequence[ChatMessage]) -> None:
        """
        Append multiple messages (e.g., loaded from a store).
        """
        with self._lock:
            self._messages.extend(messages)
            self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        """
        If max_messages is set, keep only the most recent messages.
        """
        if self._max_messages is not None and len(self._messages) > self._max_messages:
            overflow = len(self._messages) - self._max_messages
            if overflow > 0:
                del self._messages[0:overflow]

    def clear(self) -> None:
        """
        Remove all messages from memory.
        """
        with self._lock:
            self._messages.clear()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_all(self) -> List[ChatMessage]:
        """
        Return a shallow copy of all messages.
        """
        with self._lock:
            return list(self._messages)

    def get_for_model(self, native_tools: bool = False) -> List[ChatMessage]:
        """
        Returns messages prepared for sending to the model.

        If native_tools=True (e.g., OpenAI Responses with tool calling),
        removes 'tool' messages from history and keeps only
        system/user/assistant messages, so that the model sees a clean
        conversation history. Tools will be represented by the new
        assistant.tool_calls + tool messages, not by old historical tool logs.

        For planners (e.g., Ollama) or generic models, set native_tools=False
        to return the full history.
        """
        with self._lock:
            if native_tools:
                return [
                    m
                    for m in self._messages
                    if m.role in ("system", "user", "assistant")
                ]
            return list(self._messages)

    def get_recent(self, n: int) -> List[ChatMessage]:
        """
        Return the last `n` messages (or fewer, if history is shorter).
        """
        if n <= 0:
            return []
        with self._lock:
            return self._messages[-n:]
