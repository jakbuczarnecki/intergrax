# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Optional

from intergrax.llm.conversational_memory import IntergraxConversationalMemory
from intergrax.llm.messages import ChatMessage


class JsonMemoryStore:
    """
    Simple JSON-based persistence provider for IntergraxConversationalMemory.

    Responsibilities:
      - serialize a memory instance to a single JSON file,
      - load messages from a JSON file into a memory instance.

    Notes:
      - This is a convenience utility; it does not handle concurrency,
        locking, or multi-process coordination.
      - Intended for local development, simple demos, or lightweight
        persistence scenarios.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    # ------------------------------------------------------------------
    # Single JSON file (session_id + messages array)
    # ------------------------------------------------------------------

    def save(self, memory: IntergraxConversationalMemory, *, pretty: bool = True) -> None:
        """
        Save the entire memory history to a single JSON file.

        Format:
        {
          "session_id": "...",
          "messages": [
            { <dataclass ChatMessage as dict> },
            ...
          ]
        }
        """
        messages: List[ChatMessage] = memory.get_all()
        payload = {
            "session_id": memory.session_id,
            "messages": [asdict(m) for m in messages],
        }

        os.makedirs(os.path.dirname(self._file_path) or ".", exist_ok=True)
        with open(self._file_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            else:
                json.dump(payload, f, ensure_ascii=False)

    def load(self, memory: IntergraxConversationalMemory, *, replace: bool = True) -> None:
        """
        Load history from the JSON file into the provided memory instance.

        If replace=True:
          - clears the existing memory and replaces it with loaded messages.
        If replace=False:
          - appends loaded messages to the existing memory.
        """
        if not os.path.exists(self._file_path):
            # Nothing to load – silently ignore
            return

        with open(self._file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_messages = data.get("messages", [])
        messages = [ChatMessage(**m) for m in raw_messages]

        if replace:
            memory.clear()
        memory.extend(messages)

    # ------------------------------------------------------------------
    # Optional: NDJSON helpers (one message per line)
    # ------------------------------------------------------------------

    def append_ndjson(self, memory: IntergraxConversationalMemory) -> None:
        """
        Append all messages from memory to an NDJSON file.

        Each line contains:
        {
          "session_id": "...",
          "role": "...",
          "content": "...",
          "created_at": "...",
          ...any additional ChatMessage fields...
        }
        """
        os.makedirs(os.path.dirname(self._file_path) or ".", exist_ok=True)
        messages = memory.get_all()
        with open(self._file_path, "a", encoding="utf-8") as f:
            for m in messages:
                rec = {
                    "session_id": memory.session_id,
                    **asdict(m),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_from_ndjson(
        self,
        memory: IntergraxConversationalMemory,
        *,
        session_id: Optional[str] = None,
        replace: bool = True,
    ) -> None:
        """
        Load messages from NDJSON file into memory.

        If session_id is provided, only lines matching that session_id
        are loaded. Otherwise, all records are loaded.

        If replace=True:
          - clears existing memory first.
        """
        if not os.path.exists(self._file_path):
            return

        loaded: List[ChatMessage] = []

        with open(self._file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if session_id is not None and rec.get("session_id") != session_id:
                    continue

                # Remove session_id from record before constructing ChatMessage
                rec.pop("session_id", None)
                loaded.append(ChatMessage(**rec))

        if not loaded:
            return

        if replace:
            memory.clear()
        memory.extend(loaded)
