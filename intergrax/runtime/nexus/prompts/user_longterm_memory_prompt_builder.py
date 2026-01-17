# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Optional

from intergrax.llm.messages import ChatMessage, MessageRole
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry


@dataclass
class UserLongTermMemoryPromptBundle:
    """
    Prompt-ready bundle built from retrieved long-term memory entries.

    Design goals:
      - deterministic formatting (no LLM inference here),
      - compact but traceable,
      - safe: contains only retrieved entries, never the full profile.
    """
    context_messages: List[ChatMessage] = field(default_factory=list)


class UserLongTermMemoryPromptBuilder(Protocol):
    """
    Builds prompt messages to inject retrieved user long-term memory
    into the LLM context (similar role as RagPromptBuilder, but for LTM).

    NOTE: This is pure prompt construction only.
    Retrieval / embeddings / ranking live in UserProfileManager.
    """

    def build_user_longterm_memory_prompt(
        self,
        retrieved_entries: List[UserProfileMemoryEntry],
    ) -> UserLongTermMemoryPromptBundle:
        ...


class DefaultUserLongTermMemoryPromptBuilder(UserLongTermMemoryPromptBuilder):
    """
    Default deterministic LTM prompt builder.

    Output strategy:
      - single SYSTEM message containing compact bullet list,
      - includes entry_id and optional session_id for traceability,
      - avoids any inferred claims (just retrieved content).
    """

    def __init__(
        self,
        prompt_registry: YamlPromptRegistry,
        *,
        max_entries: int = 12,
        max_chars: int = 3000,
    ) -> None:
        self._prompt_registry = prompt_registry
        self._max_entries = max_entries
        self._max_chars = max_chars


    def build_user_longterm_memory_prompt(
        self,
        retrieved_entries: List[UserProfileMemoryEntry],
    ) -> UserLongTermMemoryPromptBundle:
        if not retrieved_entries:
            return UserLongTermMemoryPromptBundle(context_messages=[])

        # Filter deleted entries defensively (should already be handled upstream).
        entries = [e for e in retrieved_entries if not e.deleted]
        if not entries:
            return UserLongTermMemoryPromptBundle(context_messages=[])

        # Limit count.
        entries = entries[: self._max_entries]

        lines: List[str] = []
        
        localized = self._prompt_registry.resolve_localized(
            prompt_id="user_longterm_memory"
        )

        lines.append(localized.system)

        # Build bullet list with traceable IDs.
        for e in entries:
            entry_id = (e.entry_id or "").strip()
            session_id = (e.session_id or "").strip() if e.session_id else ""
            kind = e.kind.value
            importance = e.importance.value

            meta_bits: List[str] = []
            if entry_id:
                meta_bits.append(f"id={entry_id}")
            if session_id:
                meta_bits.append(f"session={session_id}")
            if kind:
                meta_bits.append(f"kind={kind}")
            if importance:
                meta_bits.append(f"importance={importance}")

            meta = ", ".join(meta_bits)
            content = (e.content or "").strip()

            # Keep deterministic structure.
            if meta:
                lines.append(f"- [{meta}] {content}")
            else:
                lines.append(f"- {content}")

        text = "\n".join(lines).strip()

        # Hard char limit (deterministic truncation).
        if self._max_chars and len(text) > self._max_chars:
            text = text[: self._max_chars].rstrip() + "\n[...truncated]"

        msg = ChatMessage(role="user", content=text)
        return UserLongTermMemoryPromptBundle(context_messages=[msg])