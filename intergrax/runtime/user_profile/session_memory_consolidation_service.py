# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.base import LLMAdapter
from intergrax.llm.messages import ChatMessage, MessageRole
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import (
    UserProfileMemoryEntry,
    MemoryKind,
    MemoryImportance,
)
from intergrax.runtime.user_profile.user_profile_instructions_service import UserProfileInstructionsService


@dataclass
class SessionMemoryConsolidationConfig:
    """
    Configuration for consolidating a single chat session into long-term
    user profile memory entries and optionally refreshing user-level
    system instructions.
    """

    # Target language for extracted content (facts, preferences, summaries).
    # If you introduce GLOBAL_SETTINGS, you can set the default from there.
    language: str = GLOBAL_SETTINGS.default_language

    # Maximum number of extracted USER_FACT entries to persist.
    max_facts: int = 8

    # Maximum number of extracted PREFERENCE entries to persist.
    max_preferences: int = 6

    # Whether to create a single SESSION_SUMMARY entry.
    include_session_summary: bool = True

    # Default importance used when the model does not provide one or it is invalid.
    default_fact_importance: MemoryImportance = MemoryImportance.MEDIUM
    default_preference_importance: MemoryImportance = MemoryImportance.MEDIUM
    default_summary_importance: MemoryImportance = MemoryImportance.MEDIUM

    # Maximum number of messages from the session to feed into the prompt.
    max_messages_in_prompt: int = 80

    # Soft limit for the total concatenated characters of the conversation
    # passed to the model. Older messages will be trimmed if necessary.
    max_conversation_chars: int = 6000

    # Temperature for the LLM call that extracts memory.
    temperature: Optional[float] = None

    # Whether this service should trigger regeneration of user-level
    # system instructions after storing new memory entries.
    regenerate_system_instructions: bool = True

    # If True, system instructions regeneration will be forced even if they
    # already exist and the instructions-service config would normally skip it.
    force_regenerate_system_instructions: bool = False

    # Which message roles from the conversation should be considered when
    # building the consolidation prompt. By default we focus on user/assistant
    # turns and skip system/tool messages.
    included_roles: Sequence[MessageRole] = ("user", "assistant")

    # Optional extra knobs (e.g. style hints, domain hints).
    extra: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}

            

class SessionMemoryConsolidationService:
    """
    Service responsible for converting a single chat session history
    into structured long-term memory entries for the user profile.

    High-level responsibilities:
      - Take the session conversation (ChatMessage sequence).
      - Ask the LLM to extract:
          * USER_FACT items (stable facts / goals),
          * PREFERENCE items (communication / workflow preferences),
          * optional SESSION_SUMMARY item (short global summary).
      - Map the extracted data into UserProfileMemoryEntry objects.
      - Persist them through UserProfileManager (add_memory_entry).
    """

    def __init__(
        self,
        llm: LLMAdapter,
        profile_manager: UserProfileManager,
        instructions_service: UserProfileInstructionsService,
        config: Optional[SessionMemoryConsolidationConfig] = None,
    ) -> None:
        self._llm = llm
        self._profile_manager = profile_manager
        self._instructions_service = instructions_service
        self._config = config or SessionMemoryConsolidationConfig()


    async def consolidate_session(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[ChatMessage],
    ) -> List[UserProfileMemoryEntry]:
        """
        Extract long-term memory from a single session, store it in the
        user's profile, and optionally refresh user-level system instructions.

        Parameters:
            user_id:
                Identifier of the user whose profile memory will be updated.
            session_id:
                Identifier of the session (used to tag memory entries).
            messages:
                Full or partial conversation history for this session.

        Returns:
            The list of UserProfileMemoryEntry objects that were created and stored.
            Note: system instructions regeneration is a side-effect and its
            string result (if any) is not returned here.
        """
        trimmed = self._prepare_conversation_for_prompt(messages)
        if not trimmed:
            return []

        prompt_text = self._build_prompt(
            trimmed,
            session_id=session_id,
        )

        llm_output = self._call_llm(prompt_text)
        parsed = self._parse_llm_output(llm_output)

        if parsed is None:
            # If parsing failed, we do not create any entries and we do not
            # touch system instructions.
            return []

        entries = self._build_memory_entries_from_parsed(
            user_id=user_id,
            session_id=session_id,
            parsed=parsed,
        )

        # Persist entries via UserProfileManager.
        stored_entries: List[UserProfileMemoryEntry] = []
        for entry in entries:
            # The manager is responsible for assigning entry_id / timestamps if needed.
            stored = await self._profile_manager.add_memory_entry(user_id, entry)
            stored_entries.append(stored)

        # Optionally refresh user-level system instructions as part of the
        # same pipeline, but only if something was actually stored.
        if (
            stored_entries
            and self._config.regenerate_system_instructions
        ):
            await self._instructions_service.build_and_save_system_instructions(
                user_id=user_id,
                force=self._config.force_regenerate_system_instructions,
            )

        return stored_entries

    # -------------------------------------------------------------------------
    # Internal: building the conversation snippet
    # -------------------------------------------------------------------------

    def _prepare_conversation_for_prompt(
        self,
        messages: Sequence[ChatMessage],
    ) -> List[ChatMessage]:
        """
        Take the raw session messages and trim them so that:
          - only messages with roles listed in config.included_roles
            are considered,
          - at most max_messages_in_prompt are included,
          - the total concatenated content length does not exceed
            max_conversation_chars (oldest messages are removed first).
        """
        if not messages:
            return []

        # Filter by allowed roles first.
        filtered: List[ChatMessage] = []
        for msg in messages:            
            if msg.role in self._config.included_roles:
                filtered.append(msg)

        if not filtered:
            return []

        # Use only the last N messages by default.
        if len(filtered) > self._config.max_messages_in_prompt:
            filtered = filtered[-self._config.max_messages_in_prompt :]

        # Further trim by character budget.
        total_chars = 0
        trimmed: List[ChatMessage] = []

        # Iterate from the end (most recent) backwards, accumulate until budget.
        for msg in reversed(filtered):
            content = msg.content or ""
            length = len(content)
            if total_chars + length > self._config.max_conversation_chars:
                break
            trimmed.append(msg)
            total_chars += length

        # Reverse again so that they are in chronological order.
        trimmed.reverse()
        return trimmed


    # -------------------------------------------------------------------------
    # Internal: prompt building and LLM call
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        messages: Sequence[ChatMessage],
        *,
        session_id: str,
    ) -> str:
        """
        Build a plain-text representation of the conversation and a task
        description for the LLM. The model will be asked to output a JSON
        structure describing facts, preferences and a session summary.

        The goal of this prompt is to extract ONLY long-term, re-usable
        information that will improve future conversations with this user,
        and to avoid hallucinations or speculative content.
        """
        conversation_lines: List[str] = []
        for m in messages:
            content = (m.content or "").strip()
            if not content:
                continue
            # We keep the role label mainly for context; the actual filtering
            # by included_roles happens earlier in _prepare_conversation_for_prompt.
            conversation_lines.append(f"{m.role}: {content}")

        conversation_block = "\n".join(conversation_lines) or "(empty session)"

        return f"""
You are an AI specializing in extracting long-term user profile memory
from a single chat session.

Your task:
- Read the conversation below.
- Decide which pieces of information are:
    * USER_FACT  - stable facts or long-term goals about the user
                   (e.g. background, skills, health constraints, recurring goals).
    * PREFERENCE - stable or recurring preferences about communication,
                   workflow, tools, style, or constraints (e.g. "no emojis in code").
    * SESSION_SUMMARY - a short global summary of what happened in this
                   specific session, useful as context for future work.

Important distinctions:
- USER_FACT:
    - Should be true regardless of a single session.
    - Describes who the user is, what they does, what they cares about,
      which constraints and long-term goals they have.
    - Example: "The user is a senior software engineer working with Python and .NET."
- PREFERENCE:
    - Describes how the user wants the assistant to respond or work.
    - Example: "The user prefers very technical, concise answers in English,
      without emojis and with code examples."
- SESSION_SUMMARY:
    - High-level recap of the current session only.
    - Should mention the main problem(s), decisions, and next steps.
    - Avoid very fine-grained step-by-step details.

Rules:
- Focus ONLY on information that will be useful across many future sessions.
- Ignore transient, local details that are unlikely to matter later
  (e.g. one-off examples, small talk, debug attempts that will not repeat).
- Ignore meta-information about the model itself (e.g. that you are an AI).
- Do NOT invent facts that are not clearly supported by the conversation.
- If you are not sure whether something is true or stable, either:
    * skip it, or
    * mark it with LOWER importance (LOW).
- Avoid time-sensitive language such as "today", "recently", "this week".
- Use the target language: {self._config.language}.
- Do NOT include sensitive data that should not be stored long-term
  (e.g. passwords, tokens, very private health details).
- Keep the content concise but precise.

Output format:
Return a single JSON object with the following structure:

{{
  "facts": [
    {{
      "title": "short label",
      "content": "concise description of the fact or goal",
      "importance": "LOW | MEDIUM | HIGH | CRITICAL",
      "tags": ["user", "goal"]
    }}
  ],
  "preferences": [
    {{
      "title": "short label",
      "content": "concise description of the preference",
      "importance": "LOW | MEDIUM | HIGH | CRITICAL",
      "tags": ["communication", "tone"]
    }}
  ],
  "session_summary": {{
    "title": "global summary",
    "content": "short paragraph describing the session",
    "importance": "LOW | MEDIUM | HIGH | CRITICAL",
    "tags": ["session_summary"]
  }}
}}

Constraints:
- Return ONLY valid JSON (no comments, no trailing commas, no markdown).
- If you do not want to provide a session summary, set "session_summary": null.
- Respect the importance scale: use HIGH or CRITICAL only for items that
  are clearly central for future collaboration and appear strongly in the
  conversation.

Session identifier (for your reasoning only, do not repeat it verbatim):
- session_id: {session_id}

Conversation:
{conversation_block}
"""


    def _call_llm(self, prompt_text: str) -> str:
        """
        Call the underlying LLMAdapter.generate_messages with a simple
        system+user prompt.
        """
        messages: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content=(
                    "You extract structured long-term memory from a single "
                    "chat session and output pure JSON."
                ),
            ),
            ChatMessage(
                role="user",
                content=prompt_text,
            ),
        ]

        return  self._llm.generate_messages(
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=None,
        )

    # -------------------------------------------------------------------------
    # Internal: parsing and mapping
    # -------------------------------------------------------------------------

    def _parse_llm_output(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse the LLM output as JSON. If parsing fails, attempt a
        simple brace-based extraction. If everything fails, return None.
        """
        if not text:
            return None

        text = text.strip()

        # First attempt: direct JSON parse.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Second attempt: try to extract the first JSON object in the text.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None

    def _build_memory_entries_from_parsed(
        self,
        user_id: str,  # Currently unused, but kept for future extensions.
        session_id: str,
        parsed: Dict[str, Any],
    ) -> List[UserProfileMemoryEntry]:
        """
        Map the parsed JSON structure into UserProfileMemoryEntry instances,
        applying config constraints (max items, importance defaults, etc.).
        """
        entries: List[UserProfileMemoryEntry] = []

        facts = parsed.get("facts") or []
        preferences = parsed.get("preferences") or []
        summary = parsed.get("session_summary")

        # Facts
        for item in facts[: self._config.max_facts]:
            entry = self._create_entry_from_item(
                item=item,
                session_id=session_id,
                expected_kind=MemoryKind.USER_FACT,
                default_importance=self._config.default_fact_importance,
            )
            if entry is not None:
                entries.append(entry)

        # Preferences
        for item in preferences[: self._config.max_preferences]:
            entry = self._create_entry_from_item(
                item=item,
                session_id=session_id,
                expected_kind=MemoryKind.PREFERENCE,
                default_importance=self._config.default_preference_importance,
            )
            if entry is not None:
                entries.append(entry)

        # Session summary
        if self._config.include_session_summary and summary:
            summary_entry = self._create_entry_from_item(
                item=summary,
                session_id=session_id,
                expected_kind=MemoryKind.SESSION_SUMMARY,
                default_importance=self._config.default_summary_importance,
            )
            if summary_entry is not None:
                entries.append(summary_entry)

        return entries

    def _create_entry_from_item(
        self,
        item: Dict[str, Any],
        session_id: str,
        expected_kind: MemoryKind,
        default_importance: MemoryImportance,
    ) -> Optional[UserProfileMemoryEntry]:
        """
        Create a single UserProfileMemoryEntry from a dict item produced by
        the LLM. If mandatory fields are missing, returns None.
        """
        if not isinstance(item, dict):
            return None

        content = (item.get("content") or "").strip()
        if not content:
            return None

        title = (item.get("title") or "").strip() or None
        importance_str = (item.get("importance") or "").strip().upper()
        tags = item.get("tags") or []

        # Map importance string to enum, with a safe fallback.
        importance = self._map_importance(importance_str, default_importance)

        metadata: Dict[str, Any] = {
            "tags": tags,
            "source": "session_consolidation",
        }

        return UserProfileMemoryEntry(
            content=content,
            session_id=session_id,
            kind=expected_kind,
            title=title,
            importance=importance,
            metadata=metadata,
            deleted=False,
            modified=False,
        )

    def _map_importance(
        self,
        value: str,
        default: MemoryImportance,
    ) -> MemoryImportance:
        """
        Convert the provided importance string into MemoryImportance enum.
        If conversion fails, return the default.
        """
        if not value:
            return default

        upper = value.upper()
        for level in MemoryImportance:
            if level.name == upper:
                return level

        return default
