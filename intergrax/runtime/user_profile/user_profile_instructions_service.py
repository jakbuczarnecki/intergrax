# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.base import LLMAdapter
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import (
    UserProfile,
    UserProfileMemoryEntry,
)
from intergrax.llm.messages import ChatMessage


@dataclass
class UserProfileInstructionsConfig:
    """
    Configuration for user-level system instructions generation.
    """

    # Maximum length of the final system instructions string.
    max_chars: int = 1200

    # Target language of the generated instructions, e.g. "pl" or "en".
    language: str = GLOBAL_SETTINGS.default_language

    # If False and profile.system_instructions already exists,
    # the service may simply return the existing value instead of regenerating.
    regenerate_if_present: bool = False

    # Optional extra knobs (for future use).
    # Example: {"style": "technical", "allow_bullets": True}
    extra: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class UserProfileInstructionsService:
    """
    High-level service that generates and updates user-level system_instructions
    using an LLMAdapter, based on UserProfile:

      - identity          (who the user is),
      - preferences       (how the user wants the system to behave),
      - memory_entries    (long-term facts and notes about the user).

    Responsibilities:
      - Load UserProfile via UserProfileManager.
      - Build an LLM prompt using identity, preferences and memory entries.
      - Call LLMAdapter.generate_messages() to obtain a compact, stable
        system prompt.
      - Persist the result via UserProfileManager.update_system_instructions().
    """

    def __init__(
        self,
        llm: LLMAdapter,
        manager: UserProfileManager,
        config: Optional[UserProfileInstructionsConfig] = None,
    ) -> None:
        self._llm = llm
        self._manager = manager
        self._config = config or UserProfileInstructionsConfig()

    async def build_and_save_system_instructions(
        self,
        user_id: str,
        *,
        force: bool = False,
    ) -> str:
        """
        Generate and persist user-level system instructions.

        Parameters:
          user_id:
              Identifier of the user whose profile should be updated.
          force:
              If True, always regenerate instructions even if they already exist.

        Behavior:
          - If force is False and profile.system_instructions exists and
            config.regenerate_if_present is False -> return existing value.
          - Otherwise:
              1) Load UserProfile.
              2) Build an LLM prompt from identity, preferences and memory entries.
              3) Call LLMAdapter.generate_messages() to generate compact instructions.
              4) Truncate to max_chars, strip whitespace.
              5) Save via UserProfileManager.update_system_instructions().
              6) Return the final string.
        """
        profile = await self._manager.get_profile(user_id)

        if (
            profile.system_instructions
            and not force
            and not self._config.regenerate_if_present
        ):
            return profile.system_instructions

        prompt_text = self._build_prompt(profile)
        raw_instructions = await self._call_llm(prompt_text)

        instructions = raw_instructions.strip()
        if len(instructions) > self._config.max_chars:
            instructions = instructions[: self._config.max_chars].rstrip()

        await self._manager.update_system_instructions(user_id, instructions)
        return instructions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, profile: UserProfile) -> str:
        """
        Build a single text prompt for the LLM from the given user profile.

        We intentionally pass:
          - a compact view of identity,
          - a compact view of preferences,
          - a joined view of long-term memory entries.

        The LLM is responsible for synthesizing them into a single,
        stable system prompt for this user.
        """
        identity = profile.identity
        prefs = profile.preferences
        memory_entries: List[UserProfileMemoryEntry] = profile.memory_entries

        # Compact representation of long-term memory entries.
        memory_lines: List[str] = []
        for entry in memory_entries:
            # Ignore entries marked as deleted; they should not influence
            # new instructions.
            if entry.deleted:
                continue
            memory_lines.append(f"- {entry.content}")
        memory_block = (
            "\n".join(memory_lines)
            if memory_lines
            else "(no long-term memory entries yet)"
        )

        return f"""
You are a system-instructions builder for a conversational AI assistant.

Your TASK:
- Create a compact, stable system prompt that describes this user
  and how the AI assistant should behave when talking to them.
- The instructions will be used as a SYSTEM message, prepended to EVERY conversation.
- Focus ONLY on:
    * who the user is,
    * how they prefer to communicate,
    * what their long-term goals and constraints are,
    * which facts the assistant MUST remember and respect.

Constraints:
- Language of the output: {self._config.language}.
- Maximum length: about {self._config.max_chars} characters.
- Avoid time-sensitive statements (e.g. "recently", "this week", "currently").
- Do NOT mention that you are generating instructions.
- Write ABOUT the user in third person (e.g. "The user prefers..."),
  and give direct rules for the assistant (e.g. "Always respond...").

USER IDENTITY (high-level, do not copy verbatim if too detailed):
- user_id: {identity.user_id}
- display_name: {identity.display_name}
- role: {identity.role}
- domain_expertise: {identity.domain_expertise}
- language: {identity.language}
- locale: {identity.locale}
- timezone: {identity.timezone}

USER PREFERENCES:
- preferred_language: {prefs.preferred_language}
- answer_length: {prefs.answer_length}
- tone: {prefs.tone}
- no_emojis_in_code: {prefs.no_emojis_in_code}
- no_emojis_in_docs: {prefs.no_emojis_in_docs}
- prefer_markdown: {prefs.prefer_markdown}
- prefer_code_blocks: {prefs.prefer_code_blocks}
- default_project_context: {prefs.default_project_context}

LONG-TERM USER MEMORY ENTRIES (facts, goals, stable notes):
{memory_block}

OUTPUT FORMAT:
- Return ONLY the final system prompt text, in {self._config.language}.
- Use short paragraphs and/or bullet points.
- Do NOT output JSON, XML or any machine-readable markup.
"""

    async def _call_llm(self, prompt_text: str) -> str:
        """
        Delegate generation to the underlying LLMAdapter.

        We build a small ChatMessage list:
          - one system instruction describing the meta-task,
          - one user message containing the full prompt_text.
        """
        messages: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content="You generate stable, compact user-level system instructions for an AI assistant.",
            ),
            ChatMessage(
                role="user",
                content=prompt_text,
            ),
        ]

        # LLMAdapter.generate_messages() returns a plain string.
        return await self._llm.generate_messages(
            messages,
            temperature=0.2,
            max_tokens=None,
        )
