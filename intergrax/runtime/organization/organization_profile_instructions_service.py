# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from intergrax.llm_adapters.base import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.organization.organization_profile import OrganizationProfile
from intergrax.runtime.organization.organization_profile_manager import OrganizationProfileManager


@dataclass
class OrgProfileInstructionsConfig:
    """
    Configuration for organization-level system instructions generation.
    """

    max_chars: int = 1500
    language: str = "en"
    regenerate_if_present: bool = False
    extra: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class OrganizationProfileInstructionsService:
    """
    Service that generates and updates organization-level system_instructions
    using an LLMAdapter, based on OrganizationProfile:

      - identity              (who the organization is),
      - preferences           (how the runtime should behave in this org),
      - domain/knowledge data,
      - memory_entries        (long-term org-level notes).

    Responsibilities:
      - Load OrganizationProfile via OrganizationProfileManager.
      - Build an LLM prompt using identity, preferences, summaries and memory entries.
      - Call LLMAdapter.generate_messages() to obtain a compact, stable
        organization-level system prompt.
      - Persist the result via OrganizationProfileManager.update_system_instructions().
    """

    def __init__(
        self,
        llm: LLMAdapter,
        manager: OrganizationProfileManager,
        config: Optional[OrgProfileInstructionsConfig] = None,
    ) -> None:
        self._llm = llm
        self._manager = manager
        self._config = config or OrgProfileInstructionsConfig()

    async def build_and_save_system_instructions(
        self,
        organization_id: str,
        *,
        force: bool = False,
    ) -> str:
        """
        Generate and persist organization-level system instructions.

        Parameters:
          organization_id:
              Identifier of the organization whose profile should be updated.
          force:
              If True, always regenerate instructions even if they already exist.

        Behavior:
          - If force is False and profile.system_instructions exists and
            config.regenerate_if_present is False -> return existing value.
          - Otherwise:
              1) Load OrganizationProfile.
              2) Build an LLM prompt from identity, preferences, domain/knowledge
                 summaries and memory entries.
              3) Call LLMAdapter.generate_messages() to generate compact instructions.
              4) Truncate to max_chars, strip whitespace.
              5) Save via OrganizationProfileManager.update_system_instructions().
              6) Return the final string.
        """
        profile = await self._manager.get_profile(organization_id)

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

        await self._manager.update_system_instructions(organization_id, instructions)
        return instructions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, profile: OrganizationProfile) -> str:
        """
        Build a text prompt for the LLM from the given organization profile.
        """
        identity = profile.identity
        prefs = profile.preferences

        domain_summary = profile.domain_summary or "(no domain_summary provided)"
        knowledge_summary = profile.knowledge_summary or "(no knowledge_summary provided)"
        knowledge_sources = profile.knowledge_sources or []
        tags = profile.tags or []

        return f"""
You are a system-instructions builder for a conversational AI assistant
working in the context of a specific organization (tenant).

Your TASK:
- Create a compact, stable organization-level system prompt describing
  how the assistant should behave when serving users from this organization.

Constraints:
- Language of the output: {self._config.language}.
- Maximum length: about {self._config.max_chars} characters.
- The instructions will apply across MANY sessions and users within this tenant.
- Focus on:
    * industry, products/services, typical use-cases,
    * tone, language, and formality,
    * policies regarding web search, tools, and data safety,
    * sensitive topics to avoid or handle carefully,
    * what the assistant is allowed or not allowed to do.

ORGANIZATION IDENTITY:
- organization_id: {identity.organization_id}
- name: {identity.name}
- legal_name: {identity.legal_name}
- slug: {identity.slug}
- primary_domain: {identity.primary_domain}
- industry: {identity.industry}
- headquarters_location: {identity.headquarters_location}
- default_timezone: {identity.default_timezone}

ORGANIZATION PREFERENCES:
- default_language: {prefs.default_language}
- default_output_format: {prefs.default_output_format}
- tone_of_voice: {prefs.tone_of_voice}
- allow_web_search: {prefs.allow_web_search}
- allow_tools: {prefs.allow_tools}
- sensitive_topics: {prefs.sensitive_topics}
- hard_constraints: {prefs.hard_constraints}
- soft_guidelines: {prefs.soft_guidelines}

DOMAIN & KNOWLEDGE SUMMARY:
- domain_summary: {domain_summary}
- knowledge_summary: {knowledge_summary}
- knowledge_sources: {knowledge_sources}
- tags: {tags}

OUTPUT FORMAT:
- Return ONLY the final organization-level system prompt text,
  in {self._config.language}.
- Use short paragraphs and bullet points where helpful.
- Do NOT output JSON, XML or any machine-readable markup.
"""

    async def _call_llm(self, prompt_text: str) -> str:
        """
        Delegate generation to the underlying LLMAdapter.

        We build a small ChatMessage list:
          - one system message describing the meta-task,
          - one user message containing the full prompt_text.
        """
        messages: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content="You generate stable organization-level system instructions for an AI assistant.",
            ),
            ChatMessage(
                role="user",
                content=prompt_text,
            ),
        ]

        return await self._llm.generate_messages(
            messages,
            temperature=None,
            max_tokens=None,
        )
