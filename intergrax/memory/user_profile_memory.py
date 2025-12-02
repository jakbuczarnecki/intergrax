# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core domain models for user / org profile and prompt bundles.
# These models are intentionally independent from any storage or engine logic.
# They represent the "language" in which we describe identities, preferences
# and how they should be injected into LLM prompts.
# ---------------------------------------------------------------------------


@dataclass
class UserIdentity:
    """
    High-level description of who the user is.

    This is a domain model, not something that must be sent directly to the LLM.
    It can be summarized and transformed into instructions when needed.
    """

    user_id: str

    # Human-level description
    display_name: Optional[str] = None          # e.g. "Artur"
    role: Optional[str] = None                  # e.g. "Senior .NET / Python Engineer"
    domain_expertise: Optional[str] = None      # e.g. "AI runtimes, RAG, ERP systems"

    # Environment / locale
    language: Optional[str] = None              # e.g. "pl", "en"
    locale: Optional[str] = None                # e.g. "pl-PL"
    timezone: Optional[str] = None              # e.g. "Europe/Warsaw"


@dataclass
class UserPreferences:
    """
    Stable user preferences that influence how the runtime and the LLM
    should behave by default.

    These preferences can be:
    - mirrored into system instructions (for the LLM),
    - and used programmatically by the runtime (e.g. to set max_tokens).
    """

    # Answer language & style
    preferred_language: Optional[str] = None    # e.g. "pl", "en"
    answer_length: Optional[str] = None         # e.g. "short", "detailed"
    tone: Optional[str] = None                  # e.g. "technical", "formal", "casual"

    # Formatting & content rules
    no_emojis_in_code: bool = False
    no_emojis_in_docs: bool = False
    prefer_markdown: bool = True
    prefer_code_blocks: bool = True

    # Project / domain context (high-level)
    default_project_context: Optional[str] = None
    # e.g. "Building Intergrax Drop-In Knowledge Runtime and Mooff ERP platform"

    # Arbitrary extra preferences to keep this extensible
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """
    Canonical user profile aggregate.

    It separates:
    - identity      (who the user is),
    - preferences   (how the user wants the system to behave),
    - summary       (short, compressed natural-language description used
                     as a base for system instructions).
    """

    identity: UserIdentity
    preferences: UserPreferences

    # Short natural-language summary suitable as a base for system instructions.
    # This should be kept small and periodically re-generated / compressed.
    summary_instructions: Optional[str] = None

    # Versioning / metadata hook if needed.
    version: int = 1


@dataclass
class UserProfilePromptBundle:
    """
    Small, prompt-ready bundle derived from the full profile and (optionally)
    from long-term / RAG-like profile memory.

    The engine should not need the full UserProfile on each request,
    only this compact bundle.

    Typical usage:
        - summary_instructions -> goes into system messages (small, stable)
        - hard_preferences     -> may influence runtime parameters (e.g. max_tokens)
        - user_profile_chunks  -> optional, RAG-derived facts about the user
        - org_profile_chunks   -> optional, RAG-derived facts about the org/tenant
    """

    # Short, compressed instructions to prepend as system messages.
    # This should already encode:
    #   - preferred language
    #   - tone and length defaults
    #   - "no emojis in code" rules, etc.
    summary_instructions: str = ""

    # Structured preferences that the runtime can use without asking the LLM.
    hard_preferences: Dict[str, Any] = field(default_factory=dict)

    # Optional additional context about the user, retrieved via RAG from
    # long-term profile memory. Should be used sparingly.
    user_profile_chunks: List[str] = field(default_factory=list)

    # Optional analogous chunks for organization / tenant profile.
    org_profile_chunks: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """
        Quick check to see if the bundle carries any meaningful information.
        """
        return (
            not self.summary_instructions
            and not self.hard_preferences
            and not self.user_profile_chunks
            and not self.org_profile_chunks
        )


def build_profile_prompt_bundle(profile: UserProfile) -> UserProfilePromptBundle:
    """
    Build a compact, prompt-ready bundle from a UserProfile.

    This function is intentionally conservative:
    - It does NOT call any LLM.
    - It does NOT perform RAG or long-term profile retrieval.
    - It uses existing `summary_instructions` if available,
      or constructs a minimal, deterministic fallback.

    Later, this function can be extended or replaced by a more advanced
    component that also uses long-term profile memory / RAG.
    """

    # 1) Determine summary instructions text
    if profile.summary_instructions:
        summary = profile.summary_instructions
    else:
        summary = _build_fallback_summary(profile)

    # 2) Structured hard preferences for the runtime
    hard_prefs: Dict[str, Any] = {}

    prefs = profile.preferences
    if prefs.preferred_language:
        hard_prefs["preferred_language"] = prefs.preferred_language
    if prefs.answer_length:
        hard_prefs["answer_length"] = prefs.answer_length
    if prefs.tone:
        hard_prefs["tone"] = prefs.tone

    hard_prefs["no_emojis_in_code"] = prefs.no_emojis_in_code
    hard_prefs["no_emojis_in_docs"] = prefs.no_emojis_in_docs
    hard_prefs["prefer_markdown"] = prefs.prefer_markdown
    hard_prefs["prefer_code_blocks"] = prefs.prefer_code_blocks

    if prefs.default_project_context:
        hard_prefs["default_project_context"] = prefs.default_project_context

    # Merge extra preferences as-is
    if prefs.extra:
        hard_prefs["extra"] = dict(prefs.extra)

    # At this stage we do NOT populate user_profile_chunks / org_profile_chunks.
    # These will be filled later when long-term profile memory (RAG) is introduced.
    bundle = UserProfilePromptBundle(
        summary_instructions=summary,
        hard_preferences=hard_prefs,
        user_profile_chunks=[],
        org_profile_chunks=[],
    )
    return bundle


def _build_fallback_summary(profile: UserProfile) -> str:
    """
    Build a deterministic, compact fallback summary used as system instructions
    when `profile.summary_instructions` is not explicitly set.

    This should remain short and stable; it is not meant to be a rich biography.
    """
    identity = profile.identity
    prefs = profile.preferences

    lines = []

    # Identity
    if identity.display_name:
        lines.append(f"You are talking to {identity.display_name}.")
    else:
        lines.append(f"You are talking to a user with id '{identity.user_id}'.")

    if identity.role:
        lines.append(f"The user is: {identity.role}.")
    if identity.domain_expertise:
        lines.append(f"Domain expertise: {identity.domain_expertise}.")

    # Language / style
    if prefs.preferred_language:
        lines.append(f"Always answer in {prefs.preferred_language} unless explicitly asked otherwise.")
    if prefs.tone:
        lines.append(f"Default tone: {prefs.tone}.")
    if prefs.answer_length:
        lines.append(f"Default answer length: {prefs.answer_length}.")

    # Formatting rules
    if prefs.no_emojis_in_code:
        lines.append("Never use emojis in code blocks.")
    if prefs.no_emojis_in_docs:
        lines.append("Avoid emojis in technical documentation.")
    if prefs.default_project_context:
        lines.append(f"Assume the default project context is: {prefs.default_project_context}.")

    # Fallback for minimal profile
    if not lines:
        lines.append(
            "You are talking to a user. Use a helpful, concise, and technical style by default."
        )

    return " ".join(lines)
