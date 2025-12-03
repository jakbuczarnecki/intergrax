# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class OrganizationIdentity:
    """
    Stable identification data for an organization.

    This is the organizational counterpart of `UserIdentity`.
    It should be stable over time and independent from the current session.
    """

    organization_id: str
    # Human-readable display name (short)
    name: str

    # Optional legal / extended naming
    legal_name: Optional[str] = None

    # Optional technical identifiers
    slug: Optional[str] = None           # e.g. "buildlogic", "acme-corp"
    primary_domain: Optional[str] = None # e.g. "buildlogic.pl"

    # Optional basic metadata
    industry: Optional[str] = None       # e.g. "software", "finance", "manufacturing"
    headquarters_location: Optional[str] = None  # e.g. "Warsaw, PL"
    default_timezone: Optional[str] = None       # e.g. "Europe/Warsaw"


@dataclass
class OrganizationPreferences:
    """
    Stable organization-level preferences that influence how the runtime
    should behave when working in the context of this organization.

    This is analogous to `UserPreferences`, but from a corporate / org perspective.
    """

    # Output / communication preferences
    default_language: str = "en"
    default_output_format: str = "markdown"  # e.g. "markdown", "text", "html"
    tone_of_voice: str = "neutral-professional"

    # Runtime capabilities and safety
    allow_web_search: bool = True
    allow_tools: bool = True

    # Sensitive / restricted topics (organization-specific)
    # Example: ["confidential_projects", "legal_disputes"]
    sensitive_topics: List[str] = field(default_factory=list)

    # Hard constraints: MUST be respected by the runtime
    # Example: ["Never share internal source code.", "Do not mention client names."]
    hard_constraints: List[str] = field(default_factory=list)

    # Soft guidelines: SHOULD be respected when possible
    # Example: ["Prefer concise answers.", "Use bullet points for checklists."]
    soft_guidelines: List[str] = field(default_factory=list)


@dataclass
class OrganizationProfile:
    """
    High-level organizational profile, similar in spirit to `UserProfile`.

    This is *not* meant to hold raw documents or large texts.
    Instead it holds compressed, human-curated summaries and metadata
    that can be used to build a small and stable system prompt bundle.
    """

    identity: OrganizationIdentity
    preferences: OrganizationPreferences = field(
        default_factory=OrganizationPreferences
    )

    # Optional custom summary instructions that, if provided, will be used
    # directly in the prompt bundle instead of an auto-generated summary.
    # This mirrors the behaviour of `UserProfile.summary_instructions`.
    summary_instructions: str = ""

    # Short, high-level description of what the organization does.
    # Example: "Buildlogic is a software company focused on AI-powered ERP systems."
    domain_summary: str = ""

    # Compressed summary of known organizational knowledge:
    # architecture principles, domain rules, important projects, etc.
    knowledge_summary: str = ""

    # Where the organization knowledge comes from (for traceability).
    # Example: ["Confluence: /ai/architecture", "Mooff docs v2", "Security handbook"]
    knowledge_sources: List[str] = field(default_factory=list)

    # Tags for filtering / routing.
    # Example: ["fintech", "erp", "ai", "b2b"]
    tags: List[str] = field(default_factory=list)

    # Last update timestamp of the profile (UTC).
    last_updated_utc: datetime = field(default_factory=datetime.utcnow)

    # Extra extensible metadata (for future use).
    extra: Dict[str, Any] = field(default_factory=dict)



@dataclass
class OrganizationProfilePromptBundle:
    """
    Compressed, prompt-ready representation of organization-level knowledge.

    This is the *only* part of the organizational memory that is injected
    into the system prompt in a stable way. Larger knowledge remains
    in RAG / vectorstores and is only pulled when needed.
    """

    # Main compact instruction block summarizing the organization:
    # who they are, what they do, how the assistant should behave for them.
    summary_instructions: str

    # Organization-level hard constraints that MUST be followed by the assistant.
    # These can be merged with user-level and global constraints.
    hard_constraints: List[str] = field(default_factory=list)

    # Organization-level soft guidelines that SHOULD be followed when possible.
    soft_guidelines: List[str] = field(default_factory=list)

    # Optional small, structured hints for the runtime (not directly shown to the LLM).
    # Example: {"default_project": "Mooff", "primary_domain": "erp"}
    runtime_hints: Dict[str, str] = field(default_factory=dict)

    @property
    def system_prompt(self) -> str:
        """
        Build a compact organization-level system prompt.

        Structure:
          - main summary,
          - explicit hard constraints (MUST),
          - explicit soft guidelines (SHOULD).
        """
        parts: List[str] = []

        if self.summary_instructions:
            parts.append(self.summary_instructions.strip())

        if self.hard_constraints:
            parts.append("Organization-level HARD CONSTRAINTS (MUST follow):")
            for c in self.hard_constraints:
                text = c.strip()
                if text:
                    parts.append(f"- {text}")

        if self.soft_guidelines:
            parts.append("Organization-level GUIDELINES (SHOULD follow when possible):")
            for g in self.soft_guidelines:
                text = g.strip()
                if text:
                    parts.append(f"- {text}")

        return "\n".join(parts).strip()


def build_organization_profile_prompt_bundle(
    profile: OrganizationProfile,
    max_summary_length: int = 1200,
) -> OrganizationProfilePromptBundle:
    """
    Build a compact, deterministic prompt bundle from an OrganizationProfile.

    Behaviour is intentionally symmetric with `build_profile_prompt_bundle()`:
    - If `profile.summary_instructions` is provided, it is used directly
      (optionally truncated with `max_summary_length`).
    - Otherwise, we build a fallback summary based on the profile fields.

    `max_summary_length` is a hard guardrail on the character length of
    `summary_instructions` in the final bundle.
    """

    # 1) Determine summary instructions text
    if profile.summary_instructions:
        summary_instructions = profile.summary_instructions

        if max_summary_length and max_summary_length > 0 and len(summary_instructions) > max_summary_length:
            summary_instructions = (
                summary_instructions[: max_summary_length - 3].rstrip() + "..."
            )
    else:
        summary_instructions = _build_fallback_summary(
            profile=profile,
            max_summary_length=max_summary_length,
        )

    # 2) Runtime hints and constraints
    identity = profile.identity
    prefs = profile.preferences

    runtime_hints: Dict[str, str] = {}

    if identity.primary_domain:
        runtime_hints["primary_domain"] = identity.primary_domain

    if identity.slug:
        runtime_hints["organization_slug"] = identity.slug

    if identity.default_timezone:
        runtime_hints["default_timezone"] = identity.default_timezone

    return OrganizationProfilePromptBundle(
        summary_instructions=summary_instructions,
        hard_constraints=list(prefs.hard_constraints),
        soft_guidelines=list(prefs.soft_guidelines),
        runtime_hints=runtime_hints,
    )



def _build_fallback_summary(
    profile: OrganizationProfile,
    max_summary_length: int,
) -> str:
    """
    Fallback summary builder used when `profile.summary_instructions`
    is not provided.

    This mirrors the idea used in `build_profile_prompt_bundle()` for users:
    we construct a deterministic, compact description of who the organization is,
    what it does and how the assistant should behave.
    """

    identity = profile.identity
    prefs = profile.preferences

    parts: List[str] = []

    parts.append(
        f"You are working in the context of the organization "
        f"'{identity.name}'."
    )

    if identity.legal_name and identity.legal_name != identity.name:
        parts.append(f"The legal name of the organization is: {identity.legal_name}.")

    if identity.industry:
        parts.append(f"The organization operates in the '{identity.industry}' industry.")

    if profile.domain_summary:
        parts.append(
            "High-level description of the organization:\n"
            f"{profile.domain_summary}"
        )

    if profile.knowledge_summary:
        parts.append(
            "Summary of key organizational knowledge, principles and domain rules:\n"
            f"{profile.knowledge_summary}"
        )

    parts.append(
        "When generating answers for this organization, use the following "
        "default behaviour unless the user explicitly asks otherwise:"
    )
    parts.append(
        f"- Default language: {prefs.default_language}\n"
        f"- Default output format: {prefs.default_output_format}\n"
        f"- Tone of voice: {prefs.tone_of_voice}"
    )

    capability_lines: List[str] = []
    capability_lines.append(
        f"- Web search is {'allowed' if prefs.allow_web_search else 'NOT allowed'} "
        "by default."
    )
    capability_lines.append(
        f"- Tools are {'allowed' if prefs.allow_tools else 'NOT allowed'} by default."
    )
    parts.append("Capabilities and constraints:\n" + "\n".join(capability_lines))

    if prefs.sensitive_topics:
        parts.append(
            "Treat the following topics as sensitive or restricted; "
            "handle them carefully and prefer concise, non-revealing answers "
            "unless explicitly authorized:\n"
            + ", ".join(prefs.sensitive_topics)
        )

    if profile.knowledge_sources:
        parts.append(
            "Main knowledge sources used to construct this organizational profile:\n"
            + ", ".join(profile.knowledge_sources)
        )

    summary = "\n\n".join(parts)

    if max_summary_length and max_summary_length > 0 and len(summary) > max_summary_length:
        # Hard truncation by characters; simple but deterministic.
        summary = summary[: max_summary_length - 3].rstrip() + "..."

    return summary
