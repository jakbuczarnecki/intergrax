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
    Single source of truth for an organization's long-term profile.

    Mirrors the UserProfile structure:

      - identity + preferences: stable configuration
      - system_instructions: compact, prompt-ready instructions for the runtime
      - memory_entries: list of long-term memory entries (unit-of-work)

    Legacy fields (summary_instructions, domain_summary, knowledge_summary, ...)
    are kept for backwards compatibility and potential migration,
    but new code should treat `system_instructions` as the primary
    organization-level system prompt.
    """

    identity: OrganizationIdentity
    preferences: OrganizationPreferences = field(
        default_factory=OrganizationPreferences
    )

    # New architecture: compact, prompt-ready system instructions
    # used directly (or almost directly) in the runtime as system-level
    # instructions for this organization.
    system_instructions: Optional[str] = None

    # ------------------------------------------------------------------
    # Legacy / high-level summary fields
    # ------------------------------------------------------------------

    # Legacy field: if present, can be used as a pre-composed summary
    # for older code or migration logic. New code should prefer
    # `system_instructions` instead.
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

    # Top-level modification flag for the profile (e.g. preferences,
    # system_instructions, extra). Concrete stores may use this to decide
    # whether they need to perform an UPDATE at the profile level.
    modified: bool = False