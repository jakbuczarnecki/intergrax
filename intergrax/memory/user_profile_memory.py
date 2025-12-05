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
class UserProfileMemoryEntry:
    """
    Long-term memory entry for a user profile.

    Stores stable facts, insights, or notes about the user. 
    Not tied to any runtime message structure.
    """
    entry_id: Optional[int] = None
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    deleted: bool = False
    modified: bool = False


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
    - system instructions (short, compressed natural-language description
                           used as a base for LLM system prompts),
    - memory        (long-term factual and conceptual notes about the user).
    """

    identity: UserIdentity
    preferences: UserPreferences

    # Short natural-language instructions used directly (or almost directly)
    # as system-level instructions for the runtime. This should be kept small
    # and periodically re-generated / compressed.
    system_instructions: Optional[str] = None

    # Long-term memory entries about the user (facts, insights, stable notes).
    # These are not sent directly to the LLM by default; they are used to
    # derive or update `system_instructions` and other summaries.
    memory_entries: List[UserProfileMemoryEntry] = field(default_factory=list)

    # Versioning / metadata hook if needed.
    version: int = 1    

    entry_id: Optional[int] = None
    deleted: bool = False
    modified: bool = False

