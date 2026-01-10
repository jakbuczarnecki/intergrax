# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
import uuid

from intergrax.globals.settings import GLOBAL_SETTINGS


# ---------------------------------------------------------------------------
# Core domain models for user / org profile and prompt bundles.
# These models are intentionally independent from any storage or engine logic.
# They represent the "language" in which we describe identities, preferences
# and how they should be injected into LLM prompts.
# ---------------------------------------------------------------------------

class MemoryKind(Enum):
    USER_FACT = "user_fact"
    PREFERENCE = "preference"
    SESSION_SUMMARY = "session_summary"
    ORG_FACT = "org_fact"
    POLICY = "policy"
    OTHER = "other"


class MemoryImportance(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UserProfileMemoryEntry:
    """
    Long-term memory entry for a user profile.

    Stores stable facts, insights, or notes about the user.
    Can also store session-related summaries, linked via session_id.
    """

    # Persistent identifier in the storage backend.
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # Main content of the memory entry (human-readable text).
    content: str = ""

    # Optional link to the session from which this entry was derived.
    # None means "not tied to a specific session".
    session_id: Optional[str] = None

    # High-level type of this memory entry.
    # Useful for filtering, retrieval strategies, and UI.
    kind: MemoryKind = MemoryKind.OTHER

    # Short human-readable title (e.g. "Summary of session 2025-12-09").
    title: Optional[str] = None

    # Importance level used to prioritize entries during retrieval.
    importance: MemoryImportance = MemoryImportance.MEDIUM

    # Creation timestamp in ISO format (UTC).
    # You can also store datetime and convert in the store layer;
    # here we keep string for easier serialization.
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # Additional, less frequently queried metadata.
    # Example: {"tags": ["intergrax", "memory", "profiles"], "source": "session_summarizer"}
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Unit-of-work flags used by the manager/store.
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
    language: Optional[str] = GLOBAL_SETTINGS.default_language             # e.g. "pl", "en"
    locale: Optional[str] = GLOBAL_SETTINGS.default_locale                # e.g. "pl-PL"
    timezone: Optional[str] = GLOBAL_SETTINGS.default_timezone              # e.g. "Europe/Warsaw"


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
    # e.g. "Building Intergrax nexus Runtime and Mooff ERP platform"

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

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    deleted: bool = False
    modified: bool = False

