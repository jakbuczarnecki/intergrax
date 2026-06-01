# © Artur Czarnecki. All rights reserved.

"""User/org profile instruction resolution for sessions (Phase Q+-S.1)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tracing.session.session_consolidation_diag import (
    SessionConsolidationDiagV1,
)
from intergrax.runtime.organization.organization_profile_manager import (
    OrganizationProfileManager,
)


class SessionProfileInstructionResolver:
    """Resolve and cache per-session user/org system instructions."""

    def __init__(
        self,
        *,
        user_profile_manager: Optional[UserProfileManager] = None,
        organization_profile_manager: Optional[OrganizationProfileManager] = None,
    ) -> None:
        self._user_profile_manager = user_profile_manager
        self._organization_profile_manager = organization_profile_manager

    async def user_instructions_for_session(self, session: ChatSession) -> Optional[str]:
        if not session.user_id or self._user_profile_manager is None:
            return None

        cached = session.user_profile_instructions
        if not session.needs_user_instructions_refresh and isinstance(cached, str):
            stripped = cached.strip()
            if stripped:
                return stripped

        instructions = await self._user_profile_manager.get_system_instructions_for_user(
            session.user_id
        )
        if not isinstance(instructions, str):
            return None

        stripped = instructions.strip()
        if not stripped:
            return None

        session.user_profile_instructions = stripped
        session.needs_user_instructions_refresh = False
        return stripped

    async def org_instructions_for_session(self, session: ChatSession) -> Optional[str]:
        if not session.tenant_id or self._organization_profile_manager is None:
            return None

        cached = session.org_profile_instructions
        if isinstance(cached, str):
            stripped = cached.strip()
            if stripped:
                return stripped

        instructions = (
            await self._organization_profile_manager.get_system_instructions_for_organization(
                organization_id=session.tenant_id,
            )
        )
        if not isinstance(instructions, str):
            return None

        stripped = instructions.strip()
        if not stripped:
            return None

        session.org_profile_instructions = stripped
        return stripped


@runtime_checkable
class ConsolidationEntryLike(Protocol):
    entry_type: str


def _entry_type_label(entry: Any) -> str:
    if isinstance(entry, dict):
        raw = entry.get("entry_type")
        return str(raw) if raw is not None else "unknown"
    if isinstance(entry, ConsolidationEntryLike):
        return entry.entry_type
    return "unknown"


def build_consolidation_diag(entries: Sequence[Any]) -> SessionConsolidationDiagV1:
    """Build JSON-safe consolidation debug payload without reflection."""
    type_counts: Dict[str, int] = {}
    for entry in entries:
        label = _entry_type_label(entry)
        type_counts[label] = type_counts.get(label, 0) + 1

    return SessionConsolidationDiagV1(
        entries_count=len(entries),
        entry_types=type_counts,
    )
