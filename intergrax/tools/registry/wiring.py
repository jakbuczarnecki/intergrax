# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 dependency injection for tool handlers (Phase O.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class ToolWiringContext:
    """
    Composed dependencies passed into catalog tool registration.

    Tier-3 applications build this from ``IntegrationProfile`` and runtime
    services (RAG manager, websearch executor, …). Tool handlers MUST NOT
    resolve integrations themselves.
    """

    issue_tracker: Any | None = None
    search_provider: Any | None = None
    wiki_knowledge: Any | None = None
    notification_channel: Any | None = None
    observability_backend: Any | None = None
    rag_manager: Any | None = None
    websearch_executor: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_integration_profile(
        cls,
        profile: Any,
        *,
        rag_manager: Any | None = None,
        websearch_executor: Any | None = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> ToolWiringContext:
        """
        Resolve common integration contract slots from an ``IntegrationProfile``.

        Categories without a configured slug are skipped (``None``).
        """
        from intergrax.integrations.contracts.base import IntegrationCategory

        def _optional(category: IntegrationCategory) -> Any | None:
            slug = profile.slug_for_category(category)
            if slug is None:
                return None
            try:
                return profile.resolve(category)
            except Exception:
                return None

        return cls(
            issue_tracker=_optional(IntegrationCategory.ISSUE_TRACKER),
            search_provider=_optional(IntegrationCategory.SEARCH_PROVIDER),
            wiki_knowledge=_optional(IntegrationCategory.WIKI_KNOWLEDGE),
            notification_channel=_optional(IntegrationCategory.NOTIFICATION_CHANNEL),
            observability_backend=_optional(IntegrationCategory.OBSERVABILITY_BACKEND),
            rag_manager=rag_manager,
            websearch_executor=websearch_executor,
            extras=dict(extras or {}),
        )
