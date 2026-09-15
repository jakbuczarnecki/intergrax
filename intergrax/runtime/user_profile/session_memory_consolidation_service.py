# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage, MessageRole
from intergrax.memory.strategies import (
    MemoryCandidate,
    MemoryDeduplicationRequest,
    MemoryExtractionRequest,
    MemoryPromotionAction,
    MemoryPromotionRequest,
    MemoryStrategySet,
    build_default_memory_strategies,
)
from intergrax.memory.memory_entry_materialization import materialize_user_profile_memory_entry
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import (
    MemoryImportance,
    UserProfileMemoryEntry,
)
from intergrax.runtime.user_profile.user_profile_instructions_service import UserProfileInstructionsService
from intergrax.utils.time_provider import SystemTimeProvider


@dataclass
class SessionMemoryConsolidationConfig:
    """
    Configuration for consolidating a single chat session into long-term
    user profile memory entries and optionally refreshing user-level
    system instructions.
    """

    language: str = GLOBAL_SETTINGS.default_language
    max_facts: int = 8
    max_preferences: int = 6
    include_session_summary: bool = True
    default_fact_importance: MemoryImportance = MemoryImportance.MEDIUM
    default_preference_importance: MemoryImportance = MemoryImportance.MEDIUM
    default_summary_importance: MemoryImportance = MemoryImportance.MEDIUM
    max_messages_in_prompt: int = 80
    max_conversation_chars: int = 6000
    temperature: Optional[float] = None
    regenerate_system_instructions: bool = True
    force_regenerate_system_instructions: bool = False
    included_roles: Sequence[MessageRole] = ("user", "assistant")
    extra: Dict[str, Any] = None
    deduplication_similarity_threshold: float = 0.88

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class SessionMemoryConsolidationService:
    """
    Orchestrates session consolidation into long-term user profile memory.

    Decision logic is delegated to injected memory strategies; this service
    handles conversation trimming, entry materialization, and persistence.
    """

    def __init__(
        self,
        profile_manager: UserProfileManager,
        instructions_service: UserProfileInstructionsService,
        *,
        strategies: MemoryStrategySet | None = None,
        llm: LLMAdapter | None = None,
        config: Optional[SessionMemoryConsolidationConfig] = None,
        entity_graph_service: Any | None = None,
    ) -> None:
        self._profile_manager = profile_manager
        self._instructions_service = instructions_service
        self._config = config or SessionMemoryConsolidationConfig()
        self._entity_graph_service = entity_graph_service

        if strategies is not None:
            self._strategies = strategies
        elif llm is not None:
            from intergrax.memory.strategies.defaults.sequence_deduplication import (
                SequenceMatcherDeduplicationConfig,
            )

            self._strategies = build_default_memory_strategies(
                llm,
                deduplication_config=SequenceMatcherDeduplicationConfig(
                    similarity_threshold=self._config.deduplication_similarity_threshold,
                ),
            )
        else:
            raise ValueError("SessionMemoryConsolidationService requires strategies or llm")

    async def consolidate_session(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[ChatMessage],
        run_id: Optional[str] = None,
    ) -> List[UserProfileMemoryEntry]:
        trimmed = self._prepare_conversation_for_prompt(messages)
        if not trimmed:
            return []

        extraction_request = MemoryExtractionRequest(
            user_id=user_id,
            session_id=session_id,
            messages=tuple(trimmed),
            language=self._config.language,
            max_facts=self._config.max_facts,
            max_preferences=self._config.max_preferences,
            include_session_summary=self._config.include_session_summary,
            default_fact_importance=self._config.default_fact_importance,
            default_preference_importance=self._config.default_preference_importance,
            default_summary_importance=self._config.default_summary_importance,
            temperature=self._config.temperature,
            run_id=run_id,
        )
        extraction_result = await self._strategies.extraction.extract(extraction_request)
        if not extraction_result.candidates:
            return []

        entries = [
            self._materialize_entry(candidate, run_id=run_id)
            for candidate in extraction_result.candidates
        ]

        profile = await self._profile_manager.get_profile(user_id)
        now_iso = SystemTimeProvider.utc_now().isoformat()
        for entry in entries:
            entry.valid_from = entry.valid_from or now_iso

        dedup_result = self._strategies.deduplication.deduplicate(
            MemoryDeduplicationRequest(
                existing=tuple(profile.memory_entries),
                incoming=tuple(entries),
            )
        )
        if not dedup_result.accepted:
            return []

        promotion_result = self._strategies.promotion.promote(
            MemoryPromotionRequest(
                user_id=user_id,
                session_id=session_id,
                entries=dedup_result.accepted,
            )
        )

        stored_entries: List[UserProfileMemoryEntry] = []
        for decision in promotion_result.decisions:
            if decision.action is not MemoryPromotionAction.PROMOTE:
                continue
            stored = await self._profile_manager.add_memory_entry(user_id, decision.entry)
            stored_entries.append(stored)
            if self._entity_graph_service is not None:
                self._entity_graph_service.index_memory_entry(user_id=user_id, entry=stored)

        if stored_entries and self._config.regenerate_system_instructions:
            await self._instructions_service.build_and_save_system_instructions(
                user_id=user_id,
                force=self._config.force_regenerate_system_instructions,
                run_id=run_id,
            )

        return stored_entries

    def _materialize_entry(
        self,
        candidate: MemoryCandidate,
        *,
        run_id: Optional[str] = None,
    ) -> UserProfileMemoryEntry:
        strategy_id = self._strategies.extraction.strategy_id
        return materialize_user_profile_memory_entry(
            candidate,
            strategy_id=strategy_id,
            run_id=run_id,
        )

    def _prepare_conversation_for_prompt(
        self,
        messages: Sequence[ChatMessage],
    ) -> List[ChatMessage]:
        if not messages:
            return []

        filtered: List[ChatMessage] = []
        for msg in messages:
            if msg.role in self._config.included_roles:
                filtered.append(msg)

        if not filtered:
            return []

        if len(filtered) > self._config.max_messages_in_prompt:
            filtered = filtered[-self._config.max_messages_in_prompt :]

        total_chars = 0
        trimmed: List[ChatMessage] = []

        for msg in reversed(filtered):
            content = msg.content or ""
            length = len(content)
            if total_chars + length > self._config.max_conversation_chars:
                break
            trimmed.append(msg)
            total_chars += length

        trimmed.reverse()
        return trimmed
