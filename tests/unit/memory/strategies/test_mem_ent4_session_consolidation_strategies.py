# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-4: pluggable memory strategy SPI for session consolidation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.memory.strategies import (
    MemoryCandidate,
    MemoryDeduplicationRequest,
    MemoryDeduplicationResult,
    MemoryExtractionRequest,
    MemoryExtractionResult,
    MemoryPromotionAction,
    MemoryPromotionDecision,
    MemoryPromotionRequest,
    MemoryPromotionResult,
    MemoryStrategySet,
    build_default_memory_strategies,
)
from intergrax.memory.strategies.defaults.accept_all_promotion import AcceptAllMemoryPromotionStrategy
from intergrax.memory.strategies.defaults.llm_extraction import LlmMemoryExtractionStrategy
from intergrax.memory.strategies.defaults.sequence_deduplication import (
    SequenceMatcherDeduplicationConfig,
    SequenceMatcherMemoryDeduplicationStrategy,
)
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.runtime.user_profile.session_memory_consolidation_service import (
    SessionMemoryConsolidationConfig,
    SessionMemoryConsolidationService,
)
from intergrax.runtime.user_profile.user_profile_instructions_service import UserProfileInstructionsService
from testing_support.builder import FakeLLMAdapter
from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.gate


def _deterministic_consolidation_json() -> str:
    return json.dumps(
        {
            "facts": [{"title": "Role", "content": "Senior Python engineer", "importance": "HIGH", "tags": ["user"]}],
            "preferences": [
                {
                    "title": "Tone",
                    "content": "Concise technical answers in English",
                    "importance": "MEDIUM",
                    "tags": ["communication"],
                }
            ],
            "session_summary": {
                "title": "Session recap",
                "content": "Discussed memory consolidation wiring",
                "importance": "MEDIUM",
                "tags": ["session_summary"],
            },
        }
    )


def _profile_manager_mock() -> MagicMock:
    from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile

    profile_manager = MagicMock()
    profile_manager.get_profile = AsyncMock(
        return_value=UserProfile(
            identity=UserIdentity(user_id="user-1"),
            preferences=UserPreferences(),
        )
    )
    profile_manager.add_memory_entry = AsyncMock(side_effect=lambda _uid, entry: entry)
    return profile_manager


@dataclass
class _RecordingExtraction:
    strategy_id: str = "test.extraction.recording"
    calls: int = 0
    candidates: tuple[MemoryCandidate, ...] = ()

    async def extract(self, request: MemoryExtractionRequest) -> MemoryExtractionResult:
        self.calls += 1
        if self.candidates:
            return MemoryExtractionResult(candidates=self.candidates)
        return MemoryExtractionResult(
            candidates=(
                MemoryCandidate(
                    content="injected fact",
                    kind=MemoryKind.USER_FACT,
                    session_id=request.session_id,
                    title="Injected",
                ),
            )
        )


class _KeepAllDeduplication:
    strategy_id = "test.dedup.keep_all"

    def deduplicate(self, request: MemoryDeduplicationRequest) -> MemoryDeduplicationResult:
        return MemoryDeduplicationResult(
            accepted=tuple(request.incoming),
            rejected_as_duplicate=(),
        )


class _RejectAllDeduplication:
    strategy_id = "test.dedup.reject_all"

    def deduplicate(self, request: MemoryDeduplicationRequest) -> MemoryDeduplicationResult:
        return MemoryDeduplicationResult(
            accepted=(),
            rejected_as_duplicate=tuple(request.incoming),
        )


class _SkipAllPromotion:
    strategy_id = "test.promotion.skip_all"

    def promote(self, request: MemoryPromotionRequest) -> MemoryPromotionResult:
        return MemoryPromotionResult(
            decisions=tuple(
                MemoryPromotionDecision(
                    entry=entry,
                    action=MemoryPromotionAction.SKIP,
                    reason="test_skip",
                )
                for entry in request.entries
            )
        )


@pytest.mark.asyncio
async def test_custom_extraction_strategy_is_used_by_consolidation_service() -> None:
    extraction = _RecordingExtraction()
    strategies = MemoryStrategySet(
        extraction=extraction,
        deduplication=_KeepAllDeduplication(),
        promotion=AcceptAllMemoryPromotionStrategy(),
    )
    service = SessionMemoryConsolidationService(
        profile_manager=_profile_manager_mock(),
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=strategies,
    )
    entries = await service.consolidate_session(
        user_id="user-1",
        session_id="sess-1",
        messages=[ChatMessage(role="user", content="hello")],
    )
    assert extraction.calls == 1
    assert len(entries) == 1
    assert entries[0].content == "injected fact"


@pytest.mark.asyncio
async def test_default_extraction_matches_prior_consolidation_flow() -> None:
    profile_manager = _profile_manager_mock()
    instructions_service = MagicMock(spec=UserProfileInstructionsService)
    instructions_service.build_and_save_system_instructions = AsyncMock(return_value="ok")

    service = SessionMemoryConsolidationService(
        profile_manager=profile_manager,
        instructions_service=instructions_service,
        llm=FakeLLMAdapter(fixed_text=_deterministic_consolidation_json()),
        config=SessionMemoryConsolidationConfig(include_session_summary=True),
    )
    entries = await service.consolidate_session(
        user_id="user-1",
        session_id="sess-ltm-1",
        messages=[
            ChatMessage(role="user", content="I am a senior Python engineer."),
            ChatMessage(role="assistant", content="Noted."),
        ],
    )
    kinds = {entry.kind for entry in entries}
    assert MemoryKind.USER_FACT in kinds
    assert MemoryKind.PREFERENCE in kinds
    assert MemoryKind.SESSION_SUMMARY in kinds
    assert MemoryKind.EPISODIC_EVENT in kinds
    assert profile_manager.add_memory_entry.await_count == 4


@pytest.mark.asyncio
async def test_custom_dedup_keep_all_vs_reject_all() -> None:
    candidate = MemoryCandidate(
        content="fact",
        kind=MemoryKind.USER_FACT,
        session_id="s1",
    )
    extraction = _RecordingExtraction(candidates=(candidate,))

    keep_service = SessionMemoryConsolidationService(
        profile_manager=_profile_manager_mock(),
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=MemoryStrategySet(
            extraction=extraction,
            deduplication=_KeepAllDeduplication(),
            promotion=AcceptAllMemoryPromotionStrategy(),
        ),
    )
    kept = await keep_service.consolidate_session("u", "s1", [ChatMessage(role="user", content="x")])
    assert len(kept) == 1

    reject_service = SessionMemoryConsolidationService(
        profile_manager=_profile_manager_mock(),
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=MemoryStrategySet(
            extraction=extraction,
            deduplication=_RejectAllDeduplication(),
            promotion=AcceptAllMemoryPromotionStrategy(),
        ),
    )
    rejected = await reject_service.consolidate_session("u", "s1", [ChatMessage(role="user", content="x")])
    assert rejected == []


@pytest.mark.asyncio
async def test_default_dedup_threshold_is_configurable_not_in_service() -> None:
    from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile

    existing = UserProfileMemoryEntry(content="Senior Python engineer", kind=MemoryKind.USER_FACT)
    profile_manager = _profile_manager_mock()
    profile_manager.get_profile = AsyncMock(
        return_value=UserProfile(
            identity=UserIdentity(user_id="user-1"),
            preferences=UserPreferences(),
            memory_entries=[existing],
        )
    )

    low_threshold = SequenceMatcherMemoryDeduplicationStrategy(
        SequenceMatcherDeduplicationConfig(similarity_threshold=0.5)
    )
    high_threshold = SequenceMatcherMemoryDeduplicationStrategy(
        SequenceMatcherDeduplicationConfig(similarity_threshold=0.99)
    )

    candidate = MemoryCandidate(
        content="Senior python engineer.",
        kind=MemoryKind.USER_FACT,
        session_id="s1",
    )
    extraction = _RecordingExtraction(candidates=(candidate,))

    low_service = SessionMemoryConsolidationService(
        profile_manager=profile_manager,
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=MemoryStrategySet(
            extraction=extraction,
            deduplication=low_threshold,
            promotion=AcceptAllMemoryPromotionStrategy(),
        ),
    )
    high_service = SessionMemoryConsolidationService(
        profile_manager=profile_manager,
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=MemoryStrategySet(
            extraction=extraction,
            deduplication=high_threshold,
            promotion=AcceptAllMemoryPromotionStrategy(),
        ),
    )
    low_result = await low_service.consolidate_session("u", "s1", [ChatMessage(role="user", content="x")])
    high_result = await high_service.consolidate_session("u", "s1", [ChatMessage(role="user", content="x")])
    assert low_result == []
    assert len(high_result) == 1


@pytest.mark.asyncio
async def test_promotion_skip_prevents_storage_writes() -> None:
    candidate = MemoryCandidate(content="fact", kind=MemoryKind.USER_FACT, session_id="s1")
    profile_manager = _profile_manager_mock()
    service = SessionMemoryConsolidationService(
        profile_manager=profile_manager,
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=MemoryStrategySet(
            extraction=_RecordingExtraction(candidates=(candidate,)),
            deduplication=_KeepAllDeduplication(),
            promotion=_SkipAllPromotion(),
        ),
    )
    entries = await service.consolidate_session("u", "s1", [ChatMessage(role="user", content="x")])
    assert entries == []
    profile_manager.add_memory_entry.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_protocol_strategy_without_core_changes() -> None:
    class CustomDeduplication:
        strategy_id = "external.custom.dedup"

        def deduplicate(self, request: MemoryDeduplicationRequest) -> MemoryDeduplicationResult:
            return MemoryDeduplicationResult(accepted=request.incoming, rejected_as_duplicate=())

    service = SessionMemoryConsolidationService(
        profile_manager=_profile_manager_mock(),
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=MemoryStrategySet(
            extraction=_RecordingExtraction(),
            deduplication=CustomDeduplication(),
            promotion=AcceptAllMemoryPromotionStrategy(),
        ),
    )
    entries = await service.consolidate_session("u", "s1", [ChatMessage(role="user", content="hello")])
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_strategy_failure_propagates_without_partial_writes() -> None:
    class FailingExtraction:
        strategy_id = "test.extraction.fail"

        async def extract(self, request: MemoryExtractionRequest) -> MemoryExtractionResult:
            raise RuntimeError("extraction failed")

    profile_manager = _profile_manager_mock()
    service = SessionMemoryConsolidationService(
        profile_manager=profile_manager,
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=MemoryStrategySet(
            extraction=FailingExtraction(),
            deduplication=_KeepAllDeduplication(),
            promotion=AcceptAllMemoryPromotionStrategy(),
        ),
    )
    with pytest.raises(RuntimeError, match="extraction failed"):
        await service.consolidate_session("u", "s1", [ChatMessage(role="user", content="x")])
    profile_manager.add_memory_entry.assert_not_awaited()


@pytest.mark.asyncio
async def test_consolidation_still_writes_via_profile_manager() -> None:
    profile_manager = _profile_manager_mock()
    service = SessionMemoryConsolidationService(
        profile_manager=profile_manager,
        instructions_service=MagicMock(spec=UserProfileInstructionsService),
        strategies=MemoryStrategySet(
            extraction=_RecordingExtraction(),
            deduplication=_KeepAllDeduplication(),
            promotion=AcceptAllMemoryPromotionStrategy(),
        ),
    )
    await service.consolidate_session("u", "s1", [ChatMessage(role="user", content="hello")])
    profile_manager.add_memory_entry.assert_awaited_once()


def test_build_default_strategies_respects_deduplication_configuration() -> None:
    llm = FakeLLMAdapter(fixed_text="{}")
    strategies = build_default_memory_strategies(
        llm,
        deduplication_config=SequenceMatcherDeduplicationConfig(similarity_threshold=0.75),
    )
    assert isinstance(strategies.extraction, LlmMemoryExtractionStrategy)
    assert isinstance(strategies.deduplication, SequenceMatcherMemoryDeduplicationStrategy)
    assert strategies.deduplication.similarity_threshold == 0.75
