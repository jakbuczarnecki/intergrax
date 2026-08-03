# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.tracing.history.history_summary import HistorySummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.prompts.history_prompt_builder import HistorySummaryPromptBuilder
from intergrax.runtime.nexus.responses.response_schema import HistoryCompressionStrategy
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_manager import SessionManager

LEGACY_HISTORY_COMPRESSION_DISABLED_REASON = (
    "legacy_history_compression_disabled_use_ucl"
)


class LegacyHistoryCompressionDisabledError(RuntimeError):
    reason = LEGACY_HISTORY_COMPRESSION_DISABLED_REASON

    def __init__(self) -> None:
        super().__init__(self.reason)


class HistoryLayer:

    def __init__(
        self,
        config: RuntimeConfig,
        session_manager: SessionManager,
        history_prompt_builder: HistorySummaryPromptBuilder,
    ) -> None:
        """
        HistoryLayer encapsulates legacy raw conversation history loading.

        The history_prompt_builder parameter is retained for legacy constructor
        compatibility only; must not be invoked.
        """
        self._config = config
        self._session_manager = session_manager
        _ = history_prompt_builder

    async def build_base_history(self, state: RuntimeState) -> None:
        """
        Load raw conversation history for the current session when compression
        is explicitly disabled (OFF).

        Legacy reduction strategies fail closed and must be migrated to UCL.
        """
        strategy = state.request.history_compression_strategy
        if not isinstance(
            strategy,
            HistoryCompressionStrategy,
        ):
            raise TypeError(
                "history_compression_strategy must be "
                "HistoryCompressionStrategy"
            )

        if strategy is not HistoryCompressionStrategy.OFF:
            raise LegacyHistoryCompressionDisabledError()

        session = state.session
        assert session is not None, (
            "Session must be set before building history."
        )

        raw_history: List[ChatMessage] = await self._build_chat_history(session)

        state.base_history = list(raw_history)

        try:
            raw_token_count = self._count_tokens_for_messages(raw_history)
        except Exception:
            raw_token_count = None
        state.history_token_count = raw_token_count

        state.trace_event(
            component=TraceComponent.ENGINE,
            step="history",
            message="Legacy HistoryLayer loaded raw history without optimization.",
            level=TraceLevel.INFO,
            payload=HistorySummaryDiagV1(
                base_history_length=len(raw_history),
                history_length=len(raw_history),
                history_tokens=raw_token_count,
                history_includes_current_user=False,
            ),
        )

    async def _build_chat_history(self, session: ChatSession) -> List[ChatMessage]:
        """
        Load raw conversation history for the given session.

        This method is responsible only for fetching history from SessionStore.
        """
        return await self._session_manager.get_history(
            tenant_id=session.tenant_id,
            session_id=session.id,
        )

    def _count_tokens_for_messages(self, messages: List[ChatMessage]) -> Optional[int]:
        """
        Best-effort token counting for a list of ChatMessage objects.

        Diagnostic only — the result must not influence history selection.
        """
        adapter = self._config.llm_adapter
        if adapter is None:
            return None

        try:
            return int(adapter.count_messages_tokens(messages))
        except AttributeError:
            return None
        except Exception:
            return None
