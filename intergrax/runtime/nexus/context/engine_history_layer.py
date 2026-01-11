# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from typing import TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.tracing.history.history_summary import HistorySummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.prompts.history_prompt_builder import HistorySummaryPromptBuilder
from intergrax.runtime.nexus.responses.response_schema import HistoryCompressionStrategy, RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_manager import SessionManager


@dataclass
class HistoryCompressionResult:
    """
    Result of applying a history compression strategy.

    This object groups both the compressed messages (the actual base history
    to be sent to the LLM) and all diagnostic / bookkeeping information
    that is useful for debugging and telemetry.
    """

    # Final history that should be used as the base conversation context.
    history: List[ChatMessage]

    # Whether any truncation or summarization was applied.
    truncated: bool

    # Strategy that was actually used. This may differ from the requested
    # strategy in case of fallbacks (e.g. summarization failing and falling
    # back to pure truncation).
    effective_strategy: HistoryCompressionStrategy

    # Whether a summarization step was successfully used.
    summary_used: bool

    # Token budgets used during compression. These are best-effort diagnostic
    # values and may be zero if the strategy did not rely on them.
    summary_tokens_budget: int
    tail_tokens_budget: int

    # Raw metrics for the original history.
    raw_history_messages: int
    raw_history_tokens: Optional[int]

    # Budget that was passed into the compressor for the history.
    history_budget_tokens: Optional[int]

class HistoryLayer:
    
    def __init__(
        self,
        config: RuntimeConfig,
        session_manager: SessionManager,
        history_prompt_builder: HistorySummaryPromptBuilder,
    ) -> None:
        """
        HistoryLayer encapsulates all logic related to:

          - loading raw conversation history,
          - counting tokens,
          - applying history compression strategies,
          - updating RuntimeState with base_history and debug info.
        """
        self._config = config
        self._session_manager = session_manager
        self._history_prompt_builder = history_prompt_builder

    
    async def build_base_history(self, state: RuntimeState) -> None:
        """
        Load and preprocess the conversation history for the current session.

        This step is the single place where we:
          - fetch the full session history from SessionStore,
          - compute token usage (if the adapter supports it),
          - apply token-based truncation according to the per-request
            history compression strategy.

        The resulting `state.base_history` is treated as the canonical,
        preprocessed conversation history for all subsequent steps.
        """
        session = state.session
        assert session is not None, "Session must be set before building history."

        # 1. Load raw history from SessionStore.
        raw_history: List[ChatMessage] = await self._build_chat_history(session)

        # 2. Compute token usage for the raw history, if possible.
        raw_token_count = self._count_tokens_for_messages(raw_history)
        state.history_token_count = raw_token_count

        # 3. Resolve per-request settings.
        request = state.request
        strategy = request.history_compression_strategy
        adapter = self._config.llm_adapter

        # If we cannot count tokens at all, we cannot apply token-based
        # trimming. In that case we simply keep the full history and log
        # what we know in a unified way.
        if raw_token_count is None:
            compression_result = HistoryCompressionResult(
                history=raw_history,
                truncated=False,
                effective_strategy=strategy,
                summary_used=False,
                summary_tokens_budget=0,
                tail_tokens_budget=0,
                raw_history_messages=len(raw_history),
                raw_history_tokens=None,
                history_budget_tokens=None,
            )

            state.base_history = compression_result.history
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="history",
                message="History compression summary.",
                level=TraceLevel.INFO,
                payload=HistorySummaryDiagV1(
                    base_history_length=int(compression_result.raw_history_messages),
                    history_length=int(len(compression_result.history)),
                    history_tokens=compression_result.raw_history_tokens,
                ),
            )
            return

        # 4. Compute a token budget for history based on:
        #    - the model context window,
        #    - the requested max_output_tokens (if any).
        #
        # We use a simple, conservative heuristic:
        #   - reserve a portion of the context window for the model output,
        #   - reserve a portion of the remaining input for system instructions,
        #     memory, RAG, websearch, tools, etc.
        #   - whatever remains is the history budget.
        context_window = adapter.context_window_tokens

        # Determine how many tokens we should reserve for the output.
        # If the user does not specify max_output_tokens, we assume
        # roughly 1/4 of the context window is available for the output.
        if request.max_output_tokens is not None:
            reserved_for_output = request.max_output_tokens
            # Never reserve more than half of the context window for output.
            if reserved_for_output > context_window // 2:
                reserved_for_output = context_window // 2
        else:
            reserved_for_output = context_window // 4

        if reserved_for_output < 0:
            reserved_for_output = 0
        if reserved_for_output >= context_window:
            # Degenerate case – leave at least some room for input.
            reserved_for_output = context_window // 2

        # Budget for the entire input (system + history + RAG + tools...).
        input_budget = context_window - reserved_for_output

        if input_budget <= 0:
            # Extremely small or misconfigured budget; in this case we keep
            # the history as-is and log the situation in a unified way.
            compression_result = HistoryCompressionResult(
                history=raw_history,
                truncated=False,
                effective_strategy=strategy,
                summary_used=False,
                summary_tokens_budget=0,
                tail_tokens_budget=0,
                raw_history_messages=len(raw_history),
                raw_history_tokens=raw_token_count,
                history_budget_tokens=0,
            )

            state.base_history = compression_result.history

            state.trace_event(
                component=TraceComponent.ENGINE,
                step="history",
                message="History compression summary.",
                level=TraceLevel.INFO,
                payload=HistorySummaryDiagV1(
                    base_history_length=int(compression_result.raw_history_messages),
                    history_length=int(len(compression_result.history)),
                    history_tokens=compression_result.raw_history_tokens,
                ),
            )
            return

        # Reserve a portion of the input budget for non-history input
        # (system instructions, memory, RAG/websearch/tools context).
        # The remaining portion becomes the token budget for history.
        reserved_for_meta = input_budget // 3  # ~1/3 for meta context
        if reserved_for_meta < 0:
            reserved_for_meta = 0
        if reserved_for_meta >= input_budget:
            reserved_for_meta = input_budget // 2

        history_budget_tokens = input_budget - reserved_for_meta

        # 5. Apply history compression strategy.
        compression_result = self._compress_history(
            request=request,
            raw_history=raw_history,
            raw_token_count=raw_token_count,
            strategy=strategy,
            history_budget_tokens=history_budget_tokens,
            run_id=state.run_id,
        )

        state.base_history = compression_result.history

        # 6. Update debug trace with history-related info and token stats.
        
        state.trace_event(
                component=TraceComponent.ENGINE,
                step="history",
                message="History compression summary.",
                level=TraceLevel.INFO,
                payload=HistorySummaryDiagV1(
                    base_history_length=int(compression_result.raw_history_messages),
                    history_length=int(len(compression_result.history)),
                    history_tokens=compression_result.raw_history_tokens,
                ),
            )


    async def _build_chat_history(self, session: ChatSession) -> List[ChatMessage]:
        """
        Load raw conversation history for the given session.

        This method is responsible only for fetching history from SessionStore.
        Any model-specific preprocessing (truncation, summarization, token
        accounting) should happen in `_step_build_base_history`, not here.
        """
        return await self._session_manager.get_history(session_id=session.id)


    def _count_tokens_for_messages(self, messages: List[ChatMessage]) -> Optional[int]:
        """
        Best-effort token counting for a list of ChatMessage objects.

        Design:
          - Delegates to the underlying LLM adapter if it exposes a
            `count_messages_tokens` method.
          - Returns None if no token counter is available or an error occurs.

        Note:
          - We deliberately avoid any dynamic attribute lookup (no getattr),
            to keep the integration surface with the adapter explicit and
            stable.
        """
        adapter = self._config.llm_adapter
        if adapter is None:
            return None

        try:
            # The adapter is expected to implement this method.
            return int(adapter.count_messages_tokens(messages))
        except AttributeError:
            # Adapter does not implement token counting – leave it as None.
            return None
        except Exception:
            # Any other error should not break the runtime; we just skip
            # token accounting in this case.
            return None


    def _truncate_history_by_tokens(
        self,
        messages: List[ChatMessage],
        max_tokens: int,
    ) -> List[ChatMessage]:
        """
        Truncate conversation history to fit within a token budget.

        Strategy:
        - Keep the most recent messages (suffix of the history).
        - Walk the history from the end backwards and accumulate messages
            until the token budget is exhausted.
        - If token counting is not available, this method returns the
            input list unchanged.

        Important:
        - This helper is intentionally conservative; it does NOT attempt to
            summarize older messages, it only drops them.
        - Summarization-based compression is implemented on top of this
            function in `_compress_history`.
        """
        # Degenerate cases – no budget or empty history.
        if max_tokens is None or max_tokens <= 0:
            return []

        if not messages:
            return []

        # If we cannot count tokens at all, we cannot safely truncate.
        # In that case we keep the history as-is and let the caller decide
        # how to handle potential context overflow.
        if self._count_tokens_for_messages(messages) is None:
            return messages

        truncated: List[ChatMessage] = []

        # Walk from the end (most recent) to the beginning.
        # We always keep a suffix of the conversation in chronological order.
        for msg in reversed(messages):
            # Candidate history if we include this message at the front
            # of the already-kept suffix.
            candidate = [msg] + truncated
            candidate_tokens = self._count_tokens_for_messages(candidate)
            if candidate_tokens is None:
                # If counting suddenly fails, bail out and keep what we have.
                break

            if candidate_tokens > max_tokens:
                # Adding this message would exceed the budget; stop here.
                break

            # Safe to include – prepend to keep chronological order.
            truncated.insert(0, msg)

        # If we ended up with an empty truncated list (e.g. a single message
        # already exceeds the budget), we at least keep the last message.
        if not truncated and messages:
            truncated = [messages[-1]]

        return truncated
    
    
    def _compress_history(
        self,
        *,
        request: RuntimeRequest,
        raw_history: List[ChatMessage],
        raw_token_count: Optional[int],
        strategy: HistoryCompressionStrategy,
        history_budget_tokens: int,
        run_id: Optional[str]=None,
    ) -> HistoryCompressionResult:
        """
        Apply the configured history compression strategy to the raw history
        and return a structured result object with both the final history
        and diagnostic metadata.
        """
        # Defaults for the result – will be updated below.
        effective_strategy = strategy
        truncated = False
        summary_used = False
        summary_tokens_budget = 0
        tail_tokens_budget = 0

        raw_len = len(raw_history)

        # Helper to build the result object in one place.
        def _build_result(history: List[ChatMessage]) -> HistoryCompressionResult:
            return HistoryCompressionResult(
                history=history,
                truncated=truncated,
                effective_strategy=effective_strategy,
                summary_used=summary_used,
                summary_tokens_budget=summary_tokens_budget,
                tail_tokens_budget=tail_tokens_budget,
                raw_history_messages=raw_len,
                raw_history_tokens=raw_token_count,
                history_budget_tokens=history_budget_tokens,
            )

        # 0) OFF → do not touch the history at all.
        if strategy == HistoryCompressionStrategy.OFF:
            effective_strategy = HistoryCompressionStrategy.OFF
            return _build_result(raw_history)

        # 1) If we have no token info or a non-positive budget, we cannot
        # meaningfully compress the history. Keep it as-is.
        if raw_token_count is None or history_budget_tokens <= 0:
            # We keep the requested strategy in effective_strategy for
            # diagnostic purposes, but we do not modify the history.
            return _build_result(raw_history)

        # 2) If history already fits into the budget -> nothing to do.
        if raw_token_count <= history_budget_tokens:
            return _build_result(raw_history)

        # 3) Pure truncation strategy.
        if strategy == HistoryCompressionStrategy.TRUNCATE_OLDEST:
            compressed = self._truncate_history_by_tokens(
                messages=raw_history,
                max_tokens=history_budget_tokens,
            )
            truncated = True
            effective_strategy = HistoryCompressionStrategy.TRUNCATE_OLDEST
            tail_tokens_budget = history_budget_tokens
            return _build_result(compressed)

        # 4) Summarization-based strategies.
        if strategy in (
            HistoryCompressionStrategy.SUMMARIZE_OLDEST,
            HistoryCompressionStrategy.HYBRID,
        ):
            # If the budget is extremely small, summarization will not be
            # very helpful. Fall back to pure truncation.
            if history_budget_tokens <= 64:
                compressed = self._truncate_history_by_tokens(
                    messages=raw_history,
                    max_tokens=history_budget_tokens,
                )
                truncated = True
                effective_strategy = HistoryCompressionStrategy.TRUNCATE_OLDEST
                tail_tokens_budget = history_budget_tokens
                return _build_result(compressed)

            # Basic split of the budget between summary and tail.
            summary_max_tokens = max(
                32,
                min(history_budget_tokens // 4, 256),
            )
            tail_budget = history_budget_tokens - summary_max_tokens

            if tail_budget <= 32:
                tail_budget = max(32, history_budget_tokens // 2)
                summary_max_tokens = history_budget_tokens - tail_budget

            summary_tokens_budget = summary_max_tokens
            tail_tokens_budget = tail_budget

            # Build the most recent tail first.
            tail_messages = self._truncate_history_by_tokens(
                messages=raw_history,
                max_tokens=tail_budget,
            )
            if not tail_messages:
                compressed = self._truncate_history_by_tokens(
                    messages=raw_history,
                    max_tokens=history_budget_tokens,
                )
                truncated = True
                effective_strategy = HistoryCompressionStrategy.TRUNCATE_OLDEST
                tail_tokens_budget = history_budget_tokens
                return _build_result(compressed)

            tail_len = len(tail_messages)
            prefix_len = max(0, len(raw_history) - tail_len)
            older_messages = raw_history[:prefix_len]

            if not older_messages:
                # Nothing older to summarize; we effectively behave like pure
                # truncation here, but we still mark the requested strategy.
                truncated = True
                effective_strategy = strategy
                return _build_result(tail_messages)

            prompt_bundle = self._history_prompt_builder.build_history_summary_prompt(
                request=request,
                strategy=strategy,
                older_messages=older_messages,
                tail_messages=tail_messages,
            )

            summary_msg = self._summarize_history_chunk(
                messages=older_messages,
                max_summary_tokens=summary_max_tokens,
                system_prompt=prompt_bundle.system_prompt,
                run_id=run_id,
            )

            if summary_msg is None:
                # Summarization failed; fall back to truncation.
                compressed = self._truncate_history_by_tokens(
                    messages=raw_history,
                    max_tokens=history_budget_tokens,
                )
                truncated = True
                effective_strategy = HistoryCompressionStrategy.TRUNCATE_OLDEST
                tail_tokens_budget = history_budget_tokens
                summary_tokens_budget = 0
                return _build_result(compressed)

            compressed_history: List[ChatMessage] = [summary_msg]
            compressed_history.extend(tail_messages)

            truncated = True
            summary_used = True
            effective_strategy = strategy

            return _build_result(compressed_history)

        # 5) Unknown strategy -> keep as-is.
        return _build_result(raw_history)



    
    def _summarize_history_chunk(
        self,
        messages: List[ChatMessage],
        max_summary_tokens: int,
        system_prompt: str,
        run_id: Optional[str] = None,
    ) -> Optional[ChatMessage]:
        """
        Summarize a block of older conversation history into a single
        compact system-level message.

        This helper uses the core LLM adapter synchronously. The summary
        message is NOT persisted in the SessionStore; it is meant to be
        injected into the prompt as a synthetic meta-history.
        """
        if not messages:
            return None

        if max_summary_tokens <= 0:
            return None

        adapter = self._config.llm_adapter
        if adapter is None:
            return None

        # Build a simple, robust summarization prompt.
        summary_prompt: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content=system_prompt,
            )
        ]
        summary_prompt.extend(messages)

        generate_kwargs: Dict[str, Any] = {}
        # We keep the summary small and controlled by a separate token budget.
        if max_summary_tokens > 0:
            generate_kwargs["max_tokens"] = max_summary_tokens

        try:
            raw = adapter.generate_messages(
                summary_prompt, 
                run_id=run_id,
                **generate_kwargs)
        except Exception:
            # If summarization fails for any reason, we simply return None
            # and let the caller fall back to truncation.
            return None
        
        if not isinstance(raw, str):
            return None

        if not raw:
            return None
        
        text = raw.strip()

        # We wrap the summary in a system message so that it is clearly
        # separated from user/assistant turns.
        return ChatMessage(
            role="system",
            content=f"Conversation summary (earlier turns):\n{text}",
        )