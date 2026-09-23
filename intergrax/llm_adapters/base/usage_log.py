# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider, llm_provider_slug

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


@dataclass
class LLMRunStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0


@runtime_checkable
class LLMRunStatsReader(Protocol):
    """Internal: per-adapter run-level usage snapshot access (not execution ABI)."""

    def get_run_stats(self, run_id: Optional[str] = None) -> LLMRunStats | None:
        ...


@runtime_checkable
class LLMUsageTrackable(Protocol):
    """Internal: adapter identity + usage stats source for ``LLMUsageTracker`` registration."""

    provider: LLMProvider | str
    model: str
    usage: LLMRunStatsReader


def require_llm_usage_trackable(candidate: object) -> LLMUsageTrackable:
    """Runtime guard for ``LLMUsageTrackable`` before usage-tracker registration."""
    if not isinstance(candidate, LLMUsageTrackable):
        raise TypeError(
            "LLMUsageTracker.register_adapter requires LLMUsageTrackable "
            "(provider, model, and usage stats reader)"
        )
    return candidate


class LLMAdapterUsageLog(LLMRunStatsReader):
    def __init__(self) -> None:
        self._run_stats: Dict[str, LLMRunStats] = {}

    def begin_call(
        self,
        run_id: Optional[str] = None,
        *,
        adapter: Optional[LLMAdapter] = None,
    ) -> LLMCallStats:
        """
        Begin one LLM call (not the whole runtime.run()).

        Returns a per-call context object, safe for nested/parallel use
        because it is local to the caller.

        When ``adapter`` is passed, provider/model are attached for observability metrics.
        """
        from intergrax.runtime.execution.budget.consumption import consume_llm_call

        consume_llm_call()
        rid = run_id or "general"
        if rid not in self._run_stats:
            self._run_stats[rid] = LLMRunStats()
        call = LLMCallStats(run_id=rid)
        if adapter is not None:
            call.provider = llm_provider_slug(adapter.provider)
            call.model = str(adapter.model or "")
        return call

    def end_call(
        self,
        call: LLMCallStats,
        *,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Finish one LLM call and aggregate into per-run stats.
        """
        from intergrax.runtime.execution.budget.consumption import consume_llm_token_usage

        dt_ms = int((time.perf_counter() - call.t0) * 1000)

        call.input_tokens = int(input_tokens or 0)
        call.output_tokens = int(output_tokens or 0)
        call.total_tokens = call.input_tokens + call.output_tokens
        call.duration_ms = dt_ms

        call.success = bool(success)
        call.error_type = error_type

        consume_llm_token_usage(
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            total_tokens=call.total_tokens,
        )

        st = self._run_stats.get(call.run_id)
        if st is None:
            st = LLMRunStats()
            self._run_stats[call.run_id] = st

        st.calls += 1
        st.input_tokens += call.input_tokens
        st.output_tokens += call.output_tokens
        st.total_tokens += call.total_tokens
        st.duration_ms += call.duration_ms

        if not call.success:
            st.errors += 1

        if call.provider:
            from intergrax.llm_adapters.tracking.metrics import record_llm_call

            record_llm_call(
                provider=call.provider,
                model=call.model or "",
                run_id=call.run_id,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                duration_ms=call.duration_ms,
                success=call.success,
                error_type=call.error_type,
            )

    def get_run_stats(self, run_id: Optional[str] = None) -> LLMRunStats | None:
        """
        Get aggregated stats for a given run_id.
        Returns None if no stats exist for that run_id.
        """
        rid = run_id or "general"
        st = self._run_stats.get(rid)

        if st is None:
            return LLMRunStats(
                calls=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                duration_ms=0,
                errors=0,
            )

        return LLMRunStats(
            calls=st.calls,
            input_tokens=st.input_tokens,
            output_tokens=st.output_tokens,
            total_tokens=st.total_tokens,
            duration_ms=st.duration_ms,
            errors=st.errors,
        )

    def get_all_run_stats(self) -> Dict[str, LLMRunStats]:
        """
        Get a shallow copy of all aggregated run stats.
        """
        return dict(self._run_stats)

    def reset_run_stats(self, run_id: Optional[str] = None) -> None:
        """
        Reset stats for a specific run_id (or 'general' if None).
        """
        if run_id is None:
            self._run_stats.clear()
            return

        rid = run_id or "general"
        self._run_stats.pop(rid, None)

    def export_run_stats_dict(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export aggregated stats to a JSON-serializable dict.
        Helpful for trace / logging.
        """
        rid = run_id or "general"
        st = self._run_stats.get(rid)
        if st is None:
            return {
                "run_id": rid,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "duration_ms": 0,
                "errors": 0,
            }

        return {
            "run_id": rid,
            "calls": int(st.calls),
            "input_tokens": int(st.input_tokens),
            "output_tokens": int(st.output_tokens),
            "total_tokens": int(st.total_tokens),
            "duration_ms": int(st.duration_ms),
            "errors": int(st.errors),
        }


@dataclass
class LLMCallStats:
    run_id: str
    t0: float = field(default_factory=time.perf_counter)

    # filled on end
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0

    success: bool = True
    error_type: Optional[str] = None

    # observability (set when begin_call(..., adapter=self))
    provider: str = ""
    model: str = ""


__all__ = [
    "LLMAdapterUsageLog",
    "LLMCallStats",
    "LLMRunStats",
    "LLMRunStatsReader",
    "LLMUsageTrackable",
    "require_llm_usage_trackable",
]
