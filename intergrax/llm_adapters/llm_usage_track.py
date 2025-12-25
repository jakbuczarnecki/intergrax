# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from intergrax.llm_adapters.base import LLMAdapter


@dataclass
class LLMUsageSnapshot:
    """
    One adapter snapshot for a given run_id.
    """
    adapter_label: str
    provider: Optional[str]
    model_hint: Optional[str]
    stats: LLMRunStats


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


@dataclass
class LLMRunStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0
    

class LLMUsageTracker:
    """
    Aggregates usage across multiple adapters used during a single runtime.run().

    Design goals:
      - Engine does NOT count tokens itself.
      - Each adapter counts its own tokens via begin_call/end_call.
      - Tracker just collects snapshots and builds a summary.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._adapters: Dict[str, LLMAdapter] = {}


    def register_adapter(self, adapter: LLMAdapter, label: Optional[str]) -> None:
        """
        Register an adapter used during this runtime run.
        Idempotent by label.
        """
        if not label:
            label = adapter.id
        if label not in self._adapters:
            self._adapters[label] = adapter


    def unregister_adapter(self, adapter: LLMAdapter) -> None:
        """
        Unregister an adapter from this runtime run.

        Safe to call multiple times.
        Does nothing if adapter is not registered.
        """
        to_remove = None

        for label, a in self._adapters.items():
            if a is adapter:
                to_remove = label
                break

        if to_remove is not None:
            del self._adapters[to_remove]
            
        

    def registered_labels(self) -> List[str]:
        return list(self._adapters.keys())


    def snapshot(self) -> List[LLMUsageSnapshot]:
        out: List[LLMUsageSnapshot] = []
        for label, ad in self._adapters.items():
            st = ad.usage.get_run_stats(self.run_id)
            provider = None
            try:
                # Optional: if your adapters expose provider in a strict way, wire it here.
                # Keep None if not guaranteed.
                provider = None
            except Exception:
                provider = None

            model_hint = ad.model_name_for_token_estimation
            out.append(
                LLMUsageSnapshot(
                    adapter_label=label,
                    provider=provider,
                    model_hint=model_hint,
                    stats=st,
                )
            )
        return out

    def total(self) -> LLMRunStats:
        agg = LLMRunStats()
        for _, ad in self._adapters.items():
            st = ad.usage.get_run_stats(self.run_id)
            agg.calls += st.calls
            agg.input_tokens += st.input_tokens
            agg.output_tokens += st.output_tokens
            agg.total_tokens += st.total_tokens
            agg.duration_ms += st.duration_ms
            agg.errors += st.errors
        return agg
    
    
class LLMAdapterUsageLog:

    def __init__(self) -> None:
        self._run_stats: Dict[str, LLMRunStats] = {}


    def begin_call(self, run_id: Optional[str] = None) -> LLMCallStats:
        """
        Begin one LLM call (not the whole runtime.run()).

        Returns a per-call context object, safe for nested/parallel use
        because it is local to the caller.
        """
        rid = run_id or "general"
        if rid not in self._run_stats:
            self._run_stats[rid] = LLMRunStats()
        return LLMCallStats(run_id=rid)


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
        dt_ms = int((time.perf_counter() - call.t0) * 1000)

        call.input_tokens = int(input_tokens or 0)
        call.output_tokens = int(output_tokens or 0)
        call.total_tokens = call.input_tokens + call.output_tokens
        call.duration_ms = dt_ms

        call.success = bool(success)
        call.error_type = error_type

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
    
    def get_run_stats(self, run_id: Optional[str] = None) -> Optional[LLMRunStats]:
        """
        Get aggregated stats for a given run_id.
        Returns None if no stats exist for that run_id.
        """
        rid = run_id or "general"
        return self._run_stats.get(rid)


    def get_all_run_stats(self) -> Dict[str, LLMRunStats]:
        """
        Get a shallow copy of all aggregated run stats.
        """
        return dict(self._run_stats)


    def reset_run_stats(self, run_id: Optional[str] = None) -> None:
        """
        Reset stats for a specific run_id (or 'general' if None).
        """
        rid = run_id or "general"
        if rid in self._run_stats:
            del self._run_stats[rid]


    def reset_all_run_stats(self) -> None:
        """
        Reset all stored stats.
        """
        self._run_stats.clear()


    def export_run_stats_dict(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export aggregated stats to a JSON-serializable dict.
        Helpful for debug_trace / logging.
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