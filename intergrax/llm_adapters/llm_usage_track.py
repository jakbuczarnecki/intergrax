# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import time

from intergrax.llm_adapters.llm_adapter import LLMAdapter, LLMRunStats


@dataclass
class LLMUsageSnapshot:
    """
    One adapter snapshot for a given run_id.
    """
    adapter_label: str
    provider: Optional[str]
    model_hint: Optional[str]
    stats: LLMRunStats
    

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


    def register_adapter(self, adapter: LLMAdapter, label: Optional[str] = None) -> None:
        """
        Register an adapter used during this runtime run.
        Idempotent by label.
        """
        if adapter is None:
            return
        
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
            if st is None:
                continue
            agg.calls += st.calls
            agg.input_tokens += st.input_tokens
            agg.output_tokens += st.output_tokens
            agg.total_tokens += st.total_tokens
            agg.duration_ms += st.duration_ms
            agg.errors += st.errors
        return agg
    
    
    def export(self) -> Dict[str, Any]:
        """
        Return an immutable snapshot of usage stats for the current run_id.
        Safe to serialize (JSON) and store in debug traces/logs.
        """
        adapters_out: Dict[str, Any] = {}

        for label, ad in self._adapters.items():
            st = ad.usage.get_run_stats(self.run_id)
            if st is None:
                adapters_out[label] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_ms": 0,
                    "errors": 0,
                }
            else:
                adapters_out[label] = asdict(st)

        total = self.total()
        return {
            "run_id": self.run_id,
            "total": asdict(total),
            "adapters": adapters_out,
        }