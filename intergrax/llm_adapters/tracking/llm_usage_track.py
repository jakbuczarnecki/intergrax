# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import time

from intergrax.llm_adapters.base.usage_log import (
    LLMRunStats,
    LLMRunStatsReader,
    LLMUsageTrackable,
    require_llm_usage_trackable,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import llm_provider_slug


@dataclass(frozen=True)
class LLMAdapterMeta:
    adapter_type: str
    provider: str
    model: str


@dataclass(frozen=True)
class LLMAdapterUsageEntry:
    label: str
    meta: LLMAdapterMeta
    stats: LLMRunStats
    adapter_instance_id: int


@dataclass(frozen=True)
class LLMUsageReport:
    run_id: str
    total: LLMRunStats
    entries: List[LLMAdapterUsageEntry]

    # Optional aggregation by (provider, model)
    by_provider_model: Dict[str, LLMRunStats]

    # Debug only: label -> instance_id
    adapter_instance_ids: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def pretty(self) -> str:
        lines: List[str] = []

        t = self.total
        lines.append(f"LLMUsageReport(run_id={self.run_id})")
        lines.append("Total:")
        lines.append(f"  calls        : {t.calls}")
        lines.append(f"  input_tokens : {t.input_tokens}")
        lines.append(f"  output_tokens: {t.output_tokens}")
        lines.append(f"  total_tokens : {t.total_tokens}")
        lines.append(f"  duration_ms  : {t.duration_ms}")
        lines.append(f"  errors       : {t.errors}")

        if self.by_provider_model:
            lines.append("By provider/model:")
            for key, st in self.by_provider_model.items():  # insertion order
                lines.append(
                    f"  - {key}: calls={st.calls} in={st.input_tokens} out={st.output_tokens} "
                    f"total={st.total_tokens} ms={st.duration_ms} err={st.errors}"
                )

        if self.entries:
            lines.append("Entries (registration order):")
            for e in self.entries:  # registration order
                st = e.stats
                meta = e.meta
                lines.append(f"  - {e.label} [{meta.provider}:{meta.model}] ({meta.adapter_type})")
                lines.append(
                    f"      calls={st.calls} in={st.input_tokens} out={st.output_tokens} "
                    f"total={st.total_tokens} ms={st.duration_ms} err={st.errors} "
                    f"instance_id={e.adapter_instance_id}"
                )

        return "\n".join(lines)


@dataclass
class _TrackedLLMUsageEntry:
    label: str
    trackable: LLMUsageTrackable
    adapter_type: str
    provider_slug: str
    model: str
    stats: LLMRunStatsReader


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
        self._entries: Dict[str, _TrackedLLMUsageEntry] = {}

    @staticmethod
    def _default_label(trackable: LLMUsageTrackable) -> str:
        slug = llm_provider_slug(trackable.provider)
        return f"{slug}:{trackable.model}"

    def register_adapter(
        self,
        trackable: LLMUsageTrackable,
        label: Optional[str] = None,
    ) -> None:
        """
        Register a usage-trackable adapter used during this runtime run.
        Idempotent by label.
        """
        verified = require_llm_usage_trackable(trackable)

        resolved_label = label or self._default_label(verified)

        if resolved_label not in self._entries:
            self._entries[resolved_label] = _TrackedLLMUsageEntry(
                label=resolved_label,
                trackable=verified,
                adapter_type=verified.__class__.__name__,
                provider_slug=llm_provider_slug(verified.provider),
                model=str(verified.model or ""),
                stats=verified.usage,
            )

    def unregister_adapter(self, adapter: LLMAdapter | LLMUsageTrackable) -> None:
        """
        Unregister an adapter from this runtime run.

        Safe to call multiple times.
        Does nothing if adapter is not registered.
        """
        to_remove = None

        for label, entry in self._entries.items():
            if entry.trackable is adapter:
                to_remove = label
                break

        if to_remove is not None:
            del self._entries[to_remove]

    def registered_labels(self) -> List[str]:
        return list(self._entries.keys())

    def _snapshot_stats(self, stats: LLMRunStatsReader) -> LLMRunStats:
        st = stats.get_run_stats(self.run_id)
        if st is None:
            return LLMRunStats()
        return st

    def build_report(self) -> LLMUsageReport:
        entries: List[LLMAdapterUsageEntry] = []

        adapter_instance_ids: Dict[str, int] = {}
        for label, entry in (self._entries or {}).items():
            adapter_instance_ids[label] = id(entry.trackable)

        for entry in (self._entries or {}).values():
            meta = LLMAdapterMeta(
                adapter_type=entry.adapter_type,
                provider=entry.provider_slug,
                model=entry.model,
            )

            st = self._snapshot_stats(entry.stats)

            entries.append(
                LLMAdapterUsageEntry(
                    label=entry.label,
                    meta=meta,
                    stats=st,
                    adapter_instance_id=id(entry.trackable),
                )
            )

        total = self.total()

        by_provider_model: Dict[str, LLMRunStats] = {}
        seen_ids = set()
        for e in entries:
            if e.adapter_instance_id in seen_ids:
                continue
            seen_ids.add(e.adapter_instance_id)

            key = f"{e.meta.provider}:{e.meta.model}"
            agg = by_provider_model.get(key)
            if agg is None:
                by_provider_model[key] = LLMRunStats(
                    calls=e.stats.calls,
                    input_tokens=e.stats.input_tokens,
                    output_tokens=e.stats.output_tokens,
                    total_tokens=e.stats.total_tokens,
                    duration_ms=e.stats.duration_ms,
                    errors=e.stats.errors,
                )
            else:
                agg.calls += e.stats.calls
                agg.input_tokens += e.stats.input_tokens
                agg.output_tokens += e.stats.output_tokens
                agg.total_tokens += e.stats.total_tokens
                agg.duration_ms += e.stats.duration_ms
                agg.errors += e.stats.errors

        return LLMUsageReport(
            run_id=self.run_id,
            total=total,
            entries=entries,
            by_provider_model=by_provider_model,
            adapter_instance_ids=adapter_instance_ids,
        )

    def export(self) -> Dict[str, Any]:
        return self.build_report().to_dict()

    def total(self) -> LLMRunStats:
        agg = LLMRunStats()

        seen_ids = set()
        for entry in (self._entries or {}).values():
            ad_id = id(entry.trackable)
            if ad_id in seen_ids:
                continue
            seen_ids.add(ad_id)

            st = entry.stats.get_run_stats(self.run_id)
            if st is None:
                continue

            agg.calls += st.calls
            agg.input_tokens += st.input_tokens
            agg.output_tokens += st.output_tokens
            agg.total_tokens += st.total_tokens
            agg.duration_ms += st.duration_ms
            agg.errors += st.errors

        return agg
