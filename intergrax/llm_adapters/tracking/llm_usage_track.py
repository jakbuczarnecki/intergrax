# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from intergrax.llm_adapters.base.usage_log import require_llm_usage_trackable
from intergrax.llm_adapters.contracts.llm_usage_stats import (
    LLMRunStats,
    LLMRunStatsReader,
    LLMUsageTrackable,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import llm_provider_slug
from intergrax.llm_adapters.contracts.llm_usage_report import (
    LLMAdapterMeta,
    LLMAdapterUsageEntry,
    LLMUsageReport,
)

__all__ = [
    "LLMAdapterMeta",
    "LLMAdapterUsageEntry",
    "LLMUsageReport",
    "LLMUsageTracker",
]


@dataclass
class _PhysicalUsageSource:
    trackable: LLMUsageTrackable
    adapter_type: str
    provider_slug: str
    model: str
    stats: LLMRunStatsReader


@dataclass
class _LogicalUsageEntry:
    label: str
    provider_slug: str
    model: str
    sources: Dict[int, _PhysicalUsageSource] = field(default_factory=dict)


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
        self._entries: Dict[str, _LogicalUsageEntry] = {}

    @staticmethod
    def _default_label(trackable: LLMUsageTrackable) -> str:
        slug = llm_provider_slug(trackable.provider)
        return f"{slug}:{trackable.model}"

    def _physical_source(
        self,
        verified: LLMUsageTrackable,
    ) -> _PhysicalUsageSource:
        return _PhysicalUsageSource(
            trackable=verified,
            adapter_type=verified.__class__.__name__,
            provider_slug=llm_provider_slug(verified.provider),
            model=str(verified.model or ""),
            stats=verified.usage,
        )

    def register_adapter(
        self,
        trackable: LLMUsageTrackable,
        label: Optional[str] = None,
    ) -> None:
        """
        Register a usage-trackable adapter used during this runtime run.

        Idempotent per (logical label, physical instance): the same object registered
        twice under one label is counted once. Multiple distinct instances under the
        same semantic label aggregate into one logical report entry.
        """
        verified = require_llm_usage_trackable(trackable)

        resolved_label = label or self._default_label(verified)
        instance_id = id(verified)

        logical = self._entries.get(resolved_label)
        if logical is not None and instance_id in logical.sources:
            return

        physical = self._physical_source(verified)

        if logical is None:
            logical = _LogicalUsageEntry(
                label=resolved_label,
                provider_slug=physical.provider_slug,
                model=physical.model,
            )
            self._entries[resolved_label] = logical
        elif (
            physical.provider_slug != logical.provider_slug
            or physical.model != logical.model
        ):
            raise ValueError(
                f"logical usage label '{resolved_label}' already represents "
                f"provider/model {logical.provider_slug}:{logical.model}, "
                f"cannot register physical source with provider/model "
                f"{physical.provider_slug}:{physical.model}"
            )

        logical.sources[instance_id] = physical

    def unregister_adapter(self, adapter: LLMAdapter | LLMUsageTrackable) -> None:
        """
        Unregister an adapter from this runtime run.

        Safe to call multiple times.
        Does nothing if adapter is not registered.
        """
        instance_id = id(adapter)
        labels_to_drop: List[str] = []

        for label, logical in self._entries.items():
            if instance_id not in logical.sources:
                continue
            del logical.sources[instance_id]
            if not logical.sources:
                labels_to_drop.append(label)

        for label in labels_to_drop:
            del self._entries[label]

    def registered_labels(self) -> List[str]:
        return list(self._entries.keys())

    def _snapshot_stats(self, stats: LLMRunStatsReader) -> LLMRunStats:
        st = stats.get_run_stats(self.run_id)
        if st is None:
            return LLMRunStats()
        return st

    def _aggregate_sources(self, sources: Dict[int, _PhysicalUsageSource]) -> LLMRunStats:
        agg = LLMRunStats()
        for source in sources.values():
            st = self._snapshot_stats(source.stats)
            agg.calls += st.calls
            agg.input_tokens += st.input_tokens
            agg.output_tokens += st.output_tokens
            agg.total_tokens += st.total_tokens
            agg.duration_ms += st.duration_ms
            agg.errors += st.errors
        return agg

    def _primary_instance_id(self, logical: _LogicalUsageEntry) -> int:
        return next(iter(logical.sources))

    def _primary_source(self, logical: _LogicalUsageEntry) -> _PhysicalUsageSource:
        return logical.sources[self._primary_instance_id(logical)]

    def _iter_unique_physical_sources(
        self,
    ) -> Iterator[Tuple[int, _PhysicalUsageSource]]:
        seen_ids: set[int] = set()
        for logical in (self._entries or {}).values():
            for instance_id, source in logical.sources.items():
                if instance_id in seen_ids:
                    continue
                seen_ids.add(instance_id)
                yield instance_id, source

    def build_report(self) -> LLMUsageReport:
        entries: List[LLMAdapterUsageEntry] = []

        adapter_instance_ids: Dict[str, int] = {}
        for label, logical in (self._entries or {}).items():
            if logical.sources:
                adapter_instance_ids[label] = self._primary_instance_id(logical)

        for logical in (self._entries or {}).values():
            if not logical.sources:
                continue
            primary = self._primary_source(logical)
            meta = LLMAdapterMeta(
                adapter_type=primary.adapter_type,
                provider=primary.provider_slug,
                model=primary.model,
            )

            st = self._aggregate_sources(logical.sources)

            entries.append(
                LLMAdapterUsageEntry(
                    label=logical.label,
                    meta=meta,
                    stats=st,
                    adapter_instance_id=self._primary_instance_id(logical),
                )
            )

        total = self.total()

        by_provider_model: Dict[str, LLMRunStats] = {}
        for _instance_id, source in self._iter_unique_physical_sources():
            st = self._snapshot_stats(source.stats)
            key = f"{source.provider_slug}:{source.model}"
            agg = by_provider_model.get(key)
            if agg is None:
                by_provider_model[key] = LLMRunStats(
                    calls=st.calls,
                    input_tokens=st.input_tokens,
                    output_tokens=st.output_tokens,
                    total_tokens=st.total_tokens,
                    duration_ms=st.duration_ms,
                    errors=st.errors,
                )
            else:
                agg.calls += st.calls
                agg.input_tokens += st.input_tokens
                agg.output_tokens += st.output_tokens
                agg.total_tokens += st.total_tokens
                agg.duration_ms += st.duration_ms
                agg.errors += st.errors

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
        for _instance_id, source in self._iter_unique_physical_sources():
            st = source.stats.get_run_stats(self.run_id)
            if st is None:
                continue

            agg.calls += st.calls
            agg.input_tokens += st.input_tokens
            agg.output_tokens += st.output_tokens
            agg.total_tokens += st.total_tokens
            agg.duration_ms += st.duration_ms
            agg.errors += st.errors

        return agg
