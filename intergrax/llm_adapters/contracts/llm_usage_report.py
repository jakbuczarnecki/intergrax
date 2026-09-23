# © Artur Czarnecki. All rights reserved.

"""Canonical LLM usage report DTOs (EBH-2E-R6-R2-R1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from intergrax.llm_adapters.contracts.llm_usage_stats import LLMRunStats


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
    entries: list[LLMAdapterUsageEntry]

    # Optional aggregation by (provider, model)
    by_provider_model: dict[str, LLMRunStats]

    # Debug only: label -> instance_id of first registered physical source
    adapter_instance_ids: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pretty(self) -> str:
        lines: list[str] = []

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


__all__ = [
    "LLMAdapterMeta",
    "LLMAdapterUsageEntry",
    "LLMUsageReport",
]
