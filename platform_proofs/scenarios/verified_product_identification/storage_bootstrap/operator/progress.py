"""Bounded progress sink for long-running VPI storage load operator runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapProgress,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.progress import (
    BootstrapProgressSinkPort,
)


@dataclass(frozen=True, slots=True)
class OperatorProgressEvent:
    phase: str
    batch_number: int
    records_processed: int
    total_records: int
    percentage: float
    elapsed_seconds: float
    last_committed_global_row_index: int | None
    last_identity: str | None
    throughput_records_per_second: float | None
    eta_seconds: float | None

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "phase": self.phase,
                "batch_number": self.batch_number,
                "records_processed": self.records_processed,
                "total_records": self.total_records,
                "percentage": round(self.percentage, 4),
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "last_committed_global_row_index": self.last_committed_global_row_index,
                "last_identity": self.last_identity,
                "throughput_records_per_second": (
                    round(self.throughput_records_per_second, 3)
                    if self.throughput_records_per_second is not None
                    else None
                ),
                "eta_seconds": (
                    round(self.eta_seconds, 1) if self.eta_seconds is not None else None
                ),
            },
            sort_keys=True,
        )


@dataclass(slots=True)
class JsonlOperatorProgressSink(BootstrapProgressSinkPort):
    progress_path: Path
    _last_committed_global_row_index: int | None = None

    def emit(self, progress: BootstrapProgress) -> None:
        if progress.phase.value == "COMMITTED" and progress.records_processed > 0:
            self._last_committed_global_row_index = progress.records_processed - 1
        percentage = 0.0
        if progress.total_records > 0:
            percentage = 100.0 * progress.records_processed / progress.total_records
        throughput: float | None = None
        eta_seconds: float | None = None
        if progress.elapsed_seconds > 0 and progress.records_processed > 0:
            throughput = progress.records_processed / progress.elapsed_seconds
            remaining = progress.total_records - progress.records_processed
            if remaining > 0 and throughput > 0:
                eta_seconds = remaining / throughput
        event = OperatorProgressEvent(
            phase=progress.phase.value,
            batch_number=progress.batch_number,
            records_processed=progress.records_processed,
            total_records=progress.total_records,
            percentage=percentage,
            elapsed_seconds=progress.elapsed_seconds,
            last_committed_global_row_index=self._last_committed_global_row_index,
            last_identity=progress.last_identity,
            throughput_records_per_second=throughput,
            eta_seconds=eta_seconds,
        )
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(event.to_json_line())
            handle.write("\n")
