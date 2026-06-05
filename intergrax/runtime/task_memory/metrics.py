# © Artur Czarnecki. All rights reserved.

"""In-process memory platform metrics (Phase MEM-OBS.1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryPlatformMetrics:
    memory_reads: int = 0
    memory_writes: int = 0
    retention_violations: int = 0
    ltm_hits: int = 0
    hook_blocks: int = 0

    def record_read(self) -> None:
        self.memory_reads += 1

    def record_write(self) -> None:
        self.memory_writes += 1

    def record_retention_violation(self) -> None:
        self.retention_violations += 1

    def record_ltm_hit(self) -> None:
        self.ltm_hits += 1

    def record_hook_block(self) -> None:
        self.hook_blocks += 1

    def prometheus_lines(self) -> list[str]:
        return [
            f"intergrax_memory_reads_total {self.memory_reads}",
            f"intergrax_memory_writes_total {self.memory_writes}",
            f"intergrax_memory_retention_violations_total {self.retention_violations}",
            f"intergrax_memory_ltm_hits_total {self.ltm_hits}",
            f"intergrax_memory_hook_blocks_total {self.hook_blocks}",
        ]


_GLOBAL_MEMORY_METRICS = MemoryPlatformMetrics()


def memory_platform_metrics() -> MemoryPlatformMetrics:
    return _GLOBAL_MEMORY_METRICS


def reset_memory_platform_metrics_for_tests() -> None:
    global _GLOBAL_MEMORY_METRICS
    _GLOBAL_MEMORY_METRICS = MemoryPlatformMetrics()
