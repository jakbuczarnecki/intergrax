"""Typed resume boundary derived from build-state (no filesystem I/O)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataPackResumeBoundary:
    """Where a fast resume may continue shard generation."""

    last_ready_ordinal: int | None
    next_shard_ordinal: int | None
    completed_shards: int
    build_complete: bool
