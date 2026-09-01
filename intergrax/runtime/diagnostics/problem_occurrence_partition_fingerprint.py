# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Partition fingerprint for snapshot-safe occurrence repair (DIAG-ENTERPRISE-2-R5)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_FINGERPRINT_ROW_KEY = "meta:occurrence_partition_fingerprint"
_FIELD_WRITE_GENERATION = "write_generation"
_FIELD_MIN_ROW_KEY = "min_row_key"
_FIELD_MAX_ROW_KEY = "max_row_key"


@dataclass(frozen=True, slots=True)
class ProblemOccurrencePartitionFingerprint:
    """Monotonic partition write fingerprint derived from durable occurrence writes."""

    write_generation: int
    min_row_key: str
    max_row_key: str


@dataclass(frozen=True, slots=True)
class ProblemOccurrenceRepairBoundary:
    """
    Closed repair snapshot boundary captured before a paginated scan.

    Occurrence rows belong to the snapshot when:

    ``min_row_key <= row_key <= terminal_row_key`` in ascending ``row_key`` order
  (descending ``observed_at`` with deterministic occurrence-id tie-break).

    ``write_generation`` must remain unchanged across the scan for the snapshot
    to be authoritative.
    """

    write_generation: int
    min_row_key: str
    terminal_row_key: str


def occurrence_partition_fingerprint_row_key() -> str:
    return _FINGERPRINT_ROW_KEY


def repair_boundary_stable(
    start: ProblemOccurrenceRepairBoundary,
    end: ProblemOccurrenceRepairBoundary,
) -> bool:
    return (
        start.write_generation == end.write_generation
        and start.min_row_key == end.min_row_key
        and start.terminal_row_key == end.terminal_row_key
    )


def repair_boundary_from_fingerprint(
    fingerprint: ProblemOccurrencePartitionFingerprint,
) -> ProblemOccurrenceRepairBoundary:
    return ProblemOccurrenceRepairBoundary(
        write_generation=fingerprint.write_generation,
        min_row_key=fingerprint.min_row_key,
        terminal_row_key=fingerprint.max_row_key,
    )


def encode_occurrence_partition_fingerprint(
    fingerprint: ProblemOccurrencePartitionFingerprint,
) -> dict[str, str | int]:
    return {
        _FIELD_WRITE_GENERATION: fingerprint.write_generation,
        _FIELD_MIN_ROW_KEY: fingerprint.min_row_key,
        _FIELD_MAX_ROW_KEY: fingerprint.max_row_key,
    }


def decode_occurrence_partition_fingerprint(
    payload: Mapping[str, object],
) -> ProblemOccurrencePartitionFingerprint:
    write_generation = payload.get(_FIELD_WRITE_GENERATION)
    min_row_key = payload.get(_FIELD_MIN_ROW_KEY)
    max_row_key = payload.get(_FIELD_MAX_ROW_KEY)
    if type(write_generation) is not int or isinstance(write_generation, bool):
        raise ValueError("occurrence partition fingerprint write_generation invalid")
    if type(min_row_key) is not str or not min_row_key:
        raise ValueError("occurrence partition fingerprint min_row_key invalid")
    if type(max_row_key) is not str or not max_row_key:
        raise ValueError("occurrence partition fingerprint max_row_key invalid")
    if write_generation < 1:
        raise ValueError("occurrence partition fingerprint write_generation invalid")
    return ProblemOccurrencePartitionFingerprint(
        write_generation=write_generation,
        min_row_key=min_row_key,
        max_row_key=max_row_key,
    )


def next_occurrence_partition_fingerprint(
    current: ProblemOccurrencePartitionFingerprint,
    *,
    occurrence_row_key: str,
) -> ProblemOccurrencePartitionFingerprint:
    return ProblemOccurrencePartitionFingerprint(
        write_generation=current.write_generation + 1,
        min_row_key=min(current.min_row_key, occurrence_row_key),
        max_row_key=max(current.max_row_key, occurrence_row_key),
    )


def initial_occurrence_partition_fingerprint(
    *,
    occurrence_row_key: str,
) -> ProblemOccurrencePartitionFingerprint:
    return ProblemOccurrencePartitionFingerprint(
        write_generation=1,
        min_row_key=occurrence_row_key,
        max_row_key=occurrence_row_key,
    )
