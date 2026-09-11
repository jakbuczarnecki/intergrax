# © Artur Czarnecki. All rights reserved.

"""Canonical run index registry with fail-closed duplicate protection."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.local_qualification_session.contracts import CanonicalRunRecord


class RunRegistryError(RuntimeError):
    """Invalid canonical run registration."""


@dataclass
class RunRegistry:
    planned_run_count: int
    _by_index: dict[int, CanonicalRunRecord]
    _run_id_to_index: dict[str, int]

    def __init__(self, planned_run_count: int) -> None:
        self.planned_run_count = planned_run_count
        self._by_index = {}
        self._run_id_to_index = {}

    def register(self, record: CanonicalRunRecord) -> None:
        if record.run_index < 0 or record.run_index >= self.planned_run_count:
            raise RunRegistryError(
                f"run index out of range: {record.run_index} (planned={self.planned_run_count})"
            )
        if record.run_index in self._by_index:
            raise RunRegistryError(f"duplicate run index: {record.run_index}")
        if record.run_id in self._run_id_to_index:
            raise RunRegistryError(f"duplicate run id: {record.run_id}")
        self._by_index[record.run_index] = record
        self._run_id_to_index[record.run_id] = record.run_index

    def completed_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_index))

    def records(self) -> tuple[CanonicalRunRecord, ...]:
        return tuple(self._by_index[index] for index in sorted(self._by_index))

    def is_complete(self) -> bool:
        return len(self._by_index) == self.planned_run_count
