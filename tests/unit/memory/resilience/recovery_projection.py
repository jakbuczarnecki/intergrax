# © Artur Czarnecki. All rights reserved.

"""Test-only projection that fails once then repairs via reconcile contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.memory_temporal import filter_active_memory_entries
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry


@dataclass
class FailOnceRepairableProjection:
    """Fails the first upsert; ``reconcile`` rebuilds missing entries from context."""

    projection_id: str = "fail-once-repairable"
    entries: dict[str, UserProfileMemoryEntry] = field(default_factory=dict)
    fail_next_upsert: bool = True
    repair_count: int = 0
    upsert_attempts: int = 0
    reconcile_calls: int = 0

    async def upsert_memory_entry(self, user_id: str, entry: UserProfileMemoryEntry) -> None:
        _ = user_id
        self.upsert_attempts += 1
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise TimeoutError("projection upsert failed once")
        self.entries[entry.entry_id] = entry

    async def delete_memory_entries(self, entry_ids: Sequence[str]) -> None:
        for entry_id in entry_ids:
            self.entries.pop(entry_id, None)

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        self.reconcile_calls += 1
        if context.profile is None:
            return MemoryProjectionReconciliationResult(
                projection_id=self.projection_id,
                disposition=MemoryProjectionReconciliationDisposition.CONSISTENT,
            )
        active_entries = filter_active_memory_entries(context.profile.memory_entries)
        expected_ids = set(context.authoritative_active_entry_ids)
        changed = False
        for entry in active_entries:
            if entry.entry_id not in expected_ids:
                continue
            if entry.entry_id not in self.entries:
                self.entries[entry.entry_id] = entry
                self.repair_count += 1
                changed = True
        disposition = (
            MemoryProjectionReconciliationDisposition.REPAIRED
            if changed
            else MemoryProjectionReconciliationDisposition.CONSISTENT
        )
        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=disposition,
        )
