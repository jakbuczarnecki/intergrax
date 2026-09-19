# © Artur Czarnecki. All rights reserved.

"""Typed helpers for behavioral hard-gate assertions (qualification harness only)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.memory.contracts.memory_control import MemoryControlRecallItem
from intergrax.memory.contracts.memory_recall import MemoryRecallReasonCode
from tests.qualification.memory_behavior.contracts import BehaviorViolationLedger


def assert_recall_does_not_expose_entry(
    items: Sequence[MemoryControlRecallItem],
    *,
    forbidden_entry_id: str,
    ledger: BehaviorViolationLedger,
    leak_kind: str,
) -> None:
    for item in items:
        if item.entry_id == forbidden_entry_id:
            if leak_kind == "cross_user":
                ledger.record_cross_user_leak()
            elif leak_kind == "cross_tenant":
                ledger.record_cross_tenant_leak()
            elif leak_kind == "deleted":
                ledger.record_deleted_resurrection()
            else:
                raise ValueError(f"unknown leak_kind: {leak_kind}")
            raise AssertionError(f"forbidden entry_id in recall: {forbidden_entry_id}")


def assert_recall_does_not_expose_content(
    items: Sequence[MemoryControlRecallItem],
    *,
    forbidden_substring: str,
    ledger: BehaviorViolationLedger,
    leak_kind: str,
) -> None:
    for item in items:
        if forbidden_substring in item.content:
            if leak_kind == "cross_user":
                ledger.record_cross_user_leak()
            elif leak_kind == "cross_tenant":
                ledger.record_cross_tenant_leak()
            else:
                raise ValueError(f"unknown leak_kind: {leak_kind}")
            raise AssertionError(f"forbidden content in recall: {forbidden_substring!r}")


def assert_unresolved_conflict_evidence(items: Sequence[MemoryControlRecallItem]) -> None:
    if not any(
        MemoryRecallReasonCode.CONFLICT_UNRESOLVED in item.reason_codes or item.conflict_unresolved
        for item in items
    ):
        raise AssertionError(
            "expected explicit unresolved conflict evidence "
            "(conflict_unresolved or CONFLICT_UNRESOLVED reason code)"
        )


def assert_superseded_not_current_winner(
    items: Sequence[MemoryControlRecallItem],
    *,
    superseded_entry_id: str,
    superseding_entry_id: str,
    ledger: BehaviorViolationLedger,
) -> None:
    recalled_ids = {item.entry_id for item in items}
    if superseded_entry_id in recalled_ids:
        ledger.record_superseded_as_current()
        raise AssertionError(
            f"superseded entry {superseded_entry_id} must not appear in active recall results"
        )
    if superseding_entry_id not in recalled_ids:
        raise AssertionError(f"superseding entry {superseding_entry_id} missing from recall")
    for item in items:
        if item.entry_id == superseding_entry_id and item.conflict_unresolved:
            raise AssertionError("superseding entry must not be marked conflict_unresolved")
