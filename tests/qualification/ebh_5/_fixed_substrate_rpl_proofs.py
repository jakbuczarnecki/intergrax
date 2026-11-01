# © Artur Czarnecki. All rights reserved.

"""Behavioral FIXED-substrate RPL proofs (fail-closed + ratified scope)."""

from __future__ import annotations

import asyncio

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.profile import IntegrationProfile
from tests.qualification.ebh_5._fixed_substrate_registry import (
    FixedSubstrateDecisionStatus,
    fixed_substrate_decisions,
    fixed_substrate_validation_errors,
)

_REPO_DOC = (
    __import__("pathlib").Path(__file__).resolve().parents[3]
    / "docs/project/maintainers/qualification/EBH_7_FIXED_SUBSTRATE_RATIFICATIONS.md"
)


def _substrate_row(concern_id: str):
    for row in fixed_substrate_decisions():
        if concern_id in row.inventory_concern_ids:
            return row
    raise AssertionError(concern_id)


def prove_fixed_substrate_ratification(concern_id: str) -> None:
    assert _REPO_DOC.is_file()
    assert fixed_substrate_validation_errors() == ()
    row = _substrate_row(concern_id)
    assert row.decision_status is FixedSubstrateDecisionStatus.RATIFIED


def prove_fixed_integration_vendor_fail_closed(concern_id: str) -> None:
    prove_fixed_substrate_ratification(concern_id)
    from tests.qualification.ebh_7.test_ebh_7_vendor_rpl_contract_one_pass import (
        test_ebh7_vendor_single_slug_fail_closed_without_profile,
    )

    test_ebh7_vendor_single_slug_fail_closed_without_profile(concern_id)


def prove_fixed_integration_resolve_fail_closed(concern_id: str) -> None:
    clear_catalog()
    suffix = concern_id.removeprefix("integration.")
    category = IntegrationCategory(suffix)
    with pytest.raises(Exception):
        resolve_from_profile(IntegrationProfile(), category)


def prove_memory_sqlite_conversational_fixed_positive(tmp_path) -> None:
    prove_fixed_substrate_ratification("memory.sqlite.conversational")
    from intergrax.llm.messages import ChatMessage
    from intergrax.memory.stores.sqlite_conversational_memory_store import SQLiteConversationalMemoryStore

    store = SQLiteConversationalMemoryStore(db_path=str(tmp_path / "conv.db"))

    async def _roundtrip() -> None:
        await store.append_message(
            tenant_id="tenant-a",
            session_id="session-1",
            message=ChatMessage(role="user", content="hello"),
        )
        memory = await store.load_memory(tenant_id="tenant-a", session_id="session-1")
        messages = memory.get_all()
        assert len(messages) == 1
        assert messages[0].content == "hello"

    asyncio.run(_roundtrip())


def prove_memory_sqlite_conversational_fixed_negative() -> None:
    from intergrax.memory.stores.sqlite_conversational_memory_store import SQLiteConversationalMemoryStore

    with pytest.raises(Exception):
        SQLiteConversationalMemoryStore(db_path="\0bad")
