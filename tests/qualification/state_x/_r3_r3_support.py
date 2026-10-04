# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R3 shared qualification helpers (not collected by pytest directly)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.models import (
    HumanDecisionRecord,
    HumanResponseVerdict,
    build_human_decision_record,
)
from intergrax.runtime.human.persistence_contract import (
    HumanDecisionPersistence,
    InMemoryHumanDecisionPersistence,
)
from intergrax.runtime.human.store import SQLiteHumanDecisionStore

TENANT_A = "tenant-r3r3-a"
TENANT_B = "tenant-r3r3-b"
SHARED_TASK = "task-r3r3-shared"
DECISION_ID = "hdec_r3r3_collision"


class HumanDecisionStoreFactory(Protocol):
    def __call__(self, tmp_path: Path) -> HumanDecisionPersistence: ...


def inmemory_human_decision_store(_tmp_path: Path) -> HumanDecisionPersistence:
    return InMemoryHumanDecisionPersistence()


def sqlite_human_decision_store(tmp_path: Path) -> HumanDecisionPersistence:
    return SQLiteHumanDecisionStore(db_path=tmp_path / "human-r3r3.db")


HUMAN_DECISION_STORE_FACTORIES: tuple[tuple[str, HumanDecisionStoreFactory], ...] = (
    ("InMemory", inmemory_human_decision_store),
    ("SQLite", sqlite_human_decision_store),
)


def sample_record(
    *,
    decision_id: str = DECISION_ID,
    tenant_id: str = TENANT_A,
    task_id: str = SHARED_TASK,
    verdict: HumanResponseVerdict = HumanResponseVerdict.APPROVE,
    created_at_utc: str = "2026-01-01T00:00:00+00:00",
    notes: str = "",
    run_id: str | None = "run-r3r3",
) -> HumanDecisionRecord:
    return build_human_decision_record(
        task_id=task_id,
        tenant_id=tenant_id,
        approver=local_development_approver_evidence(tenant_id=tenant_id, actor_id="approver-1"),
        verdict=verdict,
        response_text="ok",
        human_request_id="hr-r3r3",
        run_id=run_id,
        notes=notes,
        agent_id="agent-r3r3",
    ).model_copy(
        update={
            "decision_id": decision_id,
            "created_at_utc": created_at_utc,
        }
    )


def human_decision_store_factories() -> tuple[HumanDecisionStoreFactory, ...]:
    return tuple(factory for _, factory in HUMAN_DECISION_STORE_FACTORIES)
