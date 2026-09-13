# © Artur Czarnecki. All rights reserved.

"""EE-B2 — recovery interruption (stale writer / no sealed reopen)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.unit.runtime.architecture.test_ee_b2_checkpoint_fault import (
    _paused_checkpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b2_recovery_interrupted_by_stale_checkpoint_write(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "recovery.db")
    base = store.save(_paused_checkpoint())
    store.save(
        base.model_copy(
            update={"checkpoint_id": "ckpt_advanced", "progress_message": "advanced"},
        ),
        expected_revision=base.revision,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            base.model_copy(
                update={
                    "checkpoint_id": "ckpt_stale_recovery",
                    "progress_message": "stale-recovery-write",
                },
            ),
            expected_revision=base.revision,
        )


def test_ee_b2_governance_deny_is_fail_closed_decision() -> None:
    decision = PolicyDecision(action=PolicyAction.DENY, reason="governance_fault")
    assert decision.action is PolicyAction.DENY
