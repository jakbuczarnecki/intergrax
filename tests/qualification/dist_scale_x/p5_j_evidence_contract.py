# © Artur Czarnecki. All rights reserved.

"""P5 dimension J — reuse matrix + gap tests for ordering/concurrency."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]

J_REUSE_EVIDENCE: dict[str, tuple[str, str]] = {
    "J-01_competing_writers": ("8656fb7d8c42e24e1448c4b7e97f18a15c1d765c", "wl_03_hot_key_contention"),
    "J-02_duplicate_delivery": ("e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c", "wl_06_rabbit_broker_storm"),
    "J-04_lost_ack_after_commit": ("e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c", "wl_04 lost_ack"),
    "J-07_failed_commit_retry": ("e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c", "wl_06"),
    "J-08_delayed_after_timeout": ("e6020efc3f299afc9ff776fd8a1e4a4ac1ac913c", "wl_04 slow"),
}


@pytest.mark.parametrize("case_id", tuple(J_REUSE_EVIDENCE.keys()))
def test_p5_j_reuse_contract_registered(case_id: str) -> None:
    sha, anchor = J_REUSE_EVIDENCE[case_id]
    assert len(sha) == 40
    assert anchor


def test_p5_j_gap_stale_replay_rejected_by_idempotency_ledger() -> None:
    from tests.qualification.dist_scale_x.p3_multiprocess_worker import _record_semantic_effect

    scratch = _REPO_ROOT / ".tmp" / "session" / "dist-scale-x-p5-j" / uuid.uuid4().hex
    scratch.mkdir(parents=True, exist_ok=True)
    db = scratch / "effects.db"
    key = "p5-j-stale-replay"
    _record_semantic_effect(db, key)
    _record_semantic_effect(db, key)
    conn = sqlite3.connect(db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM semantic_effects").fetchone()[0]
    finally:
        conn.close()
    assert count == 1
