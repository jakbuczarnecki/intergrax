# © Artur Czarnecki. All rights reserved.

"""MP-4R6 — legacy disposition CLI surface must match wired capabilities."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CLI = _REPO_ROOT / "scripts" / "maintenance" / "human_decision_legacy_disposition_cli.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_CLI), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_strategy_choices_exclude_non_wired_strategies() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    help_text = result.stdout
    assert "provenance_recovery" not in help_text
    assert "controlled_archive" not in help_text
    assert "no_data_present" not in help_text
    assert "history_only_quarantine" in help_text
    assert "controlled_delete" in help_text


def test_cli_default_is_quarantine_dry_run(tmp_path) -> None:
    db = tmp_path / "human.db"
    from intergrax.runtime.human.store import SQLiteHumanDecisionStore

    SQLiteHumanDecisionStore(db_path=db)
    result = _run_cli("--db-path", str(db))
    assert result.returncode == 0
    assert "dry_run: True" in result.stdout


def test_cli_apply_alone_does_not_mutate_legacy_row(tmp_path) -> None:
    db = tmp_path / "human.db"
    from intergrax.runtime.human.models import HumanResponseVerdict
    from intergrax.runtime.human.persistence_errors import HumanDecisionApproverProvenanceError
    from intergrax.runtime.human.store import SQLiteHumanDecisionStore

    store = SQLiteHumanDecisionStore(db_path=db)
    with store._connection() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO human_decisions (
                decision_id, task_id, tenant_id, user_id, human_request_id,
                verdict, response_text, escalation_level, escalation_target,
                agent_id, run_id, notes, created_at_utc, approver_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-cli",
                "task",
                "tenant-a",
                "user",
                "",
                HumanResponseVerdict.APPROVE.value,
                "ok",
                0,
                None,
                None,
                "run-1",
                "",
                "2020-01-01T00:00:00+00:00",
                None,
            ),
        )
    result = _run_cli("--db-path", str(db), "--apply")
    assert result.returncode == 0
    assert "dry_run: False" in result.stdout
    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision("legacy-cli", "tenant-a")


def test_cli_controlled_delete_without_allow_delete_fails(tmp_path) -> None:
    db = tmp_path / "human.db"
    from intergrax.runtime.human.store import SQLiteHumanDecisionStore

    SQLiteHumanDecisionStore(db_path=db)
    result = _run_cli(
        "--db-path",
        str(db),
        "--strategy",
        "controlled_delete",
        "--apply",
    )
    assert result.returncode != 0
