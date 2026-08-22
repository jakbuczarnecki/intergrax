# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.debug.cli import main
from intergrax.experiments.composition import resolve_experiment_persistence
from intergrax.experiments.models import ExperimentDecision

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_debug_cli_experiments_register_list_decide(tmp_path, capsys):
    db_path = tmp_path / "experiments.db"

    assert (
        main(
            [
                "--experiments-db",
                str(db_path),
                "experiments",
                "register",
                "--hypothesis",
                "CLI experiment",
                "--capability",
                "echo.basic",
            ]
        )
        == 0
    )
    experiment_id = capsys.readouterr().out.strip()

    assert (
        main(
            [
                "--experiments-db",
                str(db_path),
                "experiments",
                "link-run",
                experiment_id,
                "run-cli-1",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "--experiments-db",
                str(db_path),
                "experiments",
                "decide",
                experiment_id,
                "--decision",
                ExperimentDecision.KEEP.value,
            ]
        )
        == 0
    )

    store = resolve_experiment_persistence(experiments_db=db_path)
    record = store.get(experiment_id)
    assert record.decision == ExperimentDecision.KEEP
    assert record.run_ids == ["run-cli-1"]

    assert main(["--experiments-db", str(db_path), "experiments", "list"]) == 0
    out = capsys.readouterr().out
    assert experiment_id in out
    assert ExperimentDecision.KEEP.value in out
