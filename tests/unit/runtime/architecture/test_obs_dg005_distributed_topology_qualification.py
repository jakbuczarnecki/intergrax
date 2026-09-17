# © Artur Czarnecki. All rights reserved.

"""OBS-DG005 — process-isolated distributed topology qualification."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from testing_support.obs_distributed_topology.qualification_harness import (
    load_json_result,
    run_dg005_topology_qualification,
)
from testing_support.obs_distributed_topology.archive_source import (
    resolve_intergrax_import_root,
)
from testing_support.obs_distributed_topology.scenario_builder import (
    build_dg005_scenario,
)
from testing_support.obs_distributed_topology.scenario_io import (
    read_scenario,
    write_scenario,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECONSTRUCTION_PKG = (
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "reconstruction"
)
_DIAG_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
_SESSION_ROOT = _REPO_ROOT / ".tmp" / "session" / "OBS-DG005-DISTRIBUTED-TOPOLOGY"


def _git_head_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _python_files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


@pytest.mark.gate
def test_dg005_reconstruction_package_does_not_import_runtime_event_bus() -> None:
    forbidden = (
        "intergrax.runtime.events.event_bus",
        "intergrax.runtime.events.runtime_event_history",
        "intergrax.contracts.runtime_event_history",
    )
    for path in _python_files_under(_RECONSTRUCTION_PKG):
        imports = _module_imports(path)
        for module in forbidden:
            assert module not in imports, f"{path.name} imports {module}"
        text = path.read_text(encoding="utf-8")
        assert "RuntimeEventBus" not in text
        assert "RuntimeEventHistoryBuffer" not in text
        assert "RuntimeEventHistoryStrategy" not in text


@pytest.mark.gate
def test_dg005_diagnostics_modules_do_not_import_sqlite_runtime_store() -> None:
    forbidden_suffixes = (
        "sqlite_runtime_event_store",
        "stores.sqlite",
        "evidence_persistence_adapter",
    )
    for path in _python_files_under(_DIAG_ROOT):
        text = path.read_text(encoding="utf-8")
        for suffix in forbidden_suffixes:
            assert suffix not in text, path.relative_to(_REPO_ROOT)


@pytest.mark.gate
def test_dg005_process_isolated_topology_qualification(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    qualification_sha = _git_head_sha()
    work_dir = tmp_path_factory.mktemp("dg005")
    report = run_dg005_topology_qualification(
        _REPO_ROOT,
        work_dir,
        qualification_sha=qualification_sha,
    )

    writer = load_json_result(report.writer_result_path)
    reader = load_json_result(report.reader_result_path)
    diagnostics = load_json_result(report.diagnostics_result_path)
    idempotent = load_json_result(report.idempotent_result_path)

    assert writer["qualification_sha"] == qualification_sha
    assert reader["qualification_sha"] == qualification_sha
    assert diagnostics["qualification_sha"] == qualification_sha
    assert idempotent["qualification_sha"] == qualification_sha

    for payload in (writer, reader, diagnostics, idempotent):
        import_root = Path(str(payload["import_root"])).resolve()
        archive_root = resolve_intergrax_import_root(report.archive_root)
        assert import_root == archive_root
        intergrax_file = Path(str(payload["intergrax_file"]))
        assert str(intergrax_file).startswith(str(archive_root))

    assert writer["history_len_after_writes"] == 0
    assert writer["writer_provider_object_id"] != reader["reader_provider_object_id"]

    scenario = read_scenario(work_dir / "scenario.json")
    expected_run_ids = [str(e.event_id) for e in scenario.primary_events] + [
        str(scenario.idempotent_event.event_id)
    ]

    assert reader["runtime_history_completeness"] == "complete"
    event_ids_in_order = reader["event_ids_in_order"]
    positions_in_order = reader["positions_in_order"]
    as_of_event_ids = reader["as_of_event_ids"]
    isolated_run_event_ids = reader["isolated_run_event_ids"]
    task_grouped_run_ids = reader["task_grouped_run_ids"]
    grouping_candidates = diagnostics["grouping_candidates"]
    assert isinstance(event_ids_in_order, list)
    assert isinstance(positions_in_order, list)
    assert isinstance(as_of_event_ids, list)
    assert isinstance(isolated_run_event_ids, list)
    assert isinstance(task_grouped_run_ids, list)
    assert isinstance(grouping_candidates, int)
    assert event_ids_in_order == expected_run_ids
    assert positions_in_order == list(range(1, len(expected_run_ids) + 1))
    assert len(as_of_event_ids) == scenario.as_of_position_index
    assert reader["foreign_tenant_visible_count"] == 0
    assert set(isolated_run_event_ids) == {
        str(e.event_id) for e in scenario.isolated_run_events
    }
    assert str(scenario.primary_run_id) in task_grouped_run_ids
    assert str(scenario.isolated_run_id) in task_grouped_run_ids
    assert idempotent["listed_count"] == len(expected_run_ids)
    assert diagnostics["execution_analyses"] == 1
    assert grouping_candidates >= 1


@pytest.mark.gate
def test_dg005_empty_backend_sees_no_writer_evidence(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    qualification_sha = _git_head_sha()
    work_dir = tmp_path_factory.mktemp("dg005-empty")
    report = run_dg005_topology_qualification(
        _REPO_ROOT,
        work_dir / "seed",
        qualification_sha=qualification_sha,
    )
    writer = load_json_result(report.writer_result_path)
    empty_db = work_dir / "empty.db"
    scenario = build_dg005_scenario(
        qualification_sha=qualification_sha,
        sqlite_db_path=empty_db,
    )
    scenario_path = work_dir / "empty-scenario.json"
    write_scenario(scenario_path, scenario)
    empty_result = work_dir / "empty-reader.json"
    completed = subprocess.run(
        [
            __import__("sys").executable,
            "-m",
            "testing_support.obs_distributed_topology.worker_cli",
            "reader",
            str(scenario_path),
            str(empty_result),
        ],
        cwd=_REPO_ROOT,
        env=_child_env_for_archive(report.archive_root, qualification_sha),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    empty_reader = load_json_result(empty_result)
    assert empty_reader["event_ids_in_order"] in ([], ())
    assert empty_reader["idempotent_run_count"] == 0
    primary_summaries = writer["primary_summaries"]
    assert isinstance(primary_summaries, list)
    assert len(primary_summaries) > 0


def _child_env_for_archive(
    archive_root: Path, qualification_sha: str
) -> dict[str, str]:
    from testing_support.obs_distributed_topology.qualification_harness import (
        _child_env,
    )

    return _child_env(archive_root, qualification_sha)
