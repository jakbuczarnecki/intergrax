# © Artur Czarnecki. All rights reserved.

"""Parent-process driver for OBS-DG005 exact-SHA topology qualification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from testing_support.obs_distributed_topology.archive_source import (
    materialize_git_archive_source_tree,
    resolve_intergrax_import_root,
)
from testing_support.obs_distributed_topology.scenario_builder import (
    build_dg005_scenario,
)
from testing_support.obs_distributed_topology.scenario_io import write_scenario


@dataclass(frozen=True, slots=True)
class Dg005QualificationReport:
    qualification_sha: str
    archive_root: Path
    child_import_root: str
    writer_result_path: Path
    reader_result_path: Path
    diagnostics_result_path: Path
    idempotent_result_path: Path


def _child_env(archive_root: Path, qualification_sha: str) -> dict[str, str]:
    env = dict(os.environ)
    import_root = str(resolve_intergrax_import_root(archive_root))
    env["INTERGRAX_DG005_QUALIFICATION_ROOT"] = import_root
    env["INTERGRAX_DG005_QUALIFICATION_SHA"] = qualification_sha
    separator = ";" if sys.platform.startswith("win") else ":"
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{import_root}{separator}{existing}" if existing else import_root
    )
    return env


def _run_worker(
    *,
    repo_root: Path,
    archive_root: Path,
    qualification_sha: str,
    role: str,
    scenario_path: Path,
    result_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "testing_support.obs_distributed_topology.worker_cli",
            role,
            str(scenario_path),
            str(result_path),
        ],
        cwd=resolve_intergrax_import_root(archive_root),
        env=_child_env(archive_root, qualification_sha),
        capture_output=True,
        text=True,
        check=False,
    )


def run_dg005_topology_qualification(
    repo_root: Path,
    work_dir: Path,
    *,
    qualification_sha: str,
) -> Dg005QualificationReport:
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_root = materialize_git_archive_source_tree(
        repo_root,
        qualification_sha,
        work_dir / "archive",
    )
    sqlite_db = work_dir / "shared-runtime-events.db"
    scenario = build_dg005_scenario(
        qualification_sha=qualification_sha,
        sqlite_db_path=sqlite_db,
    )
    scenario_path = work_dir / "scenario.json"
    write_scenario(scenario_path, scenario)

    writer_result = work_dir / "writer.json"
    idempotent_result = work_dir / "idempotent.json"
    reader_result = work_dir / "reader.json"
    diagnostics_result = work_dir / "diagnostics.json"

    for role, result_path in (
        ("writer", writer_result),
        ("idempotent_retry", idempotent_result),
        ("reader", reader_result),
        ("diagnostics", diagnostics_result),
    ):
        completed = _run_worker(
            repo_root=repo_root,
            archive_root=archive_root,
            qualification_sha=qualification_sha,
            role=role,
            scenario_path=scenario_path,
            result_path=result_path,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"DG005 worker {role} failed: stdout={completed.stdout} stderr={completed.stderr}"
            )
        if not result_path.is_file():
            raise RuntimeError(f"DG005 worker {role} did not write {result_path}")

    child_import_root = str(resolve_intergrax_import_root(archive_root))
    return Dg005QualificationReport(
        qualification_sha=qualification_sha,
        archive_root=archive_root,
        child_import_root=child_import_root,
        writer_result_path=writer_result,
        reader_result_path=reader_result,
        diagnostics_result_path=diagnostics_result,
        idempotent_result_path=idempotent_result,
    )


def load_json_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
