#!/usr/bin/env python3
"""Deterministic EQO EC-W3-11 evidence runner and mechanical reconciliation emitter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[7]
_EC08_REL = (
    "docs/project/maintainers/qualification/enterprise_quality_oversight/"
    "eqo_closure/runs/pre_g2_qualification_execution_08_245d4717"
)
EC08 = REPO_ROOT / _EC08_REL


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _uv() -> str:
    p = shutil.which("uv")
    if p:
        return p
    home = Path.home() / ".local" / "bin" / "uv.exe"
    if home.is_file():
        return str(home)
    raise RuntimeError("uv not in PATH")


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    command: str
    exit_code: int
    junit_path: Path
    log_path: Path
    started_at: str
    finished_at: str


def _parse_junit(path: Path) -> dict[str, dict[str, int]]:
    tree = ET.parse(path)
    root = tree.getroot()
    per_file: dict[str, dict[str, int]] = {}
    for case in root.iter("testcase"):
        cls = case.get("classname") or ""
        name = case.get("name") or ""
        file_key = cls.replace(".", "/") + ".py" if cls else "unknown"
        stats = per_file.setdefault(
            file_key,
            {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0},
        )
        if case.find("failure") is not None:
            stats["failed"] += 1
        elif case.find("error") is not None:
            stats["errors"] += 1
        elif case.find("skipped") is not None:
            stats["skipped"] += 1
        else:
            stats["passed"] += 1
        _ = name
    return per_file


def run_batches() -> list[BatchResult]:
    prereg = json.loads((RUN_ROOT / "DES_PREREG.json").read_text(encoding="utf-8"))
    uv = _uv()
    logs = RUN_ROOT / "logs"
    junit = RUN_ROOT / "junit"
    logs.mkdir(parents=True, exist_ok=True)
    junit.mkdir(parents=True, exist_ok=True)
    results: list[BatchResult] = []
    for batch in prereg["batches"]:
        batch_id = batch["id"]
        paths = batch["paths"]
        junit_path = junit / f"{batch_id}.xml"
        log_path = logs / f"{batch_id}.log"
        started = datetime.now(UTC).isoformat()
        cmd = [
            uv,
            "run",
            "pytest",
            *paths,
            "-vv",
            "-rA",
            "--tb=short",
            f"--junitxml={junit_path}",
        ]
        command_line = " ".join(cmd)
        child_env = os.environ.copy()
        child_env.setdefault("INTERGRAX_INBOUND_VERIFIER", "slack")
        child_env.setdefault("SLACK_SIGNING_SECRET", "eqo-w3-11-qualification-slack-signing-secret")
        child_env.setdefault("PYTHONUTF8", "1")
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
        finished = datetime.now(UTC).isoformat()
        results.append(
            BatchResult(
                batch_id=batch_id,
                command=command_line,
                exit_code=proc.returncode,
                junit_path=junit_path,
                log_path=log_path,
                started_at=started,
                finished_at=finished,
            )
        )
    return results


def write_environment_manifest() -> None:
    uv_path = _uv()
    payload = {
        "schema": "eqo.environment_manifest.v1",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "uv_path": uv_path,
        "uv_sha256": _sha256_file(Path(uv_path)) if Path(uv_path).is_file() else None,
        "python_executable": sys.executable,
        "shutil_which_uv": shutil.which("uv"),
        "source_sha": _git(["rev-parse", "HEAD"]),
        "source_tree_sha": _git(["rev-parse", "HEAD^{tree}"]),
    }
    (RUN_ROOT / "ENVIRONMENT_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_baseline_binding() -> None:
    payload = {
        "schema": "eqo.baseline_binding.v1",
        "required_ancestor": "2944320dad50398a56c738769daeed6089fdf947",
        "previous_implementation_sha": "8b78d592e85c5c53036aeeadc4ee1957797251de",
        "previous_implementation_tree_sha": "192be7b15e032e587ae41b46c6ce73ba957dbe19",
        "task_start_remote_sha": _git(["rev-parse", "origin/development"]),
        "worktree_branch": _git(["symbolic-ref", "--short", "HEAD"]),
        "recorded_utc": datetime.now(UTC).isoformat(),
    }
    (RUN_ROOT / "BASELINE_BINDING.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_matrices(batch_results: list[BatchResult]) -> None:
    source_sha = _git(["rev-parse", "HEAD"])
    file_matrix: dict[str, Any] = {
        "schema": "eqo.file_level_execution_matrix.v1",
        "source_sha": source_sha,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "files": {},
        "batches": [],
    }
    for br in batch_results:
        junit_stats = _parse_junit(br.junit_path) if br.junit_path.is_file() else {}
        file_matrix["batches"].append(
            {
                "batch_id": br.batch_id,
                "command": br.command,
                "exit_code": br.exit_code,
                "log_path": str(br.log_path.relative_to(RUN_ROOT)),
                "junit_path": str(br.junit_path.relative_to(RUN_ROOT)),
                "junit_sha256": _sha256_file(br.junit_path) if br.junit_path.is_file() else None,
                "started_at": br.started_at,
                "finished_at": br.finished_at,
            }
        )
        for rel_path, stats in junit_stats.items():
            norm = rel_path
            for p in (RUN_ROOT / "DES_PREREG.json").read_text():
                pass
            entry = file_matrix["files"].setdefault(
                norm,
                {
                    "obligation_ids": [],
                    "current_batch_outcomes": [],
                    "aggregate": {
                        "passed": 0,
                        "failed": 0,
                        "errors": 0,
                        "skipped": 0,
                        "xfailed": 0,
                    },
                    "evidence_validity": "CURRENT",
                },
            )
            for k, v in stats.items():
                entry["aggregate"][k] = entry["aggregate"].get(k, 0) + v
            entry["current_batch_outcomes"].append({"batch_id": br.batch_id, **stats, "exit_code": br.exit_code})

    # Map prereg paths explicitly
    prereg = json.loads((RUN_ROOT / "DES_PREREG.json").read_text(encoding="utf-8"))
    for batch in prereg["batches"]:
        for path in batch["paths"]:
            file_matrix["files"].setdefault(
                path,
                {
                    "obligation_ids": [],
                    "current_batch_outcomes": [],
                    "aggregate": {
                        "passed": 0,
                        "failed": 0,
                        "errors": 0,
                        "skipped": 0,
                        "xfailed": 0,
                    },
                    "evidence_validity": "CURRENT" if batch["id"] else "NOT_EXECUTED",
                },
            )

    obligation_src = json.loads((EC08 / "OBLIGATION_EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    obligations = obligation_src["obligations"]
    if len(obligations) != 34:
        raise SystemExit(f"expected 34 obligations, got {len(obligations)}")
    obligation_matrix = {
        "schema": "eqo.obligation_execution_matrix.v1",
        "source_sha": source_sha,
        "obligation_count": 34,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "reconciled": False,
        "obligations": [],
    }
    for ob in obligations:
        updated = dict(ob)
        updated["current_execution_evidence"] = str(RUN_ROOT.relative_to(REPO_ROOT).as_posix())
        updated["execution_disposition"] = ob.get("execution_disposition", "OPEN")
        obligation_matrix["obligations"].append(updated)

    queue_src = json.loads((EC08 / "REMAINING_QUALIFICATION_QUEUE.json").read_text(encoding="utf-8"))
    queue_items: list[dict[str, Any]] = []
    for q in queue_src["qualification_queue"]:
        p = q["path"]
        queue_items.append(
            {
                **q,
                "source_sha": source_sha,
                "current_result": file_matrix["files"].get(p, {}),
                "evidence_reference": f"{RUN_ROOT.name}/",
            }
        )
    for hq in queue_src.get("application_host_queue", []):
        p = hq["path"]
        queue_items.append(
            {
                **hq,
                "source_sha": source_sha,
                "current_result": file_matrix["files"].get(p, {}),
                "evidence_reference": f"{RUN_ROOT.name}/",
            }
        )
    if len(queue_items) != 31:
        raise SystemExit(f"expected 31 queue paths, got {len(queue_items)}")
    remaining_queue = {
        "schema": "eqo.remaining_qualification_queue.v1",
        "source_sha": source_sha,
        "qualification_test_file_count": 29,
        "application_host_test_file_count": 2,
        "queue_count": 31,
        "qualification_queue": queue_items,
        "formal_closure_claim": False,
    }

    (RUN_ROOT / "FILE_LEVEL_EXECUTION_MATRIX.json").write_text(
        json.dumps(file_matrix, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (RUN_ROOT / "OBLIGATION_EXECUTION_MATRIX.json").write_text(
        json.dumps(obligation_matrix, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (RUN_ROOT / "REMAINING_QUALIFICATION_QUEUE.json").write_text(
        json.dumps(remaining_queue, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_completed_batches() -> list[BatchResult]:
    prereg = json.loads((RUN_ROOT / "DES_PREREG.json").read_text(encoding="utf-8"))
    results: list[BatchResult] = []
    for batch in prereg["batches"]:
        batch_id = batch["id"]
        junit_path = RUN_ROOT / "junit" / f"{batch_id}.xml"
        log_path = RUN_ROOT / "logs" / f"{batch_id}.log"
        if not junit_path.is_file():
            continue
        started = datetime.fromtimestamp(junit_path.stat().st_mtime, tz=UTC).isoformat()
        if log_path.is_file():
            finished = datetime.fromtimestamp(log_path.stat().st_mtime, tz=UTC).isoformat()
        else:
            finished = started
        cmd = f"uv run pytest {' '.join(batch['paths'])}"
        exit_code = 1 if "<failure" in junit_path.read_text(encoding="utf-8") else 0
        if "<error" in junit_path.read_text(encoding="utf-8"):
            exit_code = max(exit_code, 2)
        results.append(
            BatchResult(
                batch_id=batch_id,
                command=cmd,
                exit_code=exit_code,
                junit_path=junit_path,
                log_path=log_path,
                started_at=started,
                finished_at=finished,
            )
        )
    return results


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    write_baseline_binding()
    write_environment_manifest()
    batches = _load_completed_batches() if "--build-only" in sys.argv else run_batches()
    build_matrices(batches)
    run_manifest = {
        "schema": "eqo.run_manifest.v1",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "batch_count": len(batches),
        "batches": [br.__dict__ for br in batches],
    }
    for br in run_manifest["batches"]:
        br["junit_path"] = str(br["junit_path"].relative_to(RUN_ROOT))
        br["log_path"] = str(br["log_path"].relative_to(RUN_ROOT))
    (RUN_ROOT / "RUN_MANIFEST.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
