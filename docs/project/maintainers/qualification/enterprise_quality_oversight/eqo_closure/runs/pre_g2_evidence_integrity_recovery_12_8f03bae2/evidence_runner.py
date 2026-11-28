#!/usr/bin/env python3
# ruff: noqa: E501
"""EC-W3-12 semantic evidence runner — actual execution records (no ambient P2C verifier injection)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from _evidence_common import (
    BATCH_TIMEOUT_SECONDS,
    Q04_LIGHTWEIGHT_TESTS,
    ec08_dir,
    extract_pytest_paths_from_command,
    git,
    inventory_binding_at_sha,
    resolve_execution_context,
    sha256_bytes,
    sha256_file,
    utc_now,
)

RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[7]
PLUGIN_PATH = RUN_ROOT / "eqo_nodecapture_plugin.py"


def _uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "uv.exe"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("uv executable not found")


def _child_env(uv_exe: str) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    uv_dir = str(Path(uv_exe).resolve().parent)
    path = child_env.get("PATH", "")
    if uv_dir not in path.split(os.pathsep):
        child_env["PATH"] = f"{uv_dir}{os.pathsep}{path}"
    for key in (
        "INTERGRAX_INBOUND_VERIFIER",
        "INTERGRAX_SLACK_SIGNING_SECRET",
        "INTERGRAX_SLACK_VERIFY_SIGNATURE",
    ):
        child_env.pop(key, None)
    child_env["PYTHONPATH"] = os.pathsep.join(p for p in (child_env.get("PYTHONPATH", ""), str(RUN_ROOT)) if p)
    return child_env


def _tracked_consistency(repo_root: Path) -> dict[str, Any]:
    status = git(repo_root, ["status", "--porcelain"])
    return {
        "porcelain_empty": status == "",
        "porcelain_lines": status.splitlines() if status else [],
    }


def _command_digest(argv: list[str]) -> str:
    return hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()


def _write_prereg() -> None:
    ec08_des = json.loads((ec08_dir(REPO_ROOT) / "DES_PREREG.json").read_text(encoding="utf-8"))
    w311 = REPO_ROOT / (
        "docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/"
        "runs/pre_g2_context_recovery_reconciliation_11_2944320d/DES_PREREG.json"
    )
    supplementary = []
    if w311.is_file():
        supplementary = json.loads(w311.read_text(encoding="utf-8")).get("batches", [])
    batches: list[dict[str, Any]] = []
    q04_k = " or ".join(Q04_LIGHTWEIGHT_TESTS)
    for cmd in ec08_des["commands"]:
        cmd_id = cmd["cmd_id"]
        if cmd_id == "Q04":
            batches.append(
                {
                    "id": "q04",
                    "paths": extract_pytest_paths_from_command(cmd["command"]),
                    "pytest_k": q04_k,
                    "deferred_nodes": "DEFERRED_FORMAL_QUALIFICATION_PROFILE",
                    "expected_coverage": cmd_id,
                }
            )
            continue
        if cmd_id.startswith("Q") or cmd_id.startswith("PKG_G_d"):
            batches.append(
                {
                    "id": cmd_id.lower(),
                    "paths": extract_pytest_paths_from_command(cmd["command"]),
                    "expected_coverage": cmd_id,
                }
            )
    supp_ids = {
        "fc1_nexus_context",
        "fc3_inbound_fail_closed",
        "p2c_http_admission",
        "gca_dur_r4",
        "erl_bridge",
        "trace_x_p3_r1",
        "d4_strict_serving",
        "gca_smoke",
        "mcp_p4_lifecycle",
        "bnd03b_witness",
        "xd12_impact",
    }
    for batch in supplementary:
        if batch["id"] in supp_ids:
            entry = dict(batch)
            entry.setdefault("classification", "SUPPLEMENTARY_REGRESSION")
            batches.append(entry)
    if not any(b["id"] == "fc3_inbound_fail_closed" for b in batches):
        batches.append(
            {
                "id": "fc3_inbound_fail_closed",
                "paths": ["tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py"],
                "classification": "SUPPLEMENTARY_REGRESSION",
                "expected_coverage": "FC3 inbound verifier fail-closed (12 controls)",
            }
        )
    payload = {
        "schema": "eqo.mapped_des_prereg.v2",
        "run_id": "ec_w3_12_evidence_integrity_recovery",
        "ticket": "governance/tickets/T-IGX-095-A.md",
        "classification": "PRE_G2_EVIDENCE_INTEGRITY_RECOVERY",
        "inventory_source": "runs/pre_g2_qualification_execution_08_245d4717/OBLIGATION_EXECUTION_MATRIX.json",
        "preregistered_at": utc_now(),
        "source_sha_at_prereg": git(REPO_ROOT, ["rev-parse", "HEAD"]),
        "batches": batches,
    }
    (RUN_ROOT / "DES_PREREG.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _environment_manifest() -> dict[str, Any]:
    pyproject = REPO_ROOT / "pyproject.toml"
    lockfile = REPO_ROOT / "uv.lock"
    manifest = {
        "schema": "eqo.environment_manifest.v1",
        "recorded_utc": utc_now(),
        "python_executable": sys.executable,
        "uv_executable": _uv(),
        "shutil_which_uv": shutil.which("uv"),
        "pyproject_sha256": sha256_file(pyproject),
        "uv_lock_sha256": sha256_file(lockfile),
        "extras_profile": [
            "dev-ci",
            "dev-ci-rag",
            "mcp",
            "llm-langchain-ollama",
            "llm-gemini",
        ],
        "fastmcp_import_ok": False,
    }
    try:
        __import__("fastmcp")
        manifest["fastmcp_import_ok"] = True
    except ImportError:
        manifest["fastmcp_import_ok"] = False
    return manifest


def run_batch(
    batch: dict[str, Any],
    *,
    env_manifest: dict[str, Any],
    source_root: Path,
    qualification_sha: str,
    qualification_tree: str,
) -> dict[str, Any]:
    if batch.get("deferred") and not batch.get("pytest_k"):
        return {
            "batch_id": batch["id"],
            "deferred": batch["deferred"],
            "skipped": True,
        }
    batch_id = batch["id"]
    paths = batch["paths"]
    pre_sha = qualification_sha
    pre_tree = qualification_tree
    worktree = git(source_root, ["rev-parse", "--git-dir"])
    node_report = (RUN_ROOT / "node_reports" / f"{batch_id}_nodes.json").resolve()
    junit_path = RUN_ROOT / "junit" / f"{batch_id}.xml"
    log_path = RUN_ROOT / "logs" / f"{batch_id}.log"
    uv_exe = _uv()
    argv = [
        uv_exe,
        "run",
        "--no-sync",
        "python",
        "-m",
        "pytest",
        *paths,
        "-vv",
        "-rA",
        "--tb=short",
        f"--junitxml={junit_path}",
        "-p",
        "eqo_nodecapture_plugin",
        f"--eqo-node-report={node_report}",
        f"--eqo-source-root={source_root}",
    ]
    if batch.get("pytest_k"):
        argv.extend(["-k", batch["pytest_k"]])
    digest = _command_digest(argv)
    record_pre = {
        "schema": "eqo.execution_record.v1",
        "phase": "pre",
        "batch_id": batch_id,
        "source_commit_sha": pre_sha,
        "source_tree_sha": pre_tree,
        "git_worktree_git_dir": worktree,
        "tracked_source_consistency": _tracked_consistency(REPO_ROOT),
        "argv": argv,
        "command_digest_sha256": digest,
        "environment_profile": env_manifest,
        "started_at_utc": utc_now(),
    }
    rec_dir = RUN_ROOT / "execution_records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "junit").mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "node_reports").mkdir(parents=True, exist_ok=True)
    (rec_dir / f"{batch_id}.pre.json").write_text(json.dumps(record_pre, indent=2), encoding="utf-8")
    child_env = _child_env(uv_exe)
    started = time.monotonic()
    proc: subprocess.CompletedProcess[bytes] | None = None
    timed_out = False
    with log_path.open("wb") as log:
        try:
            proc = subprocess.run(
                argv,
                cwd=source_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env,
                timeout=BATCH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
    elapsed = time.monotonic() - started
    log_bytes = log_path.read_bytes() if log_path.is_file() else b""
    junit_bytes = junit_path.read_bytes() if junit_path.is_file() else b""
    node_bytes = node_report.read_bytes() if node_report.is_file() else b""
    post_sha = git(source_root, ["rev-parse", "HEAD"])
    post_tree = git(source_root, ["rev-parse", "HEAD^{tree}"])
    if timed_out:
        disposition = "EXECUTION_TIMEOUT"
        returncode: int | None = 124
    elif proc is None:
        disposition = "PROCESS_EXIT_UNRESOLVED"
        returncode = None
    else:
        disposition = "PROCESS_COMPLETED"
        returncode = proc.returncode
    record_post = {
        "schema": "eqo.execution_record.v1",
        "phase": "post",
        "batch_id": batch_id,
        "process_returncode": returncode,
        "process_exit_disposition": disposition,
        "finished_at_utc": utc_now(),
        "monotonic_elapsed_seconds": elapsed,
        "stdout_stderr_path": log_path.relative_to(RUN_ROOT).as_posix(),
        "stdout_stderr_sha256": sha256_bytes(log_bytes),
        "junit_path": junit_path.relative_to(RUN_ROOT).as_posix() if junit_path.is_file() else None,
        "junit_sha256": sha256_bytes(junit_bytes) if junit_bytes else None,
        "node_report_path": node_report.relative_to(RUN_ROOT).as_posix() if node_report.is_file() else None,
        "node_report_sha256": sha256_bytes(node_bytes) if node_bytes else None,
        "source_commit_sha_after": post_sha,
        "source_tree_sha_after": post_tree,
        "source_changed_during_batch": post_sha != pre_sha or post_tree != pre_tree,
    }
    (rec_dir / f"{batch_id}.post.json").write_text(json.dumps(record_post, indent=2), encoding="utf-8")
    return {"batch_id": batch_id, "pre": record_pre, "post": record_post}


def write_static_manifests(*, qualification_sha: str, qualification_tree: str, source_root: Path) -> None:
    binding = inventory_binding_at_sha(REPO_ROOT, qualification_sha)
    (RUN_ROOT / "INVENTORY_BINDING.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    env_manifest = _environment_manifest()
    (RUN_ROOT / "ENVIRONMENT_MANIFEST.json").write_text(
        json.dumps(env_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    baseline = {
        "schema": "eqo.baseline_binding.v1",
        "required_ancestor": "4aae44a7647a18da740eaa3b8295352294941b85",
        "task_start_remote_sha": git(REPO_ROOT, ["rev-parse", "origin/development"]),
        "qualification_source_sha": qualification_sha,
        "qualification_source_tree_sha": qualification_tree,
        "source_root": source_root.as_posix(),
        "run_root": RUN_ROOT.as_posix(),
        "recorded_utc": utc_now(),
    }
    (RUN_ROOT / "BASELINE_BINDING.json").write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="EC-W3-12 semantic evidence runner")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--qualification-sha", type=str, default=None)
    parser.add_argument("--write-prereg", action="store_true")
    parser.add_argument("--manifests-only", action="store_true")
    parser.add_argument("--batch", type=str, default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    global RUN_ROOT, REPO_ROOT  # noqa: PLW0603
    RUN_ROOT = args.run_root.resolve()
    ctx = resolve_execution_context(
        source_root=args.source_root,
        run_root=RUN_ROOT,
        qualification_sha=args.qualification_sha,
    )
    REPO_ROOT = ctx.repo_root
    if args.write_prereg:
        _write_prereg()
        return 0
    if not (RUN_ROOT / "DES_PREREG.json").is_file():
        _write_prereg()
    write_static_manifests(
        qualification_sha=ctx.qualification_sha,
        qualification_tree=ctx.qualification_tree_sha,
        source_root=ctx.source_root,
    )
    if args.manifests_only:
        return 0
    env_manifest = json.loads((RUN_ROOT / "ENVIRONMENT_MANIFEST.json").read_text(encoding="utf-8"))
    if not env_manifest.get("fastmcp_import_ok"):
        print("ENVIRONMENT_BLOCKER: fastmcp not importable", file=sys.stderr)
        return 2
    prereg = json.loads((RUN_ROOT / "DES_PREREG.json").read_text(encoding="utf-8"))
    batch_results = []
    for batch in prereg["batches"]:
        if args.batch and batch["id"] != args.batch:
            continue
        batch_results.append(
            run_batch(
                batch,
                env_manifest=env_manifest,
                source_root=ctx.source_root,
                qualification_sha=ctx.qualification_sha,
                qualification_tree=ctx.qualification_tree_sha,
            )
        )
    manifest_path = RUN_ROOT / "RUN_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema": "eqo.run_manifest.v1", "execution_attempts": []}
    manifest["recorded_utc"] = utc_now()
    manifest["batch_count"] = len(batch_results)
    manifest["batches"] = batch_results
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
