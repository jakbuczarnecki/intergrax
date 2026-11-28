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
    ec08_dir,
    extract_pytest_paths_from_command,
    git,
    inventory_binding,
    sha256_bytes,
    sha256_file,
    utc_now,
)

RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[7]


def _uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "uv.exe"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("uv executable not found")


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
    q_skip = {"Q04"}
    for cmd in ec08_des["commands"]:
        cmd_id = cmd["cmd_id"]
        if cmd_id in q_skip:
            batches.append(
                {
                    "id": cmd_id.lower(),
                    "paths": extract_pytest_paths_from_command(cmd["command"]),
                    "deferred": "DEFERRED_FORMAL_QUALIFICATION_PROFILE",
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


def run_batch(batch: dict[str, Any], *, env_manifest: dict[str, Any]) -> dict[str, Any]:
    if batch.get("deferred"):
        return {
            "batch_id": batch["id"],
            "deferred": batch["deferred"],
            "skipped": True,
        }
    batch_id = batch["id"]
    paths = batch["paths"]
    pre_sha = git(REPO_ROOT, ["rev-parse", "HEAD"])
    pre_tree = git(REPO_ROOT, ["rev-parse", "HEAD^{tree}"])
    worktree = git(REPO_ROOT, ["rev-parse", "--git-dir"])
    argv = [
        _uv(),
        "run",
        "--no-sync",
        "pytest",
        *paths,
        "-vv",
        "-rA",
        "--tb=short",
        f"--junitxml={RUN_ROOT / 'junit' / (batch_id + '.xml')}",
    ]
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
    junit_dir = RUN_ROOT / "junit"
    logs_dir = RUN_ROOT / "logs"
    junit_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    junit_path = junit_dir / f"{batch_id}.xml"
    log_path = logs_dir / f"{batch_id}.log"
    (rec_dir / f"{batch_id}.pre.json").write_text(json.dumps(record_pre, indent=2), encoding="utf-8")
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    for key in (
        "INTERGRAX_INBOUND_VERIFIER",
        "INTERGRAX_SLACK_SIGNING_SECRET",
        "INTERGRAX_SLACK_VERIFY_SIGNATURE",
    ):
        child_env.pop(key, None)
    started = time.monotonic()
    with log_path.open("wb") as log:
        proc = subprocess.run(argv, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=child_env)
    elapsed = time.monotonic() - started
    log_bytes = log_path.read_bytes()
    junit_bytes = junit_path.read_bytes() if junit_path.is_file() else b""
    post_sha = git(REPO_ROOT, ["rev-parse", "HEAD"])
    post_tree = git(REPO_ROOT, ["rev-parse", "HEAD^{tree}"])
    record_post = {
        "schema": "eqo.execution_record.v1",
        "phase": "post",
        "batch_id": batch_id,
        "process_returncode": proc.returncode,
        "finished_at_utc": utc_now(),
        "monotonic_elapsed_seconds": elapsed,
        "stdout_stderr_path": log_path.relative_to(RUN_ROOT).as_posix(),
        "stdout_stderr_sha256": sha256_bytes(log_bytes),
        "junit_path": junit_path.relative_to(RUN_ROOT).as_posix() if junit_path.is_file() else None,
        "junit_sha256": sha256_bytes(junit_bytes) if junit_bytes else None,
        "source_commit_sha_after": post_sha,
        "source_tree_sha_after": post_tree,
        "source_changed_during_batch": post_sha != pre_sha or post_tree != pre_tree,
    }
    (rec_dir / f"{batch_id}.post.json").write_text(json.dumps(record_post, indent=2), encoding="utf-8")
    return {"batch_id": batch_id, "pre": record_pre, "post": record_post}


def write_static_manifests() -> None:
    binding = inventory_binding(REPO_ROOT)
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
        "required_ancestor": "8f03bae217386e26ce1d7804ca673f57f6bfccfc",
        "published_tree": "25a014baa889f2c6c79ea8109f94b5900f451c17",
        "implementation_commit_ec_w3_11": "b716ecfef7e62c4edbaa16b9d2fc522bea777cd7",
        "implementation_tree_ec_w3_11": "05637e5a76c3d1bf9124c4c687a3ce2543b3f274",
        "task_start_remote_sha": git(REPO_ROOT, ["rev-parse", "origin/development"]),
        "qualification_source_sha": git(REPO_ROOT, ["rev-parse", "HEAD"]),
        "qualification_source_tree_sha": git(REPO_ROOT, ["rev-parse", "HEAD^{tree}"]),
        "recorded_utc": utc_now(),
    }
    (RUN_ROOT / "BASELINE_BINDING.json").write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if "--write-prereg" in args:
        _write_prereg()
        return 0
    if not (RUN_ROOT / "DES_PREREG.json").is_file():
        _write_prereg()
    write_static_manifests()
    if "--manifests-only" in args:
        return 0
    env_manifest = json.loads((RUN_ROOT / "ENVIRONMENT_MANIFEST.json").read_text(encoding="utf-8"))
    if not env_manifest.get("fastmcp_import_ok"):
        print("ENVIRONMENT_BLOCKER: fastmcp not importable", file=sys.stderr)
        return 2
    prereg = json.loads((RUN_ROOT / "DES_PREREG.json").read_text(encoding="utf-8"))
    batch_results = []
    only = None
    for i, a in enumerate(args):
        if a == "--batch" and i + 1 < len(args):
            only = args[i + 1]
    for batch in prereg["batches"]:
        if only and batch["id"] != only:
            continue
        batch_results.append(run_batch(batch, env_manifest=env_manifest))
    manifest = {
        "schema": "eqo.run_manifest.v1",
        "recorded_utc": utc_now(),
        "batch_count": len(batch_results),
        "batches": batch_results,
    }
    (RUN_ROOT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
