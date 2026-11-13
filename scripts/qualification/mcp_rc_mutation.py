#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""MCP-RC mutation qualification (MRC-5): break each critical invariant on a throw-away copy and prove a test fails.

The script never touches the working tree or any accepted branch. It creates a detached git worktree under the
given scratch directory (outside the repository, see CLAUDE.md worktree rules), applies ONE source mutation at a
time, runs the targeted tests with a single pytest process, records ``killed`` / ``survived`` and removes the
worktree. A survivor is a finding, not a pass.

Usage (one terminal, one process at a time):

    python scripts/qualification/mcp_rc_mutation.py
        --scratch <parent>/_wt/_scratch/mcp-rc-mut
        --out .tmp/proof/mcp-rc-mutation/run1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_SHARED = "intergrax/applications/_shared"
_RC_TEST = "tests/unit/applications/test_mcp_rc_audit_corrections.py"


@dataclass(frozen=True)
class Mutant:
    mutant_id: str
    invariant: str
    path: str
    old: str
    new: str
    tests: tuple[str, ...]


MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        "M1-tenant-admission-bypass",
        "No verified principal means MCP_PRINCIPAL_REQUIRED in the default mode",
        f"{_SHARED}/mcp_harness_principal.py",
        "    if mode is McpIdentityMode.HARNESS_BOUND:\n        detail =",
        "    if False:\n        detail =",
        ("tests/unit/applications/test_mcp_identity_mode_fail_closed.py",),
    ),
    Mutant(
        "M2-principal-mismatch-ignored",
        "An asserted tenant/user that differs from the verified principal is a conflict",
        f"{_SHARED}/mcp_harness_principal.py",
        "            reject_identity_assertion_conflicts(\n",
        "            (lambda **_kw: None)(\n",
        ("tests/unit/applications/test_mcp_identity_mode_fail_closed.py",),
    ),
    Mutant(
        "M3-result-bound-missing-on-custom-tools",
        "Custom host tool results are bounded like every other result",
        f"{_SHARED}/mcp_server_engine.py",
        "    return bound_mcp_result_or_raise(result)\n\n\ndef _bind_custom_host_tool_for_fastmcp(",
        "    return result\n\n\ndef _bind_custom_host_tool_for_fastmcp(",
        ("tests/unit/applications/test_mcp_result_bounds_and_error_contract.py",),
    ),
    Mutant(
        "M4-deadline-propagation-dropped",
        "max_call_seconds reaches the transport cancel context",
        f"{_SHARED}/mcp_host_execution_transport_cancel.py",
        "    max_call_seconds = cancel_ctx.max_call_seconds\n",
        "    max_call_seconds = None\n",
        ("tests/unit/applications/test_mcp_deadline_and_resilience.py",),
    ),
    Mutant(
        "M5-deadline-does-not-request-governed-cancel",
        "The call bound requests the governed cancel (cancellation reaches the canonical coordinator)",
        f"{_SHARED}/mcp_host_execution_transport_cancel.py",
        "                    deadline_hit.set()\n                    transport_cancel_observed = True\n",
        "                    deadline_hit.set()\n",
        ("tests/unit/applications/test_mcp_deadline_and_resilience.py",),
    ),
    Mutant(
        "M6-admission-slot-release-lost",
        "An admission slot is released when the handler ends",
        f"{_SHARED}/mcp_engine_concurrency.py",
        "            self._inflight -= 1",
        "            pass",
        ("tests/unit/applications/test_mcp_deadline_and_resilience.py",),
    ),
    Mutant(
        "M7-error-code-mapping-swapped",
        "Legacy identity_conflict maps to MCP_TENANT_ASSERTION_CONFLICT",
        f"{_SHARED}/mcp_wire_errors.py",
        '    "identity_conflict": _C.MCP_TENANT_ASSERTION_CONFLICT,',
        '    "identity_conflict": _C.MCP_PRINCIPAL_REQUIRED,',
        ("tests/unit/applications/test_mcp_result_bounds_and_error_contract.py",),
    ),
    Mutant(
        "M8-unauthorized-execution-path",
        "Every tool that starts an agent task resolves identity in the caller's mode (no forced local-dev path)",
        f"{_SHARED}/mcp_nexus_server.py",
        "        identity_mode=coerce_mcp_identity_mode(identity_mode),\n"
        "    )\n    return await execute_mcp_agent_task(",
        "        identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT,\n    )\n    return await execute_mcp_agent_task(",
        ("tests/unit/applications/test_mcp_identity_mode_fail_closed.py",),
    ),
    # --- mutants added after the independent audit (AF-01..AF-04); killed by test_mcp_rc_audit_corrections.py ---
    Mutant(
        "M9-cross-loop-failure-payload-bound-removed",
        "Cross-loop failure payloads are bounded and sanitized at the MCP boundary",
        f"{_SHARED}/mcp_catalog_invocation_bridge.py",
        "            return bound_cross_loop_catalog_payload(terminal.payload)\n",
        "            return terminal.payload\n",
        (_RC_TEST,),
    ),
    Mutant(
        "M10-cross-loop-success-payload-bound-removed",
        "Cross-loop success payloads are bounded",
        f"{_SHARED}/mcp_catalog_invocation_bridge.py",
        "            **bound_cross_loop_catalog_payload(terminal.payload),\n",
        "            **terminal.payload,\n",
        (_RC_TEST,),
    ),
    Mutant(
        "M11-local-dev-downgrades-configured-auth",
        "LOCAL_DEV never downgrades configured auth for in-process callers",
        f"{_SHARED}/mcp_harness_principal.py",
        "    if _harness_auth_requires_principal(effective_harness_auth) and not _local_dev_harness_bypass(\n"
        "        effective_harness_auth,\n    ):",
        "    if False and not _local_dev_harness_bypass(\n        effective_harness_auth,\n    ):",
        (_RC_TEST,),
    ),
    Mutant(
        "M12-list-catalog-tools-principal-gate-removed",
        "list_catalog_tools requires a principal in the default mode",
        f"{_SHARED}/mcp_catalog_tools.py",
        '            async with sanitized_tool_errors("list_catalog_tools"):\n'
        "                require_mcp_caller_principal(identity_mode=identity_mode)\n",
        '            async with sanitized_tool_errors("list_catalog_tools"):\n',
        (_RC_TEST,),
    ),
    Mutant(
        "M13-describe-catalog-tool-principal-gate-removed",
        "describe_catalog_tool requires a principal in the default mode",
        f"{_SHARED}/mcp_catalog_tools.py",
        '            async with sanitized_tool_errors("describe_catalog_tool"):\n'
        "                require_mcp_caller_principal(identity_mode=identity_mode)\n",
        '            async with sanitized_tool_errors("describe_catalog_tool"):\n',
        (_RC_TEST,),
    ),
    Mutant(
        "M14-raw-exception-text-echoed-by-tools",
        "No tool path echoes raw exception text",
        f"{_SHARED}/mcp_wire_errors.py",
        '_C.MCP_INTERNAL_ERROR, f"tool execution failed ({type(exc).__name__})")',
        '_C.MCP_INTERNAL_ERROR, f"tool execution failed ({exc})")',
        (_RC_TEST,),
    ),
    Mutant(
        "M15-custom-tool-principal-gate-removed",
        "Custom host tools are behind the principal gate by default",
        f"{_SHARED}/mcp_server_engine.py",
        "        if gate is not None:\n            gate()\n",
        "        if False:\n            gate()\n",
        (_RC_TEST,),
    ),
    Mutant(
        "M16-failed-host-after-bound-not-deadline-exceeded",
        "A host that fails after the call bound is reported as MCP_DEADLINE_EXCEEDED",
        f"{_SHARED}/mcp_host_execution_transport_cancel.py",
        '            raise McpHostExecutionDeadlineExceeded("failed", max_call_seconds) from exc\n',
        "            raise\n",
        (_RC_TEST,),
    ),
)


def _replace_once(path: Path, old: str, new: str) -> bool:
    raw = path.read_bytes()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if text.count(old) != 1:
        return False
    text = text.replace(old, new, 1)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    return True


def _run(cmd: list[str], *, cwd: Path, timeout: float) -> tuple[int, str]:
    try:
        completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or b""
        return 124, out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)
    return completed.returncode, completed.stdout + completed.stderr


def run_mutation(repo: Path, python: str, scratch: Path, out_dir: Path, only: set[str], timeout: float) -> int:
    if scratch.exists():
        print(f"[ERROR] scratch path already exists: {scratch}", file=sys.stderr)
        return 2
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo, timeout=60)[1].strip()
    scratch.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    code, text = _run(["git", "worktree", "add", "--detach", str(scratch), head], cwd=repo, timeout=600)
    if code != 0:
        print(text, file=sys.stderr)
        return 2
    results: list[dict[str, object]] = []
    try:
        for mutant in MUTANTS:
            if only and mutant.mutant_id not in only and mutant.mutant_id.split("-", 1)[0] not in only:
                continue
            target = scratch / mutant.path
            started = time.monotonic()
            if not _replace_once(target, mutant.old, mutant.new):
                results.append(
                    {**asdict(mutant), "status": "not_applicable", "detail": "pattern not found exactly once"}
                )
                continue
            code, text = _run(
                [python, "-m", "pytest", *mutant.tests, "-q", "--tb=line", "-x"], cwd=scratch, timeout=timeout
            )
            status = "survived" if code == 0 else ("killed_by_timeout" if code == 124 else "killed")
            tail = "\n".join(text.strip().splitlines()[-6:])
            results.append(
                {
                    **asdict(mutant),
                    "status": status,
                    "exit_code": code,
                    "seconds": round(time.monotonic() - started, 1),
                    "tail": tail,
                }
            )
            _run(["git", "checkout", "--", mutant.path], cwd=scratch, timeout=120)
    finally:
        _run(["git", "worktree", "remove", "--force", str(scratch)], cwd=repo, timeout=300)
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
    report = {"repo_head": head, "results": results}
    (out_dir / "mutation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# MCP-RC mutation report (head {head})", "", "| mutant | invariant | status |", "| --- | --- | --- |"]
    lines += [f"| {r['mutant_id']} | {r['invariant']} | {r['status']} |" for r in results]
    (out_dir / "mutation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    survivors = [r for r in results if r["status"] in {"survived", "not_applicable"}]
    print("\n".join(lines))
    return 1 if survivors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scratch", type=Path, help="throw-away worktree path (must not exist)")
    parser.add_argument("--out", type=Path, default=Path(".tmp/proof/mcp-rc-mutation/run"))
    parser.add_argument("--only", default="", help="comma separated mutant ids or prefixes, e.g. M1,M3")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        for mutant in MUTANTS:
            print(f"{mutant.mutant_id}: {mutant.invariant}")
        return 0
    if args.scratch is None:
        parser.error("--scratch is required")
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    return run_mutation(args.repo.resolve(), args.python, args.scratch.resolve(), args.out, only, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
