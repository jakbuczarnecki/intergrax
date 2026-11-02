#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""EVOL-X B6 independent *technical* verification (read-only, separate worktrees).

Produces automation qualification output and a formal-authority readiness result.
Does **not** sign B6, name an auditor, or publish ``B6_ACCEPTANCE_PUBLICATION_SHA``.

Mechanical independence: separate git worktrees @ FINAL (Mode A) and TES (Mode B).
Organizational independence: still requires OD-EVOL-11 named human/eligible auditor per ECAP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_P03 = _REPO / "scripts" / "qualification" / "evol_x_p03_evidence.py"
_REGISTER = _REPO / "docs/project/maintainers/qualification/evol_x/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md"
_DEFAULT_TES = "3c4a641b3e48a13f437e830a0562e2a80b03aabb"
_DEFAULT_FINAL = "bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1"
_OD11_UNNAMED = re.compile(r"auditor identity NOT NAMED", re.IGNORECASE)
_OD11_NAMED = re.compile(r"auditor_id\s*[:=]\s*\S+", re.IGNORECASE)


def _subprocess_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def _run(cmd: list[str], *, cwd: Path, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
        env=_subprocess_env(),
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class StepLog:
    step: str
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    payload: dict[str, object] = field(default_factory=dict)


@dataclass
class FormalAuthorityGate:
    gate_id: str
    status: str
    blocker: str | None = None


def _parse_od11_register(text: str) -> tuple[bool, str]:
    if _OD11_NAMED.search(text):
        return True, "auditor_id present in register"
    if _OD11_UNNAMED.search(text):
        return False, "OD-EVOL-11 appointment approved but auditor identity not named (G-02)"
    return False, "OD-EVOL-11 row not found or ambiguous"


def _formal_authority_readiness(register_text: str) -> dict[str, object]:
    named, note = _parse_od11_register(register_text)
    gates: list[FormalAuthorityGate] = [
        FormalAuthorityGate("G-02", "PASS" if named else "BLOCKED", None if named else note),
        FormalAuthorityGate("G-07", "BLOCKED", "no signed Protocol 03 PASS record / no B6_ACCEPTANCE_PUBLICATION_SHA"),
        FormalAuthorityGate("G-08", "BLOCKED", "W7-03 FRZ-EVOL-07 Outcome A/B not operator-adjudicated"),
        FormalAuthorityGate("G-09", "BLOCKED", "OD-EVOL-12 PENDING"),
        FormalAuthorityGate("G-10", "BLOCKED", "OD-EVOLX-08 PENDING"),
    ]
    eligible = named and all(g.status == "PASS" for g in gates)
    return {
        "verdict": "ELIGIBLE" if eligible else "NOT_ELIGIBLE",
        "gates": [asdict(g) for g in gates],
        "note": ("Automated technical verification does not satisfy ECAP independent auditor sign-off (ECAP-P4)."),
    }


def _verify_mode_a_in_worktree(final_wt: Path) -> tuple[int, str, str, dict[str, object]]:
    """Run P03 Mode A using worktree sources (normative @ evidence SHA)."""
    env = _subprocess_env()
    env["PYTHONPATH"] = str(final_wt)
    code = (
        "from pathlib import Path\n"
        "from testing_support.change_safety.evidence import p03\n"
        "problems = p03.verify_mode_a(Path('.'))\n"
        "import json, sys\n"
        "if problems:\n"
        "    sys.stderr.write('\\n'.join(f'EVIDENCE_INTEGRITY_FAIL: {p}' for p in problems) + '\\n')\n"
        "    sys.exit(2)\n"
        "print(json.dumps({'status': 'EVIDENCE_INTEGRITY_OK', 'mode': 'A', "
        "'evidence_head': p03.head_sha(Path('.'))}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=final_wt,
        text=True,
        capture_output=True,
        env=env,
    )
    payload: dict[str, object] = {}
    if proc.returncode == 0:
        payload = _json_line(proc.stdout)
    return proc.returncode, proc.stdout, proc.stderr, payload


def _verify_mode_b_in_worktree(
    tes_wt: Path,
    *,
    evidence_final: str,
    junit: Path,
) -> tuple[int, str, str, dict[str, object]]:
    from scripts.qualification.evol_x_p03_evidence import outcomes_from_junit

    env = _subprocess_env()
    env["PYTHONPATH"] = str(tes_wt)
    replay = outcomes_from_junit(junit, tes_wt)
    code = (
        "from pathlib import Path\n"
        "from testing_support.change_safety.evidence import p03\n"
        "import json, sys\n"
        f"replay = {replay!r}\n"
        f"problems = p03.verify_mode_b(Path('.'), evidence_ref={evidence_final!r}, "
        "replay_outcomes=replay, certification=True)\n"
        "if problems:\n"
        "    sys.stderr.write('\\n'.join(f'EVIDENCE_INTEGRITY_FAIL: {p}' for p in problems) + '\\n')\n"
        "    sys.exit(2)\n"
        "print(json.dumps({'status': 'EVIDENCE_INTEGRITY_OK', 'mode': 'B', "
        "'test_execution_sha': p03.head_sha(Path('.'))}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tes_wt,
        text=True,
        capture_output=True,
        env=env,
    )
    payload: dict[str, object] = {}
    if proc.returncode == 0:
        payload = _json_line(proc.stdout)
    return proc.returncode, proc.stdout, proc.stderr, payload


def _wait_git_blob(repo: Path, ref: str, rel: str, *, timeout_s: float = 180.0) -> None:
    """Worktree checkout can lag; verifier uses ``git show``, not working-tree stat alone."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc = subprocess.run(
            ["git", "show", f"{ref}:{rel}"],
            cwd=repo,
            capture_output=True,
        )
        if proc.returncode == 0:
            return
        time.sleep(1.0)
    raise ValueError(f"timeout waiting for git blob @ {ref[:12]}: {rel}")


def _worktree_add(repo: Path, base: Path, ref: str) -> Path:
    wt = base / f"{ref[:12]}-{uuid.uuid4().hex[:8]}"
    wt.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "worktree", "add", "--detach", str(wt), ref], cwd=repo)
    return wt


def _worktree_remove(repo: Path, wt: Path) -> None:
    _run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo)


def _json_line(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"no JSON payload in stdout: {stdout[:500]!r}")


def run_verification(
    *,
    repo: Path,
    tes: str,
    evidence_final: str,
    session_root: Path,
    skip_mode_b: bool,
    adversarial_invalid_ref: bool,
) -> tuple[dict[str, object], list[StepLog], int]:
    logs: list[StepLog] = []
    exit_code = 0
    worktrees: list[Path] = []
    wt_base = session_root / "worktrees"
    wt_base.mkdir(parents=True, exist_ok=True)
    basetemp = repo / "build" / "pytest-b6-independent-replay"
    basetemp.mkdir(parents=True, exist_ok=True)

    sha_matrix = {
        "verification_orchestrator_head": _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip(),
        "test_execution_sha": tes,
        "evidence_final_publication_sha": evidence_final,
        "b6_acceptance_publication_sha": None,
        "certification_profile": "AUTOMATION_FIRST_SCOPED",
        "qualification_policy_id": "OD-EVOL-13",
    }

    try:
        final_wt = _worktree_add(repo, wt_base, evidence_final)
        worktrees.append(final_wt)
        bundle_m09 = (
            "docs/project/maintainers/qualification/evol_x/evol_x_p03_independent_replay/"
            "evol_m09_automated_diagnostic_qualification.v1.json"
        )
        _wait_git_blob(final_wt, "HEAD", bundle_m09)
        m09a = final_wt / bundle_m09
        if not m09a.is_file():
            raise ValueError(f"worktree @ {evidence_final[:12]} missing {m09a.name} — checkout incomplete")
        cmd_a = [sys.executable, "-c", "mode_a_in_worktree"]
        rc_a, out_a, err_a, payload_a = _verify_mode_a_in_worktree(final_wt)
        _run(["uv", "sync", "--extra", "dev-ci", "--frozen"], cwd=final_wt, capture=False)
        logs.append(
            StepLog(
                step="mode_a_verify_evidence_publish",
                command=cmd_a,
                cwd=str(final_wt),
                exit_code=rc_a,
                stdout=out_a,
                stderr=err_a,
                payload=payload_a,
            )
        )
        mode_a_ok = rc_a == 0

        if not mode_a_ok:
            exit_code = 2

        mode_b_ok = skip_mode_b
        if not skip_mode_b:
            tes_wt = _worktree_add(repo, wt_base, tes)
            worktrees.append(tes_wt)
            _run(["uv", "sync", "--extra", "dev-ci", "--frozen"], cwd=tes_wt, capture=False)
            _env = _subprocess_env()
            vl3_cmd = ["uv", "run", "python", "scripts/qualification/evol_x_vl3_manifest.py", "--check"]
            proc_vl3 = subprocess.run(vl3_cmd, cwd=tes_wt, text=True, capture_output=True, env=_env)
            logs.append(
                StepLog(
                    step="mode_b_vl3_manifest_check",
                    command=vl3_cmd,
                    cwd=str(tes_wt),
                    exit_code=proc_vl3.returncode,
                    stdout=proc_vl3.stdout,
                    stderr=proc_vl3.stderr,
                )
            )
            junit = session_root / "mode_b_junit.xml"
            pytest_cmd = [
                "uv",
                "run",
                "pytest",
                "tests/qualification/evol_x",
                "-q",
                f"--basetemp={basetemp}",
                f"--junitxml={junit}",
            ]
            proc_py = subprocess.run(pytest_cmd, cwd=tes_wt, text=True, capture_output=True, env=_env)
            logs.append(
                StepLog(
                    step="mode_b_pytest_replay",
                    command=pytest_cmd,
                    cwd=str(tes_wt),
                    exit_code=proc_py.returncode,
                    stdout=proc_py.stdout[-8000:],
                    stderr=proc_py.stderr[-4000:],
                    payload={"junit_sha256": _sha256_file(junit) if junit.is_file() else None},
                )
            )
            if proc_vl3.returncode != 0:
                exit_code = 2
                mode_b_ok = False
            elif proc_py.returncode != 0:
                exit_code = 2
                mode_b_ok = False
            else:
                cmd_b = [sys.executable, "-c", "mode_b_in_worktree"]
                rc_b, out_b, err_b, payload_b = _verify_mode_b_in_worktree(
                    tes_wt, evidence_final=evidence_final, junit=junit
                )
                logs.append(
                    StepLog(
                        step="mode_b_verify_test_execution",
                        command=cmd_b,
                        cwd=str(tes_wt),
                        exit_code=rc_b,
                        stdout=out_b,
                        stderr=err_b,
                        payload=payload_b,
                    )
                )
                mode_b_ok = rc_b == 0
                if not mode_b_ok:
                    exit_code = 2

        adversarial: dict[str, object] | None = None
        if adversarial_invalid_ref:
            tes_wt_adv = worktrees[-1] if len(worktrees) > 1 else _worktree_add(repo, wt_base, tes)
            if tes_wt_adv not in worktrees:
                worktrees.append(tes_wt_adv)
                _run(["uv", "sync", "--extra", "dev-ci", "--frozen"], cwd=tes_wt_adv, capture=False)
            bad_ref = tes  # wrong: must be FINAL, not TES
            cmd_bad = [
                "uv",
                "run",
                "python",
                "scripts/qualification/evol_x_p03_evidence.py",
                "verify-test-execution",
                "--evidence-ref",
                bad_ref,
                "--certification",
            ]
            proc_bad = subprocess.run(cmd_bad, cwd=tes_wt_adv, text=True, capture_output=True)
            adversarial = {
                "control": "invalid_evidence_ref_equals_tes",
                "expected": "fail_closed",
                "exit_code": proc_bad.returncode,
                "passed": proc_bad.returncode != 0,
            }
            logs.append(
                StepLog(
                    step="adversarial_invalid_evidence_ref",
                    command=cmd_bad,
                    cwd=str(tes_wt_adv),
                    exit_code=proc_bad.returncode,
                    stdout=proc_bad.stdout,
                    stderr=proc_bad.stderr,
                    payload=adversarial,
                )
            )
            if proc_bad.returncode == 0:
                exit_code = 2

        register_text = _REGISTER.read_text(encoding="utf-8")
        report = {
            "schema_version": "evol_x_b6_independent_verification.v1",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "disposition": (
                "AUTOMATION_TECHNICAL_VERIFICATION_PASS"
                if mode_a_ok and mode_b_ok and exit_code == 0
                else "AUTOMATION_TECHNICAL_VERIFICATION_FAIL"
            ),
            "exact_sha_matrix": sha_matrix,
            "automation_qualification": {
                "mode_a": "PASS" if mode_a_ok else "FAIL",
                "mode_b": "SKIPPED" if skip_mode_b else ("PASS" if mode_b_ok else "FAIL"),
                "profile": "AUTOMATION_FIRST_SCOPED",
            },
            "formal_authority_readiness": _formal_authority_readiness(register_text),
            "adversarial_controls": adversarial,
            "audit_log": [asdict(x) for x in logs],
            "provenance": {
                "tool": "scripts/qualification/evol_x_b6_independent_verification.py",
                "ecap_p4": "implementer/agent output is not independent organizational B6 acceptance",
            },
        }
        return report, logs, exit_code
    finally:
        for wt in worktrees:
            try:
                _worktree_remove(repo, wt)
            except subprocess.CalledProcessError:
                shutil.rmtree(wt, ignore_errors=True)
                _run(["git", "worktree", "prune"], cwd=repo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument("--tes", default=_DEFAULT_TES)
    parser.add_argument("--evidence-final", default=_DEFAULT_FINAL)
    parser.add_argument(
        "--session-root",
        type=Path,
        default=_REPO / ".tmp/session/evol-x-full-automation-closure",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="JSON report path (default: <session-root>/independent_verification_report.json)",
    )
    parser.add_argument(
        "--skip-mode-b",
        action="store_true",
        help="Mode A only (fast; incomplete automation qualification)",
    )
    parser.add_argument(
        "--adversarial-invalid-ref",
        action="store_true",
        help="Run fail-closed control: --evidence-ref TES (must fail)",
    )
    args = parser.parse_args(argv)
    session_root: Path = args.session_root
    session_root.mkdir(parents=True, exist_ok=True)
    report_path = args.report_json or (session_root / "independent_verification_report.json")

    try:
        report, _, code = run_verification(
            repo=args.repo,
            tes=args.tes,
            evidence_final=args.evidence_final,
            session_root=session_root,
            skip_mode_b=args.skip_mode_b,
            adversarial_invalid_ref=args.adversarial_invalid_ref,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        sys.stderr.write(f"verification error: {exc}\n")
        return 1

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"status": report["disposition"], "report": str(report_path)}) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
