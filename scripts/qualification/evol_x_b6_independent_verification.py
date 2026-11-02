#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""EVOL-X B6 verification (technical replay + authority + attestation).

Mechanical replay uses separate git worktrees @ FINAL (Mode A) and TES (Mode B).
Organizational acceptance is a separate field. A technical PASS does not make
formal acceptance eligible. This tool does not sign B6 and does not publish
``B6_ACCEPTANCE_PUBLICATION_SHA``.

Test credentials (``NON_PRODUCTION_TEST_ONLY``) cannot authorize formal B6.
ECAP-P4-EVOL-SCOPE-AUTO-B6 stays inactive until its governance record says
``IN_FORCE``.
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
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.qualification.evol_x_b6_attestation import (
    default_governance_path as _default_governance_path,
)
from scripts.qualification.evol_x_b6_attestation import evaluate_acceptance, load_governance

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
        return (
            False,
            "OD-EVOL-11 appointment approved but auditor identity not named (G-02); "
            "OD-EVOL-14 does not supply this identity",
        )
    return False, "OD-EVOL-11 row not found or ambiguous"


def _formal_authority_readiness(register_text: str, governance: Mapping[str, Any]) -> dict[str, object]:
    named, note = _parse_od11_register(register_text)
    activation = governance.get("ecap_amendment_activation")
    effect = governance.get("normative_effect")
    approvals = governance.get("required_approvals_present")
    approvals_map = approvals if isinstance(approvals, dict) else {}
    amendment_in_force = activation == "IN_FORCE" and effect == "IN_FORCE"
    custody_ready = (
        approvals_map.get("independent_key_custody") is True
        and governance.get("independent_attestation_environment") is True
    )
    gates: list[FormalAuthorityGate] = [
        FormalAuthorityGate("G-02", "PASS" if named else "BLOCKED", None if named else note),
        FormalAuthorityGate("G-07", "BLOCKED", "no signed Protocol 03 PASS record / no B6_ACCEPTANCE_PUBLICATION_SHA"),
        FormalAuthorityGate("G-08", "BLOCKED", "W7-03 FRZ-EVOL-07 Outcome A/B not operator-adjudicated"),
        FormalAuthorityGate("G-09", "BLOCKED", "OD-EVOL-12 PENDING"),
        FormalAuthorityGate("G-10", "BLOCKED", "OD-EVOLX-08 PENDING"),
        FormalAuthorityGate(
            "G-14",
            "PASS" if amendment_in_force else "BLOCKED",
            None if amendment_in_force else "ECAP-P4-EVOL-SCOPE-AUTO-B6 activation is not IN_FORCE",
        ),
        FormalAuthorityGate(
            "G-15",
            "PASS" if custody_ready else "BLOCKED",
            None if custody_ready else "independent key custody is not provisioned",
        ),
    ]
    eligible = named and all(g.status == "PASS" for g in gates)
    return {
        "verdict": "ELIGIBLE" if eligible else "NOT_ELIGIBLE",
        "gates": [asdict(g) for g in gates],
        "note": ("Automated technical verification does not satisfy ECAP independent auditor sign-off (ECAP-P4)."),
    }


def _verify_mode_a_in_worktree(final_wt: Path, *, orchestrator_repo: Path) -> tuple[int, str, str, dict[str, object]]:
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
        ["uv", "run", "--directory", str(orchestrator_repo), "python", "-c", code],
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
    replay_file: Path,
) -> tuple[int, str, str, dict[str, object]]:
    replay_file = replay_file.resolve()
    code = (
        "from pathlib import Path\n"
        "from testing_support.change_safety.evidence import p03\n"
        "import json, sys\n"
        f"replay = json.loads(Path({str(replay_file)!r}).read_text(encoding='utf-8'))\n"
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
        env=_subprocess_env() | {"PYTHONPATH": str(tes_wt)},
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
        target = repo / rel
        if proc.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            return
        time.sleep(0.25)
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


def decision_fields_from_run(
    *,
    mode_a_ok: bool,
    mode_b_ok: bool,
    skip_mode_b: bool,
    exit_code: int,
    governance: Mapping[str, Any],
    attestation: Mapping[str, Any] | None,
    trust_root: Mapping[str, Any] | None,
    replay_ledger: set[str] | None,
    now: datetime,
    expected_tool_digest: str | None,
) -> dict[str, Any]:
    """Separate technical, authority, attestation, and formal eligibility.

    Always leaves acceptance publication unset. Formal eligibility is not a
    technical PASS and is not a published B6 record.
    """
    if mode_a_ok and mode_b_ok and not skip_mode_b and exit_code == 0:
        technical = "PASS"
    elif skip_mode_b and mode_a_ok and exit_code == 0:
        technical = "INCOMPLETE"
    else:
        technical = "FAIL"
    mode_a = "PASS" if mode_a_ok else "FAIL"
    mode_b = "SKIPPED" if skip_mode_b else ("PASS" if mode_b_ok else "FAIL")
    fields = evaluate_acceptance(
        technical_verdict=technical,
        mode_a=mode_a,
        mode_b=mode_b,
        governance=governance,
        attestation=attestation,
        trust_root=trust_root,
        observed={"mode_a": mode_a, "mode_b": mode_b},
        expected_tool_digest=expected_tool_digest,
        replay_ledger=replay_ledger,
        now=now,
    )
    fields["b6_acceptance_publication_sha"] = None
    fields["evol_x_closed"] = False
    return fields


def _load_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def _load_replay_ledger(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(item, str) for item in data):
        raise ValueError("replay ledger must be a JSON array of strings")
    return set(data)


def _fail_closed_governance() -> dict[str, Any]:
    return {
        "schema_version": "unreadable",
        "operator_decision_id": "UNREADABLE",
        "ecap_amendment_id": "",
        "ecap_amendment_version": "",
        "ecap_amendment_activation": "PENDING_REQUIRED_APPROVALS",
        "normative_effect": "NOT_IN_FORCE",
        "test_execution_sha": "",
        "evidence_final_publication_sha": "",
        "required_approvals_present": {},
        "independent_attestation_environment": False,
    }


def run_verification(
    *,
    repo: Path,
    tes: str,
    evidence_final: str,
    session_root: Path,
    skip_mode_b: bool,
    adversarial_invalid_ref: bool,
    attestation: Mapping[str, Any] | None = None,
    trust_root: Mapping[str, Any] | None = None,
    governance: Mapping[str, Any] | None = None,
    replay_ledger: set[str] | None = None,
    now: datetime | None = None,
    expected_tool_digest: str | None = None,
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
        _run(["uv", "sync", "--extra", "dev-ci", "--frozen"], cwd=final_wt, capture=False)
        bundle_m09 = (
            "docs/project/maintainers/qualification/evol_x/evol_x_p03_independent_replay/"
            "evol_m09_automated_diagnostic_qualification.v1.json"
        )
        _wait_git_blob(final_wt, "HEAD", bundle_m09)
        _wait_git_blob(final_wt, "HEAD", "testing_support/change_safety/evidence/p03.py")
        m09a = final_wt / bundle_m09
        if not m09a.is_file():
            raise ValueError(f"worktree @ {evidence_final[:12]} missing {m09a.name} — checkout incomplete")
        cmd_a = ["uv", "run", "python", "-c", "mode_a_in_worktree"]
        rc_a, out_a, err_a, payload_a = _verify_mode_a_in_worktree(final_wt, orchestrator_repo=repo)
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
            junit = (session_root / "mode_b_junit.xml").resolve()
            session_root.mkdir(parents=True, exist_ok=True)
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
                from scripts.qualification.evol_x_p03_evidence import outcomes_from_junit

                replay = outcomes_from_junit(junit, tes_wt)
                replay_file = (session_root / "mode_b_replay_outcomes.json").resolve()
                replay_file.write_text(json.dumps(replay, sort_keys=True), encoding="utf-8")
                cmd_b = [sys.executable, "-c", "mode_b_in_worktree"]
                rc_b, out_b, err_b, payload_b = _verify_mode_b_in_worktree(
                    tes_wt, evidence_final=evidence_final, replay_file=replay_file
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
        if governance is None:
            try:
                governance = load_governance(_default_governance_path(repo))
            except (OSError, json.JSONDecodeError, ValueError):
                governance = _fail_closed_governance()
        evaluated_at = now or datetime.now(UTC)
        tool_digest = expected_tool_digest if expected_tool_digest is not None else _sha256_file(Path(__file__))
        decisions = decision_fields_from_run(
            mode_a_ok=mode_a_ok,
            mode_b_ok=mode_b_ok,
            skip_mode_b=skip_mode_b,
            exit_code=exit_code,
            governance=governance,
            attestation=attestation,
            trust_root=trust_root,
            replay_ledger=replay_ledger,
            now=evaluated_at,
            expected_tool_digest=tool_digest,
        )
        technical_verdict = decisions["technical_verification"]["verdict"]
        if technical_verdict == "PASS":
            disposition = "AUTOMATION_TECHNICAL_VERIFICATION_PASS"
        elif technical_verdict == "INCOMPLETE":
            disposition = "AUTOMATION_TECHNICAL_VERIFICATION_INCOMPLETE"
        else:
            disposition = "AUTOMATION_TECHNICAL_VERIFICATION_FAIL"
        report = {
            "schema_version": "evol_x_b6_independent_verification.v2",
            "generated_at_utc": evaluated_at.isoformat(),
            "disposition": disposition,
            "technical_verification": decisions["technical_verification"],
            "authority_verification": decisions["authority_verification"],
            "attestation_verification": decisions["attestation_verification"],
            "formal_acceptance_eligibility": decisions["formal_acceptance_eligibility"],
            "formal_acceptance_blockers": decisions["formal_acceptance_blockers"],
            "b6_acceptance_publication_sha": None,
            "evol_x_closed": False,
            "exact_sha_matrix": sha_matrix,
            "automation_qualification": {
                "mode_a": "PASS" if mode_a_ok else "FAIL",
                "mode_b": "SKIPPED" if skip_mode_b else ("PASS" if mode_b_ok else "FAIL"),
                "profile": "AUTOMATION_FIRST_SCOPED",
            },
            "formal_authority_readiness": _formal_authority_readiness(register_text, governance),
            "adversarial_controls": adversarial,
            "audit_log": [asdict(x) for x in logs],
            "provenance": {
                "tool": "scripts/qualification/evol_x_b6_independent_verification.py",
                "tool_version": "2",
                "tool_digest": tool_digest,
                "ecap_p4": "implementer/agent output is not independent organizational B6 acceptance",
                "credential_policy": "NON_PRODUCTION_TEST_ONLY signatures cannot authorize formal B6",
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
    parser.add_argument("--attestation", type=Path, default=None, help="Signed attestation JSON to evaluate")
    parser.add_argument("--trust-root", type=Path, default=None, help="Trust-root JSON (public keys only)")
    parser.add_argument(
        "--replay-ledger",
        type=Path,
        default=None,
        help="Read-only JSON array of replay bindings. This tool does not append it.",
    )
    args = parser.parse_args(argv)
    session_root: Path = args.session_root
    session_root.mkdir(parents=True, exist_ok=True)
    report_path = args.report_json or (session_root / "independent_verification_report.json")

    try:
        attestation = _load_object(args.attestation) if args.attestation else None
        trust_root = _load_object(args.trust_root) if args.trust_root else None
        replay_ledger = _load_replay_ledger(args.replay_ledger)
        report, _, code = run_verification(
            repo=args.repo,
            tes=args.tes,
            evidence_final=args.evidence_final,
            session_root=session_root,
            skip_mode_b=args.skip_mode_b,
            adversarial_invalid_ref=args.adversarial_invalid_ref,
            attestation=attestation,
            trust_root=trust_root,
            replay_ledger=replay_ledger,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        sys.stderr.write(f"verification error: {exc}\n")
        return 1

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    technical = report["technical_verification"]
    authority = report["authority_verification"]
    attestation_result = report["attestation_verification"]
    summary = {
        "technical_verification": technical["verdict"],
        "authority_verification": authority["verdict"],
        "attestation_verification": attestation_result["verdict"],
        "formal_acceptance_eligibility": report["formal_acceptance_eligibility"],
        "b6_acceptance_publication_sha": None,
        "report": str(report_path),
    }
    sys.stdout.write(json.dumps(summary) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
