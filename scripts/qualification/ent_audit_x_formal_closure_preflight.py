#!/usr/bin/env python3
"""Fail-closed preflight for ENT-AUDIT-X formal closure under IIAC (two gates).

Gate A (default) runs before the closure-phase-1 commit: every prerequisite is published and the roadmap does
not yet carry the formal closure headline. Gate B runs after the ``FORMAL_CLOSURE_PUBLICATION_SHA`` pin: the
headline is present exactly once, the pin is set, equals the status JSON and is ordered after the IIAC pass
record. Unset, malformed or missing pins always fail (they are never treated as "not pending").

The historical human Protocol 03 path is superseded by IIAC and is not supported here.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

HEADLINE = "ENT-AUDIT-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED"
LEGACY_HEADLINE = "ENT-AUDIT-X — CLOSED — INDEPENDENTLY ACCEPTED — PUBLISHED"

_QUAL = Path("docs/project/maintainers/qualification")
_ENT = _QUAL / "ent_audit_x"
_STATUS = _ENT / "ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json"
_LADDER = _ENT / "ENT_AUDIT_X_SHA_LADDER.md"
_PASS_RECORD = _ENT / "ENT_AUDIT_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
_MANIFEST = _QUAL / "iiac_evidence/ENT-AUDIT-X/certification/iiac_audit_session_manifest.ENT-AUDIT-X.json"
_EVOL_STATUS = _QUAL / "evol_x/EVOL_X_CLOSURE_EXECUTION_STATUS.json"
_ACTIVATION = _QUAL / "IIAC_POLICY_ACTIVATION_RECORD.json"
_FRZ = _QUAL / "PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
_ROADMAP = Path("docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md")

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_DIGEST64 = re.compile(r"^[0-9a-f]{64}$")
_ACCEPTING_VERDICTS = frozenset({"PASS", "PASS_SCOPED"})
_OD02_MODES = frozenset({"live", "compensating"})
_MANIFEST_DIGESTS = ("audit_report_digest", "audit_scope_manifest_digest", "test_matrix_digest", "findings_digest")

Verifier = Callable[[], list[str]]


def _iiac_in_force(repo: Path | None = None) -> bool:
    path = (repo or _REPO) / _ACTIVATION
    if not path.is_file():
        return False
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    return str(record.get("status") or "").upper() == "IN_FORCE"


def _load_json(path: Path, label: str, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"missing {label}: {path.name}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        errors.append(f"{label} is not valid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{label} must be a JSON object")
        return None
    return data


def ladder_pin(ladder_text: str, role: str) -> str | None:
    """SHA of the last ladder row whose first cell is exactly ``role``; None if absent, unset or malformed."""
    pin: str | None = None
    for line in ladder_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] != role:
            continue
        cell = cells[1].strip("`* ")
        pin = cell if _SHA40.match(cell) else None
    return pin


def _git(repo: Path, *args: str) -> tuple[int, str]:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return done.returncode, done.stdout.strip()


def _git_ancestor(repo: Path, sha: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", sha, "HEAD")[0] == 0


def _oldest_commit(repo: Path, *log_args: str) -> str | None:
    code, out = _git(repo, "log", "--format=%H", *log_args)
    lines = out.split()
    return lines[-1] if code == 0 and lines else None


def _check_evol_predecessor(repo: Path, errors: list[str]) -> None:
    evol = _load_json(repo / _EVOL_STATUS, "EVOL-X closure status", errors)
    if evol is None:
        return
    closed = str(evol.get("parent_evol_x_status", "")).upper().startswith("CLOSED")
    iiac_b6 = evol.get("protocol_03") == "SUPERSEDED_BY_IIAC_B6_ACCEPTANCE" or str(
        evol.get("formal_ecap_independent_b6", "")
    ).startswith("ACCEPTED_VIA_IIAC")
    handoff = str(evol.get("w7_06_handoff_publication_sha") or "")
    if not (closed and iiac_b6 and _SHA40.match(handoff)):
        errors.append("ENT preflight: EVOL predecessor is not IIAC-closed with a published ENT handoff pin (N-10)")


def _check_manifest(repo: Path, policy: str, ladder_text: str, errors: list[str]) -> None:
    if not (repo / _PASS_RECORD).is_file():
        errors.append(f"IIAC IN_FORCE: missing {_PASS_RECORD.as_posix()}")
    manifest = _load_json(repo / _MANIFEST, "ENT IIAC audit manifest", errors)
    if manifest is None:
        return
    if manifest.get("stage_id") != "ENT-AUDIT-X":
        errors.append("IIAC manifest stage_id must be ENT-AUDIT-X")
    if str(manifest.get("verdict", "")) not in _ACCEPTING_VERDICTS:
        errors.append("IIAC manifest verdict must be PASS or PASS_SCOPED")
    if policy and manifest.get("policy_version") != policy:
        errors.append("IIAC manifest policy_version differs from the in-force policy")
    sessions = [
        str(manifest.get(key, "")).strip()
        for key in ("implementation_session_id", "auditor_session_id", "certifier_session_id")
    ]
    if not all(sessions) or len(set(sessions)) != len(sessions):
        errors.append("IIAC manifest sessions must be present and pairwise distinct")
    for key in _MANIFEST_DIGESTS:
        if not _DIGEST64.match(str(manifest.get(key, ""))):
            errors.append(f"IIAC manifest {key} missing or not a sha256")
    if not manifest.get("limitations"):
        errors.append("IIAC manifest limitations must be recorded")
    if not manifest.get("assurance_level"):
        errors.append("IIAC manifest assurance_level must be declared (process_enforced is expected)")
    for key, role in (
        ("test_execution_sha", "TEST_EXECUTION_SHA"),
        ("evidence_publication_sha", "EVIDENCE_PUBLICATION_SHA"),
    ):
        pinned = ladder_pin(ladder_text, role)
        if pinned is None or manifest.get(key) != pinned:
            errors.append(f"IIAC manifest {key} does not match the SHA ladder {role}")


def _check_frz_aud(repo: Path, mode: str | None, errors: list[str]) -> None:
    path = repo / _FRZ
    if not path.is_file():
        errors.append("missing freeze acceptance checklist")
        return
    status: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\|\s*FRZ-AUD-(\d{2})\s*\|", line)
        if not match:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        status[int(match.group(1))] = cells[4] if len(cells) > 4 else ""
    for number in range(1, 11):
        value = status.get(number)
        if value is None:
            errors.append(f"FRZ-AUD-{number:02d} row missing in freeze checklist")
        elif not value or value.upper().startswith("OPEN"):
            errors.append(f"FRZ-AUD-{number:02d} still OPEN in freeze checklist")
    row10 = status.get(10, "").upper()
    if row10 and not row10.startswith("OPEN"):
        if mode == "compensating" and "COMPENSATING" not in row10:
            errors.append("FRZ-AUD-10 must read COMPENSATING (SCOPED) when od_ent_02_mode is compensating")
        if mode == "live" and "LIVE" not in row10:
            errors.append("FRZ-AUD-10 must read LIVE when od_ent_02_mode is live")


def _check_od_ent_02(
    repo: Path, mode: object, compensating: Callable[[Path], list[str]] | None, live: Verifier | None, errors: list[str]
) -> None:
    if mode not in _OD02_MODES:
        errors.append("IIAC IN_FORCE: set od_ent_02_mode to live|compensating in the status JSON")
        return
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    if mode == "compensating":
        if compensating is None:
            from scripts.governance import verify_od_ent_02_compensating_control as verifier

            problems = verifier.collect_errors(repo, published=True)
        else:
            problems = compensating(repo)
        errors.extend(f"OD-ENT-02 compensating: {item}" for item in problems)
    else:
        if live is None:
            from scripts.governance import verify_ent_audit_branch_protection as verifier

            problems = verifier.verify_branch_protection()
        else:
            problems = live()
        errors.extend(f"OD-ENT-02 live: {item}" for item in problems)


def _check_pins(
    gate: str,
    status: dict,
    ladder_text: str,
    is_ancestor: Callable[[str], bool],
    errors: list[str],
) -> None:
    pins = {
        "B6_ACCEPTANCE_PUBLICATION_SHA": "b6_acceptance_publication_sha",
        "ACCEPTED_BASELINE_SHA": "accepted_baseline_sha",
    }
    if gate == "B":
        pins["FORMAL_CLOSURE_PUBLICATION_SHA"] = "formal_closure_publication_sha"
    for role, key in pins.items():
        pinned = ladder_pin(ladder_text, role)
        if pinned is None:
            errors.append(f"SHA ladder {role} is unset or malformed")
            continue
        if status.get(key) != pinned:
            errors.append(f"status JSON {key} does not equal the SHA ladder {role}")
        if not is_ancestor(pinned):
            errors.append(f"SHA ladder {role} is not an ancestor of HEAD")
    if gate == "A" and status.get("formal_closure_publication_sha"):
        errors.append("formal_closure_publication_sha is already set before closure (Gate A)")


def _check_roadmap(repo: Path, gate: str, errors: list[str]) -> None:
    path = repo / _ROADMAP
    if not path.is_file():
        errors.append("missing roadmap")
        return
    text = path.read_text(encoding="utf-8-sig")
    if LEGACY_HEADLINE in text:
        errors.append("roadmap carries the superseded 'INDEPENDENTLY ACCEPTED' ENT closure headline")
    count = text.count(HEADLINE)
    if gate == "A" and count:
        errors.append("roadmap already claims formal ENT closure (premature at Gate A)")
    if gate == "B" and count != 1:
        errors.append(f"roadmap must carry the IIAC closure headline exactly once at Gate B (found {count})")


def _check_ordering(repo: Path, ladder_text: str, errors: list[str]) -> None:
    """EBH-7: the pass record is published in an earlier commit than the first CLOSED headline."""
    pass_commit = _oldest_commit(repo, "--diff-filter=A", "--", _PASS_RECORD.as_posix())
    headline_commit = _oldest_commit(repo, f"-S{HEADLINE}", "--", _ROADMAP.as_posix())
    if pass_commit is None or headline_commit is None:
        errors.append("cannot order the IIAC pass record before the roadmap closure headline (history missing)")
        return
    if (
        pass_commit == headline_commit
        or _git(repo, "merge-base", "--is-ancestor", pass_commit, headline_commit)[0] != 0
    ):
        errors.append("the pass record must be published in an earlier commit than the closure headline (EBH-7)")
    formal = ladder_pin(ladder_text, "FORMAL_CLOSURE_PUBLICATION_SHA")
    b6 = ladder_pin(ladder_text, "B6_ACCEPTANCE_PUBLICATION_SHA")
    if formal and _git(repo, "merge-base", "--is-ancestor", headline_commit, formal)[0] != 0:
        errors.append("FORMAL_CLOSURE_PUBLICATION_SHA must contain the closure headline commit")
    if formal and b6 and _git(repo, "merge-base", "--is-ancestor", b6, formal)[0] != 0:
        errors.append("B6_ACCEPTANCE_PUBLICATION_SHA must precede FORMAL_CLOSURE_PUBLICATION_SHA")


def evaluate(
    gate: str,
    repo: Path,
    *,
    iiac_in_force: bool | None = None,
    is_ancestor: Callable[[str], bool] | None = None,
    compensating_verifier: Callable[[Path], list[str]] | None = None,
    live_verifier: Verifier | None = None,
) -> list[str]:
    if gate not in {"A", "B"}:
        raise ValueError("gate must be 'A' or 'B'")
    if iiac_in_force is None:
        iiac_in_force = _iiac_in_force() if repo == _REPO else _iiac_in_force(repo)
    if not iiac_in_force:
        return ["IIAC policy is not IN_FORCE: the historical Protocol 03 path is superseded and not supported here"]

    errors: list[str] = []
    activation = _load_json(repo / _ACTIVATION, "IIAC activation record", errors)
    policy = str((activation or {}).get("policy_version", "")).strip()
    if activation is not None and not policy:
        errors.append("IIAC activation record has no policy_version")
    status = _load_json(repo / _STATUS, "ENT closure status JSON", errors)
    ladder_file = repo / _LADDER
    if not ladder_file.is_file():
        errors.append("missing ENT SHA ladder")
        return errors
    ladder_text = ladder_file.read_text(encoding="utf-8-sig")
    ancestor = is_ancestor or (lambda sha: _git_ancestor(repo, sha))

    _check_evol_predecessor(repo, errors)
    _check_manifest(repo, policy, ladder_text, errors)
    if status is not None:
        if status.get("formal_b6_status") not in {"ACCEPTED", "ACCEPTED_VIA_IIAC_B6"}:
            errors.append("formal_b6_status is not ACCEPTED / ACCEPTED_VIA_IIAC_B6")
        if str(status.get("frz_aud_status", "OPEN")).upper().startswith("OPEN"):
            errors.append("frz_aud_status still OPEN")
        _check_pins(gate, status, ladder_text, ancestor, errors)
        _check_od_ent_02(repo, status.get("od_ent_02_mode"), compensating_verifier, live_verifier, errors)
        _check_frz_aud(repo, status.get("od_ent_02_mode"), errors)
    _check_roadmap(repo, gate, errors)
    if gate == "B" and is_ancestor is None:
        _check_ordering(repo, ladder_text, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("A", "B"), default="A")
    parser.add_argument("--repo-root", type=Path, default=_REPO)
    args = parser.parse_args(argv if argv is not None else [])

    repo = args.repo_root.resolve() if args.repo_root != _REPO else _REPO
    errors = evaluate(args.gate, repo)
    if errors:
        print(f"ENT-AUDIT-X formal closure preflight (Gate {args.gate}): FAIL", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"ENT-AUDIT-X formal closure preflight (Gate {args.gate}): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
