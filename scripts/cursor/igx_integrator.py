#!/usr/bin/env python3
"""Cursor-only least-privilege GitHub PR integration under OD-AF-INT A.

Claude Code does not use this command. The authorized Cursor worker can publish
and merge only exact-SHA, reviewed, scoped PRs. No unrestricted git/gh proxy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = "jakbuczarnecki/intergrax"
DECISION = "Approved: OD-AF-INT A"
TARGETS = {"arch-freeze/rw-train": "product", "development": "docs"}
SHA = re.compile(r"^[0-9a-f]{40}$")
BRANCH = re.compile(r"^(?:af2|integ|feat|fix)/[A-Za-z0-9._/-]{1,110}$")
TICKET = re.compile(r"^governance/tickets/T-IGX-\d{3}-[A-Z]\.md$")
DOC_CI_PROFILE = frozenset(
    {
        "ci-smoke",
        "ticket-governance",
        "governance-policy-pin",
        "governance-fast",
        "evol-x-b4-preflight",
        "ent-audit-fast",
        "ent-audit-pr-gate",
    }
)
CI_PASS = frozenset({"SUCCESS", "PASS", "NEUTRAL"})
CI_PENDING = frozenset({"PENDING", "IN_PROGRESS", "QUEUED", "WAITING"})
CI_SKIP = frozenset({"SKIPPING", "SKIPPED", "CANCELLED"})
PROTECTED = (
    ".claude/",
    "agent-env/",
    ".cursor/",
    "governance/agent-control-plane/",
    "scripts/cursor/",
    "scripts/hooks/",
    "scripts/governance/",
    "governance/TRUSTED_POLICY_PIN.yaml",
    "CLAUDE.md",
    "docs/project/maintainers/qualification/ARCH_FREEZE_SHA_LADDER.md",
    "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md",
    "docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md",
)
PROTECTED_DOC_NAMES = (
    "ROADMAP",
    "OPERATOR_DECISION",
    "FRZ_ACCEPTANCE",
    "IIAC_POLICY_ACTIVATION",
    "FREEZE_CERTIFICATION",
    "ARCH_FREEZE_SHA_LADDER",
    "ARCH_FREEZE_CLOSED",
)
DOC_ROOTS = ("docs/", "governance/tickets/")
PRODUCT_ROOTS = ("intergrax/", "agents/", "applications/", "tests/", "scripts/")


class Blocked(RuntimeError):
    """A fail-closed refusal; do not bypass or silently retry."""


def _decode_bytes(data: bytes, *, strict: bool) -> str:
    try:
        return data.decode("utf-8", errors="strict" if strict else "replace")
    except UnicodeDecodeError as exc:
        raise Blocked("command output is not valid UTF-8") from exc


def run(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, cwd=cwd, capture_output=True, check=False)


def command(argv: list[str], *, cwd: Path | None = None) -> str:
    structured = "--json" in argv
    done = run(argv, cwd=cwd)
    if done.returncode:
        err = _decode_bytes(done.stderr, strict=False).strip()[:250]
        raise Blocked(f"command failed: {argv[0]} {argv[1] if len(argv) > 1 else ''}: {err}")
    if not done.stdout:
        return ""
    text = _decode_bytes(done.stdout, strict=structured)
    if structured:
        try:
            json.loads(text)
        except ValueError as exc:
            raise Blocked("command returned malformed JSON") from exc
    return text.strip()


def gh_api(path: str) -> Any:
    try:
        return json.loads(command(["gh", "api", f"repos/{REPO}/{path}"]))
    except ValueError as exc:
        raise Blocked(f"invalid GitHub response: {path}") from exc


def valid_path(path: str, kind: str) -> bool:
    if not isinstance(path, str) or not path or path.startswith(("/", "\\")) or "\\" in path:
        return False
    if any(part in ("", ".", "..") for part in path.split("/")):
        return False
    if path in PROTECTED or any(path.startswith(p) for p in PROTECTED if p.endswith("/")):
        return False
    if any(part in path.upper() for part in PROTECTED_DOC_NAMES):
        return False
    if kind == "docs":
        return path.startswith(DOC_ROOTS) and not path.endswith(".py")
    return kind == "product" and path.startswith(PRODUCT_ROOTS)


def validate_packet(p: dict[str, Any]) -> None:
    if p.get("operator_decision") != DECISION:
        raise Blocked("OD-AF-INT A not bound to packet")
    if p.get("target") not in TARGETS or p.get("kind") != TARGETS[p["target"]]:
        raise Blocked("target/kind outside approved integration scope")
    branch = p.get("branch")
    if not isinstance(branch, str) or not BRANCH.fullmatch(branch) or branch in TARGETS:
        raise Blocked("invalid worker branch")
    for field in ("head_sha", "base_sha"):
        if not isinstance(p.get(field), str) or not SHA.fullmatch(p[field]):
            raise Blocked(f"invalid {field}")
    if not isinstance(p.get("ticket"), str) or not TICKET.fullmatch(p["ticket"]):
        raise Blocked("missing ticket")
    paths = p.get("write_set")
    if not isinstance(paths, list) or not paths or any(not isinstance(x, str) for x in paths):
        raise Blocked("missing approved write set")
    if len(paths) != len(set(paths)) or any(not valid_path(x, p["kind"]) for x in paths):
        raise Blocked("duplicate or forbidden write-set path")
    if not p.get("implementer") or p.get("implementer") == p.get("reviewer"):
        raise Blocked("implementer/reviewer independence failure")
    if p.get("reviewer_role") != "igx-reviewer" or p.get("review_verdict") not in ("ACCEPT", "ACCEPT_WITH_FINDINGS"):
        raise Blocked("independent reviewer ACCEPT required")
    if p.get("review_head_sha") != p["head_sha"]:
        raise Blocked("review not bound to candidate SHA")
    digest = p.get("review_report_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise Blocked("missing review report digest")
    if not isinstance(p.get("review_report_file"), str) or not p["review_report_file"].startswith(".tmp/proof/"):
        raise Blocked("review report must be under .tmp/proof")


def verify_evidence(p: dict[str, Any], checkout: Path) -> None:
    root = (checkout / ".tmp/proof").resolve()
    proof = (checkout / p["review_report_file"]).resolve()
    if proof == root or root not in proof.parents or not proof.is_file():
        raise Blocked("review evidence missing or outside worktree proof directory")
    if hashlib.sha256(proof.read_bytes()).hexdigest() != p["review_report_sha256"]:
        raise Blocked("review evidence digest mismatch")
    try:
        record = json.loads(proof.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Blocked("review evidence must be JSON") from exc
    if any(
        record.get(k) != p.get(key)
        for k, key in (("head_sha", "head_sha"), ("reviewer", "reviewer"), ("verdict", "review_verdict"))
    ):
        raise Blocked("review record not bound to candidate and reviewer")


def check_ticket(p: dict[str, Any], checkout: Path) -> None:
    ticket = (checkout / p["ticket"]).resolve()
    if checkout.resolve() not in ticket.parents:
        raise Blocked("ticket path escaped repository")
    if ticket.is_file():
        text = ticket.read_text(encoding="utf-8")
    else:
        record = gh_api(f"contents/{p['ticket']}?ref=development")
        if not isinstance(record, dict) or record.get("encoding") != "base64":
            raise Blocked("approved ticket not found in worktree or canonical development")
        try:
            text = base64.b64decode(record["content"]).decode("utf-8")
        except (KeyError, UnicodeError, ValueError) as exc:
            raise Blocked("canonical ticket could not be decoded") from exc
    if not re.search(r"(?m)^Approved: .+", text):
        raise Blocked("ticket lacks Approved line")
    lifecycle = r"(plan_approved|in_progress|quality_gate|ready_for_audit|closed)"
    if not re.search(rf"(?m)^## Lifecycle state\s*\n\s*{lifecycle}\s*$", text):
        raise Blocked("ticket lifecycle does not authorize execution")


def local_worktree(checkout: Path) -> None:
    if not (checkout / ".git").is_file():
        raise Blocked("integrator must run in a separate worktree")
    if not checkout.resolve().as_posix().lower().startswith("d:/projekty/_wt/intergrax/"):
        raise Blocked("worktree outside D:/Projekty/_wt/intergrax/")


def verify_local(p: dict[str, Any], checkout: Path) -> None:
    local_worktree(checkout)
    if command(["git", "branch", "--show-current"], cwd=checkout) != p["branch"]:
        raise Blocked("local branch mismatch")
    if command(["git", "rev-parse", "HEAD"], cwd=checkout) != p["head_sha"]:
        raise Blocked("head SHA moved since review")
    if command(["git", "status", "--porcelain"], cwd=checkout):
        raise Blocked("dirty integration worktree")
    if "jakbuczarnecki/intergrax" not in command(["git", "remote", "get-url", "origin"], cwd=checkout).lower():
        raise Blocked("unexpected git remote")
    command(["git", "merge-base", "--is-ancestor", p["base_sha"], p["head_sha"]], cwd=checkout)
    diff = set(command(["git", "diff", "--name-only", p["base_sha"], p["head_sha"]], cwd=checkout).splitlines())
    if not diff or not diff.issubset(set(p["write_set"])):
        raise Blocked("local diff outside write set")


def target_sha(branch: str) -> str:
    ref = gh_api(f"git/ref/heads/{branch}")
    value = ref.get("object", {}).get("sha", "")
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise Blocked("invalid target ref")
    return value


def branch_head_sha(branch: str) -> str | None:
    try:
        return target_sha(branch)
    except Blocked:
        return None


def list_pr_files(number: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for page in range(1, 12):
        files = gh_api(f"pulls/{number}/files?per_page=100&page={page}")
        if not isinstance(files, list):
            raise Blocked("invalid changed-file listing")
        out.extend(files)
        if len(files) < 100:
            return out
    raise Blocked("too many PR files for bounded verification")


def verify_pr(p: dict[str, Any], number: int) -> None:
    pr = gh_api(f"pulls/{number}")
    if pr.get("state") != "open" or pr.get("draft") or pr.get("merged"):
        raise Blocked("PR not open and ready")
    if pr.get("head", {}).get("sha") != p["head_sha"] or pr.get("head", {}).get("ref") != p["branch"]:
        raise Blocked("PR head changed")
    if pr.get("base", {}).get("ref") != p["target"] or pr.get("base", {}).get("sha") != p["base_sha"]:
        raise Blocked("PR base changed")
    files = list_pr_files(number)
    changed = {f.get("filename") for f in files}
    if not changed or not changed.issubset(set(p["write_set"])) or any(not valid_path(x, p["kind"]) for x in changed):
        raise Blocked("PR changed paths outside scope")
    if p["kind"] == "docs":
        for f in files:
            if f.get("filename", "").startswith("governance/tickets/") and not isinstance(f.get("patch"), str):
                raise Blocked("ticket patch unavailable: approval changes cannot be ruled out")
            if re.search(r"(?m)^[+-]Approved:\s*", f.get("patch") or ""):
                raise Blocked("integrator cannot add/remove ticket approval lines")
    if target_sha(p["target"]) != p["base_sha"]:
        raise Blocked("target SHA moved")


def parse_pr_checks(argv: list[str]) -> list[dict[str, Any]]:
    done = run(argv)
    if done.returncode:
        err = _decode_bytes(done.stderr, strict=False).lower()
        if "--required" in argv and "no required checks" in err:
            return []
        msg = _decode_bytes(done.stderr, strict=False).strip()[:250]
        raise Blocked(f"CI query failed: {msg}")
    if not done.stdout:
        raise Blocked("CI query returned no data")
    text = _decode_bytes(done.stdout, strict=True)
    try:
        checks = json.loads(text)
    except ValueError as exc:
        raise Blocked("invalid CI response") from exc
    if not isinstance(checks, list):
        raise Blocked("invalid CI response shape")
    return checks


def qualify_ci(number: int, kind: str) -> None:
    required = parse_pr_checks(["gh", "pr", "checks", str(number), "--required", "--json", "name,state", "-R", REPO])
    for item in required:
        state = item.get("state")
        if state in CI_SKIP:
            raise Blocked("required checks red, pending, or skipped")
        if state in CI_PENDING or state not in CI_PASS:
            raise Blocked("required checks red, pending, or skipped")

    all_checks = parse_pr_checks(["gh", "pr", "checks", str(number), "--json", "name,state", "-R", REPO])
    if not all_checks:
        raise Blocked("CI check information unavailable")

    by_name = {c.get("name"): c.get("state") for c in all_checks if isinstance(c.get("name"), str)}

    if kind == "docs":
        for name in DOC_CI_PROFILE:
            state = by_name.get(name)
            if state is None:
                raise Blocked(f"mandatory documentation check missing: {name}")
            if state in CI_SKIP:
                raise Blocked(f"mandatory documentation check skipped: {name}")
            if state in CI_PENDING or state not in CI_PASS:
                raise Blocked(f"mandatory documentation check not successful: {name}")
    elif not required:
        raise Blocked("no required CI checks: cannot prove safe merge")

    for item in all_checks:
        state = item.get("state")
        if not isinstance(state, str):
            raise Blocked("CI check information unavailable")
        if state in CI_SKIP:
            continue
        if state in CI_PENDING:
            raise Blocked("CI checks still pending")
        if state not in CI_PASS:
            raise Blocked("applicable CI check failed")


def required_checks(number: int, *, kind: str = "product") -> None:
    qualify_ci(number, kind)


def open_prs_for_branch(branch: str, target: str) -> list[dict[str, Any]]:
    raw = command(
        [
            "gh",
            "pr",
            "list",
            "-R",
            REPO,
            "--head",
            branch,
            "--base",
            target,
            "--state",
            "open",
            "--json",
            "number,headRefOid,baseRefName,headRefName,state",
        ]
    )
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise Blocked("invalid open PR listing") from exc
    if not isinstance(data, list):
        raise Blocked("invalid open PR listing shape")
    return data


def publish(p: dict[str, Any], checkout: Path) -> None:
    verify_local(p, checkout)
    if target_sha(p["target"]) != p["base_sha"]:
        raise Blocked("target moved; requalify packet")
    title, body = p.get("pr_title"), p.get("pr_body_file")
    if not isinstance(title, str) or not (8 <= len(title) <= 160):
        raise Blocked("missing PR title")
    if not isinstance(body, str) or not body.startswith(".tmp/proof/"):
        raise Blocked("missing PR body")
    path = (checkout / body).resolve()
    if (checkout / ".tmp/proof").resolve() not in path.parents or not path.is_file():
        raise Blocked("PR body file missing")

    remote_sha = branch_head_sha(p["branch"])
    if remote_sha and remote_sha != p["head_sha"]:
        try:
            command(["git", "merge-base", "--is-ancestor", remote_sha, p["head_sha"]], cwd=checkout)
        except Blocked as exc:
            raise Blocked("remote branch diverged") from exc
    if remote_sha != p["head_sha"]:
        command(["git", "push", "origin", f"{p['head_sha']}:refs/heads/{p['branch']}"], cwd=checkout)

    matches = [pr for pr in open_prs_for_branch(p["branch"], p["target"]) if pr.get("state") == "OPEN"]
    if len(matches) > 1:
        raise Blocked("duplicate open PRs for integration branch")
    if len(matches) == 1:
        pr = matches[0]
        number = pr.get("number")
        head = pr.get("headRefOid")
        if not isinstance(number, int) or not isinstance(head, str) or head != p["head_sha"]:
            raise Blocked("existing PR head does not match candidate")
        if pr.get("baseRefName") != p["target"] or pr.get("headRefName") != p["branch"]:
            raise Blocked("existing PR branch target mismatch")
        print(json.dumps({"result": "PUBLISHED", "pr": number, "reused": True}))
        return

    command(
        [
            "gh",
            "pr",
            "create",
            "-R",
            REPO,
            "--base",
            p["target"],
            "--head",
            p["branch"],
            "--title",
            title,
            "--body-file",
            str(path),
        ],
        cwd=checkout,
    )
    created = open_prs_for_branch(p["branch"], p["target"])
    if len(created) != 1:
        raise Blocked("could not confirm created PR identity")
    number = created[0].get("number")
    if not isinstance(number, int):
        raise Blocked("could not confirm created PR identity")
    print(json.dumps({"result": "PUBLISHED", "pr": number, "reused": False}))


def merge(p: dict[str, Any]) -> None:
    number = p.get("pr_number")
    if not isinstance(number, int) or number <= 0:
        raise Blocked("missing PR number")
    kind = p.get("kind")
    if kind not in TARGETS.values():
        raise Blocked("invalid integration kind")
    verify_pr(p, number)
    required_checks(number, kind=kind)
    verify_pr(p, number)
    command(["gh", "pr", "merge", str(number), "--merge", "--match-head-commit", p["head_sha"], "-R", REPO])
    final = gh_api(f"pulls/{number}")
    merged_sha = final.get("merge_commit_sha")
    if not final.get("merged") or not isinstance(merged_sha, str) or not SHA.fullmatch(merged_sha):
        raise Blocked("merge not independently confirmed by GitHub")
    print(json.dumps({"result": "MERGED", "pr": number, "sha": merged_sha}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "publish", "merge"))
    parser.add_argument("--packet", type=Path, required=True)
    a = parser.parse_args(argv)
    try:
        p = json.loads(a.packet.read_text(encoding="utf-8"))
        if not isinstance(p, dict):
            raise Blocked("packet must be a JSON object")
        validate_packet(p)
        cwd = Path.cwd()
        local_worktree(cwd)
        verify_evidence(p, cwd)
        check_ticket(p, cwd)
        if a.action == "publish":
            publish(p, cwd)
        elif a.action == "merge":
            merge(p)
        else:
            print(json.dumps({"result": "VALIDATED", "target": p["target"], "head_sha": p["head_sha"]}))
        return 0
    except (Blocked, ValueError, OSError, TypeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
