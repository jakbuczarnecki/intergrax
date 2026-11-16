#!/usr/bin/env python3
"""Verify that delegate approval lines in tickets point at a delegation record that resolves (read-only, stdlib only).

Existing validators accept any text after ``Approved:`` (EQO-A finding A6-002, A2-038 operator line; plan
WP-7 V3/V4). For every ticket line of the form

    Approved: operator-delegate <label> @ <ISO-8601 with offset> | Delegation: <repo path>@<40-hex sha>

this verifier checks, and fails with the named code when:

  APPROVAL_TIMESTAMP        the timestamp is not ISO-8601 with an explicit UTC offset
  DELEGATION_MISSING        a delegate line carries no ``| Delegation:`` reference
  DELEGATE_LABEL_MISSING    the delegate line names no delegate session label
  DELEGATION_NOT_A_PATH     the reference is free text (for example a chat remark)
  DELEGATION_UNPINNED       a path without ``@<sha>``
  DELEGATION_UNTRACKED_PATH the path lies under ``.tmp/`` (never tracked, cannot be resolved later)
  DELEGATION_UNRESOLVED     the sha is not a commit, or the path does not exist at that commit
  DELEGATION_NOT_ANCESTOR   the referenced commit is not an ancestor of the commit that added the line
  DELEGATION_SCOPE_MISMATCH the delegation record does not name this ticket
  DELEGATE_NOT_IN_RECORD    the delegate label does not occur in the delegation record
  DELEGATE_IS_IMPLEMENTER   the delegate equals (or shares a component with) the ticket's implementer session
  PERSONAL_WITH_DELEGATION  an ``operator @`` (personal) line points at a delegation file

The implementer comes from a ticket line ``Implementer session: <id>`` or ``--implementer``; when unknown
the check is reported as a warning (IMPLEMENTER_UNKNOWN), and is an error with ``--require-implementer``.
Lines that are neither delegate nor personal-with-delegation, and tickets without an Approved line, are out of
scope here (ticket-approval validators and the legacy-exceptions file belong to PT-E02).

Exit codes: 0 PASS, 1 FAIL, 2 usage or environment error. Writes nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.governance.iiac_verifier_common import (  # noqa: E402
    EXIT_USAGE,
    REPO_ROOT,
    Finding,
    GitUnavailableError,
    commit_exists,
    ensure_git_repo,
    is_ancestor_or_equal,
    is_sha40,
    normalize_identity,
    report,
    run_git,
    show_blob,
)

TOOL = "verify_iiac_delegation_refs"

_APPROVED = re.compile(r"^\s*(?:[-*]\s*)?Approved:\s*(?P<rest>\S.*?)\s*$")
_DELEGATE = re.compile(
    r"^operator-delegate(?:\s+(?P<label>[A-Za-z0-9._-]+))?\s*@\s*(?P<ts>\S+)(?:\s*\|\s*Delegation:\s*(?P<ref>.+?))?(?:\s+\(.*\))?$"
)
_PERSONAL = re.compile(r"^operator(?:\s+[A-Za-z][\w -]*?)?\s*@\s*(?P<ts>\S+)\s*\|\s*Delegation:\s*(?P<ref>.+?)\s*$")
_ISO_OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$")
_IMPLEMENTER = re.compile(r"^\s*(?:[-*]\s*)?Implementer session:\s*(?P<id>\S.*?)\s*$", re.IGNORECASE)
_ID_SPLIT = re.compile(r"[+,;|\s]+")
_PATHLIKE = re.compile(r"^[\w./-]+\.(md|json|txt|yaml|yml)$")


def _timestamp_ok(value: str) -> bool:
    if not _ISO_OFFSET.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _approval_commit(repo: Path, ticket_rel: str, line: str, override: str | None) -> tuple[str, list[Finding]]:
    if override:
        return override, []
    proc = run_git(repo, "log", "--format=%H", f"-S{line.strip()}", "--", ticket_rel)
    hashes = proc.stdout.decode().split() if proc.returncode == 0 else []
    if hashes:
        return hashes[-1], []  # oldest commit that introduced the line
    head = run_git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    return head, [Finding("APPROVAL_COMMIT_UNKNOWN", f"{ticket_rel}: line not found in history, using HEAD", "warn")]


def _scope_names_ticket(record: str, ticket_id: str) -> bool:
    upper = record.upper()
    if ticket_id.upper() in upper:
        return True
    family = re.match(r"^(T-IGX-\d+)-[A-Z0-9]+$", ticket_id.upper())
    return bool(family and f"{family.group(1)}-*" in upper)


def _share_component(a: str, b: str) -> bool:
    parts_a = {t.casefold() for t in _ID_SPLIT.split(a) if len(t) >= 3}
    parts_b = {t.casefold() for t in _ID_SPLIT.split(b) if len(t) >= 3}
    return bool(parts_a & parts_b)


def check_ticket(
    repo: Path,
    ticket: Path,
    *,
    implementer: str | None,
    require_implementer: bool,
    approval_commit: str | None,
) -> list[Finding]:
    try:
        ticket_rel = ticket.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return [Finding("TICKET_OUTSIDE_REPO", f"{ticket} is not inside {repo}")]
    text = ticket.read_text(encoding="utf-8-sig")
    ticket_id = ticket.stem
    impl = implementer
    for raw in text.splitlines():
        m = _IMPLEMENTER.match(raw)
        if m and impl is None:
            impl = m.group("id")

    out: list[Finding] = []
    for raw in text.splitlines():
        approved = _APPROVED.match(raw)
        if not approved:
            continue
        rest = approved.group("rest")
        where = f"{ticket_rel}"
        personal = _PERSONAL.match(rest)
        if personal:
            out.append(
                Finding(
                    "PERSONAL_WITH_DELEGATION",
                    f"{where}: operator line points at a delegation: {personal.group('ref')}",
                )
            )
            continue
        delegate = _DELEGATE.match(rest)
        if not delegate:
            continue  # personal or free-form line: other validators own it
        out += _check_delegate_line(
            repo,
            where,
            ticket_id,
            raw,
            delegate,
            impl,
            require_implementer,
            approval_commit,
            ticket_rel,
        )
    return out


def _check_delegate_line(
    repo: Path,
    where: str,
    ticket_id: str,
    raw_line: str,
    m: re.Match[str],
    implementer: str | None,
    require_implementer: bool,
    approval_commit: str | None,
    ticket_rel: str,
) -> list[Finding]:
    out: list[Finding] = []
    label = m.group("label")
    ref = (m.group("ref") or "").strip()
    if not _timestamp_ok(m.group("ts")):
        out.append(Finding("APPROVAL_TIMESTAMP", f"{where}: {m.group('ts')!r} is not ISO-8601 with an offset"))
    if not label:
        out.append(Finding("DELEGATE_LABEL_MISSING", f"{where}: delegate line names no delegate session"))
    if not ref:
        out.append(Finding("DELEGATION_MISSING", f"{where}: delegate line has no '| Delegation:' reference"))
        return out

    path, sep, sha = ref.rpartition("@")
    if not sep or not is_sha40(sha):
        kind = "DELEGATION_UNPINNED" if _PATHLIKE.match(ref) else "DELEGATION_NOT_A_PATH"
        out.append(Finding(kind, f"{where}: reference {ref!r} is not <repo path>@<40-hex sha>"))
        return out
    path = path.strip().replace("\\", "/")
    if path.startswith(".tmp/") or path.startswith("./.tmp/"):
        out.append(Finding("DELEGATION_UNTRACKED_PATH", f"{where}: {path} lies under .tmp/ and is never tracked"))
        return out
    if not commit_exists(repo, sha):
        out.append(Finding("DELEGATION_UNRESOLVED", f"{where}: {sha[:12]} is not a commit in this repository"))
        return out
    blob = show_blob(repo, sha, path)
    if blob is None:
        out.append(Finding("DELEGATION_UNRESOLVED", f"{where}: {path} does not exist at {sha[:12]}"))
        return out
    record = blob.decode("utf-8", errors="replace")

    commit, extra = _approval_commit(repo, ticket_rel, raw_line, approval_commit)
    out += extra
    if not is_ancestor_or_equal(repo, sha, commit):
        out.append(
            Finding(
                "DELEGATION_NOT_ANCESTOR",
                f"{where}: {sha[:12]} is not an ancestor of the approval commit {commit[:12]}",
            )
        )
    if not _scope_names_ticket(record, ticket_id):
        out.append(Finding("DELEGATION_SCOPE_MISMATCH", f"{where}: {path} does not name {ticket_id}"))
    if label:
        if label.casefold() not in record.casefold():
            out.append(Finding("DELEGATE_NOT_IN_RECORD", f"{where}: delegate {label!r} does not occur in {path}"))
        if implementer:
            if normalize_identity(label) == normalize_identity(implementer) or _share_component(label, implementer):
                out.append(
                    Finding(
                        "DELEGATE_IS_IMPLEMENTER", f"{where}: delegate {label!r} is the implementer {implementer!r}"
                    )
                )
        else:
            level = "error" if require_implementer else "warn"
            out.append(
                Finding(
                    "IMPLEMENTER_UNKNOWN",
                    f"{where}: no 'Implementer session:' line, delegate!=implementer unchecked",
                    level,
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickets", nargs="+", type=Path, help="ticket markdown files inside the repository")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--implementer", default=None, help="implementer session id (overrides the ticket line)")
    parser.add_argument("--require-implementer", action="store_true", help="unknown implementer is an error")
    parser.add_argument(
        "--approval-commit", default=None, help="commit the approval line was added in (default: found in history)"
    )
    args = parser.parse_args(argv)

    for ticket in args.tickets:
        if not ticket.is_file():
            print(f"{TOOL}: usage error: ticket not found: {ticket}", file=sys.stderr)
            return EXIT_USAGE
    findings: list[Finding] = []
    try:
        ensure_git_repo(args.repo)
        for ticket in args.tickets:
            findings += check_ticket(
                args.repo,
                ticket,
                implementer=args.implementer,
                require_implementer=args.require_implementer,
                approval_commit=args.approval_commit,
            )
    except GitUnavailableError as exc:
        print(f"{TOOL}: environment error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return report(TOOL, findings)


if __name__ == "__main__":
    raise SystemExit(main())
