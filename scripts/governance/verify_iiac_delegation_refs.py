#!/usr/bin/env python3
"""Verify that delegate approval lines in tickets point at a delegation record that resolves (read-only, stdlib only).

Existing validators accept any text after ``Approved:`` (EQO-A finding A6-002, A2-038 operator line; plan
WP-7 V3/V4). Two forms of the delegate line exist and both are read:

  documented two-line form (``governance/tickets/README.md``, item 3), the form of most tickets:

      Approved: operator-delegate @ <ISO-8601 with offset>
      Delegation: <repo path>@<40-hex sha>          <- must be the line DIRECTLY after

  single-line form (older tickets, optional delegate label):

      Approved: operator-delegate <label> @ <ISO-8601 with offset> | Delegation: <repo path>@<40-hex sha>

The delegate label is optional in both forms. A label, when present, is checked against the record and
the implementer. When there is no label the delegate identity is not on the line and is not machine-readable in
the record, so ``delegate != implementer`` cannot be proven: that is reported as DELEGATE_IDENTITY_UNKNOWN
(a warning; an error with ``--require-implementer``). What IS checked without a label: the reference resolves,
its commit is an ancestor of the commit that added the line, and the record names this ticket.

Findings (error unless marked warn):

  APPROVAL_TIMESTAMP        the timestamp is not ISO-8601 with an explicit UTC offset
  DELEGATION_MISSING        a delegate line carries no Delegation reference at all
  DELEGATION_NOT_ADJACENT   two-line form: a Delegation line exists but not directly after the Approved line
  DELEGATION_NOT_A_PATH     the reference is free text (for example a chat remark)
  DELEGATION_UNPINNED       a path without ``@<sha>``
  DELEGATION_UNTRACKED_PATH the path lies under ``.tmp/`` (never tracked, cannot be resolved later)
  DELEGATION_UNRESOLVED     the sha is not a commit, or the path does not exist at that commit
  DELEGATION_NOT_ANCESTOR   the referenced commit is not an ancestor of HEAD (never published in the checked history)
  DELEGATION_OUTSIDE_APPROVAL_HISTORY warn: an ancestor of HEAD but not of the commit that added the line (the
                            record was published on another branch or later; PR-branch workflows do this)
  DELEGATION_SCOPE_MISMATCH the record does not name this ticket (exact ticket id, or its ``T-IGX-NNN-*`` family)
  DELEGATE_NOT_IN_RECORD    a delegate label does not occur in the record as a whole token
  DELEGATE_IS_IMPLEMENTER   the delegate label equals (or shares a component with) the implementer session
  DELEGATE_IDENTITY_UNKNOWN warn: no label, so delegate != implementer is unprovable (see above)
  IMPLEMENTER_UNKNOWN       warn: no ``Implementer session:`` line and no ``--implementer``
  PERSONAL_WITH_DELEGATION  an ``operator @`` (personal) line points at a delegation, inline or on the next line

Matching limits (documented, not hidden): token matching is case-insensitive and treats ``[A-Za-z0-9._-]`` as
word characters, so ``signer-one`` does not match inside ``signer-one-b`` but does match ``signer-one:``; the scope
check cannot tell a record that merely mentions the ticket id from one that delegates it.

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
    reject_option_like,
    report,
    run_git,
    show_blob,
)

TOOL = "verify_iiac_delegation_refs"

_APPROVED = re.compile(r"^\s*(?:[-*]\s*)?Approved:\s*(?P<rest>\S.*?)\s*$")
_DELEGATION_LINE = re.compile(r"^\s*(?:[-*]\s*)?Delegation:\s*(?P<ref>\S.*?)\s*$")
_DELEGATE = re.compile(
    r"^operator-delegate(?:\s+(?P<label>[A-Za-z0-9._-]+))?\s*@\s*(?P<ts>\S+)"
    r"(?:\s*\|\s*Delegation:\s*(?P<ref>.+?))?(?:\s+\(.*\))?$"
)
_PERSONAL = re.compile(
    r"^operator(?:\s+[A-Za-z][\w -]*?)?\s*@\s*(?P<ts>\S+)(?:\s*\|\s*Delegation:\s*(?P<ref>.+?))?(?:\s+\(.*\))?\s*$"
)
_ISO_OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$")
_IMPLEMENTER = re.compile(r"^\s*(?:[-*]\s*)?Implementer session:\s*(?P<id>\S.*?)\s*$", re.IGNORECASE)
_ID_SPLIT = re.compile(r"[+,;|\s]+")
_PATHLIKE = re.compile(r"^[\w./-]+\.(md|json|txt|yaml|yml)$")
_TOKEN_CHARS = r"A-Za-z0-9._-"


def _timestamp_ok(value: str) -> bool:
    if not _ISO_OFFSET.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _has_token(text: str, token: str) -> bool:
    """Whole-token, case-insensitive match.

    A neighbouring ``[A-Za-z0-9_-]`` character, or a ``.`` followed by a word character, continues the token; a
    sentence-final ``.`` or the ``..`` of a range does not.
    """
    pattern = rf"(?<![{_TOKEN_CHARS}])" + re.escape(token) + r"(?![A-Za-z0-9_-])(?!\.[A-Za-z0-9_])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _in_named_range(record: str, ticket_id: str) -> bool:
    """Ranges such as ``T-IGX-043-A..F`` name every single-letter ticket suffix between the bounds."""
    match = re.match(r"^(T-IGX-\d+-)([A-Z])$", ticket_id.upper())
    if not match:
        return False
    prefix, letter = match.groups()
    pattern = rf"{re.escape(prefix)}([A-Z])\s*\.\.\s*(?:{re.escape(prefix)})?([A-Z])(?![A-Za-z0-9_-])"
    for found in re.finditer(pattern, record.upper()):
        if found.group(1) <= letter <= found.group(2):
            return True
    return False


def _scope_names_ticket(record: str, ticket_id: str) -> bool:
    if _has_token(record, ticket_id) or _in_named_range(record, ticket_id):
        return True
    family = re.match(r"^(T-IGX-\d+)-[A-Z0-9]+$", ticket_id.upper())
    return bool(family and _has_token(record, f"{family.group(1)}-*"))


def _share_component(a: str, b: str) -> bool:
    parts_a = {t.casefold() for t in _ID_SPLIT.split(a) if len(t) >= 3}
    parts_b = {t.casefold() for t in _ID_SPLIT.split(b) if len(t) >= 3}
    return bool(parts_a & parts_b)


def _approval_commit(repo: Path, ticket_rel: str, line: str, override: str | None) -> tuple[str, list[Finding]]:
    if override:
        return override, []
    proc = run_git(repo, "log", options=("--format=%H", f"-S{line.strip()}"), pathspec=(ticket_rel,))
    hashes = proc.stdout.decode().split() if proc.returncode == 0 else []
    if hashes:
        return hashes[-1], []  # oldest commit that introduced the line
    head = run_git(repo, "rev-parse", operands=("HEAD",)).stdout.decode().strip()
    return head, [Finding("APPROVAL_COMMIT_UNKNOWN", f"{ticket_rel}: line not found in history, using HEAD", "warn")]


def _delegation_after(lines: list[str], index: int) -> tuple[str | None, bool]:
    """Return (reference on the line directly after ``index``, whether a Delegation line exists later in the file)."""
    if index + 1 < len(lines):
        match = _DELEGATION_LINE.match(lines[index + 1])
        if match:
            return match.group("ref"), True
    later = any(_DELEGATION_LINE.match(line) for line in lines[index + 2 :])
    return None, later


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
    lines = ticket.read_text(encoding="utf-8-sig").splitlines()
    impl = implementer
    if impl is None:
        for raw in lines:
            found = _IMPLEMENTER.match(raw)
            if found:
                impl = found.group("id")
                break

    out: list[Finding] = []
    for index, raw in enumerate(lines):
        approved = _APPROVED.match(raw)
        if not approved:
            continue
        rest = approved.group("rest")
        next_ref, later_exists = _delegation_after(lines, index)
        personal = _PERSONAL.match(rest)
        if personal and not rest.startswith("operator-delegate"):
            ref = personal.group("ref") or next_ref
            if ref:
                out.append(
                    Finding("PERSONAL_WITH_DELEGATION", f"{ticket_rel}: operator line points at a delegation: {ref}")
                )
            continue
        delegate = _DELEGATE.match(rest)
        if not delegate:
            continue  # free-form line: other validators own it
        inline = (delegate.group("ref") or "").strip()
        ref = inline or (next_ref or "")
        out += _check_delegate_line(
            repo,
            ticket_rel,
            ticket.stem,
            raw,
            delegate.group("label"),
            delegate.group("ts"),
            ref,
            later_exists and not inline and not next_ref,
            impl,
            require_implementer,
            approval_commit,
        )
    return out


def _check_delegate_line(
    repo: Path,
    where: str,
    ticket_id: str,
    raw_line: str,
    label: str | None,
    timestamp: str,
    ref: str,
    delegation_not_adjacent: bool,
    implementer: str | None,
    require_implementer: bool,
    approval_commit: str | None,
) -> list[Finding]:
    out: list[Finding] = []
    if not _timestamp_ok(timestamp):
        out.append(Finding("APPROVAL_TIMESTAMP", f"{where}: {timestamp!r} is not ISO-8601 with an offset"))
    if not ref:
        kind = "DELEGATION_NOT_ADJACENT" if delegation_not_adjacent else "DELEGATION_MISSING"
        detail = (
            "a Delegation line exists but not directly after the Approved line"
            if delegation_not_adjacent
            else "delegate line has no Delegation reference"
        )
        out.append(Finding(kind, f"{where}: {detail}"))
        return out

    path, sep, sha = ref.rpartition("@")
    if not sep or not is_sha40(sha):
        kind = "DELEGATION_UNPINNED" if _PATHLIKE.match(ref) else "DELEGATION_NOT_A_PATH"
        out.append(Finding(kind, f"{where}: reference {ref!r} is not <repo path>@<40-hex sha>"))
        return out
    path = path.strip().replace("\\", "/")
    if path.startswith((".tmp/", "./.tmp/")):
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

    approval_line = raw_line.split("|", 1)[0].strip() if "|" in raw_line else raw_line.strip()
    commit, extra = _approval_commit(repo, where, approval_line, approval_commit)
    out += extra
    head = run_git(repo, "rev-parse", operands=("HEAD",)).stdout.decode().strip()
    if not is_ancestor_or_equal(repo, sha, head):
        out.append(Finding("DELEGATION_NOT_ANCESTOR", f"{where}: {sha[:12]} is not an ancestor of HEAD {head[:12]}"))
    elif not is_ancestor_or_equal(repo, sha, commit):
        out.append(
            Finding(
                "DELEGATION_OUTSIDE_APPROVAL_HISTORY",
                f"{where}: {sha[:12]} is in the history of HEAD but not of the approval commit {commit[:12]}",
                "warn",
            )
        )
    if not _scope_names_ticket(record, ticket_id):
        out.append(Finding("DELEGATION_SCOPE_MISMATCH", f"{where}: {path} does not name {ticket_id}"))
    if label:
        if not _has_token(record, label):
            out.append(Finding("DELEGATE_NOT_IN_RECORD", f"{where}: delegate {label!r} does not occur in {path}"))
        if implementer and (
            normalize_identity(label) == normalize_identity(implementer) or _share_component(label, implementer)
        ):
            out.append(
                Finding("DELEGATE_IS_IMPLEMENTER", f"{where}: delegate {label!r} is the implementer {implementer!r}")
            )
    else:
        level = "error" if require_implementer else "warn"
        out.append(
            Finding(
                "DELEGATE_IDENTITY_UNKNOWN",
                f"{where}: no delegate label on the line, delegate != implementer cannot be proven",
                level,
            )
        )
    if implementer is None:
        level = "error" if require_implementer else "warn"
        out.append(
            Finding(
                "IMPLEMENTER_UNKNOWN",
                f"{where}: no 'Implementer session:' line, delegate != implementer unchecked",
                level,
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickets", nargs="+", type=Path, help="ticket markdown files inside the repository")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--implementer", default=None, help="implementer session id (overrides the ticket line)")
    parser.add_argument(
        "--require-implementer",
        action="store_true",
        help="unknown implementer or unknown delegate identity is an error",
    )
    parser.add_argument(
        "--approval-commit", default=None, help="commit the approval line was added in (default: found in history)"
    )
    args = parser.parse_args(argv)

    for ticket in args.tickets:
        if not ticket.is_file():
            print(f"{TOOL}: usage error: ticket not found: {ticket}", file=sys.stderr)
            return EXIT_USAGE
    try:
        if args.approval_commit:
            reject_option_like(args.approval_commit)
        findings: list[Finding] = []
        ensure_git_repo(args.repo)
        for ticket in args.tickets:
            findings += check_ticket(
                args.repo,
                ticket,
                implementer=args.implementer,
                require_implementer=args.require_implementer,
                approval_commit=args.approval_commit,
            )
    except ValueError as exc:
        print(f"{TOOL}: usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except GitUnavailableError as exc:
        print(f"{TOOL}: environment error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return report(TOOL, findings)


if __name__ == "__main__":
    raise SystemExit(main())
