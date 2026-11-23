#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Strict verifier for ticket approval lines and delegation references (EQO-A PT-E02, finding EQOA-A6-002).

Modes (exactly one):

* ``--since <base>``  tickets added or whose approval block changed in ``merge-base(<base>, HEAD)..HEAD``. Legacy
  exceptions are read from the merge-base (a pull request cannot excuse its own ticket by editing the exceptions
  file) and apply only to tickets that already existed there. This is the mode of the ``ticket-governance`` job.
* ``--all``           every tracked ticket at HEAD (read-only replay, used by node PT-V06). Applied exceptions are
  printed one by one and counted; stale exception entries are errors.
* ``--ticket <path>`` one ticket from the working tree (approval treated as added at HEAD).

The default behaviour of ``verify_plan_approval.py`` and ``verify_ticket_lifecycle.py`` is not changed by this
script. Read ``scripts/governance/ticket_approval.py`` for the grammar and the honest limits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.governance.ticket_approval import (
    EXCEPTIONS_REL,
    TICKET_NAME_RE,
    Git,
    TicketResult,
    approval_lines,
    check_ticket,
    legacy_approval_digest,
    load_exceptions,
    parse_state,
)

_REPO = Path(__file__).resolve().parents[2]


def _signature(text: str) -> tuple[str, ...]:
    """The approval line plus the two lines below it (the delegate block), or () when absent."""

    lines = text.replace("\r\n", "\n").split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("Approved:"):
            return tuple(lines[i : i + 3])
    return ()


def _approval_commit(git: Git, base: str, path: str, head_text: str) -> str:
    head_sig = _signature(head_text)
    log = git.out("log", "--reverse", "--format=%H", f"{base}..HEAD", "--", path) or ""
    for commit in (ln.strip() for ln in log.splitlines() if ln.strip()):
        blob = git.show(commit, path)
        if blob is not None and _signature(blob) == head_sig:
            return commit
    return git.rev_parse("HEAD") or "HEAD"


def _changed_tickets(git: Git, base: str) -> list[str]:
    diff = git.out("diff", "--name-only", "--diff-filter=AMR", f"{base}..HEAD", "--", "governance/tickets/")
    if diff is None:
        raise RuntimeError(f"git diff failed for {base}..HEAD")
    return sorted(p for p in (ln.strip() for ln in diff.splitlines()) if TICKET_NAME_RE.match(p))


def run_since(git: Git, base_ref: str) -> list[TicketResult]:
    head = git.rev_parse("HEAD")
    base_commit = git.rev_parse(base_ref)
    if not head or not base_commit:
        raise RuntimeError(f"cannot resolve HEAD or base '{base_ref}' (is the history fetched?)")
    merge_base = (git.out("merge-base", base_commit, head) or "").strip()
    if not merge_base:
        raise RuntimeError(f"no merge-base between '{base_ref}' and HEAD (shallow clone?)")
    exceptions_text = git.show(merge_base, EXCEPTIONS_REL)
    exceptions, exc_errors = load_exceptions(exceptions_text)
    results: list[TicketResult] = []
    if exc_errors:
        bad = TicketResult(EXCEPTIONS_REL)
        bad.errors.extend(exc_errors)
        results.append(bad)
    for path in _changed_tickets(git, merge_base):
        text = git.show("HEAD", path)
        if text is None:
            continue
        before = git.show(merge_base, path)
        if before is not None and _signature(before) == _signature(text) and parse_state(before) == parse_state(text):
            res = TicketResult(path)
            res.notes.append("approval block and lifecycle state unchanged since the merge-base (not re-validated)")
            results.append(res)
            continue
        commit = _approval_commit(git, merge_base, path, text)
        results.append(
            check_ticket(git, path, text, commit, exceptions, allow_exceptions=before is not None),
        )
    return results


def run_all(git: Git) -> list[TicketResult]:
    head = git.rev_parse("HEAD") or "HEAD"
    listing = git.out("ls-tree", "-r", "--name-only", "HEAD", "governance/tickets/") or ""
    paths = sorted(p for p in (ln.strip() for ln in listing.splitlines()) if TICKET_NAME_RE.match(p))
    exceptions, exc_errors = load_exceptions(git.show("HEAD", EXCEPTIONS_REL))
    results: list[TicketResult] = []
    if exc_errors:
        bad = TicketResult(EXCEPTIONS_REL)
        bad.errors.extend(exc_errors)
        results.append(bad)
    for path in paths:
        text = git.show("HEAD", path)
        if text is None:
            continue
        results.append(check_ticket(git, path, text, head, exceptions, allow_exceptions=True))
    known = set(paths)
    for ticket, entry in sorted(exceptions.items()):
        stale = TicketResult(f"{EXCEPTIONS_REL}#{ticket}")
        if ticket not in known:
            stale.errors.append("stale exception: the ticket is not tracked at HEAD")
        else:
            text = git.show("HEAD", ticket) or ""
            if entry["kind"] == "no_approval_line" and approval_lines(text):
                stale.errors.append("stale exception: the ticket now has an approval line (it must validate by itself)")
            if entry["kind"] == "unresolvable_delegation_reference" and entry.get(
                "approval_block_sha256"
            ) != legacy_approval_digest(text):
                stale.errors.append("stale exception: the approval lines changed since the entry was written")
        if stale.errors:
            results.append(stale)
    return results


def run_ticket(git: Git, root: Path, ticket: Path) -> list[TicketResult]:
    path = ticket if ticket.is_absolute() else root / ticket
    if not path.is_file():
        bad = TicketResult(str(ticket))
        bad.errors.append("ticket file not found")
        return [bad]
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    exc_path = root / EXCEPTIONS_REL
    exceptions, exc_errors = load_exceptions(exc_path.read_text(encoding="utf-8") if exc_path.is_file() else None)
    results: list[TicketResult] = []
    if exc_errors:
        bad = TicketResult(EXCEPTIONS_REL)
        bad.errors.extend(exc_errors)
        results.append(bad)
    head = git.rev_parse("HEAD") or "HEAD"
    results.append(
        check_ticket(git, rel, path.read_text(encoding="utf-8"), head, exceptions, allow_exceptions=True),
    )
    return results


def report(results: list[TicketResult]) -> int:
    failures = [r for r in results if not r.ok]
    excepted = [r for r in results if r.exception_applied]
    for res in results:
        status = "FAIL" if not res.ok else ("EXCEPTION" if res.exception_applied else "ok")
        print(f"[{status}] {res.ticket}")
        for err in res.errors:
            print(f"    - {err}", file=sys.stderr if not res.ok else sys.stdout)
        for note in res.notes:
            if status != "ok" or "unchanged" in note or "EXCEPTION" in note:
                print(f"    . {note}")
    print(
        f"[INFO] verify_ticket_approvals: {len(results)} checked, {len(failures)} failed, "
        f"{len(excepted)} accepted ONLY through the legacy exceptions file"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strict verification of ticket approval lines and delegation references."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--since", metavar="BASE", help="tickets added or approval-changed since merge-base(BASE, HEAD)")
    mode.add_argument("--all", action="store_true", help="read-only replay over every tracked ticket at HEAD")
    mode.add_argument("--ticket", type=Path, help="one ticket from the working tree")
    parser.add_argument("--repo", type=Path, default=_REPO, help="repository root (tests only)")
    parser.add_argument("--json", type=Path, help="also write the result as JSON to this path (use a .tmp/ path)")
    args = parser.parse_args(argv)

    root = args.repo.resolve()
    git = Git(root)
    try:
        if args.since:
            results = run_since(git, args.since)
        elif args.all:
            results = run_all(git)
        else:
            results = run_ticket(git, root, args.ticket)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [
                    {
                        "ticket": r.ticket,
                        "ok": r.ok,
                        "errors": r.errors,
                        "notes": r.notes,
                        "exception_applied": r.exception_applied,
                    }
                    for r in results
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return report(results)


if __name__ == "__main__":
    raise SystemExit(main())
