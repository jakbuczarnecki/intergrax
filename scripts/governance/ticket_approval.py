# © Artur Czarnecki. All rights reserved.

"""Strict ticket-approval grammar and delegation-reference resolver (EQO-A PT-E02, finding EQOA-A6-002).

This module is NEW and is used only by ``verify_ticket_approvals.py``. ``verify_plan_approval.py`` and
``verify_ticket_lifecycle.py`` do not call it: their legacy default (any line starting ``Approved:``) is unchanged.

What the strict mode checks, mechanically:

* exactly one ``Approved:`` line, in one of two forms (``operator`` or ``operator-delegate``), with an ISO 8601 time
  that carries an offset;
* for the delegate form: a ``Delegation: <repo-relative path>@<sha40>`` line directly below, a ``Delegate session:``
  line below that, and an ``Implementer session:`` field in the ticket that differs from the delegate session;
* the delegation record is a TRACKED file at that sha (``git cat-file -e``), the sha is a strict ancestor of the
  commit that adds the approval, and the record text names the ticket (exact id, ``X..Y`` range or ``NNN-*``);
* the lifecycle state requires an approval when it is a post-plan state.

What it cannot check (honest limit, label kept as ``process_enforced``): who actually typed the line, and whether a
session id was copied from another session. Host session ids are self-asserted by the writer of the line.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

POST_PLAN_STATES = frozenset({"plan_approved", "in_progress", "quality_gate", "ready_for_audit", "closed"})
PRE_PLAN_STATES = frozenset({"draft", "plan_pending"})

EXCEPTIONS_SCHEMA = "intergrax.ticket_legacy_approval_exceptions.v1"
EXCEPTIONS_REL = "governance/tickets/legacy_approval_exceptions.json"
EXCEPTION_KINDS = frozenset({"no_approval_line", "unresolvable_delegation_reference"})

TICKET_NAME_RE = re.compile(r"^governance/tickets/(T-IGX-[0-9]{3}-[A-Z])\.md$")
_APPROVED_ANY_RE = re.compile(r"^Approved:", re.MULTILINE)
_STATE_RE = re.compile(r"^##\s+Lifecycle state\s*\n+\s*(\w+)\s*$", re.MULTILINE | re.IGNORECASE)
_TS = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})"
_PERSONAL_RE = re.compile(rf"^Approved: operator @ ({_TS})$")
_DELEGATE_RE = re.compile(rf"^Approved: operator-delegate @ ({_TS})$")
_DELEGATION_RE = re.compile(r"^Delegation: ([^\s@|]+)@([0-9a-f]{40})$")
_SESSION_ID = r"(?:local|cloud|remote)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_DELEGATE_SESSION_RE = re.compile(rf"^Delegate session: ({_SESSION_ID})$")
_IMPLEMENTER_SESSION_RE = re.compile(r"^Implementer session:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
_IMPLEMENTER_SESSION_VALUE_RE = re.compile(rf"^{_SESSION_ID}$")
_TICKET_ID_RE = re.compile(r"T-IGX-([0-9]{3})-([A-Z])(?:\.\.([A-Z])|\*)?")
_BATCH_WILDCARD_RE = re.compile(r"T-IGX-([0-9]{3})-\*")


@dataclass(frozen=True)
class ApprovalBlock:
    kind: str  # "personal" | "delegate"
    timestamp: str
    delegation_path: str | None
    delegation_sha: str | None
    delegate_session: str | None
    raw: str  # exact text of the approval lines (used to bind legacy exceptions)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.raw.encode("utf-8")).hexdigest()


@dataclass
class TicketResult:
    ticket: str
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    exception_applied: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_state(text: str) -> str | None:
    match = _STATE_RE.search(text)
    return match.group(1).strip() if match else None


def approval_lines(text: str) -> list[str]:
    return [ln.rstrip("\r") for ln in text.replace("\r\n", "\n").split("\n") if _APPROVED_ANY_RE.match(ln)]


def parse_approval_block(text: str) -> tuple[ApprovalBlock | None, list[str]]:
    """Parse the approval block of a ticket. Returns (block, errors); block is None when absent or invalid."""

    lines = text.replace("\r\n", "\n").split("\n")
    idx = [i for i, ln in enumerate(lines) if _APPROVED_ANY_RE.match(ln)]
    if not idx:
        return None, []
    if len(idx) > 1:
        return None, [f"{len(idx)} lines start with 'Approved:' (exactly one is allowed)"]
    head = lines[idx[0]]
    personal = _PERSONAL_RE.match(head)
    delegate = _DELEGATE_RE.match(head)
    if not personal and not delegate:
        return None, [
            "approval line does not match 'Approved: operator @ <ISO8601 with offset>' or "
            "'Approved: operator-delegate @ <ISO8601 with offset>'"
        ]
    stamp = (personal or delegate).group(1)  # type: ignore[union-attr]
    try:
        datetime.fromisoformat(stamp)
    except ValueError:
        return None, [f"approval timestamp is not a valid ISO 8601 time: {stamp}"]
    if personal:
        following = next((ln for ln in lines[idx[0] + 1 : idx[0] + 2]), "")
        if following.startswith(("Delegation:", "Delegate session:")):
            return None, ["personal-form approval must not carry Delegation:/Delegate session: lines"]
        return ApprovalBlock("personal", stamp, None, None, None, head), []
    errors: list[str] = []
    deleg_line = lines[idx[0] + 1] if idx[0] + 1 < len(lines) else ""
    sess_line = lines[idx[0] + 2] if idx[0] + 2 < len(lines) else ""
    dm = _DELEGATION_RE.match(deleg_line)
    sm = _DELEGATE_SESSION_RE.match(sess_line)
    if not dm:
        errors.append(
            "delegate approval needs 'Delegation: <repo-relative path>@<40-hex sha>' on the line directly below"
        )
    if not sm:
        errors.append(
            "delegate approval needs 'Delegate session: <host session id>' on the line below the Delegation line"
        )
    if errors:
        return None, errors
    assert dm is not None and sm is not None
    raw = "\n".join([head, deleg_line, sess_line])
    return ApprovalBlock("delegate", stamp, dm.group(1), dm.group(2), sm.group(1), raw), []


def implementer_session(text: str) -> tuple[str | None, str | None]:
    """Return (value, error) of the ``Implementer session:`` field."""

    found = _IMPLEMENTER_SESSION_RE.findall(text.replace("\r\n", "\n"))
    if not found:
        return None, "ticket has no 'Implementer session: <host session id>' field"
    if len(found) > 1:
        return None, "ticket has more than one 'Implementer session:' field"
    if not _IMPLEMENTER_SESSION_VALUE_RE.match(found[0]):
        return None, "'Implementer session:' value is not a host session id (expected local_/cloud_/remote_<uuid>)"
    return found[0], None


def normalize_record_path(path: str) -> str | None:
    """Return the normalised repo-relative path, or None when it is not an acceptable tracked-file reference."""

    norm = path.replace("\\", "/")
    parts = norm.split("/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm) or ".." in parts or "" in parts:
        return None
    if parts[0] in {".tmp", ".git"}:
        return None
    return norm


def record_names_ticket(record_text: str, ticket_id: str) -> bool:
    """True when the record text names the ticket: exact id, a letter range ``T-IGX-NNN-X..Y`` or ``T-IGX-NNN-*``."""

    match = TICKET_NAME_RE.match(f"governance/tickets/{ticket_id}.md")
    if not match:
        return False
    num, letter = ticket_id.split("-")[2], ticket_id.split("-")[3]
    for m in _TICKET_ID_RE.finditer(record_text):
        if m.group(1) != num:
            continue
        if m.group(3):
            if m.group(2) <= letter <= m.group(3):
                return True
        elif m.group(2) == letter:
            return True
    return any(m.group(1) == num for m in _BATCH_WILDCARD_RE.finditer(record_text))


class Git:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def out(self, *args: str) -> str | None:
        done = self.run(*args)
        return done.stdout if done.returncode == 0 else None

    def show(self, rev: str, path: str) -> str | None:
        return self.out("show", f"{rev}:{path}")

    def exists(self, rev: str, path: str) -> bool:
        return self.run("cat-file", "-e", f"{rev}:{path}").returncode == 0

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return self.run("merge-base", "--is-ancestor", ancestor, descendant).returncode == 0

    def rev_parse(self, rev: str) -> str | None:
        out = self.out("rev-parse", "--verify", f"{rev}^{{commit}}")
        return out.strip() if out else None


def resolve_delegation(git: Git, block: ApprovalBlock, ticket_id: str, approval_commit: str) -> list[str]:
    """Verify the Delegation reference of a delegate-form approval against git (read-only)."""

    assert block.delegation_path is not None and block.delegation_sha is not None
    errors: list[str] = []
    path = normalize_record_path(block.delegation_path)
    if path is None:
        return [
            f"delegation path '{block.delegation_path}' is not a tracked repo-relative path "
            "(absolute, '..', '.tmp/' and '.git/' paths are refused)"
        ]
    sha = block.delegation_sha
    if git.run("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        return [f"delegation sha {sha} is not a commit in this repository"]
    if not git.exists(sha, path):
        return [f"delegation record '{path}' is not a tracked file at {sha[:12]}"]
    if sha == approval_commit or not git.is_ancestor(sha, approval_commit):
        errors.append(
            f"delegation sha {sha[:12]} is not a strict ancestor of the approval commit {approval_commit[:12]}"
        )
    record = git.show(sha, path) or ""
    if not record_names_ticket(record, ticket_id):
        errors.append(
            f"delegation record '{path}' at {sha[:12]} does not name {ticket_id} (id, range or batch wildcard)"
        )
    return errors


def load_exceptions(text: str | None) -> tuple[dict[str, dict], list[str]]:
    """Parse the legacy-exceptions JSON. Returns ({ticket path: entry}, errors)."""

    if text is None:
        return {}, []
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, [f"legacy exceptions file is not valid JSON: {exc}"]
    if not isinstance(doc, dict) or doc.get("schema") != EXCEPTIONS_SCHEMA or not isinstance(doc.get("entries"), list):
        return {}, [f"legacy exceptions file must have schema '{EXCEPTIONS_SCHEMA}' and an 'entries' list"]
    by_ticket: dict[str, dict] = {}
    errors: list[str] = []
    for i, entry in enumerate(doc["entries"]):
        if not isinstance(entry, dict):
            errors.append(f"exception entry {i} is not an object")
            continue
        ticket = entry.get("ticket")
        kind = entry.get("kind")
        if not isinstance(ticket, str) or not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            errors.append(f"exception entry {i}: 'ticket' and a non-empty 'reason' are required")
            continue
        if kind not in EXCEPTION_KINDS:
            errors.append(f"exception entry {ticket}: unknown kind {kind!r}")
            continue
        if kind == "unresolvable_delegation_reference" and not re.fullmatch(
            r"[0-9a-f]{64}", str(entry.get("approval_block_sha256", ""))
        ):
            errors.append(f"exception entry {ticket}: 'approval_block_sha256' (64 hex) is required for kind {kind}")
            continue
        if ticket in by_ticket:
            errors.append(f"exception entry {ticket} is listed twice")
            continue
        by_ticket[ticket] = entry
    return by_ticket, errors


def legacy_approval_digest(text: str) -> str:
    """sha256 of the exact legacy approval lines (every ``Approved:`` line plus ``Delegation:`` lines)."""

    lines = text.replace("\r\n", "\n").split("\n")
    keep = [ln for ln in lines if ln.startswith(("Approved:", "Delegation:"))]
    return hashlib.sha256("\n".join(keep).encode("utf-8")).hexdigest()


def check_ticket(
    git: Git,
    rel_path: str,
    text: str,
    approval_commit: str,
    exceptions: dict[str, dict],
    *,
    allow_exceptions: bool,
) -> TicketResult:
    """Strict check of one ticket text. ``approval_commit`` is the commit that adds (or the HEAD that holds) it."""

    result = TicketResult(rel_path)
    match = TICKET_NAME_RE.match(rel_path)
    if not match:
        result.errors.append("path is not governance/tickets/T-IGX-<NNN>-<L>.md")
        return result
    ticket_id = match.group(1)
    state = parse_state(text)
    if state is None:
        result.errors.append("ticket has no '## Lifecycle state' section")
        return result
    entry = exceptions.get(rel_path) if allow_exceptions else None
    lines = approval_lines(text)

    if not lines:
        if state in PRE_PLAN_STATES:
            result.notes.append(f"state {state}: no approval line yet (allowed)")
            return result
        if entry and entry["kind"] == "no_approval_line":
            result.exception_applied = f"no_approval_line: {entry['reason']}"
            result.notes.append(f"[EXCEPTION] legacy no_approval_line (state {state}): {entry['reason']}")
            return result
        result.errors.append(f"state '{state}' requires an approval line and none is present")
        return result

    # A ticket in a pre-plan state may already carry an approval line; if it does, the line must be valid.
    block, errors = parse_approval_block(text)
    if block is not None and block.kind == "delegate":
        impl, impl_err = implementer_session(text)
        if impl_err:
            errors.append(impl_err)
        elif impl == block.delegate_session:
            errors.append("delegate session equals the implementer session (R-3: the implementer must not sign)")
        errors.extend(resolve_delegation(git, block, ticket_id, approval_commit))
    if errors:
        if entry and entry["kind"] == "unresolvable_delegation_reference":
            if entry.get("approval_block_sha256") == legacy_approval_digest(text):
                result.exception_applied = f"unresolvable_delegation_reference: {entry['reason']}"
                result.notes.append(
                    f"[EXCEPTION] legacy unresolvable_delegation_reference: {entry['reason']} "
                    f"(would fail: {'; '.join(errors)})"
                )
                return result
            result.notes.append("exception entry exists but the approval lines changed since it was written: void")
        result.errors.extend(errors)
        return result
    result.notes.append(f"{block.kind} approval OK" if block else "ok")
    return result
