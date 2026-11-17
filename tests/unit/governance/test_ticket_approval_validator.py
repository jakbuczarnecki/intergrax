# © Artur Czarnecki. All rights reserved.

"""Strict ticket-approval verifier (EQO-A PT-E02, finding EQOA-A6-002): grammar, delegation resolver, legacy exceptions.

Every test builds a throw-away git repository under ``tmp_path``; nothing reads or writes the real repository.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from scripts.governance import verify_ticket_approvals as vta
from scripts.governance.ticket_approval import (
    EXCEPTIONS_REL,
    EXCEPTIONS_SCHEMA,
    Git,
    legacy_approval_digest,
    load_exceptions,
    normalize_record_path,
    parse_approval_block,
    record_names_ticket,
)

TS = "2026-11-16T10:00:00+01:00"
IMPL = "local_11111111-1111-4111-8111-111111111111"
DELEG = "local_22222222-2222-4222-8222-222222222222"
TICKET = "governance/tickets/T-IGX-900-A.md"
RECORD = "docs/records/DELEGATION-TEST.md"

TEMPLATE = """# T-IGX-900-A - test ticket

## Scope

Scope text.

## Approved

{approval}

## Implementer

{implementer}

## Lifecycle state

{state}
"""


def _ticket(
    approval: str, *, state: str = "plan_approved", implementer: str | None = f"Implementer session: {IMPL}"
) -> str:
    return TEMPLATE.format(approval=approval, implementer=implementer or "none recorded", state=state)


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root.parent / "nohooks").mkdir(exist_ok=True)
        self._run("init", "-q", "-b", "main")

    def _run(self, *args: str) -> str:
        done = subprocess.run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.autocrlf=false",
                "-c",
                f"core.hooksPath={self.root.parent / 'nohooks'}",
                *args,
            ],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )
        return done.stdout.strip()

    def commit(self, files: dict[str, str | None], message: str = "c") -> str:
        for rel, content in files.items():
            target = self.root / rel
            if content is None:
                target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
        self._run("add", "-A")
        self._run("commit", "-q", "-m", message)
        return self._run("rev-parse", "HEAD")

    def branch(self, name: str) -> None:
        self._run("checkout", "-q", "-b", name)

    def checkout(self, name: str) -> None:
        self._run("checkout", "-q", name)

    @property
    def git(self) -> Git:
        return Git(self.root)


def _scenario(
    tmp_path: Path,
    approval_fn,
    *,
    state: str = "plan_approved",
    implementer: str | None = f"Implementer session: {IMPL}",
    record_text: str = "Delegation record. Scope: T-IGX-900-A.\n",
) -> tuple[Repo, str, str]:
    """Return (repo, record_sha, approval_commit). ``approval_fn(record_sha) -> approval block text``."""

    repo = Repo(tmp_path / "repo")
    record_sha = repo.commit({RECORD: record_text, "README.md": "base\n"}, "base")
    repo.branch("feature")
    files: dict[str, str | None] = {TICKET: _ticket(approval_fn(record_sha), state=state, implementer=implementer)}
    approval_commit = repo.commit(files, "add ticket")
    return repo, record_sha, approval_commit


def _delegate_block(sha: str, *, path: str = RECORD, session: str = DELEG, ts: str = TS) -> str:
    return f"Approved: operator-delegate @ {ts}\nDelegation: {path}@{sha}\nDelegate session: {session}"


def _failed(results, needle: str) -> bool:
    return any((not r.ok) and any(needle in e for e in r.errors) for r in results)


def _ok(results) -> bool:
    return all(r.ok for r in results)


# --- grammar (unit) -------------------------------------------------------------------------------------------------


def test_personal_form_accepted() -> None:
    block, errors = parse_approval_block(_ticket(f"Approved: operator @ {TS}"))
    assert errors == [] and block is not None and block.kind == "personal"


@pytest.mark.parametrize(
    "line",
    [
        "Approved: yes, go ahead",
        "Approved: operator",
        "Approved: operator @ 2026-11-16",
        "Approved: operator @ 2026-11-16T10:00:00",
        "Approved: operator @ 2026-13-45T10:00:00+01:00",
        f"Approved: operator @ {TS} (chat acceptance)",
        f"Approved: someone-else @ {TS}",
        f"Approved: operator-delegate @ {TS} | Delegation: {RECORD}@{'a' * 40}",
    ],
)
def test_free_text_and_malformed_approval_lines_are_refused(line: str) -> None:
    block, errors = parse_approval_block(_ticket(line))
    assert block is None and errors


def test_two_approval_lines_are_refused() -> None:
    block, errors = parse_approval_block(_ticket(f"Approved: operator @ {TS}\nApproved: operator @ {TS}"))
    assert block is None and any("exactly one" in e for e in errors)


def test_personal_form_must_not_carry_delegation_lines() -> None:
    block, errors = parse_approval_block(_ticket(f"Approved: operator @ {TS}\nDelegation: {RECORD}@{'a' * 40}"))
    assert block is None and errors


def test_delegate_form_needs_delegation_and_session_lines() -> None:
    _, no_deleg = parse_approval_block(_ticket(f"Approved: operator-delegate @ {TS}"))
    assert any("Delegation:" in e for e in no_deleg)
    _, no_session = parse_approval_block(
        _ticket(f"Approved: operator-delegate @ {TS}\nDelegation: {RECORD}@{'a' * 40}")
    )
    assert any("Delegate session" in e for e in no_session)
    _, bad_session = parse_approval_block(
        _ticket(
            f"Approved: operator-delegate @ {TS}\nDelegation: {RECORD}@{'a' * 40}\nDelegate session: operator-delegate"
        )
    )
    assert any("Delegate session" in e for e in bad_session)


@pytest.mark.parametrize(
    "path",
    [".tmp/session/x/DELEG.md", "/abs/DELEG.md", "C:/abs/DELEG.md", "docs/../x.md", ".git/config", "docs//x.md"],
)
def test_untracked_or_escaping_record_paths_are_refused(path: str) -> None:
    assert normalize_record_path(path) is None


def test_record_scope_matching() -> None:
    assert record_names_ticket("see T-IGX-900-A and T-IGX-900-B", "T-IGX-900-A")
    assert record_names_ticket("T-IGX-900-A..F", "T-IGX-900-C")
    assert record_names_ticket("T-IGX-900-A*", "T-IGX-900-A")
    assert record_names_ticket("all of T-IGX-900-*", "T-IGX-900-Q")
    assert not record_names_ticket("T-IGX-900-A..C", "T-IGX-900-D")
    assert not record_names_ticket("T-IGX-901-A", "T-IGX-900-A")
    assert not record_names_ticket("no ticket here", "T-IGX-900-A")


# --- delegation resolver (git) --------------------------------------------------------------------------------------


def test_valid_delegate_approval_passes(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, _delegate_block)
    results = vta.run_since(repo.git, "main")
    assert _ok(results), [r.errors for r in results]


def test_valid_personal_approval_passes(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, lambda _s: f"Approved: operator @ {TS}")
    assert _ok(vta.run_since(repo.git, "main"))


def test_delegate_without_delegation_reference_fails(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, lambda _s: f"Approved: operator-delegate @ {TS}\nDelegate session: {DELEG}")
    assert _failed(vta.run_since(repo.git, "main"), "Delegation:")


def test_delegation_record_only_in_tmp_fails(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, lambda s: _delegate_block(s, path=".tmp/session/x/DELEG.md"))
    assert _failed(vta.run_since(repo.git, "main"), "not a tracked repo-relative path")


def test_delegation_record_missing_at_sha_fails(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, lambda s: _delegate_block(s, path="docs/records/OTHER.md"))
    assert _failed(vta.run_since(repo.git, "main"), "not a tracked file")


def test_delegation_sha_unknown_fails(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, lambda _s: _delegate_block("f" * 40))
    assert _failed(vta.run_since(repo.git, "main"), "not a commit")


def test_delegation_sha_not_an_ancestor_fails(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "repo")
    repo.commit({"README.md": "base\n"}, "base")
    repo.branch("side")
    side_sha = repo.commit({RECORD: "Scope T-IGX-900-A\n"}, "record on a side branch")
    repo.checkout("main")
    repo.branch("feature")
    repo.commit({TICKET: _ticket(_delegate_block(side_sha))}, "add ticket")
    assert _failed(vta.run_since(repo.git, "main"), "not a strict ancestor")


def test_record_not_naming_the_ticket_fails(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, _delegate_block, record_text="Delegation for T-IGX-901-A only.\n")
    assert _failed(vta.run_since(repo.git, "main"), "does not name T-IGX-900-A")


def test_record_range_scope_passes(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, _delegate_block, record_text="Tickets T-IGX-900-A..F are delegated.\n")
    assert _ok(vta.run_since(repo.git, "main"))


def test_delegate_equal_to_implementer_fails(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, lambda s: _delegate_block(s, session=IMPL))
    assert _failed(vta.run_since(repo.git, "main"), "equals the implementer session")


def test_missing_or_malformed_implementer_session_fails(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path / "a", _delegate_block, implementer=None)
    assert _failed(vta.run_since(repo.git, "main"), "Implementer session")
    repo2, _, _ = _scenario(tmp_path / "b", _delegate_block, implementer="Implementer session: some-agent")
    assert _failed(vta.run_since(repo2.git, "main"), "not a host session id")


# --- lifecycle and ranges -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["plan_approved", "in_progress", "quality_gate", "ready_for_audit", "closed"])
def test_post_plan_state_without_approval_line_fails(tmp_path: Path, state: str) -> None:
    repo, _, _ = _scenario(tmp_path, lambda _s: "PENDING. No approval has been given.", state=state)
    assert _failed(vta.run_since(repo.git, "main"), "requires an approval line")


def test_plan_pending_with_placeholder_passes(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, lambda _s: "PENDING. No approval has been given.", state="plan_pending")
    assert _ok(vta.run_since(repo.git, "main"))


def test_missing_lifecycle_state_fails(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "repo")
    repo.commit({"README.md": "base\n"}, "base")
    repo.branch("feature")
    repo.commit({TICKET: "# T-IGX-900-A\n\nApproved: operator @ " + TS + "\n"}, "add")
    assert _failed(vta.run_since(repo.git, "main"), "Lifecycle state")


def test_unreadable_base_raises(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, _delegate_block)
    with pytest.raises(RuntimeError):
        vta.run_since(repo.git, "no-such-base")


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good, _, _ = _scenario(tmp_path / "good", _delegate_block)
    assert vta.main(["--repo", str(good.root), "--since", "main"]) == 0
    bad, _, _ = _scenario(tmp_path / "bad", lambda s: _delegate_block(s, session=IMPL))
    assert vta.main(["--repo", str(bad.root), "--since", "main"]) == 1
    assert vta.main(["--repo", str(bad.root), "--since", "does-not-exist"]) == 1
    capsys.readouterr()


# --- legacy exceptions: never silent, never self-service ------------------------------------------------------------


def _exceptions(entries: list[dict]) -> str:
    return json.dumps({"schema": EXCEPTIONS_SCHEMA, "entries": entries})


def _legacy_repo(tmp_path: Path, *, exceptions_in_base: bool, ticket_exists_in_base: bool = True) -> Repo:
    repo = Repo(tmp_path / "repo")
    base_files: dict[str, str | None] = {"README.md": "base\n"}
    if ticket_exists_in_base:
        base_files[TICKET] = _ticket("PENDING.", state="in_progress")  # legacy: past plan state, no approval line
    if exceptions_in_base:
        base_files[EXCEPTIONS_REL] = _exceptions(
            [{"ticket": TICKET, "kind": "no_approval_line", "reason": "legacy test entry"}]
        )
    repo.commit(base_files, "base")
    repo.branch("feature")
    return repo


def test_legacy_ticket_without_line_is_accepted_only_through_the_exceptions_file_and_reported(tmp_path: Path) -> None:
    repo = _legacy_repo(tmp_path, exceptions_in_base=True)
    repo.commit({TICKET: _ticket("PENDING.", state="quality_gate")}, "legacy ticket advances")
    results = vta.run_since(repo.git, "main")
    assert _ok(results)
    assert [r.exception_applied for r in results] == ["no_approval_line: legacy test entry"]
    assert vta.report(results) == 0


def test_same_legacy_ticket_without_exception_fails(tmp_path: Path) -> None:
    repo = _legacy_repo(tmp_path, exceptions_in_base=False)
    repo.commit({TICKET: _ticket("PENDING.", state="quality_gate")}, "legacy ticket advances")
    assert _failed(vta.run_since(repo.git, "main"), "requires an approval line")


def test_pull_request_cannot_excuse_its_own_ticket_by_editing_the_exceptions_file(tmp_path: Path) -> None:
    repo = _legacy_repo(tmp_path, exceptions_in_base=False, ticket_exists_in_base=False)
    repo.commit(
        {
            TICKET: _ticket("PENDING.", state="in_progress"),
            EXCEPTIONS_REL: _exceptions([{"ticket": TICKET, "kind": "no_approval_line", "reason": "self-service"}]),
        },
        "new ticket plus a self-written exception",
    )
    assert _failed(vta.run_since(repo.git, "main"), "requires an approval line")


def test_new_ticket_cannot_use_an_exception_that_already_exists_at_the_merge_base(tmp_path: Path) -> None:
    repo = _legacy_repo(tmp_path, exceptions_in_base=True, ticket_exists_in_base=False)
    repo.commit({TICKET: _ticket("PENDING.", state="in_progress")}, "new ticket named in an old exception")
    assert _failed(vta.run_since(repo.git, "main"), "requires an approval line")


def test_an_exception_does_not_excuse_an_approval_line_that_is_invalid(tmp_path: Path) -> None:
    repo = _legacy_repo(tmp_path, exceptions_in_base=True)
    repo.commit({TICKET: _ticket("Approved: fine by me", state="in_progress")}, "free-text approval on a legacy ticket")
    assert _failed(vta.run_since(repo.git, "main"), "does not match")


def test_unresolvable_delegation_exception_is_bound_to_the_exact_approval_lines(tmp_path: Path) -> None:
    legacy_line = f"Approved: operator-delegate @ {TS} | Delegation: .tmp/x/DELEG.md"
    old = _ticket(legacy_line, state="closed")
    repo = Repo(tmp_path / "repo")
    entry = {
        "ticket": TICKET,
        "kind": "unresolvable_delegation_reference",
        "reason": "legacy",
        "approval_block_sha256": legacy_approval_digest(old),
    }
    repo.commit({"README.md": "x\n", TICKET: old, EXCEPTIONS_REL: _exceptions([entry])}, "base")
    repo.branch("feature")
    # unchanged lines, state changes: excepted
    repo.commit(
        {TICKET: _ticket(legacy_line, state="closed").replace("Scope text.", "Scope text edited.")}, "edit text"
    )
    res = vta.run_since(repo.git, "main")
    assert _ok(res)
    # the approval line itself changes: the exception is void and the new line is judged on its own
    repo.commit({TICKET: _ticket(legacy_line + " changed", state="closed")}, "edit approval")
    assert not _ok(vta.run_since(repo.git, "main"))


def test_exceptions_file_schema_errors_fail_closed() -> None:
    assert load_exceptions("{not json")[1]
    assert load_exceptions(json.dumps({"schema": "x", "entries": []}))[1]
    assert load_exceptions(_exceptions([{"ticket": TICKET, "kind": "bogus", "reason": "r"}]))[1]
    assert load_exceptions(_exceptions([{"ticket": TICKET, "kind": "no_approval_line", "reason": " "}]))[1]
    dup = {"ticket": TICKET, "kind": "no_approval_line", "reason": "r"}
    assert load_exceptions(_exceptions([dup, dup]))[1]
    assert load_exceptions(
        _exceptions([{"ticket": TICKET, "kind": "unresolvable_delegation_reference", "reason": "r"}])
    )[1]
    assert load_exceptions(None) == ({}, [])


def test_broken_exceptions_file_in_base_is_reported(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "repo")
    repo.commit({"README.md": "x\n", EXCEPTIONS_REL: "{broken"}, "base")
    repo.branch("feature")
    repo.commit({TICKET: _ticket("PENDING.", state="plan_pending")}, "add")
    assert _failed(vta.run_since(repo.git, "main"), "not valid JSON")


def test_unchanged_approval_and_state_is_not_revalidated(tmp_path: Path) -> None:
    repo = _legacy_repo(tmp_path, exceptions_in_base=False)
    repo.commit(
        {TICKET: _ticket("PENDING.", state="in_progress").replace("Scope text.", "Scope text edited.")}, "text only"
    )
    results = vta.run_since(repo.git, "main")
    assert _ok(results) and any("unchanged" in n for r in results for n in r.notes)


def test_replay_all_reports_stale_exceptions(tmp_path: Path) -> None:
    repo = Repo(tmp_path / "repo")
    entries = [
        {"ticket": "governance/tickets/T-IGX-900-B.md", "kind": "no_approval_line", "reason": "gone"},
        {"ticket": TICKET, "kind": "no_approval_line", "reason": "now approved"},
    ]
    repo.commit({TICKET: _ticket(f"Approved: operator @ {TS}"), EXCEPTIONS_REL: _exceptions(entries)}, "base")
    results = vta.run_all(repo.git)
    stale = [r for r in results if not r.ok and "stale exception" in " ".join(r.errors)]
    assert len(stale) == 2
    assert vta.report(results) == 1


def test_replay_all_counts_exceptions_and_never_hides_them(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = Repo(tmp_path / "repo")
    entry = {"ticket": TICKET, "kind": "no_approval_line", "reason": "legacy"}
    repo.commit({TICKET: _ticket("PENDING.", state="closed"), EXCEPTIONS_REL: _exceptions([entry])}, "base")
    assert vta.report(vta.run_all(repo.git)) == 0
    out = capsys.readouterr().out
    assert "[EXCEPTION]" in out and "1 accepted ONLY through the legacy exceptions file" in out


def test_single_ticket_mode(tmp_path: Path) -> None:
    repo, _, _ = _scenario(tmp_path, _delegate_block)
    assert _ok(vta.run_ticket(repo.git, repo.root, Path(TICKET)))
    assert not _ok(vta.run_ticket(repo.git, repo.root, Path("governance/tickets/T-IGX-900-Z.md")))
