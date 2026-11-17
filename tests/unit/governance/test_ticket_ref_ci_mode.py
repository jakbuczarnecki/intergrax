# © Artur Czarnecki. All rights reserved.

"""Commit-Ref gate, ``--ci`` mode (EQO-A PT-E02): skip env ignored, git failures and empty ranges are errors."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from scripts.governance import verify_commit_ticket_refs as refs

REF_OK = "Ref: governance/tickets/T-IGX-900-A.md"


def _git(root: Path, *args: str) -> str:
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
            f"core.hooksPath={root.parent / 'nohooks'}",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _commit(root: Path, files: dict[str, str], message: str) -> None:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "nohooks").mkdir()
    _git(root, "init", "-q", "-b", "main")
    _commit(root, {"README.md": "base\n", "governance/tickets/T-IGX-900-A.md": "# t\n"}, "base")
    _git(root, "checkout", "-q", "-b", "feature")
    monkeypatch.setattr(refs, "_REPO", root)
    monkeypatch.delenv("INTERGRAX_SKIP_TICKET_REF_GATE", raising=False)
    return root


def _args(*extra: str) -> list[str]:
    return ["--mode", "range", "--rev-range", "main..HEAD", *extra]


def test_product_commit_without_ref_fails_in_ci_mode(repo: Path) -> None:
    _commit(repo, {"scripts/x.py": "x = 1\n"}, "product commit without a trailer")
    assert refs.main(_args("--ci")) == 1


def test_product_commit_with_ref_to_a_tracked_ticket_passes_in_ci_mode(repo: Path) -> None:
    _commit(repo, {"scripts/x.py": "x = 1\n"}, f"product commit\n\n{REF_OK}")
    assert refs.main(_args("--ci")) == 0


def test_ref_to_a_ticket_that_is_not_tracked_fails_in_ci_mode_only(repo: Path) -> None:
    _commit(repo, {"scripts/x.py": "x = 1\n"}, "product commit\n\nRef: governance/tickets/T-IGX-999-Z.md")
    assert refs.main(_args("--ci")) == 1
    assert refs.main(_args()) == 0  # default mode is unchanged: only the shape of the line is checked


def test_skip_env_is_ignored_in_ci_mode(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _commit(repo, {"scripts/x.py": "x = 1\n"}, "product commit without a trailer")
    monkeypatch.setenv("INTERGRAX_SKIP_TICKET_REF_GATE", "1")
    assert refs.main(_args("--ci")) == 1
    assert "IGNORED in --ci mode" in capsys.readouterr().err


def test_skip_env_still_skips_in_default_mode(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _commit(repo, {"scripts/x.py": "x = 1\n"}, "product commit without a trailer")
    monkeypatch.setenv("INTERGRAX_SKIP_TICKET_REF_GATE", "1")
    assert refs.main(_args()) == 0


def test_unreadable_range_is_an_error_in_ci_mode_and_silent_otherwise(repo: Path) -> None:
    argv = ["--mode", "range", "--rev-range", "no-such-base..HEAD"]
    assert refs.main([*argv, "--ci"]) == 1
    assert refs.main(argv) == 0  # documents the unchanged legacy behaviour that CI mode closes


def test_empty_range_argument_is_an_error_in_ci_mode(repo: Path) -> None:
    assert refs.main(["--mode", "range", "--rev-range", "", "--ci"]) == 1
    assert refs.main(["--mode", "range", "--rev-range", ""]) == 0


def test_non_product_commit_needs_no_ref(repo: Path) -> None:
    _commit(repo, {"docs/a.md": "doc\n"}, "docs only")
    assert refs.main(_args("--ci")) == 0
