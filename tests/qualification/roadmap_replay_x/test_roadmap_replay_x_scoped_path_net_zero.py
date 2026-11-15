# © Artur Czarnecki. All rights reserved.

"""PDC-X F-56 — a scoped path absent from the prod diff is accepted only when it is net-zero.

A scoped PRE-ENT path that is not in ``git diff <pin>..HEAD`` is a violation unless the file
exists neither at HEAD nor at the DUP-X pin (added after the pin and deleted again). Every
other case stays reported. The tests run against throwaway git repositories, so no real
history is needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import tests.qualification.roadmap_replay_x._prod_delta_disposition as disp
from tests.qualification.roadmap_replay_x._prod_delta_disposition import (
    ProdDeltaDisposition,
    scoped_prod_delta_disposition_violations,
)

_PATH = "intergrax/integrations/providers/embedding_provider/ollama/manifest.py"
_GOOD_HEAD = disp.REPLAY_X_L2_QUALIFICATION_HEAD


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "diff.renames=true",
            *args,
        ],
        cwd=repo,
        text=True,
    ).strip()


def _write(repo: Path, rel: str, text: str = "x = 1\n") -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _init_repo(tmp_path: Path, *, pinned_files: tuple[str, ...] = ()) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, "intergrax/keep.py")
    for rel in pinned_files:
        _write(repo, rel, "pinned = 1\n" * 5)
    return repo, _commit(repo, "pin")


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    pin: str,
    *,
    head_sha: str = _GOOD_HEAD,
    disposition: str = "scoped delta",
) -> None:
    rows = (
        ProdDeltaDisposition(
            path=_PATH,
            disposition=disposition,
            regression_evidence="tests/qualification/roadmap_replay_x/__init__.py",
            qualification_head_sha=head_sha,
        ),
    )
    monkeypatch.setattr(disp, "_DUP_X_PIN", pin)
    monkeypatch.setattr(disp, "REPLAY_X_L2_SCOPED_PROD_DELTA", rows)
    monkeypatch.setattr(disp, "REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS", frozenset(r.path for r in rows))
    monkeypatch.setattr(disp, "scoped_regression_evidence_violations", lambda *_args: ())


def test_f56_net_zero_deleted_path_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, pin = _init_repo(tmp_path)
    _write(repo, _PATH)
    _commit(repo, "add after pin")
    (repo / _PATH).unlink()
    _commit(repo, "delete again")
    _configure(monkeypatch, pin)
    assert disp._prod_diff_paths(repo, "intergrax", "agents", "applications") == frozenset()
    assert scoped_prod_delta_disposition_violations(repo) == ()


def test_f56_path_existing_at_head_but_not_in_diff_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, pin = _init_repo(tmp_path, pinned_files=(_PATH,))
    _write(repo, "docs/unrelated.md")
    _commit(repo, "unrelated change")
    _configure(monkeypatch, pin)
    assert scoped_prod_delta_disposition_violations(repo) == (f"scoped_path_not_in_prod_diff:{_PATH}",)


def test_f56_path_added_after_pin_and_still_present_is_in_diff_not_net_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, pin = _init_repo(tmp_path)
    _write(repo, _PATH)
    _commit(repo, "add after pin")
    _configure(monkeypatch, pin)
    assert disp._is_net_zero_absent(repo, _PATH) is False
    assert scoped_prod_delta_disposition_violations(repo) == ()


def test_f56_path_deleted_after_existing_at_pin_is_not_net_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, pin = _init_repo(tmp_path, pinned_files=(_PATH,))
    (repo / _PATH).unlink()
    _commit(repo, "delete pinned file")
    _configure(monkeypatch, pin)
    assert disp._is_net_zero_absent(repo, _PATH) is False


def test_f56_path_renamed_away_after_existing_at_pin_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `git diff --name-only` lists only the rename target, so the old scoped path is "not in diff".
    repo, pin = _init_repo(tmp_path, pinned_files=(_PATH,))
    _git(repo, "mv", _PATH, "intergrax/integrations/providers/embedding_provider/ollama/moved.py")
    _commit(repo, "rename")
    _configure(monkeypatch, pin)
    assert disp._prod_diff_paths(repo, "intergrax", "agents", "applications") == frozenset(
        {"intergrax/integrations/providers/embedding_provider/ollama/moved.py"}
    )
    assert scoped_prod_delta_disposition_violations(repo) == (f"scoped_path_not_in_prod_diff:{_PATH}",)


def test_f56_wrong_head_is_still_reported_for_path_in_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, pin = _init_repo(tmp_path)
    _write(repo, _PATH)
    _commit(repo, "add after pin")
    _configure(monkeypatch, pin, head_sha="0" * 40)
    assert scoped_prod_delta_disposition_violations(repo) == (f"wrong_qualification_head:{_PATH}",)


def test_f56_missing_disposition_is_still_reported_for_path_in_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, pin = _init_repo(tmp_path)
    _write(repo, _PATH)
    _commit(repo, "add after pin")
    _configure(monkeypatch, pin, disposition="  ")
    assert scoped_prod_delta_disposition_violations(repo) == (f"missing_disposition:{_PATH}",)


def test_f56_unknown_pin_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _init_repo(tmp_path)
    _configure(monkeypatch, "f" * 40)
    with pytest.raises(subprocess.CalledProcessError):
        scoped_prod_delta_disposition_violations(repo)
