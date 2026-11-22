"""T-IGX-058-A: Cursor shell guard covers git/gh global-option and destructive variants."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GUARD = ROOT / "scripts/hooks/igx_cursor_integration_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("igx_cursor_integration_guard_test", GUARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load()


def _permission(cmd: str) -> str:
    return guard.decide({"command": cmd})["permission"]


@pytest.mark.parametrize(
    "cmd",
    [
        # push and push equivalents, literal and with global options
        "git push origin HEAD",
        "git -C D:/Projekty/_wt/x push origin HEAD",
        'git -C "D:/Projekty/my repo" push',
        "git -C 'D:/x y' push",
        "git --git-dir=D:/x/.git push",
        "git --git-dir D:/x/.git --work-tree D:/x push",
        "git.exe -c core.askPass=x push origin b",
        "GIT.EXE -c http.extraHeader=x push",
        "git.cmd push",
        "git --no-pager push --force-with-lease",
        "git -c a=b -c c=d --no-pager -C x push",
        '"C:\\Program Files\\Git\\bin\\git.exe" push origin b',
        "& 'C:\\Program Files\\Git\\cmd\\git.exe' -C x push",
        'git "push" origin b',
        "git status && git -C x push",
        "git status; git push",
        "git -C x send-pack origin refs/heads/b",
        "git http-push https://example.invalid/r b",
        "git subtree push --prefix=docs origin b",
        "git -C x subtree push -P docs origin b",
        "git -C x `\npush origin b",
        # reset --hard
        "git reset --hard origin/development",
        "git -C x reset --hard",
        "git reset -q --hard HEAD~1",
        "git --git-dir=x reset HEAD --hard",
        # force branch deletion
        "git branch -D feat/x",
        "git -C x branch -D feat/x",
        "git branch -rD origin/x",
        "git branch --delete --force feat/x",
        "git branch -d -f feat/x",
        "git branch -df feat/x",
        "git branch --force --delete feat/x",
        "git branch -d --force feat/x",
        # forced worktree removal
        "git worktree remove --force D:/_wt/x",
        "git -C x worktree remove -f D:/_wt/x",
        "git worktree remove D:/_wt/x --force",
        "git worktree remove -ff D:/_wt/x",
        # ref deletion
        "git update-ref -d refs/heads/feat/x",
        "git -C x update-ref --delete refs/heads/feat/x",
        # gh pr create / merge
        "gh pr create --base main",
        "gh.exe pr create --base main",
        "gh pr new --fill",
        "gh pr merge 17 --admin",
        "gh -R owner/repo pr merge 17",
        "gh pr --repo owner/repo merge 17 --squash",
        # gh api mutating
        "gh api repos/a/b/pulls -X POST",
        "gh api -XPOST repos/a/b/pulls",
        "gh api --method=PATCH repos/a/b/pulls/1",
        "gh api --method DELETE repos/a/b/git/refs/heads/x",
        "gh api repos/a/b/pulls -f title=x -f head=b",
        "gh api repos/a/b/issues/1/comments -F body=@x.md",
        "gh api repos/a/b/pulls --raw-field title=x",
        "gh api repos/a/b/pulls --input body.json",
        "gh api graphql -f query='mutation { mergePullRequest(input: {}) { clientMutationId } }'",
        "gh api graphql -F query=@q.graphql",
        "gh api graphql --input q.json",
    ],
)
def test_guard_denies_variants(cmd: str) -> None:
    assert _permission(cmd) == "deny", cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "git status --short",
        "git -C D:/_wt/x status",
        "git diff --check",
        "git log --oneline -5",
        "git -C x log --grep=push",
        "git show HEAD",
        "git rev-parse HEAD",
        "git add -A",
        'git commit -m "Tighten guard"',
        "git -C x commit -m wip",
        "git merge --no-ff feat/x",
        'git -C D:/_wt/int merge --no-ff -m "merge S1" igx/s1',
        "git worktree add -b feat/x D:/_wt/x HEAD",
        "git -C x worktree add --detach D:/_wt/y",
        "git worktree list",
        "git worktree remove D:/_wt/x",
        "git branch feat/x",
        "git branch -d feat/x",
        "git branch --delete feat/x",
        "git branch -f feat/x HEAD",
        "git branch --list --format=%(refname)",
        "git reset --soft HEAD~1",
        "git reset HEAD file.py",
        "git update-ref refs/heads/feat/x HEAD",
        "git fetch origin development",
        "gh pr view 17",
        "gh pr list --state open",
        "gh -R owner/repo pr checks 17",
        "gh api repos/a/b/pulls/17",
        "gh api repos/a/b/pulls -X GET -f state=open",
        "gh api --method GET repos/a/b/pulls -f state=open",
        "gh api graphql -f query='query { viewer { login } }'",
        "python scripts/cursor/igx_integrator.py merge --packet .tmp/proof/packet.json",
        "python scripts/cursor/igx_factory.py integrate --step S1 --run-tests",
        "uv run pytest tests/unit/governance -q",
    ],
)
def test_guard_allows_local_and_read_commands(cmd: str) -> None:
    assert _permission(cmd) == "allow", cmd


@pytest.mark.parametrize("value", [None, [], {}, {"command": 7}])
def test_guard_fails_closed_on_invalid_payload(value: object) -> None:
    assert guard.decide(value)["permission"] == "deny"


def test_guard_long_flag_list_does_not_backtrack() -> None:
    cmd = "git " + " ".join(f"-c k{i}=v{i} --opt{i}" for i in range(400)) + " status"
    started = time.perf_counter()
    assert _permission(cmd) == "allow"
    assert time.perf_counter() - started < 2.0


def test_guard_cli_contract_stdin_to_json() -> None:
    def run(payload: str) -> dict[str, str]:
        done = subprocess.run(
            [sys.executable, str(GUARD)], input=payload, capture_output=True, text=True, check=True, timeout=30
        )
        return json.loads(done.stdout)

    assert run(json.dumps({"command": "git -C x push"}))["permission"] == "deny"
    assert run(json.dumps({"command": "git -C x status"}))["permission"] == "allow"
    assert run("not json")["permission"] == "deny"
