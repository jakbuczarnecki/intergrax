from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MOD = ROOT / "scripts/hooks/cursor_development_push.py"
_GIT_IDENT = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _load():
    spec = importlib.util.spec_from_file_location("cursor_development_push_test", MOD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _bootstrap_ff_development(tmp_path: Path) -> tuple[Path, str, str]:
    """Bare remote + repo on development with one extra local commit (fast-forward push)."""
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "development"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", *_GIT_IDENT, "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            *_GIT_IDENT,
            "commit",
            "-q",
            "-m",
            "base\n\nRef: governance/tickets/T-IGX-058-D.md",
        ],
        cwd=repo,
        check=True,
    )
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True)
    subprocess.run(["git", "push", "-q", "origin", "development"], cwd=repo, check=True)
    subprocess.run(["git", "fetch", "origin", "development"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("ahead\n", encoding="utf-8")
    subprocess.run(["git", *_GIT_IDENT, "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            *_GIT_IDENT,
            "commit",
            "-q",
            "-m",
            "ahead\n\nRef: governance/tickets/T-IGX-058-D.md",
        ],
        cwd=repo,
        check=True,
    )
    head_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, base_sha, head_sha


def _snapshot_tree(repo: Path) -> dict[str, str | bytes]:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    tracked = (repo / "tracked.txt").read_bytes() if (repo / "tracked.txt").is_file() else b""
    untracked = (repo / "untracked.txt").read_bytes() if (repo / "untracked.txt").is_file() else b""
    index = _git(repo, "write-tree").stdout.strip()
    return {"head": head, "tracked": tracked, "untracked": untracked, "index_tree": index}


def test_push_refspec_matrix() -> None:
    m = _load()
    assert m.push_target_is_allowed_development("origin development")
    assert not m.push_target_is_allowed_development("origin main")
    assert not m.push_target_is_allowed_development("upstream development")
    assert not m.push_target_is_allowed_development("origin development --force")


@pytest.mark.parametrize(
    "tail",
    [
        "origin development --no-verify",
        "--no-verify origin development",
        "-f origin development",
        "origin development refs/heads/development",
        "origin development --follow-tags",
        "origin development --recurse-submodules=on-demand",
        "Origin development",
        "origin DEVELOPMENT",
    ],
)
def test_rejects_unapproved_options_and_refspecs(tail: str) -> None:
    assert not _load().push_target_is_allowed_development(tail)


def test_preflight_uses_actual_checkout_and_project_uv(monkeypatch, tmp_path):
    m = _load()
    monkeypatch.delenv("IGX_HOOKS", raising=False)
    monkeypatch.delenv("IGX_CURSOR_PUSH_PRECHECK_SKIP", raising=False)
    root = tmp_path / "repo"
    subdir = root / "nested"
    subdir.mkdir(parents=True)
    base, head = "a" * 40, "b" * 40
    calls = []

    def git(cwd, *args):
        calls.append((cwd, args))
        outputs = {
            ("rev-parse", "--show-toplevel"): str(root),
            ("symbolic-ref", "HEAD"): "refs/heads/development",
            ("rev-parse", "origin/development"): base,
            ("rev-parse", "HEAD"): head,
            ("merge-base", "--is-ancestor", base, head): "",
        }
        return subprocess.CompletedProcess(args, 0, outputs[args], "")

    gates = []

    def run_gate(cwd, uv, script, *args):
        gates.append((cwd, uv, script, args))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(m, "_run_git", git)
    monkeypatch.setattr(m, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(m, "_run_gate", run_gate)
    assert m.validate_development_push(subdir) is None
    assert calls[0] == (subdir, ("rev-parse", "--show-toplevel"))
    assert all(cwd == root for cwd, _ in calls[1:])
    assert ("status", "--porcelain") not in {a for _, a in calls}
    assert gates == [
        (
            root,
            "uv",
            "scripts/governance/verify_commit_ticket_refs.py",
            ("--ci", "--mode", "range", "--rev-range", f"{base}..{head}"),
        ),
        (root, "uv", "scripts/governance/validate_policy_registry.py", ()),
    ]


@pytest.mark.parametrize("gate", ["Ticket Ref gate", "Policy registry validation"])
def test_gate_failure_preserves_real_diagnostics(monkeypatch, tmp_path, gate):
    m = _load()
    monkeypatch.delenv("IGX_HOOKS", raising=False)
    monkeypatch.delenv("IGX_CURSOR_PUSH_PRECHECK_SKIP", raising=False)
    root = tmp_path
    base, head = "a" * 40, "b" * 40

    def git(_, *args):
        outputs = {
            ("rev-parse", "--show-toplevel"): str(root),
            ("symbolic-ref", "HEAD"): "refs/heads/development",
            ("rev-parse", "origin/development"): base,
            ("rev-parse", "HEAD"): head,
            ("merge-base", "--is-ancestor", base, head): "",
        }
        return subprocess.CompletedProcess(args, 0, outputs[args], "")

    def run_gate(_, _uv, script, *args):
        fail = ("verify_commit" in script) == (gate == "Ticket Ref gate")
        return subprocess.CompletedProcess(args, int(fail), "specific detail" if fail else "", "")

    monkeypatch.setattr(m, "_run_git", git)
    monkeypatch.setattr(m, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(m, "_run_gate", run_gate)
    error = m.validate_development_push(root)
    assert error == f"{gate} failed (exit 1): specific detail"


def test_run_gate_uses_uv_without_sync(monkeypatch, tmp_path):
    m = _load()
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(m.subprocess, "run", run)
    m._run_gate(tmp_path, "uv", "scripts/governance/validate_policy_registry.py")
    argv, kwargs = calls[0]
    assert argv[:6] == ["uv", "run", "--no-sync", "--project", str(tmp_path), "python"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["capture_output"] is True


@pytest.fixture
def ff_repo(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.delenv("IGX_HOOKS", raising=False)
    monkeypatch.delenv("IGX_CURSOR_PUSH_PRECHECK_SKIP", raising=False)
    monkeypatch.setattr(m, "_uv_executable", lambda: "uv")

    def run_gate(cwd, _uv, script, *args):
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(m, "_run_gate", run_gate)
    repo, base, head = _bootstrap_ff_development(tmp_path)
    return m, repo, base, head


def _assert_unchanged(repo: Path, before: dict[str, str | bytes]) -> None:
    after = _snapshot_tree(repo)
    assert after == before


@pytest.mark.parametrize(
    "mutate",
    [
        "clean",
        "untracked",
        "modified_tracked",
        "staged_only",
        "mixed",
    ],
)
def test_dirty_working_tree_positive_cases(ff_repo, mutate: str) -> None:
    m, repo, _base, _head = ff_repo
    if mutate == "untracked":
        (repo / "untracked.txt").write_text("noise\n", encoding="utf-8")
    elif mutate == "modified_tracked":
        (repo / "tracked.txt").write_text("dirty working tree\n", encoding="utf-8")
    elif mutate == "staged_only":
        (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
        subprocess.run(["git", *_GIT_IDENT, "add", "staged.txt"], cwd=repo, check=True)
    elif mutate == "mixed":
        (repo / "untracked.txt").write_text("u\n", encoding="utf-8")
        (repo / "tracked.txt").write_text("mix\n", encoding="utf-8")
        (repo / "stage-me.txt").write_text("s\n", encoding="utf-8")
        subprocess.run(["git", *_GIT_IDENT, "add", "stage-me.txt"], cwd=repo, check=True)
    before = _snapshot_tree(repo)
    assert m.validate_development_push(repo) is None
    _assert_unchanged(repo, before)


def test_dirty_tree_ticket_ref_gate_still_denies(ff_repo, monkeypatch) -> None:
    m, repo, _base, _head = ff_repo
    (repo / "noise.txt").write_text("x\n", encoding="utf-8")
    before = _snapshot_tree(repo)

    def run_gate(_cwd, _uv, script, *args):
        if "verify_commit" in script:
            return subprocess.CompletedProcess(args, 1, "", "ticket ref failure")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(m, "_run_gate", run_gate)
    error = m.validate_development_push(repo)
    assert error is not None
    assert error.startswith("Ticket Ref gate failed")
    _assert_unchanged(repo, before)


def test_dirty_tree_policy_registry_gate_still_denies(ff_repo, monkeypatch) -> None:
    m, repo, _base, _head = ff_repo
    (repo / "tracked.txt").write_text("y\n", encoding="utf-8")
    before = _snapshot_tree(repo)

    def run_gate(_cwd, _uv, script, *args):
        if "validate_policy_registry" in script:
            return subprocess.CompletedProcess(args, 1, "", "policy failure")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(m, "_run_gate", run_gate)
    error = m.validate_development_push(repo)
    assert error is not None
    assert error.startswith("Policy registry validation failed")
    _assert_unchanged(repo, before)


def test_local_development_not_fast_forward_denies(tmp_path, monkeypatch) -> None:
    m = _load()
    monkeypatch.delenv("IGX_HOOKS", raising=False)
    monkeypatch.delenv("IGX_CURSOR_PUSH_PRECHECK_SKIP", raising=False)
    monkeypatch.setattr(m, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(
        m,
        "_run_gate",
        lambda *_a, **_k: subprocess.CompletedProcess((), 0, "", ""),
    )
    repo, _base, _head = _bootstrap_ff_development(tmp_path)
    (repo / "remote_only.txt").write_text("remote\n", encoding="utf-8")
    subprocess.run(["git", *_GIT_IDENT, "add", "remote_only.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", *_GIT_IDENT, "commit", "-q", "-m", "remote\n\nRef: governance/tickets/T-IGX-058-D.md"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "push", "-q", "origin", "development"], cwd=repo, check=True)
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=repo, check=True)
    subprocess.run(["git", "fetch", "origin", "development"], cwd=repo, check=True)
    (repo / "dirty.txt").write_text("d\n", encoding="utf-8")
    before = _snapshot_tree(repo)
    error = m.validate_development_push(repo)
    assert error is not None
    assert "fast-forward" in error
    _assert_unchanged(repo, before)


def test_missing_origin_development_denies(tmp_path, monkeypatch) -> None:
    m = _load()
    monkeypatch.delenv("IGX_HOOKS", raising=False)
    monkeypatch.delenv("IGX_CURSOR_PUSH_PRECHECK_SKIP", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "development"], cwd=repo, check=True)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    subprocess.run(["git", *_GIT_IDENT, "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", *_GIT_IDENT, "commit", "-q", "-m", "c\n\nRef: governance/tickets/T-IGX-058-D.md"],
        cwd=repo,
        check=True,
    )
    (repo / "noise.txt").write_text("n\n", encoding="utf-8")
    before = _snapshot_tree(repo)
    error = m.validate_development_push(repo)
    assert error is not None
    assert "origin/development" in error
    _assert_unchanged(repo, before)


def test_wrong_current_branch_denies(tmp_path, monkeypatch) -> None:
    m = _load()
    monkeypatch.delenv("IGX_HOOKS", raising=False)
    monkeypatch.delenv("IGX_CURSOR_PUSH_PRECHECK_SKIP", raising=False)
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "development"], cwd=repo, check=True)
    (repo / "f.txt").write_text("f\n", encoding="utf-8")
    subprocess.run(["git", *_GIT_IDENT, "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", *_GIT_IDENT, "commit", "-q", "-m", "c\n\nRef: governance/tickets/T-IGX-058-D.md"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=repo, check=True)
    subprocess.run(["git", "push", "-q", "origin", "development"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=repo, check=True)
    (repo / "noise.txt").write_text("n\n", encoding="utf-8")
    before = _snapshot_tree(repo)
    error = m.validate_development_push(repo)
    assert error is not None
    assert "development" in error
    _assert_unchanged(repo, before)
