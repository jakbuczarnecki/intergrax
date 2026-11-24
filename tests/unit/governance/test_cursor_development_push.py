from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MOD = ROOT / "scripts/hooks/cursor_development_push.py"


def _load():
    spec = importlib.util.spec_from_file_location("cursor_development_push_test", MOD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            ("status", "--porcelain"): "",
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
            ("status", "--porcelain"): "",
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
