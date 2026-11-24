"""Cursor candidate publication policy: no network or actual remote writes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SHA = "a" * 40
HEAD = "b" * 40


@pytest.fixture
def mod(monkeypatch):
    directory = ROOT / "scripts/cursor"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("cursor_candidate_test", directory / "igx_publish_candidate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packet():
    return {
        "schema": 1,
        "wave": "EC-V3-W01",
        "mode": "CHATGPT_DIRECT_IMPLEMENTATION",
        "target": "arch-freeze/rw-train",
        "kind": "product",
        "branch": "ec-v3/w01-vision-fix",
        "base_sha": SHA,
        "ticket": "governance/tickets/T-IGX-043-C.md",
        "write_set": ["intergrax/allowed.py"],
        "pr_title": "fix: candidate publication",
    }


def test_no_review_or_grok_required(mod):
    mod.validate_authorization(packet(), "EC-V3-W01")


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "main"),
        ("branch", "development"),
        ("branch", "ec-v3/../main"),
        ("branch", "ec-v3/x.lock"),
        ("mode", "legacy"),
        ("base_sha", "bad"),
        ("write_set", [".claude/settings.json"]),
        ("write_set", [".cursor/publication/EC-V3-W01.json"]),
        ("write_set", ["scripts/cursor/igx_integrator.py"]),
        ("write_set", ["governance/tickets/T-IGX-043-C.md"]),
    ],
)
def test_denied_authorizations(mod, field, value):
    p = packet()
    p[field] = value
    with pytest.raises(mod.Blocked):
        mod.validate_authorization(p, p["wave"])


def test_diff_scope(mod):
    p = packet()
    mod.assert_scoped({"intergrax/allowed.py"}, p)
    with pytest.raises(mod.Blocked):
        mod.assert_scoped({"intergrax/allowed.py", "intergrax/extra.py"}, p)


def test_target_advancement_checks_ancestry_not_equality(mod, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mod, "remote_sha", lambda *_: HEAD)
    monkeypatch.setattr(mod, "git", lambda _, *args: calls.append(args) or "")
    mod.verify_target(packet(), tmp_path)
    assert ("merge-base", "--is-ancestor", SHA, HEAD) in calls


def test_publish_creates_only_draft_and_is_idempotent(mod, monkeypatch, tmp_path):
    p = packet()
    calls = []
    drafts = []
    remote = [None]

    def git(_, *args):
        calls.append(args)
        if args[0] == "push":
            remote[0] = HEAD
        return ""

    def command(args, **_):
        calls.append(tuple(args))
        if args[:3] == ["gh", "pr", "create"]:
            drafts.append({"number": 17})
        return ""

    monkeypatch.setattr(mod, "verify_local", lambda *_: HEAD)
    monkeypatch.setattr(mod, "verify_target", lambda *_: None)
    monkeypatch.setattr(mod, "open_prs", lambda _: list(drafts))
    monkeypatch.setattr(mod, "remote_sha", lambda *_: remote[0])
    monkeypatch.setattr(mod, "git", git)
    monkeypatch.setattr(mod, "command", command)
    monkeypatch.setattr(mod, "verify_draft", lambda *_: {"html_url": "https://example.invalid/pr/17"})

    assert mod.publish(p, tmp_path)["result"] == "PUBLISHED_DRAFT"
    assert mod.publish(p, tmp_path)["pr"] == 17
    pushes = [call for call in calls if call[0] == "push"]
    creates = [call for call in calls if call[:3] == ("gh", "pr", "create")]
    assert len(pushes) == 1
    assert pushes[0][-1] == f"{HEAD}:refs/heads/{p['branch']}"
    assert len(creates) == 1 and "--draft" in creates[0]
    assert all("merge" not in call and "--force" not in call for call in calls)


def test_ready_pr_is_refused(mod, monkeypatch):
    p = packet()
    repo = {"full_name": mod.REPO}
    monkeypatch.setattr(
        mod,
        "gh_api",
        lambda _: {
            "state": "open",
            "draft": False,
            "head": {"ref": p["branch"], "repo": repo},
            "base": {"ref": p["target"], "repo": repo},
        },
    )
    with pytest.raises(mod.Blocked, match="draft"):
        mod.verify_draft(p, 17)


def test_authorization_must_come_from_canonical_development(mod, monkeypatch):
    import base64
    import json

    p = packet()
    requested = []

    def api(path):
        requested.append(path)
        return {
            "type": "file",
            "encoding": "base64",
            "content": base64.b64encode(json.dumps(p).encode()).decode(),
        }

    monkeypatch.setattr(mod, "gh_api", api)
    assert mod.read_authorization(p["wave"]) == p
    assert requested == ["contents/.cursor/publication/EC-V3-W01.json?ref=development"]
    monkeypatch.setattr(mod, "gh_api", lambda _: {"type": "file", "encoding": "base64", "content": "!"})
    with pytest.raises(mod.Blocked):
        mod.read_authorization(p["wave"])


@pytest.mark.parametrize("path", ["scripts/Governance/hidden.py", "intergrax/foo.py:stream", "intergrax/dir./x.py"])
def test_windows_alias_paths_are_refused(mod, path):
    p = packet()
    p["write_set"] = [path]
    with pytest.raises(mod.Blocked):
        mod.validate_authorization(p, p["wave"])


def test_draft_wrong_target_or_head_is_refused(mod, monkeypatch):
    p = packet()
    repo = {"full_name": mod.REPO}
    pr = {
        "state": "open",
        "draft": True,
        "head": {"ref": p["branch"], "sha": HEAD, "repo": repo},
        "base": {"ref": "main", "repo": repo},
    }
    monkeypatch.setattr(mod, "gh_api", lambda _: pr)
    with pytest.raises(mod.Blocked, match="draft"):
        mod.verify_draft(p, 17, HEAD)
    pr["base"]["ref"] = p["target"]
    with pytest.raises(mod.Blocked, match="head"):
        mod.verify_draft(p, 17, SHA)


def test_direct_mutations_still_denied():
    path = ROOT / "scripts/hooks/igx_cursor_integration_guard.py"
    spec = importlib.util.spec_from_file_location("cursor_candidate_guard_test", path)
    assert spec and spec.loader
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    assert guard.decide({"command": "git push origin ec-v3/w01-vision-fix"})["permission"] == "deny"
    assert guard.decide({"command": "gh pr create --draft --base arch-freeze/rw-train"})["permission"] == "deny"
    assert guard.decide({"command": "gh pr merge 17 --admin"})["permission"] == "deny"
    assert guard.decide({"command": "python scripts/cursor/igx_publish_candidate.py --wave EC-V3-W01"})[
        "permission"
    ] == "allow"


def test_preflight_is_read_only_for_fresh_and_existing_candidate(mod, monkeypatch, tmp_path):
    p = packet()
    calls = []
    head = [SHA]
    monkeypatch.setattr(mod, "verify_checkout", lambda *_: head[0])
    monkeypatch.setattr(mod, "verify_local", lambda *_: calls.append("verify_local") or head[0])
    monkeypatch.setattr(mod, "verify_remote_candidate", lambda *_: (None, None))
    monkeypatch.setattr(mod, "command", lambda *_a, **_kw: pytest.fail("unexpected remote mutation"))
    monkeypatch.setattr(mod, "git", lambda *_a: pytest.fail("unexpected git operation"))
    result = mod.preflight(p, tmp_path)
    assert result["result"] == "PREFLIGHT_OK"
    assert result["state"] == "BASE_READY"
    assert calls == []
    head[0] = HEAD
    result = mod.preflight(p, tmp_path)
    assert result["state"] == "CANDIDATE_PRESENT"
    assert calls == ["verify_local"]


def test_preflight_refuses_remote_divergence_before_any_push(mod, monkeypatch, tmp_path):
    p = packet()
    monkeypatch.setattr(mod, "verify_checkout", lambda *_: SHA)
    monkeypatch.setattr(mod, "verify_target", lambda *_: None)
    monkeypatch.setattr(mod, "open_prs", lambda _: [])
    monkeypatch.setattr(mod, "remote_sha", lambda *_: HEAD)
    calls = []

    def git(_, *args):
        calls.append(args)
        if args[0] == "merge-base":
            raise mod.Blocked("remote candidate diverged")
        return ""

    monkeypatch.setattr(mod, "git", git)
    with pytest.raises(mod.Blocked, match="diverged"):
        mod.preflight(p, tmp_path)
    assert [call[0] for call in calls] == ["fetch", "merge-base"]


def test_candidate_requires_exactly_one_ref_per_commit(mod, monkeypatch, tmp_path):
    p = packet()
    monkeypatch.setattr(mod, "verify_checkout", lambda *_: HEAD)
    monkeypatch.setattr(mod, "diff_paths", lambda *_: {"intergrax/allowed.py"})
    message = [f"fix: candidate\n\nRef: {p['ticket']}\n"]

    def git(_, *args):
        if args[0] == "rev-list":
            return HEAD
        if args[2] == "--format=%P":
            return SHA
        if args[2] == "--format=%B":
            return message[0]
        raise AssertionError(args)

    monkeypatch.setattr(mod, "git", git)
    assert mod.verify_local(p, tmp_path) == HEAD
    message[0] += "Ref: governance/tickets/T-IGX-058-A.md\n"
    with pytest.raises(mod.Blocked, match="exactly one"):
        mod.verify_local(p, tmp_path)


def test_cli_preflight_does_not_call_publish(mod, monkeypatch, capsys):
    import sys

    p = packet()
    monkeypatch.setattr(sys, "argv", ["publisher", "--wave", p["wave"], "--preflight"])
    monkeypatch.setattr(mod, "read_authorization", lambda _: p)
    monkeypatch.setattr(mod, "preflight", lambda *_: {"result": "PREFLIGHT_OK"})
    monkeypatch.setattr(mod, "publish", lambda *_: pytest.fail("preflight must not publish"))
    assert mod.main() == 0
    assert '"result": "PREFLIGHT_OK"' in capsys.readouterr().out


def test_live_w01_command_and_preflight_pass_cursor_shell_guard():
    path = ROOT / "scripts/hooks/igx_cursor_integration_guard.py"
    spec = importlib.util.spec_from_file_location("cursor_candidate_live_guard_test", path)
    assert spec and spec.loader
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    command = (
        "Set-Location D:/Projekty/_wt/intergrax/ec-w01; "
        'python "D:/Projekty/intergrax/scripts/cursor/igx_publish_candidate.py" --wave EC-V3-W01'
    )
    assert guard.decide({"command": command})["permission"] == "allow"
    assert guard.decide({"command": command + " --preflight"})["permission"] == "allow"
