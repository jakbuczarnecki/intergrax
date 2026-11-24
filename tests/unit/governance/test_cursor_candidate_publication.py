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
