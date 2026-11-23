# © Artur Czarnecki. All rights reserved.
"""Cursor-only OD-AF-INT A regression and negative tests (no network calls)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EXEC = ROOT / "scripts/cursor/igx_integrator.py"
GUARD = ROOT / "scripts/hooks/igx_cursor_integration_guard.py"
SHA1 = "a" * 40
SHA2 = "b" * 40


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return load(EXEC)


def packet():
    return {
        "operator_decision": "Approved: OD-AF-INT A",
        "target": "arch-freeze/rw-train",
        "kind": "product",
        "branch": "af2/t-igx-055-c",
        "head_sha": SHA1,
        "base_sha": SHA2,
        "ticket": "governance/tickets/T-IGX-055-C.md",
        "write_set": ["intergrax/allowed.py"],
        "implementer": "worker-1",
        "reviewer": "reviewer-2",
        "reviewer_role": "igx-reviewer",
        "review_verdict": "ACCEPT",
        "review_head_sha": SHA1,
        "review_report_sha256": "c" * 64,
        "review_report_file": ".tmp/proof/af/review.json",
    }


def test_nominal_packet_scoped_and_sha_bound(mod) -> None:
    mod.validate_packet(packet())


@pytest.mark.parametrize(
    "field,value",
    [
        ("operator_decision", None),
        ("operator_decision", "Approved: unrelated"),
        ("target", "main"),
        ("target", "development"),
        ("branch", "development"),
        ("branch", "arch-freeze/rw-train"),
        ("head_sha", "abc"),
        ("review_head_sha", SHA2),
        ("reviewer", "worker-1"),
        ("reviewer_role", "igx-implementer"),
        ("review_verdict", "PASS"),
        ("review_report_sha256", "not-a-digest"),
        ("review_report_file", "../review.json"),
    ],
)
def test_packet_denies_unapproved_conditions(mod, field, value) -> None:
    p = packet()
    p[field] = value
    with pytest.raises(mod.Blocked):
        mod.validate_packet(p)


@pytest.mark.parametrize(
    "name",
    [
        ".claude/settings.json",
        "CLAUDE.md",
        "agent-env/rules/intergrax-core.md",
        "governance/agent-control-plane/role_registry.yaml",
        "scripts/cursor/igx_integrator.py",
        "scripts/hooks/igx_stop_guard.py",
        "scripts/governance/verify_trusted_policy_pin.py",
        "docs/project/maintainers/qualification/ARCH_FREEZE_SHA_LADDER.md",
        "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md",
        "docs/project/maintainers/qualification/INTEGRAX_ROADMAP.md",
        "../evil.py",
        "intergrax/../../evil.py",
    ],
)
def test_product_integrator_cannot_touch_protected_paths(mod, name) -> None:
    assert mod.valid_path(name, "product") is False


def test_product_and_docs_target_separation(mod) -> None:
    assert mod.valid_path("intergrax/platform/foo.py", "product")
    assert not mod.valid_path("intergrax/platform/foo.py", "docs")
    assert mod.valid_path("docs/project/technical/guide.md", "docs")
    assert mod.valid_path("governance/tickets/T-IGX-055-C.md", "docs")
    assert not mod.valid_path("governance/TRUSTED_POLICY_PIN.yaml", "docs")
    assert not mod.valid_path("docs/project/maintainers/qualification/ARCH_FREEZE_SHA_LADDER.md", "docs")


def test_review_evidence_tamper_detected(mod, tmp_path: Path) -> None:
    p = packet()
    place = tmp_path / ".tmp/proof/af/review.json"
    place.parent.mkdir(parents=True)
    place.write_text(json.dumps({"head_sha": SHA1, "reviewer": "reviewer-2", "verdict": "ACCEPT"}))
    import hashlib

    p["review_report_sha256"] = hashlib.sha256(place.read_bytes()).hexdigest()
    mod.verify_evidence(p, tmp_path)
    place.write_text('{"head_sha":"changed"}')
    with pytest.raises(mod.Blocked, match="digest mismatch"):
        mod.verify_evidence(p, tmp_path)


def test_ticket_requires_operator_approval_and_state(mod, tmp_path: Path) -> None:
    p = packet()
    ticket = tmp_path / p["ticket"]
    ticket.parent.mkdir(parents=True)
    ticket.write_text("## Lifecycle state\nin_progress\n\nApproved: OD-AF-INT A\n")
    mod.check_ticket(p, tmp_path)
    ticket.write_text("## Lifecycle state\nplan_pending\n\nApproved: OD-AF-INT A\n")
    with pytest.raises(mod.Blocked):
        mod.check_ticket(p, tmp_path)


def test_ticket_on_development_can_qualify_an_older_train_worktree(mod, monkeypatch, tmp_path: Path) -> None:
    import base64

    p = packet()
    remote = "## Lifecycle state\nin_progress\n\nApproved: operator @ 2026-11-22T09:04:11+01:00\n"

    def fetch(path: str):
        assert path == f"contents/{p['ticket']}?ref=development"
        return {"encoding": "base64", "content": base64.b64encode(remote.encode()).decode()}

    monkeypatch.setattr(mod, "gh_api", fetch)
    mod.check_ticket(p, tmp_path)


def test_pr_requires_exact_base_head_and_allowed_files(mod, monkeypatch) -> None:
    p = packet()
    number = 17
    pr = {
        "state": "open",
        "draft": False,
        "merged": False,
        "head": {"sha": SHA1, "ref": p["branch"]},
        "base": {"sha": SHA2, "ref": p["target"]},
    }
    monkeypatch.setattr(
        mod,
        "gh_api",
        lambda path: (
            pr
            if path == "pulls/17"
            else {"object": {"sha": SHA2}}
            if path.startswith("git/ref/heads/")
            else [{"filename": "intergrax/allowed.py", "patch": "@@ -1 +1 @@"}]
        ),
    )
    mod.verify_pr(p, number)
    pr["head"]["sha"] = SHA2
    with pytest.raises(mod.Blocked, match="head changed"):
        mod.verify_pr(p, number)


def _patch_pr_checks(monkeypatch, mod, *, required, all_checks):
    def parse(argv: list[str]):
        if "--required" in argv:
            return required
        return all_checks

    monkeypatch.setattr(mod, "parse_pr_checks", parse)


@pytest.mark.parametrize(
    "required,all_checks,kind",
    [
        ([], [{"name": "smoke", "state": "SUCCESS"}], "product"),
        ([{"name": "smoke", "state": "FAILURE"}], [{"name": "smoke", "state": "FAILURE"}], "product"),
        ([{"name": "smoke", "state": "PENDING"}], [{"name": "smoke", "state": "PENDING"}], "product"),
        ([{"name": "smoke", "state": "SKIPPING"}], [{"name": "smoke", "state": "SKIPPING"}], "product"),
    ],
)
def test_required_checks_fail_closed(mod, monkeypatch, required, all_checks, kind) -> None:
    _patch_pr_checks(monkeypatch, mod, required=required, all_checks=all_checks)
    with pytest.raises(mod.Blocked):
        mod.required_checks(17, kind=kind)


def test_required_checks_accept_success(mod, monkeypatch) -> None:
    ok = [{"name": "smoke", "state": "SUCCESS"}]
    _patch_pr_checks(monkeypatch, mod, required=ok, all_checks=ok)
    mod.required_checks(17, kind="product")


def test_docs_profile_passes_without_github_required(mod, monkeypatch) -> None:
    all_ok = [{"name": name, "state": "SUCCESS"} for name in mod.DOC_CI_PROFILE]
    _patch_pr_checks(monkeypatch, mod, required=[], all_checks=all_ok)
    mod.required_checks(756, kind="docs")


def test_docs_profile_blocks_skipped_mandatory(mod, monkeypatch) -> None:
    checks = [{"name": name, "state": "SUCCESS"} for name in mod.DOC_CI_PROFILE]
    for index, item in enumerate(checks):
        if item["name"] == "ci-smoke":
            checks[index] = {"name": "ci-smoke", "state": "SKIPPING"}
            break
    _patch_pr_checks(monkeypatch, mod, required=[], all_checks=checks)
    with pytest.raises(mod.Blocked, match="mandatory documentation check skipped"):
        mod.required_checks(756, kind="docs")


def test_parse_pr_checks_handles_no_github_required(mod, monkeypatch) -> None:
    class Done:
        returncode = 1
        stdout = b""
        stderr = b"no required checks reported on the branch"

    monkeypatch.setattr(mod, "run", lambda *a, **k: Done())
    assert (
        mod.parse_pr_checks(["gh", "pr", "checks", "756", "--required", "--json", "name,state", "-R", mod.REPO]) == []
    )


def test_docs_profile_blocks_missing_mandatory(mod, monkeypatch) -> None:
    partial = [{"name": "ci-smoke", "state": "SUCCESS"}]
    _patch_pr_checks(monkeypatch, mod, required=[], all_checks=partial)
    with pytest.raises(mod.Blocked, match="mandatory documentation check missing"):
        mod.required_checks(756, kind="docs")


def test_command_rejects_invalid_utf8_structured_output(mod, monkeypatch) -> None:
    class Done:
        returncode = 0
        stdout = b"\xff\xfe"
        stderr = b""

    monkeypatch.setattr(mod, "run", lambda *a, **k: Done())
    with pytest.raises(mod.Blocked, match="UTF-8"):
        mod.command(["gh", "pr", "view", "1", "--json", "number"])


def test_command_reports_unicode_stderr_on_failure(mod, monkeypatch) -> None:
    class Done:
        returncode = 1
        stdout = b""
        stderr = "Błąd repozytorium".encode()

    monkeypatch.setattr(mod, "run", lambda *a, **k: Done())
    with pytest.raises(mod.Blocked, match="Błąd"):
        mod.command(["gh", "status"])


def test_publish_reuses_existing_open_pr(mod, monkeypatch, tmp_path: Path, capsys) -> None:
    p = packet()
    p.update(
        {
            "target": "development",
            "kind": "docs",
            "branch": "integ/sfx-r2-corr-001",
            "write_set": ["docs/project/guide.md"],
            "pr_title": "Doc update for integrator test",
            "pr_body_file": ".tmp/proof/publish/body.md",
        }
    )
    body = tmp_path / p["pr_body_file"]
    body.parent.mkdir(parents=True)
    body.write_text("body", encoding="utf-8")
    create_calls: list[list[str]] = []

    def track(argv, **kwargs):
        create_calls.append(argv)
        if argv[:3] == ["git", "push", "origin"]:
            return ""
        if len(argv) > 2 and argv[1:3] == ["pr", "list"]:
            return json.dumps(
                [
                    {
                        "number": 756,
                        "headRefOid": p["head_sha"],
                        "baseRefName": "development",
                        "headRefName": p["branch"],
                        "state": "OPEN",
                    }
                ]
            )
        return ""

    monkeypatch.setattr(mod, "command", track)
    monkeypatch.setattr(mod, "verify_local", lambda *_: None)
    monkeypatch.setattr(mod, "target_sha", lambda _: p["base_sha"])
    monkeypatch.setattr(mod, "branch_head_sha", lambda _: p["head_sha"])
    mod.publish(p, tmp_path)
    assert not any(len(c) > 2 and c[1:3] == ["pr", "create"] for c in create_calls)
    out = json.loads(capsys.readouterr().out)
    assert out["pr"] == 756 and out["reused"] is True


def test_publish_blocks_duplicate_open_prs(mod, monkeypatch, tmp_path: Path) -> None:
    p = packet()
    p.update(
        {
            "target": "development",
            "kind": "docs",
            "branch": "integ/sfx-r2-corr-001",
            "write_set": ["docs/project/guide.md"],
            "pr_title": "Doc update for integrator test",
            "pr_body_file": ".tmp/proof/publish/body.md",
        }
    )
    body = tmp_path / p["pr_body_file"]
    body.parent.mkdir(parents=True)
    body.write_text("body", encoding="utf-8")

    def track(argv, **kwargs):
        if len(argv) > 2 and argv[1:3] == ["pr", "list"]:
            row = {
                "headRefOid": p["head_sha"],
                "baseRefName": "development",
                "headRefName": p["branch"],
                "state": "OPEN",
            }
            return json.dumps([{**row, "number": 756}, {**row, "number": 757}])
        return ""

    monkeypatch.setattr(mod, "command", track)
    monkeypatch.setattr(mod, "verify_local", lambda *_: None)
    monkeypatch.setattr(mod, "target_sha", lambda _: p["base_sha"])
    monkeypatch.setattr(mod, "branch_head_sha", lambda _: p["head_sha"])
    with pytest.raises(mod.Blocked, match="duplicate open PRs"):
        mod.publish(p, tmp_path)


def test_publish_blocks_remote_divergence(mod, monkeypatch, tmp_path: Path) -> None:
    p = packet()
    p.update(
        {
            "target": "development",
            "kind": "docs",
            "branch": "integ/sfx-r2-corr-001",
            "write_set": ["docs/project/guide.md"],
            "pr_title": "Doc update for integrator test",
            "pr_body_file": ".tmp/proof/publish/body.md",
        }
    )
    body = tmp_path / p["pr_body_file"]
    body.parent.mkdir(parents=True)
    body.write_text("body", encoding="utf-8")
    remote = "d" * 40

    def track(argv, **kwargs):
        if argv[:3] == ["git", "merge-base", "--is-ancestor"]:
            raise mod.Blocked("not ancestor")
        return ""

    monkeypatch.setattr(mod, "command", track)
    monkeypatch.setattr(mod, "verify_local", lambda *_: None)
    monkeypatch.setattr(mod, "target_sha", lambda _: p["base_sha"])
    monkeypatch.setattr(mod, "branch_head_sha", lambda _: remote)
    with pytest.raises(mod.Blocked, match="remote branch diverged"):
        mod.publish(p, tmp_path)


@pytest.mark.parametrize(
    "shell",
    [
        "git.exe push origin mybranch",
        "git push origin HEAD",
        "gh pr merge 17 --admin",
        "gh.exe pr create --base main",
        "gh api repos/a/b/pulls -X POST",
        "git reset --hard origin/development",
    ],
)
def test_cursor_shell_hook_denies_direct_mutations(shell: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IGX_CURSOR_PUSH_PRECHECK_SKIP", "1")
    assert load(GUARD).decide({"command": shell})["permission"] == "deny"


def test_cursor_shell_hook_allows_origin_development_push(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IGX_CURSOR_PUSH_PRECHECK_SKIP", "1")
    assert load(GUARD).decide({"command": "git push origin development"})["permission"] == "allow"
    assert load(GUARD).decide({"command": "git push origin HEAD:refs/heads/development"})["permission"] == "allow"


@pytest.mark.parametrize(
    "shell",
    [
        "git status --short",
        "gh pr view 17",
        "git diff --check",
        "python scripts/cursor/igx_integrator.py merge --packet .tmp/proof/packet.json",
    ],
)
def test_cursor_shell_hook_allows_safe_inspection_and_guarded_runner(shell: str) -> None:
    assert load(GUARD).decide({"command": shell})["permission"] == "allow"


def test_cursor_role_added_without_changing_existing_role_semantics() -> None:
    roles = yaml.safe_load((ROOT / "governance/agent-control-plane/role_registry.yaml").read_text())
    assert roles["default_role"] == "orchestrator"
    assert roles["roles"]["orchestrator"]["may_commit"] is False
    assert roles["roles"]["implementer"]["may_commit"] is True
    assert roles["roles"]["reviewer"]["may_commit"] is False
    assert roles["roles"]["integrator"]["runtime"] == "cursor"
    assert roles["subagent_role_map"]["igx-integrator"]["runtime"] == "cursor"
    assert not (ROOT / ".claude/agents/igx-integrator.md").exists()
    assert (ROOT / ".cursor/agents/igx-integrator.md").is_file()


def test_claude_codex_generated_fragments_exclude_cursor_only_roles() -> None:
    script = load(ROOT / "scripts/governance/sync_adapter_fragments.py")
    content = script._role_summary() + "\n" + script._subagent_summary()
    assert "**integrator**" not in content
    assert "| igx-integrator |" not in content


def test_cursor_integrator_not_exposed_as_a_claude_agent() -> None:
    assert not (ROOT / ".claude/agents/igx-integrator.md").exists()
    assert (ROOT / ".cursor/agents/igx-integrator.md").exists()
    fragment = (ROOT / "governance/agent-control-plane/adapters/claude/DELIVERY_DEVOS.fragment.md").read_text(
        encoding="utf-8"
    )
    assert "**integrator**" not in fragment
    assert "| igx-integrator |" not in fragment
