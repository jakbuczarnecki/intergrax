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


@pytest.mark.parametrize("field,value", [
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
])
def test_packet_denies_unapproved_conditions(mod, field, value) -> None:
    p = packet()
    p[field] = value
    with pytest.raises(mod.Blocked):
        mod.validate_packet(p)


@pytest.mark.parametrize("name", [
    ".claude/settings.json", "CLAUDE.md", "agent-env/rules/intergrax-core.md",
    "governance/agent-control-plane/role_registry.yaml",
    "scripts/cursor/igx_integrator.py", "scripts/hooks/igx_stop_guard.py",
    "scripts/governance/verify_trusted_policy_pin.py",
    "docs/project/maintainers/qualification/ARCH_FREEZE_SHA_LADDER.md",
    "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md",
    "docs/project/maintainers/qualification/INTEGRAX_ROADMAP.md",
    "../evil.py", "intergrax/../../evil.py",
])
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


def test_pr_requires_exact_base_head_and_allowed_files(mod, monkeypatch) -> None:
    p = packet()
    number = 17
    pr = {
        "state": "open", "draft": False, "merged": False,
        "head": {"sha": SHA1, "ref": p["branch"]},
        "base": {"sha": SHA2, "ref": p["target"]},
    }
    monkeypatch.setattr(mod, "gh_api", lambda path: pr if path == "pulls/17" else
                        {"object": {"sha": SHA2}} if path.startswith("git/ref/heads/") else
                        [{"filename": "intergrax/allowed.py", "patch": "@@ -1 +1 @@"}])
    mod.verify_pr(p, number)
    pr["head"]["sha"] = SHA2
    with pytest.raises(mod.Blocked, match="head changed"):
        mod.verify_pr(p, number)


@pytest.mark.parametrize("checks", [
    [], [{"name": "smoke", "state": "FAILURE"}],
    [{"name": "smoke", "state": "PENDING"}],
    [{"name": "smoke", "state": "SKIPPING"}],
])
def test_required_checks_fail_closed(mod, monkeypatch, checks) -> None:
    monkeypatch.setattr(mod, "command", lambda argv, **kwargs: json.dumps(checks))
    with pytest.raises(mod.Blocked):
        mod.required_checks(17)


def test_required_checks_accept_success(mod, monkeypatch) -> None:
    monkeypatch.setattr(mod, "command", lambda argv, **kwargs:
                        json.dumps([{"name": "smoke", "state": "SUCCESS"}]))
    mod.required_checks(17)


@pytest.mark.parametrize("shell", [
    "git push origin HEAD:refs/heads/development",
    "git.exe push origin mybranch",
    "gh pr merge 17 --admin",
    "gh.exe pr create --base main",
    "gh api repos/a/b/pulls -X POST",
    "git reset --hard origin/development",
])
def test_cursor_shell_hook_denies_direct_mutations(shell: str) -> None:
    assert load(GUARD).decide({"command": shell})["permission"] == "deny"


@pytest.mark.parametrize("shell", [
    "git status --short", "gh pr view 17", "git diff --check",
    "python scripts/cursor/igx_integrator.py merge --packet .tmp/proof/packet.json",
])
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
    current_claude = (ROOT / "governance/agent-control-plane/adapters/claude/DELIVERY_DEVOS.fragment.md").read_text(encoding="utf-8")
    banner = "<!-- GENERATED by sync_adapter_fragments.py — ADVISORY; merge manually into harness docs -->\n\n"
    assert current_claude == banner + content
