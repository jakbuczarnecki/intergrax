# © Artur Czarnecki. All rights reserved.
"""Codex adapter coverage, authority boundaries, and native lifecycle payloads."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_guard():
    spec = importlib.util.spec_from_file_location("codex_guard_test", ROOT / ".codex/hooks/igx_codex_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guard(tmp_path, monkeypatch):
    module = load_guard()
    (tmp_path / ".git").mkdir()
    monkeypatch.delenv("IGX_HOOKS", raising=False)
    return module, tmp_path


def event(root, name, session="owner", **kwargs):
    return {"cwd": str(root), "hook_event_name": name, "session_id": session, **kwargs}


def activate(module, root, session="owner"):
    plan = root / "plan with spaces.md"
    plan.write_text("# Plan\n- [ ] first\n- [ ] second\n", encoding="utf-8")
    result = module.handle(
        event(
            root,
            "UserPromptSubmit",
            session,
            prompt='$intergrax-codex:codex-igx-run-plan "plan with spaces.md"',
        )
    )
    assert "activated" in result["hookSpecificOutput"]["additionalContext"]
    return plan


def test_all_canonical_workflows_have_codex_adapter_and_invocation_policy():
    expected = {}
    for source in (ROOT / "agent-env/skills").glob("*/SKILL.md"):
        meta = yaml.safe_load(source.read_text(encoding="utf-8").split("---", 2)[1])
        expected[source.parent.name] = (source, meta.get("disable-model-invocation") is True)
    for source in (ROOT / "agent-env/rules").glob("*.md"):
        meta = yaml.safe_load(source.read_text(encoding="utf-8").split("---", 2)[1])
        if meta["loading"] in ("auto", "explicit"):
            expected[source.stem] = (source, meta["loading"] == "explicit")
    folders = ROOT / ".codex/plugin/skills"
    assert {p.name for p in folders.iterdir()} == {f"codex-{name}" for name in expected}
    for name, (source, explicit) in expected.items():
        adapter = folders / f"codex-{name}"
        text = (adapter / "SKILL.md").read_text(encoding="utf-8")
        meta = yaml.safe_load(text.split("---", 2)[1])
        assert meta["name"] == f"codex-{name}"
        assert source.relative_to(ROOT).as_posix() in text
        policy = yaml.safe_load((adapter / "agents/openai.yaml").read_text(encoding="utf-8"))
        assert policy["policy"]["allow_implicit_invocation"] is not explicit


def test_codex_guidance_contains_root_stub_and_always_rules():
    text = (ROOT / "AGENTS.override.md").read_text(encoding="utf-8")
    assert (ROOT / "AGENTS.md").read_text(encoding="utf-8").strip() in text
    for phrase in ("No AI-tool signature", "One `uv run pytest`", "may_commit", "English only", "No polling"):
        assert phrase in text
    assert (ROOT / ".codex/AGENTS.md").read_text(encoding="utf-8").strip() in text
    assert len(text.encode()) < 32768


def test_native_profiles_preserve_registered_roles_without_foreign_model_or_sandbox():
    roles = yaml.safe_load((ROOT / "governance/agent-control-plane/role_registry.yaml").read_text(encoding="utf-8"))
    profiles = {p.stem: tomllib.loads(p.read_text(encoding="utf-8")) for p in (ROOT / ".codex/agents").glob("*.toml")}
    assert set(profiles) == load_guard().AGENTS
    for name, profile in profiles.items():
        assert profile["name"] == name
        assert roles["subagent_role_map"][name]["maps_to"] in profile["developer_instructions"]
        assert not {"model", "model_reasoning_effort", "sandbox_mode", "approval_policy", "isolation"} & profile.keys()
        assert not any(model in profile["developer_instructions"] for model in ("`opus`", "`sonnet`", "`haiku`"))
    for name in ("igx-researcher", "igx-reviewer"):
        assert profiles[name]["features"]["shell_tool"] is False
    assert "Never edit product" in profiles["ci-investigator"]["developer_instructions"]


def test_project_config_is_portable_and_disables_only_antigravity_collisions():
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text(encoding="utf-8"))
    assert set(config) == {"agents", "features", "plugins", "skills"}
    assert config["agents"]["max_concurrent_threads_per_session"] == 3
    disabled = {entry["name"] for entry in config["skills"]["config"] if not entry["enabled"]}
    assert disabled == {path.parent.name for path in (ROOT / ".agents/skills").glob("*/SKILL.md")}
    catalog = json.loads((ROOT / ".agents/plugins/marketplace.json").read_text(encoding="utf-8"))
    plugin = catalog["plugins"][0]
    assert (ROOT / plugin["source"]["path"] / "plugin.json").is_file()
    assert config["plugins"][f"{plugin['name']}@{catalog['name']}"]["enabled"]


@pytest.mark.parametrize(
    "command",
    [
        "git push origin development",
        "git -C other push --force origin branch",
        "git reset --hard HEAD",
        "git commit --no-verify -m x",
        "$env:INTERGRAX_SKIP_GATE='1'",
        "git switch -c bad",
        "git checkout HEAD~1",
        "gh pr create --title x",
        "Get-Content .env",
        'Get-Content "D:\\other\\.env.production"',
    ],
)
def test_shell_guards_deny_forbidden_actions(guard, command):
    module, root = guard
    response = module.handle(event(root, "PreToolUse", tool_name="Bash", tool_input={"command": command}))
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    "command", ["git status --short", "rg -n symbol intergrax/runtime", "uv run pytest tests/unit/governance -q"]
)
def test_shell_allows_scoped_read_and_verification(guard, command):
    module, root = guard
    assert module.handle(event(root, "PreToolUse", tool_name="Bash", tool_input={"command": command})) == {}


@pytest.mark.parametrize("name", ["default", "worker", "general-purpose", "igx-integrator", "security-review"])
def test_generic_cursor_only_and_audit_children_are_denied(guard, name):
    module, root = guard
    response = module.handle(event(root, "PreToolUse", tool_name="spawn_agent", tool_input={"agent_type": name}))
    assert response["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_secret_read_and_patch_denials(guard):
    module, root = guard
    for name, args in (
        ("Read", {"file_path": ".env.local"}),
        ("apply_patch", {"command": "*** Begin Patch\n*** Update File: folder/.env\n@@\n+x\n*** End Patch"}),
    ):
        assert (
            module.handle(event(root, "PreToolUse", tool_name=name, tool_input=args))["hookSpecificOutput"][
                "permissionDecision"
            ]
            == "deny"
        )


def test_plan_ownership_progress_completion_and_foreign_state(guard):
    module, root = guard
    foreign = root / ".tmp/session/autonomy/active-plan"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("another harness", encoding="utf-8")
    plan = activate(module, root)
    assert module.handle(event(root, "Stop", "other", last_assistant_message="done")) == {}
    first = module.handle(event(root, "Stop", turn_id="1", last_assistant_message="PLAN COMPLETE"))
    assert first["decision"] == "block"
    assert module.handle(event(root, "Stop", turn_id="1")) == first
    plan.write_text("- [x] first\n- [ ] second\n", encoding="utf-8")
    assert "second" in module.handle(event(root, "Stop", turn_id="2"))["reason"]
    plan.write_text("- [x] first\n- [x] second\n", encoding="utf-8")
    assert module.handle(event(root, "Stop", turn_id="3")) == {}
    assert not module.state_path(root, "owner").exists()
    assert foreign.read_text(encoding="utf-8") == "another harness"


def test_plan_escalation_interrupt_stop_and_stall_cap(guard):
    module, root = guard
    activate(module, root)
    assert module.handle(event(root, "Stop", last_assistant_message="ESCALATION: missing access")) == {}
    assert module.handle(event(root, "Stop", turn_id="1"))["decision"] == "block"
    assert module.handle(event(root, "Stop", turn_id="1", stop_hook_active=True))["decision"] == "block"
    assert "stalled" in module.handle(event(root, "Stop", turn_id="1", stop_hook_active=True))["systemMessage"]
    module.handle(event(root, "Interrupt"))
    assert module.handle(event(root, "Stop", turn_id="4")) == {}
    module.handle(event(root, "UserPromptSubmit", prompt="$codex-igx-run-plan stop"))
    assert not module.state_path(root, "owner").exists()


def test_plan_rejects_outside_worktree_and_kill_switch(guard, monkeypatch):
    module, root = guard
    response = module.handle(event(root, "UserPromptSubmit", prompt="$codex-igx-run-plan ../outside.md"))
    assert "not activated" in response["hookSpecificOutput"]["additionalContext"]
    assert not module.state_path(root, "owner").exists()
    monkeypatch.setenv("IGX_HOOKS", "off")
    assert module.handle(event(root, "PreToolUse", tool_name="Bash", tool_input={"command": "git push"})) == {}


def test_registered_hook_command_runs_from_nested_directory():
    hooks = json.loads((ROOT / ".codex/hooks.json").read_text(encoding="utf-8"))["hooks"]
    commands = {entry["hooks"][0]["command"] for entries in hooks.values() for entry in entries}
    assert len(commands) == 1
    command = next(iter(commands))
    # Run the exact Python program from the registered Windows-safe command.
    program = command.removeprefix('python -c "').removesuffix('"')
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT / "tests/unit/governance",
        input=json.dumps(
            event(ROOT, "PreToolUse", tool_name="Bash", tool_input={"command": "git push origin development"})
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("install_status", [0, 1])
def test_installer_preserves_user_config_after_success_or_failure(tmp_path, monkeypatch, install_status):
    spec = importlib.util.spec_from_file_location("codex_setup_test", ROOT / ".codex/setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    home = tmp_path / "user"
    home.mkdir()
    config = home / "config.toml"
    original = b'model = "existing-model"\n\n[plugins.existing]\nenabled = false\n'
    config.write_bytes(original)
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(setup.shutil, "which", lambda _: "codex")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args, 0)
        config.write_bytes(original + f'\n[plugins."{setup.PLUGIN}"]\nenabled = true\n'.encode())
        return subprocess.CompletedProcess(args, install_status, stdout="{}", stderr="")

    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.main() == install_status
    assert config.read_bytes() == original
    assert len(calls) == 2


def test_large_dump_guard_allows_a_bounded_powershell_read():
    module = load_guard()
    command = f'Get-Content "{ROOT / "uv.lock"}"'
    denied = module.handle(event(ROOT, "PreToolUse", tool_name="Bash", tool_input={"command": command}))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert (
        module.handle(
            event(
                ROOT,
                "PreToolUse",
                tool_name="Bash",
                tool_input={
                    "command": command + " -TotalCount 20",
                },
            )
        )
        == {}
    )
