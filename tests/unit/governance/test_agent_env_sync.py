# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SYNC = ROOT / "scripts" / "governance" / "sync_agent_env.py"

ALWAYS_RULES = ["intergrax-core", "intergrax-english-only-commits"]
ALWAYS_MDC_MAX_BYTES = 6 * 1024
CORE_EXTRACT_PHRASES = (
    "Default branch",
    "ESCALATION:",
    "max 3 files",
    "git push",
    ".tmp/",
    "uv run pytest",
    "sleep loops",
)
OPERATOR_ONLY_SKILLS = (
    "igx-run-plan",
    "sfx-blueprint",
    "sfx-discover",
    "sfx-integration-setup",
    "sfx-present",
    "sfx-resolve",
    "sfx-session",
    "sfx-specify",
    "sfx-test-plan",
)


def test_agent_env_adapters_match_ssot() -> None:
    completed = subprocess.run(
        [sys.executable, str(SYNC), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_every_rule_has_loading_mode_and_description() -> None:
    sys.path.insert(0, str(SYNC.parent))
    try:
        import sync_agent_env  # noqa: WPS433
    finally:
        sys.path.remove(str(SYNC.parent))
    rules = sync_agent_env.load_rules()
    assert rules["intergrax-core"]["loading"] == "always"
    assert [rid for rid, meta in rules.items() if meta["loading"] == "always"] == ALWAYS_RULES
    assert sync_agent_env.size_problems() == []
    assert all(meta["loading"] in sync_agent_env.LOADING_MODES for meta in rules.values())


def _mdc_body(rule_id: str) -> str:
    text = (ROOT / ".cursor" / "rules" / f"{rule_id}.mdc").read_text(encoding="utf-8").replace("\r\n", "\n")
    return text.split("---\n", 2)[2]


def test_always_rules_include_generated_core_and_scoped_cursor_overrides() -> None:
    always = sorted(
        p.stem
        for p in (ROOT / ".cursor" / "rules").glob("*.mdc")
        if "alwaysApply: true" in p.read_text(encoding="utf-8").split("---", 2)[1]
    )
    assert always == sorted([*ALWAYS_RULES, "od-af-int-a", "cursor-native-orchestration"])


def test_always_core_mdc_inlines_the_invariants() -> None:
    body = _mdc_body("intergrax-core")
    for phrase in CORE_EXTRACT_PHRASES:
        assert phrase in body, f"always core .mdc lost: {phrase!r}"
    assert "English only" in _mdc_body("intergrax-english-only-commits")


def test_always_mdc_has_no_at_import_and_stays_small() -> None:
    bodies = [_mdc_body(rid) for rid in ALWAYS_RULES]
    for body in bodies:
        assert "@agent-env" not in body
    assert sum(len(b.encode("utf-8")) for b in bodies) <= ALWAYS_MDC_MAX_BYTES


def test_core_extract_markers_do_not_cut_normative_text() -> None:
    core = (ROOT / "agent-env" / "rules" / "intergrax-core.md").read_text(encoding="utf-8")
    assert core.count("<!-- igx:active:start -->") == core.count("<!-- igx:active:end -->") >= 1
    for phrase in CORE_EXTRACT_PHRASES:
        assert phrase in core


def test_operator_only_skills_are_flagged_in_both_adapters() -> None:
    flagged = sorted(p.name for p in (ROOT / "agent-env" / "skills").iterdir() if p.name in OPERATOR_ONLY_SKILLS)
    assert flagged == sorted(OPERATOR_ONLY_SKILLS)
    assert len(OPERATOR_ONLY_SKILLS) == 9
    for tool in (".cursor", ".claude"):
        skills = ROOT / tool / "skills"
        for skill in skills.iterdir():
            head = (skill / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
            has_flag = "disable-model-invocation: true" in head
            if skill.name in OPERATOR_ONLY_SKILLS:
                assert has_flag, f"{tool}/skills/{skill.name} lacks the flag"
            elif skill.name.startswith(("igx-devos-", "igx-autonomy", "igx-worktree")):
                assert not has_flag, f"{tool}/skills/{skill.name} must stay model-invocable"
    for tool in (".cursor", ".claude"):
        head = (ROOT / tool / "skills" / "igx-autonomy" / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
        assert "disable-model-invocation" not in head


def test_cursor_adapters_carry_no_claude_model_slugs() -> None:
    for path in (ROOT / ".cursor" / "agents").glob("*.md"):
        assert not re.search(r"^model:\s*(haiku|opus|sonnet)\b", path.read_text(encoding="utf-8"), re.MULTILINE)


def test_cursor_reviewer_uses_composer_without_claude_routing() -> None:
    cursor = (ROOT / ".cursor" / "agents" / "igx-reviewer.md").read_text(encoding="utf-8")
    claude = (ROOT / "agent-env" / "agents" / "igx-reviewer.md").read_text(encoding="utf-8")
    assert "model: composer-2.5" in cursor
    assert "reviewer uses `composer-2.5`" in cursor
    assert "runs this agent on `opus`" not in cursor
    assert "runs this agent on `opus`" in claude


def test_cursor_agents_use_documented_keys_and_derived_readonly() -> None:
    cursor_agents = ROOT / ".cursor" / "agents"
    names = sorted(p.stem for p in cursor_agents.glob("igx-*.md"))
    assert names == ["igx-architect", "igx-implementer", "igx-integrator", "igx-researcher", "igx-reviewer"]
    assert not (ROOT / ".claude" / "agents" / "igx-integrator.md").exists()
    for name in names:
        head = (cursor_agents / f"{name}.md").read_text(encoding="utf-8").split("---", 2)[1]
        keys = {line.split(":", 1)[0] for line in head.strip().splitlines()}
        assert keys == {"name", "description", "model", "readonly"}
    for name, readonly in (("igx-reviewer", "true"), ("igx-researcher", "true"), ("igx-implementer", "false"), ("igx-integrator", "false")):
        assert f"readonly: {readonly}" in (cursor_agents / f"{name}.md").read_text(encoding="utf-8")


def test_retired_rules_have_no_cursor_adapter() -> None:
    assert not (ROOT / ".cursor" / "rules" / "intergrax-token-budget.mdc").exists()
    assert (ROOT / ".cursor" / "rules" / "intergrax-core.mdc").is_file()


def test_generator_rejects_long_always_rule_without_active_block(tmp_path: Path, monkeypatch) -> None:
    sys.path.insert(0, str(SYNC.parent))
    try:
        import sync_agent_env  # noqa: WPS433
    finally:
        sys.path.remove(str(SYNC.parent))
    (tmp_path / "long-rule.md").write_text(
        "---\ndescription: x\nloading: always\n---\n" + "\n".join(f"line {i}" for i in range(30)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sync_agent_env, "_RULES", tmp_path)
    try:
        sync_agent_env._cursor_rule("long-rule", {"description": "x", "loading": "always"})
    except ValueError as exc:
        assert "igx:active" in str(exc)
    else:
        raise AssertionError("expected ValueError for a long always-on rule without igx:active block")


CHECKER = ROOT / "scripts" / "ci" / "check_cursor_token_setup.py"


def test_cursor_token_setup_checker_passes() -> None:
    completed = subprocess.run([sys.executable, str(CHECKER)], cwd=ROOT, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_cursor_token_setup_under_test", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_fails_when_token_budget_is_always_on_again(tmp_path: Path, monkeypatch) -> None:
    checker = _load_checker()
    rules = tmp_path / "rules"
    rules.mkdir()
    for rule_id in (*checker.ALWAYS_RULES, "intergrax-token-budget"):
        (rules / f"{rule_id}.md").write_text("---\ndescription: x\nloading: always\n---\nbody\n", encoding="utf-8")
    monkeypatch.setattr(checker, "RULES_DIR", rules)
    monkeypatch.setattr(checker, "TOKEN_BUDGET_RULE", rules / "intergrax-token-budget.md")
    errors: list[str] = []
    checker._check_always_on_rules(errors)
    checker._check_retired_token_budget(errors)
    assert any("must be exactly" in e for e in errors)
    assert any("loading: retired" in e for e in errors)


def test_checker_fails_when_core_mdc_is_a_pointer(tmp_path: Path, monkeypatch) -> None:
    checker = _load_checker()
    cursor_rules = tmp_path / "cursor-rules"
    cursor_rules.mkdir()
    for rule_id in checker.ALWAYS_RULES:
        (cursor_rules / f"{rule_id}.mdc").write_text(
            f"---\ndescription: x\nalwaysApply: true\n---\nCanonical rule (SSOT): @agent-env/rules/{rule_id}.md\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(checker, "CURSOR_RULES", cursor_rules)
    errors: list[str] = []
    checker._check_always_on_rules(errors)
    assert any("@agent-env" in e for e in errors)
    assert any("lost required text" in e for e in errors)
