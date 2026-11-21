# © Artur Czarnecki. All rights reserved.
"""Verify AGENTS stub split (F2), bootstrap F3/READ_BUDGET, and CURSOR_TOKEN_SETUP."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "docs" / "project" / "maintainers" / "bootstrap"
CURSOR_SETUP = ROOT / "docs" / "project" / "technical" / "guides" / "CURSOR_TOKEN_SETUP.md"
CURSORIGNORE = ROOT / ".cursorignore"
CURSORINDEXINGIGNORE = ROOT / ".cursorindexingignore"
H2_IGNORE_DIRS = (
    "docs/project/architecture/satellites/",
    "docs/project/maintainers/plans/satellites/",
)
H2_INDEXING_IGNORE_DIRS = (
    "docs/audit_results/legacy/",
    "docs/audit_results/20??-??-??/",
    "docs/audit_results/20??-??-??_run-*/",
)
H2_IGNORE_PATHS = (
    "docs/project/technical/guides/AGENT_CREATION_GUIDE.md",
    "docs/project/technical/guides/APPLICATION_CREATION_GUIDE.md",
    "docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md",
    "docs/project/technical/guides/SYSTEM_INVARIANTS.md",
    "docs/project/technical/guides/MATURITY_TAXONOMY.md",
    "docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md",
    "docs/project/architecture/intergrax_runtime_architecture.md",
    "docs/project/maintainers/plans/AUDIT_IDEAL_2026.md",
)
AGENT_INSTRUCTIONS = ROOT / "docs" / "project" / "technical" / "guides" / "AGENT_INSTRUCTIONS.md"
SYMBOL_INDEX = ROOT / "docs" / "project" / "technical" / "guides" / "SYMBOL_INDEX.md"
AGENTS_STUB = ROOT / "AGENTS.md"
ITERATION_RULE = ROOT / "agent-env" / "rules" / "intergrax-iteration.md"
TOKEN_BUDGET_RULE = ROOT / "agent-env" / "rules" / "intergrax-token-budget.md"
CURSOR_RULES = ROOT / ".cursor" / "rules"
RULES_DIR = ROOT / "agent-env" / "rules"
ALWAYS_RULES = ("intergrax-core", "intergrax-english-only-commits")
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
SETUP_REQUIRED_PHRASES = ("intergrax-core", "ESCALATION:")
SETUP_STALE_PHRASES = ("Only `intergrax-token-budget.md` is always-on", "STOP and ask")
PLAN_READ_SCOPE_MARKER = "## Cursor read scope (token budget)"
PLAN_DIR = ROOT / "docs" / "project" / "maintainers" / "plans"
SKIP_PLAN_HUBS = frozenset({"AUDIT_IDEAL_2026.md", "IDEAL_HARNESS_L3.md"})

SESSION_MARKER = "ONE_DOMAIN_ONE_CHAT"
READ_BUDGET_MARKER = "READ_BUDGET"
OUTPUT_BUDGET_MARKER = "OUTPUT_BUDGET"
O1_MARKER = "O1"
I1_MARKER = "I1"
STUB_MARKER = "AGENT_INSTRUCTIONS.md"
STUB_MAX_LINES = 45
CI_HOTFIX_RULE = ROOT / "agent-env" / "rules" / "intergrax-ci-hotfix.md"
FULL_MIN_LINES = 150
REQUIRED_BOOTSTRAPS: tuple[str, ...] = ()
HEP_BOOTSTRAP = "hep_step.txt"
FORBIDDEN_BROAD_ACCESS_PHRASES = ("full repository access",)
BROAD_ACCESS_GUARD_PATHS = (
    ROOT / ".cursor",
    ROOT / ".claude" / "skills",
    ROOT / "agent-env",
    ROOT / "docs" / "project" / "maintainers" / "bootstrap",
    ROOT / "docs" / "project" / "technical" / "guides" / "CURSOR_TOKEN_SETUP.md",
)
BROAD_ACCESS_GUARD_SUFFIXES = (".md", ".mdc", ".txt", ".py")


def _broad_access_guard_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    candidates: list[Path] = []
    for suffix in BROAD_ACCESS_GUARD_SUFFIXES:
        candidates.extend(path.rglob(f"*{suffix}"))
    return sorted(candidates)


def _check_broad_access_phrases(errors: list[str]) -> None:
    for guard_path in BROAD_ACCESS_GUARD_PATHS:
        for candidate in _broad_access_guard_candidates(guard_path):
            text = candidate.read_text(encoding="utf-8")
            for phrase in FORBIDDEN_BROAD_ACCESS_PHRASES:
                if phrase in text:
                    errors.append(f"{candidate.relative_to(ROOT)}: forbidden broad Cursor access phrase: {phrase!r}")


def _frontmatter_value(text: str, key: str) -> str:
    head = text.replace("\r\n", "\n").split("---\n", 2)
    if len(head) < 3 or head[0] != "":
        return ""
    for line in head[1].splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return ""


def _mdc_body(text: str) -> str:
    parts = text.replace("\r\n", "\n").split("---\n", 2)
    return parts[2] if len(parts) == 3 else ""


def _check_always_on_rules(errors: list[str]) -> None:
    """Always-on set is exactly core + English-only; each ``.mdc`` carries the extract, not a pointer."""
    always = sorted(
        path.stem
        for path in RULES_DIR.glob("*.md")
        if _frontmatter_value(path.read_text("utf-8"), "loading") == "always"
    )
    if always != sorted(ALWAYS_RULES):
        errors.append(f"agent-env rules with loading: always must be exactly {sorted(ALWAYS_RULES)}, found {always}")
    for rule_id in ALWAYS_RULES:
        adapter = CURSOR_RULES / f"{rule_id}.mdc"
        if not adapter.is_file():
            errors.append(f".cursor/rules/{rule_id}.mdc missing (run sync_agent_env.py)")
            continue
        text = adapter.read_text(encoding="utf-8")
        if _frontmatter_value(text, "alwaysApply") != "true":
            errors.append(f".cursor/rules/{rule_id}.mdc must have alwaysApply: true")
        if "@agent-env" in _mdc_body(text):
            errors.append(
                f".cursor/rules/{rule_id}.mdc must inline the rule text, not an unexpanded @agent-env pointer"
            )
    core = CURSOR_RULES / "intergrax-core.mdc"
    if core.is_file():
        body = _mdc_body(core.read_text(encoding="utf-8"))
        for phrase in CORE_EXTRACT_PHRASES:
            if phrase not in body:
                errors.append(f".cursor/rules/intergrax-core.mdc extract lost required text {phrase!r}")
    english = CURSOR_RULES / "intergrax-english-only-commits.mdc"
    if english.is_file() and "English only" not in _mdc_body(english.read_text(encoding="utf-8")):
        errors.append(".cursor/rules/intergrax-english-only-commits.mdc must contain 'English only'")
    total = sum(
        len(_mdc_body((CURSOR_RULES / f"{rid}.mdc").read_text(encoding="utf-8")).encode("utf-8"))
        for rid in ALWAYS_RULES
        if (CURSOR_RULES / f"{rid}.mdc").is_file()
    )
    if total > ALWAYS_MDC_MAX_BYTES:
        errors.append(f"always-on .mdc bodies are {total} bytes, max {ALWAYS_MDC_MAX_BYTES}")


def _check_retired_token_budget(errors: list[str]) -> None:
    if not TOKEN_BUDGET_RULE.is_file():
        errors.append("missing agent-env/rules/intergrax-token-budget.md (retired tombstone)")
    elif _frontmatter_value(TOKEN_BUDGET_RULE.read_text("utf-8"), "loading") != "retired":
        errors.append("intergrax-token-budget.md must have loading: retired (content lives in intergrax-core.md)")
    if (CURSOR_RULES / "intergrax-token-budget.mdc").exists():
        errors.append(".cursor/rules/intergrax-token-budget.mdc must not exist (retired rule has no adapter)")


def main() -> int:
    errors: list[str] = []

    if not CURSOR_SETUP.is_file():
        errors.append("missing docs/project/technical/guides/CURSOR_TOKEN_SETUP.md")
    else:
        setup = CURSOR_SETUP.read_text(encoding="utf-8")
        if "O1" not in setup:
            errors.append("CURSOR_TOKEN_SETUP.md must document O1 terse output policy")
        if I1_MARKER not in setup:
            errors.append("CURSOR_TOKEN_SETUP.md must document I1 input token budget")
        for required in SETUP_REQUIRED_PHRASES:
            if required not in setup:
                errors.append(f"CURSOR_TOKEN_SETUP.md must mention {required!r}")
        for stale in SETUP_STALE_PHRASES:
            if stale in setup:
                errors.append(f"CURSOR_TOKEN_SETUP.md still carries the retired wording {stale!r}")

    if not CURSORIGNORE.is_file():
        errors.append("missing .cursorignore")
    else:
        ignore = CURSORIGNORE.read_text(encoding="utf-8")
        for path in H2_IGNORE_DIRS:
            if path not in ignore:
                errors.append(f".cursorignore must exclude token-heavy dir {path} (H2)")
        for path in H2_IGNORE_PATHS:
            if path not in ignore:
                errors.append(f".cursorignore must exclude bulky guide {path} (H2)")
        if "docs/audit_results/" in ignore:
            errors.append(
                ".cursorignore must not broadly exclude docs/audit_results/ "
                "(use .cursorindexingignore for bulky campaign artifacts)"
            )

    if not CURSORINDEXINGIGNORE.is_file():
        errors.append("missing .cursorindexingignore (audit campaign indexing exclusions)")
    else:
        indexing_ignore = CURSORINDEXINGIGNORE.read_text(encoding="utf-8")
        for path in H2_INDEXING_IGNORE_DIRS:
            if path not in indexing_ignore:
                errors.append(f".cursorindexingignore must exclude token-heavy audit dir {path} (H2 indexing)")

    if not SYMBOL_INDEX.is_file():
        errors.append("missing docs/project/technical/guides/SYMBOL_INDEX.md")

    if not AGENT_INSTRUCTIONS.is_file():
        errors.append("missing docs/project/technical/guides/AGENT_INSTRUCTIONS.md")
    elif len(AGENT_INSTRUCTIONS.read_text(encoding="utf-8").splitlines()) < FULL_MIN_LINES:
        errors.append(f"AGENT_INSTRUCTIONS.md must be full reference (>={FULL_MIN_LINES} lines)")
    elif "Operator communication (O1" not in AGENT_INSTRUCTIONS.read_text(encoding="utf-8"):
        errors.append("AGENT_INSTRUCTIONS.md must include Operator communication (O1) section")

    if not AGENTS_STUB.is_file():
        errors.append("missing root AGENTS.md stub")
    else:
        stub = AGENTS_STUB.read_text(encoding="utf-8")
        stub_lines = len(stub.splitlines())
        if stub_lines > STUB_MAX_LINES:
            errors.append(f"AGENTS.md stub too large ({stub_lines} lines; max {STUB_MAX_LINES})")
        if STUB_MARKER not in stub:
            errors.append("AGENTS.md stub must link to AGENT_INSTRUCTIONS.md")
        if "O1" not in stub and "terse" not in stub.lower() and I1_MARKER not in stub:
            errors.append("AGENTS.md stub must mention I1/O1 token budget policy")
        if "## Task routing" in stub or "## Verification" in stub:
            errors.append("AGENTS.md stub must not contain full routing/verification sections")

    _check_always_on_rules(errors)
    _check_retired_token_budget(errors)

    if not CI_HOTFIX_RULE.is_file():
        errors.append("missing agent-env/rules/intergrax-ci-hotfix.md (CI hotfix mode)")
    else:
        ci_hotfix = CI_HOTFIX_RULE.read_text(encoding="utf-8")
        if "loading: explicit" not in ci_hotfix:
            errors.append("intergrax-ci-hotfix.md must have loading: explicit")
        if "AGENT_INSTRUCTIONS.md" not in ci_hotfix:
            errors.append("intergrax-ci-hotfix.md must forbid AGENT_INSTRUCTIONS.md for hotfixes")

    iteration = ITERATION_RULE.read_text(encoding="utf-8")
    if "AGENT_INSTRUCTIONS.md" not in iteration:
        errors.append("intergrax-iteration.md must reference AGENT_INSTRUCTIONS.md (F2 stub split)")
    if SESSION_MARKER not in iteration and "ONE DOMAIN = ONE NEW CHAT" not in iteration:
        errors.append("intergrax-iteration.md must include F3 session protocol")
    if "SYMBOL_INDEX" not in iteration:
        errors.append("intergrax-iteration.md must reference SYMBOL_INDEX.md (F5)")
    if "intergrax-core" not in iteration:
        errors.append("intergrax-iteration.md must point to intergrax-core.md (I1/O1 live in core, section Token use)")
    if "intergrax-token-budget" in iteration:
        errors.append("intergrax-iteration.md must not point to the retired intergrax-token-budget.md")

    plan_gen = ROOT / "scripts" / "docs" / "generate_plan_read_scopes.py"
    if not plan_gen.is_file():
        errors.append("missing scripts/docs/generate_plan_read_scopes.py (G1-E2)")
    else:
        sys.path.insert(0, str(ROOT / "scripts" / "docs"))
        from docs_domain_common import canonical_domain_ids  # noqa: WPS433

        canonical_plans = {f"{name}.md" for name in canonical_domain_ids()}
        for path in sorted(PLAN_DIR.glob("*.md")):
            if path.name in SKIP_PLAN_HUBS or path.name not in canonical_plans:
                continue
            if PLAN_READ_SCOPE_MARKER not in path.read_text(encoding="utf-8"):
                errors.append(f"{path.relative_to(ROOT)}: missing plan read-scope block (G1-E2)")

    agents_ref = ROOT / ".cursor" / "rules" / "intergrax-agents-reference.mdc"
    if agents_ref.is_file():
        errors.append("remove redundant .cursor/rules/intergrax-agents-reference.mdc (stub replaces it)")

    for name in REQUIRED_BOOTSTRAPS:
        path = BOOTSTRAP / name
        if not path.is_file():
            errors.append(f"missing bootstrap {name}")
            continue
        text = path.read_text(encoding="utf-8")
        if SESSION_MARKER not in text:
            errors.append(f"{name}: missing {SESSION_MARKER}")
        if READ_BUDGET_MARKER not in text:
            errors.append(f"{name}: missing {READ_BUDGET_MARKER}")
        if OUTPUT_BUDGET_MARKER not in text:
            errors.append(f"{name}: missing {OUTPUT_BUDGET_MARKER}")

    hep_path = BOOTSTRAP / HEP_BOOTSTRAP
    if not hep_path.is_file():
        errors.append(f"missing bootstrap {HEP_BOOTSTRAP}")
    else:
        hep_text = hep_path.read_text(encoding="utf-8")
        if SESSION_MARKER not in hep_text:
            errors.append(f"{HEP_BOOTSTRAP}: missing {SESSION_MARKER}")
        if READ_BUDGET_MARKER not in hep_text:
            errors.append(f"{HEP_BOOTSTRAP}: missing {READ_BUDGET_MARKER}")
        if OUTPUT_BUDGET_MARKER not in hep_text:
            errors.append(f"{HEP_BOOTSTRAP}: missing {OUTPUT_BUDGET_MARKER}")
        if "STEP=" not in hep_text or "SCOPE=" not in hep_text:
            errors.append(f"{HEP_BOOTSTRAP}: must include STEP= and SCOPE= placeholders")

    _check_broad_access_phrases(errors)

    if errors:
        print("check_cursor_token_setup: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("check_cursor_token_setup: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
