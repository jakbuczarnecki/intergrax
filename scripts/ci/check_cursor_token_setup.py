# © Artur Czarnecki. All rights reserved.
"""Verify AGENTS stub split (F2), bootstrap F3/READ_BUDGET, and CURSOR_TOKEN_SETUP."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "docs" / "project" / "maintainers" / "bootstrap"
CURSOR_SETUP = ROOT / "docs" / "project" / "technical" / "guides" / "CURSOR_TOKEN_SETUP.md"
CURSORIGNORE = ROOT / ".cursorignore"
H2_IGNORE_DIRS = (
    "docs/project/maintainers/audit/",
    "docs/project/architecture/satellites/",
    "docs/project/maintainers/plans/satellites/",
)
H2_IGNORE_PATHS = (
    "docs/project/technical/guides/AGENT_CREATION_GUIDE.md",
    "docs/project/technical/guides/APPLICATION_CREATION_GUIDE.md",
    "docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md",
    "docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md",
    "docs/project/technical/guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md",
    "docs/project/technical/guides/SYSTEM_INVARIANTS.md",
    "docs/project/technical/guides/MATURITY_TAXONOMY.md",
    "docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md",
    "docs/project/architecture/intergrax_runtime_architecture.md",
    "docs/project/maintainers/plans/AUDIT_IDEAL_2026.md",
)
AGENT_INSTRUCTIONS = ROOT / "docs" / "project" / "technical" / "guides" / "AGENT_INSTRUCTIONS.md"
SYMBOL_INDEX = ROOT / "docs" / "project" / "technical" / "guides" / "SYMBOL_INDEX.md"
AGENTS_STUB = ROOT / "AGENTS.md"
ITERATION_RULE = ROOT / ".cursor" / "rules" / "intergrax-iteration.mdc"
TOKEN_BUDGET_RULE = ROOT / ".cursor" / "rules" / "intergrax-token-budget.mdc"
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
CI_HOTFIX_RULE = ROOT / ".cursor" / "rules" / "intergrax-ci-hotfix.mdc"
FULL_MIN_LINES = 150
AUDIT_BOOTSTRAPS = (
    "01_audit_all_domains.txt",
    "02_audit_one_domain.txt",
    "03_implement_plan_all_domains.txt",
    "04_implement_plan_one_domain.txt",
    "05_closeout_all_domains.txt",
    "06_interactive_layer_by_layer_audit.txt",
)
HEP_BOOTSTRAP = "hep_step.txt"
FORBIDDEN_BROAD_ACCESS_PHRASES = (
    "full repository access",
)
BROAD_ACCESS_GUARD_PATHS = (
    ROOT / "scripts" / "audit",
    ROOT / "docs" / "project" / "maintainers" / "audit",
    ROOT / ".cursor",
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
                    errors.append(
                        f"{candidate.relative_to(ROOT)}: forbidden broad Cursor access phrase: {phrase!r}"
                    )


def main() -> int:
    errors: list[str] = []

    if not CURSOR_SETUP.is_file():
        errors.append("missing docs/project/technical/guides/CURSOR_TOKEN_SETUP.md")
    elif "O1" not in CURSOR_SETUP.read_text(encoding="utf-8"):
        errors.append("CURSOR_TOKEN_SETUP.md must document O1 terse output policy")
    elif I1_MARKER not in CURSOR_SETUP.read_text(encoding="utf-8"):
        errors.append("CURSOR_TOKEN_SETUP.md must document I1 input token budget")

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

    if not TOKEN_BUDGET_RULE.is_file():
        errors.append("missing .cursor/rules/intergrax-token-budget.mdc (I1/O1 always-on)")
    else:
        token_budget = TOKEN_BUDGET_RULE.read_text(encoding="utf-8")
        if "alwaysApply: true" not in token_budget:
            errors.append("intergrax-token-budget.mdc must have alwaysApply: true")
        if I1_MARKER not in token_budget:
            errors.append("intergrax-token-budget.mdc must include I1 input token budget")
        if O1_MARKER not in token_budget or "terse" not in token_budget.lower():
            errors.append("intergrax-token-budget.mdc must include O1 terse output policy")
        if "intergrax-ci-hotfix" not in token_budget:
            errors.append("intergrax-token-budget.mdc must reference intergrax-ci-hotfix.mdc")

    if not CI_HOTFIX_RULE.is_file():
        errors.append("missing .cursor/rules/intergrax-ci-hotfix.mdc (CI hotfix mode)")
    else:
        ci_hotfix = CI_HOTFIX_RULE.read_text(encoding="utf-8")
        if "alwaysApply: false" not in ci_hotfix:
            errors.append("intergrax-ci-hotfix.mdc must have alwaysApply: false")
        if "AGENT_INSTRUCTIONS.md" not in ci_hotfix:
            errors.append("intergrax-ci-hotfix.mdc must forbid AGENT_INSTRUCTIONS.md for hotfixes")

    iteration = ITERATION_RULE.read_text(encoding="utf-8")
    if "AGENT_INSTRUCTIONS.md" not in iteration:
        errors.append("intergrax-iteration.mdc must reference AGENT_INSTRUCTIONS.md (F2 stub split)")
    if SESSION_MARKER not in iteration and "ONE DOMAIN = ONE NEW CHAT" not in iteration:
        errors.append("intergrax-iteration.mdc must include F3 session protocol")
    if "SYMBOL_INDEX" not in iteration:
        errors.append("intergrax-iteration.mdc must reference SYMBOL_INDEX.md (F5)")
    if "intergrax-token-budget" not in iteration:
        errors.append("intergrax-iteration.mdc must reference intergrax-token-budget.mdc (I1/O1)")

    plan_gen = ROOT / "scripts" / "audit" / "generate_plan_read_scopes.py"
    if not plan_gen.is_file():
        errors.append("missing scripts/audit/generate_plan_read_scopes.py (G1-E2)")
    else:
        for path in sorted(PLAN_DIR.glob("*.md")):
            if path.name in SKIP_PLAN_HUBS:
                continue
            if PLAN_READ_SCOPE_MARKER not in path.read_text(encoding="utf-8"):
                errors.append(f"{path.relative_to(ROOT)}: missing plan read-scope block (G1-E2)")

    agents_ref = ROOT / ".cursor" / "rules" / "intergrax-agents-reference.mdc"
    if agents_ref.is_file():
        errors.append("remove redundant .cursor/rules/intergrax-agents-reference.mdc (stub replaces it)")

    for name in AUDIT_BOOTSTRAPS:
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