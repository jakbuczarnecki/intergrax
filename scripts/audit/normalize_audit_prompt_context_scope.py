# © Artur Czarnecki. All rights reserved.
"""Normalize audit prompt startup wording for Cursor token discipline.

This script is intentionally narrow:
- removes broad repository access wording from audit control prompts,
- keeps audit capability intact,
- preserves explicit repository availability while forbidding broad exploration.

Run after regenerating docs/project/maintainers/audit/*.md if the generator reintroduces broad-scope wording.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REPLACEMENTS = {
    "Open a new agent chat with **full repository access**.": (
        "Open a new agent chat with repository access available, but do not perform "
        "broad repository exploration. Read only the files listed in Context budget / "
        "Canonical reads."
    ),
    "Open a new Cursor / agent chat with full repository access.": (
        "Open a new Cursor / agent chat with repository access available, but do not perform "
        "broad repository exploration. Read only the files listed in Context budget / "
        "Canonical reads."
    ),
}

TARGETS = [
    ROOT / "scripts" / "audit" / "generate_domain_audit_prompts.py",
    *sorted((ROOT / "docs" / "project" / "maintainers" / "audit").glob("*.md")),
]


def normalize_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in REPLACEMENTS.items():
        updated = updated.replace(old, new)

    if updated == text:
        return False

    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    changed: list[Path] = []
    for path in TARGETS:
        if path.is_file() and normalize_file(path):
            changed.append(path)

    for path in changed:
        print(f"normalized {path.relative_to(ROOT)}")

    print(f"normalize_audit_prompt_context_scope: {len(changed)} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
