---
name: intergrax-ci-hotfix
description: Explicit operator mode — ultra-scoped CI, test, or checker hotfix with no docs, no architecture hubs, and no full test suite. Activate only when requested.
---

# intergrax-ci-hotfix (Antigravity skill adapter)

Read and follow the canonical rule (SSOT): `agent-env/rules/intergrax-ci-hotfix.md`.

Key reminders:
- Start in a fresh chat (`CI HOTFIX = NEW CHAT`).
- State `read scope`, `edit scope`, and `tests` before reading or editing.
- Do NOT load `AGENT_INSTRUCTIONS.md`, architecture/plan hubs, or strategy docs unless explicitly authorized.
- Run only the exact failing test/checker command (or `uv run python scripts/ci/run_ci_smoke_pytest.py`) and return a <= 6-line final report.
