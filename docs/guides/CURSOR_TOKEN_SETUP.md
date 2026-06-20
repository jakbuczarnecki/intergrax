# Cursor — token-efficient setup (F2 / F3)

**Goal:** avoid duplicating ~3k+ tokens every turn and prevent multi-domain context explosion.

---

## F2 — AGENTS.md stub (repo + operator)

Cursor **auto-discovers** root `AGENTS.md` and loads it every session. You **cannot** disable it in Settings without **deleting the file** from the project.

**Solution (Option A — configured in repo):**

| File | Role | Auto-loaded by Cursor |
|------|------|------------------------|
| [`AGENTS.md`](../../AGENTS.md) (root) | **Stub** (~350 tokens) — tiers, boundaries, pointers | **Yes** (unavoidable) |
| [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) | **Full reference** — routing, verification, ADR, anti-patterns | **No** — `@docs/guides/AGENT_INSTRUCTIONS.md` on demand |
| [`.cursor/rules/intergrax-iteration.mdc`](../../.cursor/rules/intergrax-iteration.mdc) | Workflow + F3 + context budget | **Yes** (`alwaysApply: true`) |

**Project Rules (Settings → Rules):**

| Rule | Action |
|------|--------|
| `intergrax-iteration.mdc` | Keep — always on (~950 tok/tur) |
| Root **AGENTS** entry | **Keep the file** — do not delete via trash icon |
| `intergrax-agents-reference.mdc` | Optional — applied intelligently; pointer only |
| `intergrax-idea-audit.mdc` | On trigger only |

**Savings vs monolithic AGENTS.md:** ~350 + ~950 ≈ **1.3k/tur** instead of ~3.2k + ~950 ≈ **4.2k/tur** (~**2.9k × turns**).

**When agent needs full routing/verification:** `@docs/guides/AGENT_INSTRUCTIONS.md`

---

## F3 — Session protocol (mandatory)

```text
ONE DOMAIN = ONE NEW CHAT
```

| Activity | Session rule |
|----------|----------------|
| Audit one layer | New chat + `02_audit_one_domain.txt` or `06_interactive…` |
| Implement one plan item | New chat per domain (or explicit operator batch) |
| Next audit domain | **New chat** — never continue 22 domains in one thread |
| Resume audit run | New chat + bootstrap + `RESUME: docs/audit_results/…` |

Multi-domain in one chat multiplies history cost linearly → 1M–5M tokens.

Bootstrap files encode this in the first lines (`SESSION: ONE_DOMAIN_ONE_CHAT`).

---

## Repo artifacts (already configured)

- `.cursorignore` — excludes noise from indexing
- Plan hubs + `docs/plan/plan/` satellites
- `docs/guides/audit_slices/<DOMAIN>.md` — compact audit context
- Architecture **Cursor read scope** blocks
- `scripts/check_plan_hub_size.py` — CI gate
- `scripts/check_cursor_token_setup.py` — stub + F3 gate

Regenerate after plan changes: `uv run python scripts/split_domain_plan.py [DOMAIN]`
