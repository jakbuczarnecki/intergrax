# Cursor — token-efficient setup (F2 / F3)

**Goal:** avoid duplicating ~3k+ tokens every turn and prevent multi-domain context explosion.

---

## F2 — Project Rules (operator setup, one-time)

In **Cursor → Settings → Rules** (Project):

| Rule | alwaysApply | Action |
|------|-------------|--------|
| `.cursor/rules/intergrax-iteration.mdc` | **Yes** | Keep — ~950 tokens/tur |
| `AGENTS.md` as Project Rule | **No — remove** | Duplicate of iteration + on-demand reference |
| `.cursor/rules/intergrax-idea-audit.mdc` | **No** | Loads on trigger only |
| `.cursor/rules/intergrax-agents-reference.mdc` | **No** | Pointer; use `@AGENTS.md` when needed |

**Why:** `AGENTS.md` (~3.2k tokens) + `intergrax-iteration.mdc` (~1k) stacked = ~4k+ **per turn**. Removing `AGENTS.md` from always-on saves **~2.4k × number of turns** per session.

**When agent needs full routing/verification:** `@AGENTS.md` in chat.

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

Regenerate after plan changes: `uv run python scripts/split_domain_plan.py [DOMAIN]`
