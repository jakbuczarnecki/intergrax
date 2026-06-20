# Cursor — token-efficient setup (F2–F7)

**Goal:** minimize always-on and accidental bulk doc loads without losing audit quality.

---

## F2 — AGENTS.md stub

| File | Auto-loaded | ~tok |
|------|-------------|------|
| `AGENTS.md` (root stub) | Yes (Cursor) | ~350 |
| `docs/guides/AGENT_INSTRUCTIONS.md` | No — `@` on demand | ~3,300 |
| `.cursor/rules/intergrax-iteration.mdc` | Yes | ~500 |

**Do not delete** root `AGENTS.md` via Cursor Settings trash icon.

---

## F3 — Session protocol

```text
ONE DOMAIN = ONE NEW CHAT
```

No Background Agents / Task subagents for layer audits unless operator opts in.

Bootstraps encode `SESSION:` + `READ_BUDGET:` on line 1–2.

---

## F4 — Architecture hub + `docs/architecture/arch/` satellites

Split domains: ACP, TIER3, PLATFORM, TOOLS.

```bash
uv run python scripts/split_domain_architecture.py [DOMAIN ...]
uv run python scripts/check_arch_hub_size.py
```

---

## G1 — Plan hub second pass

Split domains: PLATFORM, UAEP, EXP_DX (+ prior splits).

```bash
uv run python scripts/split_domain_plan.py [DOMAIN ...]
uv run python scripts/check_plan_hub_size.py
```

---

## E2 — Architecture read scopes (all domains)

```bash
uv run python scripts/generate_architecture_read_scopes.py
```

---

## H1 — Audit token discipline CI

```bash
uv run python scripts/check_audit_token_discipline.py
```

Domain audit prompts must reference `audit_slices` + `Context budget`; bootstraps must include `READ_BUDGET`.

---

## F5-B — Symbol index + CODE_ENTRY in audit slices

[`docs/guides/SYMBOL_INDEX.md`](SYMBOL_INDEX.md) — grep before repo-wide semantic search.

```bash
uv run python scripts/generate_symbol_index.py
```

---

## F7 — Generator freshness CI

```bash
uv run python scripts/check_token_generator_freshness.py
```

Regenerates audit slices, arch read scopes, audit prompts; fails if outputs drift.

---

## Audit compact context

[`audit_slices/<DOMAIN>.md`](audit_slices/) (~300 tok) instead of full IDEAL + AUDIT_MAP.

```bash
uv run python scripts/generate_audit_read_slices.py
uv run python scripts/check_cursor_token_setup.py
```
