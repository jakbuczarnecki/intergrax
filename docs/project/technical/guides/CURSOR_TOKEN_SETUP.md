# Cursor — token-efficient setup (F2–F7)

**Goal:** minimize always-on and accidental bulk doc loads without losing audit quality.

---

## F2 — AGENTS.md stub

| File | Auto-loaded | ~tok |
|------|-------------|------|
| `AGENTS.md` (root stub) | Yes (Cursor) | ~350 |
| `.cursor/rules/intergrax-token-budget.mdc` | Yes (`alwaysApply: true`) | ~380 |
| `.cursor/rules/intergrax-hep-step.mdc` | On demand / `@` | ~120 |
| `.cursor/rules/intergrax-iteration.mdc` | On demand / `@` | ~350 |
| `docs/project/technical/guides/AGENT_INSTRUCTIONS.md` | No — `@` on demand | ~3,300 |

**Only `intergrax-token-budget.mdc` is always-on.** All other rules and full guides load on explicit `@` or operator request — see root `AGENTS.md` § Cursor rule loading.

**Do not delete** root `AGENTS.md` via Cursor Settings trash icon.

---

## CI hotfix mode

| Rule | When |
|------|------|
| [`.cursor/rules/intergrax-ci-hotfix.mdc`](../../../../.cursor/rules/intergrax-ci-hotfix.mdc) | Failing CI/test/checker — **new chat**; no `AGENT_INSTRUCTIONS.md`, no arch/plan |

Allowed verification: failing GitHub command, exact test file, or `scripts/ci/run_ci_smoke_pytest.py`.

---

## F3 — Session protocol

```text
ONE DOMAIN = ONE NEW CHAT
CI HOTFIX = NEW CHAT
```

No Background Agents / Task subagents for layer audits unless operator opts in.

Bootstraps encode `SESSION:` + `READ_BUDGET:` + `OUTPUT_BUDGET:` on lines 1–3.

HEP implement steps: [`docs/project/maintainers/bootstrap/hep_step.txt`](../../maintainers/bootstrap/hep_step.txt) — edit `STEP=` / `SCOPE=` per C13–C16. Bootstrap lines 1–3 require `@intergrax-hep-step.mdc` + mandatory preflight before read/edit. **Also `@`:** [`.cursor/rules/intergrax-hep-step.mdc`](../../../../.cursor/rules/intergrax-hep-step.mdc).

---

## I1 — Input token budget (always-on)

| Always-on | ~tok |
|-----------|------|
| I1 + mandatory preflight in `intergrax-token-budget.mdc` | ~180 |

**Mandatory preflight** (before any implementation step): state read scope, edit scope, tests. If scope exceeds operator-listed files → **STOP** and ask. Do not load `AGENT_INSTRUCTIONS.md`, `intergrax-iteration.mdc`, full hubs, `docs/audit_results/AUDIT_PROTOCOL.md`, or domain guides unless explicitly requested.

**Default:** `offset`/`limit` or `grep` before full file; max 2 plan files on docs-only steps; no parallel full hub reads; no subagents unless operator asks.

**Expand context** (full hub, subagent, >3 files): **STOP** — ask operator once; wait for OK.

Domain read scope (when load is allowed): arch/plan hub read-scope blocks + one satellite — see `intergrax-iteration.mdc` (on demand only).

---

## O1 — Terse operator replies (output token budget)

| Always-on | ~tok |
|-----------|------|
| O1 block in `intergrax-token-budget.mdc` | ~100 |
| Full 12-point template | `AGENT_INSTRUCTIONS.md` § Operator communication — **not** auto-loaded |

**Default:** terse (≤12 lines) — outcome, paths, tests, blockers. **No** diff recap or doc restatement.

**Expand only when:** operator says `pełny raport` / `full report` / `iteration summary`, or milestone / closeout / journal.

**Minimal:** `krótko` / `terse` → ≤6 lines.

---

## HEP step execution (on demand)

| Rule | When |
|------|------|
| [`.cursor/rules/intergrax-hep-step.mdc`](../../../../.cursor/rules/intergrax-hep-step.mdc) | Operator asks for HEP / EVID implementation step (C13+) |

Operator instruction + listed files = source of truth. No repo search, no full docs, no subagents, no full test suite unless listed. Token usage in final report: only if Cursor provides it — no guesses.

---

## F4 — Architecture hub + `docs/project/architecture/satellites` satellites

Split domains: all 22 architecture hubs (F4-C wave 2 complete).

Satellites are in `.cursorignore` — load with explicit `Read` or `@` when read-scope or audit cites extended §.

```bash
uv run python scripts/docs/split_domain_architecture.py [DOMAIN ...]
uv run python scripts/maintenance/check_arch_hub_size.py
uv run python scripts/docs/verify_arch_split_content.py
uv run python scripts/docs/generate_architecture_read_scopes.py
```

---

## G1 — Plan hub splits

Split domains: all token-heavy plan hubs (G1-D wave 2 complete).

Satellites (`docs/project/maintainers/plans/satellites`) are in `.cursorignore` — same explicit-load rule as F4.

```bash
uv run python scripts/docs/split_domain_plan.py [DOMAIN ...]
uv run python scripts/maintenance/check_plan_hub_size.py
uv run python scripts/docs/generate_plan_read_scopes.py
```

Plan hubs include **Cursor read scope (token budget)** blocks (~150 tok) — read §6 / open queues only; same explicit-load rule as F4 arch read-scopes.

---

## H2 — Bulky docs in `.cursorignore`

Explicit `@` / `Read` only (reduces accidental index/search noise):

**Platform audit (canonical):**

- `docs/audit_results/AUDIT_PROTOCOL.md` — load only when operator requests a platform audit
- `docs/audit_results/AUDIT_REMEDIATION_PROTOCOL.md` — load only when remediating accepted findings
- `docs/audit_results/` campaign directories — load only when operator cites a campaign path

**Satellite directories (F4 / G1 / multi-layer features):**

- `docs/project/architecture/satellites`
- `docs/project/maintainers/plans/satellites`
- `docs/project/capabilities/architecture/satellites`
- `docs/project/capabilities/plan/satellites`

**Bulky guides:**

- `docs/project/technical/guides/AGENT_CREATION_GUIDE.md`
- `docs/project/technical/guides/APPLICATION_CREATION_GUIDE.md`
- `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`
- `docs/audit_results/AUDIT_PROTOCOL.md`
- `docs/audit_results/AUDIT_REMEDIATION_PROTOCOL.md`
- `docs/project/technical/guides/SYSTEM_INVARIANTS.md` — grep `SYS-INV-*` IDs when cited; `@` full file only when cited
- `docs/project/technical/guides/MATURITY_TAXONOMY.md`
- `docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`
- `docs/project/architecture/intergrax_runtime_architecture.md` — domain pair index; use bootstrap domain list or `@` when needed
- `docs/project/maintainers/plans/AUDIT_IDEAL_2026.md`

---

## H3 — Platform audit protocol

Follow [`docs/audit_results/AUDIT_PROTOCOL.md`](../../../audit_results/AUDIT_PROTOCOL.md) for adversarial layer audits. Persist results under `docs/audit_results/YYYY-MM-DD/` per campaign model in [`docs/audit_results/README.md`](../../../audit_results/README.md).

---

## E2 — Architecture read scopes (all domains)

```bash
uv run python scripts/docs/generate_architecture_read_scopes.py
```

---

## G1-E2 — Plan read scopes (all domains)

```bash
uv run python scripts/docs/generate_plan_read_scopes.py
```

Plan hub read-scope blocks mirror architecture E2 — §6 / open P0/P1 queues only; at most one `plan/satellites` satellite per session.

---

## F5-B — Symbol index

[`docs/project/technical/guides/SYMBOL_INDEX.md`](SYMBOL_INDEX.md) — grep before repo-wide semantic search.

```bash
uv run python scripts/docs/generate_symbol_index.py
```

---

## Token setup CI

```bash
uv run python scripts/ci/check_cursor_token_setup.py
```

`check_cursor_token_setup.py` validates AGENTS stub split, bootstrap read budgets, plan read-scope blocks, and rejects broad Cursor access wording in control prompts.
