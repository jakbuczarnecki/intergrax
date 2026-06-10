# Implementation Journal

**Purpose:** Chronological, narrative record of **completed implementations** across the full Intergrax stack — Tier-0 platform, Tier-1 Nexus, Tier-2 agents, and Tier-3 applications.

**Role in documentation:**

| Layer | Source of truth | Journal adds |
|-------|-----------------|--------------|
| Architecture canon | `docs/architecture/<DOMAIN>.md`, agent/app `ARCHITECTURE.md` | Operator intent, episode context |
| Implementation status | `docs/plan/<DOMAIN>.md`, agent `IMPLEMENTATION_PLAN.md` | Cross-domain timeline, impact narrative |
| Significant decisions | `docs/adr/`, agent/app `adr/` | Link only — not a second ADR store |
| Technical proof | Git commits, tests | Verification commands and outcome |

The journal is **operator tooling** (like [`audit/README.md`](../audit/README.md)). It does **not** replace domain pairs or plan rows.

---

## When to write an entry

Write an entry when an iteration **closes** a plan row, delivers a new mechanism, or ships agent/application capability — i.e. when the iteration summary in [`.cursor/rules/intergrax-iteration.mdc`](../../../.cursor/rules/intergrax-iteration.mdc) would mark work **complete**.

| Write | Skip |
|-------|------|
| Plan phase marked **Done** | Brainstorming or design-only chat |
| New Tier-0/Tier-1 wiring or contract | Typo, formatting, comment-only |
| Tier-2 agent or Tier-3 host feature | Exploratory spike with no merged artifact |
| Audit remediation with code or canon update | `audit-only` report with no delivery |
| ADR-level change (link the ADR) | Operator says **no journal** |

**One entry per coherent iteration.** Multiple plan IDs in one PR → one entry listing all refs.

---

## Scope routing

| Work location | `tier` | Primary traceability |
|---------------|--------|----------------------|
| `intergrax/` (non-runtime) | `tier-0` | `docs/architecture/<DOMAIN>.md` + `docs/plan/<DOMAIN>.md` |
| `intergrax/runtime/` | `tier-1` | Same domain pair (ORCHESTRATION, UAEP, NEXUS_EXECUTION_FLOW, …) |
| `agents/<slug>/` | `tier-2` | `agents/<slug>/ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` if present |
| `applications/<pkg>/` | `tier-3` | `applications/<pkg>/ARCHITECTURE.md` + local plan |

Use Harness audit map layer in the entry when helpful — see [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Workflow (agent)

1. Complete implementation per iteration rules (tests, plan/architecture updates, ADR if needed).
2. Copy [`ENTRY_TEMPLATE.md`](ENTRY_TEMPLATE.md) → `entries/YYYY-MM-DD-<scope>-<slug>.md`.
3. Fill frontmatter and sections in **English**.
4. Append one row to [`INDEX.md`](INDEX.md) (newest first).
5. Deliver iteration summary in chat (operator language) **and** point to the journal file path.

**Quality rules:**

- Link `plan_ref` / GAP / AUDIT-IDEAL IDs — do not copy plan tables.
- State `adr: none` with rationale when no ADR was required.
- Set `commit` after the operator commits, or `pending` until then.

---

## File layout

```text
docs/guides/implementation-journal/
  README.md           ← this file
  ENTRY_TEMPLATE.md   ← copy for new entries
  INDEX.md            ← chronological index (newest first)
  entries/            ← one markdown file per completed iteration
```

---

## Related

- [INTERGRAX_DEVELOPMENT_STRATEGY.md](../INTERGRAX_DEVELOPMENT_STRATEGY.md) — work cycle
- [`.cursor/rules/intergrax-implementation-journal.mdc`](../../../.cursor/rules/intergrax-implementation-journal.mdc) — Cursor rule
- [`.cursor/rules/intergrax-iteration.mdc`](../../../.cursor/rules/intergrax-iteration.mdc) — Definition of Done
