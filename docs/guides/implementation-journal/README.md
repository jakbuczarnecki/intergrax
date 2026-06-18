# Implementation Journal

**Purpose:** Chronological, narrative record of **completed implementations** across the full Intergrax stack — Tier-0 platform, Tier-1 Nexus, Tier-2 agents, and Tier-3 applications.

**Role in documentation:**

| Layer | Source of truth | Journal adds |
|-------|-----------------|--------------|
| Architecture canon | `docs/architecture/<DOMAIN>.md`, agent/app `ARCHITECTURE.md` | Operator intent, episode context |
| Implementation status | `docs/plan/<DOMAIN>.md`, agent `IMPLEMENTATION_PLAN.md` | Cross-domain timeline, impact narrative |
| Significant decisions | `docs/adr/`, agent/app `adr/` | Link only — not a second ADR store |
| Technical proof | Git commits, tests | Verification commands and outcome |

The journal is **operator tooling** (alongside [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md), [`LAYER_COMPLETION_MODE.md`](../LAYER_COMPLETION_MODE.md), and [`audit/`](../audit/)). It does **not** replace domain pairs or plan rows.

**Operator request** sections are **paraphrases** of the architect's intent — not verbatim chat transcripts.

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

**One entry per coherent iteration.** Multiple plan IDs in one PR → one entry listing all refs in `plan_ref`.

---

## Scope routing

| Work location | `tiers` | Primary traceability |
|---------------|---------|----------------------|
| `intergrax/` (non-runtime) | `tier-0` | `docs/architecture/<DOMAIN>.md` + `docs/plan/<DOMAIN>.md` |
| `intergrax/runtime/` | `tier-1` | Same domain pair (ORCHESTRATION, UAEP, NEXUS_EXECUTION_FLOW, …) |
| `agents/<slug>/` | `tier-2` | `agents/<slug>/ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` if present; `agents/<slug>/adr/` when applicable |
| `applications/<pkg>/` | `tier-3` | `applications/<pkg>/ARCHITECTURE.md` + local plan |
| `intergrax/applications/_shared/` | `tier-3` | Shared host wiring — link consuming `applications/<pkg>/` host(s) in Traceability |

Multi-tier episodes (e.g. agent + application in one PR): set `tiers: [tier-2, tier-3]` and list all scopes in `scope` or Traceability.

Harness audit map layer — see [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Workflow (agent)

1. Complete implementation per iteration rules (tests, plan/architecture updates, ADR if needed).
2. Create `entries/YYYY-MM-DD/` if needed; copy [`entries/_TEMPLATE.md`](entries/_TEMPLATE.md) → `entries/YYYY-MM-DD/<scope>-<slug>.md`.
3. Assign `id` per [`ENTRY_TEMPLATE.md`](ENTRY_TEMPLATE.md) §ID assignment.
4. Fill frontmatter and sections in **English**; `plan_ref` = formal IDs only (YAML list).
5. **Prepend** one row to [`INDEX.md`](INDEX.md) (newest first — do not append).
6. Deliver iteration summary in chat (operator language) **and** point to the journal file path.

**Quality rules:**

- Link `plan_ref` / GAP / AUDIT-IDEAL IDs — do not copy plan tables.
- State `adr: none` with rationale when no ADR was required.
- Set `commit` after the operator commits, or `pending` until then.
- Run `python scripts/check_implementation_journal.py` — INDEX rows must match entry files and required sections.

### `plan_ref` grammar

| Allowed | Examples |
|---------|----------|
| Plan row ID | `M-RAG.23`, `TOOL-ENG-4`, `OBS-BUS-6` |
| AUDIT-IDEAL row | `AUDIT-IDEAL-14.3`, `AUDIT-IDEAL-28.4` |
| Named plan phase | `FAUDIT-32`, `M-RAG-DEPTH`, `TOOL-ENG` (phase header in domain plan) |
| Product backlog slot | `K.1`, `K.2` (when closing §6.3 items) |

Not allowed: sentences, `Phase …` prefixes, or informal notes — put those in **Operator request** or **Summary**.

---

## File layout

```text
docs/guides/implementation-journal/
  README.md           ← this file
  ENTRY_TEMPLATE.md   ← field reference + ID rules
  INDEX.md            ← chronological index (newest first)
  entries/
    _TEMPLATE.md      ← copy scaffold (not indexed)
    YYYY-MM-DD/       ← one folder per calendar day
      <scope>-<slug>.md
```

---

## Related

- [INTERGRAX_DEVELOPMENT_STRATEGY.md](../INTERGRAX_DEVELOPMENT_STRATEGY.md) — work cycle (WNIOSKI → journal)
- [`.cursor/rules/intergrax-implementation-journal.mdc`](../../../.cursor/rules/intergrax-implementation-journal.mdc) — Cursor rule
- [`.cursor/rules/intergrax-iteration.mdc`](../../../.cursor/rules/intergrax-iteration.mdc) — Definition of Done
