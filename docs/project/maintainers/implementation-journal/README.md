# Implementation Journal

**Purpose:** Chronological, narrative record of **completed implementations** across the full Intergrax stack — Tier-0 platform, Tier-1 Nexus, Tier-2 agents, and Tier-3 applications.

**Role in documentation:**

| Layer | Source of truth | Journal adds |
|-------|-----------------|--------------|
| Architecture canon | `docs/project/architecture/<DOMAIN>.md`, agent/app `ARCHITECTURE.md` | Operator intent, episode context |
| Implementation status | `docs/project/maintainers/plans/<DOMAIN>.md`, agent `IMPLEMENTATION_PLAN.md` | Cross-domain timeline, impact narrative |
| Significant decisions | `docs/project/technical/adr/`, agent/app `adr/` | Link only — not a second ADR store |
| Technical proof | Git commits, tests | Verification commands and outcome |

The journal is **operator tooling** (alongside [`docs/audit_results/AUDIT_PROTOCOL.md`](../../../audit_results/AUDIT_PROTOCOL.md), [`LAYER_COMPLETION_MODE.md`](../LAYER_COMPLETION_MODE.md)). It does **not** replace domain pairs or plan rows.

**Operator request** sections are **paraphrases** of the architect's intent — not verbatim chat transcripts.

---

## When to write an entry

**Default: skip.** The journal is a **milestone log**, not a Cursor session log. Git commits, PR descriptions, and plan row updates are the routine record for ordinary iterations.

Write an entry **only** when at least one **milestone trigger** below applies. Otherwise state **"no journal needed"** in the chat summary (with one-line rationale).

### Milestone triggers (write)

| Trigger | Examples |
|---------|----------|
| **Layer / phase closeout** | Full Harness LC, LCM 1–6 domain closeout, named plan phase header marked **Done** |
| **ADR or significant contract change** | New or updated harness/agent/app ADR; architecture canon changes platform contracts or capability surface |
| **Cross-domain or program closeout** | AUDIT-IDEAL band closeout, multi-domain maintenance batch with architectural impact |
| **External validation** | Partner sign-off, production PoC validation, operational L3 sign-off |
| **New harness / product capability** | New Tier-0 mechanism, new agent/application **shipped** (not scaffold-only) |
| **Operator request** | Operator explicitly asks for a journal entry |

### Default skip (do not write)

| Skip | Examples |
|------|----------|
| Routine plan row | Single `*-MAINT-*`, gate fix, typo, formatting, comment-only |
| Docs-only sync | Plan registration after audit with no code; hub/link updates; Mode I idea audit |
| Exploratory work | Brainstorming, spike with no merged artifact |
| Ordinary iteration | One coherent PR that closes a row but is not a milestone — use commit + plan update only |
| Operator opt-out | Operator says **no journal** |

**One entry per milestone** (not per Cursor chat). Multiple related plan IDs in one milestone → one entry listing all refs in `plan_ref`.

---

## Scope routing

| Work location | `tiers` | Primary traceability |
|---------------|---------|----------------------|
| `intergrax/` (non-runtime) | `tier-0` | `docs/project/architecture/<DOMAIN>.md` + `docs/project/maintainers/plans/<DOMAIN>.md` |
| `intergrax/runtime/` | `tier-1` | Same domain pair (ORCHESTRATION, UAEP, NEXUS_EXECUTION_FLOW, …) |
| `agents/<slug>/` | `tier-2` | `agents/<slug>/ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` if present; `agents/<slug>/adr/` when applicable |
| `applications/<pkg>/` | `tier-3` | `applications/<pkg>/docs/ARCHITECTURE.md` + `applications/<pkg>/docs/IMPLEMENTATION_PLAN.md`; `applications/<pkg>/docs/project/technical/adr/` when applicable |
| `intergrax/applications/_shared/` | `tier-3` | Shared host wiring — link consuming `applications/<pkg>/` host(s) in Traceability |

**Tier-3 doc layout:** Application root contains `README.md` only. All other Markdown documentation belongs under `applications/<pkg>/docs/`.

Multi-tier episodes (e.g. agent + application in one PR): set `tiers: [tier-2, tier-3]` and list all scopes in `scope` or Traceability.

Platform audit procedure — see [`docs/audit_results/AUDIT_PROTOCOL.md`](../../../audit_results/AUDIT_PROTOCOL.md).

---

## Workflow (agent)

1. Complete implementation per iteration rules (tests, plan/architecture updates, ADR if needed).
2. **If a milestone trigger applies** (see above): create `entries/YYYY-MM-DD/` if needed; copy [`entries/_TEMPLATE.md`](entries/_TEMPLATE.md) → `entries/YYYY-MM-DD/<scope>-<slug>.md`.
3. Assign `id` per [`ENTRY_TEMPLATE.md`](ENTRY_TEMPLATE.md) §ID assignment.
4. Fill frontmatter and sections in **English**; `plan_ref` = formal IDs only (YAML list).
5. **Prepend** one row to [`INDEX.md`](INDEX.md) (newest first — do not append).
6. Deliver iteration summary in chat (operator language). Include journal path **only when an entry was written**; otherwise **"no journal needed"** + rationale.

**Quality rules:**

- Link `plan_ref` / GAP / AUDIT-IDEAL IDs — do not copy plan tables.
- State `adr: none` with rationale when no ADR was required.
- Set `commit` after the operator commits, or `pending` until then.
- Run `python scripts/maintenance/check_implementation_journal.py` — INDEX rows must match entry files and required sections.

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
docs/project/maintainers/implementation-journal/
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

- [INTERGRAX_DEVELOPMENT_STRATEGY.md](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) — work cycle (CONCLUSIONS → journal)
- [`.cursor/rules/intergrax-implementation-journal.mdc`](../../../.cursor/rules/intergrax-implementation-journal.mdc) — Cursor rule
- [`.cursor/rules/intergrax-iteration.mdc`](../../../.cursor/rules/intergrax-iteration.mdc) — Definition of Done
