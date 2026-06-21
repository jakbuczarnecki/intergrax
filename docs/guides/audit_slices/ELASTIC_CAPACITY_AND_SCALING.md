# Audit read slice — `ELASTIC_CAPACITY_AND_SCALING`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/ELASTIC_CAPACITY_AND_SCALING.md`](../../audit/ELASTIC_CAPACITY_AND_SCALING.md) §0

---

## Audit-map layers

30

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | Sections matching audit-map layers 30 |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | Layers 30 · maturity §5 |
| `docs/guides/SYSTEM_INVARIANTS.md` | Grep SYS-INV-* IDs from audit dimensions only (grep IDs — do not read full file) |
| `docs/plan/ELASTIC_CAPACITY_AND_SCALING.md` | **Read-scope:** Hub §6 · [`plan/satellites/`](../plan/satellites/) satellites on demand |
| `docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md` | Read-scope block + TOC sections for layers 30 |

## Code entry (grep first — F5-B)

- `docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md` — read-scope block only
- `docs/plan/ELASTIC_CAPACITY_AND_SCALING.md` — read-scope block only
- `docs/guides/SYMBOL_INDEX.md` — symbol grep map

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
