# Audit read slice — `ADAPTIVE_HARNESS_INTELLIGENCE`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../audit/ADAPTIVE_HARNESS_INTELLIGENCE.md) §0

---

## Audit-map layers

L4 AHI

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | Sections matching audit-map layers L4 AHI |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | Layers L4 AHI · maturity §5 |
| `docs/guides/SYSTEM_INVARIANTS.md` | Grep SYS-INV-* IDs from audit dimensions only (grep IDs — do not read full file) |
| `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` | **Read-scope:** Hub §6 · [`plan/plan/`](../plan/plan/) satellites on demand |
| `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` | Read-scope block + TOC sections for layers L4 AHI |

## Code entry (grep first — F5-B)

- `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` — read-scope block only
- `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` — read-scope block only
- `docs/guides/SYMBOL_INDEX.md` — symbol grep map

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
