# Audit read slice — `CODE_CRAFT`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/CODE_CRAFT.md`](../../audit/CODE_CRAFT.md) §0

---

## Audit-map layers

11b

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | Sections matching audit-map layers 11b |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | Layers 11b · maturity §5 |
| `docs/guides/SYSTEM_INVARIANTS.md` | Grep SYS-INV-* IDs from audit dimensions only (grep IDs — do not read full file) |
| `docs/plan/CODE_CRAFT.md` | **Read-scope:** Hub §6 · [`plan/plan/`](../plan/plan/) satellites on demand |
| `docs/architecture/CODE_CRAFT.md` | Read-scope block + TOC sections for layers 11b |

## Code entry (grep first — F5-B)

- `docs/architecture/CODE_CRAFT.md` — read-scope block only
- `docs/plan/CODE_CRAFT.md` — read-scope block only
- `docs/guides/SYMBOL_INDEX.md` — symbol grep map

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
