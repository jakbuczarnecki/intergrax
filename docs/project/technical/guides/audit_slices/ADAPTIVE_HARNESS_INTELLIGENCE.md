# Audit read slice — `ADAPTIVE_HARNESS_INTELLIGENCE`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/project/maintainers/audit/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../../maintainers/audit/ADAPTIVE_HARNESS_INTELLIGENCE.md) §0

---

## Audit-map layers

L4 AHI

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | Sections matching audit-map layers L4 AHI |
| `docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | Layers L4 AHI · maturity §5 |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | Grep SYS-INV-* IDs from audit dimensions only (grep IDs — do not read full file) |
| `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md` | **Read-scope:** Hub §6 · [`plan/satellites/`](../plan/satellites) satellites on demand |
| `docs/project/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` | Read-scope block + TOC sections for layers L4 AHI |

## Code entry (grep first — F5-B)

- `docs/project/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` — read-scope block only
- `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md` — read-scope block only
- `docs/project/technical/guides/SYMBOL_INDEX.md` — symbol grep map

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
