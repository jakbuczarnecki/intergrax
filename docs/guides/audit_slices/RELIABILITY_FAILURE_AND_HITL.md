# Audit read slice — `RELIABILITY_FAILURE_AND_HITL`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/RELIABILITY_FAILURE_AND_HITL.md`](../../audit/RELIABILITY_FAILURE_AND_HITL.md) §0

---

## Audit-map layers

22

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | Sections matching audit-map layers 22 |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | Layers 22 · maturity §5 |
| `docs/guides/SYSTEM_INVARIANTS.md` | Grep SYS-INV-* IDs from audit dimensions only (grep IDs — do not read full file) |
| `docs/plan/RELIABILITY_FAILURE_AND_HITL.md` | **Read-scope:** Hub §6 · [`plan/satellites/`](../plan/satellites/) satellites on demand |
| `docs/architecture/RELIABILITY_FAILURE_AND_HITL.md` | Read-scope block + TOC sections for layers 22 |

## Code entry (grep first — F5-B)

- `docs/architecture/RELIABILITY_FAILURE_AND_HITL.md` — read-scope block only
- `docs/plan/RELIABILITY_FAILURE_AND_HITL.md` — read-scope block only
- `docs/guides/SYMBOL_INDEX.md` — symbol grep map

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
