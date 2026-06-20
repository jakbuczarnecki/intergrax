# Audit read slice — `MEMORY`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/MEMORY.md`](../../audit/MEMORY.md) §0

---

## Audit-map layers

15

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | Sections matching audit-map layers 15 |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | Layers 15 · maturity §5 |
| `docs/guides/SYSTEM_INVARIANTS.md` | Grep SYS-INV-* IDs from audit dimensions only (grep IDs — do not read full file) |
| `docs/plan/MEMORY.md` | **Hub:** Hub §6 · [`plan/plan/`](../plan/plan/) satellites on demand |
| `docs/architecture/MEMORY.md` | Read-scope block + TOC sections for layers 15 |

## Code entry (grep first — F5-B)

- `intergrax/runtime/nexus/memory/` — memory stores
- `intergrax/memory/` — LTM facades

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
