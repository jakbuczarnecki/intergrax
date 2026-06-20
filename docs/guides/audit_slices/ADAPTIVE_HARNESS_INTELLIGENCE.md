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
| `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` | **Hub:** Hub §6 open rows · [`plan/plan/`](../plan/plan/) satellites on demand |
| `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` | TOC sections for layers L4 AHI · see Cursor read scope block |

## Do not load unless cited

- Full multi-thousand-line plan files
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- `docs/guides/audit_slices/` for other domains

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
