# Audit read slice — `TIER3_APPLICATION_ENVIRONMENT`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/TIER3_APPLICATION_ENVIRONMENT.md`](../../audit/TIER3_APPLICATION_ENVIRONMENT.md) §0

---

## Audit-map layers

3, 28

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | Sections matching audit-map layers 3, 28 |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | Layers 3, 28 · maturity §5 |
| `docs/guides/SYSTEM_INVARIANTS.md` | Grep SYS-INV-* IDs from audit dimensions only (grep IDs — do not read full file) |
| `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` | **Read-scope:** Hub §6 · [`plan/satellites/`](../plan/satellites/) satellites on demand |
| `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` | Read-scope block + TOC sections for layers 3, 28 |

## Code entry (grep first — F5-B)

- `applications/` — Tier-3 hosts
- `intergrax/runtime/nexus/application/` — HarnessApplication

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
