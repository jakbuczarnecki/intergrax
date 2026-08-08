# Audit read slice — `INTEGRATIONS`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/project/maintainers/audit/INTEGRATIONS.md`](../../../maintainers/audit/INTEGRATIONS.md) §0

---

## Audit-map layers

11–12

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | §11 Integration library · §12 Provider model |
| `docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | §11–§12 |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | SYS-INV-INT-* (grep IDs — do not read full file) |
| `docs/project/maintainers/plans/INTEGRATIONS.md` | **Read-scope:** Phase INT / H-INT hub · satellites on demand |
| `docs/project/architecture/INTEGRATIONS.md` | wiring + design principles hub · [`satellites/INTEGRATIONS_provider_catalog.md`](../architecture/satellites/INTEGRATIONS_provider_catalog.md) on demand |

## Code entry (grep first — F5-B)

- `intergrax/integrations/` — integration catalog
- `intergrax/integrations/registry.py` — slug registration

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
