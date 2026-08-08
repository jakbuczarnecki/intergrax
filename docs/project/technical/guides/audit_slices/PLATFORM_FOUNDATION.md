# Audit read slice — `PLATFORM_FOUNDATION`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/project/maintainers/audit/PLATFORM_FOUNDATION.md`](../../../maintainers/audit/PLATFORM_FOUNDATION.md) §0

---

## Audit-map layers

1–2, 32

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | §1 Strategic frame · §2 Tier model · §32 Documentation governance |
| `docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | §1–§2 · §32 · maturity §5 |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | SYS-INV-TIER-* · SYS-INV-DOC-* · P2-ARCH-01 (grep IDs — do not read full file) |
| `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` | **Read-scope:** §6.1 maintenance · §6.3 out-of-scope notice · [`plan/satellites`](../plan/satellites) on demand |
| `docs/project/architecture/PLATFORM_FOUNDATION.md` | §1–§6 hub · [`satellites`](../architecture/satellites) on demand |

## Code entry (grep first — F5-B)

- `intergrax/scaffold` — scaffolding CLI
- `scripts/maintenance/check_plan_hub_size.py` — plan hub gate

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
