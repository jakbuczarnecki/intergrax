# Audit read slice — `UNIFIED_CONTEXT_LIFECYCLE`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/project/maintainers/audit/UNIFIED_CONTEXT_LIFECYCLE.md`](../../audit/UNIFIED_CONTEXT_LIFECYCLE.md) §0

---

## Audit-map layers

UCL

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | One ledger · one budget · one executor · Nexus coordinator |
| `docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | UCL · EPHEMERAL_ASSEMBLY · DURABLE_COMPACTION |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | UCL §21 explicit invariants · ADR-UCL-001 (grep IDs — do not read full file) |
| `docs/project/maintainers/plans/UNIFIED_CONTEXT_LIFECYCLE.md` | **Read-scope:** CTX-UCL-CLOSEOUT-1 closed · TOKEN-10E-1 READY_FOR_REVIEW |
| `docs/project/architecture/UNIFIED_CONTEXT_LIFECYCLE.md` | §2 purpose · §5 ownership · §6 lifecycle · §21 invariants |

## Code entry (grep first — F5-B)

- `intergrax/runtime/context_lifecycle/` — artifact repository contracts
- `intergrax/runtime/nexus/context/ucl_orchestration.py` — resolve_ucl_context_plan

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
