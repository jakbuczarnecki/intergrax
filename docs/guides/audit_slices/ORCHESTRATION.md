# Audit read slice — `ORCHESTRATION`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/ORCHESTRATION.md`](../../audit/ORCHESTRATION.md) §0

---

## Audit-map layers

3, 9

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | §3 Intake · §9 Orchestration / graph |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | §3 · §9 |
| `docs/guides/SYSTEM_INVARIANTS.md` | SYS-INV-ORCH-* (grep IDs — do not read full file) |
| `docs/plan/ORCHESTRATION.md` | **Hub:** Phase ORCH-* hub · satellites on demand |
| `docs/architecture/ORCHESTRATION.md` | §10–§26 hub · [`arch/ORCHESTRATION_extended_depth.md`](../architecture/arch/ORCHESTRATION_extended_depth.md) on demand |

## Code entry (grep first — F5-B)

- `intergrax/runtime/nexus/orchestration/` — intake, NexusLoop
- `intergrax/runtime/nexus/orchestration/graph/` — execution graph

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
