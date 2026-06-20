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
| `docs/plan/ORCHESTRATION.md` | **Hub:** Phase ORCH-* · §6.1aw maintenance |
| `docs/architecture/ORCHESTRATION.md` | Intake · scheduler · graph · NexusPlan sections |

## Do not load unless cited

- Full multi-thousand-line plan files
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- `docs/guides/audit_slices/` for other domains

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
