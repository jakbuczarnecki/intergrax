# Audit read slice — `UNIFIED_EXECUTION_RUNTIME`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/UNIFIED_EXECUTION_RUNTIME.md`](../../audit/UNIFIED_EXECUTION_RUNTIME.md) §0

---

## Audit-map layers

4–5, 8, 23–24

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | §4 Identity · §5 Policy · §8 Execution runtime · §23 Security · §24 Cost |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | §4–§5 · §8 · §23–§24 |
| `docs/guides/SYSTEM_INVARIANTS.md` | SYS-INV-POL-* · SYS-INV-UAEP-* (grep IDs — do not read full file) |
| `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` | **Hub:** §6 open queue · phase registers on demand |
| `docs/architecture/UNIFIED_EXECUTION_RUNTIME.md` | UAEP · PolicyEngine · ToolRuntime · RuntimeEvent sections |

## Do not load unless cited

- Full multi-thousand-line plan files
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- `docs/guides/audit_slices/` for other domains

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
