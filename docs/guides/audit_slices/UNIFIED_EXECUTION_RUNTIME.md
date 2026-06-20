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
| `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` | **Hub:** §6.1av hub · phase satellites on demand |
| `docs/architecture/UNIFIED_EXECUTION_RUNTIME.md` | §42.1–§42.15 hub · [`arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md`](../architecture/arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md) on demand |

## Code entry (grep first — F5-B)

- `intergrax/runtime/nexus/policy/` — PolicyEngine
- `intergrax/runtime/nexus/execution/` — UAEP / HarnessKernel
- `intergrax/runtime/nexus/events/` — RuntimeEvent spine

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
