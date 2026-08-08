# Audit read slice — `NEXUS_EXECUTION_FLOW`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/project/maintainers/audit/NEXUS_EXECUTION_FLOW.md`](../../../maintainers/audit/NEXUS_EXECUTION_FLOW.md) §0

---

## Audit-map layers

8–10

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | §8 Runtime · §9 Graph · §10 Subagents |
| `docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | §8–§10 |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | SYS-INV-FLOW-* · SYS-INV-DELEG-* (grep IDs — do not read full file) |
| `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` | **Read-scope:** Phase FLOW hub · satellites on demand |
| `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | §1–§8 hub · [`satellites/NEXUS_EXECUTION_FLOW_extended_depth.md`](../architecture/satellites/NEXUS_EXECUTION_FLOW_extended_depth.md) on demand |

## Code entry (grep first — F5-B)

- `intergrax/runtime/nexus/orchestration/nexus_loop.py` — NexusLoop
- `intergrax/runtime/nexus/orchestration/intake` — task intake

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
