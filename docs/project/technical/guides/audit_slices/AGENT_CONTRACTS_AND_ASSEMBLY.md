# Audit read slice — `AGENT_CONTRACTS_AND_ASSEMBLY`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/project/maintainers/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../../maintainers/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md) §0

---

## Audit-map layers

17–20, 31 · ACP §21

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | §12–§21 Agent / registry / ACP |
| `docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | §17–§20 · §31 |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | SYS-INV-ACP-* · SYS-INV-AGENT-* (grep IDs — do not read full file) |
| `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` | **Read-scope:** §6 open · ACP closeout registers on demand |
| `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §12–§21 hub · [`satellites`](../architecture/satellites) on demand |

## Code entry (grep first — F5-B)

- `intergrax/runtime/nexus/agent` — agent contracts, registry
- `agents` — Tier-2 agent implementations

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
