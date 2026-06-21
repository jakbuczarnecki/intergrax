# Audit read slice — `AGENT_CONTRACTS_AND_ASSEMBLY`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../audit/AGENT_CONTRACTS_AND_ASSEMBLY.md) §0

---

## Audit-map layers

17–20, 31 · ACP §21

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | §12–§21 Agent / registry / ACP |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | §17–§20 · §31 |
| `docs/guides/SYSTEM_INVARIANTS.md` | SYS-INV-ACP-* · SYS-INV-AGENT-* (grep IDs — do not read full file) |
| `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | **Read-scope:** §6 open · ACP closeout registers on demand |
| `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §12–§21 hub · [`satellites/`](../architecture/satellites/) on demand |

## Code entry (grep first — F5-B)

- `intergrax/runtime/nexus/agent/` — agent contracts, registry
- `agents/` — Tier-2 agent implementations

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
