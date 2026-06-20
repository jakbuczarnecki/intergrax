# Audit read slice — `CRITIC_VERIFICATION`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/audit/CRITIC_VERIFICATION.md`](../../audit/CRITIC_VERIFICATION.md) §0

---

## Audit-map layers

25–27, 30

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | §18 Critic · §25 Evaluation |
| `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | §25–§27 · §30 |
| `docs/guides/SYSTEM_INVARIANTS.md` | SYS-INV-EVAL-* · SYS-INV-CRIT-* (grep IDs — do not read full file) |
| `docs/plan/CRITIC_VERIFICATION.md` | **Hub:** AUDIT-IDEAL · §CVL-4 backlog · audit_history satellite |
| `docs/architecture/CRITIC_VERIFICATION.md` | CVL contracts · PEV · evaluator loop |

## Code entry (grep first — F5-B)

- `intergrax/runtime/nexus/critic/` — CVL orchestrator

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
