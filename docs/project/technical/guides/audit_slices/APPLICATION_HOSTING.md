# Audit read slice — `APPLICATION_HOSTING`

**Purpose:** Replace bulk loading of `IDEAL_HARNESS_AI_ARCHITECTURE.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and full plan/architecture files during **audit-only** sessions.

**Parent audit prompt:** [`docs/project/maintainers/audit/APPLICATION_HOSTING.md`](../../audit/APPLICATION_HOSTING.md) §0

---

## Audit-map layers

HOST

## Read instead of full guides

| Source | Read only |
|--------|-----------|
| `docs/project/technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` | Hosting lifecycle around a configured Tier-3 application |
| `docs/project/technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` | HOST · maturity four-axis A/I/P/E in domain pair |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | HOST-INV-01..12 (grep IDs — do not read full file) |
| `docs/project/maintainers/plans/APPLICATION_HOSTING.md` | **Read-scope:** APP-HOST waves · gates §6 · fidelity matrix §9 |
| `docs/project/architecture/APPLICATION_HOSTING.md` | §1 purpose · §3 invariants · §6–§12 engine/lifecycle/OS |

## Code entry (grep first — F5-B)

- `intergrax/hosting/` — HostedApplicationEngine / supervisor
- `intergrax/hosting/runner.py` — run_hosted_application

## Do not load unless cited

- Full multi-thousand-line plan or architecture files (use hub + **one** satellite)
- `docs/audit_results/` (unless RESUME)
- Unrelated domain pairs
- Other domains' `audit_slices/`

## Evidence rule (unchanged)

Audit quality requires **code paths + gate scripts + tests** — this slice reduces documentation bulk only.
