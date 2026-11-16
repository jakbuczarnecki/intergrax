# T-IGX-034-G — PRE-ENT Wave 1: QUAL-X Population B–D serial qualification (pinned; Pop E @ ENT)

## Scope

Wave 1 **`QUAL-X`** batch in the PRE-ENT predecessor certification campaign. Execute **serial pytest** for **Populations B, C, and D** only at pinned current-HEAD evidence (OD-HIST01-B option **(b)** ratified by operator **D-PREENT-03=b**).

**In scope (Pop B–D):**

- Population **B** — live-slice / deferred live qualification rows explicitly scoped to PRE-ENT predecessor replay (not global PROD-Q closure).
- Population **C** — closed-world / historical scoped qualification modules tied to predecessor batches on `AUDIT_HEAD`.
- Population **D** — package-level dimension-A mechanical gates where historical dossier assigns them to pinned SHA evidence (not platform rollup).

**Out of scope by operator routing (D-PREENT-03=b):**

- **Population E** — enterprise REG rollup, FRZ-REG global promotion, ENT-AUDIT-X Population E qualification. Owner: **ENT-AUDIT-X** campaign; residual recorded in PRE-ENT blockers only.

Deliverables: qualification log under `docs/project/maintainers/qualification/pre_ent_enterprise_certification/` (append to campaign progress); ticket checkpoint with pytest exit codes. **No** G6 PASS, **no** QUAL-X parent global closure, **no** FRZ-REG row promotion.

## Plan digest

Plan digest: (pending — PRE-ENT campaign plan `pre-ent_certification_closure`)

## Approved

Operator routing **D-PREENT-03=b** (2026-11-10): Pop **B–D** pinned @ PRE-ENT; Pop **E** deferred to ENT-AUDIT-X.

## Test contract

Serial pytest — exact module list to be frozen in first execution session from [`QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md`](../../docs/project/maintainers/qualification/QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md) **excluding Population E rollup modules** per OD-HIST01-B (b). Minimum smoke before batch close:

```text
# Representative Pop B–D gate — adjust to frozen list on first run
uv run --extra dev-ci python -m pytest -p no:xdist tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py -q --collect-only
```

Full serial contract recorded in ticket **Checkpoint** when implementer runs wave.

## Out of scope

- Population E qualification / ENT-AUDIT-X rollup
- FRZ-REG-01..10 global PASS claims
- Independent G6 auditor session (Wave 3 covers predecessor product batches, not full QUAL-X)
- Push/merge unless operator directs

## Lifecycle state

plan_approved

## Checkpoint

Routing only @ operator **D-PREENT-03=b** (2026-11-10). **Not started.** Parent: [`T-IGX-034-A.md`](T-IGX-034-A.md), [`PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md) §7 appendix.
