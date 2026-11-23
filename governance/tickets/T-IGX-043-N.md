# T-IGX-043-N — FC-1 residual: human decision store → ToolWiringContext (NT-FC1-26 case 2)

## Scope

Delivery ticket for master-plan node **PT-FC1** follow-on (**standing F-4** in continuation plan A.4): close **NT-FC1-26 case 2** — wire the harness-injected human decision store through `ToolWiringContext` without a second authority store on the host path.

Prerequisite: **T-IGX-043-A** (FC-1) landed or reviewed on `development` (PR **#680** / digest @ `a6999e97`); do not duplicate FC-1 Part B work already in 043-A (`nexus_loop.py`, `build_harness_host_runtime` composition guards).

Primary write set: `intergrax/applications/_shared/tool_wiring.py` and harness-path callers that materialize `ApplicationToolWiring` (enumerated in the plan). Disjoint from trust/supply (**051-N**) and from TENANT (**051-A**).

Tor: `development` only. Overlaps **T-IGX-043-J** on workflow files — **not** `harness_host_runtime.py` in this ticket. Serialize per [`WP12_W3_PT_TICKETS_PREP.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/WP12_W3_PT_TICKETS_PREP.md).

**Runda 1 operator defaults:** **D3** (043-M Part E), **D4** (WF E2–E4), **D5** (051-A) **BLOCKED** — no product work on those rows in this wave.

Scaffold hardened from MEP W3 prep — no PASS, FRZ promotion, or independent-verification claim.

## Plan digest

Plan digest: c4bca151a8176b642ad569145f92a3606e595ed1716c47a34205f5778d90888f

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-N.md`](plans/PLAN_T-IGX-043-N.md).

## Approved

Approved: operator @ 2026-11-23T13:30:00+01:00

Wave-packet approval for development→main sync.



## Test contract

Per plan §4: red-first **NT-FC1-26 case 2** (`test_fc1_harness_strict_persistence_wiring.py`, second sub-case); case 1 remains **043-A** regression. FC-1 harness persistence slice after FC-1 lands; `uv run python -m scripts.governance.igx_gov ticket` before push when product paths change.

## Out of scope

Full FC-1 rewrite, **T-IGX-043-M** Part E, **043-J** E2–E4 (D4 BLOCKED), TENANT **051-A** (D5 BLOCKED), train merge, FRZ promotion, FCS, `ARCH_FREEZE_CLOSED`.

## Lifecycle state

quality_gate