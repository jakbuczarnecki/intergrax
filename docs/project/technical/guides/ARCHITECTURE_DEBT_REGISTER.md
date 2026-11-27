# Architecture Debt Register

> **Currency note (EQO-A remediation, as of 2026-11-16).** The table rows are unchanged (the `DEBT-*` rows are parsed by `intergrax/runtime/architecture/debt_burn_down.py`). At `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` row `P1-ARCH-01` reads **Planned** for target APP-EVOL-8, while `docs/project/maintainers/plans/satellites/TIER3_APPLICATION_ENVIRONMENT_embedded_detail.md` L90 reads APP-EVOL-8 (P1-ARCH-01) **Done** - M1-M3; the domain plan decides the status. Nine rows read **Closed** (L8-L16); the table has no closure-evidence or reopen-condition column, and `DEBT-28-01` and `DEBT-UE-9D-01` carry no closure date. Commit `eb5c9e9eef333b676d777a8a49f320d645304e60` (2026-09-17, "refactor(context): close react context authority gap") is the first commit that records `DEBT-UE-9D-01` as Closed in this file. Line references (L<n>) are to the files at that SHA, before this note was added.

Living register of harness architecture debt (IDEAL-32.2). Review quarterly with `docs/project/maintainers/plans/IDEAL_HARNESS_L3.md` and `docs/project/maintainers/plans/AUDIT_IDEAL_2026.md`.

| ID | Layer | Description | Owner | Target wave | AUDIT-IDEAL ID | Status |
|----|-------|-------------|-------|-------------|----------------|--------|
| P1-ARCH-01 | §22 Tier-3 profile | Flat `ApplicationEnvironmentProfile` namespace growth (43+ top-level fields) | platform | APP-EVOL-8 (M1–M3) | - | **Planned** - architecture §22.6 + ADR-APP-003 **accepted** (2026-06-17) |
| P1-ARCH-03 | docs / README | Root `README.md` Overview snapshot stale vs domain canon (ECC `(planned)` vs L3 Done) | platform | doc-sync | - | **Closed** (2026-06-17) |
| P2-ARCH-03 | docs / README | Root `README.md` lacks per-layer maturity visibility for architects (only aggregate L3 statement) | platform | doc-sync | - | **Closed** (2026-06-17) |
| DEBT-19-01 | §19 Registry | Durable cross-host registry snapshot store | platform | W1 (Band 2az) | AUDIT-IDEAL-19.1 | **Closed** (2026-06-09) |
| DEBT-25-01 | §25 Evaluation | Shadow eval path automation | platform | W2 (Band 2az) | AUDIT-IDEAL-25.1 | **Closed** (2026-06-09) |
| DEBT-28-01 | §28 Tier-3 | Product durable queue default beyond SQLite scaffold | product | W2 / Band 3 | AUDIT-IDEAL-28.1 | **Closed** |
| DEBT-ECP-01 | §30 Ops | Sync `architecture/ELASTIC_CAPACITY_AND_SCALING.md` §22 after ECP-DEPTH | platform | W1 (Band 2az) | AUDIT-IDEAL-30.1 | **Closed** (2026-06-12 honest maturity) |
| DEBT-ECP-02 | §30 Ops | Production elasticity - live signal bridge, K8s/Celery adapters, E2E loop | platform | ECP-PROD | AUDIT-IDEAL-30.4 | **Closed** (2026-06-12) |
| DEBT-MEM-01 | §15 Memory | Org memory 2.5 (organizational LTM scope) | platform | W1 (Band 2az) | AUDIT-IDEAL-15.1 | **Closed** (2026-06-09) |
| DEBT-UE-9D-01 | §19 Execution | Sync bounded tool loop without CE (`run_bounded_tool_loop` → `BoundedReactPattern` → `append_native_tool_messages`) when `context_engine` is unwired | platform | UE-9D | MEM-XINT-4-R | **Closed** - iterative ReAct requires `run_bounded_tool_loop_async` + CE |

## Closed-Entry Evidence & Reopen Annex (EQO-W4, `EDER-F-0183`)

> **Parser-safety note (as of 2026-11-26 @ `646e63953`):** The 7-column debt register table above (L7–L18) is unchanged so `intergrax/runtime/architecture/debt_burn_down.py` (`_DEBT_ROW_PATTERN`) parses the exact same records. This 4-column annex supplies historical closure-commit evidence and reopen conditions for all 9 **Closed** entries and adds no new PASS, certification, or FRZ claim.

| Entry ID | Closure Date | Evidence Pointer | Reopen Condition |
| --- | --- | --- | --- |
| `P1-ARCH-03` | 2026-06-17 | Commit `e4c20f180` (*docs: sync root README with ECC L3 closeout (P1-ARCH-03)*) | Root `README.md` Overview snapshot drifts from canonical domain L3 status without sync in the same change. |
| `P2-ARCH-03` | 2026-06-17 | Commit `eb99f407a` (*docs: add README platform maturity table (P2-ARCH-03)*) | Root `README.md` drops per-layer maturity visibility or omits newly promoted platform layers. |
| `DEBT-19-01` | 2026-06-09 | Commit `f9d1354ff` (*implement audit-ideal W1*); W1 Band 2az registry snapshot store | Durable cross-host registry snapshot store contract is bypassed, removed, or regresses in qualification tests. |
| `DEBT-25-01` | 2026-06-09 | Commit `f9d1354ff` (*implement audit-ideal W1*); W2 Band 2az shadow eval automation | Shadow eval path automation is disabled or loses deterministic gate coverage without replacement. |
| `DEBT-28-01` | 2026-06-10 | Commit `7fa375560` (*Close AUDIT-IDEAL W2 P1 batch: product security, durable queue, Graph RAG, and L4 evidence gates*) | Product durable queue default regresses to SQLite-only scaffold outside ratified single-node/C10-C15 profile boundaries. |
| `DEBT-ECP-01` | 2026-06-12 | Commit `ef65ce4ed` (*docs(ecp): honest maturity — architecture complete, production elasticity deferred*) | `docs/project/architecture/ELASTIC_CAPACITY_AND_SCALING.md` §22 overstates production elasticity maturity beyond verified runtime evidence. |
| `DEBT-ECP-02` | 2026-06-12 | Commits `c30cd9301` (*feat(ecp): close ECP-PROD with HITL queue and URL-gated K8s adapters*) & `666a44da4` (*docs(ecp): journal entry and debt register for ECP-PROD closeout*) | Live signal bridge, K8s/Celery elasticity adapters, or E2E elasticity loop fails regression checks or loses fail-closed gating. |
| `DEBT-MEM-01` | 2026-06-09 | Commit `f9d1354ff` (*implement audit-ideal W1*); W1 Band 2az org memory 2.5 | Organizational LTM scope breaks tenant isolation or drops required organizational scope enforcement. |
| `DEBT-UE-9D-01` | 2026-09-17 | Commit `eb5c9e9ee` (*refactor(context): close react context authority gap*) | Sync `run_bounded_tool_loop` re-introduces an unwired Context Engine bypass for iterative ReAct tool execution. |
