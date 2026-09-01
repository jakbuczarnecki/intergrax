# Architecture Debt Register

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
| DEBT-UE-9D-01 | §19 Execution | Sync bounded tool loop without CE (`run_bounded_tool_loop` → `BoundedReactPattern` → `append_native_tool_messages`) when `context_engine` is unwired | platform | UE-9D | - | **Open** - TRANSITIONAL |
