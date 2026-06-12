# Architecture Debt Register

Living register of harness architecture debt (IDEAL-32.2). Review quarterly with `docs/plan/IDEAL_HARNESS_L3.md` and `docs/plan/AUDIT_IDEAL_2026.md`.

| ID | Layer | Description | Owner | Target wave | AUDIT-IDEAL ID | Status |
|----|-------|-------------|-------|-------------|----------------|--------|
| DEBT-19-01 | §19 Registry | Durable cross-host registry snapshot store | platform | W1 (Band 2az) | AUDIT-IDEAL-19.1 | **Closed** (2026-06-09) |
| DEBT-25-01 | §25 Evaluation | Shadow eval path automation | platform | W2 (Band 2az) | AUDIT-IDEAL-25.1 | **Closed** (2026-06-09) |
| DEBT-28-01 | §28 Tier-3 | Product durable queue default beyond SQLite scaffold | product | W2 / Band 3 | AUDIT-IDEAL-28.1 | **Closed** |
| DEBT-ECP-01 | §30 Ops | Sync `architecture/ELASTIC_CAPACITY_AND_SCALING.md` §22 after ECP-DEPTH | platform | W1 (Band 2az) | AUDIT-IDEAL-30.1 | **Closed** (2026-06-12 honest maturity) |
| DEBT-ECP-02 | §30 Ops | Production elasticity — live signal bridge, K8s/Celery adapters, E2E loop | platform | ECP-PROD | AUDIT-IDEAL-30.4 | **Open** |
| DEBT-MEM-01 | §15 Memory | Org memory 2.5 (organizational LTM scope) | platform | W1 (Band 2az) | AUDIT-IDEAL-15.1 | **Closed** (2026-06-09) |
