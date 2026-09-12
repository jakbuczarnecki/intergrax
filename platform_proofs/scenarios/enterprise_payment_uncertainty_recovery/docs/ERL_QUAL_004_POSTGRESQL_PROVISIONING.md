---
qualification_id: ERL-QUAL-004
scenario_slug: enterprise_payment_uncertainty_recovery
document_type: postgresql_provisioning_implementation
lifecycle: IMPLEMENTATION_FOUNDATION
status: IMPLEMENTED
---

# ERL-QUAL-004 — PostgreSQL Provisioning

**Enterprise Payment Uncertainty Recovery**

[← Public scenario page](../README.md) · [Data Provisioning Architecture](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md) · [PostgreSQL Schema](ERL_QUAL_004_POSTGRESQL_SCHEMA_IMPLEMENTATION.md)

---

## 1. Purpose

PostgreSQL provisioning is **one replaceable implementation** of `ScenarioProvisioningPort`. It reads the vendor-neutral `dataset/` package and materializes rows into the lab PostgreSQL instance under `infrastructure/`. The dataset remains qualification truth; the adapter performs **materialization only**.

```text
dataset/ (JSON)
        ↓
ScenarioProvisioningPort
        ↓
PostgreSqlScenarioProvisioner  (provisioning/postgresql/)
        ↓
commerce / external_sor / reconciliation schemas
```

Application workflow code must not import this adapter directly — the proof harness binds a port implementation.

---

## 2. Adapter responsibility

| Concern | Owner |
| --- | --- |
| Business truth | `dataset/` manifest, shared entities, variant slices |
| Lifecycle orchestration | `contracts/provisioning/lifecycle.py` |
| PostgreSQL materialization | `provisioning/postgresql/` |
| DDL / migrations | `database/migrations/` |
| Docker lab runtime | `infrastructure/docker/` |

The adapter:

- validates manifest and variant selection;
- checks lab PostgreSQL availability;
- inserts scenario rows in a single transaction;
- verifies required rows on state availability;
- deletes scenario rows on cleanup.

---

## 3. Lifecycle

| Phase | Behavior |
| --- | --- |
| **Prepare** | Load and validate dataset; verify PostgreSQL is reachable; ensure core schema exists. |
| **Provision** | Map logical entities to rows; commit atomically. |
| **State availability** | Confirm organization, order, payment intent, application knowledge, external effect, external reality, and reconciliation case rows exist. |
| **Cleanup** | Delete provisioned rows in FK-safe order. |

Typed failures use `ProvisioningFailureCode` from the scenario contract (`INVALID_DATASET`, `MISSING_SCENARIO_VARIANT`, `PROVISIONING_UNAVAILABLE`, `INCOMPLETE_PROVISIONING`, `CLEANUP_FAILURE`, etc.).

---

## 4. Dataset flow

1. `manifest.json` resolves the variant path (shared with the reference in-memory provisioner).
2. Shared JSON files (`order`, `external_effect`, `application_knowledge_at_entry`, …) are loaded and cross-validated.
3. Variant slice supplies `external_reality` and `reconciliation` parameters.
4. `materialization.py` maps dataset fields to PostgreSQL columns using **data-driven tables** (capture outcome, truth establishment, reconciliation availability) — not `if variant == A` branches.

Investigation attempts, evidence references, and resolution records are **not** seeded at scenario entry; the reconciliation case is opened in `OPEN` state for runtime workflow.

---

## 5. Ownership

- **Dataset** — what is true for qualification.
- **Provisioner** — how that truth becomes PostgreSQL rows for the lab.
- **Database** — runtime persistence and constraints.
- **Application / Integrax** — workflow and reliability processing (out of scope for this adapter).

---

## 6. Limitations

- Requires the Docker lab from [PostgreSQL Infrastructure](ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md) and `infrastructure/config/postgres.env`.
- Schema bootstrap applies `001_erl_qual_004_core_schema.sql` when tables are missing; production-style migration orchestration is not included.
- Deterministic surrogate UUIDs and timestamps are derived from logical dataset identifiers — not duplicated business literals in code.
- Communication events and inventory context are not separate tables; reservation knowledge is stored on `application_knowledge`.

---

## 7. Validation

| Layer | Location |
| --- | --- |
| Unit (loader, mapping, typed failures) | `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_erl_qual_004_postgres_provisioning.py` |
| Integration (full lifecycle, real PostgreSQL) | `tests/integration/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_erl_qual_004_postgres_provisioning_integration.py` (skipped when lab is down) |

---

## Document validation

| Check | Result |
| --- | --- |
| PostgreSQL is one provisioning implementation | **Pass** |
| Dataset remains source of truth | **Pass** |
| No application / ERL logic in adapter | **Pass** |
| Port boundary preserved | **Pass** |
