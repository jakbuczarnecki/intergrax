---
qualification_id: ERL-QUAL-004
scenario_slug: enterprise_payment_uncertainty_recovery
document_type: postgresql_schema_implementation
lifecycle: IMPLEMENTATION_FOUNDATION
status: IMPLEMENTED
---

# ERL-QUAL-004 — PostgreSQL Schema Implementation

**Enterprise Payment Uncertainty Recovery**

[← Public scenario page](../README.md) · [Data Model Architecture](ERL_QUAL_004_POSTGRESQL_DATA_MODEL_ARCHITECTURE.md) · [Database README](../database/README.md)

---

## 1. Purpose

This document records the **physical PostgreSQL schema** for the ERL-QUAL-004 lab. DDL lives under `database/`; behavior and semantics remain defined in [ERL_QUAL_004_POSTGRESQL_DATA_MODEL_ARCHITECTURE.md](ERL_QUAL_004_POSTGRESQL_DATA_MODEL_ARCHITECTURE.md).

This is **lab schema only**—not production database readiness, not Integrax platform storage, and not a reconciliation workflow implementation.

---

## 2. Implemented entities

| Entity | Table | Schema zone |
| --- | --- | --- |
| Organization | `commerce.organizations` | Commerce / application |
| Order | `commerce.orders` | Commerce / application |
| Payment intent | `commerce.payment_intents` | Commerce / application |
| Application knowledge | `commerce.application_knowledge` | Commerce / application |
| External payment effect | `external_sor.external_payment_effects` | External SoR |
| External reality | `external_sor.external_reality` | External SoR |
| Reconciliation case | `reconciliation.reconciliation_cases` | Reconciliation |
| Investigation attempt | `reconciliation.investigation_attempts` | Reconciliation |
| Evidence reference | `reconciliation.evidence_references` | Reconciliation |
| Resolution record | `reconciliation.resolution_records` | Reconciliation |

Migration: `database/migrations/001_erl_qual_004_core_schema.sql`  
Snapshot: `database/schema/erl_qual_004_core_schema.sql`

---

## 3. Relationship overview

```text
commerce.organizations
        │
        └── commerce.orders
                 │
                 └── commerce.payment_intents
                          ├── commerce.application_knowledge
                          └── external_sor.external_payment_effects
                                   └── external_sor.external_reality

commerce.orders / payment_intents
        └── reconciliation.reconciliation_cases
                 ├── reconciliation.investigation_attempts
                 ├── reconciliation.evidence_references
                 └── reconciliation.resolution_records
```

`application_knowledge` references `external_payment_effects` for correlation but stores **belief** (`known_status`, `uncertainty_explicit`). `external_reality` holds **SoR terminal truth** (`terminal_outcome`, `processed_at`, `truth_availability_state`) in a separate table and schema.

---

## 4. Ownership boundaries

| Owner | PostgreSQL responsibility |
| --- | --- |
| **Database** | Persistence, referential integrity, CHECK constraints for enumerated lab semantics |
| **Commerce application** | Rows in `commerce.*` driven by workflow ports (future) |
| **External SoR simulator** | Rows in `external_sor.*` seeded/updated by provisioning (future)—not commerce writes on canonical path |
| **Integrax** | Reliability processing on observability spine; evidence refs may point to spine artifacts |
| **This schema** | No workflow, APIs, ORM, or reconciliation algorithms |

---

## 5. UNKNOWN and truth separation

- `application_knowledge.known_status` allows `UNKNOWN`, `CONFIRMED`, `FAILED`.
- `application_knowledge_unknown_explicit_check` requires `uncertainty_explicit = true` when status is `UNKNOWN`.
- `external_reality.terminal_outcome` uses SoR vocabulary (`PAYMENT_COMPLETED`, `PAYMENT_FAILED`, `TRUTH_INDETERMINATE`)—not application `known_status` values.
- **UNKNOWN is not FAILURE:** distinct CHECK domains; no constraint maps timeout or silence to `FAILED` in application knowledge.

---

## 6. Limitations

- No seed data, provisioning adapter, or runtime loaders in this task.
- No indexes beyond primary/unique keys implied by constraints.
- Inventory and communication-event logical entities are not separate tables; reservation knowledge may be stored on `application_knowledge.inventory_reservation_knowledge` until a future slice requires dedicated tables.
- Schema not validated against a live Docker instance in unit tests—static contract tests only.

---

## 7. Validation

Unit tests: `tests/unit/platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/test_erl_qual_004_postgres_schema.py`  
Static helpers: `database/contract.py`

---

## Document validation

| Check | Result |
| --- | --- |
| Matches approved data model architecture entity set | **Pass** |
| External Reality ≠ Application Knowledge (separate tables/schemas) | **Pass** |
| UNKNOWN explicit in application knowledge | **Pass** |
| No ORM, APIs, or workflow logic claimed | **Pass** |
| Production readiness not claimed | **Pass** |
