# ERL-QUAL-004 — PostgreSQL schema

Lab relational schema for enterprise payment uncertainty recovery. Architecture source: [ERL_QUAL_004_POSTGRESQL_DATA_MODEL_ARCHITECTURE.md](../docs/ERL_QUAL_004_POSTGRESQL_DATA_MODEL_ARCHITECTURE.md).

## Layout

```text
database/
├── contract.py          # static validation helpers
├── migrations/          # ordered, reviewable DDL
├── schema/              # canonical snapshot (mirrors migration 001)
└── README.md
```

## Apply (lab)

After [infrastructure](../infrastructure/README.md) PostgreSQL is healthy:

```powershell
$env:PGPASSWORD = "<from postgres.env>"
psql -h 127.0.0.1 -p <ERL_QUAL_004_POSTGRES_HOST_PORT> -U <POSTGRES_USER> -d <POSTGRES_DB> `
  -f platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/database/migrations/001_erl_qual_004_core_schema.sql
```

Provisioning adapters (future) should apply migrations in lexical order—not hand-edit live databases.

## Zones

| Schema | Ownership (lab) |
| --- | --- |
| `commerce` | Application workflow state (orders, intents, application knowledge) |
| `external_sor` | Integration effect + authoritative external reality |
| `reconciliation` | Cases, investigation attempts, evidence refs, resolutions |
