---
qualification_id: ERL-QUAL-004
scenario_slug: enterprise_payment_uncertainty_recovery
document_type: postgresql_infrastructure
lifecycle: IMPLEMENTATION_FOUNDATION
status: DOCUMENTED
---

# ERL-QUAL-004 — PostgreSQL Infrastructure Foundation

**Enterprise Payment Uncertainty Recovery**

[← Public scenario page](../README.md) · [Vendor Infrastructure Architecture](ERL_QUAL_004_VENDOR_INFRASTRUCTURE_ARCHITECTURE.md) · [Infrastructure README](../infrastructure/README.md)

---

## 1. Purpose

This document describes the **first implemented infrastructure layer** for the standalone E2E lab: a reproducible **PostgreSQL Docker** environment under `infrastructure/`. It hosts future materialized commerce state produced by a PostgreSQL provisioner — it does **not** define schema, migrations, or business logic.

Boundary preservation:

```text
Dataset (business truth)
        ≠
Provisioning (materialization lifecycle)
        ≠
Infrastructure (containers, PostgreSQL availability)
        ≠
Scenario application (workflow)
```

---

## 2. Why PostgreSQL

PostgreSQL was selected in [Vendor Infrastructure Architecture § 3](ERL_QUAL_004_VENDOR_INFRASTRUCTURE_ARCHITECTURE.md#3-database-decision) because ERL-QUAL-004 requires:

- transactional operational state for orders, payment uncertainty, and inventory alignment;
- relational constraints matching enterprise commerce patterns;
- reproducible lab environments via official container images.

Vector databases and event streaming remain **out of scope** for v1.

---

## 3. Container ownership

| Owner | Responsibility |
| --- | --- |
| **Infrastructure** (`infrastructure/`) | Compose project, pinned image, volume lifecycle, loopback port binding, health check, env contract |
| **Provisioning** (future) | Seed and tear down variant materialization in PostgreSQL via `ScenarioProvisioningPort` |
| **Application** (future) | Business ports and tools — reads/writes through contracts, not raw dataset files |
| **Dataset** | Vendor-neutral JSON under `dataset/` — unchanged when infrastructure changes |

---

## 4. Layout

```text
infrastructure/
├── config/
│   ├── postgres.env.example   # committed configuration contract
│   └── postgres.env             # operator copy (gitignored)
├── docker/
│   └── docker-compose.yml       # PostgreSQL service only (v1 foundation)
├── contract.py                  # static validation helpers
├── health_verification.py       # container + pg_isready verification
└── README.md                    # operator commands
```

---

## 5. Lifecycle

| Phase | Action |
| --- | --- |
| **Configure** | Copy `postgres.env.example` → `postgres.env`; set non-production credentials and host port |
| **Up** | `docker compose --env-file … -f infrastructure/docker/docker-compose.yml up -d` |
| **Verify** | `uv run python infrastructure/health_verification.py` — container running and `pg_isready` succeeds |
| **Down** | `docker compose … down` — stop container; `down -v` removes lab volume |

Compose project name: `erl-qual-004-postgres` (via `COMPOSE_PROJECT_NAME`). Named volume: `erl_qual_004_postgres_data`.

Image pin: **`postgres:16.6`** (no `latest` tag).

---

## 6. Configuration boundary

All environment-specific values live in `infrastructure/config/postgres.env` (from example):

| Variable | Role |
| --- | --- |
| `COMPOSE_PROJECT_NAME` | Isolated compose project for this scenario |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | PostgreSQL container bootstrap |
| `ERL_QUAL_004_POSTGRES_HOST_PORT` | Loopback-published host port (`127.0.0.1:…`) |

The compose file references `env_file` and does **not** embed secrets. Operators must not commit `postgres.env`.

---

## 7. Health verification

Infrastructure verification (not application health):

1. Configuration file present and contract-valid (`contract.py`).
2. Compose reports a running `postgres` service container.
3. In-container `pg_isready` for configured user and database.

Failure output uses an actionable `BLOCKED_ENVIRONMENT` message suitable for future `--validate-only` integration.

---

## 8. Current limitations

- **Lab only** — not production deployment, HA, backup, or cloud topology.
- **Database service only** — no scenario application, external payment boundary, or Integrax runtime services in compose yet.
- **No schema** — empty PostgreSQL instance; provisioning adapter and DDL are future tasks.
- **Docker required** for runtime verification; unit tests validate compose and config contracts without executing Docker.

---

## 9. References

| Artifact | Role |
| --- | --- |
| [ERL_QUAL_004_VENDOR_INFRASTRUCTURE_ARCHITECTURE.md](ERL_QUAL_004_VENDOR_INFRASTRUCTURE_ARCHITECTURE.md) | Architecture decisions and future compose stack |
| [ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md) | Provisioning boundary (future PostgreSQL adapter) |
| [infrastructure/README.md](../infrastructure/README.md) | Operator quick start |

---

**Implementation status:** PostgreSQL Docker foundation implemented. Provisioning adapter and full E2E compose stack remain future work.
