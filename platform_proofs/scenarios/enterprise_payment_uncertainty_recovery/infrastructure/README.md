# ERL-QUAL-004 — PostgreSQL infrastructure (lab)

Scenario-local Docker stack for the relational lab database. This layer owns **container lifecycle and availability only** — not dataset content, provisioning materialization, or application workflow.

## Quick start (repository root)

```powershell
Copy-Item `
  platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/config/postgres.env.example `
  platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/config/postgres.env
docker compose `
  --env-file platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/config/postgres.env `
  -f platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/docker/docker-compose.yml `
  up -d
uv run python `
  platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/health_verification.py
```

```bash
cp platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/config/postgres.env.example \
   platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/config/postgres.env
docker compose \
  --env-file platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/config/postgres.env \
  -f platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/docker/docker-compose.yml \
  up -d
uv run python \
  platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/health_verification.py
```

Teardown (removes the named volume — lab data only):

```bash
docker compose \
  --env-file platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/config/postgres.env \
  -f platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/infrastructure/docker/docker-compose.yml \
  down -v
```

## Documentation

See [ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md](../docs/ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md).
