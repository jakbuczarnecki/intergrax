# GCA OD-R2-03 — Docker Compose local qualification

**Scope:** `DOCKER-LOCAL-VERIFIED` evidence only when an operator runs the steps below on Docker Desktop (Linux containers). This project does **not** replace the ratified Linux VM / systemd production profile and is **not** operator acceptance or `TARGET-VERIFIED` proof.

**Documentation-only milestones** (not runtime enums): `CONFIGURED` (repo artifacts) → `A1-PREPARED` (static audit + non-executing `compose config -q`; **this task**) → `A6-RUNTIME-VERIFIED` (future authorized `build` / `run` / `serve`) → `OPERATOR-PROFILE-QUALIFIED` (Package B + OP-01…03 on real qual host) → `PRODUCTION-ACCEPTED` (out of scope). Legacy RUNBOOK label `DOCKER-LOCAL-VERIFIED` applies only after operator-run Docker steps below — **not** after A1 alone.

**Qualification states (OD-R2-03):** `CONFIGURED` → `REPO-TESTED` (pytest gates) → `DOCKER-LOCAL-VERIFIED` (operator-run Docker under **A6**) → `TARGET-VERIFIED` / `OPERATOR-ACCEPTED` (out of scope here).

## A1 vs A6 boundary

| Stage | Authorized in A1 | Authorized in A6 (separate) |
|-------|------------------|-----------------------------|
| `docker compose config -q` | Yes (non-executing) | Yes |
| `docker compose build` | **No** | Yes (when authorized) |
| `docker compose run` / `up` / `start` | **No** | Yes (when authorized) |
| Strict `serve` / live activation | **No** | Yes only with OP-01…03 + activation window |
| `docker compose down -v` | **No** (destructive; operator confirmation) | Same |

Missing operator policy, trusted roots, credentials, and activation remain **missing** after A1 — no LAB credential fallback.

## Prerequisites

- Docker Desktop with Linux container mode.
- Materialized GCA runtime image context (canonical builder):

```bash
uv run python scripts/build/build_application_image.py \
  --application governed_contractor_application \
  --tag governed_contractor-application:latest \
  --context-dir applications/governed_contractor_application/docker/runtime-context \
  --keep-context
```

- Optional: copy `intergrax-gca-docker-local-qualification.env.example` → `intergrax-gca-docker-local-qualification.env` (never commit secrets).

## Build qualification overlay image

From repository root:

```bash
docker compose -f applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml \
  -p intergrax-gca-docker-local-qualification build
```

## CONFIGURED verification (no strict serving)

Runs the sanctioned admission gate only (`acquire_cross_process_writer=False`). Creates durable layout on the named volume.

```bash
docker compose -f applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml \
  -p intergrax-gca-docker-local-qualification run --rm \
  gca-strict-local-qualification --configured-verify
```

## Informational storage probe (not admission)

```bash
docker compose -f applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml \
  -p intergrax-gca-docker-local-qualification run --rm \
  gca-strict-local-qualification --storage-probe
```

Ext4 / VM mount guarantees remain **TARGET-VERIFIED** on the operator Linux VM (`target-storage-probe.sh` on VM). Docker overlay filesystem is not claimed as production ext4.

## Strict serving (Package B required)

Without authorized operator governance wiring, serving must fail closed:

```text
BLOCKED — PACKAGE B OPERATOR SERVE CONFIGURATION REQUIRED
```

Default Compose sets `INTERGRAX_GCA_STRICT_SERVE_EXEC` to the sanctioned in-repo module `governed_contractor_application.host.package_b_strict_production_serving_command` (profile v1 opener + D4 uvicorn factory). Operator governance bundle path, tenant, API keys, and provider pins remain **out-of-repo** — startup fails closed until supplied (same contract as `deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production-entrypoint.sh`).

Do **not** use `governed_contractor_application.host.main:app` as a qualification substitute.

## Persistence and writer lock

- Durable state: Docker named volume `intergrax_gca_strict_local_qualification_data` → `/var/lib/intergrax/production-v1/data`.
- SQLite databases, WAL files, continuation seal, and `.intergrax-single-node-production-v1.writer.lock` must share this mount.
- Exactly **one** Compose service replica; a second container mounting the same volume can contend on the native file lock (expected fail-closed).

## Shutdown

```bash
docker compose -f applications/governed_contractor_application/deploy/docker-compose-local-qualification/docker-compose.yml \
  -p intergrax-gca-docker-local-qualification stop
```

`stop_grace_period: 120s` aligns with the systemd unit `TimeoutStopSec=120` (lifecycle drain — not reimplemented here).

## Cleanup (operator authorization only)

Deleting qualification volumes is destructive. Do not run `docker compose down -v` unless this isolated project (`-p intergrax-gca-docker-local-qualification`) is confirmed.

## Evidence capture

Redirect logs under `.tmp/session/<task-id>/` per repository session-artifact rules.
