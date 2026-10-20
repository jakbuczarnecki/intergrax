# GCA OD-R2-03 — Docker Compose local qualification

**Scope:** `DOCKER-LOCAL-VERIFIED` evidence only when an operator runs the steps below on Docker Desktop (Linux containers). This project does **not** replace the ratified Linux VM / systemd production profile and is **not** operator acceptance or `TARGET-VERIFIED` proof.

**Qualification states:** `CONFIGURED` (repo artifacts) → `REPO-TESTED` (pytest gates) → `DOCKER-LOCAL-VERIFIED` (operator-run Docker) → `TARGET-VERIFIED` / `OPERATOR-ACCEPTED` (out of scope here).

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

When Package B is supplied out-of-repo, set `INTERGRAX_GCA_STRICT_SERVE_EXEC` to a command that calls `open_gca_strict_production_serving_session_from_single_node_production_profile_v1` and the D4 uvicorn factory (same contract as `deploy/linux/single-node-production-profile-v1/intergrax-gca-strict-production-entrypoint.sh`).

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
