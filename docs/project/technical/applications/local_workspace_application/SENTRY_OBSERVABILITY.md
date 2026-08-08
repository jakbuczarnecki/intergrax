# © Artur Czarnecki. All rights reserved.

# Sentry observability — Local Workspace

This document describes the local Docker Compose Sentry proof for LKW controlled problem signals exported through the platform observability vendor contract.

## What this proves

LKW emits a controlled platform problem signal through the shared observability path into a **local Sentry stack**:

```text
LKW controlled failure (proof endpoint)
→ platform ProblemReporter
→ ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ObservabilityVendorIntegrationContract
→ ObservabilityVendorPayload (PROBLEMS)
→ Sentry provider transport
→ local Sentry Relay/Web
→ Sentry issue in local UI
```

This is local proof/dev wiring, not production Sentry hardening.

## Local stack

The Sentry overlay is repo-owned and starts a trimmed self-hosted Sentry **24.8.0** proof stack:

```text
applications/local_workspace_application/docker/docker-compose.sentry.yml
applications/local_workspace_application/docker/sentry.services.yml
```

Services (pinned images):

```text
sentry-nginx      → http://127.0.0.1:9000   (Sentry UI)
sentry-relay      → internal ingest (DSN target for LKW container)
sentry-web        → Sentry application
postgres/redis/kafka/clickhouse → required Sentry backend
sentry-snuba-bootstrap → Snuba/ClickHouse storage bootstrap (one-shot)
sentry-snuba-api  → Snuba query API (after bootstrap)
sentry-upgrade    → DB migrations (runs once before sentry-web)
sentry-bootstrap  → proof org/project + local DSN env (after sentry-web healthy)
```

Canonical proof does **not** require an external SaaS DSN or `INTERGRAX_SENTRY_DSN` on the host.

## Start LKW + local Sentry (canonical one-script path)

From repository root, start **all** local proof services (base LKW, Qdrant, Ollama, OTel, Elasticsearch, Kibana, and local Sentry):

Windows:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat
```

Linux/macOS:

```bash
chmod +x applications/local_workspace_application/scripts/run-local-docker-all.sh
applications/local_workspace_application/scripts/run-local-docker-all.sh
```

These helpers auto-discover every top-level overlay matching `docker-compose.*.yml` in `applications/local_workspace_application/docker/`. Internal fragments such as `sentry.services.yml` are included by `docker-compose.sentry.yml` and are **not** discovered directly.

Pass through Docker Compose commands when needed, for example:

```bash
applications/local_workspace_application/scripts/run-local-docker-all.sh down -v
applications/local_workspace_application/scripts/run-local-docker-all.sh ps
```

### Manual compose path (alternative)

```bash
docker compose \
  -f applications/local_workspace_application/docker/docker-compose.yml \
  -f applications/local_workspace_application/docker/docker-compose.sentry.yml \
  up --build
```

Windows:

```bat
docker compose ^
  -f applications\local_workspace_application\docker\docker-compose.yml ^
  -f applications\local_workspace_application\docker\docker-compose.sentry.yml ^
  up --build
```

## First start

First start can take **several minutes**. The Sentry stack is resource-heavy (Postgres, Redis, Kafka, ClickHouse, Snuba, Relay, workers). Startup order:

```text
postgres/redis/kafka/clickhouse healthy
→ sentry-snuba-bootstrap (Snuba/ClickHouse storage bootstrap)
→ sentry-snuba-api
→ sentry-upgrade (migrations)
→ sentry-web healthy
→ sentry-bootstrap (proof account + generated.env)
→ local_workspace (sources /proof/generated.env at process start)
```

ClickHouse/Snuba may emit noisy background warnings during warm-up. Missing-table errors (for example `generic_metric_counters_aggregated_local` or `metrics_counters_v2_local`) should **not** persist after `sentry-snuba-bootstrap` completes successfully.

## Sentry UI URL

```text
http://127.0.0.1:9000
```

## Initial login/bootstrap

`sentry-bootstrap` creates a local proof account and project after `sentry-upgrade` has run migrations and `sentry-web` is healthy:

```text
email:    admin@intergrax.local
password: proof-local-only
org:      intergrax-local
project:  lkw-proof
```

These credentials are for **local proof only**. Do not reuse in production.

The shared Sentry services use a deterministic local-proof secret key (`SENTRY_SECRET_KEY` default: `intergrax-local-sentry-proof-secret-key-not-for-production`). This is **not production-safe** and must not be reused outside this local proof stack.

`sentry-relay` runs in managed mode and requires `credentials.json` beside `config.yml`. The repo ships a **local-proof-only** Relay credential at:

```text
applications/local_workspace_application/docker/sentry/relay/credentials.json
```

It was generated once with `getsentry/relay:24.8.0 credentials generate` and is mounted read-only into the relay container. **Not production-safe** — do not reuse outside this stack. To regenerate (only if needed): run the same command against that directory with `--overwrite`.

Bootstrap writes the local DSN for the LKW container to:

```text
applications/local_workspace_application/docker/sentry-proof/generated.env
```

That path is **local proof runtime state** — created or overwritten atomically by `sentry-bootstrap` (`generated.env.tmp` → `generated.env`). It is listed in `.gitignore` together with `.bootstrapped`; do not commit bootstrap output or external SaaS DSNs.

Docker Compose resolves `env_file` when creating the container, before `depends_on` completes, so the Sentry overlay does **not** use `env_file` for `generated.env`. Instead, `local_workspace` mounts `./sentry-proof` read-only at `/proof` and `start-local-workspace-sentry-proof.sh` sources `/proof/generated.env` immediately before `uvicorn` starts. If the file is missing, the container exits with a clear local-proof error.

The committed template is:

```text
applications/local_workspace_application/docker/sentry-proof/generated.env.placeholder
```

## Run controlled LKW error proof

With the stack running:

```bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat
```

or:

```bash
applications/local_workspace_application/scripts/run-sentry-observability-proof.sh
```

The helper calls:

```text
POST /v1/local_workspace/proof/sentry-error
```

Expected helper output:

```text
proof_result=PASS
backend=sentry
sentry_mode=local_docker
sentry_ui=http://127.0.0.1:9000
problem_kind=lkw.proof_controlled_failure
problem_error_code=LKW_PROOF_CONTROLLED_FAILURE
run_id=<...>
correlation_id=<...>
sentry_event_sent=true
safety_check=passed
sentry_search_hint=tag:intergrax.problem_kind:lkw.proof_controlled_failure
```

The helper does not print DSNs, secrets, or full event payloads.

## View issue in local Sentry UI

Open **Issues** at `http://127.0.0.1:9000` and search by tags from the proof output.

Suggested filters:

```text
tag:intergrax.problem_kind:lkw.proof_controlled_failure
tag:intergrax.problem_error_code:LKW_PROOF_CONTROLLED_FAILURE
tag:intergrax.run_id:<run_id>
tag:intergrax.correlation_id:<correlation_id>
```

Expected issue title/message:

```text
Intergrax problem: lkw.proof_controlled_failure
```

## Safety expectations

The proof exports metadata only:

- no raw prompt
- no chunks
- no tool args
- no file contents
- no full local paths
- no secrets
- no PII/default user capture

`LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_CONTENT` remains `false`.

## Local-only scope

This overlay is for local proof and developer triage. Production gaps remain:

- auth/TLS and secret management
- alert routing and ownership
- dashboards/runbooks
- retention policy
- CI live proof automation
- production Sentry sizing/hardening

## Optional external DSN override

An external SaaS DSN may be supplied only as an advanced operator override through `LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN`. It is **not** the canonical LKW proof path.

## Related docs

- Platform proof narrative: [`docs/project/proofs/LKW_PLATFORM_PROOF.md`](../../../proofs/LKW_PLATFORM_PROOF.md)
- Kibana/Elasticsearch proof: [`KIBANA_OBSERVABILITY.md`](KIBANA_OBSERVABILITY.md)
- Observability architecture: [`docs/project/architecture/OBSERVABILITY.md`](../../../architecture/OBSERVABILITY.md)
