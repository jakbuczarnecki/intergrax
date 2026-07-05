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
applications/local_workspace_application/docker/docker-compose.sentry.services.yml
```

Services (pinned images):

```text
sentry-nginx      → http://127.0.0.1:9000   (Sentry UI)
sentry-relay      → internal ingest (DSN target for LKW container)
sentry-web        → Sentry application
postgres/redis/kafka/clickhouse/snuba → required Sentry backend
sentry-bootstrap → migrations + proof org/project + local DSN env
```

Canonical proof does **not** require an external SaaS DSN or `INTERGRAX_SENTRY_DSN` on the host.

## Start LKW + local Sentry

From repository root:

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

First start may take several minutes while Sentry migrations and bootstrap complete.

## Sentry UI URL

```text
http://127.0.0.1:9000
```

## Initial login/bootstrap

`sentry-bootstrap` creates a local proof account and project:

```text
email:    admin@intergrax.local
password: proof-local-only
org:      intergrax-local
project:  lkw-proof
```

These credentials are for **local proof only**. Do not reuse in production.

Bootstrap writes the local DSN for the LKW container to:

```text
applications/local_workspace_application/docker/sentry-proof/generated.env
```

Do not commit real external DSNs. The generated file contains only the local proof DSN.

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

- Platform proof narrative: [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md)
- Kibana/Elasticsearch proof: [`KIBANA_OBSERVABILITY.md`](KIBANA_OBSERVABILITY.md)
- Observability architecture: [`docs/architecture/OBSERVABILITY.md`](../../../docs/architecture/OBSERVABILITY.md)
