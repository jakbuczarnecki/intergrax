# Sentry observability — Local Workspace

This document describes the optional Sentry issue-triage proof for LKW controlled problem signals exported through the platform observability vendor contract.

## What this proves

LKW emits a controlled platform problem signal through the shared observability path into Sentry:

```text
Controlled LKW problem proof
→ PlatformProblemSignal / ProblemReporter
→ ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ObservabilityVendorIntegrationContract
→ ObservabilityVendorPayload (PROBLEMS)
→ Sentry provider transport
→ Sentry issue-shaped event
→ Sentry UI
```

This is local proof/dev wiring, not production Sentry hardening.

## Stack / required DSN

This proof requires a Sentry project DSN. It does **not** start a self-hosted Sentry stack.

Set:

```text
INTERGRAX_SENTRY_DSN=<your project DSN>
```

Do not commit real DSNs. Sentry receives policy-safe metadata only; raw content export remains disabled.

## Start with Docker Compose overlay

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

The overlay configures LKW to export to Sentry via `LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_*` env vars.

## Run controlled Sentry proof helper

From repository root with Sentry env configured:

```bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat
```

or:

```bash
applications/local_workspace_application/scripts/run-sentry-observability-proof.sh
```

Optional ids:

```bash
applications/local_workspace_application/scripts/run-sentry-observability-proof.sh \
  --run-id lkw-sentry-proof-demo \
  --correlation-id corr-demo-1
```

Expected helper output:

```text
proof_result=PASS
backend=sentry
problem_kind=lkw.proof_controlled_failure
problem_error_code=LKW_PROOF_CONTROLLED_FAILURE
run_id=<...>
correlation_id=<...>
sentry_event_sent=true
safety_check=passed
```

The helper does not print DSNs, secrets, or full event payloads.

## How to view in Sentry UI

Open your Sentry project **Issues** view and search by tags from the proof output.

Suggested filters:

```text
tag:intergrax.problem_kind:lkw.proof_controlled_failure
tag:intergrax.problem_error_code:LKW_PROOF_CONTROLLED_FAILURE
tag:intergrax.run_id:<run_id>
tag:intergrax.correlation_id:<correlation_id>
```

Expected event title/message:

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

- auth/DSN secret management
- alert routing and ownership
- dashboards/runbooks
- retention policy
- CI live proof automation

## Related docs

- Platform proof narrative: [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md)
- Kibana/Elasticsearch proof: [`KIBANA_OBSERVABILITY.md`](KIBANA_OBSERVABILITY.md)
- Observability architecture: [`docs/architecture/OBSERVABILITY.md`](../../../docs/architecture/OBSERVABILITY.md)
