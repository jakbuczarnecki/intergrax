# Kibana observability UI — Local Workspace

This document describes the optional local Kibana UI for inspecting LKW observability documents stored in Elasticsearch.

## Stack

Kibana is included in the Elasticsearch observability Docker Compose overlay:

```text
applications/local_workspace_application/docker/docker-compose.elasticsearch.yml
```

The overlay starts:

```text
elasticsearch  → http://127.0.0.1:9200
kibana         → http://127.0.0.1:5601
```

Both services use the same pinned Elastic version in the local proof stack:

```text
8.15.3
```

## Start

From repository root on Windows:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat
```

Or manually:

```bash
docker compose \
  -f applications/local_workspace_application/docker/docker-compose.yml \
  -f applications/local_workspace_application/docker/docker-compose.elasticsearch.yml \
  up --build
```

Open Kibana:

```text
http://127.0.0.1:5601
```

## Data view

Create a Kibana data view for the LKW observability index:

```text
Name / index pattern: intergrax-lkw-observability
Timestamp field: @timestamp
```

Then use **Discover** to inspect events.

## Useful filters

Filter by run id:

```text
intergrax.run_id: "run_d28d5f36f5ca4240b8693ae46eaa5946"
```

Filter by event type:

```text
intergrax.event_type: "tool_requested"
intergrax.event_type: "tool_completed"
intergrax.event_type: "task_completed"
```

Filter by tool:

```text
intergrax.tool_id: "rag.retrieve"
```

Filter by agent:

```text
intergrax.agent_id: "local_search"
```

## Suggested Discover columns

Add these columns in Kibana Discover:

```text
@timestamp
intergrax.run_id
intergrax.event_id
intergrax.event_type
intergrax.agent_id
intergrax.tool_id
intergrax.capability
intergrax.status
```

## Relationship to proof helper

Kibana is for visual exploration and dashboards.

The proof helper remains the canonical repeatable CLI validation tool:

```bat
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat run_...
```

Use Kibana to inspect and understand runs. Use the helper to prove:

```text
duplicate_check=0
safety_check=passed
```

## Local-only scope

This is a local proof/dev UI only. The overlay does not add:

```text
auth/TLS
production index lifecycle management
dashboards as code
alerting
retry/backoff
dead-letter handling
```

Those remain separate production hardening concerns.
