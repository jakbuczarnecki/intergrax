# NPSC-5F/R3 — Governed Evidence Export

> **Status:** PASS / IMPLEMENTATION COMPLETE (not FROZEN — await R3 Final)

## Purpose

Close **OBS-03**: eliminate raw `RuntimeEvent.model_dump` from supported journal/OTLP/logging export paths; enforce allowlisted `ObservabilityExportEnvelope` projection before any external sink.

## Provenance

| Artifact | SHA / version |
| --- | --- |
| R1 Final | `455c09f342f995ac0a6fcb03ffef2f4d3e36a447` |
| R2 implementation | `632507420f0ab8360aede43a2740e8fccc44efb4` |
| R2 Final | `76c92847f67da22d97943b55896a88c814d7e39d` |
| NPSC-5E Final | `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7` |
| R3 baseline (branch `development`) | recorded at implementation start |

## Parallel-session ownership

- **Session A (R3):** export projection, journal export safety, OTLP/logging boundaries, qualification.
- **Sessions B/C/D:** execution, scale, certification — must not regress R1 durability or R2 journal read contracts.

## Pre-R3 bypass

`serialize_runtime_event` returned `event.model_dump(mode="json")`, leaking arbitrary `payload` into `JournalExportSnapshot`, logger `extra`, and OTLP JSON.

## Canonical export flow

```text
RuntimeEvent
  → runtime_event_export_source_from_event (allowlisted payload)
  → RuntimeEventExportSource (+ safe top-level trace/phase metadata)
  → envelope_from_runtime_event_source
  → ObservabilityExportEnvelope
  → envelope_is_content_safe (structural key scan)
  → export sinks (OTLP JSON, logger, ObservabilityExporter plugins)
```

## Safe projection contract

- Single allowlist: `_SAFE_RUNTIME_EVENT_PAYLOAD_KEYS` in `export_boundary.py`.
- Unknown payload keys: **not exported**.
- Nested arbitrary payload: **not exported** (no generic metadata bag on envelope).

## Allowlisted runtime payload fields

`tool_id`, `capability`, `latency_ms`, `duration_ms`, `hit_count`, `error_code`, `policy_rule_id`, `args_digest`, `collection_id`, `payload_schema_id`, `schema_id`, `event_count`, `parser_trace_count`, `status`.

## Forbidden content model

`FORBIDDEN_EXPORT_CONTENT_FIELDS` — structural key prohibition via `envelope_is_content_safe` (field names, not value substring heuristics).

## Journal snapshot contract

- `JournalExportSnapshot.events`: `tuple[ObservabilityExportEnvelope, ...]`
- Schema: `journal_export.v2` (semantic change from v1 raw dict rows).
- `is_complete` / `has_continuation`: explicit bounded snapshot semantics (R2 page contract).

## Bounded export semantics

`build_journal_export_snapshot` uses `read_run_journal_page` only (never `load_complete_run_journal` for export).

## Completeness interaction with R2

Export may be a **partial page** when `has_continuation=True`; consumers must not treat as full run history.

## OTLP projection

`render_journal_otlp_json(JournalExportSnapshot)` — typed input only; span attributes from envelope fields.

## Logging safety

`make_journal_export_runtime_plugin` logs `snapshot.to_dict()` where events are envelope JSON only.

## Exporter plugin boundary

`ObservabilityExporter.export(ObservabilityExportEnvelope)` unchanged; sink-neutral safety before export.

## Extension safety

`envelope_with_observability_extensions` + `apply_observability_export_policy` retain sanitized application attributes; forbidden application fields dropped.

## Schema compatibility/version decision

**Bump to `journal_export.v2`**. Security correctness dominates legacy raw-row compatibility.

## Tenant/identity preservation

`tenant_id`, `event_id`, `run_id`, `task_id`, `attempt_id`, `execution_id` preserved on projection without synthesis.

## R1/R2 preservation

No changes to persistence append, tenant routing, `RunJournalReadPage`, or store pagination.

## R4 source-of-truth separation

Durable evidence remains `RuntimeEventPersistence`; export envelopes are lossy observability projections, not replay authority.

## Regression matrix

| Gate | Result |
| --- | --- |
| R3 unit architecture suite | PASS |
| `tests/unit/runtime/observability/**` | PASS |
| P0 reconciliation (OBS-03) | PASS |
| R1/R2 Final (mandatory subprocess matrix) | run at R3 Final |

## Security canary tests

`SECRET_PROMPT_123`, `SECRET_TOKEN_456`, `RAW_TOOL_ARGS_789`, `PRIVATE_BODY_ABC`, `DO_NOT_EXPORT_123` absent from envelope, snapshot, OTLP, and logger extra.

## Static quality

`ruff` / `pyright` on changed production modules — zero new errors required.

## Final verdict

**NPSC-5F/R3: PASS / IMPLEMENTATION COMPLETE** — raw runtime payload cannot traverse supported journal export, OTLP, or logging paths.

**Next:** NPSC-5F/R3 Final — qualification freeze and protected drift.
