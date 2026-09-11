# NPSC-5F/R3 Final — Governed Evidence Export Qualification and Freeze

**Status:** `FROZEN / PASS`

**Task:** NPSC-5F/R3 Final — Governed Evidence Export Qualification and Freeze

---

## Purpose

Formal enterprise export-security freeze: authoritative `RuntimeEvent` evidence stays internal; every supported journal, logger, OTLP, and external exporter path must project through the single typed `RuntimeEventExportSource → ObservabilityExportEnvelope` boundary with allowlisted metadata, structural content-safety validation, and bounded `JournalExportSnapshot` (`journal_export.v2`). Raw runtime payload serialization is not a supported export path.

Invariant:

> **No supported journal, logger, OTLP, or exporter path may receive raw `RuntimeEvent` payload; execution evidence leaving the evidence plane must pass one typed, allowlisted, redacted-by-default `ObservabilityExportEnvelope`, while canonical durable evidence remains the sole source of truth.**

---

## Provenance

| Label | SHA |
| ----- | --- |
| NPSC-5F/R2 Final | `76c92847f67da22d97943b55896a88c814d7e39d` |
| NPSC-5F/R2 implementation | `632507420f0ab8360aede43a2740e8fccc44efb4` |
| NPSC-5F/R1 Final | `455c09f342f995ac0a6fcb03ffef2f4d3e36a447` |
| NPSC-5F/P0 | `7811371da1069b661987b050a4c9bf42c02bda69` |
| NPSC-5E Final | `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7` |

---

## Implementation SHA

Commit: `0346face3ef68d8f21504822a26f8f45f2384cf9` — R3 governed evidence export (OBS-03).

| Module | R3 role |
| ------ | ------- |
| `export_boundary.py` | `ObservabilityExportEnvelope`, `RuntimeEventExportSource`, allowlist projection, `envelope_is_content_safe` |
| `journal_export.py` | `journal_export.v2`, `JournalExportSnapshot`, `serialize_runtime_event`, OTLP JSON |
| `export_bridge.py` | Logger plugin — safe snapshot only |

**Production code changed in R3 Final task:** NO (qualification, drift helper, final gate, docs only).

---

## Parallel-session reconciliation

Sessions B (Platform Execution Unification), C (Enterprise Scale & Resilience), and D (Execution Certification Acceleration) may advance `origin/development` during Final.

Drift classification `0346face..origin/development` at qualification: Session C dependency admission + docs only — **no R3-protected export contract overlap** (`PARALLEL DRIFT: ALLOWED / NON-OVERLAPPING`).

P0 post-R2 R1 drift sentinel advanced to `40cc8c11e0b57ed4cf0d99ed1b9b297820c6eaa8` (qualified `RuntimeEventType.EXECUTION_FAILED` — DIAG execution failure evidence R2); no other R1-protected paths changed between that baseline and Final sign-off.

**EXECUTION_FAILED compatibility (H1):** EventId semantics, tenant routing, persistence, journal ordering, pagination, export filtering, redaction, and reconstruction compatibility — PASS via `test_npsc5f_r3_final_h1_upstream_runtime_event_drift_reconciliation.py` on integrated HEAD `2965f2fcfed27162f06625c4b8bcd84b18d27704`.

R3 drift sentinel (`testing_support/npsc5f_r3_protected_drift.py`) scopes **exact R3 export surfaces** and does **not** file-freeze all of `runtime/observability/**` (R4 reconstruction may touch adjacent modules).

---

## R3 protected contracts

Post-implementation ownership (semantic + path-scoped drift):

- `ObservabilityExportEnvelope`, `RuntimeEventExportSource`
- `runtime_event_export_source_from_event`, `envelope_from_runtime_event_source`, `envelope_from_runtime_event`, `envelope_is_content_safe`
- `JournalExportSnapshot`, `JOURNAL_EXPORT_SCHEMA_VERSION`, `build_journal_export_snapshot`, `serialize_runtime_event`, `render_journal_otlp_json`, `make_journal_export_runtime_plugin`

Protected production paths: `export_boundary.py`, `journal_export.py`, `export_bridge.py`.

---

## Pre-R3 raw bypass

Historical gap: `serialize_runtime_event` / journal export used full `RuntimeEvent.model_dump`, bypassing redaction. **Closed in R3 implementation** — static proof: zero supported `RuntimeEvent.model_dump` / `event.model_dump` on export surfaces.

---

## Canonical safe export flow

```text
RuntimeEvent
  → runtime_event_export_source_from_event
  → RuntimeEventExportSource
  → envelope_from_runtime_event_source
  → ObservabilityExportEnvelope
  → envelope_is_content_safe (defense-in-depth)
  → JournalExportSnapshot / logger / OTLP / ObservabilityExporter
```

---

## Payload allowlist

Unknown runtime payload keys **denied by default** via `_SAFE_RUNTIME_EVENT_PAYLOAD_KEYS` in `export_boundary.py` (single authoritative allowlist).

---

## Unknown-field deny-by-default

Future keys (e.g. `future_super_secret`, nested dict/list carriers) absent from export projections.

---

## Structural content-safety semantics

`envelope_is_content_safe` — recursive forbidden-key structural validation against `FORBIDDEN_EXPORT_CONTENT_FIELDS`. **Not** general secret detection or value-based DLP.

---

## Journal export v2

`JOURNAL_EXPORT_SCHEMA_VERSION = journal_export.v2` — typed `ObservabilityExportEnvelope` collection; no supported raw v1 production path (`journal_export.v1` absent from production observability tree).

---

## Boundedness / completeness

`build_journal_export_snapshot` → `read_run_journal_page` (R2 semantics preserved). `is_complete` / `has_continuation` explicit; no false “full run export” claims without `is_complete=True`.

---

## Logger safety

`make_journal_export_runtime_plugin` logs safe snapshot extras only; security canaries occurrence **0**.

---

## OTLP safety

`render_journal_otlp_json(JournalExportSnapshot)` — typed input only; no public arbitrary `Mapping[str, Any]` journal escape hatch.

---

## Exporter protocol

`ObservabilityExporter.export(ObservabilityExportEnvelope)` — provider-neutral; sink-independent projection before sink.

---

## Extension safety

`envelope_with_observability_extensions` / `apply_observability_export_policy` — extensions cannot reintroduce forbidden raw content; post-policy envelope passes `envelope_is_content_safe`.

---

## Tenant / identity preservation

`event_id`, `run_id`, `task_id`, `attempt_id`, `execution_id`, `tenant_id`, `parent_event_id`, W3C trace fields preserved without synthetic `"unknown"` fallbacks. Cross-tenant export isolation enforced.

---

## Schema migration

Supported journal export schema: **v2 only** for safe typed snapshots.

---

## Legacy v1 rejection

Supported unsafe v1 export builder count: **0** in production observability.

---

## R1 preservation

R1 Final gate PASS — export success ≠ evidence persistence success.

---

## R2 preservation

R2 Final gate PASS — page completeness, snapshot bounds, concurrent append semantics unchanged in export consumption.

---

## R4 source-of-truth separation

Execution reconstruction consumes durable evidence, not lossy export envelopes.

---

## Security canary matrix

Canaries: `SECRET_PROMPT_123`, `SECRET_TOKEN_456`, `RAW_TOOL_ARGS_789`, `PRIVATE_BODY_ABC`, `DO_NOT_EXPORT_123`, `FUTURE_SECRET_FIELD_X` — injected at top-level forbidden keys, unknown keys, nested dict/list; expected serialized occurrence across export surfaces: **0**.

---

## Static architecture checks

- No `RuntimeEvent` raw `model_dump` on export surfaces
- No second export framework symbols
- No execution-control invocation from R3 export modules
- No supported `journal_export.v1` production path
- R3 drift classifier tests

---

## Regression matrix

| Gate | Role |
| ---- | ---- |
| `test_npsc5f_r3_final_governed_evidence_export.py` | Canonical R3 Final freeze |
| `test_npsc5f_r3_governed_evidence_export.py` | R3 implementation |
| `test_npsc5f_r2_final_*` | R2 frozen regression |
| `test_npsc5f_r1_final_*` | R1 frozen regression |
| `test_npsc5f_p0_*` | P0 reconciliation |
| `tests/unit/runtime/observability/**` | Observability adjacency |
| NPSC-5E / 5D Final, DG_001, TRACE slices, reconstruction | Predecessors |

---

## Final verdict

**R3 contract qualification:** PASS on implementation SHA `0346face` with **no unqualified R3-protected drift** since implementation through Final sign-off.

**OBS-03:** FIXED / FROZEN R3.

---

## Freeze statement

> Authoritative execution evidence remains internal `RuntimeEvent` data persisted by the canonical evidence store. Any supported journal, logging, OTLP or external observability path must project runtime events through the single typed `RuntimeEventExportSource → ObservabilityExportEnvelope` boundary before reaching a sink. Runtime payload fields are denied by default and only explicitly allowlisted safe metadata may leave the evidence plane. `journal_export.v2` contains typed safe envelopes and explicit bounded completeness metadata; raw RuntimeEvent serialization is not a supported export path. Structural forbidden-key validation remains defense-in-depth and is not a general value-based secret scanner. Export remains lossy observability data and never acquires execution, durability, lineage, authority, governance, checkpoint, recovery, ordering or replay ownership.

**Next:** NPSC-5F/R4 — Reconstruction / As-of / Bitemporal
