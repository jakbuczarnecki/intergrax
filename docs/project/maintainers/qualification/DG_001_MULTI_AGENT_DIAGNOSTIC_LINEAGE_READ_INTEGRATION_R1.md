# DG-001 — Multi-agent diagnostic lineage read integration R1

> **Task:** `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1`  
> **Architecture:** `docs/project/maintainers/architecture/DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_ARCHITECTURE_R1.md`  
> **Architecture base:** `5e2fef428329608e2437c8d5a1da87ac0b6848c7`  
> **Write-side base:** `a18e65c077ed57bf3bb64ef015b47ee1f3ceb6bf`  
> **Public qualification base:** `156f1defb42235fb5c3d720a41f1fcb873ad542e`

## Baseline

- **START_HEAD:** `ba7c4dcca3b528d96dcd1880377330fd759b1455`
- **Branch:** `development`
- Ancestry verified for architecture and public qualification bases.

## Authority map

| Concern | Authority |
|---|---|
| Runtime lifecycle/events | `RuntimeEventPersistence` |
| Transport/cross-boundary causal | `CausalEvidencePersistence` |
| Forensic admission + segment continuity | `ExecutionLineageReader` / durable lineage store |
| Resume/checkpoint projection | `TaskCheckpoint` / `ExecutionTreeSnapshot` (not read here) |
| Derived forensic topology | `ExecutionReconstructor` |
| Operator read entry | `DiagnosticReadService` |

## Attempt discovery decision

> **Correction (2026-09-10):** `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-ATTEMPT-DISCOVERY-ARCHITECTURE-R1-ROLLOUT-CORRECTION`  
> **Architecture:** `docs/project/maintainers/architecture/DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_ATTEMPT_DISCOVERY_ARCHITECTURE_R1.md`

```text
READ_INTEGRATION_R1:
BLOCKED_PENDING_ATTEMPT_DISCOVERY_ARCHITECTURE
```

**Historical PASS (superseded):** attempt set was declared as `RuntimeEvent attempts UNION CausalEvidence attempts` with lineage as per-attempt enrichment only. Independent audit confirmed a legal counterexample: durable `ExecutionLineagePersistence` attempt state (`open_attempt` / `open_segment`) may exist before any RuntimeEvent or CausalEvidence for that `AttemptId`. That discovery model is **incomplete**.

**Current verdict:** `EXISTING_DIAG2_DISCOVERY_COMPLETE: NO`. Rollout semantics (legacy `discovery_contract_version=None` vs post-v1 index-first, atomic registration, run discovery stable snapshot) defined in attempt-discovery architecture R1 rollout correction. Implementation deferred to `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION`.

## Read-only port

- `ExecutionLineageReader` ABC with:
  - `list_admissions_for_attempt`
  - `list_segments_for_attempt`
  - `read_attempt_lineage_state`
  - `read_seal`
- `ExecutionLineagePersistence(ExecutionLineageReader, ABC)`
- Diagnostics depend on `ExecutionLineageReader | None` only.

## Segment read API + pagination

- `ExecutionLineageSegmentPage`
- `list_segments_for_attempt(scope, limit, cursor=None)`
- Shared `validate_lineage_page_limit` (admissions alias preserved)
- Conformance: `InMemoryExecutionLineagePersistence` + `DocumentStoreExecutionLineagePersistence`

## Reconstruction model

- `ReconstructedAttemptLineage`, `ReconstructedLineageSegment`
- `ExecutionLineageReadStatus`: AVAILABLE / ABSENT / UNAVAILABLE
- `ExecutionLineageCompleteness`: OPEN / COMPLETE / PARTIAL / TRUNCATED
- Bounded loader: `initial_lineage_page_limit`, `max_lineage_records`, cursor-cycle detection

## Forensic completeness semantics

- COMPLETE: sealed + seal present + not degraded + all pages loaded + structural validation passed (outcome may be FAILED/CANCELLED)
- OPEN: unsealed valid active attempt
- PARTIAL: degraded or SEGMENT_UNCLEAN
- TRUNCATED: bounded max reached before pagination completes
- UNAVAILABLE: backend read failure (runtime/causal reconstruction still available)

## Checkpoint non-authority

`ExecutionReconstructor` does not import or read checkpoint resume stores/projections for forensic parent edges.

## Operator read mapping

- `DiagnosticExecutionLineageView` and nested operator-safe DTOs
- Pure projector: `diagnostic_lineage_projection.project_execution_lineage_view`
- `DiagnosticProblemOccurrenceView.execution_lineage` (backwards-compatible default `None`)
- No prompts/payloads/storage cursors/partition keys exposed

## Provider composition

- `HostDiagnosticReadDependencies.execution_lineage_reader`
- Reuses `resolve_execution_lineage_persistence(...)` + same platform `document_store`
- Provider unset → lineage field `None`, prior behavior preserved
- Explicit provider + missing/incompatible store → fail closed

## Integrity matrix (fail closed)

Scope mismatch, missing segment root admission, duplicate execution admission, unknown segment membership, missing in-segment parent, broken segment continuity, cursor cycle.

## Tests

```bash
uv run pytest tests/unit/runtime/execution/lineage/test_execution_lineage_segment_read.py -q
uv run pytest tests/unit/runtime/diagnostics/test_execution_lineage_reconstruction.py -q
uv run pytest tests/unit/applications/test_diagnostic_read_lineage_wiring.py -q
uv run pytest tests/unit/runtime/diagnostics/test_execution_reconstruction.py tests/unit/runtime/diagnostics/test_lifecycle_analysis.py tests/unit/runtime/diagnostics/test_diagnostic_assessment.py tests/unit/runtime/diagnostics/test_diagnostic_read_service.py -q
uv run pytest tests/unit/runtime/execution/lineage/ -q
uv run pytest tests/integration/runtime/test_harden_4e_diagnostic_read_truth_e2e.py -q
```

## Out of scope

Write-side admission semantics, checkpoint resume engine, DIAG-3/DIAG-4 taxonomy changes, Decision System, scope discovery lineage provider, second execution tree/store.

## Audit trail — read integration corrections pending

Architecture `DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_ATTEMPT_DISCOVERY_ARCHITECTURE_R1` registers these implement corrections (not in scope of original READ_INTEGRATION_R1):

| Item | Required semantics |
| ---- | ------------------ |
| Attempt discovery | OPTION_C; `discovery_contract_version` legacy/post-v1 marker; index-first via `activate_root_execution_lineage` only |
| Run discovery snapshot | Run `generation`-guarded pagination; separate from per-attempt `generation` |
| Truncation | Truncated pages must not be validated as complete forensic snapshots |
| Stable snapshot | Per-attempt generation-guarded bounded retry; no torn cross-generation compose |
| State/seal consistency | `sealed` / `closure_kind` / `degraded` equality across state and seal after stable snapshot |
| Provider error translation | DocumentStore operational failures → `ExecutionLineageUnavailableError` |
| Legacy vs post-v1 | `discovery_contract_version is None` without index → legal; `== 1` without index → integrity error |

## Final verdict

**PASS (forensic topology read path)** — canonical diagnostics read-side reconstructs durable forensic parent topology and segment continuity as derived read models without checkpoint authority or write-side coupling.

**BLOCKED (attempt discovery)** — `READ_INTEGRATION_R1: BLOCKED_PENDING_ATTEMPT_DISCOVERY_ARCHITECTURE` until `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION` lands per attempt-discovery architecture R1.
