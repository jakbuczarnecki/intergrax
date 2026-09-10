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

> **Correction (2026-09-10):** `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-ATTEMPT-DISCOVERY-ARCHITECTURE-R1-LEGACY-COVERAGE-CORRECTION`  
> **Audit trail:** `LEGACY_DISCOVERY_COVERAGE_CORRECTION`

Pre-index lineage-only attempts (durable `ExecutionLineageAttemptState` without RuntimeEvent, CausalEvidence, or discovery row) are **not discoverable** under bounded read contracts. Architecture corrects the invalid candidate union that treated legacy lineage as an independent enumeration source. Run-level `ExecutionAttemptDiscoveryCompleteness` (`COMPLETE` / `LEGACY_UNKNOWN` / `TRUNCATED`) is defined; `COMPLETE` requires durable `FROM_RUN_START` proof — **not currently provable** from existing write paths (`RUN_START_PROOF: NONE`). Default conservative semantics: `LEGACY_UNKNOWN`.

**Topology implementation exists**, but final R1 remains **blocked** until honest attempt-discovery coverage semantics are implemented in `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION`.

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
| Attempt discovery completeness | `ExecutionAttemptDiscoveryCompleteness`; `COMPLETE` only with `FROM_RUN_START` proof; default `LEGACY_UNKNOWN` |
| Legacy lineage enumeration | Legacy lineage enriches discovered attempts only; lineage-only hidden attempts not enumerable without migration |

## Audit trail — LEGACY_DISCOVERY_COVERAGE_CORRECTION

| Date | Task | Outcome |
| ---- | ---- | ------- |
| 2026-09-10 | `DG-001-...-ROLLOUT-CORRECTION` | OPTION_C selected; index-first; mixed rollout |
| 2026-09-10 | `DG-001-...-LEGACY-COVERAGE-CORRECTION` | `LEGACY_LINEAGE_ONLY_DISCOVERABLE: NO`; candidate union corrected; `ExecutionAttemptDiscoveryCompleteness` defined; `RUN_START_PROOF: NONE` |

Historical PASS (forensic topology) **retained**. Attempt-discovery remains **BLOCKED** pending implementation correction.

## Final verdict

**PASS (forensic topology read path)** — canonical diagnostics read-side reconstructs durable forensic parent topology and segment continuity as derived read models without checkpoint authority or write-side coupling.

**BLOCKED (attempt discovery)** — `READ_INTEGRATION_R1: BLOCKED_PENDING_ATTEMPT_DISCOVERY_ARCHITECTURE` until `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION` lands per attempt-discovery architecture R1, including honest `ExecutionAttemptDiscoveryCompleteness` semantics (`LEGACY_UNKNOWN` default; no false `COMPLETE`).

## READ_INTEGRATION_R1_IMPLEMENTATION_CORRECTION

> **Task:** `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-CORRECTION`
> **Initial implementation SHA:** `1ab3771f8c49e13de0f85c47473522c7521f201a`

### IMPLEMENTATION_CORRECTION_INITIAL_RESULT

**CORRECTION_REQUIRED** — frozen qualification matrix incomplete + audit-discovered runtime defects:

- discovery-only real post-v1 attempt dropped incorrectly
- discovery `UNAVAILABLE` misclassified as integrity
- segment truncation false corruption on partial segment sets
- missing parentless non-root admission integrity
- discovery public-read identity / row-meta atomic gaps

Historical table below retained from initial correction landing; verdict superseded by final qualification pass.

| Area | Initial landing |
| ---- | --------------- |
| Run scope + discovery contracts | `ExecutionLineageRunScope`, discovery record/page/run state |
| Attempt marker + codec v2 | `discovery_contract_version` None/1; v1 read implicit None; v2 write explicit |
| Atomic registration | First/subsequent batches via `PartitionAtomicDocumentStore`; idempotent + concurrent |
| Index-first activation | `register_attempt_for_run` before `open_attempt(..., discovery_contract_version=1)` |
| Run discovery snapshot | Generation-guarded pagination; `LEGACY_UNKNOWN` default; `COMPLETE` only with `FROM_RUN_START` fixture |
| Per-attempt stable snapshot | Generation-guarded segments/admissions/seal; bounded retry; churn → TRUNCATED |
| Truncation matrix | Multi-segment/admission prefix safe; no false corruption on missing tail |
| State/seal consistency | Fail closed on stable contradiction |
| Provider errors | DocumentStore get/query → `ExecutionLineageUnavailableError` |
| Operator projection | `attempt_discovery_read_status` / `attempt_discovery_completeness` on lineage view |
| Tests | D1–D10 discovery + updated conformance/reconstruction suites |

## READ_INTEGRATION_R1_FINAL_CORRECTION_AND_QUALIFICATION

> **Task:** `DG-001-MULTI-AGENT-DIAGNOSTIC-LINEAGE-READ-INTEGRATION-R1-FINAL-CORRECTION-AND-QUALIFICATION`
> **Initial implementation SHA:** `1ab3771f8c49e13de0f85c47473522c7521f201a`
> **Final correction SHA:** `42cb94ec080eddbd7106af86c96d78d17f5e34b7`
> **Verdict:** **PASS**

| Matrix | Result |
| ------ | ------ |
| D8–D13 attempt discovery | PASS |
| C1–C15 coverage / integrity | PASS |
| §69–§73 truncation / outage / seal | PASS |
| Discovery-only real attempt vs stale | PASS |
| Discovery unavailable ≠ integrity | PASS |
| Provider conformance (memory + document store) | PASS |

**Final verdict:** **PASS** — bounded run-scoped attempt discovery, honest coverage metadata, stable snapshots, truncation-safe reconstruction, legacy-safe marker semantics, no second store/tree.
