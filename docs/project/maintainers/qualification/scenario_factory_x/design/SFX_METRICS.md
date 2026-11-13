# SFX metrics G-1..G-5 and M-1..M-9 (design, PROPOSED)

> **Status:** PROPOSED design draft for T-IGX-037-C (wave W1). The definitions are fixed in P2. **All targets are `PROPOSED — OD-SFX-07`.** No value is reported without a run artifact.
> **Reference SHA:** `eadb1c2116fe`. Every `path:line` citation is at this SHA.
> **Related:** master plan R1 §8 (G-1..G-5, generalization test), §13 and R0 §13 (M-1..M-9) · [ADR-SFX-002](../../../../technical/adr/entries/2026-11-12/ADR-SFX-002.md) · [`SFX_CANONICALIZATION.md`](SFX_CANONICALIZATION.md) · [`schemas/`](schemas/)

## 1. Measurement rules (MS-1..MS-8)

| Rule | Statement |
|------|-----------|
| **MS-1 SHA** | Every value records `measurement_sha`: the 40-hex commit of the factory code under test, taken from a clean tree. A run from a dirty tree is invalid. |
| **MS-2 Environment** | Every value records its environment: the original Windows environment (C10/C15 restriction) and the LAB profile at L2. |
| **MS-3 Artifact** | No value without a run artifact under `.tmp/proof/sfx/<run-id>/`, which holds the raw inputs to the formula (record digests, logs, exports). |
| **MS-4 No mixing** | Values from different SHAs are never aggregated into one distribution. A cross-SHA comparison is labelled as one. |
| **MS-5 Baselines** | Baselines are taken at P3 and at P12, each pinned to its SHA. |
| **MS-6 Rerun on change** | After the GEN reveal, any change to factory code requires re-running the full GEN set and REF-1..3 at the new SHA. Earlier results become superseded and are kept for audit. |
| **MS-7 Separation** | REF results and GEN results are reported separately. The two engineering modes (`DETERMINISTIC_SCAFFOLD`, `AGENTIC_ENGINEERING`) are reported separately; only agentic runs count toward GATE-SFX-FULL. |
| **MS-8 Reporting** | Report per scenario and as the aggregate distribution (n, median, p10, p90, min, max). Percentiles use the nearest-rank method. A percentile over fewer than 20 samples is reported as `insufficient_samples`. |

**Record sources.** Lifecycle events are the versions of the `sfx-lifecycle` artifact, read with `WorkArtifactVersionRepository.list_for_artifact` (`intergrax/collaborative_work/repository.py:1311`). Other SFX records are CW artifact versions with the same owner. Formulas below name SFX contract fields from [`schemas/`](schemas/).

## 2. Generalization metrics G-1..G-5 (per scenario)

### G-1 Reuse ratio

- **Formula:** `count(b in blueprint.capability_bindings where b.resolution in {REUSE_AS_IS, REUSE_WITH_CONFIGURATION}) / count(blueprint.capability_bindings) × 10000`. It is computed on the blueprint version approved at `BLUEPRINTED→APPROVED`.
- **Unit:** basis points (0..10000).
- **Data source:** `sfx.scenario_blueprint.v1` and `sfx.capability_resolution_report.v1` versions in CW (`intergrax/collaborative_work/repository.py:1311`). The approved version is identified through the decision binding (`intergrax/collaborative_work/repository.py:1414`).
- **Target:** PROPOSED — OD-SFX-07.

### G-2 Manual interventions

- **Formula:** `count(e in lifecycle events where e.actor.actor_kind = HUMAN and e.intervention_class = MANUAL_EDIT)`, from DRAFT to `ADMISSION_DOSSIER_READY` or to the end of the measurement window. Designed steps (`DESIGNED_ANSWER`, `DESIGNED_APPROVAL`) are excluded.
- **Unit:** count.
- **Data source:** `sfx.scenario_lifecycle_record.v1` events (`ARTIFACT_REVISION` and `TRANSITION`), CW (`intergrax/collaborative_work/repository.py:1311`).
- **Integrity rule:** a human edit to the generated package must be logged as an `ARTIFACT_REVISION` event (ADR-SFX-002 OQ-002-4). If the package tree digest at `SANDBOX_CERTIFIED` differs from the last logged digest without a matching event, the measurement is invalid.
- **Target:** PROPOSED — OD-SFX-07.

### G-3 Generation success (first pass)

Four first-pass rates. The denominator of each is the number of scenarios that reached the stage.

| Sub-metric | Numerator: scenarios where… | Data source |
|------------|-----------------------------|-------------|
| G-3a Spec valid | version 1 of `sfx.business_scenario_specification.v1` passes validation | specification versions + validation verdicts (CW) |
| G-3b Blueprint resolvable | the first blueprint's resolution report has `unresolved_count = 0` | `sfx.capability_resolution_report.v1` (CW) |
| G-3c Build conformant | the first `sfx.engineering_job_record.v1` reaches `SUCCEEDED` and its qualification check `ARCHITECTURE_CONFORMANCE` (HO-4) is `PASS` | job record + `sfx.scenario_qualification_record.v1` (CW); scaffold job state from `TaskCheckpoint` (`intergrax/runtime/long_running/models.py:19`) |
| G-3d Sandbox tests pass | the first `sfx.scenario_run_report.v1` has every `test_results[].status = PASS` | run report (CW) + evidence plane (`intergrax/contracts/execution_evidence/persistence_port.py:23`) |

- **Unit:** basis points per sub-metric, reported as a four-tuple and never collapsed into one number.
- **Target:** PROPOSED — OD-SFX-07.

### G-4 Rework

- **G-4a Improve loops:** `count(TRANSITION events with from = SANDBOX_TESTED and to = BLUEPRINTED)`. Unit: count.
- **G-4b Changed-line share:** `(added + deleted lines between the generated tree and the tree at SANDBOX_CERTIFIED) / (total lines of the generated tree) × 10000`. The generated tree is the output of the first SUCCEEDED job, identified by `output.generated_tree_digest`. The diff is a text-only `--numstat`-equivalent diff, so binary files are counted separately. Unit: basis points.
- **Data source:** `sfx.engineering_job_record.v1` output digests (CW), lifecycle events (CW), and the workspace trees at the two pinned commits.
- **Target:** PROPOSED — OD-SFX-07.

### G-5 Time and cost per stage

- **Stages:** the intervals between consecutive `TRANSITION` events (§7.9 states).
- **Time:** `to_event.created_at − from_event.created_at`. Two parts are reported: elapsed time, and human-wait time, which is the time spent in states that wait on a human decision (`BLUEPRINTED` until `APPROVED`, `SANDBOX_TESTED` until `SANDBOX_CERTIFIED`). Unit: seconds.
- **Cost:** tokens and currency attributed to the stage through the run and execution IDs recorded in the stage's artifacts. Unit: tokens (integer) and currency minor units with an ISO 4217 code.
- **Data source:** lifecycle events (CW); LLM usage from `LLMUsageTracker` (`intergrax/llm_adapters/tracking/llm_usage_track.py:48`); execution links (`intergrax/collaborative_work/repository.py:1014`).
- **Target:** PROPOSED — OD-SFX-07.

## 3. Performance families M-1..M-9

| ID | Definition and formula | Unit | Data source (owner) | Target |
|----|------------------------|------|---------------------|--------|
| **M-1 TTFS** | Time to first scenario. **Business:** from `created_at` of version 1 of `sfx.business_problem_statement.v1` to the first lifecycle `TRANSITION` to `SANDBOX_TESTED`. **Engineer:** from `created_at` of version 1 of `sfx.sfx_session_state.v1` with `door ∈ {IDE, CLI}` to the first `TRANSITION` to `BUILT`. | seconds (median per participant group) | CW artifact versions (`intergrax/collaborative_work/repository.py:1311`); session `ChatSession.created_at` (`intergrax/memory/contracts/chat_session.py:36`) as a cross-check | PROPOSED — OD-SFX-07 |
| **M-2 Discovery efficiency** | (a) `count(discovery_record.turns)` up to the first specification version that passes validation; (b) `count(field_provenance values = INFERRED) / count(field_provenance) × 10000`. | (a) count; (b) basis points | `sfx.discovery_record.v1`, `sfx.business_scenario_specification.v1` (CW) | PROPOSED — OD-SFX-07 |
| **M-3 Reuse ratio (population)** | `Σ reuse bindings / Σ bindings` over all scenarios of one set (REF or GEN), using the G-1 numerator and denominator. G-1 is per scenario; M-3 is weighted by bindings over the set. | basis points | as G-1 | PROPOSED — OD-SFX-07 |
| **M-4 Build time** | `created_at` of the `TRANSITION` to `BUILT` minus `created_at` of the `sfx.engineering_handoff_request.v1`, for jobs that are `SUCCEEDED` with HO-4 `PASS`. Failed jobs count in G-3c, not here. Per mode. | seconds | handoff request, job record, lifecycle events (CW); `TaskCheckpoint` (`intergrax/runtime/long_running/models.py:19`); agentic: SEF CON-N-02 by reference | PROPOSED — OD-SFX-07 |
| **M-5 Quality** | Eval scores on the REF truth sets: REF-1 extraction accuracy; REF-2 contract conformance rate; REF-3 groundedness and citation precision. GEN scenarios use the metrics declared in their own `sfx.scenario_test_plan.v1`. | basis points per metric | `EvalRunner` (`intergrax/eval/eval_runner.py:21`) results referenced from `sfx.scenario_run_report.v1` | PROPOSED — OD-SFX-07 |
| **M-6 Runtime** | p50 and p95 latency per case (nearest rank, MS-8); throughput = completed cases per minute at fixture load. | milliseconds; cases per minute | Evidence plane runtime events (`intergrax/contracts/execution_evidence/persistence_port.py:23`), summarized in `sfx.scenario_run_report.v1.measurements` | PROPOSED — OD-SFX-07 |
| **M-7 Cost** | (a) tokens per run and per case; (b) currency per run; (c) budget-cap adherence = `runs with cost ≤ budget cap / runs × 10000`. | tokens; minor units + ISO 4217; basis points | `LLMUsageTracker` (`intergrax/llm_adapters/tracking/llm_usage_track.py:48`), run report `measurements` | PROPOSED — OD-SFX-07 |
| **M-8 Reliability** | (a) retry success = `operations with ≥ 1 retry that eventually succeeded / operations with ≥ 1 retry × 10000`; (b) idempotency violations = count of duplicate side effects (same idempotency key, different effect). (b) must be 0; this is an invariant, not an OD target. | basis points; count | Evidence plane (`intergrax/contracts/execution_evidence/persistence_port.py:23`, append idempotent on `event_id`); `TaskCheckpointPersistence` attempts (`intergrax/runtime/long_running/persistence_contract.py:45`); REF-2 reconciliation report | (a) PROPOSED — OD-SFX-07; (b) 0 |
| **M-9 Usability** | (a) SUS score (standard 10-item questionnaire, 0..100); (b) task success = `tasks completed without facilitator help / tasks attempted × 10000`. Three rounds of five participants (master plan R1 §8). | SUS points; basis points | Usability study records (no platform owner) stored as run artifacts under `.tmp/proof/sfx/<run-id>/usability/` | PROPOSED — OD-SFX-07 |

## 4. GEN-set protocol (generalization test)

| Rule | Statement |
|------|-----------|
| **GEN-1 Composition** | At least five independent business problems from different domains. REF-1..3 run alongside and are reported separately (MS-7). |
| **GEN-2 Independent author** | The GEN set is authored by an independent session that is neither an implementer nor an architect of SFX. The author records only the session identity and the digest. |
| **GEN-3 Sealed** | Each problem is a `sfx.business_problem_statement.v1` input, with no hints, expected blueprints or capability names. The set is sealed by digest: the CF-8 digest of a JCS array of the problem statements' canonical digests, sorted. **Only the hash is committed**, and it is committed before W3 starts. The content is held outside the repository by the author or the operator. |
| **GEN-4 Reveal** | The content is revealed only at SFX-P12-04. The revealed set must reproduce the sealed digest; otherwise the GEN run is void. |
| **GEN-5 Independent evaluator** | The GEN run and the G-1..G-5 measurement at SFX-P12-04 are performed by an independent evaluator, not by the implementer. |
| **GEN-6 No tuning** | No GEN-specific code paths, prompts, fixtures, configuration or inventory entries. After the reveal, any change to factory code requires re-running the full GEN set and REF-1..3 at the new SHA (MS-6). The review of every post-reveal change checks the diff for GEN content. |
| **GEN-7 No cherry-picking** | Every GEN problem is reported, including failures and aborted runs. A problem can be excluded only by an operator decision recorded before the reveal. |

## 5. Open points

- G-2 depends on logging artifact revisions in the lifecycle stream (ADR-SFX-002 OQ-002-4).
- G-3c depends on an HO-4 conformance entry point for `applications/` packages (ADR-SFX-002 OQ-002-2).
- G-5 and M-7 cost attribution per stage needs run and execution IDs to be propagated into usage records. The required field mapping is to be confirmed at SFX-P5.
- All targets are PROPOSED — OD-SFX-07 (operator, at P3 exit).
