# PDC-P7 IA-D5 - independent audit of domain D5 (Evaluation)

| Field | Value |
| --- | --- |
| **Stage** | PDC-X / P7 (per-domain audit, auditor `IA-D5`) |
| **Policy** | `IIAC-2026-11-05.1`, `OD-PDC-IIAC-SCOPE` (Option A) |
| **Audited SHA (`D5_AUDITED_SHA`)** | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (= `origin/development` when the audit started) |
| **Audit execution HEAD** | `b7889220607b462195308fa186f4ba66fb78f3a8` (descends from the audited SHA; 8 commits, 7 files, all under `docs/` or `governance/`; product, script and test delta = 0) |
| **Batch under audit** | B-D5, ticket `T-IGX-036-O`, implementation SHA `85a8c441d8e3f37e3f2d09451686dcd2f556d490` (ladder: `D5_IMPLEMENTATION_SHA`), ADR-PDC-05 |
| **Auditor session** | `PDC-P7-IA-D5-INDEPENDENT-AUDITOR-2026-11-15` (clean session; never implemented, signed or certified any D5 item; edits nothing outside this report and its evidence folder) |
| **Evidence** | `docs/project/maintainers/qualification/iiac_evidence/PDC-X/D5/` |
| **Overall verdict** | **FAIL** (see section 4; the FAIL comes from one row, `PDC-D5-R015`, and the repair is narrow) |

The auditor does not certify, does not fix and does not change any status of the requirements matrix. Verdict vocabulary follows `OD-PDC-IIAC-SCOPE`: PASS, PASS_SCOPED (Operator-authorized exclusions only; none was used), PARTIAL, FAIL.

## 1. Method

1. Read `OD-PDC-IIAC-SCOPE`, the 18 `PDC-D5-R*` rows of `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, the D5 lines of `PDC_OWNERSHIP_MATRIX.md`, ADR-PDC-05, the B-D5 batch in `PDC_IMPLEMENTATION_WBS.md` and the ticket/plan `T-IGX-036-O`.
2. Read every owner file at the audited SHA (about 2,800 lines): `intergrax/contracts/evaluation_contracts.py`, `intergrax/eval/{evaluators,typed_run,eval_comparison}.py`, `intergrax/runtime/architecture/{evaluation_assets,evaluation_automation,evaluation_registry_trends,evaluation_scenario_loader,online_evaluation_registry,agent_promotion}.py`, `intergrax/runtime/evidence/eval_evidence_contracts.py`, `intergrax/runtime/evaluation/human_review_sample_queue.py`, `intergrax/applications/_shared/evaluation_wiring.py`, `scripts/gates/check_mvp_promotion_gates.py`, plus the production writers/readers of the registry and all call sites of the new APIs (`git grep`).
3. Ran the cited tests, one `uv run pytest` at a time:
   - `tests/qualification/pdc_x/d5`: **79 passed** in 17 s (`logs/pytest_pdc_x_d5.log`);
   - one bounded run over `tests/qualification/pdc_x/xd/test_xd_08_evaluation.py` and the existing unit suites of the touched packages (21 files; `tests/unit/eval`, `tests/unit/experiments`, the architecture, evidence and application evaluation unit tests): **100 passed, 1 failed** in 28 s (`logs/pytest_d5_unit_and_xd08.log`; failure = F-D5-07);
   - `scripts/qualification/pdc_x_requirements.py --check`: `[OK] 172 rows, schema valid`.
4. Read the tests as an attacker would (can they fail? what do they not cover?) and wrote three probe scripts that attack the claims (`adversarial_scripts/probe_a.py`, `probe_b.py`, `probe_c.py`, outputs in `logs/`): executor failure, empty runs, telemetry/digest, digest hygiene, mutation after construction, promotion with free-text inputs, production writers and tenant scoping, id oracle and collision, lock lease and lock removal, a 4-process append, human queue.
5. Layer check: `git grep` for `agents` / `applications` imports under `intergrax/`: none. The B-D5 modules import only `intergrax.*` owners (contracts, integrations contracts, runtime). No new registry, scheduler or authority beside the existing owners was found.
6. Re-verified the findings routed to IA-D5 by `PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md` and its addenda 1..4: F-04 (D5 part), F-24, F-25, F-32 (section 5). The addenda 1..4 contain no further D5 item. F-09 (pytest not exiting) concerns `test_xd_03` and was not re-run (limitations).

Known reds that are **not** re-reported: the EBH-7 identity gate, the five `roadmap_replay_x` tests, `dist_scale_x`/Docker scope. Debt R-05 (`pdc_x` pytest process not exiting after the summary): **not reproduced** on the D5 directory or on the unit set; both processes exited normally.

## 2. Per-row verdicts

| Row | Requirement (short) | Verdict | Evidence (what holds / what does not) | Findings |
| --- | --- | --- | --- | --- |
| PDC-D5-R001 | Typed EvalDataset / EvalScenario / EvalRun | **PASS** | `evaluation_contracts.py`: frozen, `extra=forbid`, round-trips, `EvalComparison.advisory` is `Literal[True]`. 79 D5 tests green. Caveat: no production caller (F-D5-06). | F-D5-06 |
| PDC-D5-R002 | Dataset version bound to a content digest | **PARTIAL** | `EvalDataset.content_digest` is computed and a declared digest mismatch is rejected (tested). `GoldenDatasetAsset.content_digest` stays optional, its format is not validated, `bind_content`/`verify_content` have no caller, and it is not tied to `EvalDataset`. | F-D5-09 |
| PDC-D5-R003 | Stable scenario identity | **PASS** | `identity = id@digest(id, input, expected, sorted risk_tags)`; duplicate ids rejected; the library loader returns a digested dataset. Hygiene gaps only. | F-D5-10 |
| PDC-D5-R004 | Agent/model/prompt/profile/code-SHA on every run | **PARTIAL** | `EvalRun` carries all references and rejects blank ones. `EvalResult` (the cited owner path) and the legacy `EvalRunner`/`NexusEvalRunner` are unchanged and produce no provenance; only a caller that opts into `execute_eval_run` gets it, and nothing in production does. References are caller-asserted. | F-D5-06, F-D5-11 |
| PDC-D5-R005 | Task-success, tool-use, groundedness, citation evaluators behind one contract | **PARTIAL** | `Evaluator` protocol, registry and adapters for exact-match (task success), tool use, citation and golden retrieval exist. **No groundedness evaluator exists in product code**; `judge.groundedness` appears only as a test-injected fixed-score judge. The registry is composed only in tests. | F-D5-05 |
| PDC-D5-R006 | Safety evaluator kind via guardrail provider | **PASS** | `GuardrailSafetyEvaluator` runs `scan_output` of an `LlmGuardrailBackend`, a block becomes `FAILED`; tests green. Minor context and risk-level gaps. | F-D5-14 |
| PDC-D5-R007 | Deterministic vs model-assisted separation | **PASS** | `MODEL_ASSISTED` outcomes must name `judge_model`; judge outcomes are excluded from `rule_outcomes`, `scenario_successes`, evidence status and `EvalComparison.regressed`; the judge receives the subject only; legacy `AutomatedEvaluationRecord` now exposes `rule_passed`/`judge_passed` and the trend summary reports both. Isolation is by convention. | F-D5-17 |
| PDC-D5-R008 | Human review queue (current-head requalification) | **PASS** | Queue works as documented (FIFO, pending list, mark reviewed); `HumanReviewEvaluator` reports a missing decision as `PENDING`, never a pass (tested). The queue is in-process, tenant-less and unauthenticated. | F-D5-13 |
| PDC-D5-R009 | Reproducible evaluation runs | **PASS** | Pinned rerun gives an identical `result_digest` (xd_08 test and own probes); latency noise excluded by design; a changed pinned input changes the digest. Hygiene gaps in F-D5-10. | F-D5-10 |
| PDC-D5-R010 | Baseline vs candidate comparison | **PASS** | `compare_eval_runs` flags per-metric and per-scenario regressions and improvements, rejects cross-tenant and cross-dataset-digest comparison. An empty-vs-empty comparison is `clean` (F-D5-04). | F-D5-04 |
| PDC-D5-R011 | Quality regression detection | **PASS** | Floors, ceilings and tolerances behave as specified; a missing candidate metric breaches; breaches are evaluated on the candidate only. | - |
| PDC-D5-R012 | Cost-to-success and latency comparison | **PASS** | `cost_to_success`, mean and p95 latency in metrics and in the comparison; `None` when there is no success. | - |
| PDC-D5-R013 | Run provenance in the canonical evidence store | **PARTIAL** | `build_eval_run_evidence_report` produces a valid canonical `EvalEvidenceReport` and read-back checks tenant and digest. But (a) a run whose executor failed is recorded as `PASSED`, (b) integrity is self-attested, (c) no production code builds, writes or reads typed run evidence. | F-D5-04, F-D5-06, F-D5-11 |
| PDC-D5-R014 | Promotion stays governance-owned; evaluation advisory | **PARTIAL** | An evaluation PASS cannot authorize: `approved = governance_authorized and not evaluation_reasons`, `evaluation_advisory_only` is constant, and `check_mvp_promotion_gates.py` can only deny. But G0/G2 are still file-existence checks, the governance input is any non-placeholder `Approved:` line (reproduced: an unrelated implementation ticket yields `GRANTED`), the decision inputs are unverified free text unbound to the promoted subject, and `evaluate_agent_promotion` has no production call site. | F-D5-02, F-D5-03 |
| PDC-D5-R015 | `tenant_id` on online observations; cross-tenant denied | **FAIL** | The opt-in API (`append_for_tenant`, `TenantScopedOnlineEvaluationRegistry`, `get_observation`) denies cross-tenant reads (tested). `OnlineEvaluationObservation` itself did not gain `tenant_id`; all four host wirings call `wire_application_evaluation` without a tenant; `record_user_satisfaction` drops the tenant it holds; the legacy `list_observations()` returns every row to any caller. Reproduced. | F-D5-01, F-D5-12 |
| PDC-D5-R016 | CRITIC_VERIFICATION findings are legacy lineage | **PASS** | Reconciliation note present in `docs/project/maintainers/plans/CRITIC_VERIFICATION.md` (T-IGX-036-H); the matrix cites a path that does not exist. | F-D5-16 |
| PDC-D5-R017 | Production-feedback boundary documented | **PARTIAL** | The only boundary text is the pre-existing line "user satisfaction is evidence, not governance authorization". Nothing documents the typed contracts, ADR-PDC-05 or the advisory boundary of evaluation; no ticket owns R017 (excluded from T-IGX-036-H, delegated to the orchestrator by the O plan). | F-D5-08 |
| PDC-D5-R018 | File registry concurrency semantics | **PASS** | Exclusive in-process lock plus `O_EXCL` lock file, busy writers rejected, atomic replace. 4 processes x 25 appends: 100 stored, 0 lost, no lock left (`logs/probe_c.out.txt`). Edge cases of the lease are in F-D5-12. | F-D5-12 |

Totals: **PASS 11 · PARTIAL 6 · FAIL 1** (18 rows).

## 3. Findings

Severity scale: HIGH = a requirement's stated outcome is false on a production path; MEDIUM = a mandatory behavior is missing, unverified or misreported; LOW = hygiene/hardening with a bounded effect; INFO = no action required for the verdict.

### F-D5-01 - HIGH - tenant isolation of online evaluation observations is opt-in and unused (R015, FRZ-TEN-01, FRZ-TEN-07)

- **Claim at risk:** "`tenant_id` on `OnlineEvaluationObservation`... cross-tenant observation denied" (TENANT-X debt recorded by CTRL-X).
- **Facts:**
  - `intergrax/runtime/architecture/online_evaluation_models.py` is unchanged: `OnlineEvaluationObservation` has no `tenant_id`. A subclass `TenantOnlineEvaluationObservation` with `tenant_id: str | None = None` lives in the registry module.
  - `wire_application_evaluation(env, *, tenant_id=None)` binds the registry to a tenant only when `tenant_id` is passed. All four production call sites omit it: `harness_host_runtime.py:542`, `policy_wiring.py:154`, `runtime_config_bridge.py:197`, `scenario_runtime_baseline.py:459`.
  - `record_user_satisfaction` (`user_satisfaction.py`) receives `UserSatisfactionEvent.tenant_id` but `satisfaction_to_online_observation` drops it; `record_shadow_observation` has no tenant parameter. Without a scoped registry the rows are written with `tenant_id = None`.
  - The legacy `FileOnlineEvaluationRegistry.list_observations()` returns all rows (tenanted and untenanted) to any caller, and `default_online_evaluation_registry()` is one shared file per checkout.
- **Reproduction:** `adversarial_scripts/probe_b.py` Q1/Q2 (`logs/probe_b.out.txt`): a satisfaction event for `tenant-a` is stored with `tenant_id=None`; the `tenant-b` scoped view sees 0 rows (fail-closed for the scoped view) but the legacy unscoped `list_observations()` returns the rows of both tenants.
- **Why HIGH:** the row is TENANT_IMPACT HIGH and its outcome ("cross-tenant observation denied") holds only for callers that voluntarily use the new API; no production writer or host does. This is the dossier's F-04 for D5 (confirmed).
- **Suggested repair scope** (ticket of the D5 owner, different session): add `tenant_id` to the observation contract (or make the registry refuse untenanted writes); make `wire_application_evaluation` require the tenant and fail closed when the registry is enabled without it; pass the tenant from the four hosts; stamp `event.tenant_id` in `record_user_satisfaction`; remove or restrict the unscoped `list_observations()`/`default_online_evaluation_registry()` read for multi-tenant hosts; add a wiring-level negative test per host.
- **Owner paths:** `intergrax/runtime/architecture/{online_evaluation_models,online_evaluation_registry,online_evaluation,user_satisfaction}.py`, `intergrax/applications/_shared/{evaluation_wiring,harness_host_runtime,policy_wiring,runtime_config_bridge,scenario_runtime_baseline}.py`.

### F-D5-02 - MEDIUM - MVP promotion gate: G0/G2 remain file-existence checks; authorization is any `Approved:` line (R014, FRZ-GOV-01/02, FRZ-REG-09)

- **Facts:** the docstring says G0 and G2 "check that the required modules exist **and define** the required symbols". In `gate_g0_runnable` and `gate_g2_policy` every symbol tuple is `()`, so `_defines` returns nothing and the checks equal file existence (dossier F-32, confirmed; only G1 has symbols, and the tests exercise only G1). `governance_authorization` accepts any file under `governance/tickets/` that carries a non-placeholder `Approved:` line (and a `Delegation:` line for `operator-delegate`); it does not bind the ticket to the promotion subject, does not check the plan digest or the delegation record, and the regex matches anywhere in the file.
- **Reproduction:** `python scripts/gates/check_mvp_promotion_gates.py --governance-ticket governance/tickets/T-IGX-036-O.md --require-authorization` prints `promotion authorization: GRANTED by governance` and exits 0 (`logs/gate_script_unrelated_ticket_granted.txt`). The plan deliverable 5 says gates must not be satisfied by file existence alone.
- **Severity note:** the dossier classed F-32 INFO ("accept"). Re-assessed here as MEDIUM because it contradicts the batch's own acceptance statement and the authorization is not subject-bound.
- **Suggested repair scope:** name the required symbols for G0/G2 (or drop the claim from the docstring and the plan); bind authorization to a ticket type/subject and verify it with the existing plan-approval verifier; add G0/G2 hollow-module tests. **Owner path:** `scripts/gates/check_mvp_promotion_gates.py`, `tests/qualification/pdc_x/d5/test_xd08_eval_pass_cannot_authorize_promotion.py`.

### F-D5-03 - MEDIUM - promotion decision accepts unverified free-text governance inputs and is unwired (R014)

- **Facts:** `governance_authorized` is true when the stage path is allowed, `certification.eligible` is true and `rollback_plan_ref`/`change_ticket_ref` are non-blank strings. Nothing checks that the strings reference real artifacts, that `certification` belongs to the same `agent_id`/`agent_version`, or that an `EvalComparison` belongs to the promoted agent (`EvalComparison` carries no agent or revision reference). `evaluation_report_refs` is a list of free strings. `evaluate_agent_promotion` has no call site under `intergrax/`, `agents/`, `applications/` besides its re-export (ADR-SFX-006 S4 is still true); only `scripts/gates/check_agent_promotion_eval_gate.py` calls it.
- **Reproduction:** `probe_a.py` P8: bundle with refs `"x"`, a certification of another agent (`OTHER@9`) and an unrelated clean comparison gives `approved=True, governance_authorized=True`.
- **What holds:** evaluation cannot authorize by itself (tested, and re-checked); a regression or breach blocks. The gap is the strength of the governance side and the missing binding of evidence to subject.
- **Suggested repair scope:** bind evidence to subject (agent/version/tenant in `EvalComparison` or in the bundle check), verify the refs through the owning stores, and wire the decision into the actual promotion path owned by Governance. **Owner path:** `intergrax/runtime/architecture/agent_promotion.py`, `intergrax/contracts/evaluation_contracts.py`.

### F-D5-04 - MEDIUM - evidence status and `clean` can be green without evidence (R013, R010)

- **Facts / reproduction (`probe_a.py` P1, P2):**
  - A run whose executor raised (`telemetry.error = "executor failed: RuntimeError"`, `scenario_successes() == {s1: False}`) is recorded by `build_eval_run_evidence_report` with status **PASSED** when the only rule evaluator is a safety scan (the scan of the empty output passes). `_run_status` looks at rule outcomes only and ignores telemetry errors.
  - Two empty runs compare `clean == True`; a candidate with no applicable evaluator for a metric the baseline lacks is never flagged. `EvalRun` does not check that its outcomes/telemetry cover the dataset.
  - `human_pending > 0` is ignored by `clean`, by the evidence status and by the promotion advisory note ("missing human review never approval", FRZ-GOV-06, is only honored for the pass-rate).
- **Suggested repair scope:** include telemetry errors and coverage of the dataset in the evidence status and in `clean`; treat "no evidence" as `UNAVAILABLE`, not clean; surface `human_pending`. **Owner paths:** `intergrax/runtime/evidence/eval_evidence_contracts.py`, `intergrax/contracts/evaluation_contracts.py`, `intergrax/eval/eval_comparison.py`.

### F-D5-05 - MEDIUM - no groundedness evaluator; evaluator registry is composed only in tests (R005)

- The row names task-success, tool-use, **groundedness** and citation evaluators. Product code provides exact-match, tool-use, citation and a golden-retrieval adapter plus a judge slot; a groundedness evaluator exists only as a fixed-score judge built in the tests (`judge_fixed(0.9)` named `judge.groundedness`). `register_existing_evaluators` has no caller outside tests, so no host composes the registry. The test "registry covers kinds" proves kinds, not the named evaluators.
- **Suggested repair scope:** add (or explicitly exclude with an Operator decision) a groundedness evaluator behind the contract and compose the registry in the owning wiring. **Owner path:** `intergrax/eval/evaluators.py`, `intergrax/applications/_shared/evaluation_wiring.py`.

### F-D5-06 - MEDIUM - the typed evaluation pipeline has no production caller (dossier F-04, D5 part; R001, R004, R013)

- `execute_eval_run`, `compare_eval_runs`, `build_eval_run_evidence_report`, `read_eval_run_evidence`, `register_existing_evaluators`, `scenario_library_to_eval_dataset`, `GoldenDatasetAsset.bind_content/verify_content` are referenced only by their own modules and tests (`git grep`, section 1). `EvalRunner`, `NexusEvalRunner` and `EvalResult` are unchanged and do not use them; `eval_evidence_runner`/`eval_evidence_export` do not know the typed kinds, so a typed run is never written to `build/evidence/eval`. The requirements hold for a caller that opts in; "on every run" and "recorded as evidence" cannot be observed on any runtime path.
- **Suggested repair scope:** route one real runner (for example `NexusEvalRunner`/`EvalRunner`) through `execute_eval_run` and persist through the evidence export; decide explicitly that the contracts are a library and say so in R004/R013. **Owner paths:** `intergrax/eval/`, `intergrax/runtime/evidence/eval_evidence_*.py`.

### F-D5-07 - MEDIUM - existing test of the D5 package `intergrax/experiments` is red: `NameError: mint_run_id` (not a PDC row; B-D5 acceptance "existing suites stay green")

- `tests/unit/experiments/test_experiment_workflow.py::test_experiment_session_echo_workflow` fails at `intergrax/experiments/workflow.py:163` (`run_id=mint_run_id()`; the name is not imported). Introduced by `e4079dfb1` (2026-09-30), so it predates B-D5, but `intergrax/experiments/` is in the B-D5 write set and `ExperimentSession.run` is broken on every call. Not among the known reds. Reproduction: `uv run pytest tests/unit/experiments/test_experiment_workflow.py` (`logs/pytest_d5_unit_and_xd08.log`).
- **Suggested repair scope:** import `mint_run_id` from `intergrax.contracts.execution_identity` and keep the test. **Owner path:** `intergrax/experiments/workflow.py`. (This repair touches an ACT-N11 product path and therefore needs the rebind.)

### F-D5-08 - MEDIUM - R017 (evaluation boundary documentation) was never delivered and has no owner

- `T-IGX-036-H` excluded R017 ("ADR-bound lane"), `PLAN_T-IGX-036-O` sends it to the orchestrator, and no later ticket carries it. No architecture or plan document mentions `EvalRun`/`EvalComparison`, ADR-PDC-05 or the advisory-only boundary (`git grep` over `docs/project/architecture` and `docs/project/maintainers/plans`). The matrix row still reads OPEN.
- **Suggested repair scope:** a docs-only ticket adding the boundary (evaluation advisory; promotion Governance-owned; online feedback is evidence, tenant-scoped) to `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` and its plan pair. **Owner path:** `docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`.

### F-D5-09 - LOW - dataset asset digest is optional and unenforced (R002)

- `GoldenDatasetAsset.content_digest: str | None = None` accepts any string, nothing calls `bind_content`/`verify_content`, and the asset digest (bytes) is a different concept from `EvalDataset.content_digest` with no link. "Version free string" is still possible for every asset in `evaluation_assets` bundles. **Scope:** validate the `sha256:` format, require the digest where assets are loaded, link asset to `EvalDataset`. **Owner path:** `intergrax/runtime/architecture/evaluation_assets.py`.

### F-D5-10 - LOW - digest hygiene (R003, R009)

- `stable_payload_hash` uses `json.dumps(default=str)`: a set-valued `input_payload` hashes to different values per process (`probe_a.py` P6, three hash seeds, three digests). `EvalScenario`/`EvalDataset` are `frozen` but hold mutable dicts: mutating `input_payload` after construction changes `content_digest` while the declared digest was checked only at construction (P5). `EvalRun.result_digest` excludes `telemetry.error`, cost, latency and tokens although `scenario_successes()` and `cost_to_success` depend on them: two runs with equal `result_digest` can have 3 vs 2 successes (P3). **Scope:** reject non-JSON payload values, freeze nested containers, document or include the error in the digest. **Owner paths:** `intergrax/contracts/evaluation_contracts.py`, `intergrax/contracts/canonical_payload_hash.py` (shared; coordinate).

### F-D5-11 - LOW - evidence integrity is self-attested (dossier F-24, confirmed)

- `read_eval_run_evidence` checks only `result_digest` against the payload that carries it; both can be edited together, and cost/latency/tokens/errors are outside the digest, so a payload with `cost=999` per scenario is accepted (P3). Metadata fields `inputs_digest`, `dataset_digest`, `code_sha` are not cross-checked; comparisons are caller-supplied and not re-derived from runs. `code_sha` and `dataset_digest` patterns use `$`, so `"abcdef1\n"` is accepted (P4). The test named "tampered payload rejected" changes one field the digest covers. **Scope:** state the integrity level honestly or sign/anchor the digest; use `\Z`/`fullmatch`; re-derive or verify comparisons. **Owner paths:** `intergrax/runtime/evidence/eval_evidence_contracts.py`, `intergrax/contracts/evaluation_contracts.py`.

### F-D5-12 - LOW - registry id oracle, id collision and lock-lease edges (dossier F-25, confirmed; R015, R018)

- Existence oracle: tenant B asking for tenant A's id gets `OnlineEvaluationTenantBoundaryError`, an unknown id gets `KeyError` (Q3). Collision: `observation_id` is `shadow:{run_id}:{scenario_id}` (not tenant-qualified); when two tenants hold the same id, `get_observation` returns the first match and denies the second tenant's own valid row (Q4).
- Lock: time-based lease without owner token or heartbeat. A holder longer than `stale_lock_seconds` loses exclusivity (reproduced with 0.3 s: `A-in, B-in, A-out, B-out`) and the holder's `finally` unlinks the lock now owned by the next writer, so a third writer enters while B is inside (Q6: `A-out, C-in, C-out, B-out`). Two waiters can both remove one stale lock (TOCTOU, not reproduced). The default 60 s makes this rare; there is no cross-process test (my 4-process probe passed).
- **Scope:** filter by tenant before matching, return `KeyError` for foreign ids, qualify ids with the tenant, add an owner token to the lock file and unlink only an owned lock. **Owner path:** `intergrax/runtime/architecture/online_evaluation_registry.py`.

### F-D5-13 - LOW - human review queue and HUMAN evaluator (R008, FRZ-GOV-06)

- The queue is in-process (lost on restart), has no tenant, `mark_reviewed` accepts an empty `reviewer_id`, any caller and repeated reviews (Q7). `HumanReviewEvaluator` takes verdicts from a plain mapping that is not linked to the queue state or to a reviewer identity, and enqueues a new sample for every rerun of a pending scenario. **Scope:** durable tenant-scoped queue, reviewer identity, link decisions to samples. **Owner paths:** `intergrax/runtime/evaluation/human_review_sample_queue.py`, `intergrax/eval/evaluators.py`.

### F-D5-14 - LOW - safety evaluator context (R006)

- `GuardrailSafetyEvaluator` builds `GuardrailContext(run_id=...)` only: no tenant or agent reaches the backend, so tenant-specific guardrail policy cannot apply; `assess_guardrail_scan` ignores `risk_level`, so `allowed=True` with a HIGH/CRITICAL level passes; an evaluator exception aborts the whole run (fail-closed, acceptable). **Owner path:** `intergrax/eval/evaluators.py`, `intergrax/contracts/evaluation_contracts.py` (`EvalSubject` has no tenant).

### F-D5-15 - INFO - attribution trailer in merged history

- Commit `85a8c441d` (B-D5) carries `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`, contrary to the rule `intergrax-no-ai-attribution`. It is merged history and cannot be amended; recorded for the certifier. This audit's commit carries none.

### F-D5-16 - INFO - stale pointers

- The matrix cites `docs/project/architecture/CRITIC_VERIFICATION.md` for R016; the file does not exist (the note was added to `docs/project/maintainers/plans/CRITIC_VERIFICATION.md`, whose "1:1" link to it is also dead). `governance/tickets/T-IGX-036-O.md` still shows `Lifecycle state: plan_approved` after the merge. Applications path: the matrix names `applications/_shared/evaluation_wiring.py`, the code is `intergrax/applications/_shared/evaluation_wiring.py`.

### F-D5-17 - INFO - isolation of judge results is by convention

- The registry cannot verify that a `DETERMINISTIC`-labelled evaluator calls no model; `AutomatedEvaluationRecord.final_passed` is a caller-supplied boolean not validated against `rule AND judge`; a missing judge score counts as 0.0 (fail). Accepted as the design in ADR-PDC-05; recorded so that nobody reads "isolated" as "enforced".

### F-D5-18 - INFO - positives and non-reproductions

- Layer boundaries hold (no `agents`/`applications` import from `intergrax/` in the D5 modules); no new registry or authority; `EvalResult` stays valid; all 79 D5 tests and 100 of 101 bounded unit/xd_08 tests pass; the `pdc_x` processes exited normally (R-05 not reproduced on these runs).

## 4. Overall verdict

**FAIL** for D5 at `c08efe46a74a20595468d8d8bf06eab2dca5f573`.

- Basis: 18 rows = 11 PASS, 6 PARTIAL, 1 FAIL. `PDC-D5-R015` (TENANT_IMPACT HIGH, mandatory gap) is not met on any production path (F-D5-01). Per-row verdicts are not averaged; the worst mandatory row governs.
- Findings: **HIGH 1, MEDIUM 7, LOW 6, INFO 4** (18 total).
- The contract layer itself (typed dataset/run/comparison, digests, kinds, comparison, opt-in tenant API, file concurrency) is sound and well tested; the failures are at the seams: production wiring, the promotion gate, evidence status and one unowned doc row.
- Routing: F-D5-01..08 need a repair ticket of the D5 owner by a session that is neither this auditor nor the original implementer, then a fresh auditor (loop 1 of 2). F-D5-09..14 may be bundled in the same ticket. F-D5-15..18 need no repair for the verdict.
- Manifest verdict mapping: the IIAC manifest schema has no PARTIAL/FAIL-with-conditions value beyond `FAIL`; the manifest records `FAIL`.

## 5. Re-verification of dossier findings routed to IA-D5

| Dossier ID | Dossier severity | Status at `c08efe46a` | Here |
| --- | --- | --- | --- |
| F-04 (D5 part: tenant-scoped evaluation has no production caller) | MEDIUM | **Confirmed** | F-D5-01, F-D5-06 |
| F-24 (eval evidence digests self-attested, comparisons caller-supplied) | LOW | **Confirmed**, reproduced | F-D5-11 |
| F-25 (registry id oracle and lock removal) | LOW | **Confirmed**, reproduced (oracle, collision, lease) | F-D5-12 |
| F-32 (MVP gate docstring vs empty G0/G2 symbol tuples) | INFO | **Confirmed**, re-rated MEDIUM | F-D5-02 |
| F-09 (pytest does not exit after `test_xd_03`) | LOW | Not in D5 scope; not re-run; D5 runs exited normally | F-D5-18 |

## 6. Limitations (what was not covered)

- Not run: full `tests/unit`, the other `tests/qualification/pdc_x/*` directories (including `xd/test_xd_03`), `ruff`/`pyright` on the touched files, CI contexts, the P6 acceptance replay, FRZ criteria, ECAP section 2.C, `igx_gov ticket`.
- No mutation testing of the D5 tests. Test strength was judged by reading and by the probes; the stale-lock TOCTOU (two waiters removing one lock) was found by code reading and not reproduced.
- Network and live providers were out of scope: the guardrail backend and the judge were exercised only through fakes; no real vendor evidence exists or was sought (C10/C15 freeze, no Docker).
- Wiring claims (F-D5-01) rest on `git grep` of the four call sites and on direct calls of the registry/wiring API; no full host start was run.
- Other domains' audit reports were not read (independence). Interactions with D6 (evidence plane, audit trail) and D3 (revision references) were not audited; model/prompt/profile references are neutral strings by design and were not checked against their owners.
- The ticket regex check: `Ref: governance/tickets/T-IGX-036-O.md` is a valid existing ticket and is used as the single trailer of this docs-only commit; the ticket is a product-batch ticket, so the use of it for an audit commit is by operator instruction.
- Evidence files were produced from the auditor's checkout; probe scripts carry absolute paths of that checkout in `sys.path` and need that one line adjusted to rerun elsewhere.
