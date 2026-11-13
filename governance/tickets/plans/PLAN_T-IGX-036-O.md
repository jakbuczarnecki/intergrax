# PLAN T-IGX-036-O — PDC-P5 batch B-D5: D5 Evaluation

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-D5**: typed EvalDataset/EvalRun/EvalComparison, evaluator kinds, evidence persistence, advisory promotion). Decision record: ADR-PDC-05 (Accepted). Requirements: PDC-D5-R001, R002, R003, R004, R005, R006, R007, R009, R010, R011, R012, R013, R014, R015, R018.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Typed contracts (R001-R004).** New `intergrax/contracts/evaluation_contracts.py` with `EvalDataset` (content digest, version), `EvalScenario` (stable identity), `EvalRun` (agent, model, prompt and profile revisions, code SHA, dataset digest, seed, tenant) and `EvalComparison` (baseline vs candidate, per-metric deltas, cost-to-success, latency). Additive: existing `EvalResult` stays valid.
2. **One evaluator contract (R005-R007).** Evaluator kinds `DETERMINISTIC`, `MODEL_ASSISTED`, `HUMAN` and `SAFETY`. The model-assisted judge is isolated and never mixed with rule results. `SAFETY` is implemented through the existing guardrail integrations. Existing evaluators (for example `rag/evaluation/golden_harness`) are registered, not rewritten.
3. **Reproducible runs and comparison (R009-R012).** A rerun with pinned inputs yields a digest-identical result. Comparison flags a candidate regression and a threshold breach and carries cost and latency.
4. **Evidence persistence (R013, R015, R018).** Runs and comparisons are recorded through the canonical `EvalEvidenceReport` path. `OnlineEvaluationObservation` gains `tenant_id` and cross-tenant reads are denied. The file registries get explicit concurrency semantics (safe or rejected), or move to the evidence store.
5. **Advisory only (R014).** An evaluation PASS is an input to governance-owned promotion and never authorizes deployment by itself. `agent_promotion.py` and `check_mvp_promotion_gates.py` are corrected so that gates are not satisfied by an evaluation result or by file existence alone.

Design constraints: Governance is not Evaluation: no promotion gate owned by the evaluation package. Model, prompt and profile revisions are carried as neutral typed references, so this batch does not import code from the unmerged D3 branch. Customer-specific datasets and production benchmarks are out of scope. The cases the WBS names under `xd_08` (digest-identical rerun, eval PASS cannot authorize promotion) are written as `test_xd08_*` modules under `tests/qualification/pdc_x/d5/` and re-pointed by PDC-P6-01. R017 (boundary documentation) is docs-only and goes to the orchestrator because docs are outside this write set. This is the only wave-2 batch that writes under the shared hotspots (`intergrax/contracts/`, `intergrax/applications/_shared/`).

## Write set

`intergrax/eval/`, `intergrax/experiments/`, `intergrax/runtime/evaluation/`, `intergrax/runtime/architecture/evaluation_assets.py`, `intergrax/runtime/architecture/evaluation_automation.py`, `intergrax/runtime/architecture/evaluation_registry_trends.py`, `intergrax/runtime/architecture/evaluation_scenario_loader.py`, `intergrax/runtime/architecture/online_evaluation_registry.py`, `intergrax/runtime/architecture/agent_promotion.py`, `intergrax/runtime/evidence/eval_evidence_contracts.py`, `intergrax/contracts/evaluation_contracts.py`, `intergrax/applications/_shared/evaluation_wiring.py`, `scripts/gates/check_mvp_promotion_gates.py`, `tests/qualification/pdc_x/d5/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who is the single integration owner for the shared hotspots (`intergrax/applications/_shared/`, `intergrax/contracts/`).

## Protocol

1. Reuse the canonical mechanism named in ADR-PDC-05.
2. Write the tests first, under `tests/qualification/pdc_x/d5/`.
3. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files.
4. Commit with `Ref: governance/tickets/T-IGX-036-O.md`, push to a branch, open a PR against `development`, and **do not merge**.
5. Record the implementation SHA; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Every requirement row of the batch has a passing test at the implementation SHA.
- Layer boundaries hold (`intergrax/` does not import `agents/` or `applications/`).
- No new registry, scheduler, engine or authority beside the existing owner.
- Existing suites for the touched packages stay green (no weakened gate).
- Test log under `.tmp/proof/pdc-x-p5/B-D5/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d5 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

The implementer adds one more bounded `uv run pytest` over the existing unit suites of the touched packages and records it in the checkpoint.

## Out of scope

Other batches (in particular B-D6, which the dependency DAG places after wave 2); the `ollama` regression (B-REG-P); FRZ changes; OD signatures; merges; IIAC activation.
