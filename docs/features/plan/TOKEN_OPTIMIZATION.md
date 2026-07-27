<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Multi-layer Feature Plan

**Status:** Planned  
**Feature architecture (1:1):** [`../architecture/TOKEN_OPTIMIZATION.md`](../architecture/TOKEN_OPTIMIZATION.md)  
**Source audit instruction:** [`../../audit/TOKEN_OPTIMIZATION.md`](../../audit/TOKEN_OPTIMIZATION.md)  
**Primary anchor domain:** `CONTEXT_ENGINEERING`  
**Related domains:** `LLM_ADAPTERS`, `TOOLS`, `MEMORY`, `RAG`, `OBSERVABILITY`, `UNIFIED_EXECUTION_RUNTIME`, `AGENT_CONTRACTS_AND_ASSEMBLY`, `ADAPTIVE_HARNESS_INTELLIGENCE`

---

## Satellite registers (read on demand)

Large cross-domain sync registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) | domain plan cross-reference map, TOKEN row checklist, phase → plan mapping |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Cursor read scope (token budget)

Do not read the whole repository.

Default read scope for Token Optimization work:

1. `docs/features/architecture/TOKEN_OPTIMIZATION.md` (read-scope block only)
2. `docs/features/plan/TOKEN_OPTIMIZATION.md` (read-scope block + active TOKEN phase only)
3. The affected domain architecture/plan pair for the current TOKEN slice.
4. The minimal source files required by that domain plan item.

**On demand (one max):** [`plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) when syncing domain plan rows or cross-references.

Do not create `docs/plan/TOKEN_OPTIMIZATION.md`. This is a multi-layer feature plan, not a domain-layer plan.

**Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Planning model

This file coordinates cross-layer delivery. Concrete implementation rows must still be added to the owning domain plan files when a phase becomes actionable.

| TOKEN phase | Owning plan file |
|-------------|------------------|
| `TOKEN-ARCH-0` engine lifecycle, mechanisms, strategy taxonomy, config, plugins, claims | feature architecture + feature plan (docs-only) |
| `TOKEN-1` shared contracts, receipts, protected regions | feature plan + `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` |
| `TOKEN-2` OutputPolicy runtime | `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` and optional `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` |
| `TOKEN-3` ToolSchemaOptimizer | `docs/plan/TOOLS.md` |
| `TOKEN-4` ContextPackOptimizer | `docs/plan/CONTEXT_ENGINEERING.md` |
| `TOKEN-5` / `TOKEN-5A` MemorySummaryCompressor (helper-only first slice) | `docs/plan/MEMORY.md` |
| `TOKEN-6` telemetry and regression gates | `docs/plan/OBSERVABILITY.md` plus affected domain plans |
| `TOKEN-7` adaptive optimization | `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` |

**LKW proof workload:** LKW is the primary proof workload for Token Optimization. Token Optimization is **not** a local LKW feature — it is a cross-layer platform capability owned by runtime and domain plans. LKW proof must show measurable token savings, quality/regression safety, compression receipts, protected-region preservation, and observability attribution through the Harness Observability Spine. **LKW-PF6-0** proof design is **Done / Closed** (§LKW-PF6-0 below); **TOKEN-ARCH-0** engine architecture is **Done / Closed** (§TOKEN-ARCH-0 below); **TOKEN-1A** shared contracts is **Done / Closed** (§TOKEN-1A below). TOKEN implementation order remains **TOKEN-ARCH-0** → **TOKEN-1**..**TOKEN-7** below; LKW proof ordering introduces proof-design and baseline-measurement tasks around those phases (see [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PF — Recommended execution order).

---

## LKW-PF6-0 — Token Optimization proof design

**Status:** **Done / Closed** (docs-only).

**Maturity level:** proof design only — does not close `LKW-PF6` platform proof.

**Purpose:** Define exactly what the LKW Token Optimization proof must demonstrate before **TOKEN-1A** code starts. This section is the canonical source; [`applications/local_workspace_application/docs/PLATFORM_PROOF_LOOP.md`](../../applications/local_workspace_application/docs/PLATFORM_PROOF_LOOP.md) §10 and [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PF6-0 closeout mirror it for LKW scheduling.

**Narrative:** Intergrax proves that agent applications can be built as configurable, observable, cost-aware runtime systems — not hand-wired demos.

**Out of scope for LKW-PF6-0:** contracts, runtime behavior, optimizers, telemetry payloads, validators, benchmarks, fixtures, scripts, and any `TOKEN-*` implementation.

### Representative LKW workflows

All workflows use the existing LKW product proof shape:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

| Workflow ID | Description | Proof intent |
|-------------|-------------|--------------|
| **LKW-TOK-W1** | Small workspace indexing + search + synthesis | Minimal tenant-scoped baseline; compact corpus per-step token categories. |
| **LKW-TOK-W2** | Medium workspace search + synthesis with evidence | RAG/evidence/context-pack attribution under realistic retrieval load. |
| **LKW-TOK-W3** | Repeated synthesis with similar tool/catalog/context exposure | Recurring tool-catalog and context-pack savings across stable-exposure runs. |
| **LKW-TOK-W4** | Failure/safety-preserving run — exact regions must not be compressed | Optimization rejection/fallback when protected regions or safety boundaries would be violated. |

### Baseline measurement shape

Measured **before** optimization; must be reproducible enough to compare against optimized runs.

Required fields per measured step/run scope:

- `input_context_tokens`
- `tool_catalog_tokens`
- `retrieved_evidence_context_pack_tokens`
- `output_tokens`
- `total_tokens`
- `model`
- `provider`
- `runtime_profile`
- `workflow_id` (`LKW-TOK-W1` … `LKW-TOK-W4`)
- `run_id`
- `step_id`

### Optimized measurement shape

Later optimized proof runs must report:

- `baseline_token_usage` (per category)
- `optimized_token_usage` (per category)
- `saved_tokens`
- `saved_ratio`
- `optimization_strategy`
- `affected_source_category`
- `fallback_status`
- `validation_status`

### Token categories

Canonical categories:

| Category | Notes |
|----------|-------|
| input/context tokens | Assembled prompt/context before optimization |
| tool catalog tokens | LLM-facing tool schema/catalog view |
| RAG/evidence/context pack tokens | Retrieved fragments included in context |
| memory tokens | Memory summaries/blocks when in scope |
| output tokens | Model completion for the step |
| system/policy tokens | Where measurable separately |
| total tokens | Aggregate for the measured scope |

Attribution dimensions (required for later telemetry and public proof): `run`, `step`, `source`, `model`, `provider`, `strategy`, `output_profile`.

### Quality and regression criteria

Optimized run **fails** the proof if token savings break any of:

- tenant-scoped evidence
- evidence references
- synthesized answer integrity
- shadow artifact behavior
- safety boundaries
- exact protected regions
- platform abstraction boundaries

**Behavioral equivalence rule:** baseline and optimized results must remain behaviorally equivalent for the proof workload; only allowed formatting or verbosity differences are permitted.

### Protected-region requirements

Token Optimization must never lose or rewrite:

- code blocks, inline code, paths, URLs, env vars, enum values, hashes, dates, exact error strings, policy text, IDs, tenant identifiers required for correctness, evidence references

**TOKEN-1B** protected-region parser/validator is **Done / Closed** (§TOKEN-1B below). **LKW-PF6-0** defines proof requirements only.

### Compression receipt expectations

Future receipts ( **TOKEN-1C** ) must prove:

- original hash, optimized hash
- original token count, optimized token count
- saved tokens, saved ratio
- strategy
- protected-region validation status
- fallback reason when optimization is rejected

No receipt implementation in LKW-PF6-0.

### Observability visibility

Token savings must be visible through the **Harness Observability Spine** or an **approved domain-signal path**. No private Token Optimization telemetry bus.

Later proof attribution fields: `run_id`, `step_id`, `workflow_id`, `model`, `provider`, `profile`, `source/category`, `strategy`, `baseline_tokens`, `optimized_tokens`, `saved_tokens`, `saved_ratio`, `validation_status`, `fallback_status`.

Owner plan: [`docs/plan/OBSERVABILITY.md`](../../plan/OBSERVABILITY.md) Phase TOKEN-OBS; early slice **TOKEN-6A-lite** defines telemetry shape only.

### Public proof format (LKW-PF6-C target)

Later public-grade proof must include:

- representative workflow description
- baseline and optimized token usage
- saved tokens and saved ratio
- receipt references
- protected-region validation result
- quality/regression result
- observability attribution
- known limitations

**Redaction — must not expose:** raw prompts, raw documents, raw RAG chunks, raw synthesized content, tool args, secrets, tokens/secrets, absolute file paths, large raw artifacts.

### LKW-PF6-0 closure rule

Done / Closed when:

- [x] §Representative workflows, baseline/optimized shapes, categories, quality criteria, protected regions, receipts, observability, and public proof format are defined above.
- [x] **TOKEN-1A** remains not started.
- [x] No code/runtime/test/CI/dependency files changed.

**Next step:** **TOKEN-1A** — shared contracts + package skeleton (Phase TOKEN-1 below). Preceded by **TOKEN-ARCH-0** (§TOKEN-ARCH-0 below).

### LKW proof phase map (post-design)

| Phase | Scope | Depends on |
|-------|-------|------------|
| **LKW-PF6-A** | Baseline token measurement for §workflows | TOKEN-1A contracts (minimal) |
| **LKW-PF6-B** | First measurable savings proof | TOKEN-2/3 and/or TOKEN-4 light, TOKEN-1C receipts |
| **LKW-PF6-C** | Public-grade proof artifact | LKW-PF6-B + TOKEN-6A-lite/6B attribution |

---

## TOKEN-ARCH-0 — Token Optimization Engine architecture and mechanism strategy

**Status:** **Done / Closed** (docs-only).

**Purpose:** Define the Token Optimization Engine lifecycle, mechanism catalog, strategy taxonomy, configuration model, plugin/extensibility model, benchmark claim model, and first public proof mechanism selection before shared contracts are implemented.

**Canonical architecture:** [`../architecture/TOKEN_OPTIMIZATION.md`](../architecture/TOKEN_OPTIMIZATION.md) §8 Token Optimization Engine lifecycle, mechanisms, and extensibility.

**Out of scope for TOKEN-ARCH-0:** shared contract implementation, runtime behavior, optimizers, telemetry payloads, validators, benchmarks, fixtures, scripts, and any `TOKEN-1*` code.

### Acceptance

Done / Closed when:

- [x] engine lifecycle documented
- [x] mechanism catalog documented
- [x] strategy taxonomy documented
- [x] configuration model documented
- [x] plugin/extensibility model documented
- [x] benchmark claim model documented
- [x] first public proof candidate mechanisms documented
- [x] **TOKEN-1A** shared contracts — Done / Closed (§TOKEN-1A below)
- [x] no runtime/code/test/CI/dependency changes (TOKEN-ARCH-0 docs-only scope)

**Next step:** **TOKEN-OBS-1** HOS/domain-signal emission (per plan ordering).

---

## TOKEN-6B — Deterministic token regression benchmark runner

**Status:** **Done / Closed**.

**Purpose:** Add the first cross-source deterministic regression benchmark scaffold for helper-only Token Optimization optimizers.

**Deliverables:**

- `intergrax/runtime/token_optimization/regression.py` — fixture contracts, runner, default fixtures, summary/result dataclasses
- `scripts/check_token_regression_benchmarks.py` — local CI-friendly gate with optional `--json`, `--report`, `--report-json`, `--gate`, and `--gate-json` output
- `tests/unit/runtime/token_optimization/test_regression.py`

**Closeout:**

- deterministic token regression benchmark runner added
- default helper-only fixtures added for `tool_schema`, `context_pack`, and `memory_summary`
- CI-friendly local check script added
- summary/result dataclasses added
- receipt/validation/fallback expectations checked
- deterministic injected token counter used
- no model calls
- no external tokenizer dependency
- no HOS/domain-signal emission
- no observability exporter wiring
- no LKW proof execution

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/test_regression.py -q
uv run python scripts/check_token_regression_benchmarks.py
uv run python scripts/check_token_regression_benchmarks.py --json
```

**Next step:** **TOKEN-OBS-1A** domain signal model (first cautious slice of **TOKEN-OBS-1**); then full **TOKEN-OBS-1** / **TOKEN-OBS-2** observability emission and regression-gate reporting (per plan ordering).

**TOKEN-6B-R — strict validation expectation handling (refinement):**

- benchmark validation expectations now fail closed
- `expect_validation_pass=True` accepts only `passed` / `not_applicable` validation statuses
- `unknown`, `missing`, `failed`, `runner_error`, or unexpected statuses fail the fixture
- no HOS/exporter/runtime wiring was added
- no LLM-as-a-Judge implementation was added

---

## TOKEN-OBS-1A — Token optimization domain signal model (helper-only)

**Status:** **Done / Closed**.

**Purpose:** Add a helper-only, safe, structured token optimization domain signal layer that turns optimization outcomes and regression benchmark results into redaction-safe observability/domain signals and emits them into in-memory/no-op sinks for tests — without runtime hot-path integration or exporter wiring.

**Deliverables:**

- `intergrax/runtime/token_optimization/signals.py` — signal model, metadata sanitizer, builders, sinks, emission helper
- `tests/unit/runtime/token_optimization/test_signals.py`

**Closeout:**

- token optimization domain signal model added
- safe metadata sanitizer added
- builders added for optimization outcome/result and regression result
- in-memory and no-op signal sinks added
- no raw content/prompt/context/evidence emitted
- no HOS/domain-signal bus wiring
- no observability exporter wiring
- no runtime hot-path integration
- no LKW proof execution
- no LLM-as-a-Judge

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/test_signals.py -q
uv run pytest tests/unit/runtime/token_optimization/ -q
```

**Next step:** **TOKEN-OBS-1** HOS/domain-signal emission (remaining scope beyond helper-only signal model).

**TOKEN-OBS-1A-R refinement:**

- `receipt_ref` metadata is sanitized before being attached to signals
- raw content/prompt/context/evidence cannot bypass the sanitizer through `receipt_ref.metadata`
- receipt identity fields (`receipt_id`, `run_id`, `step_id`, `strategy_id`, `original_hash`, `optimized_hash`) are preserved
- no HOS/exporter/runtime wiring was added
- no LLM-as-a-Judge implementation was added

---

## TOKEN-OBS-1B — HOS domain-signal adapter for token optimization signals

**Status:** **Done / Closed**.

**Purpose:** Bridge the safe `TokenOptimizationSignal` model into the existing HOS/domain-signal path through an explicit helper — without runtime hot-path integration, exporter wiring, or auto-emission from optimizers or regression runners.

**Deliverables:**

- `intergrax/runtime/token_optimization/domain_events.py` — typed payload, registration, conversion, emission helper
- `tests/unit/runtime/token_optimization/test_domain_events.py`

**Closeout:**

- typed token optimization RuntimeEventPayload added
- token optimization domain event kind registration added
- safe TokenOptimizationSignal → RuntimeEventPayload conversion added
- explicit `emit_token_optimization_domain_signal(...)` helper added
- payload metadata and receipt_ref metadata are sanitized
- no optimizer auto-emission added
- no regression runner auto-emission added
- no runtime subscribers added
- no observability exporter wiring added
- no Elasticsearch/Kibana wiring added
- no LKW proof execution added
- no LLM-as-a-Judge implementation added

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/test_domain_events.py -q
uv run pytest tests/unit/runtime/token_optimization/test_signals.py -q
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run pytest tests/unit/runtime/events/test_domain_signals.py -q
uv run python scripts/check_token_regression_benchmarks.py
```

**Next step:** **TOKEN-OBS-1C** explicit opt-in emission helpers, then full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting (per plan ordering).

---

## TOKEN-OBS-1C — Explicit opt-in token optimization emission helpers

**Status:** **Done / Closed**.

**Purpose:** Combine the safe signal builders with the HOS domain-signal adapter through explicit opt-in helpers — without runtime hot-path integration, exporter wiring, or auto-emission from optimizers or regression runners.

**Deliverables:**

- `intergrax/runtime/token_optimization/emission.py` — explicit emission helpers and result type
- `tests/unit/runtime/token_optimization/test_emission.py`

**Closeout:**

- explicit opt-in emission helpers added
- optimization outcomes/results can be explicitly emitted through the safe domain-signal adapter
- regression results can be explicitly emitted through the safe domain-signal adapter
- optional dry-run/no-emit mode added
- metadata and receipt_ref metadata remain sanitized
- no optimizer auto-emission added
- no regression runner auto-emission added
- no runtime subscribers added
- no observability exporter wiring added
- no Elasticsearch/Kibana wiring added
- no LKW proof execution added
- no LLM-as-a-Judge implementation added

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/test_emission.py -q
uv run pytest tests/unit/runtime/token_optimization/test_domain_events.py -q
uv run pytest tests/unit/runtime/token_optimization/test_signals.py -q
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run pytest tests/unit/runtime/events/test_domain_signals.py -q
uv run pytest tests/unit/memory/ -q
uv run python scripts/check_token_regression_benchmarks.py
```

**Next step:** **TOKEN-OBS-1D** policy-gated runtime emission hook, then full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting (per plan ordering).

---

## TOKEN-OBS-1D — Policy-gated token optimization runtime emission hook

**Status:** **Done / Closed**.

**Purpose:** Add a small policy-gated runtime emission hook layer that future runtime call sites can use to emit token optimization signals safely — without runtime hot-path integration, exporter wiring, or auto-emission from optimizers or regression runners.

**Deliverables:**

- `intergrax/runtime/token_optimization/emission.py` — policy, status, and `maybe_emit_*` helpers
- `tests/unit/runtime/token_optimization/test_emission.py`

**Closeout:**

- policy-gated maybe_emit helpers added
- emission policy defaults to disabled
- enabled policy emits through the existing safe domain-signal adapter
- kind-level gates added for outcomes, regression results, and regression summaries
- dry-run policy mode added
- metadata and receipt_ref metadata remain sanitized
- no optimizer auto-emission added
- no regression runner auto-emission added
- no runtime subscribers added
- no observability exporter wiring added
- no Elasticsearch/Kibana wiring added
- no LKW proof execution added
- no LLM-as-a-Judge implementation added

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/test_emission.py -q
uv run pytest tests/unit/runtime/token_optimization/test_domain_events.py -q
uv run pytest tests/unit/runtime/token_optimization/test_signals.py -q
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run pytest tests/unit/runtime/events/test_domain_signals.py -q
uv run pytest tests/unit/memory/ -q
uv run python scripts/check_token_regression_benchmarks.py
```

**Next step:** **TOKEN-OBS-1E** regression emission wrapper, then full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting (per plan ordering).

---

## TOKEN-OBS-1E — Policy-gated regression benchmark emission wrapper

**Status:** **Done / Closed**.

**Purpose:** Add a thin wrapper around the deterministic token regression benchmark runner that optionally emits policy-gated domain signals for per-fixture results and aggregate summary — without modifying the core runner or benchmark script.

**Deliverables:**

- `intergrax/runtime/token_optimization/regression_emission.py` — `TokenRegressionEmissionRunResult`, `run_token_regression_benchmarks_with_emission`
- `tests/unit/runtime/token_optimization/test_regression_emission.py`

**Closeout:**

- wrapper calls existing `run_token_regression_benchmarks(...)` unchanged
- per-result emission via `maybe_emit_token_regression_result(...)` only from wrapper
- summary emission via `maybe_emit_token_regression_summary(...)` only from wrapper
- default policy disabled; `emit_results` / `emit_summary` flags gate attempts
- dry-run policy builds signals without recording events
- core `regression.py` and `scripts/check_token_regression_benchmarks.py` unchanged
- no optimizer auto-emission, no exporter wiring, no LKW proof, no LLM-as-a-Judge

**Next step:** **TOKEN-OBS-2B** regression fixture/eval matrix, then full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting (per plan ordering).

---

## TOKEN-OBS-2B — Regression fixture/eval matrix

**Status:** **Done / Closed**.

**Purpose:** Expand deterministic token regression benchmarks with a minimal eval matrix that proves both savings and safety behavior — without LKW integration or changes to the core benchmark script.

**Deliverables:**

- `intergrax/runtime/token_optimization/regression.py` — eval-matrix default fixtures, extended expectation fields, safe eval metadata on results
- `tests/unit/runtime/token_optimization/test_regression.py` — eval-matrix coverage for compactable, protected, and fallback cases
- `tests/unit/runtime/token_optimization/test_regression_report.py` — safe eval metadata on report items

**Closeout:**

- default fixtures now cover compactable, protected, and fallback eval cases across `tool_schema`, `context_pack`, and `memory_summary`
- expectations are explicit for pass/fail, validation, fallback, savings bounds, and receipt presence where applicable
- report items expose safe eval metadata (`eval_case`, `expected_behavior`, `expectation_status`) without raw fixture bodies
- `scripts/check_token_regression_benchmarks.py` unchanged and still exits 0 for the default suite
- no LKW integration, no exporter wiring, no LLM-as-a-Judge

**Next step:** full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting (per plan ordering).

---

## TOKEN-OBS-2A — Token regression benchmark report artifact

**Status:** **Done / Closed**.

**Purpose:** Add a small redaction-safe report builder for existing deterministic token regression benchmark results — without LKW integration, without modifying the core runner, and without changing the benchmark script.

**Deliverables:**

- `intergrax/runtime/token_optimization/regression_report.py` — `TokenRegressionReport`, `build_token_regression_report`, `token_regression_report_to_dict`, `format_token_regression_report`
- `tests/unit/runtime/token_optimization/test_regression_report.py`

**Closeout:**

- report builder consumes `TokenRegressionSummary` and optional `TokenRegressionEmissionRunResult`
- report items include only safe scalar benchmark fields (fixture id, source/category, strategy, token savings, validation/fallback, receipt id, pass/fail)
- emission section aggregates attempted emissions and status counts (`emitted`, `skipped_disabled`, `skipped_kind_disabled`, `dry_run`)
- metadata sanitized via existing `sanitize_signal_metadata`; no raw content, prompts, context fragments, evidence, fixture bodies, or event payloads
- deterministic output when `report_id` and `generated_at` are provided
- core `regression.py`, `regression_emission.py`, and `scripts/check_token_regression_benchmarks.py` unchanged and do not import the report module
- no LKW integration, no exporter wiring, no LLM-as-a-Judge

**Next step:** full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting (per plan ordering).

---

## TOKEN-OBS-2C — Regression gate thresholds

**Status:** **Done / Closed**.

**Purpose:** Add a formal gate artifact over existing deterministic token regression benchmark results — without LKW integration, without modifying the core runner, and without changing the benchmark script.

**Deliverables:**

- `intergrax/runtime/token_optimization/regression_gate.py` — `TokenRegressionGateThresholds`, `evaluate_token_regression_gate`, `token_regression_gate_to_dict`, `format_token_regression_gate`
- `tests/unit/runtime/token_optimization/test_regression_gate.py`

**Closeout:**

- gate evaluates `TokenRegressionSummary` with optional `TokenRegressionReport` cross-checks
- default thresholds pass the current 7-fixture benchmark suite
- stable failure reason codes: `fixture_failed`, `missing_receipt`, `expectation_not_met`, `unexpected_fallback`, `total_saved_ratio_below_threshold`, `total_saved_tokens_below_threshold`
- optional aggregate `min_total_saved_ratio` / `min_total_saved_tokens` thresholds
- metadata sanitized via existing `sanitize_signal_metadata`; no raw content, prompts, context, evidence, or event payloads in gate dict/formatter output
- core `regression.py`, `regression_report.py`, `regression_emission.py`, and `scripts/check_token_regression_benchmarks.py` unchanged and do not import the gate module
- no LKW integration, no exporter wiring, no LLM-as-a-Judge

**Next step:** full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting (per plan ordering).

---

## TOKEN-OBS-2D — Benchmark CLI report/gate output

**Status:** **Done / Closed**.

**Purpose:** Expose existing regression report and gate artifacts through the benchmark CLI — without LKW integration, without modifying core runner/report/gate modules, and without writing files.

**Deliverables:**

- `scripts/check_token_regression_benchmarks.py` — `--report`, `--report-json`, `--gate`, `--gate-json`, `--min-total-saved-ratio`, `--min-total-saved-tokens`
- `tests/unit/runtime/token_optimization/test_regression_cli.py`

**Closeout:**

- default script output and `--json` behavior unchanged
- `--report` / `--report-json` print redaction-safe report artifacts via existing report helpers
- `--gate` / `--gate-json` print redaction-safe gate artifacts via existing gate helpers; optional aggregate savings thresholds apply only in gate modes
- exit code remains non-zero on benchmark summary failures; gate modes also exit non-zero when gate status is `fail`
- mutually exclusive output mode flags fail fast with a clear CLI error
- no files written; no LKW integration, no exporter wiring, no LLM-as-a-Judge

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/test_regression_cli.py -q
uv run python scripts/check_token_regression_benchmarks.py
uv run python scripts/check_token_regression_benchmarks.py --report
uv run python scripts/check_token_regression_benchmarks.py --gate
```

**Next step:** full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting (per plan ordering).

---

## Diagnostic benchmark one-command flow

**Purpose:** Regenerate synthetic regression diagnostic artifacts, run the benchmark, and review results in one local step — without LKW integration or optimizer changes.

**Windows:**

```bat
scripts\token_optimization\run_token_regression_diagnostics.bat
```

**Shell:**

```sh
sh scripts/token_optimization/run_token_regression_diagnostics.sh
```

The wrapper:

1. Regenerates `.artifacts/token_optimization/regression_synthetic_v1`.
2. Runs the synthetic regression benchmark (`regression_synthetic_v1` fixture dataset).
3. Immediately runs the diagnostic artifact reviewer (`scripts/review_token_regression_artifacts.py`).

**Expected review status for `regression_synthetic_v1`:** `PASS WITH WARNINGS`. The `context_pack.long_workspace_document` fixture intentionally triggers truncation and dominant-savings warnings in the Top savings section (`[WARN]` on that case; smaller compaction cases show `[OK]`).

**Marketing note:** Do not describe the aggregate ~69–70% saved-token ratio as global compression quality. Most aggregate savings come from one long truncation case; use the reviewer’s per-case breakdown and safety checks for claims.

---

## TOKEN-OPT-3A — Stronger optimizer roadmap, algorithm inventory, and measurement sequencing

**Status:** **Done / Closed** (docs-only).

**Purpose:** Turn the next optimization phase into a sequential, measurable, platform-level roadmap. Do **not** build one large stronger optimizer. Build a sequential, plugin-friendly, policy-governed **Token Optimization Engine** where each algorithm/strategy is introduced as a separate measurable step.

**Canonical architecture:** [`../architecture/TOKEN_OPTIMIZATION.md`](../architecture/TOKEN_OPTIMIZATION.md) §8 Token Optimization Engine lifecycle, mechanisms, and extensibility.

**Out of scope for TOKEN-OPT-3A:** runtime optimizers, new mechanisms, corpus fixtures, benchmark changes, telemetry wiring, and any `TOKEN-OPT-3B+` implementation.

### Why this phase exists

**TOKEN-OBS-2** (diagnostic benchmark flow, regression report/gate artifacts, synthetic `regression_synthetic_v1` reviewer) proved the **diagnostic and measurement UX**: deterministic fixtures, receipt/validation/fallback expectations, per-case savings breakdown, and dominant-savings warnings. It did **not** yet prove advanced real-world optimization quality.

The current `ContextPackOptimizer` remains a conservative helper-only baseline: structural/light compaction, protected-region validation, receipts, fallback, and optional token counter. Aggregate savings from synthetic diagnostics must not be used as a broad public claim because they are dominated by one long truncation case (`context_pack.long_workspace_document`).

**The next phase should not jump directly to a large realistic corpus.** A realistic corpus is useful only after stronger mechanisms exist and can be measured separately.

### Existing implemented surfaces (baseline)

| Surface | Status | Notes |
|---------|--------|-------|
| **OutputPolicy runtime** | Done | `OutputPolicyResolver`; policy-only output shaping (`TOKEN-2`). |
| **ToolSchemaOptimizer** | Done | Helper-only schema/catalog compaction (`TOKEN-3`). |
| **ContextPackOptimizer** | Done | Light/structural compression only; no dedupe or budget packing (`TOKEN-4`). |
| **MemorySummaryCompressor** | Done | Helper-only first slice; staging, receipts, rollback metadata (`TOKEN-5A`). |
| **Telemetry / receipts / regression diagnostics** | Done | Contracts, receipts, protected regions, regression runner, report/gate CLI, synthetic diagnostic reviewer (`TOKEN-1*`, `TOKEN-6*`, `TOKEN-OBS-2*`). |

### Core sequencing decision

```text
Do not build one large stronger optimizer.
Build a sequential, plugin-friendly, policy-governed Token Optimization Engine
where each algorithm/strategy is introduced as a separate measurable step.
```

**Rule:** One optimization algorithm per task. Every algorithmic task must produce measurable attribution before the next algorithm is added.

**Recommended implementation order** (after current diagnostic baseline):

| Order | Task | Scope |
|-------|------|-------|
| 1 | **TOKEN-OPT-3A** | Roadmap and sequencing design (this section) — **Done / Closed** |
| 2 | **TOKEN-OPT-3B** | Priority-tiered context packing **contract** (data model only) — **Done / Closed** |
| 3 | **TOKEN-OPT-3C-A** | Optimization layer and pipeline composition **contract** — **Done / Closed** |
| 4 | **TOKEN-OPT-3C-B** | Deterministic exact deduplication layer — **Done / Closed** |
| 5 | **TOKEN-OPT-3D** | Budget-aware context packing prototype — **Done / Closed** |
| 6 | **TOKEN-OBS-3E** | Realistic corpus for stronger optimizer — **Done / Closed** (as part of **TOKEN-OBS-3E-F**) |
| 7 | **TOKEN-OBS-3F** | Baseline vs stronger optimizer comparison — **Done / Closed** (as part of **TOKEN-OBS-3E-F**) |
| 8 | **TOKEN-OBS-3G** | Safe public wording / proof claims — **Done / Closed** |
| 9 | **TOKEN-OPT-4A** | Extractive filtering layer — **Done / Closed** |
| 10 | **TOKEN-OPT-4B** | Extractive filtering evaluation / regression pack — **Done / Closed** |
| 11 | **TOKEN-OPT-5A** | Cache-prefix stabilization architecture / contract — **Done / Closed** |
| 12 | **TOKEN-OPT-5B** | Prompt-cache contracts and cache-prefix stability proof — **Done / Closed** (folds former **TOKEN-OPT-5C** / **TOKEN-OPT-5D**) |
| 13 | **TOKEN-OPT-5C** | Folded into **TOKEN-OPT-5B** functional block |
| 14 | **TOKEN-OPT-5D** | Folded into **TOKEN-OPT-5B** functional block |
| 15 | **TOKEN-OPT-5E** | Cache-aware compaction timing policy — **Done / Closed** |
| 16 | **TOKEN-7A** | Advisory recommendation contract and policy-only recommender — **Done / Closed** |
| 17 | **TOKEN-7B** | Advisory recommendation evaluation and report pack — **Done / Closed** |
| 18 | **TOKEN-7C** | Policy-gated advisory integration surface — **Done / Closed** |

Each algorithm ships as its own task, followed by measurement/review, before the next algorithm is layered in.

### Algorithm inventory

Vocabulary aligns with `intergrax/runtime/token_optimization/contracts.py` (`TokenOptimizationStrategyKind`, `StrategySafetyClass`, `TokenOptimizationProfile`, `TokenOptimizationSourceType`, `TokenOptimizationMechanism`).

| Algorithm / strategy | Primary source categories | Safety class | First allowed profile | Receipt | Protected-region validation | Regression measurement | Recommended phase | Initial status |
|----------------------|---------------------------|--------------|----------------------|---------|----------------------------|------------------------|-------------------|----------------|
| Measurement-only baseline | All (`TokenOptimizationSourceType`) | `measurement_only` | `measure_only` | Yes (bypass/fallback) | When content present | Yes | Baseline (shipped) | **Implemented** |
| Lossless normalization | `prompt`, `tool_catalog`, `rag_context_pack`, `memory` | `lossless` | `conservative` | Yes | Yes | Yes | TOKEN-3/4 (shipped) | **Implemented** (partial) |
| Lossless structural compaction | `rag_context_pack`, `retrieved_evidence`, `memory` | `lossless` | `conservative` | Yes | Yes | Yes | TOKEN-4 (shipped) | **Implemented** |
| Schema minimization | `tool_catalog` | `lossless` | `conservative` | Yes | Yes | Yes | TOKEN-3 (shipped) | **Implemented** |
| Exact deduplication | `rag_context_pack`, `conversation_history`, `tool_catalog` | `lossless` | `conservative` | Yes | Yes | Yes | **TOKEN-OPT-3C-B** | **Implemented** |
| Near-deduplication | `rag_context_pack`, `conversation_history` | `lossless` / `experimental` | `balanced` | Yes | Yes | Yes | Post-3C eval | Deferred |
| Priority-tier classification | `rag_context_pack`, `retrieved_evidence` | `lossless` | `conservative` | Yes | Yes | Yes | **TOKEN-OPT-3B** (contract) | **Done / Closed** |
| Budget-aware context packing | `rag_context_pack`, `retrieved_evidence` | `lossless` (default) | `conservative` | Yes | Yes | Yes | **TOKEN-OPT-3D** | **Implemented** (char-budget prototype) |
| Extractive filtering (tool/log/terminal) | `tool_output`, `terminal_output`, `log_output` | `lossy` (filter drops content) | `balanced` | Yes | Yes | Yes | **TOKEN-OPT-4A** | **Implemented** |
| Cache-prefix stabilization | `system_policy`, `prompt` | `lossless` | `conservative` | Yes | Light | Yes | **TOKEN-OPT-5A** → **TOKEN-OPT-5B** (contracts + synthetic proof) → **TOKEN-OPT-5E** | Architecture + contracts/helpers + compaction timing policy **Done / Closed**; runtime/provider integration deferred |
| Structured data compression | `structured_data` | `lossless` / `reversible` | `balanced` | Yes | Yes | Yes | TOKEN-4 extension | Deferred |
| Retrieval-on-demand | `rag_context_pack`, `retrieved_evidence` | `reversible` / `lossy` (partial) | `balanced` | Yes | Yes | Yes | RAG integration slice | Deferred |
| Safe lossy summarization | `memory`, `rag_context_pack` | `lossy` | `aggressive` (explicit `allow_lossy`) | Yes | Yes | Yes | Post-3D eval | **Excluded** from next slice |
| Semantic compression | `memory`, `rag_context_pack`, `conversation_history` | `lossy` / `experimental` | `experimental` | Yes | Yes | Yes | TOKEN-7+ | **Excluded** from next slice |
| Adaptive strategy recommendation | All (telemetry input) | `policy_only` | `balanced` | No (recommendation only) | N/A | Yes | **TOKEN-7** | Frozen |

### Measurement model — savings attribution by source and strategy

Aggregate savings alone are **not** enough. Every optimization task must report savings attributable to a single primary strategy (or explicit multi-strategy breakdown in receipt metadata when composition is unavoidable).

**Required separate attribution dimensions** (per run/step/source):

| Attribution bucket | Strategy kind / mechanism | Must not be mixed with |
|--------------------|----------------------------|------------------------|
| Whitespace / structural compaction savings | `lossless_structural_compression` | Truncation, dedupe, packing |
| Schema minimization savings | `schema_minimization` | Context packing, dedupe |
| Deduplication savings | `deduplication` | Truncation, packing |
| Budget-aware packing savings | `ranking_pruning` (packing tier drops) | Dedupe, truncation |
| Truncation savings | Lossy length cap (explicit `allow_lossy`) | Dedupe, packing, structural compaction |
| Output policy savings | `output_verbosity_shaping` | Source compression |
| Tool output / log filtering savings | `extractive_filtering` | RAG/context packing |
| Memory compression savings | `lossless_structural_compression` / future `safe_lossy_summarization` | Context dedupe |
| Cache / prefix savings | `cache_prefix_stabilization` | Content removal strategies |

**Public-claims rule:** Truncation-driven savings must **not** be mixed with deduplication or packing savings in public claims. Use per-case reviewer breakdown and strategy-attributed receipts (see §Diagnostic benchmark one-command flow marketing note).

Receipts and `TokenSavingsMeasurement` records must carry `strategy` (`TokenOptimizationStrategyRef`), `source_type`, and `category` so observability and regression gates can gate on attribution, not totals alone.

### Platform / plugin / application boundaries

| Principle | Rule |
|-----------|------|
| Engine ownership | **Token Optimization Engine** is platform-owned (`intergrax/runtime/token_optimization/`). |
| Application role | Applications provide workload, evidence, profiles, and validation expectations. Applications **do not** own optimizer algorithms. |
| LKW | LKW remains the **primary proof workload**, not the owner of the optimization engine. |
| Plugins | Strategies must be replaceable through platform/plugin contracts (`TokenOptimizationPluginDescriptor`, `TokenOptimizationPluginCapability`). |
| Plugin guardrails | Plugins must **not** bypass policy, protected-region validation, receipts, fallback, or observability. |
| Telemetry | No private telemetry bus. Emission through HOS or approved domain-signal path only. |
| Redaction | No raw prompts, raw documents, raw RAG chunks, tool args, secrets, or large raw artifacts in telemetry/receipts/reports. |

### Safety decisions (next implementation slices)

| Decision | Rule |
|----------|------|
| Semantic compression | **No** semantic compression in the next implementation slice. |
| LLM summarization | **No** LLM summarization in the next implementation slice. |
| Default lossiness | **No** lossy optimization by default (`allow_lossy=False` unless explicit). |
| Protected values | Protected values remain **must-keep**; `required=True` / `must_keep` fragments cannot disappear. |
| Receipts | Receipts are part of the product, not optional debug output (`emit_receipts` default on). |
| Validation failure | Validation failure must **fallback** to original or safer/lower optimization. |

### Next task definitions

#### TOKEN-OPT-3B — priority-tiered context packing contract

**Purpose:** Define the data contract for priority-tiered context packing **before** implementing packing behavior.

**Contracts added** (`intergrax/runtime/token_optimization/contracts.py`):

- `ContextFragmentPriority` — strongly typed priority tiers (`must_keep`, `high_priority`, `compressible`, `droppable`)
- `ContextPackingBudget` — token budget envelope with invariant validation
- `ContextPackingDecisionKind` — per-fragment action vocabulary (`keep`, `compact`, `deduplicate`, `drop`, `truncate`, `bypass`, `fallback`)
- `ContextPackingDecision` — per-fragment packing decision with token math validation
- `ContextDeduplicationMetadata` — cross-fragment duplicate linkage (contract only)
- `ContextFragmentPackingMetadata` — per-fragment packing metadata with fail-fast priority consistency checks
- `ContextPackingReceiptMetadata` — receipt explanation metadata for future receipt builders

**Closeout:**

- contracts added; no optimizer behavior changed
- no dedupe behavior added; no budget-aware packing behavior added
- `required=True` on existing `ContextFragment` remains compatible with future `must_keep` (conceptual predecessor; no migration in this task)
- priority tiers are strongly typed enums, not loose metadata strings
- contracts are plugin-friendly and application-independent (optional `TokenOptimizationStrategyRef`, extension metadata only in explicit `metadata` fields)
- vocabulary defined here is used by **TOKEN-OPT-3C-B** (exact dedupe, implemented) and **TOKEN-OPT-3D** (budget-aware packing)

**Status:** **Done / Closed**.

**Next step:** **TOKEN-OPT-3D** — budget-aware context packing prototype.

#### TOKEN-OPT-3C-A — optimization layer and pipeline composition contract

**Purpose:** Define strongly typed, extensible, plugin-friendly contracts for optimization layers and pipeline composition before implementing deduplication or a runtime engine.

**Contracts added** (`intergrax/runtime/token_optimization/contracts.py`):

- `TokenOptimizationLayerDecision` — per-layer outcome (`apply`, `bypass`, `fallback`, `override_previous`, `revert_to_original`, `failed`)
- `TokenOptimizationLayerDescriptor` — built-in/custom/plugin layer metadata
- `TokenOptimizationLayerContext` — pipeline position and layer lineage
- `TokenOptimizationLayerRequest` — `original_content` (immutable baseline) + `current_content` (working state after prior layers)
- `TokenOptimizationLayerResult` — explicit override metadata (`previous_changes_overridden`, `overridden_layer_ids`, `override_reason`)
- `TokenOptimizationLayer` — optional `Protocol` for layer implementations
- `TokenOptimizationLayerRef` — ordered layer reference for pipeline config
- `TokenOptimizationPipelineMode` — `default` (platform order) or `replace` (developer-provided list)
- `TokenOptimizationPipelineConfig` — composable pipeline of built-in and plugin/custom layers
- `TokenOptimizationPipelineResult` — aggregate pipeline outcome without execution logic

**Closeout:**

- contracts added; no dedupe implementation yet
- no pipeline runtime engine yet; no `ContextPackOptimizer` behavior changed
- every layer receives `original_content` and `current_content`; custom layers may override previous changes only explicitly
- pipeline order configurable via `DEFAULT` or `REPLACE`; developers can compose built-in and plugin/custom layer refs (e.g. `builtin.structural_compaction`, `custom.company.domain_dedupe`)
- deduplication will become the first concrete optimization layer in **TOKEN-OPT-3C-B**, not an ad-hoc flag in `ContextPackOptimizer`

**Status:** **Done / Closed**.

**Next step:** **TOKEN-OPT-3D** — budget-aware context packing prototype.

#### TOKEN-OPT-3C-B — deterministic exact deduplication layer

**Purpose:** Add the first stronger real reduction mechanism using deterministic exact deduplication.

**Rules:**

- Exact dedupe first; **no** semantic near-dedupe yet
- Required / `must_keep` fragments cannot disappear
- Receipt must explain suppressed/removed duplicates
- Dedupe savings measured **separately** from truncation

**Deliverables:**

- `intergrax/runtime/token_optimization/layers/exact_deduplication.py` — `ExactDeduplicationLayer`, `ExactDeduplicationLayerConfig`
- `intergrax/runtime/token_optimization/layers/__init__.py` — layer exports
- `tests/unit/runtime/token_optimization/test_exact_deduplication_layer.py`

**Closeout:**

- first concrete built-in `TokenOptimizationLayer` implementation
- layer implemented in its own file (no generic `layers.py`)
- exact line-based dedupe only; case-sensitive by default
- constructor config represents pipeline-level defaults; dynamic config override belongs to custom subclasses/implementations
- no env/config resolver added; no hidden env reads inside `optimize()`
- no semantic or near dedupe
- no pipeline runtime engine added; no `ContextPackOptimizer` behavior changed
- dedupe attribution is separated from truncation and structural compaction (`dedupe_saved_chars`, `duplicates_removed` in metadata)
- metadata exposes `base_config`, `effective_config`, `config_overrides`; `duplicate_groups` uses indices and key hashes only

**TOKEN-OPT-3C-B-R — formatting preservation refinement — Done / Closed**

- kept line endings are preserved (`splitlines(keepends=True)` + `"".join(kept_raw_lines)`)
- trailing newline is preserved when the final kept line originally ended with one
- `dedupe_saved_chars` is based on removed raw duplicate line lengths, not incidental newline normalization
- dedupe key ignores line ending; no algorithm expansion

**Status:** **Done / Closed**.

**Next step:** **TOKEN-OPT-3D** — budget-aware context packing prototype.

#### TOKEN-OPT-3D — budget-aware context packing prototype

**Purpose:** Pack structured context fragments into an explicit **estimated character budget** while preserving `must_keep` and preferring `high_priority` fragments. This is a **char-budget prototype** packing layer, not provider-aware token-budget optimization.

> **Important:** TOKEN-OPT-3D is a **char-budget prototype**. It must not be described as token-accurate optimization until a provider-aware tokenizer/counting adapter is introduced and measured.

**Rules:**

- `must_keep` survives; never dropped or truncated
- `high_priority` preferred over `compressible` and `droppable`
- `compressible` compacted only via safe whitespace normalization under budget pressure
- `droppable` removed first under pressure; excluded by default even when budget remains
- fallback if `must_keep` alone exceeds `max_chars` or protected-region validation fails
- packing savings reported at character level only (`budget_unit = "chars"`); no token counter

**Deliverables:**

- `intergrax/runtime/token_optimization/layers/budget_aware_packing.py` — `BudgetAwareContextPackingLayer`, `BudgetAwareContextPackingLayerConfig`, layer-local `BudgetAwarePackingInput` / `BudgetAwarePackingFragment`
- `intergrax/runtime/token_optimization/layers/__init__.py` — layer exports
- `tests/unit/runtime/token_optimization/test_budget_aware_packing_layer.py`

**Closeout:**

- standalone `BudgetAwareContextPackingLayer` implemented in its own file (no generic `layers.py`)
- char-budget prototype only; `budget_unit = "chars"`; `max_chars` is an estimated character budget, not a token budget
- no provider tokenizer / no model-specific token counter
- operates on typed layer-local `BudgetAwarePackingInput` / `BudgetAwarePackingFragment`
- temporary prototype payload passed through `request.metadata["packing_input"]` until a future engine routes structured layer payloads directly
- `must_keep` fragments are never dropped
- `droppable` fragments are dropped first under pressure
- `compressible` fragments may only use safe whitespace compaction (no semantic summarization, no partial truncation)
- no pipeline runtime engine added; no `ContextPackOptimizer` behavior changed
- no benchmark/public claim update

**Status:** **Done / Closed**.

**Next step:** public wording follows [`docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md`](../../public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md) (**TOKEN-OBS-3G** — Done / Closed).

#### TOKEN-OBS-3E-F — stronger optimizer evaluation pack

**Purpose:** Combine realistic synthetic corpus and deterministic baseline-vs-stronger comparison for the stronger optimizer mechanisms already implemented (`ExactDeduplicationLayer`, `BudgetAwareContextPackingLayer`).

**Scope:**

- internal evaluation only; synthetic data only; no private/real customer data
- direct evaluation of `ExactDeduplicationLayer` and `BudgetAwareContextPackingLayer` (no production pipeline engine)
- no benchmark CLI / no CI gate
- char-level metrics only (`baseline_chars`, `stronger_chars`, `saved_chars`, `strategy_savings_chars`)
- strategy-separated attribution (`deduplication`, `budget_aware_packing`, `fallback`, `no_op`)
- no token-accurate claims; no public wording / marketing claims
- prepares inputs for **TOKEN-OBS-3G**

**Deliverables:**

- `tests/fixtures/token_optimization/stronger_optimizer_corpus.py` — synthetic corpus + evaluation-only helper
- `tests/unit/runtime/token_optimization/test_stronger_optimizer_evaluation_pack.py` — corpus validation and behavior tests

**Closeout:**

- combines **TOKEN-OBS-3E** (realistic synthetic corpus) and **TOKEN-OBS-3F** (baseline vs stronger comparison) into one internal evaluation pack
- evaluation report is raw-content-safe (no case content in report fields)
- no `TokenOptimizationEngine`, layer registry, or benchmark CLI added

**Status:** **Done / Closed**.

**Next step:** follow [`docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md`](../../public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md) for any public wording derived from this proof.

#### TOKEN-OBS-3G — safe public wording / proof claims

**Purpose:** Add safe, bounded, non-marketing public claim guardrails for the stronger optimizer proof (TOKEN-OPT-3C-B, TOKEN-OPT-3D, TOKEN-OBS-3E-F).

**Closeout:**

- public claim guardrails added in `docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md`
- approved / conditional / forbidden wording documented
- numeric claims require explicit evidence checklist
- current proof remains synthetic-corpus and char-level only
- no runtime, benchmark, script, application, or layer changes

**Status:** **Done / Closed**.

#### TOKEN-OPT-3D-R — char-budget metadata naming refinement

**Purpose:** Remove misleading token-named receipt fields from the char-budget prototype metadata.

**Closeout:**

- removed token-named receipt fields from char-budget prototype metadata (`context_packing_receipt` and token-named totals/decision fields)
- retained char-level metadata only (`budget_unit`, `max_chars`, `packing_decisions` with `original_chars` / `output_chars`, etc.)
- no algorithm change
- no token counter / provider-aware tokenizer added

> **Important:** TOKEN-OPT-3D remains a **char-budget prototype**. It must not be described as token-accurate optimization until a provider-aware tokenizer/counting adapter is introduced and measured.

**Status:** **Done / Closed**.

#### TOKEN-OPT-4A — extractive filtering layer for tool / terminal / log output

**Purpose:** Add a deterministic extractive filtering layer for noisy tool, terminal, and log output without semantic compression or LLM summarization.

**Deliverables:**

- `intergrax/runtime/token_optimization/layers/extractive_filtering.py` — `ExtractiveFilteringLayer`, `ExtractiveFilteringLayerConfig`
- `intergrax/runtime/token_optimization/layers/__init__.py` — layer exports
- `tests/unit/runtime/token_optimization/test_extractive_filtering_layer.py`

**Closeout:**

- deterministic extractive filtering layer added
- targets `tool_output`, `terminal_output`, and `log_output`
- preserves head/tail, important error/warning lines, and traceback blocks
- collapses repeated lines deterministically
- emits char-level metadata only
- does not create token-accurate savings claims
- does not use LLM summarization
- does not use semantic compression
- does not add runtime pipeline engine
- does not change existing optimizer behavior
- savings attribution remains separate from dedupe, packing, and truncation

**Status:** **Done / Closed**.

**Next step:** **TOKEN-OPT-4B** — extractive filtering evaluation cases / regression pack.

#### TOKEN-OPT-4B — extractive filtering evaluation cases / regression pack

**Purpose:** Add a synthetic, raw-content-safe evaluation / regression pack proving that `ExtractiveFilteringLayer` safely filters noisy tool/terminal/log output while preserving failures, warnings, tracebacks, and protected regions with char-level attribution only.

**Deliverables:**

- `tests/fixtures/token_optimization/extractive_filtering_corpus.py`
- `tests/unit/runtime/token_optimization/test_extractive_filtering_evaluation_pack.py`

**Closeout:**

- synthetic evaluation corpus added for ExtractiveFilteringLayer
- direct evaluation of the real ExtractiveFilteringLayer added
- tool_output / terminal_output / log_output cases covered
- verbose progress noise case covered
- pytest failure evidence preservation covered
- traceback preservation covered
- repeated warning collapse coverage added
- protected-region fallback case covered
- short clean output bypass/no-op case covered
- safe report builder emits char-level metadata only
- strategy attribution remains extractive_filtering / fallback / no_op only
- reports are raw-content-safe
- no token-accurate claims added
- no runtime pipeline engine added
- no README or public adoption documentation updated
- next step: TOKEN-OPT-5A — cache-prefix stabilization architecture / contract — **Done / Closed**

**Status:** **Done / Closed**.

#### TOKEN-OPT-5A — cache-prefix stabilization architecture / contract

**Status:** **Done / Closed**.

**Purpose:** Define prompt-cache-aware optimization boundaries, provider responsibilities, cache attribution vocabulary, cache-safe prompt/thread assembly invariants, and cache-aware compaction timing rules before any provider/runtime cache implementation.

**Deliverables:**

- canonical architecture section in `docs/features/architecture/TOKEN_OPTIMIZATION.md`
- detailed cache-prefix stabilization addendum update in `docs/features/architecture/TOKEN_OPTIMIZATION_CACHE_PREFIX_STABILIZATION.md`

**Closeout:**

- prompt caching classified as cost/latency optimization, not content reduction
- cache-prefix stabilization documented as a first-class Token Optimization surface
- stable prefix and dynamic tail boundaries documented
- volatile prefix inputs documented as forbidden for cache-stable prefixes
- append-only prompt/thread invariant documented
- cache-safe and cache-hostile prompt/thread behaviors documented
- tool envelope cache-stability rule documented
- provider-specific prompt cache behavior assigned to `LLM_ADAPTERS`
- Token Optimization ownership limited to shared policy, attribution vocabulary, and safety boundaries
- cache metrics separated from content-reduction savings
- cache-aware compaction timing rule documented
- in-cache compaction explicitly deferred
- no runtime prompt assembly changes
- no provider API calls
- no adapter wiring
- no tests or benchmark runners added
- no README or public adoption documentation updated
- next step: TOKEN-OPT-5B — provider cache policy and capability contract

#### TOKEN-OPT-5B — prompt-cache contracts and cache-prefix stability proof

**Status:** **Done / Closed**.

**Purpose:** Add provider prompt-cache contracts, helper-level stable-prefix/dynamic-tail modeling, append-only prefix invariant checks, and synthetic prefix-stability evaluation without provider API calls or runtime prompt assembly.

**Deliverables:**

- provider prompt-cache contract types
- provider cache invalidation reason vocabulary
- helper-level prompt cache block/snapshot/stability model
- deterministic prefix hashing
- append-only prefix invariant helper
- synthetic prefix-stability corpus
- focused contract and prefix-stability tests

**Closeout:**

- prompt cache mode enum added
- cache invalidation reason enum added
- provider cache capabilities contract added
- prompt cache policy contract added
- provider cache usage snapshot contract added
- prompt cache attribution contract added
- stable prefix / dynamic tail helper model added
- deterministic prefix hash helper added
- append-only prefix invariant helper added
- synthetic cache-prefix stability corpus added
- prefix stability and invalidation cases covered
- cache usage fields kept separate from content-reduction fields
- provider/cache attribution does not compute token savings from cache reads
- no provider API calls added
- no runtime prompt assembly changes
- no adapter wiring added
- no observability emission added
- no benchmark runner added
- no semantic compression or LLM summarization added
- no README or public adoption documentation updated
- next step: TOKEN-OPT-5E — cache-aware compaction timing policy

#### TOKEN-OPT-5E — cache-aware compaction timing policy

**Status:** **Done / Closed**.

**Purpose:** Add a provider-neutral policy/helper layer deciding when compaction should run, defer, bypass, or require manual review based on cache prefix stability, cache hotness, TTL proximity, expected content-reduction benefit, expected cache invalidation cost, and safety risk.

**Deliverables:**

- cache-aware compaction target enum
- cache-aware compaction decision enum
- cache-aware compaction reason enum
- timing input and decision contracts
- deterministic compaction timing decision helper
- synthetic cache-aware compaction corpus
- focused policy tests
- architecture/addendum/plan documentation updates

**Closeout:**

- cache-aware compaction target contract added
- cache-aware compaction decision contract added
- cache-aware compaction reason contract added
- timing input and decision contracts added
- deterministic compaction timing helper added
- dynamic-tail-safe reduction path covered
- cold-history compaction path covered
- hot stable-prefix defer path covered
- near-expiry stable-prefix run path covered
- unstable-prefix defer path covered
- low-benefit bypass path covered
- full-thread rewrite review path covered
- protected/semantic risk review path covered
- synthetic compaction timing corpus added
- decision reports remain raw-content-safe
- no provider API calls added
- no runtime prompt assembly changes
- no adapter wiring added
- no observability emission added
- no benchmark runner added
- no semantic compression or LLM summarization added
- no in-cache compaction added
- no README or public adoption documentation updated

#### TOKEN-7A — advisory recommendation contract and policy-only recommender

**Status:** **Done / Closed**.

**Purpose:** Add a provider-neutral, policy-only advisory layer that recommends Token Optimization posture changes from redaction-safe scalar signals without auto-applying optimizations.

**Deliverables:**

- recommendation action enum
- recommendation reason enum
- recommendation confidence enum
- advisory signal contract
- advisory recommendation contract
- deterministic recommendation helper
- synthetic advisory recommendation corpus
- focused advisory recommendation tests
- architecture / feature plan / AHI plan documentation updates

**Closeout:**

- advisory recommendation action contract added
- advisory recommendation reason contract added
- advisory recommendation confidence contract added
- safe advisory signal contract added
- safe advisory recommendation contract added
- deterministic policy-only recommendation helper added
- protected-region risk escalation path covered
- quality-regression escalation path covered
- regression-gate failure review path covered
- insufficient-data path covered
- high-fallback disable-strategy path covered
- hot stable-cache preserve-prefix path covered
- invalidated-cache review path covered
- dynamic-tail reduction recommendation path covered
- measured-safe-savings enable-strategy path covered
- low-savings keep-current path covered
- synthetic advisory recommendation corpus added
- recommendation reports remain raw-content-safe
- auto_apply_allowed remains False
- no provider API calls added
- no runtime prompt assembly changes
- no adaptive runtime integration added
- no observability/HOS emission added
- no semantic compression or LLM summarization added
- no README or public adoption documentation updated

**Roadmap/order update:**

```text
TOKEN-OPT-5A — Done / Closed
TOKEN-OPT-5B — Done / Closed
TOKEN-OPT-5C — folded into TOKEN-OPT-5B functional block
TOKEN-OPT-5D — folded into TOKEN-OPT-5B functional block
TOKEN-OPT-5E — Done / Closed
TOKEN-7A — Done / Closed
TOKEN-7B — Done / Closed
TOKEN-7 — broader runtime/adaptive integration remains future work; no production auto-apply
```

**Next decision:** choose the next Token Optimization-only block; production auto-apply remains forbidden until explicitly designed and reviewed.

#### TOKEN-7C — policy-gated advisory integration surface

**Status:** **Done / Closed**.

**Purpose:** Add a deterministic policy gate around the Token Optimization advisory recommender so recommendations can be blocked, returned as report-only, returned as dry-run, escalated to review, or marked recommendation-ready without auto-applying optimizations.

**Deliverables:**

- advisory integration mode enum
- advisory integration status enum
- advisory gate reason enum
- advisory integration policy contract
- advisory integration request contract
- advisory integration result contract
- deterministic policy-gated advisory helper
- redaction-safe integration result dict helper
- deterministic integration result text formatter
- synthetic advisory integration corpus
- focused advisory integration tests
- architecture / feature plan documentation updates

**Closeout:**

- advisory integration mode contract added
- advisory integration status contract added
- advisory gate reason contract added
- advisory integration policy contract added
- advisory integration request contract added
- advisory integration result contract added
- deterministic policy-gated advisory helper added
- policy disabled / mode disabled path covered
- report-only path covered
- dry-run path covered
- review-only path covered
- advisory-allowed path covered
- insufficient-signals path covered
- strategy-enable blocked/allowed paths covered
- strategy-disable blocked/review paths covered
- synthetic advisory integration corpus added
- integration result serialization remains raw-content-safe
- auto_apply_allowed remains False across all integration results
- no global config resolver added
- no env/YAML config resolver added
- no provider API calls added
- no runtime prompt assembly changes
- no adaptive runtime integration added
- no observability/HOS emission added
- no benchmark CLI runner added
- no semantic compression or LLM summarization added
- no README or public adoption documentation updated

**Roadmap/order update:**

```text
TOKEN-7A — Done / Closed
TOKEN-7B — Done / Closed
TOKEN-7C — Done / Closed
TOKEN-7 — broader runtime/adaptive integration remains future work; no production auto-apply
```

**Next decision:** choose the next Token Optimization-only block; production auto-apply remains forbidden until explicitly designed and reviewed.

#### TOKEN-7B — advisory recommendation evaluation and report pack

**Status:** **Done / Closed**.

**Purpose:** Add a redaction-safe evaluation runner and report pack for the policy-only advisory recommender, proving deterministic recommendations, expected safety behavior, non-auto-apply status, and raw-content-safe reporting.

**Deliverables:**

- advisory evaluation case contract
- advisory evaluation result contract
- advisory evaluation summary contract
- advisory evaluation report contract
- deterministic advisory evaluation runner
- redaction-safe report dict formatter
- deterministic text report formatter
- synthetic advisory evaluation corpus
- focused advisory evaluation tests
- architecture / feature plan documentation updates

**Closeout:**

- advisory evaluation case contract added
- advisory evaluation result contract added
- advisory evaluation summary contract added
- advisory evaluation report contract added
- deterministic single-case evaluation helper added
- deterministic multi-case evaluation helper added
- redaction-safe advisory report dict helper added
- deterministic advisory report text formatter added
- synthetic advisory evaluation corpus added
- all expected advisory scenarios covered
- summary counts pass/fail/manual-review/insufficient-data/non-auto-apply/raw-content-safe results
- advisory reports remain raw-content-safe
- auto_apply_allowed remains False across evaluated recommendations
- no raw signal or recommendation objects emitted in reports
- no provider API calls added
- no runtime prompt assembly changes
- no adaptive runtime integration added
- no observability/HOS emission added
- no benchmark CLI runner added
- no semantic compression or LLM summarization added
- no README or public adoption documentation updated

**Roadmap/order update:**

```text
TOKEN-7A — Done / Closed
TOKEN-7B — Done / Closed
TOKEN-7 — broader runtime/adaptive integration remains future work; no production auto-apply
```

### TOKEN-OPT-3A acceptance

Done / Closed when:

- [x] §Why this phase exists, §Existing surfaces, §Algorithm inventory, §Sequencing, §Measurement model, §Boundaries, §Safety decisions, and §Next task definitions are documented above.
- [x] Aligns with feature architecture §8 and `contracts.py` vocabulary.
- [x] Each future algorithm is a separate task with measurement/review expectation.
- [x] LKW described as proof workload only.
- [x] No runtime/code/test/benchmark/script/application changes.

**Next step:** **TOKEN-OPT-3D** — budget-aware context packing prototype.

---

## TOKEN-6A — Telemetry payloads/counters for TOKEN-2..4

**Status:** **Done / Closed**.

**Purpose:** Add helper-only telemetry summary/counter payloads for TOKEN-2 OutputPolicyResolver, TOKEN-3 ToolSchemaOptimizer, and TOKEN-4 ContextPackOptimizer.

**Deliverables:**

- `intergrax/runtime/token_optimization/telemetry.py` — counter snapshot, summary payload, validation, attribute mapping
- `tests/unit/runtime/token_optimization/test_telemetry.py`

**Closeout:**

- helper-only telemetry summary/counter payloads added for TOKEN-2..4
- counter snapshot added
- token optimization summary payload added
- safe namespaced summary attribute mapping added
- validation helper added
- aggregates receipts, output-policy resolutions, tool schema outcomes, and context pack outcomes
- deduplicates receipts by receipt_id
- no HOS emission added
- no observability exporter wiring added
- no runtime event emission added
- no tokenizer/model calls added
- next step: **TOKEN-OBS-1** HOS/domain-signal emission according to plan ordering

**TOKEN-6A-R refinement:**

- summary metadata hardening added
- summary metadata passthrough is allow-listed
- forbidden/raw-content-like metadata is dropped from summaries and attributes
- summary validation rejects unsafe metadata
- no HOS emission/exporter wiring added

---

## TOKEN-6A-lite — Token savings telemetry shape

**Status:** **Done / Closed**.

**Purpose:** Add typed token-savings telemetry payload contracts and helpers for future HOS/domain-signal emission.

**Deliverables:**

- `intergrax/runtime/token_optimization/telemetry.py`
- `tests/unit/runtime/token_optimization/test_telemetry.py`

**Closeout:**

- token savings telemetry payload shape added
- payload builder from CompressionReceipt added
- payload validation helper added
- safe namespaced attribute mapping helper added
- no telemetry emission added
- no HOS/runtime/exporter wiring added
- no public proof path changed
- next step: **TOKEN-4** ContextPackOptimizer — Done / Closed (§Phase TOKEN-4 below)

---

## TOKEN-1C — Compression receipts + validation helpers

**Status:** **Done / Closed**.

**Purpose:** Add deterministic compression receipt creation and receipt integrity validation helpers for Token Optimization.

**Deliverables:**

- `intergrax/runtime/token_optimization/receipts.py`
- `tests/unit/runtime/token_optimization/test_receipts.py`

**Closeout:**

- compression receipt data model added
- deterministic content hashing added
- receipt builder added
- receipt ref helper added
- receipt integrity validator added
- uses TOKEN-1A contracts
- records TOKEN-1B protected-region validation results when supplied
- no optimization behavior added
- no token counting added
- no telemetry wiring added
- next step: **TOKEN-6A-lite** — **Done / Closed** (§TOKEN-6A-lite above)

---

## TOKEN-1B — Protected region parser/validator

**Status:** **Done / Closed**.

**Purpose:** Add deterministic protected-region detection and validation helpers as the first safety gate for Token Optimization.

**Deliverables:**

- `intergrax/runtime/token_optimization/protected_regions.py`
- `tests/unit/runtime/token_optimization/test_protected_regions.py`

**Closeout:**

- protected-region detection helper added
- protected-region validation helper added
- uses TOKEN-1A contracts
- no optimization behavior added
- no receipts added
- no telemetry wiring added

**Next step:** **TOKEN-1C** — compression receipts + validation helpers — **Done / Closed** (§TOKEN-1C above).

---

## TOKEN-1B-R — Protected terms refinement

**Status:** **Done / Closed**.

**Purpose:** Replace broad `ENV_VAR` regex guessing with explicit protected-term matching before **TOKEN-1C**.

**Closeout:**

- broad `ENV_VAR` regex guessing removed
- `ENV_VAR` protection now uses built-in protected terms + env extension + explicit `protected_terms`
- env extension variable: `INTERGRAX_TOKEN_OPTIMIZATION_PROTECTED_TERMS`
- env extension extends built-ins, does not replace them
- no runtime optimization behavior added
- no public proof path changed
- next step: **TOKEN-1C** — **Done / Closed** (§TOKEN-1C above)

---

## TOKEN-1A — Shared contracts + package skeleton

**Status:** **Done / Closed**.

**Purpose:** Add shared Token Optimization contract vocabulary and a minimal runtime package skeleton for later phases (TOKEN-1B..TOKEN-7).

**Deliverables:**

- `intergrax/runtime/token_optimization/__init__.py`
- `intergrax/runtime/token_optimization/contracts.py`
- `tests/unit/runtime/token_optimization/test_contracts.py`

**Closeout:**

- shared package skeleton added
- shared contracts added (profiles, policy, attribution, mechanisms, strategies, plugin descriptors, measurements, protected regions, receipt refs, request/result)
- plugin descriptor contracts added
- no runtime optimization behavior added
- no telemetry wiring added

**Next step:** **TOKEN-1C** — compression receipts + validation helpers — **Done / Closed** (§TOKEN-1C above).

---

## Implementation Blueprint

### Target runtime component layout

```text
intergrax/runtime/token_optimization/
  __init__.py
  contracts.py                 # shared DTOs, enums, policies
  output_policy.py             # OutputPolicyResolver and output profiles
  protected_regions.py         # protected region parser + validator
  receipts.py                  # CompressionReceipt builders/validators
  optimizer.py                 # TokenOptimizer orchestrator
  telemetry.py                 # HOS/domain-signal/metric emission helpers
  regression.py                # token-vs-quality benchmark helpers

intergrax/runtime/nexus/context/
  context_pack_optimizer.py    # CE integration after rank/budget, before format/preflight

intergrax/runtime/nexus/tools/
  tool_schema_optimizer.py     # compact LLM-facing tool catalog view

intergrax/memory/
  summary_compressor.py        # staging + validation + receipt + rollback for memory summaries
```

### Shared contracts to implement first

The first implementation slice must add contracts only, without wiring behavior into hot paths.

Required contracts:

- `OutputProfile`
- `CompressionLevel`
- `TokenOptimizationBypassReason`
- `TokenOptimizationSourceType`
- `ProtectedRegionKind`
- `TokenOptimizationPolicy`
- `OutputPolicy`
- `CompressionReceipt`
- `ProtectedRegion`
- `ProtectedRegionValidationResult`
- `TokenOptimizationRequest`
- `TokenOptimizationResult`
- `TokenOptimizationTelemetry`

Rules:

- Prefer frozen dataclasses with `slots=True` unless an existing domain requires Pydantic.
- Every runtime example using `RuntimeState` must explicitly pass `run_id`.
- Trace event calls must use `TraceLevel` enum where applicable.
- New Python files must start with the Intergrax copyright header.

### Implementation order

```text
LKW-PF6-0   Token Optimization proof design — Done / Closed
TOKEN-ARCH-0  Token Optimization Engine architecture and mechanism strategy — Done / Closed
TOKEN-1A    shared contracts + package skeleton — Done / Closed
TOKEN-1B    protected region parser/validator — Done / Closed
TOKEN-1C    compression receipts + validation helpers — Done / Closed
TOKEN-6A-lite  token savings telemetry payload shape — Done / Closed
TOKEN-2     OutputPolicy runtime resolver — Done / Closed
TOKEN-3     ToolSchemaOptimizer compact catalog view — Done / Closed
TOKEN-4     ContextPackOptimizer light/structural compression only — Done / Closed
TOKEN-6A    telemetry payloads/counters for TOKEN-2..4 — Done / Closed
TOKEN-5     MemorySummaryCompressor with staging/rollback — Planned
TOKEN-5A    MemorySummaryCompressor helper-only first slice — Done / Closed
TOKEN-6B    token regression benchmark runner + CI scripts — Done / Closed
TOKEN-OBS-1A domain signal model + safe in-memory emission — Done / Closed
TOKEN-OBS-1B HOS domain-signal adapter for token optimization signals — Done / Closed
TOKEN-OBS-1C explicit opt-in token optimization emission helpers — Done / Closed
TOKEN-OBS-1D policy-gated token optimization runtime emission hook — Done / Closed
TOKEN-OBS-1E policy-gated regression benchmark emission wrapper — Done / Closed
TOKEN-OBS-2A token regression benchmark report artifact — Done / Closed
TOKEN-OBS-2B regression fixture/eval matrix — Done / Closed
TOKEN-OBS-2C regression gate thresholds — Done / Closed
TOKEN-OBS-2D benchmark CLI report/gate output — Done / Closed
TOKEN-OPT-3A stronger optimizer roadmap, algorithm inventory, measurement sequencing — Done / Closed
TOKEN-OPT-3B priority-tiered context packing contract — Done / Closed
TOKEN-OPT-3C-A optimization layer and pipeline composition contract — Done / Closed
TOKEN-OPT-3C-B deterministic exact deduplication layer — Done / Closed
TOKEN-OPT-3D budget-aware context packing prototype — Done / Closed
TOKEN-OBS-3E realistic corpus for stronger optimizer — Done / Closed as part of TOKEN-OBS-3E-F
TOKEN-OBS-3F baseline vs stronger optimizer comparison — Done / Closed as part of TOKEN-OBS-3E-F
TOKEN-OBS-3G safe public wording / proof claims — Done / Closed
TOKEN-OPT-4A extractive filtering layer — Done / Closed
TOKEN-OPT-4B extractive filtering evaluation cases / regression pack — Done / Closed
TOKEN-OPT-5A cache-prefix stabilization architecture / contract — Done / Closed
TOKEN-OPT-5B prompt-cache contracts and cache-prefix stability proof — Done / Closed
TOKEN-OPT-5C folded into TOKEN-OPT-5B functional block
TOKEN-OPT-5D folded into TOKEN-OPT-5B functional block
TOKEN-OPT-5E cache-aware compaction timing policy — Done / Closed
TOKEN-7A    advisory recommendation contract and policy-only recommender — Done / Closed
TOKEN-7B    advisory recommendation evaluation and report pack — Done / Closed
TOKEN-7C    policy-gated advisory integration surface — Done / Closed
TOKEN-7     adaptive recommendations from telemetry, no auto-apply by default
```

TOKEN-7 — broader runtime/adaptive integration remains future work; no production auto-apply. Runtime/adaptive integration remains deferred.

Semantic compression is deliberately delayed until protected-region validation, receipts, telemetry, and regression gates exist. **TOKEN-OPT-3A** sequences stronger mechanisms one algorithm per task (§TOKEN-OPT-3A); semantic compression and LLM summarization remain excluded from the next implementation slice.

---

## Phase TOKEN-1 — Shared contracts, receipts, and protected regions

**Goal:** Establish the safe foundation used by all later Token Optimization slices.

**Owner layer:** `UNIFIED_EXECUTION_RUNTIME` for runtime policy placement; feature plan for shared contracts; `OBSERVABILITY` consulted for receipt/telemetry shape.

**Dependencies:** feature architecture accepted; **TOKEN-ARCH-0** closed.

**Deliverables:**

- `intergrax/runtime/token_optimization/__init__.py`
- `intergrax/runtime/token_optimization/contracts.py`
- `intergrax/runtime/token_optimization/protected_regions.py`
- `intergrax/runtime/token_optimization/receipts.py`
- unit tests for contracts, protected-region validation, and receipt hashing,
- lightweight CI script `scripts/check_token_optimization_contracts.py`.

**Acceptance criteria:**

- contract imports do not import CE/TOOLS/MEMORY hot-path modules,
- protected-region validator detects and preserves code, inline code, paths, URLs, env vars, enum values, hashes, dates, and exact error strings,
- receipt contains original hash, optimized hash, original tokens, optimized tokens, saved tokens, saved ratio, validation status, fallback flag,
- failed validation produces a fallback result rather than optimized content,
- no hot-path runtime behavior changes yet.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run python scripts/check_token_optimization_contracts.py
```

**Domain plan rows:** `TOKEN-UER-1` in `docs/plan/UNIFIED_EXECUTION_RUNTIME.md`.

**Status:** Planned.

---

## Phase TOKEN-2 — OutputPolicy runtime

**Goal:** Replace prompt-only verbosity control with runtime output policy.

**Owner layer:** `UNIFIED_EXECUTION_RUNTIME`; optional contract hints in `AGENT_CONTRACTS_AND_ASSEMBLY` later.

**Dependencies:** TOKEN-1A contracts.

**Deliverables:**

- `intergrax/runtime/token_optimization/output_policy.py`,
- `OutputPolicyResolver`,
- output profiles: `minimal`, `terse`, `standard`, `full`, `audit`, `machine_receipt`, `debug_verbose`,
- runtime safety bypass rules,
- integration point where LLM call max-output budget/profile is resolved,
- lightweight CI script `scripts/check_output_policy_wiring.py`.

**Acceptance criteria:**

- output profile is selected by runtime policy, not ad-hoc prompt wording,
- high-risk contexts can force standard/full clarity,
- terse mode is available for operator updates,
- audit/full mode remains explicit,
- structured output calls are not shortened unless schema explicitly allows it,
- no model-specific prompt hack is required.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run pytest tests/unit/runtime/ -q
uv run python scripts/check_output_policy_wiring.py
```

**Domain plan rows:** `TOKEN-UER-2` in `docs/plan/UNIFIED_EXECUTION_RUNTIME.md`.

**Status:** **Done / Closed**.

**Closeout:**

- OutputPolicy runtime resolver added
- deterministic resolved output policy added
- safe defaults added
- source_type-aware conservative policy behavior added
- validation helper added
- no model calls added
- no prompt assembly added
- no content optimization added
- no telemetry emission added
- no runtime hot-path wiring added
- next step: **TOKEN-3** ToolSchemaOptimizer — **Done / Closed** (§TOKEN-3 below)

---

## Phase TOKEN-3 — ToolSchemaOptimizer

**Goal:** Reduce recurring tool catalog token cost without changing tool schema semantics.

**Owner layer:** `TOOLS`.

**Dependencies:** TOKEN-1 contracts and protected-region validator; TOKEN-6 telemetry can be added after compact catalog works.

**Scope (TOKEN-3 closeout):** TOKEN-3 closes helper-only `ToolSchemaOptimizer` in `intergrax/runtime/token_optimization/tool_schema.py`.

- no runtime tool registry wiring
- no executable tool schema changes
- no prompt assembly changes
- no telemetry emission
- runtime integration into `ToolPlanningService` / `CatalogToolPlanner` / schema export path is future work (`TOKEN-TOOLS-1B`)

**Deliverables (helper-only, TOKEN-3 closeout):**

- `intergrax/runtime/token_optimization/tool_schema.py` — `ToolSchemaOptimizer` and `optimize_tool_schema_catalog`
- deterministic LLM-facing compact catalog view
- description normalization/truncation
- optional example removal via `allow_example_removal`
- protected-region validation integration
- compression receipts integration
- optional `token_counter` measurement
- unit tests in `tests/unit/runtime/token_optimization/test_tool_schema.py`

**Acceptance criteria:**

- canonical `ToolContract` registry is not mutated,
- tool names, parameter names, enum values, required fields, and JSON schema semantics are unchanged,
- tool call payloads and tool result JSON are not compressed by default,
- compact catalog can be enabled by policy/profile,
- schema preservation tests pass,
- token count of LLM-facing catalog decreases on a representative catalog fixture.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/test_tool_schema.py -q
uv run pytest tests/unit/runtime/token_optimization/ -q
```

**Domain plan rows:** `TOKEN-TOOLS-1A` (Done / Closed) and `TOKEN-TOOLS-1B` (Planned) in `docs/plan/TOOLS.md`.

**Status:** **Done / Closed**.

**Closeout (TOKEN-3):**

- `ToolSchemaOptimizer` added in `intergrax/runtime/token_optimization/tool_schema.py`
- deterministic LLM-facing tool catalog compaction added
- description normalization/truncation added
- optional example removal via `allow_example_removal` added
- required fields/types/enums/properties preserved
- protected-region validation integrated
- receipts integrated
- optional `token_counter` measurement supported
- no tokenizer/model calls added
- no runtime tool registry wiring added
- no executable tool schema mutation added
- no prompt assembly added
- no telemetry emission added
- runtime wiring into `ToolPlanningService` / `CatalogToolPlanner` / schema export path deferred to `TOKEN-TOOLS-1B`
- next step: **TOKEN-4** ContextPackOptimizer light/structural compression — Done / Closed (§Phase TOKEN-4 below)

---

## Phase TOKEN-4 — ContextPackOptimizer

**Goal:** Optimize selected context fragments using deterministic light/structural compression (helper-only slice).

**Owner layer:** `CONTEXT_ENGINEERING` (helper in `token_optimization`; runtime wiring deferred).

**Dependencies:** TOKEN-1 contracts/receipts/protected regions; TOKEN-2 output policy resolver.

**Deliverables (helper-only, TOKEN-4 closeout):**

- `intergrax/runtime/token_optimization/context_pack.py` — `ContextPackOptimizer` and `optimize_context_pack`
- deterministic light/structural context pack compaction
- required fragment preservation
- fragment order/IDs/source/provenance preservation
- protected-region validation integration
- compression receipts integration
- optional `token_counter` measurement
- unit tests in `tests/unit/runtime/token_optimization/test_context_pack.py`

**Explicit exclusions (this slice):**

- no semantic compression
- no tokenizer/model calls
- no `ContextCompiler` / `DefaultNexusContextEngine` wiring
- no RAG retrieval behavior changes
- no prompt assembly
- no telemetry emission
- runtime integration into CE pipeline is future work (`TOKEN-CE-1B`)

**Acceptance criteria:**

- accepts `ContextFragment`, mapping fragments, and raw string fragments,
- mandatory/policy (`required=True`) fragments are preserved exactly,
- fragment IDs, source types, metadata/provenance, and order are unchanged,
- fragments are not merged or removed,
- disabled policy bypasses optimization,
- protected-region validation with fallback on failure,
- receipts created for apply/bypass/fallback when `include_receipt=True`,
- optional `token_counter` measurement only.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/token_optimization/test_context_pack.py -q
uv run pytest tests/unit/runtime/token_optimization/ -q
```

**Domain plan rows:** `TOKEN-CE-1A` (Done / Closed), `TOKEN-CE-1B` (Planned), and `TOKEN-CE-2` (Planned) in `docs/plan/CONTEXT_ENGINEERING.md`.

**Status:** **Done / Closed**.

**Closeout (TOKEN-4):**

- `ContextPackOptimizer` helper added in `intergrax/runtime/token_optimization/context_pack.py`
- deterministic light/structural context pack compaction added
- required fragments preserved
- fragment order/source/provenance preserved
- protected-region validation integrated
- fallback on validation failure added
- receipts integrated
- optional `token_counter` measurement supported
- no tokenizer/model calls added
- no `ContextCompiler` / `DefaultNexusContextEngine` wiring added
- no RAG retrieval behavior changed
- no prompt assembly added
- no telemetry emission added
- next step: **TOKEN-OBS-1** HOS/domain-signal emission, according to plan ordering

---

## Phase TOKEN-5 — MemorySummaryCompressor

**Goal:** Safely compress persistent natural-language memory summaries and documentation-derived memory blocks.

**Owner layer:** `MEMORY`.

**Dependencies:** TOKEN-1 contracts/receipts/protected regions; recommended after TOKEN-4 proves runtime receipts.

**Status:** Planned (helper-only first slice **TOKEN-5A** — Done / Closed; see §TOKEN-5A).

**First slice:** **TOKEN-5A** — helper-only `MemorySummaryCompressor` with staging, receipts, validation, rollback metadata, and benchmark-ready result shape. See §TOKEN-5A below.

**Later slices (not TOKEN-5A):** live staging write flow, memory receipt storage wiring, CI script `scripts/check_memory_compression_receipts.py`, runtime hot-path integration.

**Domain plan rows:** `TOKEN-MEM-1` in `docs/plan/MEMORY.md`.

---

## TOKEN-5A — MemorySummaryCompressor helper-only first slice

**Status:** **Done / Closed**.

**Purpose:** Add a conservative MEMORY-owned `MemorySummaryCompressor` helper that compresses memory-summary candidates deterministically, records receipts and rollback metadata, and returns a benchmark-ready result shape — without live memory-store mutation, LLM-based semantic rewriting, or runtime hot-path wiring.

**Closeout notes:**

- helper-only `MemorySummaryCompressor` added (`intergrax/memory/summary_compressor.py`)
- staged result / rollback metadata added
- protected-region validation integrated
- compression receipts integrated
- optional `token_counter` supported
- optional `semantic_validation_hook` supported
- benchmark-ready result shape added
- no live memory-store overwrite
- no vector index mutation
- no embedding regeneration
- no LLM/model-based semantic rewriting
- no HOS emission
- no runtime hot-path wiring
- no LKW proof execution
- next step: **TOKEN-OBS-1** HOS/domain-signal emission according to plan ordering

**Refinement TOKEN-5A-R — unsafe lossy truncation guard:**

- `max_summary_chars` is treated as **lossy** compression, not lossless structural compaction
- no truncation under default `allow_lossy=False` policy
- lossy truncation requires explicit `allow_lossy=True` **and** `semantic_validation_hook` acceptance
- protected-region validation still guards truncation candidates; semantic-hook rejection falls back to original
- no LLM-as-a-Judge implementation was added
- no live memory-store wiring was added

**Deliverables (implementation task, not this docs-only slice):**

- `intergrax/memory/summary_compressor.py` — conservative `MemorySummaryCompressor` helper
- staged compressed candidate/result model
- rollback metadata model
- protected-region validator reuse (TOKEN-1B)
- compression receipt integration (TOKEN-1C)
- optional `token_counter` support
- optional `semantic_validation_hook` interface (callable only; no built-in LLM judge)
- deterministic light/structural compression only
- benchmark-ready result fields for future TOKEN-6B / LKW-PF6 proof
- fallback to original on validation failure or semantic-hook rejection

**Benchmark-ready result fields (required on apply/fallback outcomes):**

```text
source_type
strategy
original_hash
optimized_hash
original_tokens
optimized_tokens
saved_tokens
saved_ratio
validation_status
fallback_status
receipt / receipt_ref
rollback_metadata
semantic_validation_status   # present only when semantic_validation_hook is used
```

**Explicit exclusions (TOKEN-5A):**

- no live memory-store overwrite
- no automatic memory compaction job
- no vector index mutation
- no embedding regeneration
- no LLM/model-based semantic rewriting
- no full LLM-as-a-Judge eval engine
- no runtime hot-path wiring
- no HOS/domain-signal emission
- no observability exporter wiring
- no token regression benchmark runner
- no LKW proof execution

**Acceptance criteria:**

- helper-only: compresses in-memory candidates without mutating persistent memory stores,
- live source is never overwritten before validation (no live mutation in this slice),
- failed compression cannot corrupt persistent memory (no persistent writes in this slice),
- protected-region validation with fallback to original on failure,
- compression receipts created when `include_receipt=True`,
- rollback metadata attached to every candidate result,
- memory compression remains opt-in by profile/policy at call site,
- no user facts, dates, IDs, or policy text are silently lost,
- benchmark-ready result fields populated for TOKEN-6B / LKW-PF6 attribution.

**Semantic validation note:** future semantic validation and LLM-as-a-Judge belong to **TOKEN-OBS-2** / regression/evals work, not to TOKEN-5A or TOKEN-6B. TOKEN-5A may expose an optional `semantic_validation_hook` interface; TOKEN-OBS-2 and eval gates may consume it later. TOKEN-5A and TOKEN-6B must not implement or depend on a full LLM-as-a-Judge engine.

**LKW-PF6 alignment:** TOKEN-5A adds the `memory` / `memory tokens` optimization and proof category so later LKW-PF6 proof runs can attribute measured savings for memory-summary compression alongside tools, context/RAG, and output shaping.

**Required tests/checks (implementation task):**

```bash
uv run pytest tests/unit/memory/ -q
```

**Domain plan rows:** `TOKEN-MEM-1` in `docs/plan/MEMORY.md`.

---

## Phase TOKEN-6 — Telemetry and regression gates

**Goal:** Make token savings measurable and safe across runs, steps, models, and sources.

**Owner layer:** `OBSERVABILITY`; affected implementation owners per source domain.

**Dependencies:** TOKEN-1 contracts; can start with telemetry for TOKEN-2/TOKEN-3 before TOKEN-4/TOKEN-5 exist.

**Deliverables:**

- `intergrax/runtime/token_optimization/telemetry.py`,
- typed optimization summary payload,
- receipt payload shape,
- counters/spans emitted through HOS or approved domain-signal path,
- savings attribution model,
- token-vs-quality benchmark fixtures,
- `scripts/check_compression_receipts.py`,
- `scripts/check_token_regression_benchmarks.py` — **Done / Closed** (TOKEN-6B helper-only runner; no HOS wiring).

**TOKEN-6B closeout (Done / Closed):** `intergrax/runtime/token_optimization/regression.py` provides deterministic fixture-based regression benchmarks for `tool_schema`, `context_pack`, and `memory_summary` with receipt/validation/fallback expectation checks and a local `scripts/check_token_regression_benchmarks.py` gate. No model calls, no external tokenizer, no HOS/exporter wiring, no LKW proof execution.

**Acceptance criteria:**

- optimized model calls report raw/after/saved token counts,
- savings are attributable by run, step, source, model, provider, strategy, and output profile,
- regression checks can fail CI when token growth is uncontrolled or quality drops,
- telemetry does not create a private event bus,
- event naming respects the Observability event ownership model.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/observability/ -q
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run python scripts/check_compression_receipts.py
uv run python scripts/check_token_regression_benchmarks.py
```

**Domain plan rows:** `TOKEN-OBS-1` and `TOKEN-OBS-2` in `docs/plan/OBSERVABILITY.md`.

**Status:** Planned.

---

## Phase TOKEN-7 — Adaptive optimization

**Goal:** Use historical telemetry to recommend budgets and compression strategies.

**Owner layer:** `ADAPTIVE_HARNESS_INTELLIGENCE`.

**Dependencies:** TOKEN-6 telemetry and regression gates.

**Deliverables:**

- adaptive budget recommendation inputs,
- compact/full profile recommendation by task/step/source type,
- quality-drop escalation rules,
- operator override support,
- no autonomous production auto-apply until governance permits it.

**Acceptance criteria:**

- adaptive optimization remains policy-governed,
- runtime can escalate to fuller context when quality drops,
- recommendations are observable and reversible,
- no autonomous compression is applied without configured policy,
- AHI uses Token Optimization telemetry as input rather than duplicating token accounting.

**Required tests/checks:**

```bash
uv run pytest tests/unit/runtime/adaptive/ -q
```

**Domain plan rows:** `TOKEN-AHI-1` in `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`.

**Status:** Planned / Frozen until TOKEN-6 ships.

---

## ADR queue

| ADR | Scope | Status |
|-----|-------|--------|
| `ADR-TOKEN-001` | Multi-layer feature boundary and runtime component placement | Planned |
| `ADR-TOKEN-002` | Protected-region validation and compression receipts | Planned |
| `ADR-TOKEN-003` | Tool schema optimization safety model | Planned |
| `ADR-TOKEN-004` | Token telemetry and regression gate semantics | Planned |

---

## Domain plan row checklist

See [`plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) for the canonical domain plan row checklist, TOKEN phase → owning plan mapping, and sync instructions.

---

## First implementation prompt

Use this only after the domain plan rows above exist.

```text
Pracujemy na repozytorium `jakbuczarnecki/intergrax`, branch `development`.

Cel sesji:
Zaimplementuj TOKEN-1A/TOKEN-1B/TOKEN-1C — shared Token Optimization contracts, protected-region validator, and compression receipts.

Read scope:
- docs/features/architecture/TOKEN_OPTIMIZATION.md
- docs/features/plan/TOKEN_OPTIMIZATION.md
- docs/plan/UNIFIED_EXECUTION_RUNTIME.md rows TOKEN-UER-1/TOKEN-UER-2
- existing runtime token/cost/context budget modules only as needed

Edit scope:
- intergrax/runtime/token_optimization/__init__.py
- intergrax/runtime/token_optimization/contracts.py
- intergrax/runtime/token_optimization/protected_regions.py
- intergrax/runtime/token_optimization/receipts.py
- tests/unit/runtime/token_optimization/
- scripts/check_token_optimization_contracts.py

Do not wire behavior into LLM call path yet.
Do not implement ToolSchemaOptimizer yet.
Do not implement ContextPackOptimizer yet.
TOKEN-5A helper-only `MemorySummaryCompressor` implemented; do not wire live memory flows yet.
Do not create docs/plan/TOKEN_OPTIMIZATION.md.

Acceptance:
- contracts import cleanly,
- protected regions are detected and validated,
- receipts hash original/optimized content and record token savings,
- failed protected-region validation forces fallback,
- tests pass.

Run:
uv run pytest tests/unit/runtime/token_optimization/ -q
uv run python scripts/check_token_optimization_contracts.py
uv run python scripts/audit/check_docs_domain_pairs.py

Commit:
feat: add token optimization contracts and receipts
```

---

## Delivery rules

- One TOKEN phase or one domain-owned subset per PR.
- Update feature plan and affected domain plan together when a TOKEN phase becomes active.
- Do not implement runtime code in docs-sync PRs.
- Do not duplicate existing Context Engineering budget/preflight mechanisms.
- Do not duplicate LLM adapter token counting.
- Do not report token savings without quality/safety validation.
- Preserve architecture/plan 1:1 domain pairs.
- Preserve feature architecture/plan 1:1 feature pairs.

---

## Explicit exclusions

Token Optimization does not:

- compress private chain-of-thought,
- mutate executable code,
- rewrite strict JSON schema semantics,
- compress tool call payloads by default,
- remove required audit evidence,
- replace RAG ranking,
- replace memory lifecycle management,
- replace LLM adapter token counting,
- replace model routing.
