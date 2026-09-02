# Local Knowledge Workspace (LKW) - Platform Proof Loop

**Status:** active governance rule for LKW implementation  
**Parent:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Hardening addendum:** [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md)  
**Plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

---

## 1. Decision

LKW is not only a product proof. LKW is the first proof that the Intergrax platform can repeatedly create, configure, run, package, deploy, observe, and evolve agent applications.

A wave is not complete when LKW works manually but the reusable platform, scaffold, configuration, Docker, CI, or deployment implications are left behind.

---

## 2. Rule

Every non-trivial LKW step must include two acceptance layers:

1. **Product acceptance** - the LKW capability works.
2. **Platform acceptance** - the reusable lesson is propagated to the platform, scaffold, settings, build/deploy path, or CI/CD surface when applicable.

This prevents proving only a hand-built LKW application while leaving the platform unable to generate the next application correctly.

---

## 3. Defect and pattern classification gate

Before closing any LKW task, classify every discovered bug, workaround, repeated implementation pattern, missing diagnostic, scaffold gap, configuration mismatch, Docker/build issue, dependency issue, or CI/runbook gap as one of:

1. `LKW-specific` - belongs only to the local workspace domain; record why no platform propagation is needed.
2. `Platform-reusable` - affects how future Intergrax applications should be generated, configured, run, tested, observed, packaged, or deployed; update the reusable surface in the same task when safe.
3. `Platform-reusable deferred` - reusable, but too large for the current task; record a blocking follow-up before moving to the next LKW wave.

This gate applies to both implementation and diagnostic work.

---

## 4. Platform propagation loop

For every LKW wave, run this checklist:

| Step | Question | Required action |
|------|----------|-----------------|
| 1. LKW implementation | Did the product capability change? | Implement and test the LKW behavior. |
| 2. Defect/pattern classification | Did the task reveal a bug, workaround, repeated pattern, missing diagnostic, scaffold gap, config mismatch, Docker/build issue, dependency issue, or CI/runbook gap? | Classify it as `LKW-specific`, `Platform-reusable`, or `Platform-reusable deferred`. |
| 3. Shared platform extraction | Is the solution generic to agent applications? | Move or expose it through `intergrax`, `intergrax/applications/_shared`, runtime profiles, or approved shared contracts. |
| 4. Scaffold propagation | Should future agents/apps inherit it? | Update scaffold generators, templates, generated docs, env templates, Docker templates, or tests. |
| 5. Env/settings contract | Did configuration change? | Update `.env.example`, `host/settings.py`, validation behavior, and config docs. |
| 6. Packaging contract | Did dependencies or entrypoints change? | Update `pyproject.toml`, optional dependency groups, entrypoints, Dockerfile, `.dockerignore`, or build docs. |
| 7. Deploy/CI contract | Did the run path become verifiable? | Add or update tests, CI smoke, Docker build check, image run check, or deployment runbook. |
| 8. Documentation sync | Did the architecture or plan change? | Update architecture, implementation plan, and generated scaffold documentation. |

---

## 5. Required per-wave platform checklist

Use this list before closing any LKW implementation wave:

- [ ] Did every discovered bug, workaround, repeated pattern, missing diagnostic, scaffold gap, config mismatch, Docker/build issue, dependency issue, and CI/runbook gap receive a classification?
- [ ] Does this change belong only to LKW, or should it move to shared platform code?
- [ ] Should application scaffold generate this pattern for the next product host?
- [ ] Should agent scaffold generate this contract, test, or documentation pattern?
- [ ] Does `.env.example` match `host/settings.py` and production validation?
- [ ] Does `pyproject.toml` need a dependency split or optional dependency group?
- [ ] Does Docker still build from the monorepo root with the required files copied?
- [ ] Does Docker run expose the correct host, port, env profile, and healthcheck?
- [ ] Does CI need a new application smoke test or Docker build check?
- [ ] Does the deploy/runbook still describe the real execution path?
- [ ] Does the implementation plan identify both the LKW work and the platform propagation work?

---

## 6. Where propagation must land

| Area | Target examples |
|------|-----------------|
| Shared application runtime | `intergrax/applications/_shared` |
| Runtime/kernel/orchestration | `intergrax/runtime` |
| Agent scaffold | `intergrax/scaffold` agent templates and tests |
| Application scaffold | `intergrax/scaffold/new_application.py`, product app templates, generated docs |
| Docker/build templates | shared Docker template writers and app `docker` folders |
| Env/settings | `.env.example`, `host/settings.py`, config validation docs |
| Packaging | `pyproject.toml`, optional dependencies, build docs |
| CI/CD | GitHub Actions, smoke tests, Docker build/run checks |
| Documentation | LKW architecture/plan, app creation guide, application usage docs |

---

## 7. When not to propagate

Do not update platform or scaffold when the change is truly LKW-domain-specific, for example:

- a local workspace capability name;
- a user-file workflow that does not generalize;
- a domain-specific prompt or synthesis template;
- a temporary fixture used only for LKW tests.

But if the change affects how applications configure env, expose APIs, wire agents, build Docker images, run CI, emit trace, expose diagnostics, or validate production mode, it is platform-relevant.

---

## 8. Execution implication

The LKW implementation order becomes:

```text
1. Implement or diagnose the smallest LKW capability slice.
2. Classify every discovered bug, workaround, repeated pattern, diagnostic gap, scaffold gap, config mismatch, Docker/build issue, dependency issue, or CI/runbook gap.
3. Identify generic platform/scaffold/deploy implications.
4. Update the reusable surface in the same PR when safe.
5. If not safe, record a blocking follow-up before moving to the next LKW wave.
6. Keep tests and docs aligned with both product and platform acceptance.
```

This is the correct proof model: LKW proves the platform by forcing the platform to absorb every reusable lesson from the product implementation.

**Platform proof scope (2026-07):** The loop now explicitly covers provider-switch and production-maturity proofs beyond product capability slices - including model serving providers, persistence/database, vector stores, observability backends, metrics/tracing/error monitoring, token optimization, and scaffold/deploy propagation. Strategic roadmap: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §LKW-PF.

---

## 9. Platform proof maturity bar (LKW-PF0)

LKW-driven platform proofs use four distinct maturity levels. Future proofs - including **Token Optimization** (`LKW-PF6`) - must apply this bar before claiming closure or production readiness.

### 9.1 Platform proof

**Definition:** Evidence that a reusable platform capability works through the intended platform abstraction, contract, configuration, or integration boundary.

A platform proof may be **closed** when all of the following hold:

- the proof workload exercises the platform path;
- the result is observable or inspectable;
- the reusable lesson is captured in platform docs/project/maintainers/planss;
- application-specific code did not bypass platform boundaries;
- known production gaps are explicitly recorded.

**Does not mean:** production-grade readiness. Closing a platform proof validates the integration boundary and reusable lesson - not full production operation.

### 9.2 Operational proof

**Definition:** Evidence that an operator can run, inspect, debug, or repeat the proof in a controlled local/dev/proof environment.

An operational proof may be **closed** when all of the following hold:

- operator instructions exist;
- expected inputs/outputs are documented;
- failure or safety behavior is visible;
- proof results can be inspected;
- limitations are documented.

**Does not mean:** full production readiness. Operational proof confirms repeatability and inspectability in a proof environment - not hardened production deployment.

### 9.3 Production-grade readiness

**Definition:** A higher maturity level requiring production-oriented concerns to be implemented and verified - not merely planned or deferred.

Claim production-grade readiness only when applicable items below are **actually implemented and verified**:

- health/status;
- auth/TLS/secrets handling;
- retention/rotation;
- batching/backpressure where applicable;
- dashboard/config as code where applicable;
- CI/live proof automation where applicable;
- runbooks;
- path/security policy;
- failure recovery behavior;
- clear operator ownership.

**Rule:** Do not claim production-grade readiness from a closed platform proof alone.

### 9.4 Production hardening backlog

**Definition:** The place where known production gaps are tracked **after** a platform proof is closed.

Rules:

- **`closed proof != production complete`** - a closed platform proof remains valid; production gaps do not reopen proof scope.
- Record gaps in the owning platform plan (for example [`docs/project/maintainers/plans/OBSERVABILITY.md`](../../../docs/project/maintainers/plans/OBSERVABILITY.md) Phase OBS-VENDOR for observability vendors).
- Future hardening work continues without invalidating or reopening the already-valid platform proof.
- Do not downgrade closed proof status when adding backlog items.

### 9.5 Proof closure rules

When closing any LKW-driven platform proof wave:

1. State which maturity level is being closed (platform proof, operational proof, or production-grade).
2. If closing platform or operational proof, record remaining production gaps in the production hardening backlog.
3. Do not imply production-grade readiness unless §9.3 criteria are met.
4. Preserve platform boundary discipline - no application-specific bypass of contracts or integration paths.

### 9.6 Canonical example - Elasticsearch/Kibana observability

| Maturity level | Status | Notes |
|----------------|--------|-------|
| Platform proof | **Closed** | Elasticsearch/OpenSearch export through platform contract (`OBS-VENDOR-4A` … `OBS-VENDOR-5`); LKW proof workload and live readback (`OBS-VENDOR-7` live proof, [`ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md`](ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md)). |
| Operational proof | **Closed** (proof environment) | Operator runbook, inspectors, and controlled local proof documented in [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md). |
| Production-grade readiness | **Not claimed** | Auth/TLS, retention/rotation, batching policy, dashboards-as-code, CI/live automation, and full operational hardening remain open. |
| Production hardening backlog | **Planned** | Tracked in [`docs/project/maintainers/plans/OBSERVABILITY.md`](../../../docs/project/maintainers/plans/OBSERVABILITY.md) Phase OBS-VENDOR (`OBS-VENDOR-6`, `OBS-VENDOR-6C`, and related rows). |

**Preserved distinction:** Elasticsearch/Kibana path is **closed for platform proof**, but **not production-grade**. Full **OBS-VENDOR** production hardening remains **Planned**.

---

## 10. LKW-PF6-0 - Token Optimization proof design

**Status:** **Done / Closed** (docs-only, 2026-07-01).

**Maturity level closed:** proof design only. Does **not** close `LKW-PF6` platform proof, operational proof, or production-grade readiness.

**Purpose:** Define exactly what the LKW Token Optimization proof must demonstrate **before** `TOKEN-1A` code starts. Token Optimization is a **cross-layer platform capability** - not a private LKW feature. Narrative: Intergrax proves that agent applications can be built as configurable, observable, cost-aware runtime systems - not hand-wired demos.

Canonical detail: [`docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md`](../../../docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md) §LKW-PF6-0. Implementation schedule: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §LKW-PF6-0 closeout.

### 10.1 Representative LKW workflows

All candidate workflows follow the existing LKW product proof shape:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

| Workflow ID | Description | Proof intent |
|-------------|-------------|--------------|
| **LKW-TOK-W1** | Small workspace indexing + search + synthesis | Baseline for minimal tenant-scoped path; establishes per-step token categories on a compact corpus. |
| **LKW-TOK-W2** | Medium workspace search + synthesis with evidence | Baseline for RAG/evidence/context-pack token attribution under realistic retrieval load. |
| **LKW-TOK-W3** | Repeated synthesis run with similar tool/catalog/context exposure | Measures recurring tool-catalog and context-pack savings across runs with stable exposure. |
| **LKW-TOK-W4** | Failure/safety-preserving run where exact regions must not be compressed | Proves optimization rejection/fallback when protected regions or safety boundaries would be violated. |

Proof design only - no fixtures or scripts in this step.

### 10.2 Baseline measurement shape

Baseline measurement happens **before** optimization is applied and must be reproducible enough to compare against optimized runs.

| Field | Meaning |
|-------|---------|
| `input_context_tokens` | Tokens in assembled input/context before optimization |
| `tool_catalog_tokens` | Tokens in LLM-facing tool catalog view |
| `retrieved_evidence_context_pack_tokens` | Tokens from RAG/evidence/context pack fragments |
| `output_tokens` | Tokens in model output for the measured step |
| `total_tokens` | Sum attributable to the measured scope |
| `model` | Model identifier |
| `provider` | Provider identifier |
| `runtime_profile` | Active runtime/output/profile configuration |
| `workflow_id` | One of `LKW-TOK-W1` … `LKW-TOK-W4` |
| `run_id` | Harness run identifier |
| `step_id` | Step identifier within the run |

### 10.3 Optimized measurement shape

Optimized proof runs must later report:

| Field | Meaning |
|-------|---------|
| `baseline_token_usage` | Baseline counts per category (§10.2) |
| `optimized_token_usage` | Post-optimization counts per category |
| `saved_tokens` | `baseline − optimized` per category and total |
| `saved_ratio` | `saved_tokens / baseline` where baseline > 0 |
| `optimization_strategy` | Strategy applied (e.g. output profile, compact catalog, context-pack light mode) |
| `affected_source_category` | Which token category/source was optimized |
| `fallback_status` | Whether optimization was applied, partially applied, or rejected with fallback |
| `validation_status` | Protected-region and quality validation outcome |

### 10.4 Token categories

Canonical categories for attribution:

- input/context tokens
- tool catalog tokens
- RAG/evidence/context pack tokens
- memory tokens
- output tokens
- system/policy tokens (where measurable)
- total tokens

Categories must support later attribution by: `run`, `step`, `source`, `model`, `provider`, `strategy`, `output_profile`.

### 10.5 Quality and regression criteria

An optimized run is **not** successful if it saves tokens but breaks:

- tenant-scoped evidence
- evidence references
- synthesized answer integrity
- shadow artifact behavior
- safety boundaries
- exact protected regions
- platform abstraction boundaries

**Minimum comparison expectation:** baseline result and optimized result must remain **behaviorally equivalent** for the proof workload, with differences limited to allowed formatting or verbosity changes.

### 10.6 Protected-region requirements

Token Optimization must never lose or rewrite protected regions such as:

- code blocks
- inline code
- paths
- URLs
- env vars
- enum values
- hashes
- dates
- exact error strings
- policy text
- IDs
- tenant identifiers where required for correctness
- evidence references

`TOKEN-1B` will later implement protected-region validation. `LKW-PF6-0` defines the proof requirement only.

### 10.7 Compression receipt expectations

Future compression receipts must prove:

- original hash
- optimized hash
- original token count
- optimized token count
- saved tokens
- saved ratio
- strategy
- protected-region validation status
- fallback reason when optimization is rejected

`TOKEN-1C` will later implement receipts. No implementation in `LKW-PF6-0`.

### 10.8 Observability visibility

Token savings must be visible through the **Harness Observability Spine** or an **approved domain-signal path**. Do not introduce a private Token Optimization telemetry bus.

Proof must later show attribution by: `run_id`, `step_id`, `workflow_id`, `model`, `provider`, `profile`, `source/category`, `strategy`, `baseline_tokens`, `optimized_tokens`, `saved_tokens`, `saved_ratio`, `validation_status`, `fallback_status`.

See [`docs/project/maintainers/plans/OBSERVABILITY.md`](../../../docs/project/maintainers/plans/OBSERVABILITY.md) Phase TOKEN-OBS.

### 10.9 Public proof format

The later public-grade LKW token proof (`LKW-PF6-C`) must include:

- representative workflow description
- baseline token usage
- optimized token usage
- saved tokens
- saved ratio
- receipt references
- protected-region validation result
- quality/regression result
- observability attribution
- known limitations

**Must not expose:** raw prompts, raw documents, raw RAG chunks, raw synthesized content, tool args, secrets, tokens/secrets, absolute file paths, large raw artifacts.

### 10.10 LKW-PF6-0 closure rule

`LKW-PF6-0` is **Done / Closed** only when all of the following hold:

- [x] representative workflows are defined (§10.1),
- [x] baseline and optimized measurement shapes are defined (§10.2, §10.3),
- [x] token categories are defined (§10.4),
- [x] quality/regression criteria are defined (§10.5),
- [x] protected-region requirements are defined (§10.6),
- [x] compression receipt expectations are defined (§10.7),
- [x] observability visibility is defined (§10.8),
- [x] public proof format is defined (§10.9),
- [x] `TOKEN-1A` remains not started,
- [x] no code/runtime/test/CI/dependency files are changed.

---

## 11. LKW-PF6 - Token Optimization product proof (planned)

**Prerequisite:** Universal platform proof **TOKEN-10G** must pass before LKW product proof begins. **LKW-PF6-0** (proof design) is **Done / Closed** - see §10.

### Canonical ordering

```text
TOKEN-10A … TOKEN-10G → universal platform proof
TOKEN-10H             → checked-in proof, README promotion
LKW-PF6-A             → product baseline measurement
LKW-PF6-B             → runtime integration (consume platform contracts)
LKW-PF6-C             → baseline-vs-optimized product proof
```

### LKW-PF6-A - Product baseline

Measure real LKW flows **without** optimization: workspace search; evidence/context assembly; synthesis; tool exposure; repeated conversational steps; protected evidence path.

### LKW-PF6-B - Runtime integration

LKW supplies product policy, classifications, evidence, identity, and explicit enablement. LKW consumes stable prompt/runtime contract, router, cache-aware gate, pipeline, receipts, and metrics. **LKW must not duplicate** Token Optimization components.

### LKW-PF6-C - Product proof

Compare baseline and optimized runs: input tokens, content-reduction savings, prefix-cache reuse, latency, evidence preservation, tenant isolation, protected regions, answer quality, fallbacks, receipts, observability attribution.

**Maturity:** LKW-PF6-C closure is product proof - not automatic production-grade readiness. Distinction: proof design → platform proof → operational proof → production-grade readiness.

Canonical detail: [`docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md`](../../../docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md) §LKW-PF6 proof phase map; schedule: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §LKW-PF6.

---

## 8. Repository-wide Intergrax proof suite (PUBLIC-PROOF-GATE-1)

The canonical gateway answers: **which Intergrax capabilities can be proven on this exact commit and environment right now?**

| Artifact | Role |
|----------|------|
| `scripts/proof/intergrax_proof_manifest.py` | Typed manifest - proof membership, profiles, commands, environment/platform requirements |
| `scripts/proof/intergrax_proof_runner.py` | Master runner - selection, subprocess execution, aggregation, receipt |
| `scripts/proof/run-intergrax-proof-suite.py` | Operator entrypoint |

### Commands

```bash
uv run python scripts/proof/run-intergrax-proof-suite.py --profile quick
uv run python scripts/proof/run-intergrax-proof-suite.py --profile full
uv run python scripts/proof/run-intergrax-proof-suite.py --profile live
```

`--dry-run` resolves manifest selection without executing child proofs. `--allow-external-mutating` opts in to external mutating proofs when registered.

### Profiles

| Profile | Semantics |
|---------|-----------|
| `quick` | Fast, deterministic, local proofs - no required external-provider calls |
| `full` | All locally executable proofs for the current machine (includes `quick`) |
| `live` | Adds real external-provider proofs (includes `full`) |

### Status interpretation

| Status | Meaning |
|--------|---------|
| `PASS` | Child proof exited zero |
| `FAIL` | Child proof failed or timed out |
| `BLOCKED_ENVIRONMENT` | Required environment capability absent (not a product defect) |
| `SKIPPED_PLATFORM` | Manifest declares a different OS requirement |
| `SKIPPED_PROFILE` | Not selected for the requested profile or dry-run |

`LIVE` profile: missing optional external-provider credentials yields `PASS_WITH_BLOCKED` overall when no proof actually failed.

### Overall suite status

| Status | Meaning |
|--------|---------|
| `DRY_RUN` | Manifest selection and dry execution plan validated; child proofs were not executed. Not equivalent to `PASS`. Exit code may be 0 because the dry-run itself succeeded. |

### Receipts

Machine-readable receipts are written to `.artifacts/proof/<timestamp>-<profile>-<short-sha>.json` (gitignored). Receipts include commit SHA, dirty-worktree flag, per-proof status, and safe diagnostics - never tokens, API keys, or environment values.

### Tests vs proofs vs qualification

- **Unit/integration tests** - regression gates in CI; not public evidence by themselves.
- **Proof scripts** - bounded, operator-runnable evidence workloads referenced by the manifest.
- **Real-provider qualification** - live `PASS` against Slack/Google/M365 requires credentials and explicit `live` profile execution; implementation `PASS` alone is not external qualification.

Canonical manifest is the source of truth. Individual LKW reviewer guides (for example [`applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md`](proof/LKW_PLATFORM_PROOF.md)) explain how to run domain proofs; the suite orchestrates them without duplicating their implementation.

---
