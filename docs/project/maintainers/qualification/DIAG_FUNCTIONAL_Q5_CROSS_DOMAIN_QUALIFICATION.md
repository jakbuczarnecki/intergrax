# DIAG-FUNCTIONAL-Q5 — Cross-Domain Functional Diagnostics Qualification

## Purpose

Q5 answers whether Intergrax has **one universal Functional Diagnostics qualification framework**, or four independent harnesses that happen to share some types.

Q1–Q4 proved Functional Diagnostics across four mechanisms:

| Gate | Domain | Mechanism |
|------|--------|-----------|
| Q1 | RAG / retrieval | C1 pipeline |
| Q2 | Tool selection | workspace tool catalog |
| Q3 | Web search | Tavily + extraction |
| Q4 | Model routing | two-model Ollama routing |

Q5 re-runs all four through a **generic plugin runner** and proves:

- same `FunctionalDiagnosticAnalyzer` across domains
- reusable qualification core (`intergrax/core/qualification/`)
- domain plugins as thin adapters over proven Q1–Q4 execution
- cross-domain evidence scope isolation
- fifth synthetic plugin without core changes

## Architecture

```text
                   Functional Qualification Core
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
     reusable contracts   generic runner     generic metrics
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                     Qualification Plugin API
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         functional.rag   functional.tool_selection   functional.web_search
              │               │               │
              └───────────────┼───────────────┘
                              │
                    functional.model_routing
                              │
                              ▼
                  FunctionalDiagnosticAnalyzer (unchanged)
```

### Generic (core)

| Component | Module |
|-----------|--------|
| Plugin identity | `functional_qualification_identity.py` |
| Plugin contract | `functional_qualification_plugin.py` |
| Registry | `functional_qualification_registry.py` |
| Plan | `functional_qualification_plan.py` |
| Runner | `functional_qualification_runner.py` |
| Metrics | `functional_qualification_metrics.py` |
| Verdict | `functional_qualification_verdict.py` |
| Fidelity gates | `functional_qualification_fidelity.py` |
| Case normalization | `functional_qualification_case.py` |
| Bounded attempts | `functional_qualification_attempts.py` |
| Results / reporting | `functional_qualification_result.py`, `functional_qualification_reporting.py` |
| Comparator (pre-Q5) | `functional_diagnostic_comparator.py` |
| Expectations (pre-Q5) | `functional_diagnostic_expectation.py` |

### Domain-specific (plugins)

Located under `tests/system/functional_diagnostics_q5/plugins/`:

| Plugin ID | Adapter | Wraps |
|-----------|---------|-------|
| `functional.rag` | `RagQualificationPlugin` | `tests/system/functional_diagnostics_q1/runner.py` |
| `functional.tool_selection` | `ToolSelectionQualificationPlugin` | Q2 runner |
| `functional.web_search` | `WebSearchQualificationPlugin` | Q3 runner |
| `functional.model_routing` | `ModelRoutingQualificationPlugin` | Q4 runner |

Domain plugins contain: descriptor, expectation mapping, fidelity gate mapping, and a call to the existing canonical domain `run_qualification()`.

## Q1–Q4 Responsibility Audit

| Responsibility | Q1 | Q2 | Q3 | Q4 | Common? |
|----------------|----|----|----|----|---------|
| Case definitions | `q1/cases.py` | `q2/cases.py` | `q3/cases.py` | `q4/cases.py` | Pattern shared; content domain-specific |
| Expectations | `QualificationCaseExpectation` | same | same | same | **Yes** — `functional_diagnostic_expectation.py` |
| Oracle | `q1/oracle.py` | `q2/oracle.py` | `q3/oracle.py` | `q4/oracle.py` | Domain-specific |
| Runner | `q1/runner.py` | `q2/runner.py` | `q3/runner.py` | `q4/runner.py` | Orchestration duplicated; Q5 core extracts generic flow |
| Preflight | LKW + Qdrant index | LKW ready | LKW + Tavily | LKW + Ollama models | Domain-specific checks |
| Comparison | `compare_qualification_case` | same | same | same | **Yes** |
| Metrics | inline in runner | inline | inline | inline | **Yes** — `functional_qualification_metrics.py` |
| Fidelity | per-domain snapshot | per-domain | decision + evidence | routing + evidence | Plugin-defined gates; core aggregates |
| Artifact writing | per-domain JSON | per-domain | per-domain | per-domain | Q5 adds cross-domain envelope |
| Human report | Q1 MD | Q2 MD | Q3 MD | Q4 MD | This document |
| Repeatability | repeat groups | repeat groups | repeat groups | repeat groups | **Yes** — `QualificationRepeatabilityGroup` |
| Isolation | Q1-E cross-run | Q2-F | Q3-G | Q4-F | Scope-based; Q5 verifies globally |
| Environment | proof env / LKW | proof env | `intergrax_proof_environment` | same | Reused canonical loader |
| Runtime invocation | LkwClient | LkwClient | LkwClient | LkwClient | Domain-specific payloads |
| Trace parsing | RAG evidence fetch | tool invoke | web Tavily | routing profiles | Domain-specific |

## Universal Evidence Primitives

| Domain | Generic evidence |
|--------|------------------|
| RAG | OPERATION + CANDIDATES + SELECTION + VALIDATION |
| Tools | CANDIDATES + SELECTION + OPERATION + VALIDATION |
| Web | OPERATION + CANDIDATES + SELECTION + OUTPUT_RELATION + VALIDATION |
| Model Routing | CANDIDATES + SELECTION + OPERATION + OUTPUT_RELATION + VALIDATION |

**Universality statement:**

- same evidence primitives across 4 domains = **YES**
- same analyzer = **YES** (`FunctionalDiagnosticAnalyzer`)
- domain-specific analyzer count = **0**

## Plugin Contract

Minimal mandatory surface:

```python
class FunctionalQualificationPlugin(Protocol):
    @property
    def descriptor(self) -> QualificationPluginDescriptor: ...
    def execute(self) -> QualificationPluginResult: ...
```

Optional capabilities are expressed as **plugin-defined fidelity gates**, not fat interface methods.

## Verdict Semantics

| Verdict | Meaning |
|---------|---------|
| `PASS` | All required plugins and gates pass |
| `FAILED` | Diagnostic mismatch in at least one executed domain |
| `BLOCKED` | Required environment unavailable before evaluation |

Aggregation precedence: **FAILED > BLOCKED > PASS**

No partial pass: one required plugin `BLOCKED` → Q5 `BLOCKED`. No averaging.

## Canonical Runner

```bash
uv run python -m tests.system.functional_diagnostics_q5.runner
```

Machine artifact:

```text
.tmp/session/diag-functional-q5/qualification-report.json
```

## Extension Proof

Unit test `test_functional_qualification_runner.py` registers `functional.synthetic_test` without modifying core, analyzer, registry implementation, or runner implementation.

`extension_change_surface` in the machine report records **0** required core changes.

## Adding a Fifth Domain

1. Implement `FunctionalQualificationPlugin` under `tests/system/functional_diagnostics_q5/plugins/`
2. Register in `composition.py`
3. Add plugin ID to `QualificationPlan`

No changes to `FunctionalDiagnosticAnalyzer` or qualification core.

## Check-ID Collision Safety

`FunctionalDiagnosticCheckId` values are namespaced per specification:

| Domain | Candidate check suffix |
|--------|------------------------|
| C1 RAG | `...000002` |
| Q2 Tools | `...000011` |
| Q3 Web | `...000023` |
| Q4 Routing | `...000031` |

Static test: `test_check_ids_are_namespaced_across_domains`

## What Q5 Proves

- One reusable qualification core orchestrates four real domains
- Real Q1–Q4 execution (not cached verdicts)
- Same analyzer identity across plugins
- Cross-domain scope isolation by `(tenant_id, task_id, run_id)`
- Plugin registry with fail-closed duplicate/unknown handling
- Framework extensibility via synthetic plugin

## What Q5 Does NOT Prove

- durable persistence = **OPEN**
- production scale = **OPEN**
- H1 test-suite health = **OPEN**
- all possible AI mechanisms / vendors

## Static Tests

```bash
uv run pytest tests/unit/core/qualification/test_functional_qualification_*.py \
  tests/system/functional_diagnostics_q5/ -q
```

## Q5-R1 — Prerequisite Stabilization (2026-09-02)

### Initial live Q5 (preserved)

```text
INITIAL Q5 LIVE = FAILED
Q1 PASS
Q2 PASS
Q3 FAILED 8/11
Q4 PASS
```

Artifact: `.tmp/session/diag-functional-q5/qualification-report-initial-failure.json`

### Q3 failure audit (D / E / G-A)

| Case | Expected first fail | Actual first fail | Selected source | Canonical in candidates |
|------|---------------------|-------------------|-----------------|------------------------|
| Q3-D | EXTRACTION_VALIDATION | SELECTION | `doc/versions` | yes (rank 2) |
| Q3-E | FINAL | SELECTION | `doc/versions` | yes (rank 2) |
| Q3-G-A | healthy PASS | SELECTION | `doc/versions` | yes (rank 2) |

Root cause: canonical `python-3120` present; selector nondeterministically chose `doc/versions`. Q3-D extraction bias did emit `2023-10-01` but SELECTION failed first. Fault injection did **not** leak upstream (extraction_bias isolated from selection prompt).

### Healthy selector characterization (pre-fix)

Initial failure matrix: Q3-A PASS (canonical rank 1); Q3-D/E/G-A FAIL with identical Tavily mix (rc3 rank 1, canonical rank 2). Selector LLM chose `doc/versions` in downstream/isolation runs.

### R1 corrections

1. **Core**: `functional_qualification_attempts.py` — bounded prerequisite attempts (`SATISFIED` / `NOT_SATISFIED` / `BLOCKED`), anti-cherry-pick (first SATISFIED attempt authoritative).
2. **Q3 runner**: prerequisite-conditioned bounded retries for Q3-D and Q3-E only (max 3). Q3-A and Q3-G-A remain strict (no retry).
3. **Production selector**: deterministic pre-LLM preference for official `/downloads/release/` non-prerelease pages when `failure_layer != source_selection_bias`; improved healthy selection prompt.

### Q3-R1 live result

```text
PASS 11/11
Q3-D authoritative attempt 1 (canonical selection → extraction fail)
Q3-E authoritative attempt 1 (canonical → extraction pass → synthesis fail)
Q3-G-A PASS (canonical selection)
```

### Q5-R1 final live result

```text
PASS
plugins = 4
domains_passed = 4
cases = 44
full_case_match_rate = 100%
prerequisite_success_rate = 100%
cross_domain_isolation_pass = true
```

Artifact: `.tmp/session/diag-functional-q5/qualification-report.json`

### New core module

| Module | Role |
|--------|------|
| `functional_qualification_attempts.py` | Bounded attempt policy, precondition status, attempt records |

### Attempt metrics (Q5 aggregate)

| Metric | Value |
|--------|-------|
| total_attempts | 44 |
| prerequisite_misses | 0 |
| cases_requiring_retry | 0 |
| prerequisite_exhaustions | 0 |
| prerequisite_success_rate | 100% |

### Historical Q3 note

Historical Q3 R3 qualification PASS preserved. Q5 rerun exposed repeatability/stability issue; R1 addressed via prerequisite semantics + production selector stabilization.

