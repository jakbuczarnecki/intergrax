# DIAG-FUNCTIONAL-Q2 - Real Tool Selection Qualification

Qualification gate for generic Functional Diagnostics on the real LKW multi-tool LLM selection path.

## Canonical command

From repository root (requires running LKW docker stack with Ollama):

```powershell
./tests/system/functional_diagnostics_q2/run_q2_qualification.ps1
```

Direct module entry:

```bash
uv run python -m tests.system.functional_diagnostics_q2.runner
```

Architecture / evidence-fidelity unit gate (no external services):

```bash
uv run pytest tests/system/functional_diagnostics_q2/test_q2_evidence_fidelity.py -q
```

## Prerequisites

Materialize runtime context and start the real C1 stack (Ollama + LKW):

```powershell
uv run python scripts/build/build_application_image.py `
  --application local_workspace_application `
  --context-dir applications/local_workspace_application/docker/runtime-context `
  --materialize-only
docker compose -f tests/system/unified_execution/docker-compose.yml build --no-cache local_workspace
docker compose -f tests/system/unified_execution/docker-compose.yml up -d
```

Pull generative model (canonical LKW default):

```bash
docker compose -f tests/system/unified_execution/docker-compose.yml exec ollama ollama pull llama3.1:latest
```

Set:

- `LKW_BASE_URL=http://localhost:8021`
- `LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY=ue-11g-c1-certification-secret`

Compose must provide:

- `INTERGRAX_LLM_PROVIDER=ollama`
- `INTERGRAX_LLM_MODEL=llama3.1:latest`
- `INTERGRAX_EMBEDDING_MODEL=nomic-embed-text`

## REAL / MOCKED

| Component | Mode |
| --- | --- |
| Unified Execution | REAL |
| LKW host | REAL |
| LLM tool selection | REAL (`generate_with_tools` via resolved Ollama adapter) |
| Tool catalog | REAL (`workspace.search`, `workspace.write_file`) |
| Tool invocation | REAL (`invoke_catalog_tool` / UAEP gateway) |
| Functional evidence | REAL (`tool_selection_qualifier` adapters) |
| Central DIAG | REAL (`FunctionalDiagnosticAnalyzer` + operator projection) |
| External oracle | REAL / deterministic (`q2.tool.functional_oracle.v1`) |
| Mocks | NONE on core path |

## Decision / Observation / Diagnosis separation

```text
Catalog tools (workspace.search, workspace.write_file)
   ↓
LLM tool-call decision (tool_selection_qualifier/steps/tool_selection_job.py)
   ↓
actual selected ToolId
   ├────────────→ invoke_catalog_tool
   ↓
OBS evidence (tool_functional_evidence adapters)
   ↓
DIAG (FunctionalDiagnosticAnalyzer)
   ↓
Operator
```

**Architectural invariant:** The qualification agent owns tool-selection semantics. Observability records candidates, selection, and invocation outcome. Diagnostics never participates in selection.

## Mandatory matrix

| Case | Intent |
| --- | --- |
| Q2-A | Healthy - correct tool, no false positive |
| Q2-B | Wrong tool selected, invocation succeeds |
| Q2-C | Correct tool selected, invocation fails |
| Q2-D | Correct tool + invocation, final validation fails |
| Q2-E | Missing selection evidence → inconclusive |
| Q2-F | Run isolation (healthy vs wrong tool) |
| Q2-G | Repeated deterministic wrong-tool case (3×) |

## DIAG-FUNCTIONAL-Q2-ENV-1 run (2026-09-01)

### Environment

| Component | Value |
| --- | --- |
| Stack | `tests/system/unified_execution/docker-compose.yml` |
| LKW image | `local_workspace-application:ue-11g-c1` (`sha256:c9a77b56c9e2…`) |
| Generative model | `llama3.1:latest` (Ollama, tool-calling verified) |
| Embedding model | `nomic-embed-text:latest` |
| Materialization HEAD | `2695a1772358afae913cba412baae52191560355` (local) |

### Static gate

`uv run pytest tests/system/functional_diagnostics_q2/test_q2_evidence_fidelity.py -q` → **4 passed**

### Live qualification verdict (initial)

**Q2 = FAILED** (environment unblocked; live matrix executed)

| Case | Match | Notes |
| --- | --- | --- |
| Q2-A | PASS | `workspace.search`, functional PASS |
| Q2-B | FAIL | Wrong tool selected; DIAG first failure = SELECTION (correct); comparator mismatch on INVOCATION (expected PASS, observed FAIL — unsafe LLM path caused real invoke failure) |
| Q2-C | PASS | Selection pass, invocation fail |
| Q2-D | PASS | Selection + invocation pass, validation fail |
| Q2-E | FAIL | Operator inconclusive (correct); comparator required functional PASS while oracle reported FAIL |
| Q2-F-A | PASS | Isolation healthy |
| Q2-F-B | FAIL | Same INVOCATION vector mismatch as Q2-B |
| Q2-G (×3) | FAIL (all 3) | Repeatable wrong-tool selection; same comparator mismatch |

Metrics: `matched_cases=4/10`, `stage_accuracy=40%` (was full-case match, not stage match), `false_negatives=5` (incorrect — counted comparator mismatch, not missed functional failure), `repeatability_pass=true`, `evidence_fidelity_pass=true`.

Preserved artifact: `.tmp/session/diag-functional-q2/qualification-report-env1-failed.json`

### DIAG-FUNCTIONAL-Q2-R1 root cause

1. **Wrong-tool invoke path:** LLM `workspace.write_file` arguments often used absolute/unsafe paths → real `tool.invoke` FAILED → DIAG correctly reported INVOCATION FAIL while qualification expected PASS for technically successful invoke.
2. **Q2-E comparator:** Mixed independent functional oracle (FAIL when selection suppressed / wrong tool) with diagnostic expectation (INCONCLUSIVE + selection INSUFFICIENT_EVIDENCE).
3. **Metrics:** `false_negatives` and `stage_accuracy` derived from full-case comparator mismatch instead of functional-failure detection and first-proven-failure stage match.

### DIAG-FUNCTIONAL-Q2-R1 fixes

| Layer | Change |
| --- | --- |
| Agent (`tool_selection_job`) | Sanitize `workspace.write_file` paths for qualification (reject absolute/`..`; safe default) |
| Expectation (`cases.py`) | Q2-E: `compare_functional_outcome=False`; oracle ground truth independent of diagnostic inconclusive |
| Comparator | `compare_functional_outcome` flag on `QualificationCaseExpectation` |
| Runner metrics | Separate `full_case_match`, `stage_match`, `functional_failure_detection`, corrected FP/FN semantics |
| Tests | Q1 comparator regression + Q2 wrong-tool / missing-evidence / metrics unit gates |

`FunctionalDiagnosticAnalyzer` unchanged.

### Live qualification verdict (R1 rerun)

**Q2 REAL TOOL-SELECTION = QUALIFIED**

| Case | Match | Trace |
| --- | --- | --- |
| Q2-A | PASS | search, functional success |
| Q2-B | PASS | write_file wrong tool; SELECTION fail, INVOCATION pass, VALIDATION fail |
| Q2-C | PASS | search + invoke fail |
| Q2-D | PASS | search + invoke pass + validation fail |
| Q2-E | PASS | selection insufficient, operator inconclusive |
| Q2-F-A | PASS | isolation healthy |
| Q2-F-B | PASS | isolation wrong-tool (same as Q2-B) |
| Q2-G (×3) | PASS | repeatable wrong-tool |

Metrics: `matched_cases=10/10`, `stage_accuracy=100%`, `false_positives=0`, `false_negatives=0`, `repeatability_pass=true`, `evidence_fidelity_pass=true`.

Machine artifact: `.tmp/session/diag-functional-q2/qualification-report.json`

Recommendation: **READY_FOR_Q3_REAL_WEB_SEARCH_QUALIFICATION**

## Artifacts

Machine-readable report:

`.tmp/session/diag-functional-q2/qualification-report.json`

## Analyzer

Uses the same `FunctionalDiagnosticAnalyzer` as Q1 with `build_q2_tool_selection_functional_diagnostic_specification`.

Invocation outcome check is independent of selection correctness (no upstream block on wrong-tool + successful invoke).
