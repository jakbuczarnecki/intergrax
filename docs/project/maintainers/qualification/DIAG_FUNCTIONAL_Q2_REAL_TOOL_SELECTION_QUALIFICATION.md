# DIAG-FUNCTIONAL-Q2 — Real Tool Selection Qualification

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

Start the real C1 stack (Ollama + LKW):

```powershell
./tests/system/unified_execution/run_c1_proof.ps1
```

Set:

- `LKW_BASE_URL=http://localhost:8021`
- `LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY=ue-11g-c1-certification-secret`

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
| Q2-A | Healthy — correct tool, no false positive |
| Q2-B | Wrong tool selected, invocation succeeds |
| Q2-C | Correct tool selected, invocation fails |
| Q2-D | Correct tool + invocation, final validation fails |
| Q2-E | Missing selection evidence → inconclusive |
| Q2-F | Run isolation (healthy vs wrong tool) |
| Q2-G | Repeated deterministic wrong-tool case (3×) |

## Artifacts

Machine-readable report:

`.tmp/session/diag-functional-q2/qualification-report.json`

## Analyzer

Uses the same `FunctionalDiagnosticAnalyzer` as Q1 with `build_q2_tool_selection_functional_diagnostic_specification`.

Invocation outcome check is independent of selection correctness (no upstream block on wrong-tool + successful invoke).
