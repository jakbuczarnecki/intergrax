# DIAG-FUNCTIONAL-Q1 - Real RAG/C1 Qualification

Qualification gate for Functional Diagnostics on the real UE-11G-C1 / LKW RAG path.

## Canonical command

From repository root (requires running LKW docker stack):

```powershell
./tests/system/functional_diagnostics_q1/run_q1_qualification.ps1
```

Direct pytest (after services are up and `LKW_BASE_URL` is set):

```bash
uv run pytest tests/system/functional_diagnostics_q1/ -m qualification -vv
```

Architecture / evidence-fidelity unit gate (no external services):

```bash
uv run pytest tests/system/functional_diagnostics_q1/test_q1_evidence_fidelity.py -q
```

## Prerequisites

Start the real C1 stack (Qdrant, Ollama, Mongo, LKW):

```powershell
./tests/system/unified_execution/run_c1_proof.ps1
```

Or compose only:

```powershell
docker compose -f tests/system/unified_execution/docker-compose.yml up --build local_workspace
```

Set:

- `LKW_BASE_URL=http://localhost:8021`
- `LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY=ue-11g-c1-certification-secret`

## REAL / MOCKED

| Component | Mode |
| --- | --- |
| Unified Execution | REAL |
| Qdrant / retrieval | REAL |
| Embedding (Ollama) | REAL |
| LLM / model path | REAL (agentic LKW reflex + pipeline) |
| Functional evidence | REAL (instrumentation in `local_search` / `local_synthesizer`) |
| Central DIAG | REAL (`FunctionalDiagnosticAnalyzer` + operator projection) |
| External oracle | REAL / deterministic (`c1.rag.date_oracle.v1`) |
| Mocks | NONE on core path |

## Decision / Observation / Diagnosis separation

```text
Qdrant
   ↓
Search selection (local_search/retrieval_selection.py)
   ↓
actual selection (selected_artifact_ref in search_summary)
   ├────────────→ normal pipeline / synthesizer handoff
   ↓
OBS evidence (rag_functional_evidence adapters)
   ↓
DIAG (FunctionalDiagnosticAnalyzer)
   ↓
Operator
```

**Architectural invariant:** Search owns selection semantics. Observability records the selected artifact. Diagnostics never participates in selection.

## Evidence independence proof

Qualification expectation and DIAG specification do **not** control emitted functional evidence.

- Instrumentation records pipeline facts only (candidates, actual top-1 selection, output relation).
- `qualification_force_selection_artifact_ref` and recorder-returned fidelity summaries are **removed**.
- Q1-B injects failure **before selection** via retrieval-ranking query bias (decoy ranks first).
- Q1-C injects synthesis failure via qualification `draft` input to the real synthesize step.
- Static gate: production instrumentation must not import qualification oracle/comparator modules.

## Mandatory matrix

| Case | Intent |
| --- | --- |
| Q1-A | Healthy - no false positive |
| Q1-B | Controlled real selection failure (ranking query → decoy top-1) |
| Q1-C | Synthesis failure with correct upstream selection |
| Q1-D | Missing selection evidence → inconclusive operator view |
| Q1-E | Run isolation (healthy vs failure) |
| Q1-F | Repeated deterministic selection failure (3x) |
| Q1-H | Historical wrong-date reproduction (observational) |

## Artifacts

Machine-readable report:

`.tmp/session/diag-functional-q1/qualification-report.json`

Fields include `provider_candidates`, `actual_selected_artifact`, `emitted_selected_artifact`, `evidence_fidelity_match`, and per-kind fidelity booleans.

## Out of scope (remain open)

- Durable functional evidence persistence
- Production scale
- Q2 tool-selection qualification
- H1 test-suite health
