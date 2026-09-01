# DIAG-FUNCTIONAL-Q1 — Real RAG/C1 Qualification

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

## Mandatory matrix

| Case | Intent |
| --- | --- |
| Q1-A | Healthy — no false positive |
| Q1-B | Controlled selection failure |
| Q1-C | Synthesis failure with correct upstream |
| Q1-D | Missing selection evidence → inconclusive operator view |
| Q1-E | Run isolation (healthy vs failure) |
| Q1-F | Repeated deterministic failure (3x) |

## Artifacts

Machine-readable report:

`.tmp/session/diag-functional-q1/qualification-report.json`

## Outcome (2026-09-01)

```
Q1 REAL RAG/C1 QUALIFICATION = PASS

REAL INTEGRATION CORRECTNESS = QUALIFIED FOR Q1
GROUND-TRUTH AGREEMENT = 100% (9/9)
FALSE POSITIVES = 0
FALSE NEGATIVES = 0
STAGE ACCURACY = 100%
REPEATABILITY = PASS

DURABLE PERSISTENCE = NOT YET QUALIFIED
PRODUCTION SCALE = NOT YET QUALIFIED
```

HEAD at qualification: `ad897623566a016eddba329855537b86f067af9e`

## Out of scope (remain open)

- Durable functional evidence persistence
- Production scale
- Q2 tool-selection qualification
- H1 test-suite health
