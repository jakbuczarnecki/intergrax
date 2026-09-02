# DIAG-FUNCTIONAL-Q3 - Real Web Search Qualification

Qualification gate for generic Functional Diagnostics on the real web-search pipeline
(query construction → SearchProvider → source selection → extraction → synthesis).

## Canonical command

From repository root (requires running LKW docker stack with Ollama + search provider credentials):

```powershell
./tests/system/functional_diagnostics_q3/run_q3_qualification.ps1
```

Direct module entry:

```bash
uv run python -m tests.system.functional_diagnostics_q3.runner
```

Architecture / evidence-fidelity unit gate (no external services):

```bash
uv run pytest tests/system/functional_diagnostics_q3/test_q3_evidence_fidelity.py -q
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

Pull generative model:

```bash
docker compose -f tests/system/unified_execution/docker-compose.yml exec ollama ollama pull llama3.1:latest
```

Set:

- `LKW_BASE_URL=http://localhost:8021`
- `LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY=ue-11g-c1-certification-secret`
- **One** real search provider credential (first match wins):
  - `INTERGRAX_TAVILY_API_KEY` (preferred)
  - `INTERGRAX_BRAVE_API_KEY`
  - `INTERGRAX_GOOGLE_CSE_API_KEY` + `INTERGRAX_GOOGLE_CSE_CX`
  - `INTERGRAX_EXA_API_KEY`

Pass credentials into the LKW container via host env (docker-compose forwards `${INTERGRAX_TAVILY_API_KEY:-}` etc.).

## REAL / MOCKED

| Component | Mode |
| --- | --- |
| Unified Execution | REAL |
| LKW host | REAL |
| LLM (query/selection/extraction/synthesis) | REAL (Ollama `llama3.1:latest` via resolved adapter) |
| SearchProvider abstraction | REAL |
| Provider adapter | REAL (tavily/brave/google_cse/exa — first configured) |
| External web provider | REAL (network search request) |
| Source-selection mechanism | REAL (`web_search_qualifier` LLM) |
| Extraction mechanism | REAL (`web_search_qualifier` LLM on provider snippet) |
| Synthesis mechanism | REAL (`web_search_qualifier` LLM) |
| Functional evidence | REAL (`web_search_qualifier` adapters) |
| Central DIAG | REAL (`FunctionalDiagnosticAnalyzer` + operator projection) |
| External oracle | REAL / deterministic (`q3.web.*.v1`) |
| Mocks | **NONE** on core path |

## Stable ground truth

| Field | Value |
| --- | --- |
| Task | Python 3.12.0 release date from official python.org |
| Expected fact | `2023-10-02` / `October 2, 2023` |
| Expected authoritative source | `python.org` release page (`python-3120`) |

Oracle does **not** require identical full result ranking — only authoritative source presence, correct selection, and fact correctness where applicable.

## Mandatory matrix

| Case | Intent |
| --- | --- |
| Q3-A | Healthy — full pipeline success |
| Q3-B | Wrong query sent to real provider |
| Q3-C | Wrong source selected (official source must be in candidates) |
| Q3-D | Wrong extracted intermediate fact |
| Q3-E | Wrong final synthesis |
| Q3-F | Missing selection evidence → inconclusive |
| Q3-G | Run isolation (healthy vs wrong source) |
| Q3-H | Repeated wrong-source case (3×) |

## DIAG-FUNCTIONAL-Q3 initial run (2026-09-02)

### Environment audit

| Item | Value |
| --- | --- |
| START_HEAD | `49173085b798648a1d3c97c44ef67d0483c93627` |
| SearchProvider contract | `intergrax/integrations/contracts/search_provider.py` |
| Production adapter path | `create_tavily_search_provider` / `create_brave_search_provider` / `create_google_cse_search_provider` / `create_exa_search_provider` |
| Workload entry | `local.workspace.web_search_qualification` → `web_search_qualifier` |
| Static gate | `test_q3_evidence_fidelity.py` |

### Credentials (operator host)

| Provider | Configured |
| --- | --- |
| INTERGRAX_TAVILY_API_KEY | NO |
| INTERGRAX_BRAVE_API_KEY | NO |
| INTERGRAX_GOOGLE_CSE_* | NO |
| INTERGRAX_EXA_API_KEY | NO |

### Live qualification verdict (initial)

**Q3 = BLOCKED** — no search provider credentials on qualification host; real external search cannot execute.

Preserved artifact: `.tmp/session/diag-functional-q3/qualification-report.json`

### Recommendation

Configure a real search provider credential, rebuild/restart LKW with forwarded env, re-run canonical runner. Do not tune comparator/spec until live matrix executes.

## H1

OPEN (DIAG 100k/flaky suite not in scope).

## Remaining

Q4 model routing, durable persistence, production scale, H1.
