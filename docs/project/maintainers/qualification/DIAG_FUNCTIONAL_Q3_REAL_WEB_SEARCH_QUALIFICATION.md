# DIAG-FUNCTIONAL-Q3 - Real Web Search Qualification

Qualification gate for generic Functional Diagnostics on the real web-search pipeline
(query construction → SearchProvider → source selection → extraction → synthesis).

## Canonical command

From repository root (requires running LKW docker stack with Ollama + search provider credentials):

```powershell
./tests/system/functional_diagnostics_q3/run_q3_qualification.ps1
```

Direct module entry (loads root `.env` via `scripts/proof/intergrax_proof_environment.py` — no manual key export):

```bash
uv run python -m tests.system.functional_diagnostics_q3.runner
```

Docker stack (explicit env file):

```bash
docker compose --env-file .env -f tests/system/unified_execution/docker-compose.yml up -d
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
docker compose --env-file .env -f tests/system/unified_execution/docker-compose.yml build --no-cache local_workspace
docker compose --env-file .env -f tests/system/unified_execution/docker-compose.yml up -d
```

Pull generative model:

```bash
docker compose -f tests/system/unified_execution/docker-compose.yml exec ollama ollama pull llama3.1:latest
```

Set:

- `LKW_BASE_URL=http://localhost:8021`
- `LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY=ue-11g-c1-certification-secret`
- **One** real search provider credential (first match wins); canonical source is repository-root `.env` loaded by the Q3 runner / proof environment loader (`override=False`, process env wins):
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

## DIAG-FUNCTIONAL-Q3 live run (2026-09-02)

### Environment audit

| Item | Value |
| --- | --- |
| START_HEAD | `640f2c9d0db633cdce49e5cb256f5e6ab30fc512` |
| FINAL_HEAD | `17d04462ca7db17ba33bedf3ee4275d17b25844e` |
| INTERGRAX_TAVILY_API_KEY | YES (host + compose + LKW runtime) |
| Compose propagation | YES (`INTERGRAX_TAVILY_API_KEY: ${INTERGRAX_TAVILY_API_KEY:-}`) |
| Host/provider preflight | `tavily`, REAL network, 3 hits, auth success |
| Runtime-context materialized | YES (`web_search_qualifier`, `local.workspace.web_search_qualification`) |
| LKW image rebuild | YES (`local_workspace-application:ue-11g-c1`, `ce82b0626a6f`) |
| Static gate | `4 passed in 1.50s` |

### Live qualification verdict

**Q3 = FAILED** — real Tavily + full matrix executed; 6/11 cases MATCH, 5 MISMATCH, FP=2, FN=0, stage_accuracy=54.5%, inconclusive_accuracy=0%, repeatability=PASS, evidence_fidelity=PASS.

Preserved artifact: `.tmp/session/diag-functional-q3/qualification-report.json`

### Recommendation

**Q3-R1 REQUIRED** — tune DIAG comparator/spec/oracle for selection-stage attribution and Q3-F inconclusive projection before re-qualification.

## DIAG-FUNCTIONAL-Q3-R1 (2026-09-02)

### Environment audit

| Item | Value |
| --- | --- |
| START_HEAD | `17d04462ca7db17ba33bedf3ee4275d17b25844e` |
| FINAL_HEAD | `b8208944fa63b3b3f3c021e0df985fba853ac0bb` |
| INTERGRAX_TAVILY_API_KEY | YES (host + compose + LKW runtime) |
| Runtime-context materialized | YES |
| LKW image rebuild | YES |
| Static gate | `18 passed` (Q3 source semantics + evidence fidelity + Tavily URL + Q2 regressions) |

### Initial live result (preserved)

**INITIAL LIVE RESULT = FAILED 6/11** — artifact: `.tmp/session/diag-functional-q3/qualification-report-initial-failure.json` (if present) or prior `qualification-report.json` from first real Tavily run documented above.

### Root causes fixed in R1

1. **Expected-source identity** — oracle no longer picks first ranked `python-312*` candidate; canonical expected source is always `python-3120` final release.
2. **Extraction injection** — Q3-D upstream bias + fallback replaces correct LLM extraction with `2023-10-01` when bias is ignored.
3. **Q3-F missing evidence** — `Oct. 2, 2023` accepted by oracle; missing selection yields `INSUFFICIENT_EVIDENCE` + `INCONCLUSIVE` without false `EXTRACTION_VALIDATION` fail.
4. **Tavily URL** — factory posts to `https://api.tavily.com/search` (no trailing-slash base URL).

### Live qualification verdict (R1)

**Q3 = PASS** — 11/11 MATCH, FP=0, FN=0, stage_accuracy=100%, inconclusive_accuracy=100%, repeatability=PASS, evidence_fidelity=PASS.

Preserved artifact: `.tmp/session/diag-functional-q3/qualification-report.json`

### Recommendation

**Q3 REAL WEB SEARCH = QUALIFIED** — `READY_FOR_Q4_REAL_MODEL_ROUTING_QUALIFICATION`

## DIAG-FUNCTIONAL-Q3-R2 (2026-09-02)

### Independent audit finding (R1)

R1 live runner achieved **11/11 MATCH**, but independent audit **rejected qualification authority** because Q3-C/D used post-decision forcing:

- Q3-C discarded canonical LLM selection and programmatically chose RC/wrong candidate.
- Q3-D replaced correct LLM extraction with `2023-10-01` when bias was ignored.

Preserved history:

| Phase | Result |
| --- | --- |
| INITIAL | BLOCKED (no provider credentials) |
| INITIAL LIVE | FAILED 6/11 |
| R1 live | 11/11 — rejected by independent audit (self-fulfilling injection) |

### R2 mechanism changes

| Layer | R1 (removed) | R2 (allowed) |
| --- | --- | --- |
| Q3-C | discard canonical LLM URL → force RC | pre-decision bias only: selection prompt, candidate reordering, ranking context |
| Q3-D | replace correct LLM date with `2023-10-01` | pre-decision bias only: extraction prompt; raw extractor output == emitted fact |

New static gate: `test_q3_anti_forcing.py` (behavioral + AST).

New live gates: `selection_decision_fidelity`, `extraction_decision_fidelity`, `post_decision_forcing = NONE`.

### R2 first attempt (env wiring)

**Q3-R2 = BLOCKED** — runner did not load repository-root `.env`; `INTERGRAX_TAVILY_API_KEY` absent on host process; static gates **37 passed**; full 11-case live matrix not executed.

### R2-LIVE (2026-09-02) — canonical `.env` wiring + authentic matrix

| Item | Value |
| --- | --- |
| START_HEAD | `b35e4fc233fc085565be459bc20bfe85fa76f843` |
| FINAL_HEAD | `a28dfd970ba8a22e986c52eeeb839d02229af64e` |
| Root `.env` | YES |
| Key loaded (no manual export) | YES |
| Canonical loader | `scripts/proof/intergrax_proof_environment.py` |
| Compose `--env-file .env` | YES |
| Container key non-empty | YES |
| Tavily preflight | provider=tavily, network=REAL, auth=PASS, hits≥1 |
| Static gates | **41 passed** (+ `test_q3_proof_environment.py`) |
| Materialize + LKW rebuild | YES |

Preserved history:

| Phase | Result |
| --- | --- |
| INITIAL | BLOCKED (no provider credentials) |
| INITIAL LIVE | FAILED 6/11 |
| R1 live | 11/11 — rejected by independent audit (self-fulfilling injection) |
| R2 first attempt | BLOCKED — `.env` not loaded by runner |

### R2-LIVE authentic matrix result

**Q3 = BLOCKED** — mandatory controlled real failure not inducible without post-decision forcing.

| Gate | Result |
| --- | --- |
| Q3-C (selection bias) | INDUCED — LLM selected `python-3120rc3` (wrong) under pre-decision bias; DIAG `SELECTION` fail |
| Q3-D (extraction bias) | **NOT INDUCED** — LLM extracted `2023-10-02` despite bias; blocked at `q3_d_not_inducible` |
| Q3-F | PASS trace — selection absent → `INSUFFICIENT_EVIDENCE` → extraction PASS → operator `INCONCLUSIVE` |
| post_decision_forcing | **NONE** |

Preserved artifact: `.tmp/session/diag-functional-q3-r2/qualification-report.json`

### Recommendation

Q3-D extraction bias cannot honestly induce wrong-date failure with current Ollama `llama3.1:latest` without post-decision override. **Do not reintroduce forcing.** Options: stronger pre-decision extraction bias (R3), alternate model/temperature, or accept Q3-D as non-inducible boundary case. Q3-C selection bias remains inducible on live Tavily candidates.

## H1

OPEN (DIAG 100k/flaky suite not in scope).

## Remaining

Q4 model routing, durable persistence, production scale, H1.
