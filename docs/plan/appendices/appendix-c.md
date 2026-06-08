**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix C — Harness audit traceability (Phase Q)

**Purpose:** Every finding from the harness implementation audit (2026-06-01) maps to exactly one Phase Q deliverable. Update **Status** when the deliverable is **Done** / **Won't fix** (with reason).

**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### C.1 Nexus, loops, orchestration, errors

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| N-01 | `NexusLoop` monolith ~1200 lines | Q-N.1 | Done (`orchestration/`; ~586 lines) |
| N-02 | Duplicate `_normalize_human_response` | Q-N.2 | Done |
| N-03 | Dual retry (`RetryEngine` vs `max_run_retries`) | Q-N.3 | Done |
| N-04 | `PolicyEngine` \| `RuntimePolicyEngine` union | Q-N.4 | Done |
| N-05 | Hooks NOT_WIRED: decision, interrupt, retry | Q-N.5 | Done |
| N-06 | Hooks PARTIAL: trace persist | Q-N.6 | Done |
| N-07 | `runtime_steps/tools.py` misleading name | Q-N.7 | Done |
| N-08 | `RuntimeConfig` monolith | Q-N.8 | Done |
| N-09 | `integration_profile: object` | Q-N.9 | Done |
| N-10 | `production_mode` default in lab | Q-N.10 | Done |
| N-11 | Graph callbacks typed `object` | Q-N.11 | Done |
| N-12 | Duplicate import `InterruptType` | Q-N.12 | Done |
| N-13 | `AgentEngine` static UAEP / event_bus | Q-N.13 | Done |
| N-14 | No unit tests `nexus_loop.py` | Q-N.14 | Done |
| N-15 | Thin `GraphExecutor` unit coverage | Q-N.15 | Done |

### C.2 LLM adapters

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| L-01 | Dead `tracked_llm_call` | Q-L.1 | Done |
| L-02 | Empty `llm_adapters/__init__.py` | Q-L.2 | Done |
| L-03 | `architecture/LLM_ADAPTERS.md` missing provider table | Q-L.3 | Done |
| L-04 | `LLMProfile` docstring `max_retries` wrong | Q-L.4 | Done |
| L-05 | `supports_streaming()` default True | Q-L.5 | Done |
| L-06 | PolicyEngine ignores `llm_cost_evaluation` | Q-L.6 | Done |
| L-07 | Dual usage tracking naming | Q-L.7 | Done |
| L-08 | No structured-output conformance | Q-L.8 | Done |
| L-09 | Bedrock context_window TODO | Q-L.9 | Done |
| L-10 | OpenAI-compat `__dict__.update` fragility | Q-L.10 | Done |
| L-11 | Env vars scattered | Q-L.11 | Done |

### C.3 RAG

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| R-01 | Dead `_build_backend_where` / `_map_hits_to_chunks` | Q-R.1 | Done |
| R-02 | Four parallel retrieval paths | Q-R.2 | Done |
| R-03 | `enable_rag` vs `use_rag` in ContextBuilder | Q-R.3 | Done |
| R-04 | `NoPlannerPipeline` always `RagStep` | Q-R.4 | Done |
| R-05 | `top_k` collapses prefetch | Q-R.5 | Done |
| R-06 | `RuntimeConfig` vs `RagProfile` dual config | Q-R.6 | Done |
| R-07 | Unused `RagProfile.extras` | Q-R.7 | Done |
| R-08 | RAG metrics env not in profile | Q-R.8 | Done |
| R-09 | `rag/answers/` parallel stack | Q-R.9 | Done |
| R-10 | `UserProfileManager` bypasses `RetrievalService` | Q-R.10 | Done |
| R-11 | Three “context builder” names | Q-R.11 | Done |
| R-12 | Legacy `use_rag` plan booleans | Q-R.12 | Done |

### C.4 Memory

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| M-01 | No single memory architecture doc | Q-M.1 | Done |
| M-02 | Task memory not visible in scaffold | Q-M.2 | Done |
| M-03 | Silent default when task memory None | Q-M.3 | Done |

### C.5 Observability & metrics

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| O-01 | RAG plugin not in `platform_wiring` | Q-O.1 | Done |
| O-02 | No RAG bridge tests | Q-O.2 | Done |
| O-03 | Parser trace bypasses `ObservabilityBackend` | Q-O.3 | Done |
| O-04 | `metrics/export` substring heuristics | Q-O.4 | Done |
| O-05 | Duplicate import in `metrics/export.py` | Q-O.5 | Done |
| O-06 | `behavioral` never set in export | Q-O.6 | Done |
| O-07 | `/metrics/llm` not on lab host | Q-O.7 | Done |
| O-08 | Observability env scattered | Q-O.8 | Done |
| O-09 | RAG metrics asymmetry vs LLM | Q-O.9 | Done |
| O-10 | `trace_bridge` vs `phase_coverage` drift | Q-O.10 | Done |
| O-11 | Debug router missing type imports | Q-O.11 | Done |
| O-12 | No `trace_bridge` unit tests | Q-O.12 | Done |
| O-13 | Two Prometheus concepts unclear | Q-O.13 | Done |
| O-14 | Runtime events SQLite-first; Cassandra adoption undefined | Q-O.14 | Done |

### C.6 Legacy, style, docs

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| X-01 | Deprecated `ChatAgent` | Q-X.1 | Done |
| X-02 | `task_metadata_bridge` legacy | Q-X.2 | Done |
| X-03 | Copyright / Integrax typo | Q-X.3 | Done |
| X-04 | `tools_base` deprecation | Q-X.4 | Done |
| X-05 | M.6 Future slugs table stale | Q-X.5 | Done |
| D-01 | `docs/README` focus outdated | Q-D.1 | Done |
| D-02 | Canon §52 still “Active” | Q-D.2 | Done |
| D-03 | §0.1 “blocked until L” stale | Q-D.1 (§0.1 fix) | Done |
| D-04 | Guide missing memory/RAG naming | Q-D.4 | Done |
| D-05 | §5.2 process gates not listed for agent authors | Q-D.5 | Done |

### C.7 Tests (cross-cutting)

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| T-01 | NexusLoop unit suite | Q-T.1 / Q-N.14 | Done |
| T-02 | `rag_profile_from_env` tests | Q-T.2 | Done |
| T-03 | `ContextBuilder` tests | Q-T.3 | Done |
| T-04 | `UserProfileManager` tests | Q-T.4 | Done |
| T-05 | Single retrieval per turn test | Q-T.5 | Done |
| T-06 | Platform wiring observability E2E | Q-T.6 | Done |

### C.8 Phase Q paydown log

| Date | Q ID | Summary |
|------|------|---------|
| 2026-06-01 | Q-D.3 | §0.1 strategic objective — Harness GA vs Phase K vs Phase Q |
| 2026-06-01 | Q-O.1,Q-O.2,Q-O.5,Q-O.7 | RAG plugin bootstrap, tests, metrics lint, lab `/metrics/llm` |
| 2026-06-01 | Q-N.2,Q-N.7,Q-N.12 | Duplicate HITL normalize; tool_context_helpers; interrupt import |
| 2026-06-01 | Q-R.1–Q-R.5,Q-R.8 | RAG dead code, single retrieval path, use_rag metadata, prefetch_k |
| 2026-06-01 | Q-L.1,Q-L.2,Q-L.4 | Remove tracked_llm_call; llm_adapters exports; LLMProfile docstring |
| 2026-06-01 | Q-T.2,Q-T.3,Q-T.6 | New unit/integration tests; gate **399 passed** (+2) |
| 2026-06-01 | Q-N.1(partial),Q-N.10,Q-N.13,Q-N.15 | `hitl_runner.py`; lab `harness_production_mode`; AgentEngine `event_bus`; graph checkpoint tests |
| 2026-06-01 | Q-L.9–Q-L.11,Q-O.6,Q-O.11,Q-O.14 | Bedrock windows, OpenAI-compat delegation, LLM env appendix, metrics behavioral, debug types, trace storage §33.1 |
| 2026-06-01 | docs-consolidation | Merged LLM/RAG observability, retry, trace ADR into canon + `architecture/LLM_ADAPTERS.md`; removed satellite `docs/*.md` |
| 2026-06-01 | Q-N.1,Q-X.2,Wave 9 | `graph_runner`, `task_events`, `lifecycle_bridge`; UAEP `execution_options_for_request`; gate **417 passed** |
| 2026-06-01 | Q-X.2(partial),Q-X.4,Q-X.5 | Legacy metadata warnings; `tools_base` timeline; M.6 beta slugs; gate **415 passed** |
| — | — | *(append row per merged PR)* |

**Coverage:** 58 audit rows → 49 unique Q deliverables (some Q IDs satisfy multiple rows). **Target:** 100% **Done** or **Won't fix** — **achieved** (Phase Q complete).

**Appendix B relationship:** Closed by Phase Q where mapped. Residual items tracked in **Appendix D** (Phase Q+).

---
