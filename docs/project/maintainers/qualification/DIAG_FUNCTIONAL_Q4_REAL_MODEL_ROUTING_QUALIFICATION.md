# DIAG-FUNCTIONAL-Q4 - Real Model Routing Qualification

Qualification gate for generic Functional Diagnostics on the real LKW production LLM routing path.

## Canonical command

```powershell
./tests/system/functional_diagnostics_q4/run_q4_qualification.ps1
```

```bash
uv run python -m tests.system.functional_diagnostics_q4.runner
```

Static gates (no external services):

```bash
uv run pytest tests/system/functional_diagnostics_q4/ -q
```

## Prerequisites

Same UE-11G-C1 LKW + Ollama stack as Q1–Q3. Docker Ollama must include **both**:

- `llama3.1:latest` (PROFILE_A — quality / primary)
- `qwen2.5:7b` (PROFILE_B — budget / cheap)

Host preflight also checks local `ollama list` for both models.

```powershell
uv run python scripts/build/build_application_image.py `
  --application local_workspace_application `
  --context-dir applications/local_workspace_application/docker/runtime-context `
  --materialize-only
docker compose -f tests/system/unified_execution/docker-compose.yml build local_workspace
docker compose -f tests/system/unified_execution/docker-compose.yml up -d
docker compose -f tests/system/unified_execution/docker-compose.yml exec ollama ollama pull qwen2.5:7b
```

Environment:

- `LKW_BASE_URL=http://localhost:8021`
- `LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY=ue-11g-c1-certification-secret`

Machine artifact: `.tmp/session/diag-functional-q4/qualification-report.json`

## Production routing architecture

```text
TASK / qualification metadata
   ↓
RoutingContext (task_class, budget_remaining_ratio)
   ↓
LLMRoutingProfile.rules → LLMRoutingEvaluator
   ↓
ModelRouter policy hints (fallback ordering when applicable)
   ↓
resolve_llm_adapter → RoutingEvaluatingLLMAdapter
   ↓
selected LLMProfile → real Ollama adapter
   ↓
model.generate (real invocation)
   ↓
FunctionalEvidence (generic kinds)
   ↓
FunctionalDiagnosticAnalyzer (unchanged)
   ↓
Operator projection
```

**Decision owners**

| Concern | Owner |
| --- | --- |
| Routing context | Workload metadata → `RoutingContext` |
| Routing rules | `LLMRoutingProfile` on LKW env (qualification profile) |
| Selected profile | `LLMRoutingEvaluator` + `ModelRouter` |
| Adapter creation | `resolve_llm_adapter` / `create_adapter_for_routing_evaluation` |
| Model invocation | Ollama provider adapter |
| Diagnostics | `FunctionalDiagnosticAnalyzer` only interprets evidence |

`intergrax/runtime/token_optimization/llm_router.py` is **not** model-routing authority (Token Optimization configuration selection only).

## Routing profiles

| Profile | Provider | Model | Role |
| --- | --- | --- | --- |
| PROFILE_A | `ollama` | `llama3.1:latest` | Primary / quality (`model_routing_primary`) |
| PROFILE_B | `ollama` | `qwen2.5:7b` | Budget / cheap (`budget_remaining_ratio < 0.25`) |
| INVOKE_FAIL | `ollama` | `diag-q4-nonexistent-model-xyz` | Scoped invocation failure (`model_routing_invoke_fail`) |

Artifact identity: `llm:ollama:<model>` (e.g. `llm:ollama:llama3.1:latest`).

## REAL / MOCKED

| Component | Mode |
| --- | --- |
| Unified Execution / LKW | REAL |
| `LLMRoutingEvaluator` / `ModelRouter` | REAL |
| `resolve_llm_adapter` | REAL |
| Ollama PROFILE_A / PROFILE_B | REAL |
| `model.generate` | REAL |
| Functional evidence | REAL (`model_routing_qualifier`) |
| `FunctionalDiagnosticAnalyzer` | REAL (unchanged) |
| Oracle | REAL / deterministic (`q4.model.functional_oracle.v1`) |
| Core mocks | **NONE** |

## Mandatory matrix

| Case | Intent |
| --- | --- |
| Q4-A | Healthy routing → PROFILE_A, success |
| Q4-B | Wrong route (budget bias) → PROFILE_B, invocation succeeds, SELECTION fails first |
| Q4-C | Correct invoke-fail profile, real provider failure |
| Q4-D | Correct route + biased prompt → VALIDATION fails |
| Q4-E | Missing SELECTION evidence → INCONCLUSIVE |
| Q4-F | Isolation (healthy vs wrong route) |
| Q4-G | Wrong-route repeatability ×3 |

## DIAG-FUNCTIONAL-Q4 live run (2026-09-02) — matrix PASS (pre-R1 audit)

| Metric | Result |
| --- | --- |
| Verdict | **PASS** (implementation matrix) |
| full_case_match | 10/10 (100%) |
| stage_accuracy | 100% |
| inconclusive_accuracy | 100% (Q4-E) |
| FP / FN | 0 / 0 |
| repeatability | PASS |
| evidence_fidelity | 100% |
| routing_decision_fidelity | 100% |
| post_decision_forcing | NONE |
| post_generation_forcing | NONE |

**Independent audit:** qualification authority **not accepted** — workload ran its own `LLMRoutingEvaluator.evaluate(...)` before `RoutingEvaluatingLLMAdapter` ran production evaluation (double routing decision).

**Q4-R1 note:** First live matrix failed on Q4-D (model ignored weak bias) and adapter summary propagation; fixed before pre-R1 PASS run (not repaired mid-run).

## Q4-R1 ENTERPRISE AUTHORITY CORRECTION

**Problem:** `model_routing_job` pre-computed routing via `LLMRoutingEvaluator().evaluate(...)` while `RoutingEvaluatingLLMAdapter._refresh_inner_adapter()` performed a second authoritative evaluation. Evidence tracked the first (qualification-side) decision, not the production execution path.

**Design (R1):**

```text
ONE routing evaluation per model call
RoutingEvaluatingLLMAdapter._refresh_inner_adapter()
  → LLMRoutingEvaluator.evaluate(...)
  → on_evaluated observer (chained, restored in finally)
  → inner adapter swap
  → real model invocation
  → FunctionalEvidence from observed RoutingEvaluation only
```

- Qualification-side evaluator: **NONE**
- Observer: production `set_on_evaluated` via `begin_routing_observation` / `end_routing_observation` (restore previous observer + context provider)
- Concurrency: per-run observer install/restore; isolation matrix proves F-A vs F-B decisions do not cross-leak
- Strict typing: `ObservedRoutingDecision`, `Q4QualificationRequest`, `JsonObject` diagnostics serialization

See machine artifact after R1 run: `.tmp/session/diag-functional-q4/qualification-report.json` (prior PASS preserved as `qualification-report-pre-r1.json`).

## DIAG-FUNCTIONAL-Q4-R1 live run (2026-09-02)

| Metric | Result |
| --- | --- |
| Verdict | **PASS** |
| full_case_match | 10/10 (100%) |
| authoritative_routing_observation_fidelity | 100% |
| qualification_routing_recomputation | NONE |
| observer_cross_run_leakage | FALSE |
| post_decision_forcing | NONE |
| post_generation_forcing | NONE |
| repeatability | PASS |
| FP / FN | 0 / 0 |

## Status (post-R1)

```text
Q1 REAL RAG             ✅ QUALIFIED
Q2 REAL TOOL SELECTION  ✅ QUALIFIED
Q3 REAL WEB SEARCH      ✅ QUALIFIED
Q4 REAL MODEL ROUTING   ✅ QUALIFIED (R1 enterprise authority)

Q5 CROSS-DOMAIN FINAL   ▶ NEXT
```

Recommendation: `READY_FOR_Q5_CROSS_DOMAIN_FUNCTIONAL_DIAGNOSTICS_QUALIFICATION`

H1 (100k suite cleanup): OPEN  
Durable FunctionalEvidence persistence: OPEN  
Production scale benchmark: OPEN
