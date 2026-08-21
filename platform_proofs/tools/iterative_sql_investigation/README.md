# TOOLS-ITERATIVE-SQL-INVESTIGATION

**Proof ID:** `TOOLS-ITERATIVE-SQL-INVESTIGATION`  
**Domain:** TOOLS (platform proof — not product, not LKW)  
**Status:** EXECUTABLE (PP-3C)

## Claim

The bounded iterative tool runtime can use real SQL observations to drive subsequent evidence-dependent tool calls and reach a bounded conclusion while preserving explicit proof of the investigation chain.

## Why it matters

Reference TOOLS domain platform proof. Exercises reusable investigation semantics (multi-hop tool selection, evidence dependencies, bounded termination) independent of any product workflow.

## Architecture under proof

- [`docs/project/architecture/TOOLS.md`](../../../docs/project/architecture/TOOLS.md)
- `ToolPlanningService` / bounded ReAct via `run_bounded_tool_loop()`
- ENG-5 investigation policy (`tools_investigation_policy`)
- ENG-6 `InvestigationProof`

## Real boundaries

| Boundary | Status |
|----------|--------|
| PostgreSQL | Real, isolated Docker |
| Intergrax tool runtime | Real `ToolRegistry` + `RuntimeToolInvoker` |
| SQL tool | Proof-owned read-only contract |
| Model provider | Real adapter from `INTERGRAX_LLM_*` |
| Bounded tool loop | Canonical `run_bounded_tool_loop()` — no proof-local loop |

## Canonical dataset lifecycle

**Source of truth:** deterministic generator + fixed seed + dataset version + row count + ground-truth invariant contract.

**No canonical CSV is committed.** Bulk CSV, if ever used, is temporary transport only.

| Field | Value |
|-------|-------|
| `dataset_id` | `TOOLS-ITERATIVE-SQL-INVESTIGATION-DATASET` |
| `dataset_version` | `v1` |
| `seed` | `42` |
| `row_count` | `100000` |
| `ground_truth_version` | `A1-B1-C1` |
| `schema_identity` | `proof.parcel_events/v1` |

**Fingerprint:** canonical JSON of identity fields → SHA-256 (does not hash generated rows).

**Setup semantics:** TRUNCATE → regenerate canonical 100k → reload → verify materialized PostgreSQL invariants. Model execution aborts if verification fails.

Docker (`docker-compose.yml`) provides infrastructure only (PostgreSQL, roles, schema permissions). Experiment data materialization runs via `setup.py` / proof entrypoint — not Docker startup.

## Model isolation from ground truth

The real LLM receives only:

- analyst-safe table/schema description
- scenario question
- tool contract
- ENG-5 investigation policy (via `ToolPlanningService`)

It does **not** receive `dataset.py`, planted constants, expected answers, invariant implementation, or fingerprint semantics that reveal hidden answers.

## Scenarios

| ID | Question (summary) | PASS focus |
|----|-------------------|------------|
| **A** | North delay rate vs other regions — strongest operational explanation | Multi-hop SQL evidence; anomaly segment; not volume-only trap; valid `InvestigationProof` |
| **B** | Is heavier weight itself the likely cause of delays? | Global association then segmented verification; must not claim direct causation |
| **C** | Are staffing shortages the reason for delays? | No fabricated staffing evidence; bounded missing-evidence conclusion |

Typed semantic outcomes: `ScenarioAOutcome`, `ScenarioBOutcome`, `ScenarioCOutcome` in `evaluator.py`.

## Provider requirements

Configure via standard Intergrax env (no proof-specific HTTP client):

| Variable | Role |
|----------|------|
| `INTERGRAX_LLM_PROVIDER` | Required — adapter slug (native tool calling) |
| `INTERGRAX_LLM_MODEL` | Required — model id |
| Provider credential env | As required by selected adapter (e.g. API key); local `ollama` / `vllm` need no key |

PostgreSQL:

| Variable | Default |
|----------|---------|
| `INTERGRAX_PP_SQL_INVESTIGATION_ADMIN_DSN` | `postgresql://proof_admin:proof_admin_local@localhost:5435/iterative_sql_proof` |
| `INTERGRAX_PP_SQL_INVESTIGATION_DSN` | `postgresql://proof_runtime:proof_runtime_local@localhost:5435/iterative_sql_proof` |

## Setup

```bash
docker compose -f platform_proofs/tools/iterative_sql_investigation/docker-compose.yml up -d
```

Validate dataset materialization only (no LLM):

```bash
uv run python platform_proofs/tools/iterative_sql_investigation/run_proof.py --validate-only
```

## Canonical proof run

```bash
uv run python platform_proofs/tools/iterative_sql_investigation/run_proof.py
```

Via suite runner (FULL or LIVE profile):

```bash
uv run python scripts/proof/run-intergrax-proof-suite.py --profile full
```

Manifest entry: `TOOLS-ITERATIVE-SQL-INVESTIGATION` in `scripts/proof/intergrax_proof_manifest.py`.

## Invocation limits

- `max_iterations`: 8
- `max_tool_calls_per_round`: 2
- `max_identical_tool_call_repeats`: 2
- Existing runtime wall-time / budget semantics remain enabled

## SQL safety boundary

1. Proof-local validator — SELECT/WITH only
2. Database authorization — runtime role read-only
3. Bounded fetch — max 200 visible rows

## Limitations

- Semantic evaluator uses bounded heuristics on tool traces + final answer — not a second judge LLM
- Provider API keys are not expressible as manifest `EnvRequirement` kinds; missing credentials fail at proof start (documented gap)
- Not QUALIFIED — requires accepted real-model qualification under PP-4
- Synthetic logistics dataset — no production rules
- Does not prove product workflows, universal model certification, or production readiness

## What this does NOT prove

- LKW or product Quick Start consumption
- Public claim promotion (`docs/project/proofs/PROOFS.md` unchanged until PP-4/PP-5)
- QUALIFIED coverage
- Commercial or real-user validation

## PP-3B artifacts

| Artifact | Role |
|----------|------|
| `dataset.py` | Deterministic generator + invariant contract |
| `dataset_identity.py` / `setup.py` | Identity, fingerprint, materialization, DB verification |
| `sql_tool.py` | Read-only SQL tool |
| `runtime.py` | PostgreSQL → registry → invoker |
| `investigation_runtime.py` | Canonical planner + bounded loop wiring |
| `evaluator.py` / `proof_result.py` | Machine-checkable scenario outcomes |
| `run_proof.py` | Canonical executable entrypoint |
