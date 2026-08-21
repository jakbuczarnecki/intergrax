# TOOLS-ITERATIVE-SQL-INVESTIGATION

**Proof ID:** `TOOLS-ITERATIVE-SQL-INVESTIGATION`  
**Domain:** TOOLS (platform proof — not product, not LKW)  
**Status:** DESIGNED — deterministic infrastructure implemented (PP-3B)

## Claim

The bounded iterative tool runtime can use real SQL observations to drive subsequent evidence-dependent tool calls and reach a bounded conclusion while preserving explicit proof of the investigation chain.

## Why it matters

This is the reference TOOLS domain platform proof. It exercises reusable investigation semantics (multi-hop tool selection, evidence dependencies, bounded termination) independent of any product workflow.

## Architecture under proof

- [`docs/project/architecture/TOOLS.md`](../../../docs/project/architecture/TOOLS.md)
- ToolPlanningService / bounded ReAct path (planned PP-3C)
- ENG-5 investigation policy (planned)
- ENG-6 InvestigationProof (planned)

## Real boundaries

| Boundary | PP-3B | Planned |
|----------|-------|---------|
| PostgreSQL | Real, isolated Docker | Same |
| Intergrax tool runtime | Real `ToolRegistry` + `RuntimeToolInvoker` | Same |
| SQL tool | Proof-owned read-only contract | Same |
| Model provider | **Not yet** | PP-3C |
| Bounded tool loop | **Not yet** | PP-3C |

## Deterministic infrastructure (PP-3B)

Implemented under this directory:

| Artifact | Role |
|----------|------|
| `dataset.py` | Deterministic synthetic logistics dataset (`parcel_events`) with planted A/B/C structure |
| `sql_tool.py` | Read-only SQL validation, DB-level bounded fetch, typed output |
| `runtime.py` | Wires canonical `create_postgresql_relational_store()` → proof tool → registry → invoker |
| `docker-compose.yml` + `sql/bootstrap.sql` | Isolated PostgreSQL with read-only runtime role |
| `contracts.py` | Typed Pydantic tool I/O (`platform_proof.sql.query`) |

## Planned scenarios

| Scenario | Planted signal | Expected investigation lesson |
|----------|----------------|------------------------------|
| **A** | North aggregate delay looks worse; naive hub counts implicate high-volume hub | Rate/segmentation reveals North + express + long_haul as true anomaly that explains North elevation |
| **B** | Weight correlates with delay globally | Controlling for route/service removes weight as direct cause |
| **C** | No staffing columns exist | Staffing causation is unresolvable from available evidence |

Scenarios are **not executed** until PP-3C adds model + planner + bounded loop.

## Setup

```bash
docker compose -f platform_proofs/tools/iterative_sql_investigation/docker-compose.yml up -d
```

Environment (local disposable credentials):

| Variable | Default |
|----------|---------|
| `INTERGRAX_PP_SQL_INVESTIGATION_ADMIN_DSN` | `postgresql://proof_admin:proof_admin_local@localhost:5435/iterative_sql_proof` |
| `INTERGRAX_PP_SQL_INVESTIGATION_DSN` | `postgresql://proof_runtime:proof_runtime_local@localhost:5435/iterative_sql_proof` |

Load dataset (admin credentials):

```python
from intergrax.integrations.providers.relational_store.postgresql import create_postgresql_relational_store
from platform_proofs.tools.iterative_sql_investigation.dataset import bulk_load_parcel_events

store = create_postgresql_relational_store(
    dsn="postgresql://proof_admin:proof_admin_local@localhost:5435/iterative_sql_proof",
    tenant_schema="proof",
)
store.connect()
bulk_load_parcel_events(store, row_count=5000)
store.close()
```

## SQL safety boundary

1. **Proof-local validator** — single statement; SELECT/WITH only; rejects obvious mutating/admin keywords. Documented limits — not a general SQL parser.
2. **Database authorization (authoritative)** — runtime role has CONNECT + schema USAGE + SELECT only; no INSERT/UPDATE/DELETE/DDL; `statement_timeout` configured.
3. **Bounded results** — queries are wrapped with a subquery `LIMIT 201` so PostgreSQL returns at most 201 rows; visible cap is 200 with `truncated=True` when more exist.

## Limitations

- PP-3B does not run LLM investigation or register the proof in `scripts/proof/intergrax_proof_manifest.py`.
- SQL validator is proof-local and keyword-based; do not reuse as a universal SQL firewall.
- Dataset is synthetic logistics fiction — no production business rules.
- Subquery wrapping may not suit every advanced SQL dialect feature; proof scenarios use bounded SELECT/WITH only.

## What this does NOT prove (yet)

- Real model-driven iterative investigation
- InvestigationProof qualification
- EXECUTABLE / QUALIFIED coverage
- Public claim promotion in `docs/project/proofs/PROOFS.md`
- Product workflow consumption (LKW or otherwise)

## Educational explanation

Naive SQL investigation often stops at the first aggregate that “looks wrong.” This proof plants deliberate traps — volume confounds counts, confounders mimic causation, missing variables block conclusions — so a bounded tool loop must chain evidence-dependent queries instead of jumping to a single headline number. PP-3B establishes the deterministic substrate (data, database permissions, bounded read-only tool, runtime wiring) so PP-3C can test whether the TOOLS runtime actually behaves that way under a real model.

## NOT YET

- Real model execution
- InvestigationProof qualification
- Canonical manifest registration
- EXECUTABLE coverage promotion
- Public claim promotion
