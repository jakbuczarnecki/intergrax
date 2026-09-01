# `data.sql_analyst`

**Bundle:** `data` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

**Structured data Q&A**: discover relational schema, run read-oriented SQL queries, export results to shadow workspace. Use for analytics agents connected to Postgres/BigQuery/MotherDuck - not for arbitrary DDL/DML without host policy.

## How it works

1. `database.describe_schema` introspects tables/columns via `RelationalStore` integration.
2. `database.query` executes SQL (host policy should restrict to read-only connections).
3. `workspace.write_file` saves result CSV/JSON for downstream steps.
4. High risk tier - enforce `ToolAccessPolicy` and read-only DB roles at Tier-3.

## How to use

```python
from intergrax.skills.providers.data.manifests import DATA_SQL_ANALYST

AgentContract(id="analyst", skills=[DATA_SQL_ANALYST], risk_level=AgentRiskLevel.HIGH, ...)
```

Wire `relational_store` slug (`postgres`, `bigquery`, `motherduck`, etc.).

## What you get

| Benefit | Detail |
|---------|--------|
| **Schema-first analysis** | Describe before query reduces hallucinated columns |
| **Export path** | Results land in workspace for reports |
| **DB swap** | Change warehouse via integration profile only |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `database.query` | Execute SQL query |
| `database.describe_schema` | List tables/columns |
| `workspace.write_file` | Export query results |

## Related skills

- `research.citation_synthesis` - narrative report from exported data
- `workspace.authoring` - further edit exported files
