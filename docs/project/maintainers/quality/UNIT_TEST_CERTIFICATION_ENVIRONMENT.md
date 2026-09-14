# Unit test certification environment (HARDENING-9.1)

## Purpose

Provide one **minimal, deterministic** optional-extra profile so maintainers and nightly CI can run the full `tests/unit` tree without `uv sync --all-extras` and without ad-hoc provider installs.

## Test scope

| Surface | Contract |
| -------- | -------- |
| **Collection** | All modules under `tests/unit` must import cleanly (`pytest tests/unit --collect-only`). |
| **Quality gate execution** | Deterministic regression subset: `pytest tests/unit -m "gate and not no_ci"`. |
| **Excluded from gate marker** | Tests marked `no_ci`, `network`, `qualification`, `external_provider`, `docker`, `slow`, `external_proof`, etc. remain in the tree for local/qualification runs but are not part of the default gate selection. |

Physical layout under `tests/unit` is the collection contract; gate execution follows marker policy (see `pyproject.toml` markers and `.github/workflows/unit-tests.yml`).

## Canonical profile

**Extra name:** `dev-unit-cert`

**Dependency ownership** (each line mirrors an existing optional extra; do not duplicate ownership in core `[project].dependencies`):

| Package | Owning extra | Unit test families |
| -------- | ------------- | ------------------- |
| `fastmcp` | `mcp` | FastAPI MCP coupling (`tests/unit/applications/test_fastapi_mcp.py`, …) |
| `langchain-core` | `dev-ci-rag` / LangChain compat | compat, LLM Ollama bridge, RAG indexing, Nexus tool catalog |
| `qdrant-client` | `dev-ci-rag` / `vector-qdrant` | RAG vectorstore contract tests |
| `anthropic`, `tiktoken` | `llm-anthropic` | LLM adapter conformance |
| `pgvector` | `integrations-pgvector` | VPI platform proof unit adapters (`psycopg` comes from `dev` / `dev-ci`) |
| `torch`, `sentence-transformers`, `transformers` | `rag-local-embeddings` | HF embedding provider + VPI data-pack proof helpers |

## Install

**Local maintainer (full integration dev + unit cert):**

```powershell
uv sync --extra dev --extra dev-unit-cert
```

**CI nightly full gate (minimal CI runner + unit cert):**

```powershell
uv sync --extra dev-ci --extra dev-unit-cert --frozen
```

Default `uv` dependency groups (`test`, `quality`) remain enabled for `uv run pytest`.

## Verify collection

```powershell
uv run pytest tests/unit --collect-only -q
```

Expect **0 collection errors**.

## Verify quality gate

```powershell
uv run pytest tests/unit -m "gate and not no_ci" -q --tb=line
```

## Anti-pattern

Do **not** use:

```text
uv sync --all-extras
```

That installs unrelated heavy integrations, increases conflict risk, and violates least-dependency governance.

## Adding new optional dependencies

1. Classify the failing import: unit import graph vs qualification-only vs network/Docker/credentials.
2. If the test belongs under `tests/unit` and imports at module level, add the package to the **owning optional extra** first, then list it in `dev-unit-cert` only when that import is part of the certified unit tree.
3. If the test is external-only, prefer markers + dedicated CI job over expanding `dev-unit-cert`.
4. Update this document and `tests/unit/runtime/architecture/test_unit_certification_environment_contract.py`.

## Related

- [REPOSITORY_QUALITY_GATE_HARDENING.md](./REPOSITORY_QUALITY_GATE_HARDENING.md) — HARDENING-9 collection gate
- `.github/workflows/unit-tests.yml` — `gate-tests` job sync profile
- `.github/workflows/rag-guard.yml` — `dev-ci-rag` for RAG-focused CI (subset)
