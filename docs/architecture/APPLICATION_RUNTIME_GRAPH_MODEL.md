# Application Runtime Graph Model

**Status:** Canonical architecture (Tier-3 runtime isolation)  
**Related:** [`APPLICATION_DEPENDENCY_MODEL.md`](APPLICATION_DEPENDENCY_MODEL.md)  
**Last updated:** 2026-07-23

---

## 1. Outcome

```text
Tier-3 application
├── may depend on Tier-1 platform capabilities (Intergrax-ai + extras)
├── may depend on reusable Tier-2 agents (direct declarations only)
├── owns application-specific configuration and dependencies
└── must not depend on another Tier-3 application

Tier-2 agent
├── may depend on Tier-1 platform contracts and capabilities
├── may depend on other Tier-2 agents (explicit agents/<agent>/pyproject.toml)
├── must remain independent of every Tier-3 application
└── may be reused by any number of Tier-3 applications

Tier-1 platform
├── must remain independent of Tier-2 agents
└── must remain independent of Tier-3 applications
```

Canonical dependency direction:

```text
Tier-3 → Tier-2 → Tier-2 → … → Tier-1
Tier-3 → Tier-1
```

Tier-2 agents may depend on other Tier-2 agents. The application runtime graph
resolves the full acyclic transitive closure. Tier-3 applications declare only
direct Tier-2 dependencies; transitive Tier-2 dependencies are declared by the
importing agent. Agent cycles and agent-to-application dependencies fail closed
(`AGENT_DEPENDENCY_CYCLE`, `AGENT_TIER_VIOLATION`).

Each application image contains the **minimal transitive runtime graph** —

```text
application + direct agents + transitive agents + platform
+ direct/transitive third-party runtime deps (via uv.lock)
```

— and nothing outside that graph. Unresolved local workspace packages fail closed
(`RUNTIME_GRAPH_UNRESOLVED`).

---

## 2. Declaration source of truth

| Layer | Metadata | Declares |
|-------|----------|----------|
| Platform | root `pyproject.toml` | Intergrax-ai + capability extras |
| Agent | `agents/<agent>/pyproject.toml` | `intergrax-*-agent` + Intergrax-ai (+ optional Tier-2 peers + agent-only third-party) |
| Application | `applications/<app>/pyproject.toml` | Intergrax extras + **direct** agent packages + app-only deps |

One shared `uv.lock` remains canonical for the monorepo phase.

**Third-party semantics:** the runtime-graph manifest field
`direct_third_party_distributions` lists only third-party dependencies declared
by the Tier-3 application. Third-party packages contributed by selected agents
are resolved through those agent projects and `uv.lock`. The graph resolver does
**not** manually compute the full external transitive closure — `uv.lock` remains
the source of truth for that closure.

Agents are **not** owned by applications. Do not invent names such as “LKW agents”.

---

## 3. Build context and images

Canonical build:

```bash
uv run python scripts/build/build_application_image.py \
  --application local_workspace_application \
  --tag intergrax/local-workspace:local
```

The builder:

1. resolves the application runtime graph from project metadata (direct + transitive Tier-2 closure);
2. materializes a minimal context (no other Tier-3 trees, no unreachable agents);
3. writes `.intergrax-runtime-graph.json` (schema version 2: direct / transitive / all agent fields);
4. runs Docker BuildKit against that context;
5. removes the temporary context unless `--keep-context` / `--context-dir` is set.

Compose uses `applications/<app>/docker/runtime-context/` prepared by the same builder.
Do not build with repository-root context + `COPY applications/`.

Per-image environment: `/app/.venv` inside the container filesystem.
Local workspace execution may still use the shared root `.venv` unless
`UV_PROJECT_ENVIRONMENT` is set. These are different isolation levels.

---

## 4. Inventory (discovery snapshot)

### Tier-3 applications (with `pyproject.toml`)

| Application | Selected agents (declared) | Notable platform extras |
|-------------|---------------------------|-------------------------|
| `attestation_demo` | `boundary_demo` | — |
| `dispute_sim_application` | dispute_* (4) | — |
| `governed_contractor_application` | `external_contractor_adapter` | — |
| `intergrax_assistant_application` | echo, intergrax_assistant, legal, research | — |
| `lab_application` | echo, lab, problem_radar, research, signoff_probe | — |
| `legal_application` | legal | — |
| `local_workspace_application` | local_indexer, local_search, local_synthesizer | slack, mongodb, sentry, kafka |
| `poc_template_application` | echo | — |
| `research_application` | research | — |

### Tier-2 agents

All reusable packages under `agents/*` own `pyproject.toml` workspace members.
Import paths (`from local_search...`) are preserved; distribution names use
`intergrax-<slug>-agent` (special case: `intergrax_assistant` → `intergrax-assistant-agent`).

### Forbidden edges (package graph + AST)

```text
platform → agent
platform → application
agent → application          # package graph: AGENT_TIER_VIOLATION
application A → application B
Tier-2 dependency cycle      # package graph: AGENT_DEPENDENCY_CYCLE
```

Agent → agent edges are allowed when declared in the importing agent's
`agents/<agent>/pyproject.toml`. The complete graph must remain acyclic.

Dynamic `importlib` loading in CLI/demo glue is allowed only where static imports
would violate the tier boundary; agents and applications remain optional at import time.

### Known platform-base dependencies not yet capability-isolated

`Intergrax-ai` still pulls a large mandatory base (LLM/RAG/parser stack, including
packages such as `torch`, `transformers`, document parsers, etc.).
Application-selected extras isolate **provider SDKs** (Slack, MongoDB, Sentry, Kafka, …)
where platform packaging permits. Do not claim those base packages are
application-selected until the platform package itself is made optional/import-safe.

---

## 5. Isolation proofs

| Proof | Mechanism |
|-------|-----------|
| Declaration | application / agent `pyproject.toml` |
| Dependency graph | `uv export --project … --frozen --no-dev --no-emit-workspace` |
| Build context | materialized context + `.intergrax-runtime-graph.json` |
| Image filesystem | no other `applications/<other>/` trees; no undeclared agents |
| Image imports | selected SDKs importable; unselected provider SDKs absent where packaging allows |
| Tier boundaries | `tests/unit/architecture/test_tier_dependency_boundaries.py` |

---

## 6. Security

Build contexts and images must never include `.env`, Slack tokens, API keys,
Git metadata, or developer home paths. The materializer fails closed on
secret-like payloads (`xapp-`, `xoxb-`, Bearer tokens, private keys, assigned secrets).
