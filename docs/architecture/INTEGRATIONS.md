# Integrations

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 13–14  
**Audit instruction:** [`audit/INTEGRATIONS.md`](../audit/INTEGRATIONS.md)  
---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (INTEGRATIONS canon).

- **Implement / audit default:** manifest registration + IntegrationProfile + wiring. Catalog: [`arch/INTEGRATIONS_provider_catalog.md`](arch/INTEGRATIONS_provider_catalog.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/INTEGRATIONS.md`](../guides/audit_slices/INTEGRATIONS.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`arch/INTEGRATIONS_provider_catalog.md`](arch/INTEGRATIONS_provider_catalog.md) | provider catalog |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

**Related:**

| Document | Purpose |
|----------|---------|
| [`guides/SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) | Cross-layer MUST/MUST NOT rules — tool/integration invariants |
| [`architecture/TOOLS.md`](TOOLS.md) | Agent-facing semantic operations |
| [`architecture/SKILLS.md`](SKILLS.md) | Declarative tool composition packs |
| [`architecture/NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | Graph / routing / HITL / retries |
| [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | UnifiedTaskRunner, agent execution spine |
| [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) | Application roster, profile, intake wiring |
| [`guides/MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md) | Maturity claims vocabulary |

---

## Integration Layer Contract

The **Integration Library** (`intergrax/integrations/`) is Intergrax’s **Tier-0** modular catalog of **backend and vendor adapters** — databases, queues, search APIs, vector indexes, cloud platforms, collaboration tools, and local services.

Normative rules:

- Integrations are **Tier-0 adapters** for external systems, infrastructure backends, vendor APIs and local services.
- Integrations expose **backend-specific capabilities** to tools and platform services.
- Integrations are **not agent-facing**.
- Integrations are **not user-facing products**.
- Integrations are **not orchestration engines**.
- Integrations are **not memory engines**.
- Integrations are **not context engines**.
- Integrations are **not HITL systems**.
- Integrations **must not** own agent lifecycle or application lifecycle.

**Tools** are agent-facing semantic operations. **Applications** configure and wire integrations. **Agents never call integrations directly.**

---

## Responsibility boundary

| Concern | Owner |
|---|---|
| Vendor SDK / protocol details | Integration |
| Secrets and backend credentials access | Integration + policy/config |
| Agent-facing semantic operation | Tool |
| Tool invocation, policy, side-effect gateway | ToolRuntime |
| Agent decision | Tier-2 Agent |
| Graph / routing / HITL / retries | Nexus / runtime |
| Application roster/profile/intake | Tier-3 Application |
| LLM context assembly | ContextCompiler / ContextEngine |
| Memory state | Memory services |
| RAG retrieval orchestration | RAG service / catalog tools |
| Observability events | RuntimeEventBus / observability spine |

---

## Allowed integration responsibilities

Integrations **MAY**:

- wrap vendor SDKs or protocols,
- normalize request/response transport details,
- manage backend-specific authentication handoff,
- expose typed low-level operations to tools/platform services,
- translate backend errors into platform error types,
- support health checks,
- support capability discovery where appropriate,
- provide low-level clients for platform-owned services,
- handle retry only when it is backend/protocol-level and does not conflict with runtime retry policy.

---

## Disallowed integration responsibilities

Integrations **MUST NOT**:

- be invoked directly by agents,
- be invoked directly by Nexus graph nodes as side effects,
- decide which agent should run,
- manage global task lifecycle,
- own orchestration loops,
- own HITL approval,
- own business/product decisions,
- own prompt construction,
- own LLM calls unless the integration itself is explicitly an LLM provider adapter under the LLM adapter layer,
- write agent memory directly,
- emit private trace pipelines outside the observability spine,
- bypass ToolRuntime for agent-invokable side effects,
- implement product-specific workflows.

---

## Integration access paths

Correct access paths for integration use:

### Agent-invokable side effects

```text
Agent -> Tool / Skill -> ToolRuntime -> Integration
```

### Application intake / external surface

```text
External system -> Integration adapter -> Tier-3 intake surface -> UnifiedTaskRunner.run_task()
```

### Platform service backend

```text
Platform service -> Integration adapter
```

**Examples:**

- RAG service may use a vector database integration.
- Memory service may use a database integration.
- Observability sink may use OTEL/Sentry/log integration.
- ToolRuntime may use Slack/Google/GitHub integration through a tool.

---

## Slack / Teams / collaboration adapters

Intergrax supports Slack and Teams as **interaction surfaces** — examples of collaboration adapters, not the definition of the integration layer.

Slack and Teams adapters **may** normalize external messages into tasks and send approved outputs back, but they **must not** own runtime orchestration.

They may provide:

- task intake
- notifications
- approval requests
- progress updates
- final responses
- interactive buttons
- user context
- channel context

**Correct:**

```text
Slack event -> integration adapter -> application intake -> UnifiedTaskRunner.run_task()
    -> Nexus Runtime -> Agent execution -> Nexus final result -> integration adapter sends response
```

**Incorrect:**

```text
Slack bot -> direct agent call -> private memory -> direct tool execution
Slack bot contains orchestration logic
Slack bot directly manages agents
Slack bot stores global task state
```

---

## Cursor review checklist

Before adding or modifying an integration, Cursor must verify:

- [ ] Is this truly an integration, not a tool, skill, agent or application?
- [ ] Is the integration backend/vendor-facing rather than agent-facing?
- [ ] Are side effects exposed to agents only through ToolRuntime?
- [ ] Are secrets handled through approved config/policy mechanisms?
- [ ] Does the integration avoid orchestration, HITL and product workflow ownership?
- [ ] Are backend errors normalized?
- [ ] Is observability routed through the platform spine?
- [ ] Is retry limited to protocol/backend concerns and compatible with runtime retry?
- [ ] Is the integration wired through Tier-3 profile/config where required?
- [ ] Are maturity claims expressed through [`guides/MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md)?

---

## Adapter implementation checklist

Before implementing a new adapter, answer:

```text
1. What external system does it connect to?
2. What operations does it expose?
3. What permissions are required?
4. Is it read-only or write-capable?
5. What are risk levels?
6. What errors can happen?
7. What timeout/retry policy is needed?
8. What data should be logged?
9. What data must be protected?
10. Which tools or platform services may use it (not agents directly)?
```

Adapters should be generic and reusable.

---

## Catalog


**Last updated:** 2026-06-17 — **Full Harness LC** (re-validates M.7 P7 closeout); **185** slugs · M.7 P7 **Done** 18/18

The **Integration Library** (`intergrax/integrations/`) is Intergrax’s modular catalog of external systems — databases, queues, search APIs, vector indexes, cloud platforms, and collaboration tools. See **Integration Layer Contract** above for normative tier boundaries. Applications wire backends **by category** via `IntegrationProfile`; agents consume backends **through catalog tools**, not by importing vendor adapters.

**Related docs:**

| Document | Purpose |
|----------|---------|
| [`guides/SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) | Cross-layer tool/integration invariants |
| [`architecture/TOOLS.md`](TOOLS.md) | Agent-facing tools that compose these integrations |
| [`architecture/SKILLS.md`](SKILLS.md) | Declarative tool composition packs |
| [`architecture/NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | Graph / routing / HITL / retries |
| [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | UnifiedTaskRunner execution spine |
| [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) | Application profile and intake wiring |
| [intergrax_runtime_architecture.md](../intergrax_runtime_architecture.md) §7.1 | Architecture canon — tiers, contracts, registry rules |
| [plan/INTEGRATIONS.md](../plan/INTEGRATIONS.md) | Phase status, backlog, delivery workflow |
| [guides/AGENT_CREATION_GUIDE.md](../guides/AGENT_CREATION_GUIDE.md) Appendix E | How agents vs applications use integrations |
| [architecture/RAG.md](RAG.md) | RAG retrieval engine (consumes integration slugs) |
| Per-provider guides | `intergrax/integrations/providers/<category>/<slug>/USAGE.md` |
| [../infra/README.md](../infra/README.md) | **Local Docker infrastructure** — compose profiles, manage scripts |
| [../infra/PORTS.md](../infra/PORTS.md) | Host port matrix for integration tests |
| [guides/HARNESS_ENVIRONMENT.md](guides/HARNESS_ENVIRONMENT.md) | Lab harness stack, OTLP, verification |

---

## Harness lab stable stack (Phase S / T)

The **lab harness environment** treats these catalog slugs as **`stable`** (production-ready for the reference lab stack). Source of truth: `intergrax/integrations/registry/harness_lab_stack.py`.

| Slug | Category |
|------|----------|
| `sqlite` | relational_store |
| `postgresql` | relational_store (Tier-2 product apps) |
| `redis` | key_value_cache |
| `qdrant` | vector_store |
| `slack` | notification_channel + interaction_surface |
| `sentry` | observability_backend |
| `otel` | observability_backend |
| `lab_json` | interaction_surface |
| `log` | notification_channel |

```bash
uv run pytest tests/unit/integrations/test_harness_lab_stable_stack.py -m gate -q
```

Other slugs remain **`beta`** unless promoted explicitly. Do not mark all 185 providers stable in one release.

### M.6 P5 — Harness integration depth (Done — 33/34)

**Register:** [intergrax_runtime_architecture.md — M.6 P5](../plan/INTEGRATIONS.md#m6-p5--harness-integration-depth-done--3334) · Band **2ab**

| Wave | Focus | Status |
|------|--------|--------|
| H-INT-6 | Ops, metrics, multi-CI, local cloud | **Done** (10/10) |
| H-INT-7 | Eval observability, async bus, artifacts | **Done** (10/10) |
| H-INT-8 | Data plane lab (graph, document, logs, vectors) | **Done** (8/8) |
| H-INT-9 | P2 reserve | **Done** (5/6 — `trivy` deferred) |

**Delivered:** 25 harden (STABLE + `IntegrationHealthProbe`) · 8 greenfield (`_shared/p6`) · 4 Tier-3 presets · `HARNESS_M6_P5_PROBE_SLUGS` · debug API `GET /debug/integrations/health?stack=m6_p5`.

**Tier-3 presets (P5):** `harness_metrics_stack()`, `harness_eval_stack()`, `harness_async_stack()`, `harness_ci_stack()` — CLI: `intergrax integrations-pick harness_metrics|harness_eval|harness_async|harness_ci`.

**Deferred:** `trivy` — absorbed into **M.6 P6** [M-P6.1](../plan/INTEGRATIONS.md#m6-p6--master-register-32-slugs) (`security_scanner` / **M-P6-CAT.1**).

### M.6 P6 — Harness integration expansion (Done — 32/32)

**Register:** [intergrax_runtime_architecture.md — M.6 P6](../plan/INTEGRATIONS.md#m6-p6--harness-integration-expansion-planned) · Band **2ac** · Queue **[§6.1y](../plan/INTEGRATIONS.md#61y-harness-implementation-queue--integration-expansion-m6-p6-done)**

| Wave | Focus | Slugs | Status |
|------|--------|-------|--------|
| H-INT-10 | Security + secrets | `trivy`, `snyk`, `semgrep`, `infisical` | **Done** |
| H-INT-11 | Cloud sandbox | `e2b`, `modal`, `daytona` | **Done** |
| H-INT-12 | Identity / tenant IAM | `auth0`, `keycloak`, `workos` | **Done** |
| H-INT-13 | GitOps CI | `argocd`, `buildkite`, `jenkins` | **Done** |
| H-INT-14 | Speech catalog | `elevenlabs`, `deepgram` | **Done** |
| H-INT-15 | Enterprise ops | `newrelic`, `splunk`, `zendesk`, `statsig` | **Done** |
| H-INT-16 | Data / workflow | `prefect`, `airflow`, `typesense`, `neon`, `pulsar` | **Done** |
| H-INT-17 | Reserve | `algolia`, `confluent`, `backblaze_b2`, `triton`, `replicate`, `stripe`, `salesforce`, `hubspot` | **Done** |

**New categories (9):** `security_scanner`, `sandbox_host`, `identity_provider`, `speech_provider`, `workflow_orchestrator`, `vision_serving`, `ml_inference_host`, `billing_meter`, `crm`.

**Post-catalog wiring (M-P6-WIRE):** `wire_integration_tool_context()` resolves P6 slots into `ToolWiringContext`; `extend_tool_profile_for_integration()` auto-enables `security.scan`, `workflow.*`, and `sandbox.exec` when matching categories are configured. Speech catalog slugs bridge to Tier-0 speech tools via `IntegrationSpeechAdapter` + `SpeechProviderBackend` ([ADR-MOD-001](../adr/entries/2026-06-19/ADR-MOD-001.md) — slug identity; enum path removed under MOD-SPEECH-ARCH).

### Speech provider (`speech_provider`) — canonical tool path

Speech SaaS vendors follow the **open catalog** rules (§Open catalog below) — same as all 185+ slugs.

| Step | Mechanism |
|------|-----------|
| Register | `providers/speech_provider/<slug>/manifest.py` + `register_from_manifest()` or `IntegrationPlugin` |
| Contract | `SpeechProviderBackend` — `synthesize()`, `transcribe()`, `health()` |
| Tier-3 | `IntegrationProfile.speech_provider = <manifest \| plugin \| slug \| instance>` |
| Tools | `wire_integration_tool_context()` → `IntegrationSpeechAdapter(provider_slug=…)` → `speech.synthesize` / `speech.transcribe` |

**Do not** extend a platform `SpeechProvider` enum. **Do not** add vendor-specific branches in `wire_modality_extras()` when the integration slot is configured. Implementation paydown: [`plan/MODALITY.md`](../plan/MODALITY.md) MOD-SPEECH-ARCH.*.

**Delivered:** 32 STABLE slugs (`_shared/p7`) · 9 category contracts · 4 Tier-3 presets · `HARNESS_M6_P6_PROBE_SLUGS` · debug API `GET /debug/integrations/health?stack=m6_p6`.

**Tier-3 presets (P6):** `harness_security_stack()`, `harness_sandbox_stack()`, `harness_identity_stack()`, `harness_gitops_stack()` — CLI: `intergrax integrations-pick harness_security|harness_sandbox|harness_identity|harness_gitops`.

### M.7 P7 — Agent-developer integration expansion (Done — 18/18)

**Register:** [plan/INTEGRATIONS.md — M.7 P7](../plan/INTEGRATIONS.md#m7-p7--agent-developer-integration-expansion-done--1818) · Band **2ad**

| Wave | Focus | Slugs | Status |
|------|--------|-------|--------|
| H-INT-P7-1 | Research + RAG | `perplexity`, `arxiv`, `semantic_scholar`, `llamaparse`, `lancedb` | **Done** |
| H-INT-P7-2 | Interaction + browser + storage | `telegram`, `browserbase`, `google_drive`, `apify` | **Done** |
| H-INT-P7-3 | Workflow + wiki + identity + cache | `n8n`, `wikipedia`, `clerk`, `upstash_redis`, `upstash_qstash` | **Done** |
| H-INT-P7-4 | Data warehouse | `okta`, `bigquery`, `motherduck`, `airbyte` | **Done** |

**Delivered:** 18 STABLE slugs (`_shared/p8`) · 3 Tier-3 agent presets · `HARNESS_M7_P7_PROBE_SLUGS` · auto-wiring `search_provider` / `document_parser` / `vector_store` → catalog tools.

**Tier-3 presets (P7):** `research_web_stack()`, `document_ingest_stack()`, `chat_bot_stack()` — CLI: `intergrax integrations-pick research_web|document_ingest|chat_bot`.

**Catalog:** **185** slugs in `layout.py` (**12** core / **185** full preset).

---

## Local infrastructure (Docker)

Run backing services locally before integration tests or lab hosts. Unified stack: `infra/integration/` with **compose profiles** (`core`, `queue`, `rag`, `data`, `secrets`, `observability`, `cloud`, `heavy`, `p6`).

```bash
cd infra/integration && ./manage.sh start          # default profiles
cd infra/integration && ./manage.sh start rag      # vectors + neo4j + ollama + docling
cd infra/integration && ./manage.sh start p6       # keycloak + typesense + airflow + core (PostgreSQL)
cd infra/integration && ./manage.sh start all      # full stack
```

See [infra/PORTS.md](../infra/PORTS.md) for host ports (e.g. Redis `6379`, Qdrant `6333`, Neo4j Bolt `7687`, Weaviate `8080`, MinIO `9000`, Vault `8200`, ClickHouse HTTP `8123` / native `9002`).

**SaaS-only slugs** (no local container — use mocks or API keys): `slack`, `jira`, `confluence`, `google_cse`, `pinecone`, `cohere_rerank`, `sentry` (cloud), most `observability_backend` HTTP proxies unless self-hosted image is listed in infra.

---

## Provider layout (by category)

Integrations are grouped under **contract category** folders — the same grouping used when generating P2/P3 provider stubs:

```text
intergrax/integrations/providers/
├── layout.py                 # slug → category map
├── relational_store/         # sqlite, postgresql, mysql, …
├── document_store/           # mongodb, cassandra, dynamodb
├── key_value_cache/          # redis, memcached, elasticache
├── message_bus/              # kafka, sqs, pubsub, …
├── object_storage/           # s3, azure_blob, gcs
├── vector_store/             # pinecone, qdrant, chroma, weaviate, milvus, inmemory, vespa
├── search_provider/          # google_cse, bing, reddit, google_places, brave, serpapi, tavily, exa
├── notification_channel/     # slack, teams, discord, twilio, pagerduty, opsgenie, …
├── interaction_surface/      # lab_json, slash_command (slack/teams also register here)
├── collaboration_suite/      # ms365_graph, google_workspace
├── issue_tracker/            # jira, github, linear, azure_devops, gitlab
├── wiki_knowledge/           # confluence, notion, sharepoint
├── observability_backend/    # prometheus, elasticsearch, otel, langfuse, datadog, clickhouse, sentry, langsmith, …
├── document_parser/          # docling, pymupdf, unstructured, python_docx, openpyxl, whisper, yt_dlp
├── rerank_provider/          # cohere_rerank, jina_rerank
├── browser_automation/       # playwright, firecrawl, selenium
├── secrets_store/            # vault
├── graph_store/              # neo4j
└── cloud_platform/           # aws, azure, gcp
```

**Import path:** `from intergrax.integrations.providers.object_storage.s3.bundle import create_s3_object_storage`

Catalog identity is the string **slug** (`"s3"`, `"postgresql"`) registered at runtime — not a central enum.

---

## Open catalog (no slug enum)

| Mechanism | Role |
|-----------|------|
| `providers/<category>/<slug>/manifest.py` | `MANIFEST = IntegrationManifest(slug=…)` — canonical metadata per provider |
| `register_from_manifest(MANIFEST, factory)` | Registers in runtime catalog (`registry/catalog.py`) |
| `IntegrationProfile` | Declares slot: manifest, plugin class, slug `str`, or pre-built instance |
| `profile.resolve(IntegrationCategory.…)` | Instantiates via registered factory |
| `catalog_manifests.py` | Lightweight **preset** copies for lab/product profiles only (not exhaustive) |
| `IntegrationPlugin` | External packages: `integration_manifest()` + `create_integration()` |
| `bootstrap_catalogs()` | Unified Tier-3 bootstrap; `integration_preset="core"` or `"full"` |

Third-party integrations **must not** extend a core enum. Register a plugin or call `register_from_manifest` from application startup.

**Shipped vs plugin class:** ~167 providers register via `register_from_manifest(MANIFEST, create_*)`. External pip packages should implement `IntegrationPlugin` (`integration_manifest()` + `create_integration()`). `SqliteIntegrationPlugin` in `providers/relational_store/sqlite/plugin.py` documents the class-based pattern; shipped `register.py` keeps the manifest path for bootstrap performance.

Tier-3 hosts should call `bootstrap_application_integration_catalog()` (not bare `register_default_integrations()`).

### Named integration presets (Phase DX-4.3)

Typed factories in `intergrax.integrations.registry.presets` — use in `ApplicationManifest` / `host/environment_profile.py`:

| Preset function | Returns | Typical use |
|-----------------|---------|-------------|
| `lab_stack(enable_otel=True)` | `IntegrationProfile.lab_harness_preset` | Default lab / scaffold hosts |
| `legal_stack()` | `IntegrationProfile.legal_product()` | Legal product relational + vector + OTEL observability backend |
| `research_stack()` | `IntegrationProfile.research_product()` | Research product search + vector |
| `data_stack(enable_redis=True, enable_qdrant=False)` | Lab harness + optional redis/qdrant | Data-heavy experiments |
| `observability_stack(enable_otel=True, enable_grafana_stack=False)` | Lab harness OTEL-first; optional Grafana/Loki/Tempo triad | Trace/metrics focus |
| `harness_production_stack(secrets_slug="doppler", enable_grafana_stack=True)` | PostgreSQL + pgvector + secrets + Grafana stack + Unleash + GitHub Actions | Harness production Tier-3 (no business agents) |

CLI fragment helper: `uv run intergrax integrations pick postgres` (presets: `lab`, `legal`, `research`, `data`, `observability`, `harness_production`). See `intergrax/cli/integrations_pick.py`.

See [guides/EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md), `intergrax/integrations/examples/custom_memory_kv/`, and `tests/unit/integrations/test_external_plugin.py`.

Scaffold a new provider tree: `python -m intergrax.scaffold new-integration <slug> --category <category>`.

---

## Design principles

| Principle | What it means |
|-----------|---------------|
| **Universal contracts** | Each category (`relational_store`, `vector_store`, `message_bus`, …) defines a small Protocol. Providers implement the contract; agent logic depends on the contract only. |
| **Modular providers** | One slug = one package under `providers/<category>/<slug>/` (category = contract name). Swap Redis for ElastiCache, SQLite for PostgreSQL, or Chroma for Pinecone by changing `IntegrationProfile` — no agent refactor. |
| **Environment portability** | Tier-3 applications compose integrations at startup (`IntegrationProfile`, env vars). The same Tier-2 agent runs against lab defaults (`sqlite`, `log`, `lab_json`) or production stacks (`postgresql`, `slack`, `s3`, `qdrant`). |
| **Single entry for SDKs** | Vendor SDKs (boto3, PyMongo, chromadb, redis, …) are imported only in boundary modules: `opens.py`, `rag_store.py`, `web_client.py`, `client.py`, and `_shared/p2|p3|p4/factories.py`. CI enforces this via `scripts/check_integration_vendor_imports.py`. Tier-2 agents must **not** import provider slugs or vendor libraries. |
| **Catalog registration** | `register_default_integrations(preset="full")` or `preset="core"` (lab). Resolution: explicit slug → profile field → env → cloud defaults. |

---

## How wiring works

```text
Tier-3 application (integration_wiring.py)
        │
        ▼
IntegrationProfile  ──►  IntegrationRegistry.resolve(category)
        │                        │
        │                        ▼
        │                 providers/<slug>/bundle.py
        │                        │
        ▼                        ▼
   env + options            category contract instance
                                   │
                                   ▼
                         passed into runtime / RAG / tools
```

Agents consume integrations **through catalog tools** ([architecture/TOOLS.md](architecture/TOOLS.md)), not by importing provider adapters. Tier-3 may also pass resolved contracts into `ToolWiringContext` for tool handlers.

**Example — declarative profile:**

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.catalog_manifests import POSTGRESQL, QDRANT
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(
    relational_store=POSTGRESQL,
    vector_store=QDRANT,
    object_storage="s3",
    notification_channel="slack",
    options={
        "s3": {"bucket": "intergrax-artifacts", "prefix": "tenant-a"},
    },
)

store = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

**Example — lab defaults (no external vendors):**

```python
profile = IntegrationProfile.lab()
# relational_store → sqlite, notification_channel → log, interaction_surface → lab_json
```

**Example — product profile with SQLite observability fallback:**

Product profiles such as `IntegrationProfile.legal_product()` may omit `relational_store`. Tier-3 factories pass the profile to `wire_nexus_observability()`; when SQLite is not declared on the profile, trace and runtime-event stores fall back to default `build/` SQLite paths (same as pre-profile wiring).

```python
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.integrations.registry.profile import IntegrationProfile

observability = wire_nexus_observability(
    integration_profile=IntegrationProfile.legal_product(),
)
```

Explicit SQLite bundle (lab / tests):

```python
from intergrax.runtime.persistence.integration_profile_wiring import open_trace_store_from_profile

profile = IntegrationProfile(
    relational_store="sqlite",
    options={"sqlite": {"data_dir": "build/lab"}},
)
trace_store = open_trace_store_from_profile(profile)
```

---

## Category contracts

| Category | Contract | Typical use |
|----------|----------|-------------|
| `relational_store` | `RelationalStore` | SQL persistence, analytics warehouses |
| `document_store` | `DocumentStore` | Flexible JSON / wide-column documents |
| `key_value_cache` | `KeyValueCache` | Idempotency, rate limits, locks — **not** session or user LTM memory (see Phase MEM / `AGENT_CREATION_GUIDE` Appendix G) |
| `message_bus` | `MessageBus` | Async task queues, worker transport |
| `object_storage` | `ObjectStorage` | Artifacts, exports, large file handoff |
| `vector_store` | `VectorStore` | RAG embedding indexes |
| `search_provider` | `SearchProvider` | Web / API research |
| `notification_channel` | `NotificationChannel` | Outbound alerts (HITL, progress) |
| `interaction_surface` | `InteractionSurface` | Inbound webhooks / chat intake |
| `collaboration_suite` | `CollaborationSuite` | Mail, calendar, directory |
| `issue_tracker` | `IssueTracker` | Issues, comments, search |
| `wiki_knowledge` | `WikiKnowledge` | Runbooks, internal docs |
| `observability_backend` | `ObservabilityBackend` | Metrics, log search, error tracking (Sentry) |
| `browser_automation` | `BrowserAutomation` | Dynamic web pages (JS-heavy sites) |
| `secrets_store` | `SecretsStore` | Tenant API keys, credentials (Vault, …) |
| `graph_store` | `GraphStore` | Agent memory, tool dependency graphs |
| `document_parser` | `DocumentParser` | Document/media parsing (Docling, PyMuPDF, Unstructured, python-docx, openpyxl, whisper, yt_dlp) |
| `rerank_provider` | `RerankProvider` | Vendor reranking APIs (cohere_rerank, jina_rerank) — consumed by RAG `rerankers/` |
| `security_scanner` | `SecurityScannerBackend` | SAST/dependency scans (trivy, snyk, semgrep) — CI and release gates |
| `llm_guardrail` | `LlmGuardrailBackend` | LLM I/O safety scanners (LLM Guard, Guardrails AI, NeMo, OpenGuardrails) — §47 |
| `cloud_platform` | `CloudPlatform` | Multi-service auth + category defaults |

Contract modules: `intergrax/integrations/contracts/`.

---

## Cloud platform facades

Platform providers resolve **default slugs** for infrastructure categories when an application sets `cloud_platform` and leaves category fields empty.

| Platform | Auth | Default `object_storage` | Default `message_bus` | Default `document_store` | Default `key_value_cache` |
|----------|------|--------------------------|------------------------|--------------------------|---------------------------|
| `aws` | IAM keys, profile, STS assume-role | `s3` | `sqs` | `dynamodb` | `elasticache` |
| `azure` | Managed identity, service principal | `azure_blob` | `service_bus` | — | — |
| `gcp` | ADC, service account JSON | `gcs` | `pubsub` | — | — |

Service-level slugs (`s3`, `azure_blob`, `gcs`, …) remain available for explicit or multi-cloud setups.

---
