# Integrations — Implementation Plan

**Architecture (1:1):** [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**RAG engine (layer 14):** [`architecture/RAG.md`](../../architecture/RAG.md) ↔ [`plan/RAG.md`](RAG.md) — M-RAG, M-RAG-DEPTH, **M-RAG-GRAPH** (GraphRAG platform). This plan covers **integration catalog** slugs only; RAG adapters for `graph_store` are owned by M-RAG.38–M-RAG.51 in [`plan/RAG.md`](RAG.md).

**Last updated:** 2026-08-02 — **GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1** architecture frozen (`READY_FOR_REVIEW`); Google knowledge runtime **PLANNED** after complete Slack vertical — see [`plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md) Phase 10.

---

## Active cross-feature work — LangChain Independence

### LCI-2A — Document parser boundary migration

| Field | Value |
|-------|-------|
| **Status** | READY_FOR_REVIEW |
| **Integration contract reused** | `ParsedDocumentFragment` |
| **Provider internals changed** | no |
| **Provider optionalization** | deferred to LCI-5A / LCI-5C |

### LCI-3D — Native vector-store provider adapters

| Field | Value |
|-------|-------|
| **Status** | READY_FOR_REVIEW |
| **Provider boundary** | `VectorStoreRecord` → provider-native SDK payload → `VectorStoreHit` |
| **Isolation** | Tenant, namespace and workspace routing are system-owned at every provider boundary |
| **Scope** | Vector-store provider implementations and shared integration bridges only |
| **Out of scope** | Retrieval, reranking, Graph RAG, application tenancy and SDK upgrades |

Provider runtime matrix:

| Provider | `add_records` | `add_documents` | scoped signatures | forwarding | delete/count |
|----------|---------------|------------------|-------------------|------------|-------------|
| Vespa | native | absent | keyword-only | verified | B query/count; C unknown-ID delete |
| LanceDB | native | absent | keyword-only | verified | B metadata-scoped fallback |
| Typesense | native | absent | keyword-only | verified | C unsupported, fail closed |

### LCI-4B — Native rerank provider boundary

| Field | Value |
|-------|-------|
| **Status** | READY_FOR_REVIEW |
| **Provider boundary** | `Sequence[RerankerCandidate]` → vendor SDK text/index payload → `Sequence[RerankerResult]` |
| **Invariants** | Native `KnowledgeDocument`, identity, scope, provenance, user metadata and vector identity remain authoritative |
| **Providers** | Cohere Rerank and Jina Rerank |
| **Next** | LCI-4C Graph RAG |

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (INTEGRATIONS plan).

- **Implement / audit default:** Phase INT / H-INT hub queues · §6.1 open P0/P1 · M.6 expansion registers — satellite on demand
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/INTEGRATIONS.md`](../../technical/guides/audit_slices/INTEGRATIONS.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Phase H-INT-GRAPH — graph_store expansion (Planned)

**Purpose:** New `graph_store` vendor slugs required before RAG adapters M-RAG.49–M-RAG.51.  
**RAG coordination:** [`plan/RAG.md`](RAG.md) Wave G4 · GAP-RAG-33.

| ID | Slug | Category | Priority | Status | RAG deliverable | Notes |
|----|------|----------|----------|--------|-----------------|-------|
| H-INT-GRAPH-1 | `neptune` | graph_store | **P3** | **Done** | M-RAG.49 | AWS Neptune — OpenCypher HTTP bridge |
| H-INT-GRAPH-2 | `orientdb` | graph_store | **P3** | **Done** | M-RAG.50 | OrientDB OpenCypher HTTP bridge |
| H-INT-GRAPH-3 | `arangodb` | graph_store | **P3** | **Done** | M-RAG.51 | ArangoDB AQL HTTP bridge |

**Per-slug checklist:** contract gate → `providers/graph_store/<slug>` → health probe → bootstrap register → RAG `RagGraphStoreBackend` adapter (M-RAG.38 registry) → gate green.

**Explicitly out of scope:** Microsoft GraphRAG library vendoring (harness-native indexer M-RAG.47); TigerGraph / JanusGraph unless product reprioritizes.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/INTEGRATIONS_appendices.md`](plan/INTEGRATIONS_appendices.md) | appendices |
| [`plan/INTEGRATIONS_audit_history.md`](plan/INTEGRATIONS_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase INTEGRATIONS-2A — provider category contracts (Done)

**Purpose:** Category-specific base contracts aligned with `layout.py` `SLUG_CATEGORY` taxonomy before concrete provider migration.  
**Architecture:** [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md#provider-category-contract-layer-integrations-2a)
**Detail plan:** [`PROVIDER_CATEGORY_CONTRACTS.md`](PROVIDER_CATEGORY_CONTRACTS.md)

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **INTEGRATIONS-2A** | Code | P1 | **Done** | `intergrax/runtime/integrations/categories` + `PROVIDER_CATEGORY_CONTRACT_REGISTRY` | All 31 `SLUG_CATEGORY` folders covered; `observability_backend` aliases **`ObservabilityVendorIntegrationContract`**; `PlatformIntegrationKind` extended; focused tests green |

**INTEGRATIONS-2A status (2026-06-28):**

- Category-specific contracts for all provider folders in `layout.py`
- `observability_backend` → **`ObservabilityVendorIntegrationContract`** (no duplicate contract)
- Concrete provider migration **deferred** (INTEGRATIONS-2B)
- No LKW change; no registry/bootstrap wiring; no vendor SDK imports

---

## Phase INTEGRATIONS-2B — observability provider contract migration pilot (Done — pattern hardened)

**Purpose:** Adapt existing `observability_backend` provider packages to category contracts without duplicating providers.  
**Architecture:** [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md#provider-package-pattern-integrations-2b-follow-up)
**Pattern:** existing provider package + new contract-based integration class; legacy query facade remains backward-compatible.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **INTEGRATIONS-2B-LANGFUSE** | Code | P1 | **Done (reference pilot)** | Langfuse `LangfuseObservabilityIntegration` in existing provider package | Subclasses **`ObservabilityVendorIntegrationContract`**; sanitized envelope only; injectable transport; old **`ObservabilityBackend`** facade unchanged; focused tests green |
| **INTEGRATIONS-2B-FOLLOWUP** | Code | P1 | **Done** | Provider package pattern + scaffold hardening | Canonical layout documented; Langfuse conforms; scaffold idempotent; `enabled=True` without transport fails early; batch migration **deferred** |

**INTEGRATIONS-2B status (2026-06-28):**

- **Langfuse** accepted as reference pilot **after** scaffold/pattern hardening (INTEGRATIONS-2B-FOLLOWUP)
- **`LangfuseObservabilityIntegration`** under `intergrax/integrations/providers/observability_backend/langfuse`
- Legacy **`create_langfuse_observability_backend`** / **`register_langfuse_integration`** remain backward-compatible
- Maintenance shell generators (`wire_p2` through `wire_p7`) preserve contract-aware packages when `integration.py` exists
- Scaffold blockers for full provider migration **removed** (INTEGRATIONS-SCAFFOLD-P5-P7-CONTRACT-AWARE); full migration still **deferred** until a small multi-wave provider migration is validated
- Arize, Phoenix, Elasticsearch, and remaining observability_backend slugs **deferred** until batch migration wave
- No LKW change; no global bootstrap registration; no vendor SDK imports

---

## Phase INTEGRATIONS-2C — observability_backend provider contract migration (Done)

**Purpose:** Migrate all existing `observability_backend` provider packages to **`ObservabilityVendorIntegrationContract`** using the Langfuse reference pattern.  
**Architecture:** [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md#provider-package-pattern-integrations-2b-follow-up)

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **INTEGRATIONS-2C-WAVE1** | Code | P1 | **Done** | LLM/agent observability providers | Arize, Phoenix, Langsmith, Helicone, Braintrust, W&B — `integration.py` + contract factory; legacy facade unchanged |
| **INTEGRATIONS-2C-WAVE2** | Code | P1 | **Done** | Telemetry/APM/log vendors | Datadog, Sentry, SigNoz, Honeycomb, New Relic, Splunk, PostHog — same pattern |
| **INTEGRATIONS-2C-WAVE3** | Code | P1 | **Done** | OpenTelemetry/storage/query backends | Prometheus, Elasticsearch, OpenSearch, OTel, OpenTelemetry Collector, Grafana, Loki, Tempo, InfluxDB, ClickHouse, MLflow — same pattern |
| **INTEGRATIONS-2C-TESTS** | Test | P1 | **Done** | Parametrized conformance suite | `test_observability_provider_contract_migration.py` — shared tests for all migrated slugs |

**INTEGRATIONS-2C status (2026-06-28):**

- **All 26** `observability_backend` slugs in `layout.py` migrated (Langfuse pilot + 25 batch waves)
- Each migrated provider: `integration.py` (contract class), `create_<slug>_observability_integration` in `bundle.py`, lazy `__init__.py` exports, USAGE.md contract section
- Legacy **`create_<slug>_observability_backend`** / **`register_<slug>_integration`** remain backward-compatible; **`register.py`** still registers legacy facade only
- **`enabled=True`** without transport raises **`IntegrationConfigurationError`**; no vendor SDK imports in `integration.py`; no network I/O in tests
- **Registry v2 / contract registry wiring remains deferred**
- **Deferred providers:** none — full `observability_backend` category complete
- **OBS-EXPORT-5 linkage:** contract adapters complete; production vendor transports and operator wiring pending (not production export done)
- No LKW change; no global bootstrap registration

**Migrated slugs (26):** `langfuse`, `arize`, `phoenix`, `langsmith`, `helicone`, `braintrust`, `wandb`, `datadog`, `sentry`, `signoz`, `honeycomb`, `newrelic`, `splunk`, `posthog`, `prometheus`, `elasticsearch`, `opensearch`, `otel`, `opentelemetry_collector`, `grafana`, `loki`, `tempo`, `influxdb`, `clickhouse`, `mlflow`

---

## Phase INTEGRATIONS-2D — remaining provider category contract migration (Done)

**Purpose:** Migrate all non-`observability_backend` provider slugs in `SLUG_CATEGORY` to category-specific **`PlatformIntegrationContract`** subclasses using the Langfuse package pattern (not observability semantics).

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **INTEGRATIONS-2D-WAVE1** | Code | P1 | **Done** | Retrieval/data access | `vector_store`, `search_provider`, `document_parser`, `rerank_provider`, `wiki_knowledge` |
| **INTEGRATIONS-2D-WAVE2** | Code | P1 | **Done** | Storage/databases | `relational_store`, `document_store`, `key_value_cache`, `graph_store`, `object_storage` |
| **INTEGRATIONS-2D-WAVE3** | Code | P1 | **Done** | Communication/workflow | `message_bus`, `notification_channel`, `collaboration_suite`, `issue_tracker`, `browser_automation` |
| **CONVERSATION-CHANNEL-1** | Code | P1 | **Done** | Conversational category | `conversation_channel` contract + seven contract-defined providers (runtime unbound); multi-category with notification for slack/teams/discord/telegram |
| **INTEGRATIONS-2D-WAVE4** | Code | P1 | **Done** | Platform/security/ops | `cloud_platform`, `secrets_store`, `feature_flag`, `ci_cd`, `security_scanner`, `sandbox_host`, `identity_provider`, `workflow_orchestrator` |
| **INTEGRATIONS-2D-WAVE5** | Code | P1 | **Done** | AI/service/business | `speech_provider`, `vision_serving`, `ml_inference_host`, `billing_meter`, `crm` |
| **INTEGRATIONS-2D-TESTS** | Test | P1 | **Done** | Parametrized conformance suite | `test_provider_category_contract_migration.py` — completeness derived from `SLUG_CATEGORY` |

**INTEGRATIONS-2D status (2026-06-28):**

- **160** non-observability slugs migrated (`integration.py` + `create_<slug>_<category>_integration` in `bundle.py`, lazy `__init__.py`, USAGE.md)
- Legacy catalog factories unchanged; **`register.py`** remains legacy-compatible (no contract factory registration)
- **`enabled=True`** without injectable client raises **`IntegrationConfigurationError`**; no vendor SDK imports in `integration.py`
- **Registry v2 / contract registry wiring remains deferred**
- **Deferred slugs (9):** `llm_guardrail` catalog slugs (`llm_guard`, `guardrails_ai`, `nemo_guardrails`, `openguardrails`, `presidio`, `llama_guard`, `lakera`, `azure_content_safety`, `bedrock_guardrails`) — shared `llm_guardrail/bundles` layout without per-slug provider packages
- No LKW change; no global bootstrap registration

**Migrated categories (30):** all `SLUG_CATEGORY` folders except deferred `llm_guardrail` slugs (category contract exists; per-slug packages deferred).

---

## Phase INTEGRATIONS-2E — runtime cutover to single provider entrypoint (Done)

**Purpose:** Convert contract migration shells into real single-entrypoint providers. Each slug exposes exactly one public class: `<ProviderPascal><CategoryPascal>Integration`. Legacy catalog factories remain as compatibility shims delegating to that class; parallel public adapter/facade classes are removed or privatized.

**Distinction from INTEGRATIONS-2D:** 2D added contract-based `integration.py` beside legacy runtime adapters. 2E moves runtime behavior into the Integration class and eliminates symbol shadowing (e.g. `PineconeVectorStoreIntegration` in both `adapter.py` and `integration.py`).

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **INTEGRATIONS-2E-TESTS** | Test | P1 | **Done** | `test_provider_runtime_cutover.py` | `CUTOVER_SLUGS` registry; bundle shadowing guard; legacy shim + behavior tests |
| **INTEGRATIONS-2E-VECTOR-W1** | Code | P1 | **Done** | `pinecone`, `qdrant` | Single `*VectorStoreIntegration`; `adapter.py` removed; legacy factories shim to Integration |
| **INTEGRATIONS-2E-VECTOR-W2** | Code | P1 | **Done** | remaining 8 `vector_store` slugs | Same cutover pattern |
| **INTEGRATIONS-2E-STORAGE** | Code | P2 | **Done** | object_storage + relational/document stores | Per-provider cutover applied |
| **INTEGRATIONS-2E-COMM** | Code | P2 | **Done** | notification/channel/message/collaboration | Per-provider cutover applied |
| **INTEGRATIONS-2E-RETRIEVAL** | Code | P2 | **Done** | search/rerank/document_parser | Per-provider cutover applied |
| **INTEGRATIONS-2E-PLATFORM** | Code | P2 | **Done** | DevOps/security/platform providers | Per-provider cutover applied |
| **INTEGRATIONS-2E-AI** | Code | P2 | **Done** | AI/service/business providers | Per-provider cutover applied |
| **INTEGRATIONS-2E-OBS** | Code | P2 | **Done** | 25 `observability_backend` providers | Per-provider cutover applied |
| **INTEGRATIONS-2E-LLM-GUARDRAIL** | Code | P3 | **Deferred** | 9 `llm_guardrail` slugs | Requires **INTEGRATIONS-2F-LLM-GUARDRAIL-PACKAGE-NORMALIZATION** |

**INTEGRATIONS-2E status (2026-06-29):**

- Cut over (behavior tests pass): **185 slugs** — all `SLUG_CATEGORY` entries except deferred `llm_guardrail` (includes e.g. `chroma`, `slack`, `filesystem`, `github` across all 30 migrated categories)
- Legacy catalog factories (e.g. `create_<slug>_vector_store`, `create_<slug>_notification_channel`) are compatibility shims — construct the Integration class directly or via `.from_store()` / `.as_*()` views
- Public `adapter.py` removed for cut-over slugs; no symbol shadowing in `bundle.py`
- Contract factory unchanged; `register.py` remains legacy catalog hook
- No registry v2; no LKW change; no bootstrap change

**Cut-over registry:** `CUTOVER_SLUGS` in `tests/unit/integrations/providers/test_provider_runtime_cutover.py` — derived from `SLUG_CATEGORY` minus deferred `llm_guardrail` slugs (not a hand-maintained subset)

**Deferred (9):** `azure_content_safety`, `bedrock_guardrails`, `guardrails_ai`, `lakera`, `llama_guard`, `llm_guard`, `nemo_guardrails`, `openguardrails`, `presidio` — shared `bundles` layout; do not mark cut over until package normalization.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6, §7.7 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-13.1 | §13 Integrations | Integration marketplace catalog + trust scoring | P3 | **Done** |
| AUDIT-IDEAL-13.2 | §13 Integrations | Catalog hot-reload without host restart | P3 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

### 6.1d Harness implementation queue — integration closeout (closed)

**Purpose:** Single ordered list for **Phase INT** (Band 2l). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **INT-DOC.1–2** | Docs | **Done** | Appendix K + cross-refs | Author map complete |
| 2 | **INT-1** | Code | **Done** | `integration_runtime_bridge` | `test_integration_runtime_bridge.py` |
| 3 | **INT-2** | Code | **Done** | `integration_health_wiring` | `test_integration_health_wiring.py` |

---

### 6.1x Harness implementation queue — Integration depth (M.6 P5 done)

**Purpose:** Closeout record for **Phase M.6 P5** (Band 2ab). **Status:** **Done** (2026-06-02) — **33/34**.  
**Register:** [M.6 P5 — Master register](.#m6-p5--master-register-34-slugs) · **Execution order:** [§6.2af](.#62af-phase-m6-p5-execution-order-band-2ab--planned)
**Policy:** One slug per PR (or one harden wave ≤4 slugs); runs **in parallel** with §6.1 maintenance — pull when W-OPS / W-ADAPT / EVAL / prod stack needs the slug.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P5-CAT.1–3 | `ci_cd` extend, `security_scanner`, category mapping | **P0** | **Done** (CAT.2 deferred: `trivy`) |
| 1 | H-INT-6 | M-P5.1–M-P5.10 | Ops/metrics/CI/local cloud: prometheus, clickhouse, vault, pagerduty, github, gitlab_ci, circleci, azure_pipelines, mailpit, localstack | **P0** | **Done** |
| 2 | H-INT-7 | M-P5.11–M-P5.20 | Eval/async/artifacts: langfuse, phoenix, braintrust, mlflow, influxdb, timescaledb, temporal, redpanda, minio, s3 | **P0/P1** | **Done** |
| 3 | H-INT-8 | M-P5.21–M-P5.28 | Data plane lab: neo4j, mongodb, elasticsearch, nats, chroma, weaviate, launchdarkly, signoz | **P1/P2** | **Done** |
| 4 | H-INT-9 | M-P5.29–M-P5.34 | P2 reserve: codecov, trivy, grafana_oncall, opentelemetry_collector, snowflake, supabase | **P2** | **Done** |
| 5 | PRE | M-P5-PRE.1 | Tier-3 presets: `harness_metrics_stack`, `harness_eval_stack`, `harness_async_stack`, `harness_ci_stack` | **P0** | **Done** |

**Explicitly excluded:** Band 3 product agents; see [M.6 P5 register](.#m6-p5--harness-integration-depth-done--3334).

### 6.1y Harness implementation queue — Integration expansion (M.6 P6 Done)

**Purpose:** Ordered backlog for **Phase M.6 P6** (Band 2ac). **Status:** **Done** (2026-06-02) — **32/32**.  
**Register:** [M.6 P6 — Master register](.#m6-p6--master-register-32-slugs) · **Execution order:** [§6.2ag](.#62ag-phase-m6-p6-execution-order-band-2ac--done)
**Policy:** One slug per PR (or one CAT wave before first slug in a new category); runs **in parallel** with §6.1 maintenance — pull when security/sandbox/identity/GitOps/speech harness gaps block ops.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P6-CAT.1–9 | New categories: `security_scanner`, `sandbox_host`, `identity_provider`, `speech_provider`, `workflow_orchestrator`, `vision_serving`, `ml_inference_host`, `billing_meter`, `crm` | **P0** | **Done** |
| 1 | H-INT-10 | M-P6.1–M-P6.4 | Security + secrets: `trivy`, `snyk`, `semgrep`, `infisical` | **P0** | **Done** |
| 2 | H-INT-11 | M-P6.5–M-P6.7 | Cloud sandbox: `e2b`, `modal`, `daytona` | **P0/P1** | **Done** |
| 3 | H-INT-12 | M-P6.8–M-P6.10 | Identity: `auth0`, `keycloak`, `workos` | **P0/P1** | **Done** |
| 4 | H-INT-13 | M-P6.11–M-P6.13 | GitOps CI: `argocd`, `buildkite`, `jenkins` | **P0/P1** | **Done** |
| 5 | H-INT-14 | M-P6.14–M-P6.15 | Speech catalog: `elevenlabs`, `deepgram` | **P0** | **Done** |
| 6 | H-INT-15 | M-P6.16–M-P6.19 | Enterprise ops: `newrelic`, `splunk`, `zendesk`, `statsig` | **P1** | **Done** |
| 7 | H-INT-16 | M-P6.20–M-P6.24 | Data/workflow: `prefect`, `airflow`, `typesense`, `neon`, `pulsar` | **P1** | **Done** |
| 8 | H-INT-17 | M-P6.25–M-P6.32 | Reserve: `algolia`, `confluent`, `backblaze_b2`, `triton`, `replicate`, `stripe`, `salesforce`, `hubspot` | **P2** | **Done** |
| 9 | PRE | M-P6-PRE.1 | Tier-3 presets: `harness_security_stack`, `harness_sandbox_stack`, `harness_identity_stack`, `harness_gitops_stack` | **P0** | **Done** |
| 10 | WIRE | M-P6-WIRE.1–7 | Tool surface + sandbox/speech/identity bridges + promote gate + infra `p6` | **P0** | **Done** |

**Per-slug checklist:** see [M.6 P6 register](.#m6-p6--harness-integration-expansion-planned).

**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS`; four Tier-3 presets; gate green.

### 6.1z Harness implementation queue — Agent-developer expansion (M.7 P7 done)

**Purpose:** Ordered backlog for **Phase M.7 P7** (Band 2ad). **Status:** **Done** (2026-06-08) — **18/18**.  
**Register:** [M.7 P7 — Master register](.#m7-p7--agent-developer-integration-expansion-done--1818)
**Policy:** Reuse existing category contracts; `_shared/p8` thin factories; auto-wire `search_provider` / `document_parser` / `vector_store` catalog tools.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 1 | H-INT-P7-1 | M-P7.1–M-P7.5 | Research + RAG: `perplexity`, `arxiv`, `semantic_scholar`, `llamaparse`, `lancedb` | **P0** | **Done** |
| 2 | H-INT-P7-2 | M-P7.6–M-P7.9 | Interaction + browser + storage: `telegram`, `browserbase`, `google_drive`, `apify` | **P0** | **Done** |
| 3 | H-INT-P7-3 | M-P7.10–M-P7.14 | Workflow + wiki + identity + cache: `n8n`, `wikipedia`, `clerk`, `upstash_redis`, `upstash_qstash` | **P0/P1** | **Done** |
| 4 | H-INT-P7-4 | M-P7.15–M-P7.18 | Data warehouse: `okta`, `bigquery`, `motherduck`, `airbyte` | **P1** | **Done** |
| 5 | PRE | M-P7-PRE.1 | Tier-3 presets: `research_web_stack`, `document_ingest_stack`, `chat_bot_stack` | **P0** | **Done** |
| 6 | WIRE | M-P7-WIRE.1 | `extend_tool_profile_for_integration` — search/RAG auto-wiring | **P0** | **Done** |

**Closeout target:** catalog **185** slugs; `HARNESS_M7_P7_PROBE_SLUGS`; three Tier-3 presets; gate green.

---

### 6.2bd Phase INT execution order (Band 2l — closed 2026-06-02)

**Status:** **Done** · register: [Phase INT](plan/INTEGRATIONS.md) · queue: [§6.1d](.#61d-harness-implementation-queue--integration-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | INT-1 | `integration_runtime_bridge` | Critical |
| 2 | INT-2 | `integration_health_wiring` | High |
| 3 | INT-DOC.1–2 | Appendix K + plan sync | Low |

---

### 6.2bc Phase TS execution order (Band 2k — closed 2026-06-02)

**Status:** **Done** · register: [Phase TS](plan/TOOLS.md) · queue: [§6.1c](.#61c-harness-implementation-queue--toolsskills-closeout-closed)

Work **one TS ID per PR**; after each step update the TS master table + §6.1c + paydown log; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | TS-1 | `catalog_runtime_bridge` + `materialize_runtime_config` | Critical | TS-DOC.* (parallel OK) |
| 2 | TS-2 | Harness host LLM adapter wiring | High | — |
| 3 | TS-3 | `SkillResolverProtocol` | Medium | — |
| 4 | TS-DOC.1–2 | Appendix J + plan sync | Low | TS-1–3 |

---

### 6.2af Phase M.6 P5 execution order (Band 2ab — Planned)

**Status:** **Done** (2026-06-02) · register: [M.6 P5](.#m6-p5--harness-integration-depth-done--3334) · queue: [§6.1x](.#61x-harness-implementation-queue--integration-depth-m6-p5-done)

```text
Wave H-INT-0 (categories):  M-P5-CAT.1 → M-P5-CAT.2 → M-P5-CAT.3
Wave H-INT-6 (ops/CI):      M-P5.1 → M-P5.2 → M-P5.3 → M-P5.4 → M-P5.5 → M-P5.6 → M-P5.7 → M-P5.8 → M-P5.9 → M-P5.10
Wave H-INT-7 (eval/async):  M-P5.11 → M-P5.12 → M-P5.13 → M-P5.14 → M-P5.15 → M-P5.16 → M-P5.17 → M-P5.18 → M-P5.19 → M-P5.20
Wave H-INT-8 (data lab):    M-P5.21 → M-P5.22 → M-P5.23 → M-P5.24 → M-P5.25 → M-P5.26 → M-P5.27 → M-P5.28
Wave H-INT-9 (P2 reserve):  M-P5.29 → M-P5.30 → M-P5.31 → M-P5.32 → M-P5.33 → M-P5.34
Wave PRE (presets):         M-P5-PRE.1  (after H-INT-6 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P4 **Done**; M-P4.FU wiring **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** H-INT-6 unblocks W-OPS metrics + multi-CI; H-INT-7 unblocks EVAL/W-ADAPT; H-INT-8 is lab-only.  
**Closeout target:** catalog **136** slugs; `HARNESS_M6_P5_PROBE_SLUGS` + four Tier-3 presets; gate green.### 6.2ae Phase M.6 P4 execution order (Band 2aa — Done)

**Status:** **Done** (2026-06-02) · register: [M.6 P4](.#m6-p4--harness-platform-expansion-done) · queue: [§6.1w](.#61w-harness-implementation-queue--integration-expansion-m6-p4-closed)

```text
Wave H-INT-0 (categories):  M-P4-CAT.1 → M-P4-CAT.2  (before first slug in new category)
Wave H-INT-1 (storage):     M-P4.1 → M-P4.2 → M-P4.3 → M-P4.4
Wave H-INT-2 (obs stack):   M-P4.5 → M-P4.6 → M-P4.7
Wave H-INT-3 (secrets):     M-P4.8 → M-P4.9 → M-P4.10 → M-P4.11
Wave H-INT-4 (control):     M-P4.12 → M-P4.13 → M-P4.14 → M-P4.15 → M-P4.16
Wave H-INT-5 (enterprise):  M-P4.17 → M-P4.18 → M-P4.19 → M-P4.20 → M-P4.21 → M-P4.22 → M-P4.23 → M-P4.24 → M-P4.25 → M-P4.26 → M-P4.27 → M-P4.28
```

**Prerequisites:** Phase M core + M.6 P1/P2/P3 **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** Any wave after H-INT-0 may start when a slug is needed — prefer H-INT-1 → H-INT-2 → H-INT-3 order for W-OPS/adaptive unblock.  
**Closeout:** **Done** — catalog **127** in `layout.py`; `tests/unit/integrations/providers/test_p5_m6_p4_providers.py` (42 tests).

---

## Phase INTEGRATIONS-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates M.6 P5/P6, M.7 P7, M.12, H-INT-GRAPH; no open P0/P1  
**Prerequisites:** Phase INT **Closed** · catalog **185** slugs · AUDIT-IDEAL-13.1/13.2 **Done**  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| INTEGRATIONS-LC-S1 | **Re-audit** — M.6/M.7/M.12 register + tier-0 verdict | **Done** | High | No P0/P1 |
| INTEGRATIONS-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| INTEGRATIONS-LC-S3 | **Gate verification** | **Done** | High | 550 unit tests · 2 CI gate scripts |
| INTEGRATIONS-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** beta→stable slug promotion honesty · thin P4 provider shells · SaaS-only slugs without local container · nginx/ingress slug (ECP cross-ref)

### 6.1av Harness implementation queue — Integrations audit maintenance (planned)

**Source:** Layer 11 audit (2026-06-18) — `INTEGRATIONS` layer 13 · [`../audit_results/2026-06-18/INTEGRATIONS.md`](../../../audit_results/2026-06-18/INTEGRATIONS.md)
**Priority ladder:** **Band 1** (§6.1) — catalog honesty + provider depth; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **INT-MAINT-01** | CI | P2 | **Done** | Beta→stable promotion gate — `check_integration_maturity_labels.py` or conformance extension | STABLE slugs have health probe + test evidence |
| 2 | **INT-MAINT-02** | Code | P3 | **Done** | Thin P4 provider shells — hardening checklist + minimal health probe per shell | Each P4 shell passes probe unit test |
| 3 | **INT-MAINT-03** | Docs/Metadata | P3 | **Done** | SaaS-only slugs — honest lab-stack docs + `requires_local_container` metadata | Manifest field + USAGE honesty |
| 4 | **INT-MAINT-04** | Cross-ref | P4 | **Done** | nginx/ingress slug — cross-ref [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) | ECP owns slug; INT documents bridge |

**Suggested PR order:** INT-MAINT-01 → INT-MAINT-03 → INT-MAINT-02 → INT-MAINT-04.

**Explicitly excluded from MAINT:** H-INT-GRAPH — remains in existing phase register (post IDEAL-L3 W2).

### Speech catalog alignment (MOD-SPEECH-ARCH cross-ref)

**Source:** Idea audit 2026-06-19 · [ADR-MOD-001](../../technical/adr/entries/2026-06-19/ADR-MOD-001.md)
**Owner domain:** [`plan/MODALITY.md`](MODALITY.md) MOD-SPEECH-ARCH.*  
**Policy:** Hard cutover — delete `SpeechProvider` enum legacy; no transitional compatibility layer.

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **INT-SPEECH-ARCH.1** | Docs/Code | P2 | **Done** | Canon sync — `speech_provider` is sole vendor path for `speech.*` tools; remove enum references from integration wiring docs | [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) §Speech provider + MOD-SPEECH-ARCH gates green |

Close **INT-SPEECH-ARCH.1** in the same PR wave as **MOD-SPEECH-ARCH.4** (wiring unification).

---

## Architecture doc alignment (P2-ARCH)

| ID | Scope | Status |
|----|-------|--------|
| **P2-ARCH-06** | Clarify integration layer contract and access paths | **Done** (2026-06-20) |

---

## Phase INT-P8 — Dynamic Integration Selection & Agent Workspace Gateways (Planned)

**Status:** **Planned** — architecture & implementation backlog only (no code in this phase doc update)  
**Prerequisites:** Phase INTEGRATIONS-LC **Done** · catalog **194** shipped slugs (`layout.py`) · Full Harness LC unchanged  
**Architecture (1:1):** [`architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) — §Phase INT-P8
**Catalog (planned slugs):** [`architecture/satellites/INTEGRATIONS_provider_catalog.md`](../../architecture/satellites/INTEGRATIONS_provider_catalog.md) — §INT-P8
**Band:** 2ae (post–Full Harness LC strategic depth)  
**Policy:** One implementation ID per PR; **do not** register planned slugs in `layout.py` until the matching task PR; **do not** mark any INT-P8 task **Done** until code ships.

### Scope split

| Layer | This update | Follow-up PRs |
|-------|-------------|---------------|
| Architecture canon | INT-P8 boundaries, invariants, product mapping, non-goals | — |
| Provider catalog satellite | Planned categories/slugs (**Planned**, not shipped) | Register slug + contract + tests per task |
| Runtime / `layout.py` | **No change** | Per INT-P8.2–INT-P8.6 task |
| Tier-3 presets (`presets.py`) | Document only (INT-P8.9) | Preset functions in dedicated PR |
| ToolRuntime policy | Document alignment (INT-P8.8) | Policy gate extensions in TOOLS/runtime PR |

### Execution order (recommended)

```text
Wave ARCH:  INT-P8.12 (docs — this update) → INT-P8.1 (selection metadata design)
Wave CORE:  INT-P8.7 (selection engine contract) → INT-P8.8 (ToolRuntime policy alignment)
Wave GATE:  INT-P8.2 (mcp) ∥ INT-P8.3 (openapi_http) ∥ INT-P8.4 (local_workspace) ∥ INT-P8.5 (local_git)
Wave INTEL: INT-P8.6 (sourcegraph) — after INT-P8.5 recommended
Wave PRE:   INT-P8.9 (Tier-3 presets) — after gateway providers wired
Wave MAP:   INT-P8.10 (product mapping validation) · INT-P8.11 (non-goals gate in reviews)
```

**Parallelism:** INT-P8.2–INT-P8.5 may proceed in parallel after INT-P8.1 metadata schema and INT-P8.7 contract stub are agreed.

---

### INT-P8 master register

| ID | Title | Type | Priority | Status | Depends on | Acceptance criteria |
|----|-------|------|----------|--------|------------|---------------------|
| **INT-P8.1** | Dynamic Integration Selection Metadata | Design + Code | **P0** | **Planned** | INTEGRATIONS-LC | Extended manifest/profile fields (`capabilities`, `operations`, `read_write`, `auth_type`, `required_scopes`, `data_sensitivity`, `latency_class`, `cost_class`, `locality`, `deterministic`, `side_effect_level`, `supported_task_intents`, `suitable_agent_types`, `supports_dry_run`, `supports_rollback`, `requires_human_approval`, `rate_limit_class`, `testability`, `selection_hints`, `risk`); backward-compatible with existing **194** manifests; schema documented in architecture satellite |
| **INT-P8.2** | MCP Gateway Integration (`tool_protocol_gateway` / `mcp`) | Code | **P0** | **Planned** | INT-P8.1, INT-P8.8 | Category contract + provider: MCP server discovery, list tools/resources, fetch tool schemas, invoke tools **via ToolRuntime only**, read resources, health probe, write/side-effect policy gate, selection metadata; fake MCP server + tests blocking side effects without approval |
| **INT-P8.3** | OpenAPI HTTP Connector (`api_connector` / `openapi_http`) | Code | **P0** | **Planned** | INT-P8.1, INT-P8.8 | Load OpenAPI from file/URL; list/describe operations; request schema validation; read-only execution; write ops only with ToolRuntime approval; HTTP method risk classification; auth metadata; health probe; mock API server; tests: GET/POST, auth missing, schema invalid, blocked unsafe method |
| **INT-P8.4** | Local Workspace Integration (`workspace_store` / `local_workspace`) | Code | **P0** | **Planned** | INT-P8.1, INT-P8.8 | Root-scoped workspace; list tree, read, text search; write/delete/move gated; glob allow/deny; file size limit; path traversal + symlink escape blocked; health probe; security path tests; **not** a alias of `filesystem` object storage |
| **INT-P8.5** | Local Git Worktree Integration (`code_repository` / `local_git`) | Code | **P0** | **Planned** | INT-P8.1, INT-P8.8 | Repo detection; status, branch, changed files, diff, log, read file at ref, blame; apply_patch + commit approval-gated; push **out of scope**; branch allowlist; dirty repo detection; health probe; temp-repo tests |
| **INT-P8.6** | Code Intelligence Integration (`code_intelligence` / `sourcegraph`) | Code | **P1** | **Planned** | INT-P8.1 | `sourcegraph`: code/symbol/commit/diff/repo search, fetch file by repo/ref/path, repo allowlist, read-only contract, health probe, mock GraphQL tests, no-token-logging test; `github_code` optional follow-up — **not** in first wave |
| **INT-P8.7** | Integration Selection Engine Contract | Design + Code | **P0** | **Planned** | INT-P8.1 | Input: task intent, required capabilities, risk tolerance, locality, read/write, data sensitivity; output: category, provider slug, operation, reason, required approvals; candidate ranking, fallback, safe refusal, explainability, trace/diagnostic event |
| **INT-P8.8** | ToolRuntime Policy Gate Alignment | Design + Code | **P0** | **Planned** | INT-P8.1 | Read-only without approval; write with approval; destructive with explicit HITL; external side effects always via ToolRuntime; MCP tool classification pre-exec; OpenAPI unsafe methods blocked default; workspace write/delete/move gated; git commit/patch gated; audit trail for all side effects; cross-ref [`plan/TOOLS.md`](TOOLS.md) |
| **INT-P8.9** | New Tier-3 Integration Presets | Code | **P1** | **Planned** | INT-P8.2–INT-P8.6 (partial OK per preset) | `local_workspace_stack()`, `coding_agent_stack()`, `enterprise_api_stack()`, `mcp_gateway_stack()` documented + implemented in `registry/presets.py`; each composes existing + new slugs per architecture §INT-P8.9 |
| **INT-P8.10** | Product Mapping | Docs | **P2** | **Planned** | INT-P8.12 | Architecture + plan document which products use which INT-P8 stacks (Local Knowledge Workspace, Dispute Simulation, research/coding/automation/enterprise/audit/document agents) |
| **INT-P8.11** | Explicit Non-Goals | Docs + Review gate | **P2** | **Planned** | INT-P8.12 | Non-goals listed in architecture + plan; PR checklist rejects catalog-padding slugs and direct agent→integration bypass |
| **INT-P8.12** | Documentation and Plan Structure | Docs | **P0** | **Planned** | INTEGRATIONS-LC | `architecture/INTEGRATIONS.md` §INT-P8; provider catalog §INT-P8 planned; this phase register; **no** `layout.py` / runtime catalog / shipped slug count changes; **no** task marked Done without implementation |

---

### INT-P8.9 — Tier-3 preset composition (planned)

| Preset | Composes (existing + planned) |
|--------|------------------------------|
| `local_workspace_stack()` | `local_workspace`, `local_git`, local `document_parser`, `inmemory`/`lancedb`, `log`/`lab_json`, optional `otel` |
| `coding_agent_stack()` | `local_git`, `local_workspace`, `sourcegraph` or `github_code`, `semgrep` (existing), optional `sandbox_host`, optional `ci_cd` |
| `enterprise_api_stack()` | `openapi_http`, `identity_provider`, `secrets_store`, `observability_backend`, `notification_channel`, ToolRuntime approval policy |
| `mcp_gateway_stack()` | `mcp`, `secrets_store`, ToolRuntime policy, `observability_backend` |

---

### INT-P8.10 — Product mapping

| Product / agent class | Primary INT-P8 stack | Key integrations |
|----------------------|---------------------|------------------|
| Local Knowledge Workspace | `local_workspace_stack` | workspace, git, parser, vector |
| Dispute Simulation Workspace | `local_workspace_stack` (+ domain tools) | workspace, parser, vector |
| Research agents | `research_web_stack` + selection engine | `openapi_http`, existing search/RAG |
| Coding agents | `coding_agent_stack` | git, workspace, sourcegraph, semgrep |
| Automation agents | `enterprise_api_stack` / `mcp_gateway_stack` | openapi, MCP, notifications |
| Enterprise assistant | `enterprise_api_stack` | openapi, identity, secrets, observability |
| Repo architecture audit agents | `coding_agent_stack` (read-heavy) | git, sourcegraph, semgrep |
| Document intelligence agents | `document_ingest_stack` + workspace | workspace, parser (existing) |

---

### INT-P8.11 — Explicit non-goals

INT-P8 implementation PRs **MUST NOT** add:

- Additional LLM providers
- Additional vector DB or graph store vendors
- Additional observability / eval vendors
- Additional browser automation providers
- New project-management SaaS without Tier-3 product owner
- LangChain / LlamaIndex as integration slugs
- Zapier / Make.com in first wave
- Git **push** in first wave (`local_git`)
- Direct agent invocation of MCP tools
- OpenAPI write/unsafe HTTP methods without ToolRuntime approval

---

### INT-P8 closeout target (future)

- Selection metadata on new + migrated hot-path providers
- Up to **5** new categories, **≤6** planned first-wave slugs registered (not duplicating existing **194**)
- Four Tier-3 presets shipped in code
- Selection engine + ToolRuntime policy tests green
- Gate green; **Full Harness LC** status unchanged until explicit LC re-validation if scope warrants

**ADR:** TBD when INT-P8.1 metadata schema is implemented (likely required for selection engine contract).

---

*End of Integrations Implementation Plan.*
