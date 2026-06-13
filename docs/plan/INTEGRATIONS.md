# Integrations — Implementation Plan

**Architecture (1:1):** [`architecture/INTEGRATIONS.md`](../architecture/INTEGRATIONS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

**RAG engine (layer 14):** [`architecture/RAG.md`](../architecture/RAG.md) ↔ [`plan/RAG.md`](RAG.md) — M-RAG, M-RAG-DEPTH, **M-RAG-GRAPH** (GraphRAG platform). This plan covers **integration catalog** slugs only; RAG adapters for `graph_store` are owned by M-RAG.38–M-RAG.51 in [`plan/RAG.md`](RAG.md).

---

## Phase H-INT-GRAPH — graph_store expansion (Planned)

**Purpose:** New `graph_store` vendor slugs required before RAG adapters M-RAG.49–M-RAG.51.  
**RAG coordination:** [`plan/RAG.md`](RAG.md) Wave G4 · GAP-RAG-33.

| ID | Slug | Category | Priority | Status | RAG deliverable | Notes |
|----|------|----------|----------|--------|-----------------|-------|
| H-INT-GRAPH-1 | `neptune` | graph_store | **P3** | **Done** | M-RAG.49 | AWS Neptune — OpenCypher HTTP bridge |
| H-INT-GRAPH-2 | `orientdb` | graph_store | **P3** | **Done** | M-RAG.50 | OrientDB OpenCypher HTTP bridge |
| H-INT-GRAPH-3 | `arangodb` | graph_store | **P3** | **Done** | M-RAG.51 | ArangoDB AQL HTTP bridge |

**Per-slug checklist:** contract gate → `providers/graph_store/<slug>/` → health probe → bootstrap register → RAG `RagGraphStoreBackend` adapter (M-RAG.38 registry) → gate green.

**Explicitly out of scope:** Microsoft GraphRAG library vendoring (harness-native indexer M-RAG.47); TigerGraph / JanusGraph unless product reprioritizes.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6, §7.7 · baseline **32/32 L3**  
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
**Register:** [M.6 P5 — Master register](#m6-p5--master-register-34-slugs) · **Execution order:** [§6.2af](#62af-phase-m6-p5-execution-order-band-2ab--planned)  
**Policy:** One slug per PR (or one harden wave ≤4 slugs); runs **in parallel** with §6.1 maintenance — pull when W-OPS / W-ADAPT / EVAL / prod stack needs the slug.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P5-CAT.1–3 | `ci_cd` extend, `security_scanner`, category mapping | **P0** | **Done** (CAT.2 deferred: `trivy`) |
| 1 | H-INT-6 | M-P5.1–M-P5.10 | Ops/metrics/CI/local cloud: prometheus, clickhouse, vault, pagerduty, github, gitlab_ci, circleci, azure_pipelines, mailpit, localstack | **P0** | **Done** |
| 2 | H-INT-7 | M-P5.11–M-P5.20 | Eval/async/artifacts: langfuse, phoenix, braintrust, mlflow, influxdb, timescaledb, temporal, redpanda, minio, s3 | **P0/P1** | **Done** |
| 3 | H-INT-8 | M-P5.21–M-P5.28 | Data plane lab: neo4j, mongodb, elasticsearch, nats, chroma, weaviate, launchdarkly, signoz | **P1/P2** | **Done** |
| 4 | H-INT-9 | M-P5.29–M-P5.34 | P2 reserve: codecov, trivy, grafana_oncall, opentelemetry_collector, snowflake, supabase | **P2** | **Done** |
| 5 | PRE | M-P5-PRE.1 | Tier-3 presets: `harness_metrics_stack`, `harness_eval_stack`, `harness_async_stack`, `harness_ci_stack` | **P0** | **Done** |

**Explicitly excluded:** Band 3 product agents; see [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334).

### 6.1y Harness implementation queue — Integration expansion (M.6 P6 Done)

**Purpose:** Ordered backlog for **Phase M.6 P6** (Band 2ac). **Status:** **Done** (2026-06-02) — **32/32**.  
**Register:** [M.6 P6 — Master register](#m6-p6--master-register-32-slugs) · **Execution order:** [§6.2ag](#62ag-phase-m6-p6-execution-order-band-2ac--done)  
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

**Per-slug checklist:** see [M.6 P6 register](#m6-p6--harness-integration-expansion-planned).

**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS`; four Tier-3 presets; gate green.

### 6.1z Harness implementation queue — Agent-developer expansion (M.7 P7 done)

**Purpose:** Ordered backlog for **Phase M.7 P7** (Band 2ad). **Status:** **Done** (2026-06-08) — **18/18**.  
**Register:** [M.7 P7 — Master register](#m7-p7--agent-developer-integration-expansion-done--1818)  
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

**Status:** **Done** · register: [Phase INT](plan/INTEGRATIONS.md) · queue: [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | INT-1 | `integration_runtime_bridge` | Critical |
| 2 | INT-2 | `integration_health_wiring` | High |
| 3 | INT-DOC.1–2 | Appendix K + plan sync | Low |### 6.2bc Phase TS execution order (Band 2k — closed 2026-06-02)

**Status:** **Done** · register: [Phase TS](plan/TOOLS.md) · queue: [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed)

Work **one TS ID per PR**; after each step update the TS master table + §6.1c + paydown log; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | TS-1 | `catalog_runtime_bridge` + `materialize_runtime_config` | Critical | TS-DOC.* (parallel OK) |
| 2 | TS-2 | Harness host LLM adapter wiring | High | — |
| 3 | TS-3 | `SkillResolverProtocol` | Medium | — |
| 4 | TS-DOC.1–2 | Appendix J + plan sync | Low | TS-1–3 |

---

### 6.2af Phase M.6 P5 execution order (Band 2ab — Planned)

**Status:** **Done** (2026-06-02) · register: [M.6 P5](#m6-p5--harness-integration-depth-done--3334) · queue: [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-done)

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

**Status:** **Done** (2026-06-02) · register: [M.6 P4](#m6-p4--harness-platform-expansion-done) · queue: [§6.1w](#61w-harness-implementation-queue--integration-expansion-m6-p4-closed)

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

## Phase INT — Integration control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (INT-DOC.* + INT-1–2); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §13; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix K**.

**Priority ladder:** **Band 2l** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bd](#62bd-phase-int-execution-order-band-2l--closed) · queue: [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed)

### INT — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| INT-DOC.1 | INT0 | **Appendix K** — integration control plane (§K.1–K.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| INT-DOC.2 | INT0 | **Cross-ref sync** — plan, README, AUDIT_MAP §13, audit prompt ref #8 | **Done** | Medium | `docs/*` | Links resolve |
| INT-1 | INT1 | **`integration_runtime_bridge.py`** — explicit `integration_profile` on `RuntimeConfig` | **Done** | **Critical** | `integration_runtime_bridge.py`, `runtime_config_bridge.py` | `test_integration_runtime_bridge.py` |
| INT-2 | INT2 | **`integration_health_wiring.py`** — bootstrap health probes on `wire_application_environment` | **Done** | High | `integration_health_wiring.py`, `environment_wiring.py` | `test_integration_health_wiring.py` |

### INT — Paydown log

| Date | INT ID | Summary |
|------|--------|---------|
| 2026-06-02 | INT-DOC.1, INT-DOC.2 | Appendix K + cross-refs; AUDIT_MAP §13 |
| 2026-06-02 | INT-1, INT-2 | Integration runtime bridge + health wiring |

**Phase INT complete when:** INT-1–2 + INT-DOC.* **Done**; §6.1d queue closed. **Status: complete (2026-06-02).**

---

### Phase H — Interaction Surfaces (§18)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| H.1 | Outbound webhook delivery | **Done** | §18 | Pluggable delivery + formatters; HTTP opt-in |
| H.2 | `InteractionAdapter` protocol | **Done** | §18 | Inbound → normalized `Task` |
| H.3 | Slack inbound lab path | **Done** | §18 | Debug API intake + signature stub |
| H.4 | HITL notification templates | **Done** | §42.10 | Reusable template + `notify_hitl_pause`; Slack/Teams formatters |
| H.5 | Teams parity | **Done** | §18 | Activity parser + HMAC verifier + debug intake tests |
| H.6 | Organization Worker demo | **Done** | §38 | E2E lab: intake → HITL → notification → resume |

---

---

### Phase M — Integration Library (Tier-0 Catalog)

**Canon:** §7.1.1–§7.1.5  
**Goal:** One discoverable integration catalog so platform teams ship adapters and agent teams compose them in Tier-3 — without duplicating Redis/Postgres/Slack clients per agent.

**Principle:** evolve existing modules (`queueing/`, `distributed/`, `websearch/`, …) into catalog providers; do not fork parallel stacks.

**Catalog (2026-06-08):** **185** slugs in `layout.py` · **12** core / **185** full preset · timeline: pre-P4 **99** → M.6 P4 **127** (+28) → M.6 P5 **135** (+8 greenfield, 25 hardened) → M.6 P6 **167** (+32) → M.7 P7 **185** (+18).

**Out of scope:** `intergrax/llm_adapters/` — LLM providers are **not** part of the Integration Library (§7.1.2). RAG retrieval orchestration (`intergrax/rag/`) — [`plan/RAG.md`](RAG.md).

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M.0 | Integration backlog + categories approved | **Done** | Canon §7.1.3 catalog table |
| M.1 | Scaffold `intergrax/integrations/` package | **Done** | `contracts/`, `registry/`, `_shared/`, `providers/` |
| M.2 | Category contracts (P0 set) | **Done** | 7 P0 contracts + re-exports for queueing/notifications/interactions |
| M.3 | `IntegrationRegistry` + `IntegrationProfile` | **Done** | `catalog.register_integration`, `resolve`, env/mapping profile |
| M.4 | P0 providers — wrap existing | **Done** | See **M.4 provider tracker** below |
| M.5 | Provider conformance test harness | **Done** | `tests/unit/integrations/`, `_shared/conformance.py` |
| M.6 | P1 providers (on demand) | **Done** (beta) | postgresql, mysql, jira, confluence, prometheus, ms365_graph, aws, azure, gcp — see M.4/M.6 trackers |
| M.6 P2 | Extended providers (on demand) | **Done** (beta) | All P2/P3 slugs shipped 2026-05-30 — see **M.6 P2 tracker**; `_shared/p2/` + thin `providers/<slug>/` shells |
| M.6 P4 | Harness platform expansion | **Done** (beta) (28/28) | `_shared/p5/` · `bootstrap_m6_p4.py` · [M.6 P4 register](#m6-p4--harness-platform-expansion-done) |
| M.6 P5 | Harness integration depth (audit 2026-06-02) | **Done** (33/34) | Harden 25 STABLE + health · 8 greenfield · `trivy` → [M.6 P6](#m6-p6--harness-integration-expansion-planned) · [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334) |
| M.6 P6 | Harness integration expansion (audit 2026-06-02) | **Done** (32/32) | Security, sandbox, identity, GitOps CI, speech catalog, enterprise ops, data/workflow, modality reserve · [M.6 P6 register](#m6-p6--harness-integration-expansion-planned) · Band **2ac** |
| M.7 P7 | Agent-developer integration expansion (audit 2026-06-08) | **Done** (18/18) | Research/RAG, chat bots, browser automation, workflow glue, serverless cache/queue, warehouse analytics · [M.7 P7 register](#m7-p7--agent-developer-integration-expansion-done--1818) · Band **2ad** |
| M.12 | LLM guardrail vendor adapters | **Done** | Category `llm_guardrail` · adapters + middleware + CI · [M.12 register](#phase-m12--llm-guardrail-integrations-planned) |
| M.7 | Agent Creation Guide § integrations | **Done** | Appendix E — capabilities/tools vs `IntegrationProfile` / `wire_lab_integrations()` |
| M.8 | Lab `IntegrationProfile` example | **Done** | `applications/lab_application/` — `wire_lab_integrations()` + `log` provider |

**M.4 delivery workflow (one provider per iteration):**

1. Implement `providers/<category>/<slug>/` (wrap legacy module — no fork).
2. Register via `register_<slug>_integration()` + `register_default_integrations()`.
3. Unit tests under `tests/unit/integrations/providers/`.
4. Add `providers/<slug>/USAGE.md` — English usage guide (factory + `IntegrationProfile` + API invoke example). Extend `scripts/generate_integration_usage_docs.py` and run `uv run python scripts/generate_integration_usage_docs.py`.
5. Update canon §7.1.3 status + this tracker + migration map row.
6. Next slug in priority order.

#### M.4 provider tracker

| Slug | Category | Status | Package | Legacy source |
|------|----------|--------|---------|---------------|
| `redis` | key_value_cache | **Done** | `providers/redis/` — `create_redis_integration()` (KV, idempotency, rate limit, semaphore, rerank) |
| `sqlite` | relational_store | **Done** | `providers/sqlite/` — `create_sqlite_integration()` (trace, events, checkpoints, HITL, …) |
| `kafka` | message_bus | **Done** (+ adopcja) | `providers/kafka/` — runtime transport delegates here |
| `celery` | message_bus | **Done** | `providers/celery/` — `create_celery_integration()` (inject `app` or broker/backend env) |
| `google_cse` | search_provider | **Done** | `providers/google_cse/` — `create_google_cse_integration()` (legacy `GOOGLE_CSE_*` env) |
| `bing` | search_provider | **Done** | `providers/bing/` — `create_bing_integration()` (legacy `BING_SEARCH_V7_API_KEY`) |
| `slack` | notification + interaction | **Done** (+ adopcja) | `providers/slack/` — runtime wiring delegates here |
| `teams` | notification + interaction | **Done** (+ adopcja) | `providers/teams/` — runtime wiring delegates here |
| `webhook` | notification_channel | **Done** (+ adopcja) | `providers/webhook/` — generic HTTP + `GenericJsonPayloadFormatter` |
| `lab_json` | interaction_surface | **Done** (+ adopcja) | `providers/lab_json/` — lab intake; runtime channel ``lab`` |
| `rabbitmq` | message_bus | **Done** (+ adopcja) | `providers/rabbitmq/` — `create_rabbitmq_integration()` (requires `kv_store`) |
| `log` | notification_channel | **Done** (+ adopcja) | `providers/log/` — wraps `LoggingNotificationAdapter`; lab profile default |
| `postgresql` | relational_store | **Done** (beta) | `providers/postgresql/` — `RelationalStore` via psycopg3; only `opens.py` connects |
| `mysql` | relational_store | **Done** (beta) | `providers/mysql/` — `RelationalStore` via pymysql; only `opens.py` connects |
| `databricks` | relational_store | **Done** (beta) | `providers/databricks/` — SQL Warehouse via databricks-sql-connector; only `opens.py` connects |
| `mongodb` | document_store | **Done** (beta) | `providers/mongodb/` — flexible JSON `DocumentStore`; PyMongo only in `opens.py` |
| `pinecone` | vector_store | **Done** (beta) | `providers/pinecone/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `qdrant` | vector_store | **Done** (beta) | `providers/qdrant/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `chroma` | vector_store | **Done** (beta) | `providers/chroma/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `s3` | object_storage | **Done** (beta) | `providers/s3/` — put/get/delete/presigned_url; boto3 only in `opens.py` |
| `jira` | issue_tracker | **Done** (beta) | `providers/jira/` — REST v3; only `opens.py` creates httpx client |
| `confluence` | wiki_knowledge | **Done** (beta) | `providers/confluence/` — REST wiki; only `opens.py` creates httpx client |
| `prometheus` | observability_backend | **Done** (beta) | `providers/prometheus/` — PromQL query API; only `opens.py` creates httpx client |
| `elasticsearch` | observability_backend | **Done** (beta) | `providers/elasticsearch/` — `_search` aggregations; only `opens.py` creates httpx client |
| `ms365_graph` | collaboration_suite | **Done** (beta) | `providers/ms365_graph/` — Graph mail/calendar/directory; only `opens.py` creates httpx client |
| `cassandra` | document_store | **Done** (beta) | `providers/cassandra/` — CQL get/put/delete/query; only `opens.py` creates driver session |
| `aws` | cloud_platform | **Done** (beta) | `providers/aws/` — IAM/STS auth + category defaults; only `opens.py` creates boto3 session |
| `azure` | cloud_platform | **Done** (beta) | `providers/azure/` — MI / service principal + category defaults; only `opens.py` creates credential |
| `gcp` | cloud_platform | **Done** (beta) | `providers/gcp/` — ADC / service account + category defaults; only `opens.py` creates credentials |

#### M.6 P2 — Extended provider tracker (canon §7.1.3 P2)

Deliver after M.6 P1 priorities unless a product app blocks on a specific slug. Each P2 provider follows the same workflow as M.4 (contract → `providers/<slug>/` → tests → catalog row).

| Slug | Category | Status | Rationale / notes |
|------|----------|--------|-------------------|
| **`cassandra`** | **document_store** | **Done** (beta) | High-volume log / event retention; CQL driver via `opens.py` single entry |
| **`elasticsearch`** | **observability_backend** | **Done** (beta) | Log search / aggregations (`_search` + Lucene `query_string` via ObservabilityBackend); complements `prometheus` |
| **`databricks`** | **relational_store** | **Done** (beta) | Lakehouse SQL Warehouse; PAT via `opens.py`; `execute` / `fetch_all` for analytics agents |
| **`mongodb`** | **document_store** | **Done** (beta) | Flexible JSON documents; partition-scoped get/put/delete/query via PyMongo |
| **`pinecone`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/pinecone_vector_store.py` |
| **`qdrant`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/qdrant_vector_store.py` |
| **`chroma`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/chroma_vector_store.py` |
| **`s3`** | **object_storage** | **Done** (beta) | AWS S3 blobs; boto3 only in `opens.py` |
| **`azure_blob`** | **object_storage** | **Done** (beta) | Azure Blob; `providers/azure_blob/` + shared `CatalogObjectStorage` |
| **`gcs`** | **object_storage** | **Done** (beta) | GCS via `_shared/p2/gcs_blob.py` |
| **`dynamodb`** | **document_store** | **Done** (beta) | boto3 table facade in `_shared/p2/factories.py` |
| **`oracle`** / **`mssql`** / **`azure_sql`** / **`cloud_sql`** | **relational_store** | **Done** (beta) | SQL adapters via `_shared/p2/clients.py` |
| **`memcached`** / **`elasticache`** | **key_value_cache** | **Done** (beta) | pymemcache / Redis-compatible duck client |
| **`sqs`** / **`service_bus`** / **`pubsub`** | **message_bus** | **Done** (beta) | `CloudTaskQueue` over cloud SDK facades |
| **`email_smtp`** | **notification_channel** | **Done** (beta) | stdlib SMTP in factory open path |
| **`otel`** | **observability_backend** | **Done** (beta) | OTLP-oriented metrics facade (beta noop exporter default) |
| **`github`** / **`linear`** / **`azure_devops`** | **issue_tracker** | **Done** (beta) | REST issue trackers via httpx |
| **`notion`** / **`sharepoint`** | **wiki_knowledge** | **Done** (beta) | REST wiki adapters |
| **`google_workspace`** | **collaboration_suite** | **Done** (beta) | Gmail / Calendar REST |
| **`brave`** / **`serpapi`** | **search_provider** | **Done** (beta) | Shared `_shared/rest_search.py` hit mappers |
| **`playwright`** | **browser_automation** | **Done** (beta) | `contracts/browser_automation.py` + Playwright factory |

#### M.6 P3 / M.7 — Harness integrations (Done beta, 2026-05-29)

**M.11 harness defaults (Done beta):** default `notify_channel` injection from lab wiring (`task_defaults.py`, `LAB_HARNESS` enricher on lab run + interaction intake).

**M.10 harness Tier A (Done beta):** composite observability (`observability_backends` + role-based `resolve_observability_backend`), HITL→PagerDuty runtime path (`create_harness_notification_adapter`, `LAB_HARNESS`), integration tests.

**M.9 harness depth (Done beta):** full adapters (LangSmith, OpenSearch, Vespa, GitLab, PagerDuty, Braintrust), tools (`gitlab.create_issue`, `pagerduty.trigger_incident`, `braintrust.log_eval`), `slash_command`, lab harness profile, CI harness-smoke job. Catalog: **99** (M.9 closeout; **135** after M.6 P5).

**M.8 harness gap (Done beta):** +14 slugs via `_shared/p4/factories.py`

**M.7 harness (Done beta):** +21 slugs via `_shared/p3/factories.py` (incl. **sentry**).

#### M.7 — Document parser catalog bridge (2026-05-30)

Vendor document parsing moved from `intergrax/rag/document_loaders/parsers/` into `integrations/providers/document_parser/`. RAG uses `CatalogDocumentParser` + `resolve_document_parser()`.

**Wave 2 (2026-05-30):** `openpyxl`, `whisper`, `yt_dlp`; `cohere_rerank` / `jina_rerank`; Bing/Google CSE implementations under `integrations/.../web_client.py` (websearch re-exports); `ParserPipeline` ingestion trace; tool `rag.ingest_document`; `IntegrationProfile.legal_product()` / `research_product()` / `lab()` with `document_parser=docling`; lab `GET /v1/lab/integrations/docling/health`.

**Wave 3 (2026-05-30):** `reddit`, `google_places` search providers; Chroma/Qdrant/Pinecone SDK in `integrations/.../rag_store.py` (RAG shims); runtime SQLite delivery ledger via `sqlite/opens`; `rag.ingest_document` env flags for legal/research; parser trace export to Langfuse/Sentry.

**Wave 4 (2026-05-30):** `inmemory` vector store SDK in `integrations/.../inmemory/rag_store.py`; SQLite observability via `integration_profile_wiring` + `wire_nexus_observability(integration_profile=…)` with default-path fallback; parser pipeline spans appended to `RunTraceWriter` (`parser_trace_span.py`); vendor import governance script + CI gate; Phase Q scaffold defaults (`IntegrationProfile`, `ToolProfile` with `websearch.read_url`).

**Wave 5 (2026-05-30):** Phase P wave 3 tools (`websearch.fetch_batch`, `rag.list_collections`, `observability.query_traces`); full `IntegrationProfile` on legal/research products; Weaviate/Milvus `rag_store.py`; Redis SDK cleanup in distributed/rag shims; governance extended to `agents/` + `rag/`; parser trace export on `RunTraceWriter.finalize_run`; Phase Q scaffold wave 2 (lab vs product ToolProfile, env profile override).

| Slug | Status | Notes |
|------|--------|-------|
| `docling` | **Done** (beta) | local + server; `opens.py` only Docling/httpx imports |
| `pymupdf` | **Done** (beta) | PDF + optional Tesseract OCR |
| `unstructured` | **Done** (beta) | HTML loader |
| `python_docx` | **Done** (beta) | Word `.docx` |
| `openpyxl` | **Done** (beta) | Excel/CSV via pandas |
| `whisper` | **Done** (beta) | Audio + YouTube (uses yt_dlp opens) |
| `yt_dlp` | **Done** (beta) | YouTube audio/video download |
| `cohere_rerank` | **Done** (beta) | RAG rerank via integration resolver |
| `jina_rerank` | **Done** (beta) | RAG rerank via integration resolver |
| `reddit` | **Done** (beta) | Reddit OAuth2 search |
| `google_places` | **Done** (beta) | Google Places text search |

#### M.6 P4 — Harness platform expansion (Done)

**Status:** **Done** (2026-06-02) — **28/28 Done** · catalog **127** slugs  
**Source:** Integration harness ROI audit (2026-06-02)  
**Queue:** [§6.1w](#61w-harness-implementation-queue--integration-expansion-m6-p4-closed) · **Execution order:** [§6.2ae](#62ae-phase-m6-p4-execution-order--done)  
**Priority ladder:** **Band 2aa** (§4.0) — **Done**  
**Implementation:** `intergrax/integrations/_shared/p5/` + thin shells via `scripts/wire_p5_m6_p4_providers.py` · `register_m6_p4_integrations()` in `bootstrap_extended.py`

**Hard rules:**

- **No** LLM API slugs — use `llm_adapters/` (canon §7.1.2).
- **New categories** (`feature_flag`, `ci_cd`) require canon §5.2.4 review before merge — track **M-P4-CAT.\*** first.
- Reuse M.4 workflow: contract (or extend existing) → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → gate green.
- `ollama` bridges existing `infra/integration` Docker + `LLMAdapter` Ollama path — not a duplicate LLM stack.

**New category proposals (M-P4-CAT):**

| ID | Category | Slugs | Status | Acceptance |
|----|----------|-------|--------|------------|
| M-P4-CAT.1 | `feature_flag` | `unleash`, `launchdarkly` | **Done** | `FeatureFlagBackend` + `IntegrationCategory.FEATURE_FLAG` |
| M-P4-CAT.2 | `ci_cd` | `github_actions` | **Done** | `CiCdBackend` + `IntegrationCategory.CI_CD` |

##### M.6 P4 — Master register (28 slugs)

| Wave | ID | Slug | Category | Priority | Status | Harness ROI | Acceptance |
|------|-----|------|----------|----------|--------|-------------|------------|
| H-INT-1 | M-P4.1 | `pgvector` | vector_store | **P0** | **Done** (beta) | Unify PostgreSQL (stable) + RAG vectors + adaptive stores | `IntegrationProfile.vector_store=pgvector`; RAG hybrid query; gate unit tests |
| H-INT-1 | M-P4.2 | `duckdb` | relational_store | **P0** | **Done** (beta) | Local OLAP for `phase_w_adapt_report`, eval trends, golden scenarios | `RelationalStore` read path; CI-friendly file DB; report script optional backend |
| H-INT-1 | M-P4.3 | `influxdb` | observability_backend | **P1** | **Done** (beta) | Time-series utility U, cost, latency — adaptive KPIs | `ObservabilityBackend` query_range; W-ADAPT signal export optional |
| H-INT-1 | M-P4.4 | `timescaledb` | relational_store | **P1** | **Done** (beta) | Hypertables for adaptive + eval registry trends on Postgres | Extends `postgresql` contract; migration note in USAGE |
| H-INT-2 | M-P4.5 | `grafana` | observability_backend | **P0** | **Done** (beta) | W-OPS.4 SLO dashboards; L3 release visibility | HTTP API health + dashboard URL artifact; lab stack doc |
| H-INT-2 | M-P4.6 | `loki` | observability_backend | **P0** | **Done** (beta) | Log query for RuntimeEvents / structured logs | LogQL query adapter; complements `prometheus` |
| H-INT-2 | M-P4.7 | `tempo` | observability_backend | **P0** | **Done** (beta) | Trace backend for OTEL (`otel` slug exists; dedicated store) | Trace query by `trace_id`; lab compose profile |
| H-INT-3 | M-P4.8 | `aws_secrets_manager` | secrets_store | **P0** | **Done** (beta) | Prod harness secrets; complements `aws` facade | `SecretsStore` get/list; no secrets in agent code |
| H-INT-3 | M-P4.9 | `azure_key_vault` | secrets_store | **P0** | **Done** (beta) | Azure prod parity | MI / SP auth via `azure` patterns |
| H-INT-3 | M-P4.10 | `gcp_secret_manager` | secrets_store | **P0** | **Done** (beta) | GCP prod parity | ADC / SA via `gcp` patterns |
| H-INT-3 | M-P4.11 | `doppler` | secrets_store | **P1** | **Done** (beta) | Dev/prod secret sync for harness authors | Project/config scoped fetch; lab `.env` bridge |
| H-INT-4 | M-P4.12 | `unleash` | feature_flag | **P0** | **Done** (beta) | Gradual `AdaptiveProfile` rollout (observe→recommend) | Requires **M-P4-CAT.1**; tenant-scoped flags |
| H-INT-4 | M-P4.13 | `launchdarkly` | feature_flag | **P1** | **Done** (beta) | Enterprise feature flags + canary | Requires **M-P4-CAT.1** |
| H-INT-4 | M-P4.14 | `github_actions` | ci_cd | **P0** | **Done** (beta) | Harness release gate status; `harness-release.yml` evidence | Requires **M-P4-CAT.2**; workflow run + check suite read |
| H-INT-4 | M-P4.15 | `redpanda` | message_bus | **P1** | **Done** (beta) | Kafka-compatible async `AdaptationScheduler` / pattern miner | Lab compose; consumer/producer contract tests |
| H-INT-4 | M-P4.16 | `cloudflare_r2` | object_storage | **P1** | **Done** (beta) | S3-compatible cheap eval/adaptive artifacts | `ObjectStorage` put/get; reuse S3 adapter patterns |
| H-INT-5 | M-P4.17 | `memgraph` | graph_store | **P1** | **Done** (beta) | GraphRAG alternative; lighter lab footprint | Integration `GraphStore` contract; RAG adapter **Planned** M-RAG.39 |
| H-INT-5 | M-P4.18 | `falkordb` | graph_store | **P2** | **Done** (beta) | Redis-module graph — reuse lab `redis` stack | Bolt/Redis adapter; RAG adapter **Planned** M-RAG.39 |
| H-INT-5 | M-P4.19 | `incident_io` | notification_channel | **P1** | **Done** (beta) | Ops runbooks (`runbook/adaptive/*`) → real incidents | Outbound incident create; HITL escalation path |
| H-INT-5 | M-P4.20 | `kubernetes` | cloud_platform | **P1** | **Done** (beta) | Prod harness host deploy; health probes at scale | Extend `CloudPlatform` — scale API: [ECP-4.*](plan/ELASTIC_CAPACITY_AND_SCALING.md) |
| H-INT-5 | M-P4.21 | `servicenow` | issue_tracker | **P2** | **Done** (beta) | Enterprise change approval for policy learning | `IssueTracker` search/get; HITL change ticket |
| H-INT-5 | M-P4.22 | `bitbucket` | issue_tracker | **P2** | **Done** (beta) | Atlassian stack beside `jira` | REST issues/PRs |
| H-INT-5 | M-P4.23 | `asana` | issue_tracker | **P2** | **Done** (beta) | PM human task queue beside `linear` | Task search/create |
| H-INT-5 | M-P4.24 | `sendgrid` | notification_channel | **P2** | **Done** (beta) | Deliverability beyond raw `email_smtp` | Transactional send API |
| H-INT-5 | M-P4.25 | `mailgun` | interaction_surface | **P2** | **Done** (beta) | Inbound email → interaction intake | Webhook verify + payload normalize |
| H-INT-5 | M-P4.26 | `mlflow` | observability_backend | **P2** | **Done** (beta) | Experiment tracking beside wandb/braintrust | Run/metric log API; lab workflow §35 |
| H-INT-5 | M-P4.27 | `huggingface_hub` | object_storage | **P2** | **Done** (beta) | W-ML model artifact pull (ONNX/YOLO) | Model file get/list; modality plane bridge |
| H-INT-5 | M-P4.28 | `ollama` | interaction_surface | **P2** | **Done** (beta) | Local inference host (`infra/integration` ollama service) | Health probe + model list; cross-link [architecture/MODALITY.md](architecture/MODALITY.md) · not LLM catalog slug |

**Explicitly excluded from M.6 P4:** CRM (Salesforce, HubSpot), payment rails, blockchain, duplicate vector SaaS, LLM vendor APIs.

##### M.6 P4 — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-02 | M-P4.0 | Register 28 harness-ROI integration slugs + §6.1w + §6.2ae + Band 2aa (audit → plan) |
| 2026-06-02 | M-P4.1–M-P4.28 | All 28 M.6 P4 providers: `_shared/p5/`, layout **127**, tests `test_p5_m6_p4_providers.py`, gate green |
| 2026-06-02 | M-P4.FU | Tier-3 follow-up (no business agents): `harness_production_stack` / `harness_production_defaults`, lab env (`LAB_OBSERVABILITY_GRAFANA_STACK`, `LAB_ADAPTIVE_FEATURE_FLAG`, `LAB_SECRETS_BACKEND`), adaptive feature-flag gate, pgvector persistence + health, M6 P4 stable promotion (8 slugs), `health_check_harness_m6_p4_probes`, docs sync |
| 2026-06-02 | M-P4.FU.2 | Adaptive runtime bridge uses gated `wiring.profile`; debug `GET /debug/integrations/health`; remove `getattr` from P5 health probes (`IntegrationHealthProbe`); W-OPS integration health debug gate; gate **790** |

#### M.6 P5 — Harness integration depth (Done — 33/34)

**Deferred:** `trivy` — absorbed into **M.6 P6** [M-P6.1](#m6-p6--master-register-32-slugs) with `security_scanner` category (**M-P6-CAT.1**).

**Delivered (2026-06-02):**

- `_shared/p6/factories.py` — 8 greenfield harness slugs
- `bootstrap_m6_p5.py` + `layout.py` (+8 slugs → **135** catalog slugs)
- Health probes on harden adapters; **STABLE** promotion (25 slugs)
- Tier-3 presets: `harness_metrics_stack`, `harness_eval_stack`, `harness_async_stack`, `harness_ci_stack`
- `HARNESS_M6_P5_PROBE_SLUGS` + `health_check_harness_m6_p5_probes()` + debug API `stack=m6_p5`
- `integrations-pick` presets: `harness_metrics`, `harness_eval`, `harness_async`, `harness_ci`
- Tests: `tests/unit/integrations/providers/test_p6_m6_p5_providers.py`

#### M.6 P5 — Harness integration depth (register archive)

**Status:** **Done** (2026-06-02) — **33/34** · catalog **135** slugs in layout.py (**136** when `trivy` ships)  
**Source:** Harness integration re-audit (2026-06-02) — post M.6 P4 follow-up  
**Queue:** [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-planned) · **Execution order:** [§6.2af](#62af-phase-m6-p5-execution-order-band-2ab--planned)  
**Priority ladder:** **Band 2ab** (§4.0) — runs **in parallel** with §6.1 maintenance; **does not** unblock Band 3 product work

**Scope split:**

| Kind | Count | Meaning |
|------|-------|---------|
| **Harden** | 25 | Slug already in catalog (`beta`) — health probe, STABLE promotion, harness preset wiring, tests |
| **Greenfield** | 9 | New slug + provider tree — same M.4 workflow as P4 |

**Hard rules (inherit M.6 P4):**

- **No** LLM vendor API slugs — use `llm_adapters/` (canon §7.1.2).
- **No** CRM, payments, blockchain, or duplicate vector SaaS without explicit harness ROI.
- Reuse `_shared/p5/` HTTP patterns or existing provider packages — **do not** fork RAG/runtime stores.
- One slug (or one harden wave) per PR; gate green after each.
- `infra/integration` Docker profile must be documented in slug `USAGE.md` when a local service exists.

**New category proposals (M-P5-CAT):**

| ID | Category | Slugs | Status | Acceptance |
|----|----------|-------|--------|------------|
| M-P5-CAT.1 | `ci_cd` (extend) | `gitlab_ci`, `circleci`, `azure_pipelines`, `codecov` | **Done** | Read-only workflow/check/coverage APIs on existing `CiCdBackend` |
| M-P5-CAT.2 | `security_scanner` *(proposed)* | `trivy` | **Deferred** | `SecurityScannerBackend` with `scan_image(ref) -> ScanReport`; canon §5.2.4 review before merge |
| M-P5-CAT.3 | — *(use existing)* | `mailpit`, `localstack`, `grafana_oncall`, `opentelemetry_collector` | **Done** | Map to existing categories (`notification_channel`, `cloud_platform`, `notification_channel`, `observability_backend`) |

**Tier-3 named presets (deliver with H-INT-6 closeout):**

| Preset function | Slugs (primary) | Harness use |
|-----------------|-----------------|-------------|
| `harness_metrics_stack()` | `prometheus` + `grafana` + `otel` | W-OPS.4 SLO / metrics-first lab |
| `harness_eval_stack()` | `langfuse` + `minio` + `duckdb` | EVAL export + experiment traces |
| `harness_async_stack()` | `redpanda` or `kafka` + `redis` + optional `temporal` | W-ADAPT async / long-running |
| `harness_ci_stack()` | `github_actions` + `gitlab_ci` + optional `circleci` | Multi-CI release evidence |

##### M.6 P5 — Master register (34 slugs)

| Wave | ID | Slug | Category | Kind | Priority | Status | Harness ROI | Acceptance |
|------|-----|------|----------|------|----------|--------|-------------|------------|
| H-INT-6 | M-P5.1 | `prometheus` | observability_backend | harden | **P0** | **Done** | Metrics SLO backbone (W-OPS.4); complements Grafana stack | Health probe; `harness_metrics_stack`; infra `:9090` |
| H-INT-6 | M-P5.2 | `clickhouse` | observability_backend | harden | **P0** | **Done** | OLAP eval/adaptive trends at scale | Query adapter; infra `:8123` |
| H-INT-6 | M-P5.3 | `vault` | secrets_store | harden | **P0** | **Done** | Prod secrets alt in `harness_production_stack` | Health probe; STABLE; infra `:8200` |
| H-INT-6 | M-P5.4 | `pagerduty` | notification_channel | harden | **P0** | **Done** | HITL / incident escalation (tool already wired) | Integration health + lab smoke |
| H-INT-6 | M-P5.5 | `github` | issue_tracker | harden | **P0** | **Done** | PR/issue context for release board | Read API; links to `github_actions` evidence |
| H-INT-6 | M-P5.6 | `gitlab_ci` | ci_cd | greenfield | **P0** | **Done** | GitLab pipeline status for harness release | **M-P5-CAT.1**; `CiCdBackend` read |
| H-INT-6 | M-P5.7 | `circleci` | ci_cd | greenfield | **P0** | **Done** | Multi-CI release evidence | **M-P5-CAT.1** |
| H-INT-6 | M-P5.8 | `azure_pipelines` | ci_cd | greenfield | **P0** | **Done** | Azure DevOps CI parity | **M-P5-CAT.1**; pairs with `azure_devops` issue tracker |
| H-INT-6 | M-P5.9 | `mailpit` | notification_channel | greenfield | **P0** | **Done** | Local SMTP/HITL without SaaS | Infra `:1025`/`:8025`; email capture tests |
| H-INT-6 | M-P5.10 | `localstack` | cloud_platform | greenfield | **P0** | **Done** | S3/SQS/DynamoDB smoke in CI | Infra `:4566`; pairs with `s3`/`sqs`/`dynamodb` slugs |
| H-INT-7 | M-P5.11 | `langfuse` | observability_backend | harden | **P0** | **Done** | LLM trace + eval export (EVAL/W-ADAPT) | Infra `:3000`; `harness_eval_stack` |
| H-INT-7 | M-P5.12 | `phoenix` | observability_backend | harden | **P0** | **Done** | Arize OSS trace UI for lab | Infra `:6006` |
| H-INT-7 | M-P5.13 | `braintrust` | observability_backend | harden | **P1** | **Done** | Online eval registry bridge | API read + export hook |
| H-INT-7 | M-P5.14 | `mlflow` | observability_backend | harden | **P1** | **Done** | Experiment tracking (M.6 P4 beta hardening) | STABLE promotion path |
| H-INT-7 | M-P5.15 | `influxdb` | observability_backend | harden | **P1** | **Done** | Adaptive KPI time-series (M.6 P4 beta) | STABLE + W-ADAPT optional export |
| H-INT-7 | M-P5.16 | `timescaledb` | relational_store | harden | **P1** | **Done** | Eval/adaptive hypertables on Postgres | Extends `postgresql` patterns |
| H-INT-7 | M-P5.17 | `temporal` | message_bus | harden | **P1** | **Done** | Long-running harness workflows | Infra `heavy` profile `:7233` |
| H-INT-7 | M-P5.18 | `redpanda` | message_bus | harden | **P1** | **Done** | Kafka-compat async adaptive bus (M.6 P4 beta) | STABLE + `harness_async_stack` |
| H-INT-7 | M-P5.19 | `minio` | object_storage | harden | **P1** | **Done** | Local S3 for eval/adaptive artifacts | Infra `:9000`; preset with `harness_eval_stack` |
| H-INT-7 | M-P5.20 | `s3` | object_storage | harden | **P1** | **Done** | Prod checkpoint/eval blob store | `harness_production_stack` option |
| H-INT-8 | M-P5.21 | `neo4j` | graph_store | harden | **P1** | **Done** | GraphRAG harness eval | Infra `:7687`; health probe |
| H-INT-8 | M-P5.22 | `mongodb` | document_store | harden | **P1** | **Done** | MEM platform JSON artifacts | Infra `:27017` |
| H-INT-8 | M-P5.23 | `elasticsearch` | observability_backend | harden | **P1** | **Done** | Log search for RuntimeEvents | Infra `:9200` |
| H-INT-8 | M-P5.24 | `nats` | message_bus | harden | **P2** | **Done** | Lightweight async bus | Infra `:4222` |
| H-INT-8 | M-P5.25 | `chroma` | vector_store | harden | **P2** | **Done** | RAG lab alternative | Infra `:8000`; thin RAG bridge |
| H-INT-8 | M-P5.26 | `weaviate` | vector_store | harden | **P2** | **Done** | RAG lab alternative | Infra `:8080` |
| H-INT-8 | M-P5.27 | `launchdarkly` | feature_flag | harden | **P2** | **Done** | Enterprise canary beside Unleash | Adaptive gate smoke |
| H-INT-8 | M-P5.28 | `signoz` | observability_backend | harden | **P2** | **Done** | Self-hosted OTEL APM | Optional Grafana stack alt |
| H-INT-9 | M-P5.29 | `codecov` | ci_cd | greenfield | **P2** | **Done** | Coverage gate in release evidence | **M-P5-CAT.1** |
| H-INT-9 | M-P5.30 | `trivy` | security_scanner | greenfield | **P2** | **→ M-P6.1** | Image/SBOM scan before STABLE promote | Absorbed into [M.6 P6](#m6-p6--harness-integration-expansion-planned) (**M-P6-CAT.1**) |
| H-INT-9 | M-P5.31 | `grafana_oncall` | notification_channel | greenfield | **P2** | **Done** | On-call beside Grafana stack | Webhook/API incident create |
| H-INT-9 | M-P5.32 | `opentelemetry_collector` | observability_backend | greenfield | **P2** | **Done** | Collector admin/health (export via `otel`) | Distinct from app OTEL export slug |
| H-INT-9 | M-P5.33 | `snowflake` | relational_store | harden | **P2** | **Done** | Enterprise eval analytics | Existing beta hardening only |
| H-INT-9 | M-P5.34 | `supabase` | relational_store | harden | **P2** | **Done** | Postgres+auth lab shortcut | Existing beta hardening only |

**Explicitly excluded from M.6 P5:** CRM (Salesforce, HubSpot), payment rails, blockchain, `vespa`/`selenium` (heavy lab only), `servicenow`/`asana`/`notion`/`sharepoint`/`google_workspace` (business PM/collab), duplicate vector SaaS without infra smoke (`pinecone`, `milvus` until explicitly requested).

**Per-slug checklist (harden):** health probe → STABLE promotion → harness preset slot (if applicable) → `HARNESS_M6_P5_PROBE_SLUGS` or W-OPS extension → gate green → paydown log row.

**Per-slug checklist (greenfield):** contract/category gate → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → bootstrap register → gate green → paydown log row.

##### M.6 P5 — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-02 | M-P5.0 | Register 34 harness-depth slugs from integration re-audit; §6.1x + §6.2af + Band 2ab |
| 2026-06-02 | M-P5.1–34 | Implement 33/34 slugs: health + STABLE harden, p6 greenfield, presets, W-OPS probes; `trivy` deferred (M-P5-CAT.2) |
| 2026-06-02 | M-P5.FU | W-OPS `harness_m6_p5_health_gate`; `IntegrationBinding` JSON dict roundtrip fix; register status sync |

#### M.6 P6 — Harness integration expansion (Done — 32/32)

**Status:** **Done** (2026-06-02) — **32/32** · catalog **167** slugs in layout.py  
**Source:** Harness integration gap audit (2026-06-02) — post M.6 P5; all **32** proposed slugs registered below (includes `trivy` migrated from M-P5.30, plus `modal`, `daytona`, `workos`, `hubspot` from audit waves)  
**Queue:** [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-done) · **Execution order:** [§6.2ag](#62ag-phase-m6-p6-execution-order-band-2ac--done)  
**Priority ladder:** **Band 2ac** (§4.0) — runs **in parallel** with §6.1 maintenance; **does not** unblock Band 3 product work

**Scope:** **32 greenfield** slugs — new provider trees + category contracts where noted. No business-agent logic.

**Hard rules (inherit M.6 P4/P5):**

- **No** LLM vendor API slugs — use `llm_adapters/` (canon §7.1.2).
- Reuse `_shared/p6/` / `_shared/p7/` HTTP patterns — **do not** fork RAG/runtime stores.
- One slug (or one category CAT wave) per PR; gate green after each.
- `infra/integration` Docker profile documented in slug `USAGE.md` when a local service exists.
- **`salesforce` / `hubspot` / `stripe`:** harness-platform slugs only (metering, CRM context for support agents) — **not** Band 3 product agents.

**New category proposals (M-P6-CAT — canon §5.2.4 review before first slug in category):**

| ID | Category | Slugs | Status | Acceptance |
|----|----------|-------|--------|------------|
| M-P6-CAT.1 | `security_scanner` | `trivy`, `snyk`, `semgrep` | **Done** | `SecurityScannerBackend`: `scan_image(ref)`, `scan_repo(path)` → `ScanReport`; completes **M-P5-CAT.2** |
| M-P6-CAT.2 | `sandbox_host` | `e2b`, `modal`, `daytona` | **Done** | `SandboxHostBackend`: `create_session()`, `exec()`, `upload_artifact()`; bridges Tier-1 `sandbox.exec` tool |
| M-P6-CAT.3 | `identity_provider` | `auth0`, `keycloak`, `workos` | **Done** | `IdentityProviderBackend`: `verify_token()`, `userinfo()`, optional `list_tenants()` |
| M-P6-CAT.4 | `speech_provider` | `elevenlabs`, `deepgram` | **Done** | `SpeechProviderBackend`: TTS/STT; unifies `speech_adapters/` with Integration Library ([architecture/MODALITY.md](architecture/MODALITY.md)) |
| M-P6-CAT.5 | `workflow_orchestrator` | `prefect`, `airflow` | **Done** | `WorkflowOrchestratorBackend`: trigger run, poll status, fetch logs (eval/RAG batch jobs) |
| M-P6-CAT.6 | `vision_serving` | `triton` | **Done** | Remote CV inference host ([architecture/MODALITY.md](architecture/MODALITY.md) W-ML.4) |
| M-P6-CAT.7 | `ml_inference_host` | `replicate` | **Done** | Managed model endpoint (`predict`, health) |
| M-P6-CAT.8 | `billing_meter` | `stripe` | **Done** | Usage metering hook for harness SaaS path (canon §50 future) |
| M-P6-CAT.9 | `crm` | `salesforce`, `hubspot` | **Done** | Read-only CRM context (accounts, contacts, tickets) for support harness agents |

**Tier-3 named presets (deliver with H-INT-10 closeout or M-P6-PRE.1):**

| Preset function | Slugs (primary) | Harness use |
|-----------------|-----------------|-------------|
| `harness_security_stack()` | `trivy` + `semgrep` + optional `snyk` | STABLE promote gate + V-SEC repo policy |
| `harness_sandbox_stack()` | `e2b` + optional `modal` | Cloud `sandbox.exec` for lab/product hosts |
| `harness_identity_stack()` | `keycloak` (lab) or `auth0` (prod) | Multi-tenant debug API / host auth |
| `harness_gitops_stack()` | `argocd` + `github_actions` | Agent host deploy after eval gate |

##### M.6 P6 — Master register (32 slugs)

| Wave | ID | Slug | Category | Priority | Status | Harness ROI | Acceptance |
|------|-----|------|----------|----------|--------|-------------|------------|
| H-INT-10 | M-P6.1 | `trivy` | security_scanner | **P0** | **Done** | Image/SBOM scan before STABLE promote | **M-P6-CAT.1**; migrates M-P5.30 |
| H-INT-10 | M-P6.2 | `snyk` | security_scanner | **P0** | **Done** | SAST/SCA in agent pack promotion pipeline | **M-P6-CAT.1** |
| H-INT-10 | M-P6.3 | `semgrep` | security_scanner | **P0** | **Done** | Policy-as-code on agents/skills repos | **M-P6-CAT.1** |
| H-INT-10 | M-P6.4 | `infisical` | secrets_store | **P0** | **Done** | Dev-friendly secrets sync (lab + prod) | Health probe; pairs with `harness_production_stack` |
| H-INT-11 | M-P6.5 | `e2b` | sandbox_host | **P0** | **Done** | Cloud isolation for `sandbox.exec` | **M-P6-CAT.2**; sandbox tool bridge |
| H-INT-11 | M-P6.6 | `modal` | sandbox_host | **P1** | **Done** | Serverless agent/compute workloads | **M-P6-CAT.2** |
| H-INT-11 | M-P6.7 | `daytona` | sandbox_host | **P1** | **Done** | Dev environment sandbox alternative | **M-P6-CAT.2** |
| H-INT-12 | M-P6.8 | `auth0` | identity_provider | **P0** | **Done** | SaaS OIDC for multi-tenant harness hosts | **M-P6-CAT.3** |
| H-INT-12 | M-P6.9 | `keycloak` | identity_provider | **P0** | **Done** | Self-hosted OIDC (VPC customers) | **M-P6-CAT.3**; infra optional |
| H-INT-12 | M-P6.10 | `workos` | identity_provider | **P1** | **Done** | Enterprise SSO + directory sync | **M-P6-CAT.3** |
| H-INT-13 | M-P6.11 | `argocd` | ci_cd | **P0** | **Done** | GitOps deploy Tier-3 hosts after eval gate | Read API; `harness_gitops_stack` |
| H-INT-13 | M-P6.12 | `buildkite` | ci_cd | **P1** | **Done** | Eval-before-merge pipelines | Extends `CiCdBackend` |
| H-INT-13 | M-P6.13 | `jenkins` | ci_cd | **P1** | **Done** | Enterprise CI parity | Extends `CiCdBackend` |
| H-INT-14 | M-P6.14 | `elevenlabs` | speech_provider | **P0** | **Done** | TTS catalog slug; bridges `speech_adapters/` | **M-P6-CAT.4**; `speech.synthesize` tool |
| H-INT-14 | M-P6.15 | `deepgram` | speech_provider | **P0** | **Done** | STT for HITL voice + audio RAG ingest | **M-P6-CAT.4**; `speech.transcribe` tool |
| H-INT-15 | M-P6.16 | `newrelic` | observability_backend | **P1** | **Done** | APM gap beside Datadog/Honeycomb | Health + query API |
| H-INT-15 | M-P6.17 | `splunk` | observability_backend | **P1** | **Done** | Enterprise log search (RuntimeEvents export) | Search adapter |
| H-INT-15 | M-P6.18 | `zendesk` | issue_tracker | **P1** | **Done** | Support tickets → agent tasks / HITL | Read/create ticket API |
| H-INT-15 | M-P6.19 | `statsig` | feature_flag | **P1** | **Done** | Agent experiment gates beside Unleash/LD | Adaptive canary smoke |
| H-INT-16 | M-P6.20 | `prefect` | workflow_orchestrator | **P1** | **Done** | Batch eval / dataset refresh orchestration | **M-P6-CAT.5** |
| H-INT-16 | M-P6.21 | `airflow` | workflow_orchestrator | **P1** | **Done** | Data-eng standard for RAG reindex jobs | **M-P6-CAT.5** |
| H-INT-16 | M-P6.22 | `typesense` | vector_store | **P1** | **Done** | Fast hybrid search lab backend | Thin RAG bridge + health |
| H-INT-16 | M-P6.23 | `neon` | relational_store | **P1** | **Done** | Serverless Postgres for trace/eval lab | Extends `postgresql` patterns |
| H-INT-16 | M-P6.24 | `pulsar` | message_bus | **P1** | **Done** | Multi-tenant streaming bus | Infra optional |
| H-INT-17 | M-P6.25 | `algolia` | search_provider | **P2** | **Done** | SaaS search for product agents | Search API adapter |
| H-INT-17 | M-P6.26 | `confluent` | message_bus | **P2** | **Done** | Managed Kafka for enterprise event bus | Pairs with `kafka` slug |
| H-INT-17 | M-P6.27 | `backblaze_b2` | object_storage | **P2** | **Done** | Low-cost eval/shadow-workspace artifacts | S3-compat API |
| H-INT-17 | M-P6.28 | `triton` | vision_serving | **P2** | **Done** | Remote CV inference (W-ML.4) | **M-P6-CAT.6** |
| H-INT-17 | M-P6.29 | `replicate` | ml_inference_host | **P2** | **Done** | Hosted models without lab GPU | **M-P6-CAT.7** |
| H-INT-17 | M-P6.30 | `stripe` | billing_meter | **P2** | **Done** | Usage metering for future harness SaaS | **M-P6-CAT.8**; read-only meter events |
| H-INT-17 | M-P6.31 | `salesforce` | crm | **P2** | **Done** | Enterprise CRM context (support agents) | **M-P6-CAT.9**; read-only |
| H-INT-17 | M-P6.32 | `hubspot` | crm | **P2** | **Done** | SMB CRM context (support agents) | **M-P6-CAT.9**; read-only |

**Explicitly excluded from M.6 P6:** LLM vendor slugs; blockchain; duplicate thin observability without tool surface; `pinecone`/`milvus` until explicitly requested; Band 3 business agent implementations inside provider packages.

**Per-slug checklist (greenfield):** category CAT gate (if new) → contract → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → bootstrap register → optional preset/probe → gate green → paydown log row.

##### M.6 P6 — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-02 | M-P6.0 | Register **32** harness-expansion slugs from integration gap audit; §6.1y + §6.2ag + Band **2ac** |
| 2026-06-02 | M-P6-WIRE | Post-catalog closeout: Tier-1 tools (`security.scan`, `workflow.*`), `HostedSandboxSession` bridge, `IntegrationSpeechAdapter`, `wire_application_identity`, V-SEC promote gate script, infra `p6` profile, CI hook |

#### M.7 P7 — Agent-developer integration expansion (Done — 18/18)

**Status:** **Done** (2026-06-08) — **18/18** · catalog **185** slugs in `layout.py`  
**Source:** Integration audit for Tier-2 agent authors (2026-06-08)  
**Queue:** [§6.1z](#61z-harness-implementation-queue--agent-developer-expansion-m7-p7-done)  
**Priority ladder:** **Band 2ad** — runs in parallel with §6.1 maintenance  
**Implementation:** `intergrax/integrations/_shared/p8/` + thin shells via `scripts/wire_p8_m7_p7_providers.py` · `register_m7_p7_integrations()` in `bootstrap_m7_p7.py`

**Hard rules:**

- **No** LLM vendor API slugs — use `llm_adapters/`.
- Reuse existing category contracts — **no** new universal mechanisms.
- `telegram` dual-role slug (`notification_channel` + `interaction_surface`) like `slack`.
- Auto-enable catalog tools when profile declares `search_provider`, `document_parser`, or `vector_store`.

##### M.7 P7 — Master register (18 slugs)

| Wave | ID | Slug | Category | Priority | Status | Agent-dev ROI | Acceptance |
|------|-----|------|----------|----------|--------|---------------|------------|
| H-INT-P7-1 | M-P7.1 | `perplexity` | search_provider | **P0** | **Done** | AI-native research beside `tavily`/`exa` | `websearch.query` auto-wire |
| H-INT-P7-1 | M-P7.2 | `arxiv` | search_provider | **P0** | **Done** | Academic paper search | STABLE + conformance |
| H-INT-P7-1 | M-P7.3 | `semantic_scholar` | search_provider | **P0** | **Done** | Citation-aware research | STABLE + conformance |
| H-INT-P7-1 | M-P7.4 | `llamaparse` | document_parser | **P0** | **Done** | High-quality PDF/table ingest | `rag.ingest_document` auto-wire |
| H-INT-P7-1 | M-P7.5 | `lancedb` | vector_store | **P0** | **Done** | Embedded local vector RAG | `rag.retrieve` auto-wire |
| H-INT-P7-2 | M-P7.6 | `telegram` | notification + interaction | **P0** | **Done** | Chat bot intake/outbound | dual-category catalog factory |
| H-INT-P7-2 | M-P7.7 | `browserbase` | browser_automation | **P0** | **Done** | Managed browser sessions | `browser.fetch_page` |
| H-INT-P7-2 | M-P7.8 | `google_drive` | object_storage | **P0** | **Done** | Cloud document source | `storage.*` tools |
| H-INT-P7-2 | M-P7.9 | `apify` | browser_automation | **P1** | **Done** | Structured web scraping | conformance tests |
| H-INT-P7-3 | M-P7.10 | `n8n` | workflow_orchestrator | **P0** | **Done** | Low-code automation triggers | `workflow.*` tools |
| H-INT-P7-3 | M-P7.11 | `wikipedia` | wiki_knowledge | **P0** | **Done** | Free structured knowledge | `knowledge.*` tools |
| H-INT-P7-3 | M-P7.12 | `clerk` | identity_provider | **P1** | **Done** | Fast SaaS auth for new hosts | identity tools |
| H-INT-P7-3 | M-P7.13 | `upstash_redis` | key_value_cache | **P1** | **Done** | Serverless Redis | `cache.*` tools |
| H-INT-P7-3 | M-P7.14 | `upstash_qstash` | message_bus | **P1** | **Done** | Serverless queue | `message_bus.*` tools |
| H-INT-P7-4 | M-P7.15 | `okta` | identity_provider | **P1** | **Done** | Enterprise SSO | identity tools |
| H-INT-P7-4 | M-P7.16 | `bigquery` | relational_store | **P1** | **Done** | Warehouse analytics agents | `database.*` tools |
| H-INT-P7-4 | M-P7.17 | `motherduck` | relational_store | **P1** | **Done** | Cloud DuckDB eval trends | extends `duckdb` patterns |
| H-INT-P7-4 | M-P7.18 | `airbyte` | workflow_orchestrator | **P1** | **Done** | RAG reindex orchestration | `workflow.*` tools |

**Tier-3 named presets (M-P7-PRE.1):**

| Preset function | Slugs (primary) | Agent-dev use |
|-----------------|-----------------|---------------|
| `research_web_stack()` | `perplexity` + `wikipedia` + `inmemory` | ResearchAgent / assistant hub |
| `document_ingest_stack()` | `llamaparse` + `lancedb` + `minio`/`google_drive` | Legal/LKW ingest |
| `chat_bot_stack()` | `telegram` + `redis` + `langfuse` | Messenger bot + trace |

**ADR:** no ADR needed — extends existing Integration Library contracts and M.6 factory patterns; no Nexus semantics change.

##### M.7 P7 — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-08 | M-P7.0 | Register 18 agent-developer slugs; §6.1z + Band **2ad** |
| 2026-06-08 | M-P7.1–18 | `_shared/p8`, bootstrap, presets, tool auto-wiring, tests `test_p8_m7_p7_providers.py` |

#### Phase M.12 — LLM guardrail integrations (Done)

**Canon:** [`architecture/INTEGRATIONS.md`](../architecture/INTEGRATIONS.md) §47 · UAEP §42.11.6  
**Goal:** Ship `llm_guardrail` integration category + Tier-1 `guardrail_runtime_bridge` so Tier-3 hosts can select NeMo Guardrails, Guardrails AI, LLM Guard, OpenGuardrails, and complementary scanners without agent SDK imports.

**Status:** **Done** (2026-06-09) — **14/14** deliverables; vendor SDKs optional (pattern/HTTP fallback in CI).

**Policy:** One slug per PR; P0 slugs (`llm_guard`, `guardrails_ai`) before P1 orchestration backends; optional cloud slugs only when product host requires.

##### M.12 — Master register

| ID | Wave | Slug / deliverable | Priority | Status | Package / module | Acceptance |
|----|------|-------------------|----------|--------|------------------|------------|
| M-P12-CAT.1 | CAT | `LlmGuardrailBackend` contract + `IntegrationCategory.LLM_GUARDRAIL` | **P0** | **Done** | `integrations/contracts/llm_guardrail.py` | ADR-GR-001 · `test_llm_guardrail_contract.py` |
| M-P12-CAT.2 | CAT | `IntegrationProfile.llm_guardrail` field + resolver | **P0** | **Done** | `registry/profile.py` | Binding accessor + resolve |
| M-P12.1 | H-INT-GR-1 | `llm_guard` — Protect AI LLM Guard | **P0** | **Done** | `_adapters.py`, `_vendor_opens.py` | Pattern fallback + optional `llm-guard` |
| M-P12.2 | H-INT-GR-2 | `guardrails_ai` — Guardrails AI Hub validators | **P0** | **Done** | `_adapters.py` | Optional `guardrails-ai` import |
| M-P12.3 | H-INT-GR-3 | `nemo_guardrails` — NVIDIA NeMo Guardrails | **P1** | **Done** | `_factory.py` | Pattern adapter; Colang bundle follow-up |
| M-P12.4 | H-INT-GR-4 | `openguardrails` — OpenGuardrails SDK / gateway | **P1** | **Done** | HTTP adapter | `INTERGRAX_OPENGUARDRAILS_*` |
| M-P12.5 | H-INT-GR-5 | `presidio` — Microsoft Presidio PII | **P1** | **Done** | `_vendor_opens.py` | Optional `presidio-analyzer` |
| M-P12.6 | H-INT-GR-6 | `llama_guard` — Meta Llama Guard classifier | **P2** | **Done** | Pattern adapter | Inference host bundle follow-up |
| M-P12.7 | H-INT-GR-7 | `lakera` — Lakera Guard API | **P2** | **Done** | HTTP adapter | `INTERGRAX_LAKERA_*` |
| M-P12.8 | H-INT-GR-8 | `azure_content_safety` | **P2** | **Done** | HTTP adapter | `INTERGRAX_AZURE_CONTENT_SAFETY_*` |
| M-P12.9 | H-INT-GR-9 | `bedrock_guardrails` | **P2** | **Done** | Pattern adapter | Bedrock policy wiring follow-up |
| M-P12-PRE.1 | PRE | `harness_guardrail_stack()` preset | **P0** | **Done** | `registry/presets.py` | `harness_guardrail_stack()` |
| M-P12-WIRE.1 | WIRE | `guardrail_runtime_bridge` + middleware registration | **P0** | **Done** | `application_guardrail_middleware.py`, `guardrail_wiring.py` | `LlmGuardrailMiddleware` on Nexus |
| M-P12-WIRE.2 | WIRE | Extend `security_runtime_bridge` to compose native + vendor | **P1** | **Done** | `security_runtime_bridge.py` | Guardrail slug in `RuntimeConfig.metadata` |
| M-P12-WIRE.3 | CI | `check_harness_guardrail_wiring.py` | **P1** | **Done** | `scripts/` | Import gate green |

**Suggested PR order:** M-P12-CAT.1 → M-P12-CAT.2 → M-P12.1 → M-P12.2 → M-P12-WIRE.1 → M-P12-PRE.1 → remaining slugs.

**Optional dependency group:** `Intergrax-ai[integrations-guardrails]` — `presidio-analyzer`, `presidio-anonymizer` only (torch/docling conflict). Install `llm-guard`, `guardrails-ai`, `nemoguardrails` manually when needed.

**ADR:** [ADR-GR-001](../adr/entries/2026-06-09/ADR-GR-001.md) — Accepted.

##### M.12 — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-09 | M-P12-DOC.1 | Architecture §47 + plan register opened; UAEP GR-DOC cross-ref |
| 2026-06-09 | M-P12.* | Full M.12 implementation: contract, 9 slugs, middleware, assembly, tests, CI |
| 2026-06-09 | M-P12.FU | Follow-up: per-slug bundles, chained scanners, LLM hooks, legal host example, docs paydown |
| 2026-06-09 | M-P12.FU2 | NeMo `opens.py`, HTTP smoke tests, `USAGE.md`, lab guardrail toggle |
| 2026-06-09 | M-P12.HARD | E2E Nexus guardrail gate, `GUARDRAIL_BLOCKED` events, `UAEPBlockedError` → failed execution, PLATFORM Band 2ay |

##### M.6 P6 — Post-catalog wiring closeout (Done — 2026-06-02)

| ID | Deliverable | Status |
|----|-------------|--------|
| M-P6-WIRE.1 | `security.scan` tool + `ToolWiringContext.security_scanner` | **Done** |
| M-P6-WIRE.2 | `workflow.trigger` / `workflow.poll` / `workflow.fetch_logs` + `workflow_orchestrator` wiring | **Done** |
| M-P6-WIRE.3 | `sandbox.exec` → `SandboxHostBackend` via `HostedSandboxSession` | **Done** |
| M-P6-WIRE.4 | Speech catalog → speech tools via `IntegrationSpeechAdapter` | **Done** |
| M-P6-WIRE.5 | Harness OIDC auth via `wire_application_identity()` (lab + generic FastAPI hosts) | **Done** |
| M-P6-WIRE.6 | `check_harness_security_promote_gate.py` (wiring default; optional live scan) | **Done** |
| M-P6-WIRE.7 | Docker profile `p6` (keycloak, typesense, airflow) | **Done** |
| M-P6-WIRE.8 | `extend_tool_profile_for_integration()` + lab MCP P6 wiring + product host identity | **Done** |
| M-P6-OPS.1 | Release CLI security scan + P6 infra E2E script + `harness.reliability_smoke` P6 tools | **Done** |

#### M.6 P3 — Legacy backlog note (superseded)

Slugs below were **already in** `IntegrationSlug` unless marked *proposed*. Prioritize when a product app blocks; otherwise deliver after P2.

| Priority | Slug(s) | Category | Why agents/apps need it |
|----------|---------|----------|-------------------------|
| **High** | `mongodb` | document_store | Session state, flexible agent memory, JSON artifacts at scale |
| **High** | `pinecone`, `qdrant`, `chroma` | vector_store | Production RAG — unify Tier-3 `IntegrationProfile.vector_store` with existing `rag/` backends |
| **High** | `s3`, `azure_blob`, `gcs` | object_storage | Checkpoint blobs, sandbox exports, document ingestion pipelines |
| **High** | `email_smtp` | notification_channel | HITL and report delivery without Slack/Teams |
| **Medium** | `notion`, `sharepoint` | wiki_knowledge | Runbooks and internal docs (Confluence complement) |
| **Medium** | `github`, `linear` | issue_tracker | Dev workflows, PR/issue-aware agents |
| **Medium** | `google_workspace` | collaboration_suite | Google-tenant mail/calendar parity with MS365 |
| **Medium** | `otel` | observability_backend | Export runtime traces/metrics to Grafana Cloud, Datadog, etc. |
| **Medium** | `playwright` | browser_automation | JS-heavy sites, authenticated flows beyond static fetch |
| **Medium** | `brave`, `serpapi` | search_provider | Rate-limit / vendor diversity for research agents |
| **Low** | `oracle`, `mssql`, `azure_sql`, `cloud_sql` | relational_store | Enterprise DB deployments |
| **Low** | `dynamodb`, `memcached`, `elasticache` | document_store / KV | AWS-native persistence tiers |
| **Done (beta)** | `weaviate`, `milvus`, `snowflake`, `vault` | vector_store / relational_store / secrets | `integrations/providers/vector_store/weaviate/`, `vector_store/milvus/`, `relational_store/snowflake/`, `secrets/vault/` |

**Vector-store rule (pinecone / qdrant / chroma):** implementation **stays** in `intergrax/rag/vectorstore/`. Integration Library adds `providers/<slug>/` as a **thin registry adapter**: `opens.py` is the only module that imports vendor SDK; `bundle.create_*_vector_store()` delegates to the existing RAG provider. Tier-3 selects slug via `IntegrationProfile.vector_store`; RAG pipeline code unchanged.

**MongoDB — suggested implementation sketch (greenfield):**

```text
providers/mongodb/
├── config.py                   # INTERGRAX_MONGODB_URI, DATABASE, COLLECTION_PREFIX
├── client.py                   # PyMongo collection wrapper (internal — no driver outside opens.py)
├── adapter.py                  # MongoDocumentStore implements DocumentStore
├── opens.py                    # ONLY place that constructs MongoClient
├── bundle.py                   # create_mongodb_document_store()
├── register.py
└── tests/                      # mocked collection; integration_live optional
```

**Prerequisite (mongodb):** `DocumentStore` contract — **Done** (`contracts/document_store.py`). Partition key maps to MongoDB `_id` or compound `{tenant_id, key}` index.

**Pinecone — suggested implementation sketch (catalog bridge):**

```text
providers/pinecone/
├── config.py                   # INTERGRAX_PINECONE_API_KEY, INDEX, NAMESPACE, ENV
├── adapter.py                  # Thin VectorStore registry facade (delegates to rag/)
├── opens.py                    # ONLY place that imports pinecone SDK / builds Pinecone client
├── bundle.py                   # create_pinecone_vector_store() → rag PineconeVectorStore
├── register.py
└── tests/                      # mocked delegate; guard: no pinecone import outside opens.py
```

**Prerequisite (pinecone):** `contracts/vector_store.py` — **Done** (re-exports `rag/vectorstore/contracts/vector_store.py`). Registered under `IntegrationCategory.VECTOR_STORE`.

**Cassandra — suggested implementation sketch (greenfield):**

```text
contracts/document_store.py     # DocumentStore — get/put/delete/query by partition key
providers/cassandra/
├── config.py                   # INTERGRAX_CASSANDRA_CONTACT_POINTS, KEYSPACE, USER, PASSWORD
├── client.py                   # CQL session (internal — no direct driver import outside opens.py)
├── adapter.py                  # CassandraDocumentStore implements DocumentStore
├── opens.py                    # ONLY place that constructs cassandra driver session
├── bundle.py                   # create_cassandra_integration()
├── register.py
└── tests/                      # testcontainers or mocked session; integration_live optional
```

**Prerequisite (cassandra):** `DocumentStore` contract — **Done** (`contracts/document_store.py`). Runtime event / trace backends remain SQLite-first until an explicit adoption milestone names Cassandra as a target store.

**Elasticsearch — suggested implementation sketch (greenfield):**

```text
providers/elasticsearch/
├── config.py                   # INTERGRAX_ELASTICSEARCH_URL, USER, PASSWORD, INDEX_PREFIX
├── client.py                   # REST search client (internal — no httpx outside opens.py)
├── adapter.py                  # ElasticsearchObservabilityBackend implements ObservabilityBackend
├── opens.py                    # ONLY place that constructs httpx client / ES connection
├── bundle.py                   # create_elasticsearch_observability_backend()
├── register.py
└── tests/                      # mocked _search / ES|QL responses; integration_live optional
```

**Contract note:** start with `ObservabilityBackend` (`query_instant` / `query_range`) mapped to ES\|QL or index-scoped aggregations where feasible; add optional `search_logs(query, *, limit)` on the contract in a follow-up if PromQL-shaped methods prove awkward for log-only clusters.

**Databricks — suggested implementation sketch (greenfield):**

```text
providers/databricks/
├── config.py                   # INTERGRAX_DATABRICKS_HOST, HTTP_PATH, TOKEN, CATALOG, SCHEMA
├── client.py                   # SQL connection wrapper (internal — no driver import outside opens.py)
├── adapter.py                  # DatabricksRelationalStore implements RelationalStore
├── opens.py                    # ONLY place that opens databricks-sql-connector / REST session
├── bundle.py                   # create_databricks_relational_store()
├── register.py
└── tests/                      # mocked cursor / Statement Execution API; integration_live optional
```

**Contract note:** implements existing `RelationalStore` (`connect`, `execute`, `fetch_all`, `close`). Optional `tenant_schema` maps to Unity Catalog ``catalog.schema`` (default schema per connection). Not a replacement for domain runtime stores (SQLite-first) — target is analytics / reporting agents and batch read paths.


1. Create package skeleton:

```text
intergrax/integrations/
├── __init__.py
├── contracts/
│   ├── __init__.py
│   └── base.py              # IntegrationMetadata, HealthStatus, IntegrationError
├── registry/
│   ├── __init__.py
│   ├── catalog.py           # slug → provider entry (lazy import)
│   └── factory.py           # resolve(category, slug | env)
├── _shared/
│   ├── config.py            # pydantic BaseIntegrationConfig
│   └── health.py
└── providers/
    └── .gitkeep
```

2. Add `IntegrationMetadata` dataclass: `slug`, `categories`, `status` (`stable` | `beta` | `deprecated`), `env_prefix`.

3. Register package in `pyproject.toml` / existing import paths (no new top-level dependency unless provider-specific).

#### M.2 — Category contracts (step-by-step)

For each category in §7.1.2, implement a **minimal** Protocol in `integrations/contracts/`:

| Contract | Minimum methods | Notes |
|----------|-----------------|-------|
| `RelationalStore` | `connect()`, `execute()`, `fetch_all()`, `close()` | **Done** — `contracts/relational_store.py`; sqlite/postgresql/mysql/**databricks** (beta) |
| `KeyValueCache` | `get`, `set`, `delete`, `set_if_absent` | Maps to existing `IdempotencyStore` / Redis helpers |
| `MessageBus` | `enqueue`, `get_status`, `get_result` | Re-export / implement `queueing.contracts.TaskQueue` |
| `SearchProvider` | `search(query, *, limit)` → `SearchResult[]` | Align with `websearch/providers/base.py` |
| `NotificationChannel` | `notify(message)` | Align with `runtime/notifications/adapter_contract.py` |
| `InteractionSurface` | `can_handle`, `to_inbound`, `channel` | Align with `runtime/interactions/adapter_contract.py` |
| `CloudPlatform` | `slug`, `default_region`, `resolve(category)`, `health` | **Done** — `contracts/cloud_platform.py`; **`aws`**, **`azure`**, **`gcp`** providers (beta) |
| `CollaborationSuite` | `get_message`, `list_messages`, `send_mail`, `list_calendar_events`, `get_user` | **Done** — `contracts/collaboration_suite.py`; `ms365_graph` provider |
| `DocumentStore` | `get`, `put`, `delete`, `query` (partition-scoped) | **Done** — `contracts/document_store.py`; `cassandra`, **`mongodb`** (beta) providers |
| `VectorStore` | `add_documents`, `query`, `delete`, … | **Done** — `contracts/vector_store.py` re-exports `rag/`; **`pinecone`**, **`qdrant`**, **`chroma`** (beta) |
| `ObjectStorage` | `put`, `get`, `delete`, `presigned_url` | **Done** — `contracts/object_storage.py`; **`s3`** (beta) |
| `IssueTracker` | `get_issue`, `add_comment`, `search_issues` | **Done** — `contracts/issue_tracker.py`; `jira` provider |
| `WikiKnowledge` | `get_page`, `search_pages` | **Done** — `contracts/wiki_knowledge.py`; `confluence` provider |
| `ObservabilityBackend` | `query_instant`, `query_range` | **Done** — `contracts/observability_backend.py`; `prometheus`, **`elasticsearch`** (beta) providers |

**Rule:** if a contract already exists elsewhere, **re-export or inherit** — do not define a third variant.

#### M.3 — IntegrationRegistry (step-by-step)

1. `catalog.py` — static registry:

```python
INTEGRATION_ENTRIES: dict[str, IntegrationEntry] = {
    "sqlite": IntegrationEntry(categories=("relational_store",), factory="..."),
    "redis": IntegrationEntry(categories=("key_value_cache",), factory="..."),
    # ...
}
```

2. `factory.py`:

```python
def resolve(category: str, slug: str | None = None, *, config: Mapping[str, Any] | None = None) -> Any:
    """slug defaults from env INTERGRAX_INTEGRATION_<CATEGORY> or IntegrationProfile."""
```

3. `IntegrationProfile` — pydantic model loaded from env or YAML in Tier-3 `settings.py`.

4. `health_check_all(profile)` — optional startup probe for lab/production.

#### M.4 — Adding a new provider (checklist for implementers)

Copy this checklist into every `providers/<slug>/README.md`:

```text
[ ] 1. Pick category contract(s) from integrations/contracts/
[ ] 2. Create providers/<slug>/ with adapter.py, config.py, config.example.yaml
[ ] 3. Implement contract — no business logic, no Nexus imports
[ ] 4. Register slug in registry/catalog.py
[ ] 5. Add unit tests with fakes or testcontainers (default: no live vendor)
[ ] 6. Optional: pytest -m integration_live with CI secrets
[ ] 7. Wire in one Tier-3 application as reference (lab or product)
[ ] 8. Update canon §7.1.3 status column
```

**Example — wrapping existing Redis idempotency store:**

```text
providers/redis/
├── adapter.py       # RedisKeyValueCache implements KeyValueCache
├── config.py        # REDIS_URL, REDIS_PREFIX
└── tests/
    └── test_redis_cache.py  # fakeredis or mock
```

Delegate to `intergrax/distributed/providers/redis_idempotency_store.py` internally.

**Example — new Jira provider (greenfield):**

```text
providers/jira/
├── adapter.py       # JiraIssueTracker implements IssueTracker
├── config.py        # JIRA_BASE_URL, JIRA_API_TOKEN
├── config.example.yaml
├── README.md
└── tests/
    └── test_jira_issue_tracker.py  # responses mocked from fixtures/
```

Expose agent tools via Tier-0 tool registration (`jira.get_issue`, `jira.create_comment`) — ToolRuntime policy in Tier-1.

#### M.4b — Cloud platform providers (aws / azure / gcp)

Each platform folder exposes **one auth entry point** and registers sub-service slugs:

```text
providers/aws/
├── adapter.py       # CloudPlatform: IAM profile, region, resolve("object_storage") → S3
├── config.py        # AWS_REGION, AWS_PROFILE, AWS_ROLE_ARN
├── services/        # thin wrappers delegating to category contracts
│   ├── s3.py
│   ├── sqs.py
│   └── dynamodb.py
└── tests/

providers/azure/
├── adapter.py       # Managed identity + service principal
├── services/
│   ├── blob.py
│   └── service_bus.py
└── ...

providers/gcp/
├── adapter.py       # ADC + service account
├── services/
│   ├── gcs.py
│   └── pubsub.py
└── ...
```

**Checklist:** implement infrastructure services (S3, SQS, Blob, GCS, Pub/Sub, …) only. LLM wiring stays in `intergrax/llm_adapters/` — do not register Bedrock, Azure OpenAI, or Vertex under `integrations/`.

#### M.5 — Migration map (legacy → catalog)

| Legacy location | Target slug | Action |
|-----------------|-------------|--------|
| `distributed/providers/redis_kv_store.py` (+ siblings) | `redis` | **Done** — single entry `integrations/providers/key_value_cache/redis/create_redis_integration()` |
| `queueing/providers/kafka/` | `kafka` | **Done** — runtime transport + tests delegate to `integrations/providers/message_bus/kafka/` |
| `queueing/providers/celery/` | `celery` | **Done** — `integrations/providers/message_bus/celery/create_celery_integration()` |
| `queueing/providers/rabbitmq/` | `rabbitmq` | **Done** — runtime transport + tests delegate to `integrations/providers/message_bus/rabbitmq/` |
| `websearch/providers/google_cse_provider.py` | `google_cse` | **Done** — `integrations/providers/search_provider/google_cse/create_google_cse_integration()` |
| `websearch/providers/bing_provider.py` | `bing` | **Done** — `integrations/providers/search_provider/bing/create_bing_integration()` |
| `runtime/notifications/adapters/webhook_adapter.py` | `webhook` | **Done** — `integrations/providers/notification_channel/webhook/create_webhook_integration()` |
| `runtime/notifications/adapters/logging_adapter.py` | `log` | **Done** — `integrations/providers/notification_channel/log/`; factory delegates |
| `runtime/notifications/adapters/` | `slack`, `teams` | **Done** — runtime delegates |
| `runtime/interactions/adapters/lab_json_adapter.py` | `lab_json` | **Done** — `integrations/providers/interaction_surface/lab_json/create_lab_json_integration()` |
| `runtime/*/stores/sqlite_*.py` (+ store openers) | `sqlite` | **Done** — single entry `integrations/providers/relational_store/sqlite/create_sqlite_integration()` |
| (new) | `postgresql` | **Done** — `integrations/providers/relational_store/postgresql/`; **only** `opens.py` calls `psycopg.connect` |
| (new) | `mysql` | **Done** — `integrations/providers/relational_store/mysql/`; **only** `opens.py` calls `pymysql.connect` |
| (new) | `jira` | **Done** — `integrations/providers/issue_tracker/jira/`; **only** `opens.py` creates httpx client |
| (new) | `confluence` | **Done** — `integrations/providers/wiki_knowledge/confluence/`; **only** `opens.py` creates httpx client |
| (new) | `prometheus` | **Done** — `integrations/providers/observability_backend/prometheus/`; **only** `opens.py` creates httpx client |
| (new) | `ms365_graph` | **Done** — `integrations/providers/collaboration_suite/ms365_graph/`; **only** `opens.py` creates httpx client + token fetch |
| (new) | `cassandra` | **Done** — `integrations/providers/document_store/cassandra/`; **only** `opens.py` creates driver session |
| (new) | `aws` | **Done** — `integrations/providers/cloud_platform/aws/`; **only** `opens.py` creates boto3 session |
| (new) | `azure` | **Done** — `integrations/providers/cloud_platform/azure/`; **only** `opens.py` creates Azure credential |
| (new) | `gcp` | **Done** — `integrations/providers/cloud_platform/gcp/`; **only** `opens.py` creates Google credentials |
| (new) | `elasticsearch` | **Done** — `integrations/providers/observability_backend/elasticsearch/`; **only** `opens.py` creates httpx client |
| (new) | `databricks` | **Done** — `integrations/providers/relational_store/databricks/`; **only** `opens.py` calls `databricks.sql.connect` |
| (new) | `mongodb` | **Done** — `integrations/providers/document_store/mongodb/`; **only** `opens.py` calls `pymongo.MongoClient` |
| `rag/vectorstore/providers/pinecone_*` | `pinecone` | **Done** — `providers/pinecone/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/providers/qdrant_*` | `qdrant` | **Done** — `providers/qdrant/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/providers/chroma_*` | `chroma` | **Done** — `providers/chroma/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/bootstrap/vectorstore_bootstrap.py` | integration catalog | **Done** — `create_default_vectorstore_manager()` resolves via `IntegrationProfile.vector_store` |
| `rag/vectorstore/providers/*` | other vector slugs | Catalog entry only until bridge provider ships |

**Not migrated to `integrations/`:** `intergrax/llm_adapters/` — LLM providers are a separate Tier-0 concern (§7.1.2 out-of-scope table).

#### M.6 — Testing strategy

| Layer | Location | Marker |
|-------|----------|--------|
| Contract unit tests | `tests/unit/integrations/` | default gate |
| Provider unit tests | `intergrax/integrations/providers/<slug>/tests/` | default gate |
| Registry / factory | `tests/unit/integrations/test_registry.py` | gate |
| Live vendor smoke | `tests/integration/integrations/` | `integration_live` (CI optional) |

Conformance test pattern: given a fake backend, assert all Protocol methods behave consistently (including error types).

#### M.7 — Agent Creation Guide (Appendix E)

Documented in [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) Appendix E:

- Agents: `capabilities`, `allowed_tools`, `ToolRequest` — no integration slug imports.
- Applications: `IntegrationProfile`, `wire_lab_integrations()`, `register_default_integrations()`.
- Env: `INTERGRAX_INTEGRATION_<CATEGORY>` overrides.

Tier-3 composition example (product factory):

```python
# applications/my_app/factory.py
from intergrax.integrations import (
    IntegrationCategory,
    IntegrationProfile,
    register_default_integrations,
)

def create_app():
    register_default_integrations()
    profile = IntegrationProfile.lab()  # or build_profile_from_env()

    cloud = profile.resolve(IntegrationCategory.CLOUD_PLATFORM)       # aws | azure | gcp
    db = profile.resolve(IntegrationCategory.RELATIONAL_STORE)        # sqlite | postgresql
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    storage = profile.resolve(IntegrationCategory.OBJECT_STORAGE)
    notifier = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
    # wire into Nexus factories, not into agents/
```

Agents reference capabilities in `AgentContract` (e.g. `allowed_tools=["websearch.query"]`) — not integration slugs.

#### M.8 — Definition of done (Phase M incremental)

Each provider PR is **done** when:

1. Contract conformance tests pass.
2. Registered in `catalog.py` with metadata.
3. `providers/<slug>/USAGE.md` — English: env vars, factory call, `IntegrationProfile` resolve, minimal invoke example.
4. At least one Tier-3 app or lab factory can select it via `IntegrationProfile`.
5. No new direct vendor imports added under `agents/`.

Szablony utrzymywane przez `scripts/generate_integration_usage_docs.py` (regeneracja po dodaniu providera).

---

---

#### V-KG — Knowledge Graph Evolution Path (Harness)

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-KG.1 | Graph-RAG architecture contract | **Done** | Medium | Canon section + terminology alignment |
| V-KG.2 | Hybrid retrieval reference path (vector + keyword + graph) | **Done** | Medium | Reference implementation notes |
| V-KG.3 | Graph-backed explainability trace fields | **Done** | Medium | Trace schema supports graph provenance |#### V-V6 — Phase V Closeout (L3/L4 Evidence & CI)

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-V6.1 | Bounded adaptive governance contracts (policy-learning envelopes, human gates) | **Done** | High | `adaptive_governance.py` + unit tests |
| V-V6.2 | L3/L4 maturity gate evidence aggregator | **Done** | **Critical** | `maturity_gate_evidence.py` + `maturity_gate_evidence_report.json` |
| V-V6.3 | CI closeout gate (`phase_v_closeout_gate.py --enforce`) | **Done** | **Critical** | Regression workflow runs closeout after gate tests |

#### Phase V — Execution matrix (dependencies and order)

Phase V should be executed in dependency-aware waves:

```text
Wave V0 (planning):      V-CG.1 + V-AM.1 + ownership/cadence baseline
Wave V1 (foundations):   V-CG.2 -> V-CG.4 + V-ALG.1 + V-PE.1 + V-EVAL.1
Wave V2 (quality):       V-CE.1 -> V-CE.3 + V-PE.2 -> V-PE.4 + V-EVAL.2 -> V-EVAL.3
Wave V3 (governance):    V-ALG.2 -> V-ALG.4 + V-SEC.1 -> V-SEC.4 + V-COST.1 -> V-COST.2
Wave V4 (ops maturity):  V-AM.2 -> V-AM.4 + V-EVAL.4 + V-COST.3 -> V-COST.4
Wave V5 (advanced):      V-MA.1 -> V-MA.3 + V-KG.1 -> V-KG.3
Wave V6 (closeout):      L3/L4 gate evidence + docs sync + priority reset
```

Critical dependency rules:

- `V-CG.1` must precede `V-CG.2/V-CG.4` and dependency-health metrics in `V-AM`.
- `V-PE.1` and `V-EVAL.1` must precede prompt/eval regression gates.
- `V-ALG.1` must precede production promotion flow (`V-ALG.2`).
- `V-SEC.*` and `V-COST.*` deny/degrade behavior must be validated before L3 gate.

#### Phase V — KPI thresholds and acceptance metrics

Minimum quantitative targets for Phase V completion:

| Area | Metric | Target |
|------|--------|--------|
| Capability graph | Changed harness PRs with graph impact artifact | **>= 95%** |
| Compatibility | Graph-edge compatibility gate pass on default branch | **100% required** |
| Lifecycle governance | Production-eligible agents with owner + certification metadata | **100% required** |
| Context quality | Context regression suite pass rate | **>= 95%** |
| Prompt quality | Prompt regression/adversarial suite pass rate | **>= 95%** |
| Evaluation ops | Critical capabilities with baseline + post-change scores | **100% required** |
| Security hardening | Adversarial defense suite pass rate (prompt/tool/retrieval) | **100% required** |
| Cost governance | Budget/quota policy test pass rate | **100% required** |
| Architecture metrics | Modularity/dependency/governance/observability coverage reported | **100% runs** |
| Architecture debt | Critical debt items trending (rolling 30d) | **non-increasing** |

#### Phase V — Operating cadence and governance ceremonies

- **Weekly:** Architecture hardening triage (V-* progress, blockers, scope control).
- **Weekly:** Security/cost review for new deny/degrade paths and policy regressions.
- **Bi-weekly:** Architecture review board for high-impact V-* design changes.
- **Monthly:** Architecture debt review (index trend + mitigation decisions).
- **Per release candidate:** L3/L4 evidence review (gates below) before release approval.

#### Phase V — Stream ownership model

| Stream | Primary owner | Supporting owners |
|--------|----------------|-------------------|
| V-CG | Platform architecture | Runtime + DevEx |
| V-ALG | Runtime governance | Platform + QA |
| V-CE / V-PE | Runtime + Prompt systems | QA/Eval |
| V-EVAL | Evaluation engineering | Runtime + Product quality |
| V-AM | Platform observability | Runtime + DevEx |
| V-SEC | Security engineering | Runtime + Platform |
| V-COST | Runtime economics | Platform + FinOps |
| V-MA | Orchestration/runtime | QA |
| V-KG | Knowledge systems | Runtime + Eval |

Owner rules:

- Every V-* PR must include a single accountable owner.
- Cross-stream dependencies must list an explicit approver before merge.
- Ownership metadata for production-impacting components must be reflected in registries where applicable.

#### Phase V — L3/L4 gate evidence (architecture maturity)

L3 readiness requires:

1. `V-CG.*`, `V-ALG.*`, `V-EVAL.1-4`, `V-SEC.1-4`, `V-COST.1-2`, `V-AM.1-3` complete.
2. KPI thresholds marked **100% required** above are satisfied.
3. Security and compatibility gates are green for two consecutive release cycles.
4. Architecture governance artifacts updated (canon + plan + traceability appendices).

L4 readiness requires:

1. L3 criteria met and stable.
2. `V-COST.3-4`, `V-MA.*`, `V-KG.*`, and adaptive loops with bounded governance controls.
3. Closed-loop evaluation feedback demonstrates measurable quality/cost improvement over baseline.
4. Policy-learning/adaptive behavior remains human-governed and auditable.

#### Phase V — Definition of done

1. Capability graph compatibility validation is active in CI for harness-critical changes.
2. Agent lifecycle governance gates exist and are enforced for production-eligible agents.
3. Context/prompt/evaluation governance artifacts are versioned and regression-tested.
4. Architecture health metrics are measurable and reviewed on a recurring cadence.
5. Security/data/cost hardening controls are testable, observable, and documented.
6. All changes remain harness-only (no implicit K.1/K.2 scope creep).
7. Coverage matrix (Appendix H) has **no `Uncovered` rows** for harness-scope architecture domains.

#### Phase V — Paydown log

| Date | V ID | Summary |
|------|------|---------|
| 2026-06-02 | V-CG.1, V-AM.1, V-ALG.1 | Typed baseline contracts added (`intergrax/runtime/architecture/`) + report-only artifacts script (`scripts/phase_v_foundations_report.py`) + unit tests |
| 2026-06-02 | V-CG.2, V-CG.3, V-CG.4 | Lineage/impact/compatibility modules + capability graph guard script (`scripts/phase_v_capability_graph_guard.py`) + enforce switch + unit tests |
| 2026-06-02 | V-AM.2, V-ALG.2, V-EVAL.1 | Metrics pipeline contracts + promotion flow evaluator + unified evaluation mode contracts + governance artifacts script (`scripts/phase_v_governance_report.py`) + unit tests |
| 2026-06-02 | V-ALG.3, V-ALG.4, V-EVAL.2 | Lifecycle/deprecation governance contracts + production ownership guard + evaluation asset bundle contracts + governance report extensions + unit tests |
| 2026-06-02 | V-EVAL.3, V-AM.3 | Automated evaluators (`evaluation_automation.py`) + architecture coverage report (`architecture_coverage.py`) + governance report persistence + unit tests |
| 2026-06-02 | V-AM.4, V-EVAL.4 | Debt governance cadence/policy report (`debt_governance.py`) + release trend/comparison report (`evaluation_registry_trends.py`) + governance script artifacts + unit tests |
| 2026-06-02 | V-SEC.1, V-SEC.2 | Prompt injection defense profile (`prompt_security.py`) + tool injection defense controls (`tool_security.py`) + governance artifacts + adversarial unit tests |
| 2026-06-02 | V-SEC.3, V-SEC.4 | Retrieval poisoning defense (`retrieval_security.py`) + tenant isolation/audit verification (`tenant_security.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-COST.1, V-COST.2, V-COST.3, V-COST.4 | Budget envelopes + quota deny/degrade + cost forecast/anomaly + optimization guardrails (`cost_*.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-CE.1, V-CE.2, V-PE.1, V-PE.2 | Context quality scoring/dedup (`context_engineering.py`) + prompt registry/composition (`prompt_registry_governance.py`, `prompt_composition.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-CE.3, V-CE.4, V-PE.3, V-PE.4 | Context regression benchmark + retrieval effectiveness + policy overlays + prompt regression suite + governance artifacts + unit tests |
| 2026-06-02 | V-MA.1, V-MA.2, V-MA.3, V-KG.1, V-KG.2, V-KG.3 | Multi-agent coordination catalog/selection/acceptance + Graph-RAG/hybrid retrieval/provenance contracts + governance artifacts + unit tests |
| 2026-06-02 | V-V6.1, V-V6.2, V-V6.3 | Bounded adaptive governance + L3/L4 maturity evidence + `phase_v_closeout_gate.py` CI enforcement |
| 2026-06-03 | H-APP.* | Phase H-APP: ApplicationEnvironmentProfile, unified wiring, 43 tasks, gate 510 |
| 2026-06-05 | V-REM.0.* | Plan audit: 9 Phase V + 1 Phase A gaps reclassified Partial; Phase V-REM + Appendix J + §6.1z queue opened |
| — | — | *(append row per merged PR)* |

---

---

## Appendix K


---

## Appendix K — Adaptive Harness Intelligence traceability (Phase W-ADAPT)

**Purpose:** 100% mapping from [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) (AHIA) to concrete **W-ADAPT.\*** IDs. **Canonical phase narrative:** [Phase W-ADAPT](#phase-w-adapt--adaptive-harness-intelligence-l4-runtime).

**Status:** **70/70 Done** (Band 2y closed 2026-06-05) — Waves W-ADAPT-0 through W-ADAPT-7 complete.

### K.1 AHIA component → W-ADAPT ID matrix

| AHIA component (§9) | Existing module to reuse | W-ADAPT ID |
|---------------------|--------------------------|------------|
| SignalCollector | `metrics/export.py`, `execution_guard.py`, `online_evaluation_registry.py` | W-ADAPT-1.4–1.11 |
| HarnessOutcomeSignal + utility | — (new) | W-ADAPT-1.1, W-ADAPT-1.8 |
| SignalStore | — (new SQLite) | W-ADAPT-1.3 |
| BanditStateStore | — (new) | W-ADAPT-2.1 |
| RoutingTuningEngine | `rag/routing/query_router.py`, LLM profiles | W-ADAPT-2.2, W-ADAPT-3.7, W-ADAPT-4.10 |
| ExecutionStrategyEngine | `history_evaluator.py`, `nexus_factory.py` | W-ADAPT-2.3, W-ADAPT-4.10 |
| PolicyLearningEngine | `adaptive_governance.py`, `tool_security.py` | W-ADAPT-2.4, W-ADAPT-4.6, W-ADAPT-4.9 |
| EvaluationFeedbackEngine | `evaluation_registry_trends.py` | W-ADAPT-2.5, W-ADAPT-5.3 |
| ProposalBuilder | `adaptive_governance.py` (`AdaptiveLoopProposal`) | W-ADAPT-2.6 |
| AdaptationEngine facade | — (new) | W-ADAPT-2.7 |
| Governance gate | `adaptive_governance.py`, `capability_graph_compatibility.py` | W-ADAPT-2.8–2.9 |
| ProfileVersionStore | — (new; pattern from `agent_promotion.py`) | W-ADAPT-3.1–3.2, W-ADAPT-3.5 |
| AdaptationExecutor | `runtime_governance_bridge.py` (extend) | W-ADAPT-3.3–3.4, W-ADAPT-4.4–4.5, W-ADAPT-4.8 |
| VerificationLoop | `evaluation_registry_trends.py`, `execution_guard.py` | W-ADAPT-5.1–5.5 |
| ProcessPatternMiner | trace persistence | W-ADAPT-6.* |
| AdaptationScheduler | Celery/message bus pattern from W-ML | W-ADAPT-2.12, W-ADAPT-5.12, W-ADAPT-6.5 |
| AdaptiveProfile (Tier-3) | `environment_profile.py` | W-ADAPT-4.1, W-ADAPT-7.1–7.2 |
| Ops reports / CI | `phase_v_governance_report.py` pattern | W-ADAPT-1.12, W-ADAPT-2.11, W-ADAPT-5.6–5.8 |
| Runtime L4 evidence | `maturity_gate_evidence.py` | W-ADAPT-5.7, W-ADAPT-5.11 |
| Author docs | AGENT_CREATION_GUIDE appendices | W-ADAPT-7.3–7.4 |

### K.2 Adaptive loop kind → implementation wave

| `AdaptiveLoopKind` | Engine | Apply wave | Authority default |
|--------------------|--------|------------|-------------------|
| `ROUTING_TUNING` | W-ADAPT-2.2 | W-ADAPT-4.10 | RECOMMEND |
| `EXECUTION_STRATEGY_TUNING` | W-ADAPT-2.3 | W-ADAPT-4.10 | RECOMMEND |
| `POLICY_LEARNING` | W-ADAPT-2.4 | W-ADAPT-4.6, W-ADAPT-4.9 | AUTO_WITH_HUMAN_GATE |
| `EVALUATION_FEEDBACK` | W-ADAPT-2.5 | observe only (W-ADAPT-5.3) | OBSERVE_ONLY |

### K.3 Lifecycle mode → task coverage

| Mode | Code | Primary tasks |
|------|------|---------------|
| Observe | L4-O | W-ADAPT-1.* |
| Recommend | L4-R | W-ADAPT-2.* |
| Shadow | L4-S | W-ADAPT-3.* |
| Canary | L4-C | W-ADAPT-4.3 |
| Apply | L4-A | W-ADAPT-4.4–4.10 |
| Verify | L4-V | W-ADAPT-5.* |

### K.4 Paydown log

| Date | W-ADAPT ID | Summary |
|------|------------|---------|
| 2026-06-05 | W-ADAPT-1.1–1.12 | Observe (L4-O): contracts, SignalStore, SignalCollector, Nexus/Runtime hooks, `phase_w_adapt_report.py` |
| 2026-06-05 | W-ADAPT-0.2–0.5 | ADR-ADAPT-001 + `intergrax/runtime/adaptive/` scaffold + gate import tests |
| 2026-06-05 | W-ADAPT-0.1 | Phase W-ADAPT register + §6.1t + §6.2ac + Appendix K + Band 2y |
| 2026-06-02 | W-ADAPT-2.1–2.12 | Recommend (L4-R): AdaptationEngine, ProposalBuilder, bandit store, proposal report |
| 2026-06-02 | W-ADAPT-3.1–3.7 | Shadow (L4-S): ProfileVersionStore, shadow executor, integration tests |
| 2026-06-02 | W-ADAPT-4.1–4.10 | Apply (L4-A): canary, apply, rollback, policy-learning HITL |
| 2026-06-02 | W-ADAPT-5.1–5.12 | Verify (L4-V): VerificationLoop, auto-rollback, L4 runtime closeout gate, runbooks |
| 2026-06-02 | W-ADAPT-6.1–6.5 | ProcessPatternMiner, trace sequence reader, pattern report export |
| 2026-06-02 | W-ADAPT-7.1–7.7 | Tier-3 AdaptiveProfile wiring, debug routes, business outcome webhook, acceptance E2E |
| 2026-06-02 | W-ADAPT-OPS | Lab L4-O observe default (`LAB_ADAPTIVE_OBSERVE`); CI/release `--enforce-l4-runtime`; canon §54 + AHIA sync |

---
