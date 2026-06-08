**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix B — Technical debt backlog

**Purpose:** consolidated backlog for review and **incremental paydown**.  
**Source:** canon §2 map, §0.5 maturity, Phase G–K gaps, lab sign-off findings (2026-05-27).  
**How to use:** pick items by priority; apply §0.6 (Tier-1 only when reusable across agents).  
**Status:** `Open` | `Done` | `Deferred`

### B.0 Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-05-29 | M.6-gcp | `providers/gcp/` — cloud_platform facade; ADC/service account + category slug defaults |
| 2026-05-29 | M.6-azure | `providers/azure/` — cloud_platform facade; token health + category slug defaults |
| 2026-05-29 | M.6-aws | `providers/aws/` — cloud_platform facade; STS health + category slug defaults |
| 2026-05-29 | M.6-cassandra | `providers/cassandra/` + `contracts/document_store.py`; CQL partition-scoped CRUD |
| 2026-05-29 | M.6-ms365_graph | `providers/ms365_graph/` + `contracts/collaboration_suite.py`; Graph mail/calendar/directory |
| 2026-05-30 | M.6-prometheus | `providers/prometheus/` + `contracts/observability_backend.py`; PromQL query API |
| 2026-05-30 | M.6-confluence | `providers/confluence/` + `contracts/wiki_knowledge.py`; REST wiki; single-entry `opens.py` |
| 2026-05-30 | M.6-jira | `providers/jira/` + `contracts/issue_tracker.py`; REST v3; single-entry `opens.py` |
| 2026-05-30 | M.6-mysql | `providers/mysql/` — beta `RelationalStore` (pymysql); single-entry `opens.py` |
| 2026-05-30 | M.6-provider-layout | Providers grouped under `providers/<category>/<slug>/`; `layout.py` slug map; tests mirrored by category |
| 2026-05-30 | M.6-p2-batch | P2/P3 integrations — 22 slugs (`azure_blob`, `gcs`, `dynamodb`, cloud queues, SQL variants, SMTP, OTEL, GitHub/Linear/Azure DevOps, Notion/SharePoint, Google Workspace, Brave/SerpAPI, Playwright); `_shared/p2/`; **324** integration unit tests |
| 2026-05-30 | M.7-agent-guide-integrations | `AGENT_CREATION_GUIDE.md` Appendix E — agents vs Tier-3 wiring |
| 2026-05-30 | N.2.1-unified-wiring | `ApplicationBuildContext`, `builder_key`/`factory_path`, lab+legal on `build_application_registry` |
| 2026-05-30 | N.2-conformance | `build_registry_from_manifest`, `load_agent_from_binding` + unit tests |
| 2026-05-30 | N.1-manifest | `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` + unit tests |
| 2026-05-30 | N.10-new-stack | `scaffold new-stack` — agent + application; `TIER3_READINESS.md` |
| 2026-05-30 | N.9-scaffold-acceptance | `test_scaffold_acceptance.py` — lab/product runtime E2E; fix product `agent_factories.py` indent |
| 2026-05-30 | N.8-agent-guide-4e | `AGENT_CREATION_GUIDE.md` Step 4E — `new-application`, Docker scripts, §7.4.8 links |
| 2026-05-30 | N.4-product-scaffold | `--profile product` → FastAPI Core host, `agent_factories.py`, auth stub env; `new_application_product.py` |
| 2026-05-30 | N.5-docker-build-scripts | `build-docker.sh` / `build-docker.bat` in scaffold + lab/legal/research/poc; `docker_templates.py` |
| 2026-05-30 | N.0-docs | Canon §7.4.8–§7.4.10 + Phase N plan (application environment, manifest, scaffold steps) |
| 2026-05-30 | M.8-lab-profile | `wire_lab_integrations()` + `providers/log/` — lab uses `IntegrationProfile.lab()` |
| 2026-05-30 | M.4-kafka-rabbitmq-adopt | Queueing bootstrap + integration tests use `integrations/providers/{kafka,rabbitmq}/` only |
| 2026-05-30 | M.4-rabbitmq | `providers/rabbitmq/` + runtime `build_rabbitmq_transport()` delegate |
| 2026-05-29 | M.4-lab_json | `providers/lab_json/` + runtime `create_interaction_adapter(LAB)` delegate — **M.4 P0 complete** |
| 2026-05-29 | M.4-webhook | `providers/webhook/` + runtime `create_notification_adapter(WEBHOOK)` delegate |
| 2026-05-29 | M.4-teams-adopt | Runtime notifications/interactions/verifier + long_running delegate to `providers/teams/` |
| 2026-05-29 | M.4-teams | `providers/teams/` — dual category catalog entry |
| 2026-05-29 | M.4-slack-adopt | Runtime notifications/interactions/verifier + long_running delegate to `providers/slack/` |
| 2026-05-29 | M.4-slack | `providers/slack/` — dual category + resolve dispatches by category |
| 2026-05-29 | M.4-bing | `providers/bing/` — SearchProvider adapter over legacy Bing v7 |
| 2026-05-29 | M.4-google_cse | `providers/google_cse/` — SearchProvider adapter over legacy CSE |
| 2026-05-29 | M.4-celery | `providers/celery/` — message bus + worker helpers; no `kv_store` |
| 2026-05-29 | M.4-kafka | `providers/kafka/` + transport delegate; requires `kv_store` |
| 2026-05-29 | M.4-sqlite-adopt | Runtime `open_*` + apps delegate to `integrations/providers/relational_store/sqlite/` |
| 2026-05-29 | M.4-sqlite | `providers/sqlite/` + bundle (10 domain stores); lazy bootstrap + package `__init__` |
| 2026-05-29 | M.4-redis | Complete bundle: `create_redis_integration()` — KV, idempotency, rate limit, semaphore, rerank |
| 2026-05-27 | B.08, B.10 | `wire_nexus_observability` + SQLite defaults in Legal / Research / Lab factories; integration test |
| 2026-05-27 | B.01, B.02 | `RuntimeCheckpoint` full snapshot + UAEP mid-step cursor/resume; acceptance `05b` |
| 2026-05-27 | B.12, B.14 | Production `POST /v1/interactions/intake` on lab; Legal legacy `AgentEngine` removed |
| 2026-05-27 | B.05 | Escalation notification template + scheduler wiring in lab + SAFETY_VIOLATION timeout→escalate |
| 2026-05-27 | B.09, B.17 | Injectable `trace_store` on debug API; gate uses `pytest -m gate` (`testpaths` includes `agents/`) |
| 2026-05-27 | Platform stabilization | All Tier-3 hosts: validating runtime events, plugin bootstrap, resilient delivery (lab/legal/research/poc); shared `_shared/platform_wiring` + `notification_wiring` |
| 2026-05-27 | Infra paydown | SQLite DLQ ledger + debug `/notifications/*`; `ValidatingRuntimeEventPersistence`; Tier-3 plugin bootstrap |
| 2026-05-27 | B.07, B.11, B.13, B.18, B.24 | Schema registry + phase coverage + `RuntimePlugin`; metrics export + `GET /debug/tasks/{id}/metrics`; retry/DLQ delivery; echo + research_mock HTTP trace acceptance; agents vendor import gate test |
| 2026-05-27 | K.3–K.5 | `coerce_replay_policy_engine` + `ExecutionGuard.evaluate_replay`; ChatAgent production import guard; CI gate paths aligned with full gate (**394** tests) |
| 2026-05-27 | B.06, §18 | `BEFORE/AFTER_TOOL_CALL` + agent-selection hooks; product interaction intake on legal/research (**397** gate) |

### B.1 Runtime & §42 convergence

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.01 | **UAEP mid-step checkpoint** — resume inside a long-running step (not only between steps / HITL) | §42.9.3, §26 | **High** | **Done** | Long-running domain agents (Legal, Research) | Tier-1 | `uaep_step_cursor`, `should_resume_uaep_step`, optional `resume_step` (2026-05-27) |
| B.02 | **Full checkpoint snapshot** — plan + graph node states + UAEP index + pending decisions in one durable blob | §42.9.2 | **High** | **Done** | Multi-agent graphs, crash recovery | Tier-1 | `plan_snapshot`, `graph_snapshot`, `pending_decisions` in `RuntimeCheckpoint` (2026-05-27) |
| B.03 | **Policy engine facade** — single `PolicyEngine` for replay, validation, runtime policy | §42.11 | **Medium** | **Done** | Indirect — consistent governance for all agents | Tier-1 | `PolicyEngine` + `coerce_policy_engine`; Nexus/UAEP/interrupt handler (2026-05-27) |
| B.04 | **Dual `AgentDecision` cleanup** — converge tools-agent variant with canonical §42.7 enum | §42.7 | **Medium** | **Done** | Agents emitting decisions must use one contract | Tier-1 | `ToolPlanDecision` in `tools.core.tool_plan_decision`; no `tools_agent` re-export (2026-06-02) |
| B.05 | **Escalation policy production path** — `SAFETY_VIOLATION` / HITL expiry → real escalation (not stub) | §42.38, §42.10 | **Medium** | **Done** | HITL-heavy agents | Tier-1 | `escalation.v1` template, `wire_long_running_scheduler`, lab startup, SAFETY_VIOLATION timeout→escalate (2026-05-27) |
| B.06 | **Hook / middleware parity** — full §42.20 pipeline vs current Nexus-embedded hooks | §42.20, §42.22 | **Low** | **Done** | Extension agents via plugins | Tier-1 | Lifecycle + **tool call** + **agent selection** hooks; decision/interrupt/retry hooks remain optional (2026-05-27) |
| B.07 | **§42 maturity remainder** — schema versioning (§42.29), full `ExecutionPhase` coverage, plugin contracts | §42 | **Medium** | **Done** (baseline) | Platform stability for new agents | Tier-1 | `runtime/schema/registry.py`, `events/phase_coverage.py`, `plugins/contract.py` (2026-05-27) |

### B.2 Observability & debug surface

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.08 | **Application trace store split** — factories used `InMemoryRunTraceStore` while debug API reads SQLite | §33, §42.24 | **High** | **Done** | HTTP `/debug/tasks/*` 503 in product apps | Tier-3 | `wire_nexus_observability` + `open_run_trace_store` (2026-05-27) |
| B.09 | **Debug API trace reader** — only SQLite file path; no injectable in-memory / shared store handle | §19 | **Medium** | **Done** | Lab tests, local dev without file I/O | Tier-1 | `trace_store` on `create_debug_router` / `create_debug_app`; lab passes Nexus store (2026-05-27) |
| B.10 | **NexusLoop runtime events in app factories** — all Tier-3 factories pass runtime events to Nexus | §42.24 | **Medium** | **Done** | Events 503 on `/debug/tasks/{id}/events` | Tier-3 | Legal + Research default SQLite; lab when path passed (2026-05-27) |
| B.11 | **Metrics layer** — event-first, trace-second, **metrics-third** unified export | §42.1, §33 | **Low** | **Done** | Ops visibility, SLOs | Tier-0 | `runtime/metrics/export.py` + `GET /debug/tasks/{run_id}/metrics` (2026-05-27) |

### B.3 Interaction surfaces (§18)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.12 | **Production Slack / Teams webhooks** — inbound intake on product hosts | §18 | **Medium** | **Done** | Organization Worker, HITL from chat | Tier-0 / Tier-3 | `POST /v1/interactions/intake` on lab/legal/research/poc via `wire_interaction_intake_service` (2026-05-27) |
| B.13 | **Outbound delivery hardening** — retries, DLQ, delivery receipts for HITL notifications | §18, §42.10 | **Low** | **Done** | HITL agents in prod | Tier-0 | `RetryingNotificationDelivery` + `SQLiteDeliveryLedger` + debug `/debug/notifications/*` (2026-05-27) |

### B.6 Integration Library (§7.1)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.18 | **Integration catalog package** — `intergrax/integrations/` scaffold | §7.1.1 | **High** | **Done** | All agents needing external systems | Tier-0 | M.1–M.3 + M.5 (2026-05-29) |
| B.19 | **P0 provider wraps** — M.4 catalog slugs | §7.1.3 | **High** | **Done** | Lab + first prod apps | Tier-0 | All P0 slugs wrapped + runtime adoption (2026-05-29) |
| B.20 | **PostgreSQL relational_store** — production DB adapter | §7.1.3 | **Medium** | **Done** (beta) | Multi-tenant applications | Tier-0 | `providers/postgresql/` — domain stores SQLite-first |
| B.21 | **Jira + Confluence providers** — issue/wiki ingestion | §7.1.3 | **Medium** | **Done** (beta) | PM / research agents | Tier-0 | Integrations + catalog tools (Phase O.4, 2026-05-30) |
| B.22 | **MS365 Graph provider** — mail, calendar | §7.1.3 | **Medium** | **Done** (beta) | Org worker, scheduling agents | Tier-0 | `providers/ms365_graph/`; client credentials via `opens.py` |
| B.23 | **Prometheus observability_backend** — PromQL query API | §33, §7.1.3 | **Low** | **Done** (beta) | Ops / SLO | Tier-0 | `providers/prometheus/`; complements B.11 metrics layer design |
| B.28 | **Cassandra document_store** — wide-column adapter for high-volume retention | §7.1.3 P2 | **Medium** | **Done** (beta) | Runtime event archive at scale; ops telemetry | Tier-0 | `providers/cassandra/`; single-entry `opens.py` |
| B.29 | **Elasticsearch observability_backend** — log search / aggregations | §7.1.3 P2 | **Medium** | **Done** (beta) | Ops log triage; optional RAG over logs | Tier-0 | `providers/elasticsearch/`; single-entry `opens.py`; complements B.23 |
| B.30 | **Databricks relational_store** — SQL Warehouse / Unity Catalog SQL | §7.1.3 P2 | **Medium** | **Done** (beta) | Analytics agents, lakehouse reporting | Tier-0 | `providers/databricks/`; single-entry `opens.py`; PAT |
| B.31 | **MongoDB document_store** — flexible JSON persistence | §7.1.3 P2 | **Medium** | **Done** (beta) | Agent memory, unstructured artifacts | Tier-0 | `providers/mongodb/`; PyMongo only in `opens.py`; reuses `DocumentStore` |
| B.32 | **Pinecone vector_store bridge** — catalog entry → `rag/` | §7.1.3 P2 | **Medium** | **Done** (beta) | Production RAG agents | Tier-0 | `providers/pinecone/` thin adapter; SDK only in `opens.py` |
| B.33 | **Qdrant + Chroma vector_store bridges** — same pattern as B.32 | §7.1.3 P2 | **Low** | **Done** (beta) | Self-hosted / dev RAG | Tier-0 | `providers/qdrant/`, `providers/chroma/`; RAG bootstrap via catalog |
| B.34 | **Object storage contract + S3 provider** — blobs for artifacts / sandboxes | §7.1.3 P2 | **Medium** | **Done** (beta) | Large file handoff, exports | Tier-0 | `contracts/object_storage.py`, `providers/s3/`; boto3 only in `opens.py` |
| B.35 | **Notion + SharePoint wiki_knowledge** — internal docs ingestion | §7.1.3 P3 | **Low** | **Done** (beta) | Research / runbook agents | Tier-0 | REST adapters; `_shared/p2/factories.py` |
| B.36 | **GitHub + Linear issue_tracker** — dev workflow sources | §7.1.3 P3 | **Low** | **Done** (beta) | Code-aware agents | Tier-0 | REST; thin provider shells |
| B.37 | **email_smtp notification_channel** — outbound mail without chat | §7.1.3 P3 | **Low** | **Done** (beta) | HITL, scheduled reports | Tier-0 | stdlib SMTP in factory open path |
| B.38 | **OpenTelemetry observability_backend** — trace/metric export | §33, §7.1.3 P3 | **Low** | **Done** (beta) | Unified ops dashboards | Tier-0 | `providers/otel/`; beta noop exporter default |
| B.39 | **Playwright browser_automation** — dynamic web interaction | §7.1.3 P3 | **Low** | **Done** (beta) | Research on JS-heavy sites | Tier-0 | `providers/playwright/`; browser launch in factory |
| B.25 | **AWS cloud_platform facade** — auth + S3/SQS/DynamoDB/ElastiCache defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | AWS-hosted applications | Tier-0 | `providers/aws/`; infrastructure only |
| B.26 | **Azure cloud_platform facade** — MI + Blob/Service Bus/Azure SQL defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | Azure-hosted applications | Tier-0 | `providers/azure/`; infrastructure only |
| B.27 | **GCP cloud_platform facade** — ADC + GCS/Pub/Sub/Cloud SQL defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | GCP-hosted applications | Tier-0 | `providers/gcp/`; infrastructure only |
| B.24 | **Direct vendor SDK in agents** — audit + lint rule | §5.2, §7.1.4 | **Medium** | **Done** | Prevents catalog bypass | Tier-2 | `scripts/check_agents_vendor_imports.py` + gate test `test_vendor_import_guard_b24` (2026-05-27) |

### B.7 Tool Library (§7.1.6)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.40 | **Tool Library scaffold** — catalog, profile, wiring context | §7.1.6 | **High** | **Done** | All agents using external capabilities | Tier-0 | Phase O.2; apps wire tools O.8 (2026-05-30) |
| B.41 | **Context tools** — `rag.retrieve`, `websearch.query` | §7.1.7, §22.1 | **High** | **Done** | RAG / research agents | Tier-0 | Phase O.3 (2026-05-30) |
| B.42 | **Jira catalog tools** — `jira.get_issue`, `jira.search_tasks`, … | §7.1.6 | **Medium** | **Done** | PM / legal workflow agents | Tier-0 | Phase O.4 (2026-05-30) |
| B.43 | **Unified tool model** — deprecate `use_rag` / `use_websearch` flags | §7.1.7, §22.2 | **High** | **Done** | Consistent tool policy + MCP | Tier-1 | Phase O.5 (2026-05-30) |
| B.44 | **Legacy ToolBase migration** | §5.2.2 | **Medium** | **Done** | Single registry | Tier-0 | Phase O.7; `tools_base` deprecated |
| B.45 | **MCP tool export from catalog** | §7.1.6 | **Low** | **Done** | External MCP clients | Tier-3 | Phase O.6 |

### B.4 Legacy & composition

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.14 | **`ChatAgent` / legacy engine removal** — `LEGAL_USE_LEGACY_AGENT_ENGINE` removed | §39, §41 | **Medium** | **Done** | Single execution path for all agents | Tier-1 / Tier-3 | Legal `fastapi_router` requires `UnifiedTaskRunner`; legacy flags removed (2026-05-27) |
| B.15 | **Legal full E2E gate (real LLM)** — deferred acceptance with live model | — | **Low** | **Deferred** | Legal quality assurance | Tier-2 / CI | K.6; separate from Agent OS gate; enable when CI budget approved |
| B.16 | **Lab agent auto-discovery** — manifest-driven roster + scaffold | §7.4 | **Low** | **Done** | Onboarding friction | Tier-3 | Phase N: `ApplicationManifest`, `new-stack` (N.10); explicit `AgentBinding` remains by design (2026-05-30) |
| B.28 | **Per-application `.env.example` missing** — only root `.env.example`; lab/legal vars in README only | §7.4.8 | **Medium** | **Done** | Deployable POC friction | Tier-3 | N.7 backfill + scaffold (2026-05-30) |
| B.29 | **`new-application` scaffold (lab)** — Tier-3 hosts hand-copied from legal/lab | §7.4.8 | **High** | **Done** | Lab + product profiles via CLI; gate acceptance | Tier-3 / platform | N.10 `new-stack` optional |
| B.30 | **No application-level Dockerfile** — only `infra/docker/docling/` | §7.4.8 | **Medium** | **Done** | Per-app `docker/` + build scripts on lab/legal/research/poc | Tier-3 | N.5–N.7 (2026-05-30) |

### B.5 Test & certification hygiene

| ID | Item | Canon | Priority | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------------|------|----------------|
| B.17 | **`agents/` gate collection** — `signoff_probe` test marks `gate` but lives under `agents/` (may not be collected by default `pytest tests/`) | — | **Low** | **Done** | Sign-off smoke not in main gate count | Test infra | `testpaths` includes `agents/`; canonical gate: `uv run pytest -m gate -q` (2026-05-27) |
| B.18 | **HTTP observability acceptance** — trace on echo + multi-agent mock (graph path) | Appendix A #9–10 | **Low** | **Done** | Certification confidence | Test | `test_lab_application_runs_echo_with_trace_observability`, `test_lab_application_runs_research_mock_with_graph_trace` (2026-05-27) |

### B.8 Suggested priority order (for planning)

```text
1. ~~B.08, B.10~~ — observability consistency (Done 2026-05-27)
2. ~~B.01, B.02~~ — checkpoint / full snapshot (Done 2026-05-27)
3. ~~B.03, B.04~~ — governance facade + AgentDecision cleanup (Done 2026-05-27)
4. ~~B.12, B.14~~ — product interaction + legacy removal (Done 2026-05-27)
5. ~~B.05~~ — escalation production path (Done 2026-05-27)
6. ~~B.09, B.17~~ — debug trace injection + gate collection (Done 2026-05-27)
7. ~~B.06~~ — hook parity doc + lifecycle wiring (Done 2026-05-27)
8. ~~B.07, B.11, B.13, B.18, B.24~~ — §42 baseline, metrics export, delivery hardening, HTTP trace acceptance, vendor import guard (Done 2026-05-27)
9. ~~Platform stabilization~~ — all Tier-3 factories aligned (Done 2026-05-27)
10. B.15 — Legal E2E real LLM (**Deferred** — product/CI decision)
11. ~~Phase Q~~ — Harness audit remediation — **Done** (Appendix C)
12. ~~Phase Q+ / Phase R~~ — **Done** (Appendices D, E)
13. ~~Phase S — Harness environment GA~~ — **Done**
14. ~~Phase T — Harness cleanliness~~ — **Done**
15. Phase U — Harness production hardening — **Done**
16. Harness completion backlog (§4.1) — **Done** (2026-06-02)
17. Phase K — K.1/K.2 business agents — **Deferred**
18. Tier-3 product apps / Legal E2E — **Deferred**
```

**Note:** Platform harness (Q–U) is complete. **Harness completion** (legacy + CI) is active. Business agents and product applications are **end of list**.

---
