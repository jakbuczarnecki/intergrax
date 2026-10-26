# DIST-SCALE-X — Vendor / Backend Scale Qualification Matrix (P0)

| Field | Value |
| --- | --- |
| **Task** | `DIST-SCALE-X-P0` |
| **Baseline SHA** | `35344557d3ce67a631f82df96c86b79b92b4db93` |
| **Classification legend** | **M** mandatory physical scale · **R** representative-equivalence · **F** functional-only · **E** external sandbox · **U** unsupported for distributed production |
| **Disposition** | Supporting P0 — **not** independent closure |

---

## 1. Production-capable backends @ START_HEAD

Discovery sources: `intergrax/queueing/providers/**`, `intergrax/runtime/persistence/sqlite_composition.py`, `intergrax/collaborative_work/persistence.py`, `intergrax/integrations/_shared/cloud_task_queue.py`, TOOL-LIFE-X TL10/TL12 evidence, [REAL_VENDOR_ENTERPRISE_QUALIFICATION.md](REAL_VENDOR_ENTERPRISE_QUALIFICATION.md) App. A (compose layout — **not executed** in P0).

| Backend / adapter | Code anchor | DIST class | P2–P6 batch | Equivalence / gap notes |
| --- | --- | --- | --- | --- |
| **PostgreSQL** (relational + CW) | `PostgreSQLIntegrationConfig`, `open_postgresql_collaborative_work_repositories` | **M** | P2–P4, WL-5 | — |
| **DocumentStore TaskQueue** | `intergrax/queueing/providers/document_store/document_store_task_queue.py`, `colocated_worker.py` | **M** | P2, P3, P6 | — |
| **RabbitMQ TaskQueue** | `rabbitmq_task_queue.py`, `BrokerBackedTaskQueueBase` | **M** *or* **R** | P2, P5 | **Operator:** designate as sole broker **M** OR secondary **R** after proof |
| **Kafka TaskQueue** | `kafka_task_queue.py`, `BrokerBackedTaskQueueBase` | **R** default; **M** if operator selects | P2 or P5 | **EVIDENCE GAP** for DIST: BG-Q5–Q7 replay on Kafka @ CLOSURE_SHA not present @ P0 |
| **Celery TaskQueue** | `celery_task_queue.py` | **R** | P2 optional | **EVIDENCE GAP:** no DIST behavioral equivalence proof vs **M** broker (contract `TaskQueue` alone insufficient) |
| **SQLite runtime bundle** | `SQLiteRuntimePersistenceBundle`, per-family SQLite stores | **R** | P4 smoke | **Equivalence (contract):** same persistence protocols as STATE-X inventory (SX-F01–F20). **Not** multi-node proof |
| **In-memory / fakes** | test doubles | **F** | — | **NOT SUFFICIENT** for roadmap exit item 2 |
| **Redis** (distributed rate limit / optional KV) | W2 path; LKW Kafka wiring uses Redis KV state | **R** | P3 WL-6, P5 L if effective | **Equivalence (partial):** `tests/unit/llm_adapters/test_distributed_rate_limit.py` — LLM RL only; **EVIDENCE GAP** for full-platform storm @ DIST |
| **HTTP loopback (Class B tools)** | TOOL-LIFE-X strict-host / F8 E2E | **M** (DIST WL-4 scope) | **P3 WL-4** | Physical loopback only — **not** cloud vendor |
| **MCP tool host** | TL11 qualification | **F** | P6 optional mix | Does not replace WL-4 HTTP **M** |
| **Cloud LLM / SaaS HTTP vendors** | Integrations registry | **E** | Out of DIST parent CLOSED | **PROD-Q** Dimension B |
| **CloudTaskQueue** (e.g. QStash) | `CloudTaskQueue` in `integrations/_shared/cloud_task_queue.py` | **E** | — | Sandbox + operator |
| **Vector / graph / search stores** | Integration providers | **F** (default) | — | **N/A WITH EVIDENCE** unless representative host declares production-effective RAG path |
| **MySQL/Oracle CW** | Not materialized @ START_HEAD per REAL_VENDOR doc | **U** / **F** | — | Not claimed for distributed production |

---

## 2. Representative-equivalence (**R**) — contract and behavioral justification

| Pair / backend | Contract-level justification | Behavioral proof @ START_HEAD | DIST disposition |
| --- | --- | --- | --- |
| **Kafka vs Rabbit (broker)** | Shared `TaskQueue` API + `BrokerBackedTaskQueueBase` submit/ack/redelivery hooks | BG-Q5–Q7 proven on **in-process / DocumentStore** worker paths; **not** re-run on Kafka in DIST suite | **EVIDENCE GAP** — must **RE-VALIDATE** on chosen **M** broker; Kafka **R** until replay @ CLOSURE_SHA |
| **Celery vs Rabbit** | Both implement `TaskQueue`; different worker process model | BG-Q8 plugin pattern for custom queue — not Celery-vs-Rabbit scale equivalence | **EVIDENCE GAP** — Celery **R** only; optional P2 slice |
| **SQLite vs PostgreSQL (state)** | STATE-X `ATOMICITY_MATRIX`; same protocol surfaces per family | STATE-X semantic PASS on SQLite/PG adapters separately; cross-store non-atomicity explicit | **R** for dev/single-node; **M** PG for distributed durability claims |
| **Redis distributed RL vs local RL** | Shared `ProviderRateLimitPort` family (W2-C) | `test_distributed_rate_limit.py` unit scope | **R** for LLM path; **EVIDENCE GAP** for WL-6 full stack |

**Rule:** Without behavioral proof row filled @ CLOSURE_SHA, **R** does **not** satisfy mandatory physical scale for that backend class.

---

## 3. Mandatory physical backends before P2 (minimum set)

| ID | Backend | Class | Operator decision required? |
| --- | --- | --- | --- |
| MB-1 | PostgreSQL | **M** | Pin image/version from compose (before `QUAL INFRA RUN`) |
| MB-2 | DocumentStore TaskQueue | **M** | Representative DocumentStore provider for queue |
| MB-3 | Multi-process workers (LKW or P0-designated host) | **M** (topology) | Confirm host application |
| MB-4 | One message broker (RabbitMQ **or** Kafka) | **M** (exactly one) | **YES** — `THRESHOLD_LOCK_RECORD` §broker |
| MB-5 | HTTP loopback WL-4 | **M** | P3 — no cloud vendor |

---

## 4. Operator decisions before P2 (no infra launch in P0)

1. **Broker M:** RabbitMQ vs Kafka (other broker → **R** only).  
2. **Redis** in WL-6 / resource exhaustion scope: yes/no.  
3. **Representative host** for workers: default LKW vs GCA (document in P1).  
4. **Compose profile pins** for PG + broker (`infra/integration/docker-compose.yml`).  
5. **Soak / dedicated host** (WL-7) — duration lock deferred to `DIST_SCALE_X_THRESHOLD_LOCK_RECORD` (post-SAC freeze).

---

## 5. Backend combination matrix (P2–P6)

| Combo | Purpose | Allowed when |
| --- | --- | --- |
| PG + DocumentStore queue + N workers | WL-1,2 baseline | MB-1,2,3 + qual infra authorized |
| PG + broker **M** + N workers | Horizontal scale dim B | MB-4 locked |
| PG + HTTP loopback | WL-4 | P3 only |
| SQLite-only persistence | Smoke / **R** regression | **Not** sole proof for exit item 2–3 |
| Cloud vendor HTTP | — | **E** — PROD-Q only |

---

## 6. Traceability

- Inventory surfaces: [DIST_SCALE_X_P0_CLOSED_WORLD_INVENTORY.md](DIST_SCALE_X_P0_CLOSED_WORLD_INVENTORY.md)  
- WL bindings: [DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md](DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md)  
- SAC: [DIST_SCALE_X_SAC_PROPOSAL.md](DIST_SCALE_X_SAC_PROPOSAL.md)
