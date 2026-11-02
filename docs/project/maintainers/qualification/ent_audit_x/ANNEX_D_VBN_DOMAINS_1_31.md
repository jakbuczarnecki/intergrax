# ANNEX D — VBN roadmap domains 1–31

**PLAN_BASELINE_SHA:** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**PLAN_PUBLICATION_SHA:** `6328f0cf915ed1f4cee13acb5eb68f07a7438b8e`  
**Inventory replay @ HEAD:** 69 concerns in `vendor_backend_concern_inventory()`; gate `test_ebh5_g06_01_roadmap_31_domains_accounted` (EBH-5).

| Dom | Domain | Production-effective surface | Provider/backend | Owner | Existing evidence @ HEAD | Current-HEAD gap | RPL | Required ENT check | Test / nodeid | Closure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | RELATIONAL_PERSISTENCE | `persistence.relational_sqlite_lab`; integration relational | sqlite / lab store | Integrations / STATE | EBH-5 row + INT-CONFIG CERT scoped | Re-orchestrate; historical PASS ≠ auto | QUALIFIED lab path | VBN-SQLITE-EFFECTIVE | `test_ebh_5_l2_integrity_gates` | FREEZE GREEN + drift 0 |
| 2 | NOSQL_DOCUMENT | integration document store concern | vendor bundle | Integrations | inventory row | HEAD replay required | NOT YET / QUALIFIED per row | VBN-INVENTORY-DRIFT | g06 domain test | classified drift |
| 3 | KEY_VALUE | `integration.key_value_cache` | KV provider | Integrations | CONFIG-X + EBH-5 | cache domain N/A cross-ref | QUALIFIED | VBN-SHADOW-SELECTION | config_x gates | no shadow selector |
| 4 | CACHE | N/A — KV concern covers cache selection | — | Integrations composition | `_roadmap_domain_na_evidence` CACHE | bounded discovery only | N/A WITH EVIDENCE | VBN-INVENTORY-DRIFT | `test_ebh5_g06_01_roadmap_six_na_domains_have_discovery_evidence` | N/A validated |
| 5 | VECTOR | vector store integration | embedding/vector bundle | Integrations / RAG | EBH-2G scoped + inventory | HEAD replay | per matrix row | VBN-REPLACEABILITY-MATRIX | RPL gates | matrix PASS |
| 6 | GRAPH | graph store integration | provider bundle | Integrations | EBH-2G-R2 evidence | HEAD replay | QUALIFIED | VBN-REPLACEABILITY-MATRIX | ebh_5 factory rpl | RPL PASS |
| 7 | OBJECT_BLOB | blob/object concern | storage adapter | Integrations | inventory | replay @ HEAD | per row | VBN-REACHABILITY | behavioral package C | reachability proof |
| 8 | EVENT_STREAMING | event bus / streaming concerns (2) | configured transport | Runtime/OBS | HARNESS-W5 scoped | coupling check | QUALIFIED OTLP path | VBN-OBS-OPS-COUPLING | harness_01 gates | no lock-in finding |
| 9 | QUEUES_WORKERS | worker queue concerns (2) | queue backend | Execution/Runtime | HARNESS-W4 scoped | boundedness + vendor | per row | VBN-SILENT-FALLBACK | W4 qual slice | fail-closed |
| 10 | DISTRIBUTED_COORDINATION | distributed config / runtime state | composition wiring | Runtime OBS | NA disposition COORDINATION | static discovery limits | N/A WITH EVIDENCE | VBN-INVENTORY-DRIFT | g06 NA test | N/A OK |
| 11 | TRANSACTION_IDEMPOTENCY | idempotency concerns (2) | durable store | STATE-X | STATE-X-R3 scoped | tenant/idempotency replay | per row | VBN-INVENTORY-DRIFT | state_x qual | drift 0 |
| 12 | CHECKPOINTS_EXECUTION | checkpoint concerns (4) | execution store | Execution/STATE | STATE-X + harness | HEAD replay | per row | VBN-REACHABILITY | ee_a1 gate | proven paths |
| 13 | AGENT_MEMORY | memory provider concerns | memory backend | Memory | EBH-2H scoped | replay | per row | VBN-CORE-HARDCODING | ebh6 hardcoding register | PROHIBITED=0 |
| 14 | RAG_RETRIEVAL | retrieval/rerank (3) | RAG providers | RAG/Integrations | EBH-2G | replay | QUALIFIED rerank | VBN-CONTRACT-LEAKAGE | ebh_2i gate | no vendor in contracts |
| 15 | LLM_EMBEDDING | LLM/embedding (10) | model providers | Integrations | EBH-2E/F scoped | largest domain — full matrix | mixed | VBN-REPLACEABILITY-MATRIX | ebh_5 dual slug rpl | row-level PASS |
| 16 | AGENT_RUNTIMES | runtime hosting concerns (4) | agent runtime | Hosting/Runtime | EBH-2F/R1 | replay | per row | VBN-SHADOW-SELECTION | harness gates | no shadow registry |
| 17 | HOSTING_SUBSTRATE | substrate concerns (2) | host execution | Runtime | EBH-2F | replay | per row | VBN-REACHABILITY | host replaceability tests | structural proof |
| 18 | EXTERNAL_MCP | external integrations (6) | MCP/adapters | Integrations | EBH-2F + INT | replay | per row | VBN-CORE-HARDCODING | ebh6 register | no prohibited core |
| 19 | LOGGING | observability profile sinks | via OTLP/event | Observability | NA disposition LOGGING | not standalone category | N/A WITH EVIDENCE | VBN-OBS-OPS-COUPLING | runtime_event_delivery_wiring | N/A OK |
| 20 | METRICS | metrics via observability profile | OTLP | Observability | NA disposition METRICS | same as logging | N/A WITH EVIDENCE | VBN-OBS-OPS-COUPLING | W5 gates | N/A OK |
| 21 | TRACING | trace export concern | OTLP | Observability | inventory row (1) | replay | QUALIFIED transport | VBN-OBS-OPS-COUPLING | ADR-OBS-005 regressions | PASS |
| 22 | DIAGNOSTICS | diagnostics concern | platform diag | Diagnostics | inventory | replay | per row | VBN-OBS-OPS-COUPLING | harness qual | no exec truth |
| 23 | ALERTING | alerting concern | integration | Integrations | inventory | replay | per row | VBN-INVENTORY-DRIFT | g06 | drift OK |
| 24 | AUDIT_EVIDENCE | audit/evidence store | evidence spine | Trace/Governance | TRACE-X scoped | replay | per row | VBN-INVENTORY-DRIFT | trace_x qual | tenant safe |
| 25 | SECRETS | secrets concern | secret refs | CONFIG/Integrations | inventory | replay | per row | VBN-SILENT-FALLBACK | config_x fail-closed | no silent default |
| 26 | AUTH_IDENTITY | auth identity concern | identity provider | Governance | inventory | replay | per row | VBN-CONTRACT-LEAKAGE | gov_x2 qual | typed contracts |
| 27 | CONFIG_PROFILE | config/profile (9) | profile activation | CONFIG-X | CONFIG-X PASS historical + HEAD replay | shadow/effective | per row | VBN-SHADOW-SELECTION | config_x final gates | reconciled |
| 28 | NETWORK_HTTP | HTTP client surfaces | via integrations | Integrations | NA disposition NETWORK | no duplicate row | N/A WITH EVIDENCE | VBN-INVENTORY-DRIFT | g06 NA | N/A OK |
| 29 | TELEMETRY_OTLP | OTLP transport (2) | OTLP | Observability | HARNESS-W5 CLOSED scoped | replay | QUALIFIED | VBN-OBS-OPS-COUPLING | W5 batch tests | PASS |
| 30 | EVAL_ANALYTICS | eval analytics | none in scanned roots | Platform | NA disposition EVAL | bounded scan limits | N/A WITH EVIDENCE | VBN-INVENTORY-DRIFT | g06 NA | monitor S8 |
| 31 | DISCOVERED | discovered concerns (8) | mixed | Integrations | inventory DISCOVERED bucket | classify each row | per concern | VBN-INVENTORY-DRIFT | inventory integrity | no orphan rows |

**Freshness rule:** EBH-5 parent CLOSED @ scoped SHA `33ecf19c…` — ENT FREEZE mode must replay g06 + hardcoding register @ **audit execution SHA**, not cite ledger alone.
