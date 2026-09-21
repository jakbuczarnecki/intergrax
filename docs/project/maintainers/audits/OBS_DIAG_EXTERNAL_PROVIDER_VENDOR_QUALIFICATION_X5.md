# OBS-DIAG-X5 — External Provider & Vendor Qualification

**Verdict:** `PASS — PROVIDER SUPPORT MATRIX RECONCILED`

Reconciliation note: prior docs sometimes implied broader production qualification than live proofs supported. X5 downgrades catalog-only integrations to **ADAPTER ONLY** / **SUPPORTED + NOT QUALIFIED** unless normal operation, failure, and recovery were executed.

## A. Verdict

`PASS — PROVIDER SUPPORT MATRIX RECONCILED` — OBS/DIAG spine providers (sqlite-file, Kafka, OTLP export semantics) qualified with live or executed contract proofs; remaining catalog integrations honestly labeled.

## B. Git baseline

| Pole | Wartość |
| ---- | ------- |
| branch | `development` |
| START_HEAD | `4245102fb22bb6ff70f4ac1babdeb43c18a7f8f6` |
| origin/development | `4245102fb22bb6ff70f4ac1babdeb43c18a7f8f6` |
| final HEAD | `d2e58034a48db6bb6a5d2356ff07128b037dbfc3` |
| target files clean | yes (no parallel WIP on provider targets) |
| HEAD drift | none at start |
| unrelated WIP | none at start |

## C. Provider inventory (OBS/DIAG scope)

| Provider | Domain | Contract | Declared support | Adapter |
| -------- | ------ | -------- | ---------------- | ------- |
| sqlite-file | persistence | EvidencePersistencePort / DocumentStore | qualification default | yes |
| kafka | transport | MessageProducer/Consumer, TaskQueue | IntegrationStatus.STABLE | yes |
| mongodb | persistence | DocumentStore → Problem/FunctionalEvidence | IntegrationStatus.STABLE | yes |
| otel | telemetry | ObservabilityExporter | IntegrationStatus.STABLE | yes |
| opentelemetry_collector | telemetry | OTLP transport target | IntegrationStatus.STABLE | yes |
| in-process-async | transport | colocated worker spine | platform-internal | yes |

Inventory source: `testing_support/obs_diag_provider_qualification/inventory.py` + integration manifests under `intergrax/integrations/providers/{message_bus,observability_backend,document_store}/`.

## D. Final qualification matrix

| Provider | Live env | Normal | Failure | Recovery | Fresh client | Final status |
| -------- | -------: | -----: | ------: | -------: | -----------: | ------------ |
| sqlite-file | AVAILABLE | PASS | PASS (typed conformance) | PASS | PASS | SUPPORTED + QUALIFIED |
| kafka | AVAILABLE @ localhost:9092 | PASS (X4) | PASS | PASS | PASS (X4 fresh read) | SUPPORTED + QUALIFIED |
| mongodb | UNAVAILABLE / env-gated | PASS when URI set (D1-R1) | NOT EXECUTED in X5 | NOT EXECUTED | PASS (D1-R1) | SUPPORTED + NOT QUALIFIED |
| otel export | in-process | PASS | PASS | PASS (HARDEN-3C) | n/a | SUPPORTED + QUALIFIED |
| opentelemetry_collector | NOT EXECUTED | — | — | — | — | ADAPTER ONLY |
| catalog telemetry (22 others) | NOT EXECUTED | — | — | — | — | ADAPTER ONLY |
| catalog message_bus (11 others) | NOT EXECUTED | — | — | — | — | ADAPTER ONLY |

## E. Persistence — sqlite-file

```text
provider: sqlite-file (SqliteFileDocumentStore / SQLiteRuntimeEventStore)
contract: ProblemPersistence, EvidencePersistencePort (DG-005)
durability: writer close → fresh reader (integration + DG-005)
concurrency semantics: DocumentStore CAS via conformance helpers
tenant isolation: assert_problem_persistence_conformance
failure: contract typed errors (conformance); no in-memory fallback in host wiring
recovery: file-backed state survives adapter reopen
final status: SUPPORTED + QUALIFIED
```

## F. Persistence — mongodb

```text
provider: mongodb document store
contract: FunctionalEvidencePersistence / DocumentStore
durability: D1-R1 process boundary when INTERGRAX_MONGODB_URI configured
failure/recovery: not re-run in X5 session (env-gated)
final status: SUPPORTED + NOT QUALIFIED (honest — spine default is sqlite-file)
```

## G. Transport — kafka

```text
provider: kafka (confluent-kafka / librdkafka)
delivery semantics: at-least-once; duplicate without commit proven in X5
producer failure: publish flush fails closed (unreachable broker)
consumer restart: redelivery without commit (X5)
recovery: publish succeeds after broker available
platform spine: X4 cross-process OBS/DIAG (P4)
final status: SUPPORTED + QUALIFIED
```

## H. Telemetry — otel

```text
provider: otel exporter (runtime observability/exporters/otlp)
export mode: derived export (best-effort)
failure isolation: HARDEN-3C + export_policy — business/canonical event retained
health visibility: ObservabilityExporterHealthRegistry + export metrics
recovery: exporter retry bounded in adapter tests
canonical evidence unaffected: PROVEN (unit gates)
final status: SUPPORTED + QUALIFIED (semantic); live collector endpoint NOT required for DIAG truth
```

## I. Support status changes

| Provider | Before | After | Reason |
| -------- | ------ | ----- | ------ |
| provider matrix row | OPEN (X7) | RECONCILED (X5) | Live proofs + honest catalog downgrade |
| mongodb (spine claim) | implied production | SUPPORTED + NOT QUALIFIED | spine uses sqlite-file; Mongo env-gated |
| observability_backend catalog | implicit “supported” | ADAPTER ONLY | no live failure/recovery proof in X5 |

## J. Vendor-boundary proof

```text
vendor imports in OBS/DIAG semantic core (excl. exporters/): 0
vendor config in contracts: 0
provider branching in diagnostic_orchestrator: 0
```

Gate: `tests/unit/runtime/architecture/test_obs_diag_x5_provider_gates.py`

## K. Failure/recovery summary

| Provider | Failure injected | Expected | Actual | Recovery |
| -------- | ---------------- | -------- | ------ | -------- |
| kafka | 127.0.0.1:1 broker | publish fails | RuntimeError deliver | localhost:9092 publish OK |
| kafka | no commit + new consumer | redelivery | same payload | commit on happy path |
| sqlite-file | new adapter instance | read durable | conformance PASS | same file |
| otel | exporter exception | fail-open export | HARDEN-3C PASS | retry path in tests |

## L. Ownership/lifecycle

Host diagnostic composition: custom persistence **borrowed**; forbidden in-memory Problem fallback (`diagnostic_composition.py`). Kafka/sqlite adapters opened via integration opens/factories — not core vendor branching.

## M. Source-of-truth invariants

```text
canonical RuntimeEvent truth: PROVEN
canonical Problem truth: PROVEN
telemetry as derived export: PROVEN
vendor as diagnostic authority: NO
```

## N. External infrastructure

```text
Kafka: confluent-kafka; Docker profile infra/integration; localhost:9092
Health probe: kafka_broker_ready / AdminClient.list_topics
```

## O. Tests (X5 suites)

```text
tests/integration/providers/obs_diag/test_x5_kafka_transport_failure_recovery.py
tests/integration/providers/obs_diag/test_x5_sqlite_file_persistence_qualification.py
tests/unit/runtime/architecture/test_obs_diag_x5_provider_gates.py
```

## P. Regression suites

X2/X2A/X2B, X3/X3A, X4, X4A: re-run in session post-gate (see session pytest log).

## Q–S. Files modified

Production: none (qualification support + gates only).

Support: `testing_support/obs_diag_provider_qualification/*`

Tests: `tests/integration/providers/obs_diag/*`, `test_obs_diag_x5_provider_gates.py`

Docs: `OBSERVABILITY.md`, `DIAGNOSTICS.md`, this audit.

## T. Remaining X5 limitations

External HITL vendor service qualification remains **NOT_PROVEN** (out of persistence/transport/telemetry matrix). Per-vendor telemetry backends beyond OTLP semantic proofs remain **ADAPTER ONLY** until live endpoint qualification is executed.

## U. Commit / provenance (corrected in X5A)

| Field | SHA / note |
| ----- | ---------- |
| X5 original evidence commit | `2282a2aea9893325735a02e5dd773da4ff97632d` |
| Erroneous final SHA recorded in initial X5 audit | `d2e58034a48db6bb6a5d2356ff07128b037dbfc3` (did not match X5 commit — corrected by OBS-DIAG-X5A) |
| X5A integrity closure | see [`OBS_DIAG_PROVIDER_EVIDENCE_INTEGRITY_X5A.md`](OBS_DIAG_PROVIDER_EVIDENCE_INTEGRITY_X5A.md) |

## V. Ready for X6?

`YES` after X5A — provider claims reconciled with lifecycle and anti-drift gates; exact-SHA backbone gate (X6) is the next closure step.

---

> Wszystkie wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie rzeczywistego kodu i commitu znajdującego się na GitHub. Sam raport Cursor AI nie jest wystarczającym dowodem poprawności.

> OBS-DIAG-X5 nie może zostać uznane za zakończone, jeżeli provider otrzymuje status `SUPPORTED + QUALIFIED` bez realnego live proofu normal operation, failure i recovery, jeżeli niedostępny provider powoduje silent fallback do in-memory implementation, jeżeli telemetry vendor staje się źródłem prawdy albo jeżeli vendor-specific implementation leakage pojawia się w core runtime/OBS/DIAG contracts.
