# Proof Receipts — Architecture

**Status:** Canonical architecture (domain pair 1:1)  
**Plan (1:1):** [`plan/PROOF_RECEIPTS.md`](../plan/PROOF_RECEIPTS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Proof consumer:** LKW ([`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PR)  
**Last updated:** 2026-07-10 — **PROOF-RECEIPTS-1E closed**

---

## A. Purpose

Platform proofs must leave **structured, queryable evidence** that reviewers and automation can inspect without parsing ad-hoc markdown logs. Markdown reviewer guides (for example [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../public-adoption/LKW_PLATFORM_PROOF.md)) explain *how* to run a proof; they are **not** the source of truth for proof outcomes.

The source of truth is a **`ProofReceipt`** document persisted through the provider-neutral **`DocumentStore`** contract and selected vendor integration (MongoDB for the default live proof path).

---

## B. Ownership boundaries

| Layer | Owns |
|-------|------|
| **Platform** | `ProofReceipt` model (`intergrax/proofs/receipts/contracts.py`); `ProofReceiptStore` engine (`intergrax/proofs/receipts/store.py`); `DocumentRecord` mapping (`intergrax/proofs/receipts/document_store.py`); `DocumentStore` contract; integration catalog/profile selection |
| **LKW (Tier-3)** | Proof workload execution; building domain/provider evidence payloads; invoking platform persistence — **no** direct MongoDB/pymongo imports |
| **Providers** | Vendor backends behind `DocumentStore` (MongoDB default for live proof path; Cassandra, DynamoDB, … as future alternatives) |
| **Reviewer docs** | Human runbooks and step guides only — never authoritative proof storage |

---

## C. Platform proof pattern

```text
Application proof workload (e.g. LKW)
  → build ProofReceipt (domain + provider + guardrail evidence)
  → ProofReceiptStore.put()
  → DocumentStore contract (provider-neutral data surface)
  → DocumentStoreVendorIntegrationContract (platform integration boundary)
  → concrete vendor integration (MongoDB, Cassandra, DynamoDB — PROOF-RECEIPTS-1C)
  → vendor SDK / backend
  → persisted structured receipt
```

`DocumentStore` is the **provider-neutral data surface** — `get`, `put`, `delete`, `query`, and `close` without vendor-specific types. `DocumentStoreVendorIntegrationContract` is the **platform integration boundary** between that surface and concrete vendors. It mirrors `ObservabilityVendorIntegrationContract` used for Elasticsearch/Sentry-style observability proofs: category-specific config, `supported_operations`, `for_provider(...)`, safe `public_view()`, and `as_document_store()` for vendor implementations.

MongoDB remains the **default/natural live vendor** for ProofReceipt storage. **PROOF-RECEIPTS-1D** adds the optional LKW Docker overlay (`docker-compose.mongodb.yml`) with `lkw-mongodb`, `lkw-mongo-express`, and a platform-backed `DocumentStore` smoke path — **closed**. The smoke record is infrastructure connectivity data, not a `ProofReceipt`. LKW must **never** depend on MongoDB or pymongo directly — only on `ProofReceiptStore` → `DocumentStore` (recording in **PROOF-RECEIPTS-1E**).

This mirrors existing platform integration proofs:

| Proof domain | Platform contract | Typical live vendor |
|--------------|-------------------|---------------------|
| Async background tasks | `message_bus.*` / `TaskQueue` | Kafka (LKW.4E) |
| Problem/error signals | `PlatformProblemSignal` / observability export | Sentry |
| Event/log timeline | Observability vendor integration | Elasticsearch / Kibana |
| **Structured proof receipts** | **`ProofReceiptStore` → `DocumentStore` → `DocumentStoreVendorIntegrationContract`** | **MongoDB** |

---

## D. ProofReceipt model

**Code:** `intergrax/proofs/receipts/contracts.py`

Minimum fields:

| Field | Role |
|-------|------|
| `schema_version` | Contract version (`intergrax.proof_receipt.v1`) |
| `proof_id` | Stable receipt identifier |
| `proof_kind` | Proof taxonomy key (e.g. `platform_background_task`) |
| `application_id` | Proof workload application (e.g. `local_workspace_application`) |
| `result` | `PASS` \| `FAIL` \| `ERROR` |
| `recorded_at` | UTC timestamp |
| `run_id` | Platform run identifier |
| `correlation_id` | Optional cross-surface correlation |
| `task_id` | Optional background task identifier |
| `provider_evidence` | Vendor-neutral provider facts (message bus, document store, …) |
| `domain_evidence` | Workload-specific proof facts (markers, search hits, …) |
| `guardrails` | Negative controls (no mock queue, no in-memory bypass, …) |
| `metadata` | Extension bag |

The model is **generic** — not hardcoded to LKW.4E or LKW only.

---

## E. DocumentStore mapping

**Code:** `intergrax/proofs/receipts/document_store.py`

| DocumentRecord field | Mapping |
|---------------------|---------|
| `partition_key` | `proof_receipts/<application_id>` |
| `row_key` | `proof/<proof_kind>/<run_id>` |
| `data` | Full `ProofReceipt` JSON payload |
| `ttl_seconds` | `None` by default |

Mapping is **provider-neutral**. No MongoDB-specific field names or BSON types leak into the receipt contract.

---

## F. ProofReceiptStore engine

**Code:** `intergrax/proofs/receipts/store.py`

`ProofReceiptStore` depends only on `DocumentStore`:

- `put(receipt)` — upsert through `DocumentStore.put`
- `get(application_id, proof_kind, run_id)` — point read
- `query(application_id, proof_kind=…, limit=…)` — partition query with optional `row_key_prefix`
- `close()` — delegate resource release

Host/application wiring selects the `DocumentStore` implementation from **`IntegrationProfile`** / config via a concrete `DocumentStoreVendorIntegrationContract` subclass — the same layered pattern as Kafka message bus, Sentry observability, and Elasticsearch timeline proofs.

---

## G. DocumentStore vendor integration contract

**Code:** `intergrax/runtime/integrations/document_store.py`

| Artifact | Role |
|----------|------|
| `DocumentStoreVendorIntegrationContract` | Platform-owned category contract deriving from `PlatformIntegrationContract` — **only active `document_store` contract** (PROOF-RECEIPTS-1C) |
| `DocumentStoreVendorIntegrationConfig` | Typed, secret-safe config (`database_name`, `collection_name`, `namespace`) |
| `DocumentStoreVendorOperation` | Declared operations: `get`, `put`, `delete`, `query`, `close` |
| `DocumentStoreVendorKind` | Well-known vendor slugs (`mongodb`, `cassandra`, …) |
| `as_document_store()` | Boundary method — MongoDB, Cassandra, and DynamoDB return existing `DocumentStore` adapters |

Default capabilities: `READ`, `WRITE`, `HEALTH_CHECK`. Integration identity: `{provider_id}:document_store` (for example `mongodb:document_store`).

---

## H. Strict boundaries

1. **No pymongo outside the MongoDB provider package** (`intergrax/integrations/providers/document_store/mongodb/`).
2. **LKW must not import pymongo** or call MongoDB APIs directly.
3. **No LKW-only MongoDB helper** — persistence goes through `ProofReceiptStore` → `DocumentStore`.
4. **No markdown as source of truth** — `.proof_docs/` and reviewer guides are operational aids only.
5. **No in-memory/fake store as live proof acceptance** — unit-test doubles may validate the store contract only.

---

## I. PROOF-RECEIPTS-1D — LKW MongoDB Docker proof stack (closed)

**Overlay:** `applications/local_workspace_application/docker/docker-compose.mongodb.yml`  
**Runner:** `applications/local_workspace_application/scripts/run-lkw-mongodb-proof-stack.bat`  
**Validator:** `applications/local_workspace_application/scripts/verify_lkw_mongodb_stack.py`

```text
Platform smoke (PROOF-RECEIPTS-1D)
  → create_mongodb_integration()
  → MongoDBDocumentStoreIntegration.as_document_store()
  → DocumentStore.put() / get()
  → MongoDB (lkw-mongodb, volume lkw_mongodb_data)

Reviewer (inspection only)
  → Mongo Express (http://localhost:8086 default)
  → MongoDB
```

- Smoke partition `platform_smoke` / row `mongodb_document_store` is **not** a `ProofReceipt`.
- No `ProofReceiptStore` call in 1D; no pymongo imports in LKW application code.
- Public Step 9 in `LKW_PLATFORM_PROOF.md` documents Mongo Express receipt inspection for **PROOF-RECEIPTS-1E**.

---

## J. PROOF-RECEIPTS-1E — LKW background-task receipt recording (closed)

**Workload proof:** `applications/local_workspace_application/scripts/run-lkw-background-task-proof.py`
**Runner:** `applications/local_workspace_application/scripts/run-lkw-background-task-proof.bat` (Kafka + MongoDB overlays)
**Recording helper:** `intergrax/proofs/receipts/recording.py`

```text
LKW background-task workload
  → live Kafka execution
  → ProofReceipt
  → ProofReceiptStore
  → DocumentStore
  → MongoDBDocumentStoreIntegration
  → MongoDB
  → Mongo Express reviewer inspection
```

- `ProofReceipt` in MongoDB is the source of truth; markdown is reviewer guidance only.
- The infrastructure smoke record from 1D is not a `ProofReceipt`.
- No direct pymongo or MongoDB write exists in LKW.
- No in-memory fallback is accepted for live proof acceptance.
- Final `proof_result=PASS` depends on receipt write/read/query verification.

---

## K. Future reviewer path (target)

After PROOF-RECEIPTS-1C–1E closeout:

1. Run LKW proof workload (background task proof with receipt recording).
2. Application records a `ProofReceipt` through `ProofReceiptStore`.
3. `IntegrationProfile` selects `document_store=mongodb`.
4. Reviewer inspects receipt in **Mongo Express** / Mongo UI — not by opening markdown closeout files.

Public Step 9 in `LKW_PLATFORM_PROOF.md` is available for Mongo Express receipt inspection.

---

## L. Code references

| Artifact | Path |
|----------|------|
| ProofReceipt contract | `intergrax/proofs/receipts/contracts.py` |
| Document mapping | `intergrax/proofs/receipts/document_store.py` |
| Recording helper | `intergrax/proofs/receipts/recording.py` |
| Store engine | `intergrax/proofs/receipts/store.py` |
| DocumentStore contract | `intergrax/integrations/contracts/document_store.py` |
| DocumentStore vendor contract | `intergrax/runtime/integrations/document_store.py` |
| MongoDB provider | `intergrax/integrations/providers/document_store/mongodb/` |
| LKW proof schedule | `applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md` §LKW-PR |
