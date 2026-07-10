# Proof Receipts — Implementation Plan

**Architecture (1:1):** [`architecture/PROOF_RECEIPTS.md`](../architecture/PROOF_RECEIPTS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Proof consumer:** LKW-PR ([`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PR)  
**Last updated:** 2026-07-10 — **PROOF-RECEIPTS-1E closed**

---

## A. Status

| Item | Value |
|------|-------|
| **Track status** | **Closed** — PROOF-RECEIPTS-1A through **1E** complete |
| **Architecture** | [`architecture/PROOF_RECEIPTS.md`](../architecture/PROOF_RECEIPTS.md) — **PROOF-RECEIPTS-1B closed** |
| **Current proof consumer** | LKW (structured receipt after live proofs) |
| **Default live vendor** | MongoDB (`document_store` integration) |
| **Next step** | — (Proof Receipts wave complete) |

**Note:** LKW.4E (Kafka-backed background task live proof) remains **closed**. Proof receipts are the **next platform proof wave** — not a markdown closeout of LKW.4E.

Historical **LKW.5** in the LKW plan refers to the **closed persistence proof** (`LKW_DATA_HOME` + Qdrant). The proof-receipt wave uses **LKW-PR** to avoid renumbering that closed wave.

---

## B. Staged rollout

| ID | Task | Scope | Status |
|----|------|-------|--------|
| **PROOF-RECEIPTS-1A** | Define ProofReceipt contract and MongoDB-backed storage architecture | `ProofReceipt` model; `DocumentRecord` mapping; `ProofReceiptStore`; architecture + plan docs; unit tests | **Done** |
| **PROOF-RECEIPTS-1B** | DocumentStore vendor integration base contract | `DocumentStoreVendorIntegrationContract`; category config/operations; `as_document_store()` boundary; unit tests | **Done** |
| **PROOF-RECEIPTS-1C** | Complete `document_store` vendor category cutover | MongoDB, Cassandra, DynamoDB on `DocumentStoreVendorIntegrationContract`; remove `DocumentStoreIntegrationContract` | **Done** |
| **PROOF-RECEIPTS-1D** | LKW Docker proof stack with MongoDB / Mongo Express | Compose overlay for live vendor-backed receipt persistence | **Closed** |
| **PROOF-RECEIPTS-1E** | LKW proof receipt recording through platform | LKW proof workloads write `ProofReceipt` via `ProofReceiptStore` after live proofs | **Closed** |

---

## C. PROOF-RECEIPTS-1A deliverables (closed)

| Deliverable | Location |
|-------------|----------|
| `ProofReceipt` Pydantic model | `intergrax/proofs/receipts/contracts.py` |
| `DocumentRecord` mapping helpers | `intergrax/proofs/receipts/document_store.py` |
| `ProofReceiptStore` engine | `intergrax/proofs/receipts/store.py` |
| Architecture canon | `docs/architecture/PROOF_RECEIPTS.md` |
| Unit tests | `tests/unit/proofs/receipts/test_proof_receipts_contract.py` |

**Acceptance (1A):**

- [x] Platform-owned `ProofReceipt` model exists
- [x] Provider-neutral `ProofReceiptStore` delegates only to `DocumentStore`
- [x] No pymongo imports outside MongoDB provider package (this task adds none)
- [x] No LKW MongoDB-specific code introduced
- [x] Documentation states markdown is reviewer guide, not source of truth
- [x] Documentation states MongoDB is default/natural live vendor
- [x] LKW-PR wave added to LKW implementation plan
- [x] LKW.4E not redefined

---

## D. PROOF-RECEIPTS-1B deliverables (closed)

| Deliverable | Location |
|-------------|----------|
| `DocumentStoreVendorIntegrationContract` | `intergrax/runtime/integrations/document_store.py` |
| Category config / operations / kinds | same module |
| Package exports | `intergrax/runtime/integrations/__init__.py` |
| Unit tests | `tests/unit/runtime/integrations/test_document_store_vendor_integration_contract.py` |
| Architecture layer docs | `docs/architecture/PROOF_RECEIPTS.md` §G |

**Acceptance (1B):**

- [x] Contract derives from `PlatformIntegrationContract`
- [x] `integration_kind` is `document_store`
- [x] `for_provider(...)` builds stable `{provider_id}:document_store` identity
- [x] Default capabilities: `READ`, `WRITE`, `HEALTH_CHECK`
- [x] Default operations: `get`, `put`, `delete`, `query`, `close`
- [x] Safe `public_view()` — no secrets
- [x] Base `as_document_store()` raises `NotImplementedError`
- [x] No MongoDB live proof implemented
- [x] `ProofReceiptStore` still depends only on `DocumentStore`

**Depends on:** PROOF-RECEIPTS-1A  
**Blocks:** PROOF-RECEIPTS-1C

---

## E. PROOF-RECEIPTS-1C deliverables (closed)

| Deliverable | Detail |
|-------------|--------|
| Vendor category cutover | MongoDB, Cassandra, DynamoDB inherit `DocumentStoreVendorIntegrationContract` |
| `as_document_store()` | Each vendor returns existing provider-backed `DocumentStore` adapter |
| Old contract removal | `DocumentStoreIntegrationContract` removed from active runtime modules |
| Registry alias | `document_store` folder maps to `DocumentStoreVendorIntegrationContract` |
| Unit tests | `test_document_store_vendor_category_cutover.py`; provider tests updated |

**Depends on:** PROOF-RECEIPTS-1B  
**Blocks:** PROOF-RECEIPTS-1D

---

## F. PROOF-RECEIPTS-1D deliverables (closed)

| Deliverable | Detail |
|-------------|--------|
| Docker compose overlay | `applications/local_workspace_application/docker/docker-compose.mongodb.yml` — MongoDB (`lkw-mongodb`) + Mongo Express (`lkw-mongo-express`) |
| Live vendor path | Platform `MongoDBDocumentStoreIntegration` → `as_document_store()` → `DocumentStore.put/get` smoke — **not** a `ProofReceipt` |
| Reviewer inspection | Mongo Express at `http://localhost:8086` (default) — inspection only; not on LKW runtime path |
| Persistent volume | Named volume `lkw_mongodb_data` |
| Proof runner | `applications/local_workspace_application/scripts/run-lkw-mongodb-proof-stack.bat` |
| Smoke validator | `applications/local_workspace_application/scripts/verify_lkw_mongodb_stack.py` |

**Compose command (repository root):**

```bash
docker compose \
  -f applications/local_workspace_application/docker/docker-compose.yml \
  -f applications/local_workspace_application/docker/docker-compose.mongodb.yml \
  up --build
```

**Defaults:** database `intergrax_proofs`; collection `proof_receipts`; MongoDB service `lkw-mongodb`; auth source `admin`. Smoke record partition `platform_smoke` / row `mongodb_document_store` is **infrastructure connectivity data only** — not a `ProofReceipt`. **ProofReceipt recording remains PROOF-RECEIPTS-1E.**

**Depends on:** PROOF-RECEIPTS-1C  
**Blocks:** PROOF-RECEIPTS-1E  
**Closeout:** live PASS via `run-lkw-mongodb-proof-stack.bat` (platform put/get + MongoDB restart persistence read-back).

---

## G. PROOF-RECEIPTS-1E scope

| Deliverable | Detail |
|-------------|--------|
| Recording helper | `intergrax/proofs/receipts/recording.py` — `record_and_verify_proof_receipt()` |
| LKW proof recording | `run-lkw-background-task-proof.py` persists `ProofReceipt` after live Kafka background-task proof |
| Combined proof stack | `run-lkw-background-task-proof.bat` composes Kafka + MongoDB overlays |
| Platform boundary | No pymongo or direct MongoDB access in Tier-3 |
| Public reviewer docs | Step 9 in `LKW_PLATFORM_PROOF.md` — receipt inspection via Mongo Express |

**Path:**

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

**Depends on:** PROOF-RECEIPTS-1D
**Closeout:** live PASS via `run-lkw-background-task-proof.bat` (workload + receipt write/read/query verification)

---

## H. Integration analogy

| Existing proof | Platform surface | Vendor |
|----------------|------------------|--------|
| LKW.4E background tasks | `message_bus.*` | Kafka |
| LKW-OBS / Sentry path | Observability export / problem signals | Sentry |
| Elasticsearch timeline | `ObservabilityVendorIntegrationContract` | Elasticsearch |
| **LKW-PR proof receipts** | **`ProofReceiptStore` → `DocumentStore` → `DocumentStoreVendorIntegrationContract`** | **MongoDB** |

---

## I. Out of scope (this track)

- Rewriting LKW.4E closeout as markdown-only proof
- pymongo imports in Tier-3 applications
- In-memory DocumentStore as live proof backend
- Public LKW Step 9 before live proof succeeds (PROOF-RECEIPTS-1E)
- Standalone IntegrationProfile wiring task — profile/config selection remains an architectural requirement, not a separately scheduled proof-receipt step
