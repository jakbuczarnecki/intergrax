# Proof Receipts — Implementation Plan

**Architecture (1:1):** [`architecture/PROOF_RECEIPTS.md`](../architecture/PROOF_RECEIPTS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Proof consumer:** LKW-PR ([`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PR)  
**Last updated:** 2026-07-10 — **PROOF-RECEIPTS-1B closed**

---

## A. Status

| Item | Value |
|------|-------|
| **Track status** | In progress — architecture + contracts + vendor integration boundary defined |
| **Architecture** | [`architecture/PROOF_RECEIPTS.md`](../architecture/PROOF_RECEIPTS.md) — **PROOF-RECEIPTS-1B closed** |
| **Current proof consumer** | LKW (structured receipt after live proofs) |
| **Default live vendor** | MongoDB (`document_store` integration) |
| **Next step** | **PROOF-RECEIPTS-1C** — implement MongoDB DocumentStore vendor integration |

**Note:** LKW.4E (Kafka-backed background task live proof) remains **closed**. Proof receipts are the **next platform proof wave** — not a markdown closeout of LKW.4E.

Historical **LKW.5** in the LKW plan refers to the **closed persistence proof** (`LKW_DATA_HOME` + Qdrant). The proof-receipt wave uses **LKW-PR** to avoid renumbering that closed wave.

---

## B. Staged rollout

| ID | Task | Scope | Status |
|----|------|-------|--------|
| **PROOF-RECEIPTS-1A** | Define ProofReceipt contract and MongoDB-backed storage architecture | `ProofReceipt` model; `DocumentRecord` mapping; `ProofReceiptStore`; architecture + plan docs; unit tests | **Done** |
| **PROOF-RECEIPTS-1B** | DocumentStore vendor integration base contract | `DocumentStoreVendorIntegrationContract`; category config/operations; `as_document_store()` boundary; unit tests | **Done** |
| **PROOF-RECEIPTS-1C** | Implement MongoDB DocumentStore vendor integration | Concrete MongoDB subclass; provider registration; `as_document_store()` live adapter | Planned |
| **PROOF-RECEIPTS-1D** | Wire ProofReceiptStore through IntegrationProfile/config | Host/application wiring selects `DocumentStore` from profile; expose store to proof workloads | Planned |
| **PROOF-RECEIPTS-1E** | Record LKW proof receipts through platform engine | LKW proof scripts write `ProofReceipt` via `ProofReceiptStore` after live proofs | Planned |
| **PROOF-RECEIPTS-1F** | Add MongoDB + Mongo Express Docker proof stack | Compose overlay for live vendor-backed receipt persistence | Planned |
| **PROOF-RECEIPTS-1G** | Add reviewer-visible Step 9 to LKW_PLATFORM_PROOF.md | Public reviewer path for MongoDB receipt inspection — only after live stack ships | Planned |

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

## E. PROOF-RECEIPTS-1C scope (next)

| Deliverable | Detail |
|-------------|--------|
| MongoDB vendor subclass | `DocumentStoreVendorIntegrationContract` concrete implementation |
| Provider registration | Ensure `mongodb` document store resolves from catalog/profile |
| Conformance | `assert_document_store` passes for MongoDB adapter |
| Config | `INTERGRAX_MONGODB_URI`, database name env alignment |

**Depends on:** PROOF-RECEIPTS-1B  
**Blocks:** PROOF-RECEIPTS-1D

---

## F. Integration analogy

| Existing proof | Platform surface | Vendor |
|----------------|------------------|--------|
| LKW.4E background tasks | `message_bus.*` | Kafka |
| LKW-OBS / Sentry path | Observability export / problem signals | Sentry |
| Elasticsearch timeline | `ObservabilityVendorIntegrationContract` | Elasticsearch |
| **LKW-PR proof receipts** | **`ProofReceiptStore` → `DocumentStore` → `DocumentStoreVendorIntegrationContract`** | **MongoDB** |

---

## G. Out of scope (this track)

- Rewriting LKW.4E closeout as markdown-only proof
- pymongo imports in Tier-3 applications
- In-memory DocumentStore as live proof backend
- MongoDB Docker stack (1F)
- Public LKW Step 9 before live stack (1G)
