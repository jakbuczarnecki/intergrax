# Proof Receipts — Implementation Plan

**Architecture (1:1):** [`architecture/PROOF_RECEIPTS.md`](../architecture/PROOF_RECEIPTS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Proof consumer:** LKW-PR ([`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PR)  
**Last updated:** 2026-07-10 — **PROOF-RECEIPTS-1A closed**

---

## A. Status

| Item | Value |
|------|-------|
| **Track status** | In progress — architecture + contracts defined |
| **Architecture** | [`architecture/PROOF_RECEIPTS.md`](../architecture/PROOF_RECEIPTS.md) — **PROOF-RECEIPTS-1A closed** |
| **Current proof consumer** | LKW (structured receipt after live proofs) |
| **Default live vendor** | MongoDB (`document_store` integration) |
| **Next step** | **PROOF-RECEIPTS-1B** — complete MongoDB DocumentStore provider wiring if gaps exist |

**Note:** LKW.4E (Kafka-backed background task live proof) remains **closed**. Proof receipts are the **next platform proof wave** — not a markdown closeout of LKW.4E.

Historical **LKW.5** in the LKW plan refers to the **closed persistence proof** (`LKW_DATA_HOME` + Qdrant). The proof-receipt wave uses **LKW-PR** to avoid renumbering that closed wave.

---

## B. Staged rollout

| ID | Task | Scope | Status |
|----|------|-------|--------|
| **PROOF-RECEIPTS-1A** | Define ProofReceipt contract and MongoDB-backed storage architecture | `ProofReceipt` model; `DocumentRecord` mapping; `ProofReceiptStore`; architecture + plan docs; unit tests | **Done** |
| **PROOF-RECEIPTS-1B** | Complete MongoDB DocumentStore provider wiring | Close any gaps in `integrations/providers/document_store/mongodb/` registration, profile resolution, and conformance | Planned |
| **PROOF-RECEIPTS-1C** | Wire ProofReceiptStore through IntegrationProfile/config | Host/application wiring selects `DocumentStore` from profile; expose store to proof workloads | Planned |
| **PROOF-RECEIPTS-1D** | Record LKW proof receipts through platform engine | LKW proof scripts write `ProofReceipt` via `ProofReceiptStore` after live proofs | Planned |
| **PROOF-RECEIPTS-1E** | Add MongoDB + Mongo Express Docker proof stack | Compose overlay for live vendor-backed receipt persistence | Planned |
| **PROOF-RECEIPTS-1F** | Add reviewer-visible Step 9 to LKW_PLATFORM_PROOF.md | Public reviewer path for MongoDB receipt inspection — only after live stack ships | Planned |

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

## D. PROOF-RECEIPTS-1B scope (next)

| Deliverable | Detail |
|-------------|--------|
| Provider registration | Ensure `mongodb` document store resolves from catalog/profile |
| Conformance | `assert_document_store` passes for MongoDB adapter |
| Config | `INTERGRAX_MONGODB_URI`, database name env alignment |
| Gap report | Document any blockers before 1C wiring |

**Depends on:** PROOF-RECEIPTS-1A  
**Blocks:** PROOF-RECEIPTS-1C

---

## E. Integration analogy

| Existing proof | Platform surface | Vendor |
|----------------|------------------|--------|
| LKW.4E background tasks | `message_bus.*` | Kafka |
| LKW-OBS / Sentry path | Observability export / problem signals | Sentry |
| Elasticsearch timeline | Observability vendor integration | Elasticsearch |
| **LKW-PR proof receipts** | **`ProofReceiptStore` → `DocumentStore`** | **MongoDB** |

---

## F. Out of scope (this track)

- Rewriting LKW.4E closeout as markdown-only proof
- pymongo imports in Tier-3 applications
- In-memory DocumentStore as live proof backend
- MongoDB Docker stack (1E)
- Public LKW Step 9 before live stack (1F)
