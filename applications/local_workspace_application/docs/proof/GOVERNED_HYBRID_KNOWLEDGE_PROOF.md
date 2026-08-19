# Governed Hybrid Knowledge Proof

**Intergrax determines what evidence is required for an answer to be admissible, resolves it through authorized indexed and live sources, revalidates authority at execution time, and preserves structural proof of why a past answer was valid.**

```mermaid
flowchart LR
  POLICY["Approved Indexed Policy"]
  LIVE["Current External Status"]
  AUTH["Runtime Authority"]

  POLICY --> REQ["Evidence Requirements"]
  LIVE --> ACQ["Authorized Evidence Acquisition"]
  AUTH --> ACQ

  REQ --> GATE["Evidence Admissibility Gate"]
  ACQ --> GATE

  GATE -->|SATISFIED| ANSWER["Answer allowed"]
  GATE -->|UNSATISFIED| STOP["No answer"]

  ANSWER --> HISTORY["Durable Structural Proof"]
  STOP --> HISTORY

  classDef indexed fill:#e8f4fc,stroke:#2563eb,color:#1e3a5f
  classDef live fill:#fef3e2,stroke:#d97706,color:#78350f
  classDef gov fill:#f3e8ff,stroke:#7c3aed,color:#4c1d95
  classDef gate fill:#ecfdf5,stroke:#059669,color:#064e3b

  class POLICY indexed
  class LIVE live
  class AUTH gov
  class GATE gate
```

---

## Quick start

**Prerequisites:** Python 3.12, `uv`, repo checkout. No cloud credentials. No manual HTTP service.

```bash
uv run python -m proof_infrastructure.governed_hybrid_knowledge_proof
```

Optional machine-readable output:

```bash
uv run python -m proof_infrastructure.governed_hybrid_knowledge_proof --json
```

**Duration:** local deterministic proof (seconds on a developer machine).

---

## What this proof demonstrates

| Guarantee | Mechanism exercised |
|-----------|---------------------|
| Explicit evidence requirements | HYBRID Query Policy + product/provider obligations |
| Indexed + live admissibility | `evaluate_execution_admissibility` |
| Authority revalidated at execution | `WorkspaceLiveAccessRuntimeAuthority` |
| Required evidence failure blocks synthesis | `INSUFFICIENT_EVIDENCE`, LLM calls = 0 |
| Provider non-invocation is measurable | Project Status `read_request_count` |
| Structural historical proof | `WorkspaceAskServiceV2.get_run` / `WorkspaceAskRepository` |

This document claims **Intergrax behavior only** — not comparisons to other platforms.

---

## Story — ORION deployment readiness

**Project:** ORION  
**Question (all scenarios):** `Is ORION ready for deployment?`

**Approved indexed policy** (managed document path):

```text
Deployment Policy — Approved

A project is ready for deployment only when:
1. readiness score is at least 90
2. no security blocker is OPEN
```

**External live status (controlled HTTP):** readiness `94`, blocker `SEC-417`.

---

## 01 — Reality matters

```mermaid
sequenceDiagram
  participant P as Indexed Policy
  participant L as Live Project Status
  participant G as Admissibility
  participant A as Answer
  Note over L: SEC-417 OPEN
  P->>G: policy obligation satisfied
  L->>G: live obligation satisfied
  G->>A: SATISFIED
  A-->>Reviewer: NO
```

| Check | Expected |
|-------|----------|
| Live HTTP reads | 1 |
| Admissibility | SATISFIED |
| Decision | **NO** |

---

## 02 — Freshness matters

Only external state changes: `SEC-417` OPEN → CLOSED.

```mermaid
stateDiagram-v2
  [*] --> OPEN: initial fixture
  OPEN --> CLOSED: control API only
  note right of CLOSED
    Same policy, query, binding, adapter
  end note
```

| Check | Expected |
|-------|----------|
| Live HTTP reads | 1 |
| Admissibility | SATISFIED |
| Decision | **YES** |

---

## 03 — Authority matters

Binding ACTIVE during planning; **DISABLED** after indexed retrieval, before live HTTP.

```mermaid
flowchart TD
  PLAN["Evidence plan authorized"]
  IDX["Indexed retrieval completes"]
  REV["Binding DISABLED via configuration"]
  AUTH["Runtime authority denies"]
  HTTP["Project Status HTTP reads = 0"]
  GATE["Admissibility UNSATISFIED"]
  PLAN --> IDX --> REV --> AUTH --> HTTP
  AUTH --> GATE
```

| Check | Expected |
|-------|----------|
| Live HTTP reads | **0** |
| LLM calls | **0** |
| Ask status | `INSUFFICIENT_EVIDENCE` |
| Decision | **CANNOT DETERMINE** |

---

## 04 — History matters

Current live state may be CLOSED; historical Ask #1 still explains **why NO was valid then**.

```mermaid
flowchart LR
  RUN["Persisted Ask run"]
  RUN --> CFG["configuration_revision"]
  RUN --> OBL["required_evidence_obligations"]
  RUN --> IDX["indexed provenance"]
  RUN --> LIVE["live provenance hash"]
  RUN --> ADM["admissibility SATISFIED"]
  RUN --> ANS["answer NO"]
```

**Limitation (explicit):** EPHEMERAL live bodies are **not** durably retained. Historical proof uses structural identity — `content_hash`, binding/capability IDs, timestamps, admissibility — not raw live payload replay.

---

## Expected terminal output (representative)

```text
============================================================
INTERGRAX — GOVERNED HYBRID KNOWLEDGE PROOF
============================================================

01 REALITY — current blocker changes the decision
HTTP reads: 1
Admissibility: SATISFIED
Decision: NO
RESULT: PASS

02 FRESHNESS — reality changes, policy does not
HTTP reads: 1
Admissibility: SATISFIED
Decision: YES
RESULT: PASS

03 AUTHORITY — revoked means physically not called
HTTP reads: 0
Admissibility: UNSATISFIED
LLM calls: 0
Decision: CANNOT DETERMINE
RESULT: PASS

04 HISTORY — why was NO valid then?
Decision: NO
RESULT: PASS

============================================================
4 / 4 PROOFS PASSED
============================================================
```

---

## Real boundaries exercised

| Layer | Component |
|-------|-----------|
| Application | `WorkspaceAskServiceV2` |
| Indexed path | managed local document → `WorkspaceDocumentIndexingService` / `local.workspace.index` → `local.workspace.search` → `WorkspaceIndexedEvidenceRetrieverV1` |
| Connection | `TenantConnection` → `TenantConnectionRehydrator` → `KnowledgeConnectionRegistry` → `KnowledgeConnectionRegistryIntegrationResolverV1` |
| Live path | `LiveCapabilityExecutorV1` + `ProjectStatusReadLiveHandlerV1` |
| Authority revoke | `LiveAccessLifecycleService.disable` → `WorkspaceLiveAccessRuntimeAuthority` reload |
| Admissibility | `evaluate_execution_admissibility` |
| Synthesis | `HybridAskAnswerAssemblerV2` + deterministic proof LLM |
| Persistence | `WorkspaceAskRepository.get_run_v2` |
| External HTTP | `proof_infrastructure.controlled_project_status_service` |

No fake search-result injection, no manual integration registration, no direct configuration mutation by the proof harness.

---

## Architecture deep link

Hybrid Ask architecture and COMM-5C3 Project Status boundary:

[`HYBRID_ASK_ARCHITECTURE.md`](../HYBRID_ASK_ARCHITECTURE.md)

---

## Automated tests

```bash
uv run pytest tests/unit/proof_infrastructure/test_governed_hybrid_knowledge_proof.py -v
```
