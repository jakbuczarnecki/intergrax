# Llm Adapters — Implementation Plan

**Architecture (1:1):** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (LLM_ADAPTERS plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites/) satellites on demand. Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/LLM_ADAPTERS.md`](../guides/audit_slices/LLM_ADAPTERS.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/LLM_ADAPTERS_audit_history.md`](plan/LLM_ADAPTERS_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-18) — all AUDIT-IDEAL §6 rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-6.1 | §6 LLM | Structured output validation on 100% reference + certified agent paths | P1 | **Done** |
| AUDIT-IDEAL-6.2 | §6 LLM | Live cost/latency/quality model routing (AHI prod path) | P2 | **Done** — `check_live_model_routing_wiring.py` |
| AUDIT-IDEAL-6.3 | §6 LLM | Central `ModelCatalog` + unified context window resolution | P0 | **Done** — `CatalogCapabilityAdapter` |
| AUDIT-IDEAL-6.4 | §6 LLM | Tokenizer-consistent context preflight (adapter path) | P0 | **Done** — `count_message_tokens(adapter=)` |
| AUDIT-IDEAL-6.5 | §6 LLM | Profile failover chain on retriable provider errors | P1 | **Done** — LC-3 |
| AUDIT-IDEAL-6.6 | §6 LLM | ACP `StepLLMRouter` backed by `LLMAdapter` (single DX) | P1 | **Done** — M-LLM-X.5.4 |
| AUDIT-IDEAL-6.7 | §6 LLM | Developer `USAGE.md` + startup validation | P2 | **Done** — `check_llm_profile_runtime.py` + doctor |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

## Phase M-LLM-X — LLM Developer Excellence (post-audit 2026-06-14)

**Source:** Deep production audit 2026-06-14 — contract L3, model metadata L1–L2, routing L1–L2, DX L2.  
**Canon:** [`architecture/LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) §Model catalog · §Routing · §Audit register  
**Goal:** Elevate Tier-0 LLM layer from **production L3 foundation** to **best-in-class developer engine** — correct context for any model string, unified token accounting, runtime routing/failover, single agent API.  
**Status:** **LC baseline Done** (2026-06-14) — P0/P1 closed · **M-LLM-X partial waves** = P2+ backlog (not blocking layer maturity)  
**Priority ladder:** Band **2ba** (after M-LLM-R closeout) · queue [§6.1ax](#61ax-harness-implementation-queue--llm-developer-excellence-m-llm-x)  
**Execution order:** [§6.2af](#62af-phase-m-llm-x-execution-order)  
**Target maturity:** Model metadata **L3**, routing **L3**, DX **L3+** (see architecture maturity table).

**Hard rules (non-negotiable):**

- **No** new per-adapter hardcoded context dicts without catalog entry or prefix rule — migrate to `ModelCatalog`.
- **`LLMProfile.options["context_window_tokens"]`** MUST override catalog for **all** providers (not Ollama-only).
- **Preflight / history budget** MUST use `adapter.count_messages_tokens` when adapter is in scope.
- **No** vendor SDK imports in Tier-2 — unchanged tier boundary.
- One **M-LLM-X.\*** task group per PR → update master table + architecture audit register → gate green.
- **ADR:** [ADR-LLM-002](../adr/entries/2026-06-14/ADR-LLM-002.md) **Done** — prerequisite for M-LLM-X.1 code merge.  
**ADR:** [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md) **Accepted** — prerequisite for M-LLM-X.9 code merge.

**Explicitly excluded:** Central LLM gateway microservice (needs separate platform ADR), rewriting all 19 SDK clients, product HTTP DTOs, Phase K agents.

---
