# external_contractor_adapter — architecture

**Status:** GEC-0 scaffold + GEC-1 contracts + GEC-2 `ExternalWorkIntegration` available (2026-07-20) — ACP **reflex** stub; domain adapter logic planned GEC-3  
**Vertical:** Governed External Contractor (GEC)  
**Implementation tracker:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
**Agent ADRs:** [`adr/README.md`](adr/README.md)  
**Host architecture:** [`applications/governed_contractor_application/docs/ARCHITECTURE.md`](../../../applications/governed_contractor_application/docs/ARCHITECTURE.md)

This agent is a **Tier-2 domain adapter**, not a second orchestration system. Nexus owns multi-step task orchestration; this package maps an external contractor product into Intergrax contracts.

---

## Purpose

Adapt an external A2A-style contractor agent into the governed GEC flow:

```text
Nexus Task
  → ExternalContractorAdapterAgent
  → External contractor integration
  → External A2A Contractor Agent
```

The external product remains responsible for domain execution. This adapter owns discovery, lifecycle mapping, correlation, and normalization only.

---

## Capabilities

| Capability | Role |
|------------|------|
| `external_contractor.adapt` | Primary adapter capability (scaffold + host default) |

Additional fine-grained capabilities (quote sync, deliverable fetch, etc.) may be added in GEC-3 without moving ownership of HITL or policy into this package.

---

## Typed inputs and outputs (GEC-1 platform; wired in GEC-3)

| Direction | Platform type | Notes |
|-----------|---------------|-------|
| In | Governed `task_id` / `run_id` + `ExternalTaskCorrelation` | Intergrax identity remains primary |
| In | `QuoteAcceptanceEvidence` (from runtime HITL refs) | Adapter consumes; does not create |
| Out | `CommercialQuote` / `MoneyAmount` | For Tier-3 presentation |
| Out | `ExternalWorkStatus` timeline | Not Nexus `TaskState` |
| Out | `ExternalDeliverableRef` | Workspace-safe resource URI |
| Out | Normalized tool/evidence facts | For receipts — not partner hardcoding |

Canonical modules: `intergrax.contracts.external_work`, `intergrax.contracts.money` ([ADR-EXTWORK-001](../../../docs/adr/entries/2026-07-20/ADR-EXTWORK-001.md)).

Integration boundary (GEC-2 Done): `intergrax.integrations.contracts.external_work.ExternalWorkIntegration` ([ADR-EXTWORK-002](../../../docs/adr/entries/2026-07-20/ADR-EXTWORK-002.md)).

---

## External integration dependency

Depends on the **provider-neutral `ExternalWorkIntegration`** Protocol (GEC-2). GEC-3 will inject a host-bound implementation and map lifecycle steps onto `discover` / `create_work` / `get_quote` / `submit_quote_acceptance` / `get_work` / `get_timeline` / `get_deliverables` / `get_evidence`. Partner URLs and credentials are supplied by the Tier-3 host — never hardcoded in this agent. The agent must not own the integration boundary.

---

## Lifecycle states (adapter view)

```text
discover_card
  → create_or_correlate_external_task
  → fetch_quote
  → await_acceptance (runtime HITL — not adapter-owned)
  → continue_after_accept | stop_after_reject
  → sync_status
  → fetch_deliverables
  → emit_normalized_evidence
```

Idempotency keys must cover create/continue/status transitions (GEC-3).

---

## `Agent.run()` / typed `on_next_step` alignment

| Item | GEC-0 scaffold | Target |
|------|----------------|--------|
| Pattern | ACP **reflex** (`CognitivePattern.REFLEX`) | May evolve (ADR) if multi-phase mapping needs another pattern |
| Entry | Typed cognitive hooks / `on_next_step` | Domain work in `steps/domain_job.py` (and successors) |
| LLM | Stub adapter for offline smoke | Real LLM only if mapping needs it; prefer deterministic tool/integration calls |

Do **not** build an internal orchestration graph that duplicates Nexus.

---

## Ownership matrix

| Concern | Owner |
|---------|-------|
| Nexus task graph / scheduling | Runtime Nexus |
| Policy allow/deny | Runtime policy + Tier-3 bundles |
| HITL quote accept/reject | Runtime HITL + Tier-3 surfaces |
| Agent Card discovery, quote fetch, status sync, deliverable fetch | **This adapter** |
| Wallet / payment approval | Tier-3 / runtime — **prohibited here** |
| Workspace escape / external publication approval | Tier-3 / runtime — **prohibited here** |
| Reusable contracts | `intergrax/` — **not** this package as long-term home |

---

## Idempotency and retries

| Requirement | Rule |
|-------------|------|
| Idempotency | External create/continue must use deterministic keys from Intergrax identity + stage |
| Retries | Safe for reads/status; mutating calls only with idempotency |
| Crash recovery | Correlate existing external task; do not double-create when key present |

---

## External task correlation

Maintain a stable mapping:

```text
intergrax run_id / task_id  ↔  external_task_id  ↔  quote_id
```

Surface correlation in adapter outputs so Tier-3 receipts and traces can join facts.

---

## Status and evidence normalization

| External concept | Adapter duty |
|------------------|--------------|
| Partner status enums/strings | Map to governed status model (GEC-1) |
| Tool calls / work evidence | Normalize to platform evidence shapes |
| Opaque partner blobs | Do not require core to understand partner JSON |

---

## Prohibited responsibilities

- Quote acceptance or rejection decisions
- Wallet or payment approval
- Policy decisions or governance bypasses
- Workspace allowlist escape
- External publication approval
- Competing domain contractor implementation (for example local code review)
- Importing `applications.*`
- Owning Nexus orchestration

---

## Layout

| Path | Role |
|------|------|
| `external_contractor_adapter_agent.py` | `ExternalContractorAdapterAgent` — cognitive pattern hooks |
| `contract.py` | `AgentContract` + `cognitive_pattern` |
| `capabilities.py` | Capability ids |
| `steps/domain_job.py` | Domain step entry — GEC-3 implementation point |
| `prompts/system.md` | Prompt assets |
| `schemas/` | Agent-local I/O helpers (prefer platform contracts when shared) |
| `tests/` | Agent smoke / unit tests |
| `docs/adr/` | Agent ADRs |

---

## Tier hygiene

- Imports only `intergrax.*` and `external_contractor_adapter.*`
- Tools / integrations resolved by Tier-3 host profiles
- Registration: `AgentBinding.mount(...)` in `applications/governed_contractor_application/manifest.py`
