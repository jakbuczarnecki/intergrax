# external_contractor_adapter — Implementation Plan

**The implementation map** for the Tier-2 GEC adapter agent.

**Status:** Working draft (2026-07-20) — **GEC-0…GEC-3 Done** (provider-neutral mapping baseline); HITL resume deferred to host **GEC-4**  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Host tracker:** [`applications/governed_contractor_application/docs/IMPLEMENTATION_PLAN.md`](../../../applications/governed_contractor_application/docs/IMPLEMENTATION_PLAN.md)  
**Agent ADRs:** [`adr/README.md`](adr/README.md)

Principle: **domain adapter only** · **reuse Tier-0** · **no Tier-3 imports** · **no orchestration ownership** · **mapping ≠ governance**

---

## Documentation model

| Topic | Where |
|-------|--------|
| Purpose, ownership, prohibited duties | **ARCHITECTURE.md** |
| Task status | **This file** |
| Agent ADRs | **`docs/adr/`** |
| Vertical phases GEC-0…GEC-11 | Host `IMPLEMENTATION_PLAN.md` |

---

## 0. Scope at a glance

| Field | Value |
|-------|-------|
| Agent id | `external_contractor_adapter` |
| Class | `ExternalContractorAdapterAgent` |
| Mapper | `ExternalWorkAdapter` |
| Primary capability | `external_contractor.adapt` |
| Pattern | ACP reflex |
| Tier | Tier-2 (`agents/external_contractor_adapter/`) |
| Host | `applications/governed_contractor_application/` |
| Integration | Injected `ExternalWorkIntegration` only |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| GEC-0 | Canonical ACP scaffold + architecture docs | **Done** | High | `new-agent` |
| GEC-1 | Platform external-work contracts (consume-only) | **Done** | High | Owned under `intergrax/contracts/` |
| GEC-2 | Provider-neutral `ExternalWorkIntegration` (platform) | **Done** | High | Consume-only; ADR-EXTWORK-002 |
| GEC-3.1 | Consume platform contractor contracts (GEC-1) | **Done** | High | No app-local contracts |
| GEC-3.2 | Wire `ExternalWorkIntegration` via DI | **Done** | High | Agent ctor + host `settings.external_work_integration` |
| GEC-3.3 | Lifecycle mapping in `ExternalWorkAdapter` / `steps/` | **Done** | High | Sync only; idempotent correlate |
| GEC-3.4 | Status + evidence normalization | **Done** | High | Timeline / deliverables / evidence refs |
| GEC-3.5 | Resume/stop on HITL decision signal | Deferred | High | Host **GEC-4** — adapter may forward evidence only |
| GEC-A1 | Extend prompts only if needed | Planned | Low | Prefer deterministic Protocol calls |
| GEC-A2 | Agent ADR for pattern evolution (if leaving reflex) | Planned | Medium | |

Host-owned phases (HITL UX, policy packs, receipts, public API, partner handoff, stub/live proof) remain in the application plan.

---

## 2. Verification

```bash
uv run pytest agents/external_contractor_adapter/tests -q
uv run pytest applications/governed_contractor_application/tests -q
```

---

## 3. Non-goals (agent package)

- Quote acceptance, payment approval, policy decisions
- Transport (HTTP / A2A / REST / JSON-RPC)
- Partner URL hardcoding / provider registries
- Local competing contractor agent
- ProofReceipt store ownership
- Polling, background workers, retry engines, resume ownership
- Marking host GEC-4…GEC-11 complete from this file
