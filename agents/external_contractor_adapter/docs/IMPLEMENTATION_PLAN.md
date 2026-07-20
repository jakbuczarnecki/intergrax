# external_contractor_adapter — Implementation Plan

**The implementation map** for the Tier-2 GEC adapter agent.

**Status:** Working draft (2026-07-20) — **GEC-0 scaffold Done**; domain adapter work tracked under host plan **GEC-3**  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Host tracker:** [`applications/governed_contractor_application/docs/IMPLEMENTATION_PLAN.md`](../../../applications/governed_contractor_application/docs/IMPLEMENTATION_PLAN.md)  
**Agent ADRs:** [`adr/README.md`](adr/README.md)

Principle: **domain adapter only** · **reuse Tier-0** · **no Tier-3 imports** · **no orchestration ownership**

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
| Primary capability | `external_contractor.adapt` |
| Pattern | ACP reflex (scaffold) |
| Tier | Tier-2 (`agents/external_contractor_adapter/`) |
| Host | `applications/governed_contractor_application/` |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| GEC-0 | Canonical ACP scaffold + architecture docs | **Done** | High | `new-agent` |
| GEC-3.1 | Consume platform contractor contracts (GEC-1) | Planned | High | No app-local contracts |
| GEC-3.2 | Wire external contractor integration (GEC-2) | Planned | High | Provider-neutral |
| GEC-3.3 | Implement lifecycle mapping in `steps/` | Planned | High | Idempotent correlate |
| GEC-3.4 | Status + evidence normalization | Planned | High | |
| GEC-3.5 | Resume/stop on HITL decision signal | Planned | High | No accept ownership |
| GEC-A1 | Extend prompts only if needed | Planned | Low | Prefer deterministic tools |
| GEC-A2 | Agent ADR for pattern evolution (if leaving reflex) | Planned | Medium | |

Host-owned phases (HITL UX, policy packs, receipts, public API, partner handoff, stub/live proof) remain in the application plan — not duplicated as Done here.

---

## 2. Verification

```bash
uv run pytest agents/external_contractor_adapter/tests -q
uv run pytest applications/governed_contractor_application/tests -q
```

---

## 3. Non-goals (agent package)

- Quote acceptance, payment approval, policy decisions
- Partner URL hardcoding
- Local competing contractor agent
- ProofReceipt store ownership
- Marking GEC-3…GEC-11 complete from this file alone
