# boundary_demo agent - Implementation Plan

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../docs/project/architecture/intergrax_runtime_architecture.md)
Host tracker: [`applications/attestation_demo/docs/IMPLEMENTATION_PLAN.md`](../../../applications/attestation_demo/docs/IMPLEMENTATION_PLAN.md)

Principle: **stable PoC agent** · **no receipt logic in Tier-2** · **no Tier-3 imports in agent code**

---

## Documentation model

| Topic | Where |
|-------|--------|
| Purpose, contracts, runtime layout | **ARCHITECTURE.md** (this directory) |
| Task status, phases | **This file** |
| Agent architecture decisions | **`docs/project/technical/adr`** - [`adr/README.md`](adr/README.md) |
| EBE platform / host work | `../../applications/attestation_demo` |

---

## 0. Scope at a glance

| Field | Value |
|-------|-------|
| Agent id | `boundary_demo_agent` |
| Class | `BoundaryDemoAgent` |
| Primary capability | `attestation.demo` |
| Tier | Tier-2 (`agents/boundary_demo`) |
| Host | `applications/attestation_demo` |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| BOUNDARY-1 | UAEP `records.put` step | **Done** | High | `boundary_demo_agent.py` |
| BOUNDARY-2 | Registry skill/tool resolution (empty author `allowed_tools`) | **Done** | High | `test_boundary_demo_skill_resolution.py` |
| BOUNDARY-3 | Attestation demo host wiring | **Done** | High | `attestation_demo/host/tool_wiring.py` |
| BOUNDARY-4 | Agent smoke tests | **Done** | High | `agents/boundary_demo/tests` |
| BOUNDARY-5 | Docs layout (`docs`) | **Done** | Medium | Scaffold parity |

---

## 2. Verification

```bash
uv run pytest agents/boundary_demo/tests -q
uv run pytest tests/unit/agents/test_boundary_demo_skill_resolution.py -q
uv run pytest applications/attestation_demo/tests -q
```
