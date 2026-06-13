# attestation_demo — Implementation Plan

**The implementation map** for this Tier-3 partner PoC host — phases, status, and verification.

Status: **Done** (PoC v1 — ready for partner sharing)

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Application ADRs: [`adr/README.md`](adr/README.md)

---

## Documentation model

| Topic | Where |
|-------|--------|
| Host purpose, EBE contract, trust model | **ARCHITECTURE.md** |
| Task queue and verification | **This file** |
| Partner quickstart + sample payloads | **README.md** · **partner_handoff/** |
| Deploy runbook | **BUILD_AND_DEPLOY.md** |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| EBE-1 | `execution_boundary_event.v1` + invoker hook + memory buffer | **Done** | High | `intergrax/runtime/attestation/` |
| EBE-2 | `ExecutionBoundaryExportProfile` + wiring bridge | **Done** | High | `attestation_runtime_bridge.py` |
| EBE-3 | `attestation_demo` host + `POST /poc/run` | **Done** | High | Tier-3 scaffold layout |
| EBE-4 | `boundary_demo_agent` + `records.put` lab wiring | **Done** | High | `host/tool_wiring.py` |
| EBE-5 | README + sample JSON + trust model | **Done** | High | Partner handoff |
| EBE-6 | Platform OBSERVABILITY pair + harness ADR + partner handoff | **Done** | High | ADR-OBS-002, `partner_handoff/` |
| EBE-7 | Webhook sink | Deferred | Low | Phase 2 |
| EBE-8 | HarnessKernel step-level events | Deferred | Low | Phase 2 |
| EBE-9 | Host-side event signing | Deferred | Low | Phase 2 |

---

## 2. Verification

```bash
uv run pytest applications/attestation_demo/attestation_demo_tests -q
uv run pytest tests/unit/runtime/attestation/ -q
uv run pytest tests/unit/applications/test_application_deploy_triad.py -q -k attestation_demo
uv run pytest tests/unit/applications/test_agent_app_doc_pair.py -q -k attestation_demo
uv run pytest tests/unit/scaffold/test_adr_scaffold.py -q -k attestation_demo
python scripts/check_harness_adr.py
```
