# lab_application — Implementation Plan

**The implementation map** for this Tier-3 harness lab — phases, status, gaps, and verification.

Status: **Done** (operational harness lab) — maintenance via §6.1

Architecture: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md)  
Platform plan: [`docs/intergrax_runtime_architecture.md`](../../docs/intergrax_runtime_architecture.md) · Phase AA-LABAPP

---

## Documentation model

| Topic | Where |
|-------|--------|
| Debug API, manifest flags, integrations | **ARCHITECTURE.md** |
| Harness maintenance tasks | **This file** + platform **§6.1** |
| Deploy | `BUILD_AND_DEPLOY.md` |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| LABAPP-1 | `build_harness_host_runtime` factory | **Done** | High | AA-LABAPP.2 |
| LABAPP-2 | Dynamic manifest roster from settings | **Done** | High | Echo, research, org worker flags |
| LABAPP-3 | Deploy triad + gate | **Done** | High | `test_application_deploy_triad` |
| LABAPP-4 | Adaptive observe profile default | **Done** | Medium | L4-O — `LAB_ADAPTIVE_OBSERVE` |
| LABAPP-5 | New harness features | Maintenance | Medium | Platform §6.1 only — no product scope |

---

## 2. Verification

```bash
uv run pytest applications/lab_application/lab_application_tests -q
uv run pytest tests/unit/applications/test_application_deploy_triad.py -q
```
