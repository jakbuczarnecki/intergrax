# legal_application - Implementation Plan

**The implementation map** for this Tier-3 product host - phases, status, gaps, and verification.

Status: **Shell Done** - domain steps live in `agents/legal` (Band 3)

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
Agent plan: [`agents/legal/docs/IMPLEMENTATION_PLAN.md`](../../../agents/legal/docs/IMPLEMENTATION_PLAN.md)
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../docs/project/architecture/intergrax_runtime_architecture.md) · Phase AA-LEGAPP

---

## Documentation model

| Topic | Where |
|-------|--------|
| Product host, auth, serving, deploy | **ARCHITECTURE.md** |
| Host vs agent task split | **This file** + `agents/legal/docs/IMPLEMENTATION_PLAN.md` |
| Deploy | `BUILD_AND_DEPLOY.md` |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| LEGAPP-1 | Product manifest + environment profile | **Done** | High | AA-LEGAPP.1 |
| LEGAPP-2 | `build_harness_host_runtime` wiring | **Done** | High | No legacy runtime bridge |
| LEGAPP-3 | Deploy triad + compliance smoke route | **Done** | High | Gate deploy triad |
| LEGAPP-4 | Serving API extensions | Planned | Medium | With LEG-4 agent steps |
| LEGAPP-5 | Production auth hardening | Planned | Medium | Env + policy bundle |

---

## 2. Verification

```bash
uv run pytest applications/legal_application/tests -q
uv run pytest tests/unit/applications/test_application_deploy_triad.py -q
```
