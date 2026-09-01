# legal agent - Implementation Plan

**The implementation map** for this Tier-2 agent - phases, status, gaps, and verification.

Status: **Scaffold Done** - domain UAEP steps **Band 3** (explicit reprioritization required)

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Spec input: [`SPEC_FROM_LEGACY.md`](SPEC_FROM_LEGACY.md)  
Host: [`applications/legal_application`](../../../applications/legal_application/)
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../docs/project/architecture/intergrax_runtime_architecture.md) · Phase AA-LEG

---

## Documentation model

| Topic | Where |
|-------|--------|
| UAEP layout, skills, runtime | **ARCHITECTURE.md** |
| Task status, domain port queue | **This file** |
| Legacy behavioral requirements | `SPEC_FROM_LEGACY.md` |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| LEG-1 | Hard reset to UAEP scaffold | **Done** | High | Phase AA-LEG |
| LEG-2 | `legal.contract_review` skill on contract | **Done** | High | Tier-0 skill pack |
| LEG-3 | ARCHITECTURE baseline | **Done** | High | AA-LEG.1.3 |
| LEG-4 | Port domain steps from legacy spec | Planned | High | One PR per `@step` - Band 3 |
| LEG-5 | Observability + policy hooks | Planned | Medium | With host `legal_application` |

---

## 2. Verification

```bash
uv run pytest agents/legal/tests -q
uv run pytest applications/legal_application/tests -q
```
