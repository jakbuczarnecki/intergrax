# signoff_probe agent — Implementation Plan

**The implementation map** for this Tier-2 harness probe — phases, status, gaps, and verification.

Status: **Done** (Appendix A sign-off probe)

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../docs/project/architecture/intergrax_runtime_architecture.md) · Phase AA-SIG

---

## Documentation model

| Topic | Where |
|-------|--------|
| Sign-off flow, capability `signoff.probe` | **ARCHITECTURE.md** |
| Task status | **This file** |
| Harness GA checklist | Platform plan **Appendix A** |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| SIG-1 | Scaffold layout parity | **Done** | High | `test_signoff_probe_matches_scaffold_layout` |
| SIG-2 | UAEP smoke + capability contract | **Done** | High | Gate integration |
| SIG-3 | ARCHITECTURE + IMPLEMENTATION_PLAN | **Done** | Medium | Phase AA conformance |
| SIG-4 | Domain expansion | N/A | — | Probe only — do not grow into product agent |

---

## 2. Verification

```bash
uv run pytest agents/signoff_probe/tests -q
uv run pytest tests/unit/scaffold/test_signoff_scaffold_parity.py -q
```
