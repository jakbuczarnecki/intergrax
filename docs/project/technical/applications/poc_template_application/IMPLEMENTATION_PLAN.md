# poc_template_application — Implementation Plan

**The implementation map** for this Tier-3 reference shell — phases, status, gaps, and verification.

Status: **Done** (canonical lab scaffold reference)

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../architecture/intergrax_runtime_architecture.md) · Phase AA-POC

---

## Documentation model

| Topic | Where |
|-------|--------|
| Minimal H-APP host layout | **ARCHITECTURE.md** |
| Conformance with scaffold CLI | **This file** |
| Authoring guide | `applications/TIER3_READINESS.md` |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| POC-1 | Manifest + `build_harness_host_runtime` | **Done** | High | AA-POC.1 |
| POC-2 | Parity with `new-application` scaffold | **Done** | High | AA-POC.2 |
| POC-3 | Deploy triad | **Done** | High | Docker gate |
| POC-4 | Stay minimal — do not accrete product logic | Ongoing | High | Copy via scaffold, not edit in place |

---

## 2. Verification

```bash
uv run pytest applications/poc_template_application/tests -q
uv run pytest tests/unit/applications/test_scaffold_application.py -q
```

New apps: `python -m intergrax.scaffold new-stack <name> --profile lab`
