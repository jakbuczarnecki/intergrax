# Execution Identity Authority Hardening (HARDENING-7)

**Classification:** `MAINTAINER_HARDENING`  
**Status:** `COMPLETE` (HARDENING-7 audit on `development`)  
**Parent:** [`EXECUTION_IDENTITY_AUTHORITY_MODEL.md`](EXECUTION_IDENTITY_AUTHORITY_MODEL.md) (EE-A2)

---

## Single authority

| Surface | Role |
| --- | --- |
| `intergrax/contracts/execution_identity_authority.py` | Port contract (`ExecutionIdentityAuthorityPort`) — no minting |
| `intergrax/contracts/execution_identity.py` | Primitive mint helpers + validation + active context bind API |
| `intergrax/runtime/execution/identity_authority.py` | **Sole production mint owner** for Run / Attempt / Execution lifecycle |

`DefaultExecutionIdentityAuthority` delegates **ExecutionId** creation only through:

- `mint_root_execution_identity()` — root admission
- `mint_child_execution_id()` — child segments under an active parent tree

Run / Attempt minting remains on the authority port methods; background transport uses `mint_background_transport_identity()`.

---

## Identity flow

```text
ExecutionIdentityAuthority (identity_authority.py)
        |
        v
MintedExecutionIdentity / RootTaskIdentity
        |
        v
ExecutionRuntime + ExecutionBoundary (bind_active_execution_identity)
        |
        v
Workflow · Recovery · Evidence · Checkpoint · Audit
        |
        v
Execution Spine (unchanged — HARDENING-6)
```

Recovery and checkpoint planes **reuse** existing Run / Execution; retry transitions mint **AttemptId** only via `mint_retry_attempt_id()` / `AttemptLifecycleService`.

---

## Forbidden patterns (production runtime)

| Pattern | Why forbidden |
| --- | --- |
| `mint_execution_id()` outside `identity_authority.py` runtime authority functions | Side generator bypass |
| `ExecutionId(...)` construction outside contract validators | Non-canonical identity |
| `bind_active_execution_identity()` outside `execution/boundary.py` | Context hijack |
| Nexus / `background_execution` calling `mint_*` lifecycle IDs | Plane isolation (NPSC-3C-F-R1 gate) |
| Recovery resume minting a new ExecutionId for the same logical run | Identity fork |

---

## Documented exceptions (non-production)

Conformance and qualification harnesses may call contract `mint_*` helpers directly when building synthetic fixtures. Allowlisted in EE-A2-H2 global freeze:

- `runtime/execution/decision_finalization_conformance.py`
- `runtime/observability/persistence_conformance.py`
- `runtime/diagnostics/*_persistence_conformance.py`
- `collaborative_work/repository_qualification_suite.py`
- harness / qualification runners under `applications/_shared` and `core/qualification`

These are **not** execution admission paths.

---

## HARDENING-7 finding and fix

**Finding:** `DefaultExecutionIdentityAuthority` called `mint_execution_id()` inside port methods, failing `test_runtime_authority_module_mints_execution_id_only_in_runtime_authority_functions` (NPSC-3C-F-R1).

**Fix:** Port `mint_execution_identity` / `mint_child_execution_identity` delegate ExecutionId minting to `mint_root_execution_identity` and `mint_child_execution_id` respectively; only those module functions invoke `mint_execution_id()`.

**Regression gates:** `tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py`, EE-A2-H1/H2/H3 certification modules.

---

## Audit summary (intergrax/)

| Area | Result |
| --- | --- |
| Execution / Nexus / background mint scan | No forbidden `mint_*` outside owner (exempt conformance only) |
| Self-healing / autonomy | No direct `mint_execution_id`; autonomy loop uses `default_execution_identity_authority.mint_execution_identity` |
| Recovery / checkpoint | No primitive execution mint in recovery modules (EE-A2 frozen plane tests) |
| Evidence correlation | Events carry caller-supplied execution tuple; EventId mint is evidence-scoped, not execution lifecycle |
