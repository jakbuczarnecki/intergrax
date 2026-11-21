# T-IGX-051-C — ARCH-FREEZE remediation T-AF2-RW-TRUST (trust boundary wave)

## Scope

Delivery ticket for **T-AF2-RW-TRUST** / WP-12 W3 trust-boundary remediation on the `development` tor. Master-plan mapping: `llm/messages.py`, `graph_assembly.py`, `context/provider_lifecycle.py`, `context/planner.py` and related consumers.

Residuals routed here (MEP-AF2 v2.1, PDC table A): **EX-D2-01**, **EX-D2-02**, **EX-D2-05**, **EX-D2-07**, **EX-L-05**, **PDC-D2-R004** (RAG red root cause where product-bound).

Design goals: trust classes for model messages (untrusted content must not reach `role=system`), most-restrictive egress default, orphan `tool_call_id` fail-closed in STRICT, planner ledger sizing fidelity (**EX-D2-05**).

Write set is **disjoint** from PT-FC3 embedding/vector paths and from PT-WF harness evaluation paths (no planned edit of `harness_host_runtime.py` in this ticket).

Train reference: parallel implementation on `af2/pt-fc3` / PR train is allowed; do not merge `arch-freeze/rw-train` without operator mandate.

Scaffold only — no product commits, no PASS claim.

## Plan digest

Plan digest: e871de56efe9d9b98c1ebd62587824bfc8556c64eed15374360a6b0c8a5647ef

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-051-C.md`](plans/PLAN_T-IGX-051-C.md).

## Approved

PENDING.

## Test contract

Defined in the plan (injection fixtures, STRICT negative tests, regression nodes). Not run at scaffold time.

## Out of scope

TENANT **051-A**, GATES **051-F**, SUPPLY **051-B/N**, merge to `rw-train`, FRZ promotion, FCS.

## Lifecycle state

`draft`