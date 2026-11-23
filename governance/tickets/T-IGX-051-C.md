# T-IGX-051-C — ARCH-FREEZE remediation T-AF2-RW-TRUST (trust boundary wave)

## Scope

Delivery ticket for **T-AF2-RW-TRUST** / WP-12 W3 trust-boundary remediation on the `development` tor. Master-plan mapping: `llm/messages.py`, `graph_assembly.py`, `context/provider_lifecycle.py`, `context/planner.py` and related consumers.

Residuals routed here (MEP-AF2 v2.1, PDC table A): **EX-D2-01**, **EX-D2-02**, **EX-D2-05**, **EX-D2-07**, **EX-L-05**, **PDC-D2-R004** (RAG red root cause where product-bound).

Design goals: trust classes for model messages (untrusted content must not reach `role=system`), most-restrictive egress default, orphan `tool_call_id` fail-closed in STRICT, planner ledger sizing fidelity (**EX-D2-05**).

Write set is **disjoint** from PT-FC3 embedding/vector paths and from PT-WF harness evaluation paths (no planned edit of `harness_host_runtime.py` in this ticket).

Train reference: parallel implementation on `af2/pt-fc3` / PR train is allowed; do not merge `arch-freeze/rw-train` without operator mandate.

Scaffold only — no product commits, no PASS claim.

## Plan digest

Plan digest: c91d7ca4c3d970908b9bb4c76a88213175accb880a61f5bc8248fd3a08dfa1c9

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-051-C.md`](plans/PLAN_T-IGX-051-C.md).

## Approved

Approved: operator @ 2026-11-21T12:00:00+01:00

Approval source: OD-AF-AUTO-03 (autonomy session 2026-11-21); digest verified against canonical plan file.



## Test contract

Defined in the plan (injection fixtures, STRICT negative tests, regression nodes). Not run at scaffold time.

## Out of scope

TENANT **051-A**, GATES **051-F**, SUPPLY **051-B/N**, merge to `rw-train`, FRZ promotion, FCS.

## Lifecycle state

in_progress

§7 fala 5A slice 3+: `NT-TRUST-06` **1 passed**, tenant-scope **11 passed**, §7.2 **64 passed** @ `dc083f6ea`; all `NT-TRUST-01`..`06` green; fala 5 close @ `e61e5136a`: trust+RAG **80 passed**, FC-3 bundle **260 passed**, 3 skipped — plan §7.2