# T-IGX-076-A — ARCH-FREEZE platform foundation

## Scope

Product implementation of the approved ARCH-FREEZE foundation. The platform must refuse a non-stable vector store in a strict host, run accepted work from a durable row visible to a second process, classify memory durability separately from atomicity and sharing, pin skill instruction and policy content, apply the existing MCP identity refusal on every strict composition and execution path, and keep an unknown external effect from starting another external operation.

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-076-A.md`](plans/PLAN_T-IGX-076-A.md).

Write set, by package:

- Vector refusal and removal of silent in-memory vector stubs: integration catalog status, Milvus and Weaviate stores, the host composition that opens a vector store.
- Out-of-process tasks: queue wiring, document-store task queue, Celery adapter, logical task execution.
- Session and memory admission: durability evidence, not a conditional-store type check.
- Skill instruction and policy pinning, without weakening platform, application, or client policy.
- MCP strict composition: existing identity assertion driven by host posture.
- External-effect recovery: existing ledger and reconciliation; no automatic compensation on an unknown result.
- Verification of mechanisms that already pass, plus one existing full gate. No new orchestration system.

## Plan digest

Plan digest: 2c70d92ea5b6ff7773d7bf83bdaaff61c49ab94069273e163a64e1b513a40275

## Approved

Approved: operator-delegate @ 2026-11-26T11:08:00+01:00
Delegation: operator chat commission 2026-11-26 to implement the approved ARCH-FREEZE foundation plan without a further planning round

## Test contract

One `uv run pytest` at a time. Package tests are named in the canonical plan. The closing step is one existing full gate after the packages land. Do not weaken tests. Do not mark freeze acceptance rows as passed in this ticket.

## Out of scope

Scenario Factory product code, new vendor integrations, live provider accounts, a second execution engine, a second identity system, a second policy engine, a second task queue, and a new memory subsystem. Kafka remains an unbuilt broker. Mongo organization memory stays refused until it meets the same evidence gate. Procedural memory is not a new store.

## Lifecycle state

in_progress
