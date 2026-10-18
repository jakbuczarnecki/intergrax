# PROD-Q-R1-R7-Q — Stable CREATE invocation identity / durable duplicate-dispatch denial

| Field | Value |
|---|---|
| Status | **RATIFIED — SCOPED DIRECTION; IDENTITY CONTRACT DESIGN AND IMPLEMENTATION NOT QUALIFIED** |
| Operator ratification | 2026-10-18 |
| Evidence HEAD | `89b1a74db963b017f0fff5055fff353121fa284d` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — NOT CLOSED |
| Prior authority | Reflex ADR-01; F02 Option B; R2 OD-R2-01…04; GCA SQLite decision 01; PIS ADR-03 |
| Production delta | 0 — decision only |

## 1. Exact problem and authority split

At evidence HEAD, `GovernedExternalWorkOrchestrator.create()` checks GCA state keyed by canonical `execution_id`; then `_new_invocation` constructs a fresh provider `invocation_id` per entry. `ProviderInvocationStore.insert_invocation_if_absent` and `GovernedProviderInvocationDispatchGate` deduplicate by `invocation_id`, not by the semantic identity of an external CREATE. If GCA GER/state/F02 and write-uncertainty denial are unavailable after an ambiguous persistence failure/restart, the same logical CREATE can obtain a new invocation ID and be admitted to physical dispatch. This is an in-scope production blocker, not an optional freeze debt.

Existing PIS remains the **only** durable provider-invocation intent/outcome authority. GCA orchestrator remains the **only** execution semantic owner. No separate identity registry, deduplication store, shadow gate or additional persistence authority is approved.

## 2. Approved direction (bounded)

**RATIFIED:** Introduce/reuse a canonical, stable, typed physical CREATE invocation identity *bound to an authorized logical operation* before provider dispatch. Its definition must derive from existing canonical execution/run/attempt, operation type, tenant/principal and idempotency semantics after **closed-world contract analysis**. Do not assume `execution_id` alone uniquely describes every legitimate invocation, or that new `invocation_id` per retry is acceptable. Distinguish retry of the *same* external effect from a *new separately authorized* operation; preserve canonical identity across process restart.

The existing PIS `insert_invocation_if_absent` and gate must be used as the admission source: **INSERTED** may admit a new physical dispatch only after all existing policy/identity checks; **EQUIVALENT_NOOP** (or conflict/unknown outcome) must never call the provider again. A repeated intent must result in a typed denial or sanctioned lookup/reconciliation of the already persisted outcome, not another provider mutation. Do not infer that an equivalent intent necessarily means successful external completion. PIS final outcome and GCA host finalization truth stay distinct.

Atomic durable intent-before-effect ordering, timeout/crash-window and exactly-once *admission*, not an impossible unqualified exactly-once physical effect claim, must be explicitly evidenced. Provider idempotency keys, if applicable, must preserve same-effect identity and cannot substitute for PIS admission proof.

## 3. Mandatory preimplementation contract gate

Before editing code, independently inventory and resolve contract semantics for:
- `RuntimeRevision`, canonical execution identity, task/run/attempt and `execution_id` generation;
- `ProviderInvocation` / provider invocation ID, idempotency key and serialization;
- `ProviderInvocationStore` and `SQLiteProviderInvocationStore` key and uniqueness semantics;
- `GovernedProviderInvocationDispatchGate`, `insert_invocation_if_absent`, `EQUIVALENT_NOOP`, and dispatch/reconciliation callers;
- `GovernedExternalWorkOrchestrator.create` / `_new_invocation` and authorized new vs repeated operations;
- GCA SQLite store restart, missing/uncertain GER/F02, provider outcome persistence timing.

Choose **one** typed mapping with an explicit uniqueness/collision contract, tenant partitioning, retry vs new-operation rules and backward compatibility/reconciliation for previously persisted random invocation IDs. Preserve existing IDs where identity contract requires it. Any required new semantic authority, incompatible persisted identity format, unclear retry/new-op semantics or new storage index beyond ratified existing authority must trigger `STOP — ARCHITECTURE DECISION REQUIRED` before implementation. Operator has **not** approved an arbitrary formula/hash or unilateral modification of PIS semantics.

## 4. Mandatory safety and qualification gates

1. CREATE physical effect success, GCA write ambiguous, neither uncertainty marker nor schema invalidation persisted, restart with no GER/state/F02: same logical CREATE is **not dispatched again**; PIS durable intent recognized through stable identity.
2. GCA clean + PIS intent exists: no repeated provider call; no false success when outcome unknown.
3. GCA and PIS both intact: ordinary existing flow preserved.
4. Different valid newly authorized CREATE: independently admitted; identical-effect retries collapse.
5. Tenant/principal isolation; action and attempt semantics; collisions and malformed identity deny.
6. Concurrent duplicate requests across processes; a single admitted physical dispatch where PIS guarantees allow; stable identity across restart.
7. Failed/uncertain PIS write: deny provider dispatch; no false persisted outcome.
8. Post-dispatch lost outcome: deny re-execution absent explicit separately sanctioned reconciliation.
9. Missing/corrupt F02: no accept/cancel and no CREATE redispatch; no EBE/PIS used as substitute snapshot.
10. Provider mocks count actual physical calls, with **real SQLite PIS + real SQLite GCA** for durable restart gates; unit in-memory only is insufficient for production evidence.
11. Strict HTTP → AC3 → orchestrator → PIS/gate → provider integration must be qualified separately; no global gate accepted from unit tests.
12. Preserve governance ≠ execution, contract-first, layer boundaries, exactly-one semantic authority, pluginability/replaceability and negative tests.

## 5. Status and execution limits

Approved **architecture direction**, not acceptance of code. Do not promote `FRZ-REL-03/05`, `FRZ-PRD-08`, `FRZ-SEC-01/04`, `FRZ-REG-02/03` or relevant OWN/BND/CTR/TYP/RPL criteria. `IMPL-01`, `R7-Q`, `PROD-Q` and all global FRZ remain OPEN/NOT QUALIFIED.

First next step is bounded closed-world design verification and smallest valid implementation under this decision. Do not open another generic ADR. Test-source writing is authorized; **pytest=0 / uv run=0** until specific operator approval. No broad integration suite, background jobs or parallel testing. Independent GitHub exact-SHA audit required before closure.
