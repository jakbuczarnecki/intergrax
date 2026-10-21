# PROD-Q-R1-R7-Q — Stable CREATE invocation identity / durable duplicate-dispatch denial

| Field | Value |
|---|---|
| Status | **RATIFIED DIRECTION — CONTRACT GATE CLOSED; IMPLEMENTATION @ `389cb82be`; SCOPED QUALIFICATION EXECUTED @ `1bf97ce8a1f59241a59d834ec009ccc270029702` — evidence [`PROD_Q_R1_R7_Q_STABLE_CREATE_IDENTITY_SCOPED_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_STABLE_CREATE_IDENTITY_SCOPED_QUALIFICATION_EVIDENCE.md) **READY FOR AUDIT**; parent IMPL-01 / R7-Q / PROD-Q / FRZ OPEN** |
| Operator ratification | 2026-10-18 (direction); contract gate record 2026-10-21 |
| Evidence HEAD | `c541c950f8ab2473f7a755fd2b0d29efb549ca0e` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — NOT CLOSED |
| Prior authority | Reflex ADR-01; F02 Option B; R2 OD-R2-01…04; GCA SQLite decision 01; PIS ADR-03 |
| Production delta | Stable CREATE `invocation_id` derivation + orchestrator admission path @ `389cb82be` (see §6); scoped qualification **23 passed** @ `1bf97ce8…` (see evidence doc) |

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

First next step after §6 contract gate: operator authorization to execute scoped qualification (`test_prod_q_r1_r7_q_impl_01_stable_create_invocation_identity.py`, dual-SQLite Q2, strict HTTP CREATE chain) — not further identity redesign absent qual failure. Do not open another generic ADR. **pytest=0 / uv run=0** until specific operator approval. Independent GitHub exact-SHA audit required before closure.

## 6. Mandatory contract gate closure (2026-10-21)

**Verdict:** `STABLE CREATE IDENTITY — CONTRACT READY FOR IMPLEMENTATION AUTHORIZATION` (contract closed; implementation already merged — authorization applies to **qualification execution**, not a second identity design pass).

### 6.1 Canonical CREATE effect identity (single definition)

| Layer | Field / rule |
|-------|----------------|
| Authorized logical operation | One physical CREATE per **`execution_id`** under GR-3 canonical execution identity (`task_id`, `run_id`, `attempt_id` bind via active execution context → `execution_id`). |
| Stable provider key | `invocation_id = derive_governed_external_work_create_provider_invocation_id(tenant_id, principal_id, execution_id, provider_id, idempotency_key)` → `gca-pinv:{stable_payload_hash({kind: governed_external_work_create.v1, …})}` (`intergrax/contracts/governed_external_work_provider_invocation_identity.py`). |
| Retry vs new operation | **Same effect:** unchanged tuple above → same `invocation_id`. **New authorized CREATE:** new `execution_id` (new GR-3 execution) and/or different `idempotency_key` and/or different `tenant_id` / `principal_id` / `provider_id` → distinct `invocation_id`. |
| Restart stability | Derivation is pure; PIS row keyed by `invocation_id` survives restart. Orchestrator `_resolve_create_provider_invocation_id` prefers persisted GER/F02 `create_provider_invocation_id` when present (legacy random ids). |
| PIS admission | `insert_invocation_if_absent`: no row → **INSERTED**; row + `provider_invocations_equivalent` → **EQUIVALENT_NOOP**; row + not equivalent → **conflict** (`ProviderInvocationConflictError`); infra/ambiguous commit → **UNKNOWN** (`ProviderInvocationPersistenceError`, uncertain-write marker on SQLite store). |
| Gate | `GovernedProviderInvocationDispatchGate`: physical `execute()` only on **INSERTED**; **EQUIVALENT_NOOP** → `dispatch_not_admitted` (not provider success). |
| EQUIVALENT_NOOP semantics | Typed duplicate intent only — **not** provider success attestation; orchestrator `_handle_create_duplicate_invocation_admission` reconciles via GER or PIS outcome or `EXECUTION_OUTCOME_UNKNOWN`. |
| Legacy | Pre-stable random `inv-*` rows remain authoritative when GER/F02 still references them; new creates without GER use derived `gca-pinv:*`. |
| Tenant/principal isolation | Hashed into `invocation_id`; mismatched replay with same idempotency but different tenant/principal yields different ids (no cross-tenant collapse). |
| GER/F02 lost, PIS durable | GCA redispatch blocks when state/GER present; without GER, duplicate admission path uses PIS intent/outcome — no second provider CREATE on equivalent id. |

**Ownership unchanged:** GCA orchestrator semantic owner; PIS sole durable intent/outcome; gate sole physical dispatch; no registry/dedup store.

### 6.2 Scenario matrix (contract)

| Scenario | Canonical identity | PIS state | Gate decision | Provider calls | Recovery | Verdict |
|----------|-------------------|-----------|---------------|----------------|----------|---------|
| First correct CREATE | Derived `gca-pinv:*` | INSERTED intent | INSERTED → dispatch | 1 | GER/F02 as today | Admit once |
| Retry same effect | Same derived id | EQUIVALENT_NOOP | No dispatch | 0 (held at 1) | UNKNOWN or GER replay | Fail-closed reconcile |
| Separate authorized CREATE | New `execution_id` / idem / scope | New INSERTED | Dispatch | +1 | Normal | Independent admit |
| Concurrent duplicate | Same id | One INSERTED, peer EQUIVALENT_NOOP | Single dispatch | 1 | SQLite serialization | Single physical effect |
| Failed/uncertain PIS commit | Id minted in memory | No durable intent / marker | No dispatch | 0 | Uncertain-write deny | Fail-closed |
| Provider CREATE ok, GER/F02 lost | Stable id in PIS | Intent ± outcome | EQUIVALENT_NOOP on retry | 0 on retry | PIS outcome projection | No redispatch |
| Restart after ambiguous GCA write | Unchanged derived id | Prior intent if committed | Per PIS + GCA uncertain flags | Per admission | OD-R2 SQLite recovery | No silent redispatch |
| UNKNOWN provider outcome | Stable id | Intent + UNKNOWN outcome | First dispatch done | 1 | Host UNKNOWN state | No false success |
| Legacy random invocation id | GER-stored `inv-*` | Existing row | Per row | 0 if GER blocks | `_resolve_create_provider_invocation_id` | Compatibility |
| Tenant/principal mismatch | Different derived id | Separate rows | Independent INSERTED | Per authorized op | Partitioned hash | No collision |
| Strict HTTP re-entry | Same as orchestrator CREATE | Same as above | Same gate path | Same | Sanctioned port → orchestrator | No bypass |

### 6.3 Implementation evidence (not qualification)

- `389cb82bef494eccc8d52eb3e5d79da6ffbb5e76` — derive + orchestrator + unit test sources.
- Qualification sources present; **not executed** under operator pytest=0 policy.

**Cursor status cap:** READY FOR AUDIT (independent GitHub SHA audit required for closure).
