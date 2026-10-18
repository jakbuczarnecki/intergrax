# PROD-Q-R1-R7-Q-F02 — Durable follow-on snapshot (prior-create state) architecture decision

| Field | Value |
|-------|-------|
| **Status** | **RATIFIED — SCOPED OPTION B / OPERATOR APPROVED** (2026-10-18; production implementation and durability not yet qualified) |
| **Date** | 2026-10-18 |
| **Branch** | `development` |
| **START_HEAD** | `903f4be08ec9cadd4d3ded133761cd0bd79fe0d3` |
| **Parent** | **PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01** · [`PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_OWNERSHIP_RECOVERY_CORRECTIVE_ADR_01.md`](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_OWNERSHIP_RECOVERY_CORRECTIVE_ADR_01.md) §F02 |
| **Preserved** | Reflex **ADR-01 A+B** · **ADR-03** PIS authority |
| **Production delta** | **0** — documentation / architecture decision only |
| **Does not close** | IMPL-01 · R7-Q · PROD-Q · any global **FRZ-*** PASS |

---

## 1. Ownership graph (source-backed @ `903f4be08…`)

```text
Strict HTTP / reflex (Tier-2)
  └─ GovernedSanctionedExternalWorkOperationPort  [facade — NOT prior-state owner]
       └─ GovernedExternalWorkOrchestrator        [exactly-one writer + resolver]
            ├─ GovernedExecutionStore              [GER, host state, EBE JSON, **proposed follow-on record**]
            ├─ ProofReceiptStore
            ├─ ContinuationStateStore
            ├─ PolicyBundleArtifactStore
            └─ ProviderInvocationStore (durable @ composition)  [intent/outcome — NOT snapshot carrier]

PIS: admission + outcome digests — no ExternalWorkSnapshot payload.
EBE v2 (`put_event_json`): attestation evidence — not follow-on execution authority.
```

| Concern | Exactly-one owner | Mechanism |
|---------|-------------------|-----------|
| Prior-create typed view for accept/cancel | `GovernedExternalWorkOrchestrator` | `resolve_create_adapter_result_for_follow_on` (read) · create `_finalize_side_effect` (write) |
| Physical persistence | `GovernedExecutionStore` implementation (injected, process-scoped per F01) | Extended protocol method — **same store**, no parallel service |
| Port session dict | **Removed** (recovery IMPL) | `sanctioned_external_work_operation_port.py` L31–37, L134–141, L156 |

---

## 2. Field requirement matrix — `accept` / `cancel`

**Code authority:** `orchestrator.py` `accept` L403–508 · `cancel` L510–594 · adapter `forward_quote_acceptance` / `cancel_and_map` (`external_work_adapter.py`).

### 2.1 `ExternalWorkSnapshot` fields

| Field | Accept | Cancel | Used by orchestrator | Used by adapter (correlation path) |
|-------|--------|--------|----------------------|-------------------------------------|
| `schema_version` | validate | validate | indirect (Pydantic) | indirect |
| `correlation.task_id` | **yes** | **yes** | identity gate L450–455 / L542–547 | **yes** |
| `correlation.run_id` | **yes** (non-empty) | **yes** (non-empty) | L446–448 / L538–540 | **yes** |
| `correlation.external_task_id` | **yes** | **yes** | `_new_invocation` L467 / L559 | **yes** |
| `correlation.provider_id` | **yes** (meta) | **yes** (meta) | L426–440 / L529–537 | **yes** |
| `correlation.correlation_id` | optional | optional | meta L442–445 | optional |
| `correlation.idempotency_key` | no (accept supplies new) | no | — | accept path uses new idempotency |
| `status` | no direct | no direct | — | post-mutation mapping only |
| `quote` | no | no | — | enrich paths only |
| `created_at` / `updated_at` | no | no | — | canonical snapshot integrity |
| `provider_state_label` / `failure_ref` / `deliverable_count` / `metadata` | no | no | — | not read on follow-on entry |

**Minimum for follow-on:** full canonical `ExternalWorkSnapshot` (frozen contract) — orchestrator requires `create_result.snapshot is not None`; partial correlation-only reconstruction is **forbidden** (would omit validated snapshot fields and invite drift).

### 2.2 `ExternalWorkAdapterResult` beyond snapshot

| Field | Accept | Cancel |
|-------|--------|--------|
| `snapshot` | **required** | **required** |
| `provider` | fallback `provider_id` if meta empty | fallback `provider_id` |
| `used` / `proof` / `policy_decision` | not read on entry | not read on entry |

**Rehydration target:** `ExternalWorkAdapterResult` with `used=True`, `reason` from persisted create outcome, `snapshot`, optional `provider` — sufficient for current orchestrator + adapter.

---

## 3. Existing data availability matrix

| Source | Carries full `ExternalWorkSnapshot`? | Carries `ExternalWorkAdapterResult`? | Durable @ strict prod (today) | Trust for follow-on authority |
|--------|--------------------------------------|----------------------------------------|--------------------------------|------------------------------|
| Port `create_result_by_execution_id` | yes (session) | yes | **no** (shadow) | **reject** |
| `GovernedExecutionResultV2` (`put_result`) | **no** — `provider_invocation`, digested `provider_outcome`, `proof` only | **no** | same as GCA execution store | insufficient |
| `ProviderInvocation` / PIS outcome | partial correlation ids | **no** | **yes** (composition) | intent/outcome only — **not** snapshot |
| `put_event_json` / `ExecutionBoundaryEventV2` | **no** — identity, policy, proof sections only (`boundary_event.py` L194–218) | **no** | same as GCA execution store | **evidence only** — using as recovery truth would conflate attestation with execution authority |
| `get_state` | **no** | **no** | same | gating only (UNKNOWN, denied, …) |
| `ContinuationStateStore` | **no** — `GovernedContinuationRequest` | **no** | same as host stores | quote continuation — not create snapshot |

**Option A verdict:** **Not proven.** No existing `GovernedExecutionStore` record contains the required immutable `ExternalWorkSnapshot` + provider descriptor binding. GER + PIS cannot be composed into a valid snapshot without inferring timestamps, status, quote, and metadata (**forbidden** by task closed-world rules).

---

## 4. Option A vs Option B

| Criterion | Option A — reuse existing record only | Option B — minimal typed extension on `GovernedExecutionStore` |
|-----------|----------------------------------------|----------------------------------------------------------------|
| Schema change | none | add one frozen record type + two protocol methods |
| Parallel store | no | **no** — same owner, same injection site as GER |
| Proven fields @ HEAD | **fails** — see §3 | **yes** — explicit snapshot persistence |
| Evidence vs authority | N/A | EBE remains evidence; new record is **execution host fact** written by orchestrator at create finalize |
| Replaceability | unchanged | backend implements new methods alongside existing GER APIs |
| Operator ratification | would claim no extension | **required** — extends `GovernedExecutionStore` semantic contract |

---

## 5. Recommended design — **Option B**

### 5.1 Typed contract (proposed — Tier-0)

**New file:** `intergrax/contracts/governed_external_work_create_follow_on.py`

```python
class GovernedExternalWorkCreateFollowOnRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_external_work_create_follow_on.v1"]
    execution_id: ExecutionId
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    tenant_id: str          # non-empty
    principal_id: str         # non-empty
    create_action: Literal["external_work.create"]  # bind to CREATE GER only
    create_provider_invocation_id: str              # matches GER.provider_invocation.invocation_id
    snapshot: ExternalWorkSnapshot
    provider: ExternalWorkProviderDescriptor | None = None
    snapshot_content_digest: str                    # validate_content_digest(stable canonical snapshot JSON)
    recorded_at: datetime                           # aware UTC

    # model_validator: snapshot.correlation.task_id/run_id align with task_id/run_id;
    # execution_id key consistency; digest matches snapshot payload
```

**Protocol extension** (`applications/governed_contractor_application/host/stores.py`):

```python
def put_create_follow_on_record(
    self, execution_id: str, record: GovernedExternalWorkCreateFollowOnRecordV1
) -> None: ...

def get_create_follow_on_record(
    self, execution_id: str
) -> GovernedExternalWorkCreateFollowOnRecordV1 | None: ...
```

JSON helpers: `parse_governed_external_work_create_follow_on_record_json` — explicit schema dispatch (mirror GER / EBE patterns). **No** `Any` / `object` / ad-hoc dict at the boundary.

### 5.2 Write lifecycle and ordering (create finalize)

**Writer:** `GovernedExternalWorkOrchestrator._finalize_side_effect` only, when:

1. `action == ACTION_CREATE_EXTERNAL_WORK` and `after_create is True`;
2. adapter path reached successful ALLOW + `adapter_result.snapshot is not None`;
3. `put_result(ger)` succeeded (CREATE GER persisted);
4. **then** `put_create_follow_on_record` with record bound to `ger` identity and `invocation.invocation_id`.

**Ordering:** `put_result` → `put_create_follow_on_record` → `put_state` (attestation pending) → `put_event_json` (EBE). If follow-on write fails → `put_state(EXECUTION_FAILED)`, reason `create_follow_on_record_persistence_failed`, **no** accept/cancel until record exists.

**Idempotent rewrite:** second identical record (equivalence predicate on digest + invocation id) is no-op; conflicting payload → fail closed (store raises or orchestrator maps to failed state).

**Not written when:** policy denied, UNKNOWN outcome, missing snapshot, attestation failed before GER, or create did not complete successful mutation path.

### 5.3 Read / resolver lifecycle

**Resolver:** `GovernedExternalWorkOrchestrator.resolve_create_adapter_result_for_follow_on(execution_id) -> ExternalWorkAdapterResult | None`

**Algorithm (fail closed):**

1. Load `get_state(execution_id)` — reject if `None`, `EXECUTION_OUTCOME_UNKNOWN`, create denied states, or `EXECUTION_FAILED` without attested CREATE GER.
2. Load CREATE `get_result` — require `GovernedExecutionResultV2`, `action == external_work.create`, consistent `execution_id`.
3. Load `get_create_follow_on_record` — **required**; validate Pydantic + digest + identity binding against GER (`tenant_id`, `principal_id`, `task_id`, `run_id`, `attempt_id`, `create_provider_invocation_id`).
4. Compare follow-on request `tenant_id` / `principal_id` (from port payload) to record — mismatch → `None` / port maps `prior_create_snapshot_missing` or dedicated `prior_create_identity_mismatch` (IMPL choice — must deny provider mutation).
5. Rehydrate `ExternalWorkAdapterResult(used=True, reason=…, snapshot=record.snapshot, provider=record.provider, status=record.snapshot.status)`.
6. **Do not** read `get_event_json` for snapshot fields. **Do not** call provider `get_work` to fill gaps.

**Port:** `_resolve_prior_create_result` delegates to orchestrator only; **remove** session cache.

### 5.4 Tenant / principal / revision rules

| Check | Rule |
|-------|------|
| Tenant | Record `tenant_id` == GER `tenant_id` == active governance tenant on follow-on request (`_require_canonical_execution_identity`) |
| Principal | Record `principal_id` == GER `principal_id`; follow-on request principal must match record (strict serving binding — F01 debt separate) |
| Execution identity | `execution_id` key == record.execution_id == snapshot-bound task/run; port `payload.execution_id` must match |
| Revision | `snapshot.schema_version` + record `schema_version` — unsupported version → deny (no reflection) |
| Provider correlation | `create_provider_invocation_id` matches CREATE GER; snapshot correlation matches GER task/run/external_task_id |
| Stale snapshot | `record.recorded_at` < GER `execution_completed_at` tolerance N/A — single write; if GER superseded by non-CREATE action on same id (should not happen) → deny |

### 5.5 Restart, UNKNOWN, idempotency

| Scenario | Behavior |
|----------|----------|
| Process restart after successful create | Accept/cancel allowed iff durable store returns GER + follow-on record + valid state |
| Restart mid-finalize (GER without record) | **Deny** follow-on — incomplete persistence fails closed |
| `EXECUTION_OUTCOME_UNKNOWN` | **Deny** — no follow-on mutations |
| Repeated accept/cancel (new invocations) | Each mutation new PIS intent; prior-create record **immutable**; idempotency via accept/cancel keys on adapter, not snapshot store |
| Corrupt JSON / digest mismatch | **Deny** — treat as missing record |
| Wrong tenant on accept | **Deny** before adapter |

**Recovery vs retry:** GR-7-A7 — reloading snapshot is **not** permission to re-dispatch CREATE; separate PIS rules govern provider retries.

### 5.6 Shadow authority

- Single writer (orchestrator finalize), single resolver (orchestrator method), single store protocol.
- Record is **host execution fact**, not governance decision, not observability event, not PIS substitute.
- EBE remains attest-only; PIS remains intent/outcome only.

---

## 6. Minimal implementation blueprint (files)

| File | Change |
|------|--------|
| `intergrax/contracts/governed_external_work_create_follow_on.py` | **add** `GovernedExternalWorkCreateFollowOnRecordV1` + parse helper + equivalence helper |
| `intergrax/contracts/__init__.py` or package exports | export new symbol (if project convention requires) |
| `applications/governed_contractor_application/host/stores.py` | extend `GovernedExecutionStore` + `InMemoryGovernedExecutionStore` |
| `applications/governed_contractor_application/host/orchestrator.py` | write on CREATE finalize; add `resolve_create_adapter_result_for_follow_on` |
| `applications/governed_contractor_application/host/sanctioned_external_work_operation_port.py` | remove `create_result_by_execution_id`; delegate resolver |
| `applications/governed_contractor_application/host/strict_external_work_host_wiring.py` (or session resources type) | remove `create_result_by_execution_id` from session bag |
| Future durable GCA store backend | implement new methods with same media as GER (F01 — **TRACKED FREEZE DEBT**) |

**Forbidden:** new snapshot microservice; Tier-2 import of `applications/`; inferring snapshot from `get_event_json` or provider polling.

---

## 7. Acceptance gates (mandatory)

### Positive

| ID | Gate |
|----|------|
| G-F02-P01 | After attested CREATE, `get_create_follow_on_record(execution_id)` returns valid record; digest verifies |
| G-F02-P02 | Accept/cancel in **same process** without port dict succeed when record present |
| G-F02-P03 | After simulated restart (new store instance, replayed persistence fixture), accept/cancel succeed **iff** GER + record reloaded |
| G-F02-P04 | Resolver returns typed `ExternalWorkAdapterResult`; orchestrator accept/cancel unchanged |

### Negative

| ID | Gate |
|----|------|
| G-F02-N01 | Missing record after CREATE → `prior_create_snapshot_missing`, no provider mutation |
| G-F02-N02 | Wrong `tenant_id` on accept → denied before adapter |
| G-F02-N03 | `EXECUTION_OUTCOME_UNKNOWN` state → denied |
| G-F02-N04 | Corrupt record / digest mismatch → denied |
| G-F02-N05 | Port session dict absent — no code path uses shadow cache |
| G-F02-N06 | EBE-only store (GER + event, no follow-on record) → denied |

**Executable evidence:** **DEFERRED — EVIDENCE REQUIRED** (pytest forbidden in this task). Proposed: unit tests on `InMemoryGovernedExecutionStore` + orchestrator resolver with injected clock; qualification harness restart fixture under F01 durable store slice.

---

## 8. Enterprise audit matrix

| Principle | Source evidence @ HEAD | Proposed change | Residual risk |
|-----------|------------------------|-----------------|---------------|
| Contracts over implementations | `GovernedExecutionStore` protocol `stores.py` L40–56 | Typed `GovernedExternalWorkCreateFollowOnRecordV1` | Durable physical backend still debt (F01) |
| Layer boundaries | Tier-2 port → orchestrator only | No Tier-2 store access | — |
| Exactly-one owner | Corrective ADR §F02 | Orchestrator write/read | IMPL discipline |
| Strong typing | `ExternalWorkSnapshot` in `intergrax/contracts/external_work.py` | Frozen record model | — |
| Replaceability | Store injected in composition | New protocol methods on same port | Backend must implement all methods |
| No duplicate mechanism | Session dict defect | Remove dict | — |
| Governance ≠ execution | Accept re-evaluates policy in adapter | Snapshot is correlation + state view, not ALLOW | — |
| Evidence ≠ authority | EBE v2 lacks snapshot | Do not parse EBE for follow-on | — |
| Fail-closed | Port already `prior_create_snapshot_missing` | Extend to incomplete persistence | — |
| Observability | Reliability emits separate events | No change | — |
| Regression protection | G-F02-* gates | Deferred tests | OPEN until executed |

---

## 9. FRZ touchpoints (status unchanged — **OPEN**)

| FRZ | Relation to F02 |
|-----|-----------------|
| **FRZ-REL-03** | Follow-on idempotency remains on accept/cancel keys + PIS; snapshot record is read-only anchor |
| **FRZ-REL-05** | CREATE termination snapshot frozen before accept/cancel |
| **FRZ-PRD-08** | Removes session bypass for prior-create state |
| **FRZ-SEC-01/04** | Tenant/principal bound in record; removes shadow dict |
| **FRZ-CTR-***, **FRZ-BND-***, **FRZ-OWN-***, **FRZ-TYP-***, **FRZ-RPL-***, **FRZ-REG-02/03** | Typed extension at sanctioned host store boundary; single owner; backends replaceable; gates deferred |

Global checklist statuses: **not updated** (operator closure only).

---

## 10. Ratified baseline decision (historical proposal)

**Ratified Option B:** extend `GovernedExecutionStore` with `GovernedExternalWorkCreateFollowOnRecordV1` persistence as the **sole authoritative** prior-create carrier for accept/cancel, written only by `GovernedExternalWorkOrchestrator` on successful CREATE finalize, read only via `resolve_create_adapter_result_for_follow_on`, with session port cache **removed** and **no** use of `get_event_json` or PIS for snapshot reconstruction.

**Reject alternatives:** Option A (insufficient @ HEAD); session dict fallback; provider `get_work` recovery; EBE-as-authority.

---

## 11. Operator ratification — 2026-10-18

**RATIFIED — SCOPED OPTION B**, following independent exact-SHA review of the proposed F02 document @ `cf2c0d8277d0202094e72b8098e604077e45af10`. Operator approved the typed `GovernedExternalWorkCreateFollowOnRecordV1` on the **existing** `GovernedExecutionStore`, with orchestrator as exactly-one writer/resolver, no session cache, and no use of EBE or PIS as a snapshot carrier.

**Two mandatory operator safety conditions:**

1. **Actual production durability.** A protocol method or an in-memory/process-scoped store is **not** evidence of restart durability. Strict production follow-on mutations must be **fail-closed** unless the sanctioned GCA execution-store backend is independently qualified durable across process restart. Existing demo filesystem or in-memory backends cannot silently be promoted to production. Any missing compliant backend remains an **IN-SCOPE BLOCKER** for strict integrated qualification, not a merely deferred optional improvement. Do not invent a new persistence framework or shadow store. If the existing owned backend/contracts cannot meet this requirement without new architecture, **STOP — ARCHITECTURE DECISION REQUIRED**.
2. **Partial-finalization truth and retry safety.** If provider CREATE has already occurred and `put_result` succeeds but `put_create_follow_on_record` fails, **never** convert the physical effect into an ordinary retryable provider failure or claim it did not occur. Preserve actual PIS/provider outcome and distinguish **external-effect outcome** from **host finalization incomplete**. Deny all follow-on requests until authoritative record completeness is restored through sanctioned recovery; do **not** ordinary re-dispatch CREATE. The semantic state/contract to represent this distinction must reuse an existing authority; if unavailable, STOP for architecture decision before implementation. Revisit the originally proposed unconditional `EXECUTION_FAILED` transition: it is **not ratified** where it misrepresents an already-executed provider effect.

**Required gate before parent acceptance:** exact-SHA contract/code proof + lightweight deterministic positive/negative tests of both conditions; actual full HTTP AC3 / restart durability test remains DEFERRED where resource budget prohibits execution, and cannot be counted PASS. The existing F01–F05 corrective prerequisites still apply. `IMPL-01`, `DIM20`, `R7-Q`, `PROD-Q` and global `FRZ-*` remain NOT CLOSED / OPEN. No production change was authorized beyond the approved bounded design, and no implementation is asserted by this ratification.

---

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.*
