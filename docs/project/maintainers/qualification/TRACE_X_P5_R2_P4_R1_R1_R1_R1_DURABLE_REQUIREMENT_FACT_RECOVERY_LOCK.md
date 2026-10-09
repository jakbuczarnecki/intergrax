# TRACE-X-P5-R2-P4-R1-R1-R1-R1 — Durable Requirement Fact Recovery Lock

| Field | Value |
|---|---|
| **Status** | **BLOCKED ON R1-R1-R1-R1-R1** (P2 typed pin/staging API — child lock) |
| **Production delta** | **0** |
| **FINAL_COMMIT** | `2c1fccf4b611afcdd7407e37c4ba33e76f81ce9a` (superseded for staging contract by child) |
| **Child** | [`TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_P2_PIN_RECOVERY_STAGING_CONTRACT_LOCK.md`](TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_P2_PIN_RECOVERY_STAGING_CONTRACT_LOCK.md) |
| **Rejected R1-R1-R1 baseline** | `3f37a80cea783fad4ca6b75ce8199ee8ed9c888d` |
| **Parent** | [`TRACE_X_P5_R2_P4_R1_R1_R1_REQUIREMENT_EVENT_CANONICAL_RETRY_IDENTITY_RECONCILIATION.md`](TRACE_X_P5_R2_P4_R1_R1_R1_REQUIREMENT_EVENT_CANONICAL_RETRY_IDENTITY_RECONCILIATION.md) |
| **Blocker** | `R2-P4-REQUIREMENT-FACT-DURABLE-RECOVERY-29` — **RESOLVED IN DESIGN** (this artifact) |
| **P4 / R1 / R1-R1 / R1-R1-R1** | **BLOCKED** on corrected P4 implementation after audit |
| **FRZ-TRC-11** | **OPEN** |
| **P5 / CERT** | **NOT ENTERED** |

## 1. Reconciliation boundary

Parent R1-R1-R1 locks **what** must be equal on retry (`RuntimeEvent` model equality, immutable
`ExecutionIntegrationConfigurationProvenanceRequirementFact`) but left **where** that fact is
recovered after process crash (Case C: pin durable, spine absent, in-memory fact lost).

This artifact closes only durable recovery sourcing and retry ownership. It does **not** reopen:

- typed requirement event;
- runtime emitter ownership;
- pin → spine → I/O sequencing;
- deterministic `EventId`;
- exact `RuntimeEvent` equality;
- stable timestamp requirement;
- mandatory durability;
- Option B as-of semantics;
- Docker E2E mandate.

## 2. Blocker disposition

| Blocker | Disposition |
|---|---|
| `R2-P4-REQUIREMENT-FACT-DURABLE-RECOVERY-29` | **RESOLVED IN DESIGN** — composite durable source below; one minimal additive extension on sanctioned P2 pin envelope for fields absent from every other owner |

## 3. Closed-world durable inventory

Production locus names refer to code @ rejected R1-R1-R1 baseline `3f37a80c…` unless noted.

| Required field | Candidate source | Durable? | Exact for obligation? | Canonical owner | Usable for Case C? |
|---|---|---:|---:|---|---:|
| `tenant_id` | P2 pin `record.tenant_id` · `KvExecutionIntegrationConfigurationPinningStore` | Yes | Yes | Integrations P2 pin (`applications/_shared/integrations/persistence.py`) | **Yes** |
| `execution_id` | P2 pin `record.execution_id` | Yes | Yes | P2 pin | **Yes** |
| `IntegrationConfigurationSubject` | P2 pin `record.pin_subject` (+ decode pair) | Yes | Yes | P2 pin | **Yes** (per subject row) |
| `task_id` | `RuntimeCheckpoint` on governed `Task` · `MarketplaceToolExecutionIntent.task_id` | Yes (task / intent) | Yes when execution scope known | Task store / intent repo | **Partial** — intent lacks `run_id`/`attempt_id`; checkpoint requires resumed task context |
| `run_id` | `RuntimeCheckpoint.run_id` | Yes | Yes with task resume | Task runtime checkpoint | **Partial** — only after task reload |
| `attempt_id` | `RuntimeCheckpoint.attempt_id` | Yes | Yes with task resume | Task runtime checkpoint | **Partial** — same |
| `task_id`/`run_id`/`attempt_id` (pin moment) | **Pin envelope extension** `requirement_execution_correlation` (§5) | Yes (after P4 impl) | Yes | P2 pin (extended) | **Yes** |
| factual `timestamp` | P2 pin body today | Yes | **No** — no time field in envelope | — | **No** |
| factual `timestamp` | `MarketplaceToolExecutionIntent` | Yes | **No** — intake time ≠ pin boundary | Intent repo | **No** |
| factual `timestamp` | positioned requirement spine event | Yes | Yes | Runtime spine | **No** in Case C (absent) |
| factual `timestamp` | **Pin extension** `requirement_boundary_recorded_at` (§5) | Yes (after P4 impl) | Yes | P2 pin (extended) | **Yes** |
| `node_id` / `agent_id` / `step_id` | Active execution context at pin | **No** after crash | — | — | **No** |
| `node_id` / `agent_id` / `step_id` | Pin extension `requirement_emission_context` (optional stable values) | Yes | Yes if captured at pin | P2 pin (extended) | **Yes** (default empty/`None`) |
| `correlation_id` / `traceparent` / `tracestate` | Active evidence context at pin (`peek_active_execution_evidence_context`) | **No** after crash | — | — | **No** |
| same trace fields | Pin extension `requirement_emission_context` | Yes | Yes if non-empty at pin | P2 pin (extended) | **Yes** |
| `phase` / `severity` / `event_type` / registry fields | Spine metadata registry + requirement contract | N/A (derived) | Yes (deterministic) | Runtime event catalog | **Yes** (derive, not store) |
| `payload` dimensions | P2 pin + schema contract | Yes | Yes | Pin + contracts | **Yes** |
| `event_id` | Deterministic digest of logical key | N/A | Yes | Contracts (R1-R1-R1) | **Yes** (recompute) |
| full `RuntimeEvent` (Case D) | Positioned spine `INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED` | Yes | Yes | `RuntimeEventPersistence` / store | **Yes** |

### 3.1 Stores explicitly insufficient alone

| Store | Verdict |
|---|---|
| `MarketplaceToolExecutionIntent` (`intergrax/contracts/tools/marketplace_tool_execution_intent.py`) | **Insufficient** — has `tenant_id`, `task_id`, `execution_request_id`, provenance ops ids; **no** `run_id`, `attempt_id`, `ExecutionId`, `IntegrationConfigurationSubject`, requirement boundary timestamp, or trace metadata |
| Positioned runtime spine (pre-commit) | **Insufficient** for Case C — event absent by definition |
| `ExecutionLineageAdmissionRecord` | **Insufficient alone** — durable and exact **given** `ExecutionLineageAttemptScope`, but **no** public execution-global reverse index; scope must come from checkpoint or pin extension |
| In-memory `ExecutionIntegrationConfigurationProvenanceRequirementFact` at commit port | **Insufficient** after restart |

## 4. Selected recovery model (priority order applied)

1. **Existing durable execution facts** — P2 pin for tenant, execution, subject, payload dimensions.
2. **Lineage / checkpoint** — corroborate `task_id` + `run_id` + `attempt_id` on resume (`RuntimeCheckpoint` + governed task reload); not relied on as sole source at pin boundary.
3. **Minimal additive fields on sanctioned P2 pin envelope** — single extension object written **once** on first successful pin CAS, immutable on idempotent pin replay:

```text
record.requirement_recovery_staging.v1
  requirement_boundary_recorded_at   # timezone-aware UTC, pin success instant
  requirement_execution_correlation
    task_id
    run_id
    attempt_id
  requirement_emission_context     # optional; only non-null values used on rebuild
    node_id | agent_id | step_id
    correlation_id
    traceparent
    tracestate
```

**Not** a standalone requirement-fact store: extension lives inside the existing P2 KV value keyed
`(tenant_id, execution_id, subject)` — same owner as pin content.

**Case D:** staged extension is **not** read for equality; the **accepted positioned spine event**
is the sole canonical source for every `RuntimeEvent` field (including `timestamp` =
`RuntimeEvent.timestamp` on that record).

**Case C:** rebuild fact exclusively from **P2 pin row (including extension)** + deterministic
registry derivation; verify spine absence via deterministic `EventId` lookup / projection for
logical key before append.

## 5. Canonical timestamp (exactly one)

| Case | Canonical field | Record |
|---|---|---|
| **C** (no spine fact) | `requirement_boundary_recorded_at` | P2 pin JSON envelope `record.requirement_recovery_staging.v1` in `encode_integration_configuration_provenance` / `KvExecutionIntegrationConfigurationPinningStore.pin` |
| **D** (spine accepted) | `timestamp` | Positioned `RuntimeEvent` for `integration_configuration_provenance_requirement_committed` |

Forbidden on all retry paths: `datetime.now(...)`, min/max spine heuristics, intent
`requested_at`, hash-derived synthetic time.

## 6. Correlation recovery (`task_id`, `run_id`, `attempt_id`, `execution_id`)

| Field | Durable source | Linkage proof |
|---|---|---|
| `execution_id` | P2 `record.execution_id` | Port / handler closure + pin index `read_all(tenant_id, execution_id)` |
| `task_id` | Pin extension `requirement_execution_correlation.task_id` | Written at pin from `peek_active_execution_task_id()` / governed task invariant |
| `run_id` | Pin extension `requirement_execution_correlation.run_id` | Written at pin from `require_active_execution_identity()[0]` |
| `attempt_id` | Pin extension `requirement_execution_correlation.attempt_id` | Written at pin from `require_active_execution_identity()[1]` |

**Resume corroboration (no global scan):** reload governed `Task` → `RuntimeCheckpoint` must match
extension triple for active configured execution; mismatch → fail closed.

**ExecutionId → attempt/run/task:** deterministic within one obligation via pin extension (not
reverse index scan). Lineage admission under
`ExecutionLineageAttemptScope(tenant_id, task_id, run_id, attempt_id)` must admit
`execution_id` when corroboration runs — integrity check only.

## 7. Subject recovery

For `tenant_id` + `ExecutionId`:

1. Pin store **`read_pin_records`** (sanctioned typed read — see child lock R1-R1-R1-R1-R1);
   `read_all` is provenance-only and does **not** carry subject or staging.
2. For each pin record with `mode == CONFIGURED_ADOPTED`, treat as one independent obligation.
3. Never collapse subjects; never guess subject from catalog or adoption binding.

Pin-only subject set is **complete** for configured-adopted obligations pinned for that execution.

## 8. Crash Case C — algorithm after process restart

```text
restart
→ configured execution re-enters (resume / retry business port)
→ ExecutionBoundConfiguredRelationalStorePort._initialize_adapter
    input: tenant_id, execution_id from port construction (execution-scoped)
→ materialize_validate_and_pin (idempotent)
→ for each CONFIGURED_ADOPTED subject from read_all(tenant_id, execution_id):
      decode pin envelope incl. requirement_recovery_staging.v1
      if spine has no accepted event for deterministic EventId(logical key):
          build ExecutionIntegrationConfigurationProvenanceRequirementFact from pin + extension
          map to RuntimeEvent (no defaults for event_id/timestamp)
          commit port → RuntimeEventBus.record
→ on append success, continue RelationalStoreExecutionAdapter business I/O
```

Every input field maps to §3–§6; no in-memory-only fact.

## 9. Crash Case D — algorithm after process restart

```text
restart
→ same execution-scoped re-entry as §8
→ for each subject obligation:
      load positioned spine event by EventId (deterministic) or idempotent append API
      rebuild RuntimeEvent from stored positioned.event (all fields)
      retry append → reconcile_idempotent_event_acceptance → original position
      assert full model equality PASS vs pre-crash canonical event
→ business I/O continues
```

Mechanically demonstrable via production `RuntimeEventPersistence.append` + integrity rules
already used for governance facts.

## 10. Orphan-pin discovery (pin exists, requirement event absent)

**No new global index.**

Discovery is **execution-scoped** only:

- Inputs: `tenant_id`, `execution_id` from bound port / qualified handler wiring.
- Enumerate subjects: P2 provenance index key
  (`_provenance_index_kv_key`) via `read_all`.
- For each subject: deterministic `EventId` → spine read; absent ⇒ orphan for that obligation.

Diagnostics may surface the same predicate read-only (R1-R1); production repair uses §8.

## 11. Retry ownership (exact component)

| Concern | Owner |
|---|---|
| Repair orchestration after restart | `ExecutionBoundConfiguredRelationalStorePort._initialize_adapter` · `intergrax/integrations/execution_bound_configured_relational_store_port.py` |
| Pin idempotency | `ExecutionBoundIntegrationResolution.materialize_validate_and_pin` |
| Staging write at pin | `KvExecutionIntegrationConfigurationPinningStore.pin` (extended envelope) |
| Fact → event | `RuntimeEventIntegrationConfigurationProvenanceRequirementRecorder` (planned P4) via `ExecutionIntegrationConfigurationProvenanceRequirementCommitPort` |
| Persistence | `RuntimeEventBus.record` → `RuntimeEventPersistence.append` |
| `ExecutionId` source | Constructor argument on port / marketplace qualified execution binding (same invocation scope as pin) |
| Correlation at repair | Pin extension; corroborate with `RuntimeCheckpoint` on task resume |
| Commit invoke | Inserted between pin return and `RelationalStoreExecutionAdapter` construction (R1-R1 §5) |

No `RequirementRecoveryService`, second retry engine, or parallel durable intent repository.

## 12. `MarketplaceToolExecutionIntent` verification

**Not sufficient** for canonical requirement-event identity. Durable fields today:
`tenant_id`, `task_id`, `execution_request_id`, `binding_operation_id`, `capability_identity`,
`provenance` (configured/UCA ops ids) — none provide `run_id`, `attempt_id`, `ExecutionId`,
`IntegrationConfigurationSubject`, factual pin timestamp, or W3C trace triple. Intent remains
pre-EE dispatch truth, not requirement-fact recovery authority.

## 13. End-to-end graph (locked)

```text
P2 pin KV (incl. requirement_recovery_staging.v1)
  + [Case D: positioned requirement spine event]
→ recover ExecutionIntegrationConfigurationProvenanceRequirementFact
→ deterministic RuntimeEvent builder (registry-derived category/ops_hint/phase/severity)
→ RuntimeEventBus.record
→ reconcile_idempotent_event_acceptance
→ configured execution business I/O may continue
```

## 14. Corrected P4 qualification implications

After audit acceptance, corrected P4 must prove (Docker-backed where noted):

1. Crash after pin before event + **process restart** (Case C).
2. Recovery reconstructs original fact from pin extension (+ spine for Case D).
3. Exact timestamp preserved (`requirement_boundary_recorded_at` / spine `timestamp`).
4. Exact `attempt_id` / `run_id` / `task_id` / `execution_id`.
5. Exact subject per obligation.
6. Rebuilt `RuntimeEvent` equals pre-crash canonical event.
7. First append after failed pre-acceptance crash succeeds.
8. Accepted-before-crash retry returns original position (Case D).
9. No new store/lifecycle — only P2 envelope extension + existing pin/spine/bus paths.

## 15. Tracker status

| ID | Status after this docs-only reconciliation |
|---|---|
| `TRACE-X-P5-R2-P4` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1-R1` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1-R1-R1` | **BLOCKED ON R1-R1-R1-R1** (implementation) |
| `TRACE-X-P5-R2-P4-R1-R1-R1-R1` | **BLOCKED ON R1-R1-R1-R1-R1** |
| `TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1` | **READY FOR AUDIT** (child qualification doc) |
| `FRZ-TRC-11` | **OPEN** |
| `P5` | **NOT ENTERED** |
| `CERT` | **NOT ENTERED** |

## 16. Independent audit notice

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
