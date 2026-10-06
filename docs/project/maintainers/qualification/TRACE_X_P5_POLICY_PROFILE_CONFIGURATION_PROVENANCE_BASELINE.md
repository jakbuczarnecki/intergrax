# TRACE-X-P5-P0 — Policy, Profile & Configured→Effective Provenance Baseline

**Parent:** TRACE-X — End-to-End Traceability & Evidence Certification

**Stage:** TRACE-X-P5-P0 (discovery + qualification baseline — **production delta = 0**)

**START_HEAD:** `0102eeabc6d1d59efbecff52737491c96b1d3f0c`

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p5_support.py`

**Status:** TRACE-X-P5-P0 = **READY FOR AUDIT** · TRACE-X-P5 = **CURRENT** (not CLOSED) · **FRZ-TRC-07 / 08 / 11** remain **OPEN**

---

## 1. Scope

**In scope:** closed-world inventories for policy provenance, effective profile revision provenance, and configured→effective configuration provenance; semantic owner matrix; execution join classification; FRZ-TRC-07/08/11 current-HEAD disposition; tenant negatives; authority separation; mechanical gates TXP5P0-Q01..Q13.

**Out of scope:** production contract/runtime changes; FRZ PASS promotion; CONFIG-X provider activation proof; P6 (FRZ-TRC-09/10); new global trace envelopes.

---

## 2. Closed-world inventories (@ START_HEAD)

| Domain | Discovered | Registry | Unknown | Orphan | Duplicate registry keys |
|---|---:|---:|---:|---:|---:|
| Policy | 13 | 13 | 0 | 0 | 0 |
| Profile revision | 17 | 17 | 0 | 0 | 0 |
| Configuration | 8 | 8 | 0 | 0 | 0 |

Discovery: AST scan (`intergrax`, `agents`, `applications`) for typed contract classes + registered composition anchors. Registry parity gates: `test_txp5p0_q02`..`q04`.

---

## 3. Semantic owner matrix (canonical)

| Concern | Exactly-one owner | Canonical contract |
|---|---|---|
| Policy decision / bundle identity (governed boundary) | Governance + execution boundary evidence composition | `PolicyDecisionSection` |
| Governance evidence persistence | Governance evidence plane | `GovernanceEvidenceRef` / `GovernanceEvidenceSection` |
| Policy basis for obligation derivation (app/workspace scope) | Obligation derivation | `PolicyEvidenceBasisV1` |
| Effective profile revision materialization | Profile resolution | `EffectiveProfileRevision` |
| Active revision selection | Profile resolution / activation | `ActiveEffectiveProfileRevisionBinding` |
| Execution-used revision (pinning) | Profile resolution / execution pinning | `EffectiveProfileExecutionBinding` |
| Host admission gate | Profile resolution (`EffectiveProfileRevisionAdmission`) | `EffectiveProfileRevisionAdmissionPort` |
| Configured capability identity | Integrations / INT-CONFIG realization | `ConfiguredCapabilityBinding` |
| Configured≠effective invariant | Integrations contracts | `validate_realization_request_invariants` |
| Runtime inspection | Read/projection only | inspection sections — **not** provenance authority |
| Factual execution reconstruction | `ExecutionReconstructor` | `ExecutionReconstruction` (derived; no policy/profile/config joins today) |

---

## 4. Producer → persistence → execution → reconstruction

### 4.1 Policy (`FRZ-TRC-07`)

```text
policy evaluation / governance recording
  → GovernanceDecisionEvidenceFact (persisted) + GovernanceEvidenceRef
  → ExecutionBoundaryEvent.policy (PolicyDecisionSection)
  → optional RuntimeInspection governance projection (read-only)
```

**Execution join (typed):** `ExecutionBoundaryEvent.task_id` + `run_id` (+ boundary `event_id`) → `PolicyDecisionSection` bundle fields; `GovernanceEvidenceRef.evidence_id` → persisted governance fact.

**Gap:** `ExecutionReconstructor` does not compose policy revision into global execution reconstruction (**P5-GAP-01** → proposed **TRACE-X-P5-R1**).

### 4.2 Profile revision (`FRZ-TRC-08`)

```text
configured layers
  → materialize_effective_profile_revision → EffectiveProfileRevision (store)
  → activation → ActiveEffectiveProfileRevisionBinding
  → admit_root_execution → pin_effective_profile_revision_for_execution
  → EffectiveProfileExecutionBinding (tenant_id, execution_id, revision_id, fingerprint)
  → checkpoint metadata on Task (evidence only)
  → runtime inspection providers (read-only)
```

**Execution join (typed):** `EffectiveProfileExecutionBinding.execution_id` + `tenant_id` → `revision_id` + `fingerprint`.

**Re-audit vs P0 TX-B01:** profile-resolution contracts + pinning store constitute a **domain SSOT** for pinned revisions; P0 “no global SSOT” is **superseded at contract level** but **global TRACE-X closure** still blocked when admission port is unwired (**P5-GAP-02**) and reconstructor lacks join (**P5-GAP-01** overlaps execution-wide attribution).

### 4.3 Configured→effective (`FRZ-TRC-11`)

```text
configuration payload (configuration_type/version/fingerprint)
  → validate_realization_request_invariants (configured fingerprint match — fail-closed)
  → ExistingCapabilityConfigurationRealizationService.realize_admitted
  → ConfiguredCapabilityBinding
  → (optional) task_id/run_id on request — correlation within INT-CONFIG scope only
```

**Execution join (scoped):** realization `request_id` / optional `RunId`+`TaskId` → `ConfiguredCapabilityBinding.configuration_fingerprint`.

**Gap:** no canonical global chain to `ExecutionId` / `RuntimeEvent` outside INT-CONFIG (**P5-GAP-04** → proposed **TRACE-X-P5-R2**).

---

## 5. FRZ disposition (@ current HEAD — no PASS)

| Criterion | Disposition | Evidence present | Missing for closure |
|---|---|---|---|
| FRZ-TRC-07 | PARTIAL_CURRENT_HEAD | `PolicyDecisionSection`, GOV-X1/X2 scoped negatives, obligation `PolicyEvidenceBasisV1` | Global execution→policy revision reconstruction; all execution paths |
| FRZ-TRC-08 | PARTIAL_CURRENT_HEAD | Profile resolution SSOT + `EffectiveProfileExecutionBinding` | Optional admission wiring; reconstructor join; full-path negatives |
| FRZ-TRC-11 | PARTIAL_CURRENT_HEAD | INT-CONFIG-REAL-X within scope; fingerprint invariants | Global configured→execution→evidence chain |

---

## 6. Tenant audit (P5 local)

| Check | Mechanism | Result |
|---|---|---|
| Config realization tenant | `verify_admitted_authorization_evidence` TENANT_MISMATCH | fail-closed (TXP5P0-Q10) |
| Profile pinning tenant | pinning store keyed by `(tenant_id, execution_id)` | cross-tenant read returns None (TXP5P0-Q11) |
| Policy / inspection | governance read adapters enforce scope tenant + task/run/attempt/execution | fail-closed on mismatch |

**Global FRZ-TEN:** not claimed.

---

## 7. Authority separation

| Invariant | Status |
|---|---|
| policy evidence ≠ authorization ≠ execution admission | preserved — P5 read-only inventory |
| profile revision evidence ≠ execution authority | preserved — admission port separate from evidence |
| configured capability ≠ effective capability | preserved — INT-CONFIG binding is configured identity only |

---

## 8. Blockers / debt / proposed children

| ID | Classification | Summary | Child |
|---|---|---|---|
| P5-GAP-01 | IN-SCOPE BLOCKER | No reconstructor join for policy revision | TRACE-X-P5-R1 |
| P5-GAP-02 | IN-SCOPE BLOCKER | Optional profile admission on `HostTask` | TRACE-X-P5-R1 |
| P5-GAP-03 | TRACKED FREEZE DEBT | TX-B01 reclassified — contract SSOT exists; global closure still open | — |
| P5-GAP-04 | IN-SCOPE BLOCKER | No global config fingerprint → execution evidence chain | TRACE-X-P5-R2 |
| P5-GAP-05 | TRACKED FREEZE DEBT | TXP1R1-Q02 / TXP1R1-Q23 | — |

**Recommendation:** **TRACE-X-P5 = BLOCKED** for independent closure until children remediate blockers; **TRACE-X-P5-P0 = READY FOR AUDIT** for baseline inventory only.

---

## 9. Mechanical gates

Entrypoint: `tests/qualification/trace_x/test_trace_x_p5_p0_baseline.py` (TXP5P0-Q01..Q13).

Negative sensitivity: synthetic unregistered surfaces TXP5P0-Q06..Q08.

---

## 10. Tests (@ P5-P0 implementation)

```text
uv run pytest tests/qualification/trace_x/test_trace_x_p5_p0_qualification_gates.py -p no:xdist -q
```

**Result:** 13 passed @ implementation time (qualification gates only).

---

## 11. Enterprise audit matrix (P5-P0 baseline)

| Dimension | Result |
|---|---|
| Layer boundaries | PASS — owners locked in §3 |
| Ownership | PASS — inventories name single owner per identity class |
| Contracts / typing | PASS — no dict-bag provenance in scope |
| Composition | PARTIAL — host admission optional |
| Pluginability | PASS — strategies / ports pattern |
| Bypass resistance | PARTIAL — unwired admission path |
| Fail-closed | PASS — fingerprint + tenant negatives gated |
| Tenant isolation (local) | PASS |
| Regression protection | PASS — closed-world parity gates |

---

**Independent audit required.** Cursor report is not closure authority.
