# UCA-6C R6 — Governed Capability Acquisition Enterprise Certification

## 1. Certification metadata

| Field | Value |
| ----- | ----- |
| **Task** | UCA-6C-R6-CERT-RERUN-3 |
| **Date** | 2026-09-26 (operator session, UTC+2) |
| **Branch** | `development` |
| **CERTIFIED_UCA_CODE_BASELINE** | `a38f70bc878a4e61807ce93cb0bfa600fdaca168` |
| **AUDIT_HEAD** | `a38f70bc878a4e61807ce93cb0bfa600fdaca168` |
| **FINAL_EVIDENCE_COMMIT** | *(recorded after docs-only commit; see §3)* |

**Separation:** `CERTIFIED_UCA_CODE_BASELINE` is the audited **code** SHA. The evidence/documentation commit that replaces this file is a **docs-only** commit on the same branch and does not change the certified code baseline.

**Method:** read-only certification — bounded architecture review (§14 read scope), negative searches, sequential bounded pytest (T1–T8), fresh-process import proofs (T8), bounded Pyright and Ruff on certified seams. **Production mutation = 0**, **test mutation = 0**, **config mutation = 0** during evidence collection.

## 2. Scope

Enterprise certification of **Governed Capability Acquisition (UCA-6C R6)** at the certified code baseline. Verifies exactly-one ownership, hard layer boundaries, contract-first acquisition, canonical Execution and HITL, durable restart/reentry, fencing/idempotency, SQLite/CW typed seam, Tool Registry boundary, and fail-closed relational category semantics. Does **not** perform Architecture Freeze, R7 scenario work, or full-platform Integrations migration proof.

**Explicitly not claimed:** all Integrations migration tests green (`test_provider_category_contract_migration.py` not in scope).

## 3. Certified baseline

| Check | Result |
| ----- | ------ |
| `BRANCH` | `development` |
| `HEAD` | `a38f70bc878a4e61807ce93cb0bfa600fdaca168` |
| `origin/development` | `a38f70bc878a4e61807ce93cb0bfa600fdaca168` |
| Tracked worktree at audit start | **clean** |
| Baseline drift | **none** |

Replaces prior certification candidates: `e01f169e…`, `ba7bb571…`, `c471a338…` as the **final** UCA R6 code baseline for this certification cycle.

## 4. Baseline lineage / remediation history

| Commit | Role | Classification |
| ------ | ---- | -------------- |
| `69e4a698b…` | Historical certification artifact (`docs(uca): certify governed capability acquisition r6`) | **SUPERSEDED / NOT FINAL ACCEPTED CERTIFICATION** — later independent audit found certified-seam static blockers, subsequently remediated and re-certified here |
| `7f609cf20d5645e2703dd2da4081c9eaf41c4b04` | Certified-seam static blocker remediation | **UCA code remediation** |
| `ba7bb5719b1ee518dba00e6f18894d5ea85f9cdb` | Restore canonical relational provider typing | **UCA code remediation** |
| `c471a33879b4d69a0bc07ea7a0ae5496b097df0c` | Close relational category contract blockers | **UCA code remediation** |
| `a38f70bc878a4e61807ce93cb0bfa600fdaca168` | Enforce relational category invariant | **UCA code remediation (CERTIFIED_UCA_CODE_BASELINE)** |
| `33175f8e6`, `29c721356` | EBH enterprise documentation sync | **OUT-OF-SCOPE PARALLEL DOCUMENTATION CHANGE** (no UCA code/test surface change) |
| `29a270072` | Env example UTF-8 normalization | **OUT-OF-SCOPE** (config example; not UCA certified seam) |

## 5. Authoritative architecture

1. `docs/project/maintainers/architecture/UCA_6C_CANONICAL_HITL_BOUNDARY_RECONCILIATION.md`
2. `docs/project/technical/adr/entries/2026-09-23/ADR-UCA-6C-AGENT-GOVERNANCE-CANONICAL-HITL-RECONCILIATION.md`
3. `docs/project/technical/adr/entries/2026-09-22/ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md`
4. `docs/project/technical/adr/entries/2026-09-22/ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY.md`
5. `docs/project/maintainers/architecture/EXECUTION_ENGINE_OWNERSHIP_MODEL.md`

## 6. Canonical UCA flow

```text
Worker recovery
    ↓
canonical capability discovery
    ↓
true capability gap
    ↓
acquisition coordination
    ↓
capability qualification
    ↓
binding
    ↓
canonical Execution request
    ↓
Execution Engine admission
    ↓
Execution Engine owns lifecycle
    ↓
ExecutionIdentityAuthority owns identity
    ↓
bound capability execution
    ↓
ToolRuntime
    ↓
exact tool invocation
    ↓
Governance authorities
    ↓
canonical Execution-owned HITL
    ↓
durable suspended-operation reentry
    ↓
same protected operation continues
```

No parallel ownership path evidenced by T3 architecture gates and UCA unit corpus.

## 7. Enterprise owner matrix

| Concern | Canonical owner | Duplicate semantic owner? | Alternate path? | Bypass? | Result |
| ------- | ----------------- | ------------------------- | --------------- | ------- | ------ |
| Need | Consumer | No | No | No | PASS |
| Capability discovery | Capability Catalog | No | No | No | PASS |
| Marketplace recommendation | Marketplace | No | No | No | PASS |
| Acquisition coordination | Capability Acquisition | No | No | No | PASS |
| Capability qualification | Capability Qualification | No | No | No | PASS |
| Provider/environment qualification | Core Qualification | No | No | No | PASS |
| Binding | Qualification / domain handoff | No | No | No | PASS |
| Worker responsibility recovery | Autonomous Work | No | No | No | PASS |
| Execution lifecycle | Execution Engine | No | No | No | PASS |
| Execution identity | ExecutionIdentityAuthority | No | No | No | PASS |
| Tool invocation | ToolRuntime | No | No | No | PASS |
| Governance decision | Governance | No | No | No | PASS |
| Agent Governance approval | Agent Runtime Governance | No | No | No | PASS |
| Declarative HITL | Declarative Policy | No | No | No | PASS |
| Meaningful Side Effect authority | MSE Governance | No | No | No | PASS |
| Human pause/resume lifecycle | ExecutionContinuationPort | No | No | No | PASS |
| Suspended invocation payload | SuspendedExecutionOperationStore | No | No | No | PASS |
| Code synthesis | CodeCraft | No | No | No | PASS |
| Sandbox execution | Sandbox | No | No | No | PASS |
| Nexus orchestration | Execution Engine internal only | No | No public UCA dependency | No | PASS |
| Relational category semantics | `RelationalStoreIntegrationContract` | No | No | No | PASS |
| SQLite provider mechanics | SQLite integration provider | No | No | No | PASS |
| Runtime SQLite composition | `runtime/persistence` | No | No | No | PASS |
| CW persistence materialization contract | Collaborative Work | No | No | No | PASS |
| Tool Registry runtime-safe root | Tools subsystem | No | No | No | PASS |
| Tool Registry composition | explicit registry leaf modules | No root re-export | No | PASS |

## 8. GCF invariants

| Invariant | Result | Evidence |
| --------- | ------ | -------- |
| GCF-INV-001 Coordination != Ownership | PASS | T3 + `test_uca6c_r6_architecture_gates.py` |
| GCF-INV-002 Qualification != Authorization | PASS | T1-B sequential authority / governance tests |
| GCF-INV-003 Acquisition != Execution lifecycle | PASS | T3 EE ownership; acquisition coordination surfaces |
| GCF-INV-004 Binding != Execution | PASS | T1 worker resume / binding tests |
| GCF-INV-005 Capability growth != Authority growth | PASS | T1-A negative gates |
| GCF-INV-006 No second HITL | PASS | T1-B HITL hardening; T3 |
| GCF-INV-007 No second Execution Engine | PASS | `test_ee_a1_execution_engine_ownership_certification_gate.py` |
| GCF-INV-008 No public Nexus dependency | PASS | Architecture gates; bounded UCA contract packages |
| GCF-INV-009 ToolRuntime mandatory | PASS | T1-C / canonical tool runtime tests |
| GCF-INV-010 True capability gap only after canonical discovery | PASS | T1-A discovery / durable obstacle tests |

## 9. Certification invariants C-01–C-20

| ID | Result | Evidence |
| -- | ------ | -------- |
| C-01 Discovery ownership | PASS | T1-A catalog/discovery |
| C-02 Acquisition coordination only | PASS | T1-A acquisition; no EE lifecycle in acquisition |
| C-03 Contract-first acquisition | PASS | contracts + strategy policy modules |
| C-04 Qualification boundary | PASS | T1 qualification vs governance |
| C-05 Binding != Execution | PASS | T1 binding / resume |
| C-06 Execution ownership | PASS | T3 |
| C-07 Worker recovery != Execution resume | PASS | T1 continuation / resume family |
| C-08 No pre-approval transport | PASS | grep: no `governance_approval_evidence` in AW / capability_acquisition |
| C-09 Authority separation | PASS | T1-B governance sequencing |
| C-10 Sequential authority behavior | PASS | T1-B sequential authority closure |
| C-11 Single ExecutionContinuationPort | PASS | T1 continuation single-owner |
| C-12 Single SuspendedExecutionOperationStore | PASS | T1-B suspended operation tests |
| C-13 Durable restart | PASS | T1 restart / identity correlation |
| C-14 Multi-host safety | PASS | T1 multi-host fencing |
| C-15 Crash windows | PASS | T1 crash window module |
| C-16 Canonical idempotency | PASS | T1 durable terminal / recovery E2E |
| C-17 SQLite composition boundary | PASS | T4 + `sqlite_composition.py` |
| C-18 Tool Registry root boundary | PASS | T5 + T8 |
| C-19 SQLite/CW typed contract seam | PASS | T4, T6, T7 |
| C-20 SQLite invalid config fail-closed | PASS | T4; relational fail-closed in T7 |

## 10. Proof-family matrix

| Proof family | Result | Primary evidence |
| ------------ | ------ | ---------------- |
| Discovery / gap | PASS | T1-A |
| Acquisition / qualification / binding | PASS | T1-A, T1-B |
| Execution admission / lifecycle | PASS | T1-B, T3 |
| Execution identity | PASS | T3 `test_ee_a2_identity_authority_certification.py` |
| Governance / HITL / sequential authority | PASS | T1-B |
| Worker E2E / restart / reentry | PASS | T1 |
| Multi-host / fencing / crash / idempotency | PASS | T1-B |
| SQLite provider + runtime composition | PASS | T4 |
| Relational category contract (RC-01–RC-08) | PASS | T7 |
| CW persistence / materialization | PASS | T6 |
| Tool Registry import boundary | PASS | T5, T8 |

## 11. Dynamic test inventory

Measured from tracked files at `CERTIFIED_UCA_CODE_BASELINE` (`git ls-files`, pattern `test_uca6c*.py`):

| Bucket | Files |
| ------ | ----: |
| **All tracked UCA unit** | **48** |
| T1-A `tests/unit/autonomous_work/*` | 20 |
| T1-B `tests/unit/runtime/execution/*` | 19 |
| T1-C `tests/unit/runtime/nexus/*` + `tests/unit/tools/*` | 4 |
| T1-D remainder | 5 |
| **Partition check** T1-A+B+C+D == all | **True** |
| Tracked UCA integration (`tests/integration`) | **0** |

## 12. Test results T1–T8

| Phase | Files / scope | Passed | Failed | Skipped | Result |
| ----- | ------------- | -----: | -----: | ------: | ------ |
| T1-A | 20 files | 122 | 0 | 1 | PASS |
| T1-B | 19 files | 114 | 0 | 0 | PASS |
| T1-C | 4 files | 26 | 0 | 0 | PASS |
| T1-D | 5 files | 29 | 0 | 0 | PASS |
| T2 | — | — | — | — | **NO TRACKED UCA INTEGRATION FILES** |
| T3 | 4 required gate modules | 32 | 0 | 0 | PASS |
| T4 | 3 SQLite seam modules | 16 | 0 | 0 | PASS |
| T5 | `tests/unit/tools/registry/` (incl. `test_runtime_import_boundary.py`) | 25 | 0 | 0 | PASS |
| T6 | `test_persistence_provider_binding.py` | 24 | 0 | 0 | PASS |
| T7 | `test_provider_category_contracts.py` | 19 | 0 | 0 | PASS |
| T8 | 2 sequential `uv run python -c` imports | 2 | 0 | 0 | PASS |

**T1-A skip (NON-BLOCKING SKIP):** `tests/integration/autonomous_work/conftest.py:71` — PostgreSQL backend unavailable. Optional integration backend; not the sole proof of any C-01–C-20 or GCF invariant; expected when PG is not configured.

**T2:** `integrationUca.Count == 0` — reported as required; not a failure.

## 13. Former blockers B1–B12

| ID | Status | Evidence |
| -- | ------ | -------- |
| B1 Clock / stale datetime patch | **CLOSED** | T1-B caller-held resume authority tests |
| B2 Tool Registry import/collection cycle | **CLOSED** | T5, T8, T1-C |
| B3 SQLite/CW typing regression | **CLOSED** | T4, T6; Pyright 0 on CW/SQLite certified symbols |
| B4 Permissive SQLite path coercion | **CLOSED** | grep `Path(str(` → 0 in sqlite provider; T4 |
| B5 Weak Tool Registry boundary | **CLOSED** | T5 `test_runtime_import_boundary.py` |
| B6 CW Protocol missing bodies | **CLOSED** | T6; Pyright clean on CW Protocols |
| B7 SQLite db_path str/Path mismatch | **CLOSED** | `SqliteRelationalStoreClient.db_path -> Path`; T4 |
| B8 RootExecutionLaunchRequest generic arity | **CLOSED** | Pyright clean on dispatch + `root_execution_launch.py`; T3 |
| B9 SQLite-local for_provider / private category dependency | **CLOSED** | grep sqlite tree: 0 `_CONNECT_READ_WRITE_HEALTH` / `category_for_provider` imports |
| B10 Relational schema_id hierarchy incompatibility | **CLOSED** | T7 RC proofs; Pyright clean on `RelationalStoreIntegrationContract` |
| B11 Relational for_provider hierarchy incompatibility | **CLOSED** | T7; typed `for_provider -> Self` |
| B12 Relational integration_kind direct-construction bypass | **CLOSED** | T7 foreign-kind rejection tests |

## 14. Static typing results (certified seam Pyright)

**Command:** bounded `uv run pyright` on §26 file list at `AUDIT_HEAD`.

**Certified symbols (zero diagnostics required):** `RelationalStoreIntegrationContract` (`schema_id`, `integration_kind`, `for_provider`); SQLite `db_path` / integration / bundle / runtime composition; CW `CollaborativeWorkPersistenceFactory`, `CollaborativeWorkMaterializationBinder`, `CollaborativeWorkPersistenceProvider`; `RootExecutionLaunchRequest`; `QualifiedCapabilityExecutionDispatchService._build_launch_request`.

**Result:** **0 diagnostics** on certified UCA relational/SQLite/CW/Execution symbols.

## 15. Exact Pyright finding classification

| Location | Diagnostic | Symbol | Classification |
| -------- | ---------- | ------ | -------------- |
| `data.py:79` | reportIncompatibleVariableOverride | `KeyValueCacheIntegrationContract.schema_id` | **OUT-OF-SCOPE STATIC DEBT** |
| `data.py:89` | reportIncompatibleMethodOverride | `KeyValueCacheIntegrationContract.for_provider` | **OUT-OF-SCOPE STATIC DEBT** |
| `data.py:113` | reportIncompatibleVariableOverride | `GraphStoreIntegrationContract.schema_id` | **OUT-OF-SCOPE STATIC DEBT** |
| `data.py:123` | reportIncompatibleMethodOverride | `GraphStoreIntegrationContract.for_provider` | **OUT-OF-SCOPE STATIC DEBT** |

**OUT-OF-SCOPE STATIC DEBT disclosure:** These diagnostics belong to non-UCA Integration categories and are not used by the certified relational/SQLite/CW UCA seam. They do not establish global Integrations static cleanliness and must not be represented as fixed by this certification.

No other Pyright diagnostics on the bounded file set. **Certified Pyright blocker count = 0.**

## 16. Ruff classification

| Command | Result |
| ------- | ------ |
| `ruff check` (§29 file list) | **All checks passed** |
| `ruff format --check` | Would reformat `intergrax/contracts/root_execution_launch.py`, `intergrax/runtime/integrations/contracts.py` |

**NON-BLOCKING PRE-EXISTING QUALITY OBSERVATION:** format drift on two files only; no semantic/layer/import finding. No Ruff blocker on certified seams.

## 17. Negative architecture checks (bounded)

| Check | Result |
| ----- | ------ |
| Second Execution Engine | **absent** (T3) |
| Second HITL | **absent** (T1-B, T3) |
| Direct tool invocation bypass | **absent** (T3 zero-bypass gate) |
| AW-owned Execution lifecycle | **absent** |
| Capability Acquisition-owned execution | **absent** |
| Capability Qualification-owned authorization | **absent** |
| Pre-approval transport | **absent** (grep AW/UCA acquisition) |
| Public Nexus dependency (UCA surfaces) | **absent** |
| Duplicate ExecutionContinuationPort | **absent** |
| Duplicate SuspendedExecutionOperationStore | **absent** |
| SQLite-owned runtime composition | **absent** (T4) |
| CW-owned provider selection | **absent** (T6 handoff) |
| Tool Registry root composition export | **absent** (T5, T8) |
| Private category helper leakage into SQLite provider | **absent** (grep) |

## 18. Out-of-scope debt

- **Integrations migration:** `tests/unit/integrations/providers/test_provider_category_contract_migration.py` — not executed; not UCA PASS/FAIL input.
- **Pyright:** four KeyValueCache/GraphStore override diagnostics (§15).
- **Ruff format:** two certified-surface files (§16).

## 19. Out-of-scope parallel documentation changes

Commits `33175f8e6`, `29c721356` (EBH documentation sync) between remediation commits — **OUT-OF-SCOPE PARALLEL DOCUMENTATION CHANGE** relative to UCA code baseline.

## 20. Architecture checkpoint (18×)

| # | Question | Answer |
| - | -------- | ------ |
| 1 | Capability Acquisition coordinates only? | **YES** |
| 2 | Qualification separated from Authorization? | **YES** |
| 3 | Binding separated from Execution? | **YES** |
| 4 | Execution Engine sole lifecycle owner? | **YES** |
| 5 | ExecutionIdentityAuthority sole identity owner? | **YES** |
| 6 | ToolRuntime mandatory boundary? | **YES** |
| 7 | No second HITL? | **YES** |
| 8 | Durable resume continues same operation? | **YES** |
| 9 | Restart/fencing/multi-host still proven? | **YES** |
| 10 | SQLite does not own runtime composition? | **YES** |
| 11 | CW consumes typed contracts? | **YES** |
| 12 | Tool Registry root lightweight? | **YES** |
| 13 | SQLite does not use private category helpers locally? | **YES** |
| 14 | Relational category invariant fail-closed? | **YES** |
| 15 | Relational schema invariant fail-closed? | **YES** |
| 16 | No duplicate semantic owner? | **YES** |
| 17 | No architecture bypass? | **YES** |
| 18 | Certified seams strongly typed? | **YES** |

## 21. Exit criteria

All required gates at audit HEAD:

- GCF-INV-001–010 = PASS
- C-01–C-20 = PASS
- T1 = PASS (1 non-blocking skip)
- T2 = NO TRACKED FILES (acceptable)
- T3–T8 = PASS
- B1–B12 = CLOSED
- Certified Pyright blockers = 0
- Certified Ruff blockers = 0
- Architecture checkpoint = 18× YES
- Production / test / config mutation during CERT = 0

## 22. Final verdict

```text
UCA-6C R6 ENTERPRISE CERTIFICATION = PASS

CERTIFIED_UCA_CODE_BASELINE =
a38f70bc878a4e61807ce93cb0bfa600fdaca168

BLOCKER COUNT = 0

R6-FREEZE ELIGIBILITY = YES

ARCHITECTURE FROZEN = NO

NEXT REQUIRED STEP =
INDEPENDENT GITHUB AUDIT OF UCA-6C-R6-CERT-RERUN-3
```

---

**Historical certification attempt:** `69e4a698b…`
**Status:** SUPERSEDED / INVALID AS FINAL CERTIFICATION
**Reason:** later independent audit found certified-seam static blockers, which were subsequently remediated and re-certified at `a38f70bc…`.

**GITHUB AUDIT REQUIRED:** The final certification document, exact certified code baseline, evidence commit, test results, static finding classification, ownership matrix, architecture invariants, and all certified seams must be independently audited against the code stored on GitHub. This artifact does not constitute final acceptance. R6-FREEZE must not begin until the exact GitHub evidence commit for UCA-6C-R6-CERT-RERUN-3 has been independently accepted.
