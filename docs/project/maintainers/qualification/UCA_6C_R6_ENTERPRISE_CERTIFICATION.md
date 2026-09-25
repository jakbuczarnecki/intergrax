# UCA-6C R6 — Governed Capability Acquisition Enterprise Certification

## 1. Certification metadata

| Field | Value |
| ----- | ----- |
| **Task** | UCA-6C-R6-CERT-RERUN — Enterprise Certification Rerun |
| **Date** | 2026-09-25 (operator session, UTC+2) |
| **Branch** | `development` |
| **CERTIFIED_UCA_CODE_BASELINE** | `e01f169e7a544db15bb2ad62560c6996bf4163a1` |
| **AUDIT_HEAD** | `e01f169e7a544db15bb2ad62560c6996bf4163a1` |
| **Artifact parent code baseline** | `e01f169e7a544db15bb2ad62560c6996bf4163a1` (evidence collected at this SHA; docs-only commit follows) |
| **Worktree state before evidence write** | clean tracked tree; `START_HEAD == origin/development == CERTIFIED_UCA_CODE_BASELINE` |

**Method:** read-only certification — static audit of §14 surfaces, targeted Ruff/Pyright, bounded pytest matrix (sequential `uv run pytest`), fresh-process import proofs. **Production mutation = 0**, **test mutation = 0** during evidence collection.

## 2. Scope

This artifact records enterprise certification of **Governed Capability Acquisition (UCA-6C R6)** on the certified code baseline. It does **not** implement features, refactor production, or perform Architecture Freeze. Certification verifies ownership, contracts, governance sequencing, durable restart/reentry, SQLite/CW seams, and Tool Registry boundaries via tests and bounded static review.

## 3. Baseline drift

| Commit | Scope | UCA impact | Classification |
| ------ | ----- | ---------- | -------------- |
| — | — | — | **No drift** — `START_HEAD`, `origin/development`, and `CERTIFIED_UCA_CODE_BASELINE` are identical (`e01f169e7a544db15bb2ad62560c6996bf4163a1`). |

**Recorded pre-flight:** `BRANCH=development`, `WORKTREE_STATE=clean`, `STASH_STATE=9 stashes present (untouched)`.

## 4. Authoritative architecture

Primary references (authority for certified flow):

1. `docs/project/maintainers/architecture/UCA_6C_CANONICAL_HITL_BOUNDARY_RECONCILIATION.md`
2. `docs/project/technical/adr/entries/2026-09-23/ADR-UCA-6C-AGENT-GOVERNANCE-CANONICAL-HITL-RECONCILIATION.md`
3. `docs/project/technical/adr/entries/2026-09-22/ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md`
4. `docs/project/technical/adr/entries/2026-09-22/ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY.md`

## 5. Canonical flow

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
ExecutionIdentityAuthority owns ExecutionId
    ↓
bound capability execution
    ↓
ToolRuntime
    ↓
exact tool invocation
    ↓
Governance authorities
    ↓
canonical Execution-owned HITL when required
    ↓
durable suspended-operation reentry
    ↓
same protected operation continues
```

## 6. Enterprise owner matrix

| Concern | Canonical owner | Duplicate found? | Result |
| ------- | ----------------- | ---------------- | ------ |
| Need | Consumer | No | PASS |
| Capability discovery | Capability Catalog | No | PASS |
| Marketplace recommendation | Marketplace | No | PASS |
| Acquisition coordination | Capability Acquisition | No | PASS |
| Capability qualification | Capability Qualification | No | PASS |
| Provider/environment qualification | Core Qualification | No | PASS |
| Binding | Qualification / domain handoff | No | PASS |
| Worker responsibility recovery | Autonomous Work | No | PASS |
| Execution lifecycle | Execution Engine | No | PASS |
| Execution identity | ExecutionIdentityAuthority | No | PASS |
| Tool invocation | Tools / ToolRuntime | No | PASS |
| Governance decision | Governance | No | PASS |
| Agent Governance approval | Agent Runtime Governance | No | PASS |
| Declarative HITL authority | Declarative Policy | No | PASS |
| Meaningful Side Effect authority | MSE Governance | No | PASS |
| Human pause/resume lifecycle | ExecutionContinuationPort | No | PASS |
| Suspended invocation payload | SuspendedExecutionOperationStore | No | PASS |
| Code synthesis | CodeCraft | No | PASS |
| Sandbox execution | Sandbox | No | PASS |
| Nexus orchestration | Execution Engine internal only | No public UCA dependency | PASS |
| SQLite provider mechanics | SQLite integration provider | No | PASS |
| Runtime SQLite composition | `runtime/persistence` | No | PASS |
| CW materialization contracts | Collaborative Work | No | PASS |
| Tool Registry runtime-safe root | Tools subsystem | No | PASS |
| Tool Registry composition | explicit `registry.wiring` / `factory` / `bootstrap` / `catalog` leaf modules | No root composition re-export | PASS |

**Static evidence:** `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py`, `test_uca6c_r6_architecture_gates.py`, `test_platform_execution_unification_u5_final_zero_bypass.py`, acquisition/qualification unit corpus; no `governance_approval_evidence` in `intergrax/autonomous_work` or `intergrax/capability_acquisition` (grep at audit HEAD).

## 7. GCF invariants

| Invariant | Result | Evidence |
| --------- | ------ | -------- |
| GCF-INV-001 coordination != ownership | PASS | Architecture gates + `test_uca6c_r6_continuation_single_owner.py` |
| GCF-INV-002 qualification != authorization | PASS | `test_uca6c_r6_r5_7_sequential_authority_generations.py`, architecture pause gates |
| GCF-INV-003 acquisition != lifecycle | PASS | `acquisition_service.py` coordination surface; EE ownership gate |
| GCF-INV-004 binding != execution | PASS | `test_uca6c_worker_qualified_capability_resume.py`, binding/execution unit tests |
| GCF-INV-005 capability growth != authority growth | PASS | `test_uca6c_r6_tigae_negative_gates.py` |
| GCF-INV-006 no second HITL | PASS | `test_uca6c_r6_r5_6_h1_reentry_hardening.py`, agent governance pause gates |
| GCF-INV-007 no second Execution Engine | PASS | `test_ee_a1_execution_engine_ownership_certification_gate.py` |
| GCF-INV-008 no public Nexus dependency | PASS | No Nexus imports under `intergrax/contracts/autonomous_work` or `capability_acquisition` (grep) |
| GCF-INV-009 ToolRuntime mandatory | PASS | `test_uca6c_r5_canonical_tool_runtime.py` |
| GCF-INV-010 true gap after complete discovery | PASS | `test_uca6c_r6_r5_9_durable_obstacle_capability_need.py`, catalog discovery tests |

## 8. Certification invariants C-01–C-20

| ID | Result | Evidence |
| -- | ------ | -------- |
| C-01 Discovery ownership | PASS | `catalog_canonical_discovery_service.py` + AW discovery unit tests |
| C-02 Acquisition coordination only | PASS | `acquisition_service.py` / registry; no lifecycle in acquisition package |
| C-03 Contract-first acquisition | PASS | Typed ports in contracts; strategy policy module |
| C-04 Qualification boundary | PASS | Qualification vs governance tests in AW + execution corpus |
| C-05 Binding != Execution | PASS | `qualified_capability_binding_service.py` + worker resume tests |
| C-06 Execution ownership | PASS | T3 EE ownership + dispatch adapter tests |
| C-07 Worker recovery != Execution resume | PASS | `test_uca6c_r_production_resume.py`, continuation single-owner |
| C-08 No pre-approval transport | PASS | grep: no `governance_approval_evidence` in AW/UCA acquisition paths |
| C-09 Authority separation | PASS | Sequential authority + agent/declarative/MSE tests |
| C-10 Sequential authority behavior | PASS | `test_uca6c_r6_r5_7_r2_sequential_authority_closure.py` |
| C-11 Single ExecutionContinuationPort | PASS | `test_uca6c_r6_continuation_single_owner.py` |
| C-12 Single SuspendedExecutionOperationStore | PASS | suspended_operation test family (T1-B) |
| C-13 Durable restart | PASS | `test_uca6c_durable_restart_identity_correlation.py`, restart E2E tests |
| C-14 Multi-host safety | PASS | `test_uca6c_r6_r5_9_r2_multi_host_fencing.py`, reclaim transport tests |
| C-15 Crash windows | PASS | `test_uca6c_r6_r5_9_r3_crash_windows.py` |
| C-16 Canonical idempotency | PASS | durable terminal outcome + distributed recovery E2E |
| C-17 SQLite composition boundary | PASS | T4 + `sqlite_composition.py` / provider bundle gates |
| C-18 Tool Registry root boundary | PASS | T5 + `tools/registry/__init__.py` exact `__all__` |
| C-19 SQLite/CW typed contract seam | PASS | T4, T6, `test_persistence_provider_binding.py` |
| C-20 SQLite invalid config fail-closed | PASS | T4; grep: no `Path(str(` in sqlite provider tree |

## 9. Proof-family matrix

| Proof family | Result | Evidence |
| ------------ | ------ | -------- |
| Discovery | PASS | T1-A catalog/discovery tests |
| Acquisition | PASS | AW R2/R3 tests + acquisition surfaces |
| Qualification | PASS | binding service + qualification contracts |
| Binding | PASS | worker qualified capability resume tests |
| Execution admission | PASS | execution intake/dispatch tests |
| Execution identity | PASS | T3 `test_ee_a2_identity_authority_certification.py` |
| Agent Governance | PASS | agent governance pause/resume tests |
| Declarative HITL | PASS | sequential authority + HITL hardening tests |
| MSE governance | PASS | strict governance composition tests |
| Sequential authorities | PASS | T1-B sequential authority modules |
| Worker E2E | PASS | worker governed execution E2E family |
| Restart | PASS | restart handoff + true restart E2E |
| Reentry | PASS | canonical reentry fencing tests |
| Multi-host | PASS | multi-host fencing + cross-host transport |
| Fencing | PASS | claim authority propagation tests |
| Crash windows | PASS | crash window module |
| Idempotency | PASS | durable terminal + recovery E2E |
| SQLite runtime composition | PASS | T4 runtime persistence tests |
| SQLite/CW typed provider seam | PASS | T6 + T4 |
| Tool Registry import boundary | PASS | T5 + T8 cold import |

## 10. Test inventory

| Category | Count |
| -------- | ----: |
| Tracked UCA unit files (`test_uca6c*.py`) | **48** |
| Tracked UCA integration files | **0** |

**Grouping (files must sum to 48):**

| Group | Files |
| ----- | ----: |
| T1-A `tests/unit/autonomous_work/**/test_uca6c*.py` | 19 |
| T1-B `tests/unit/runtime/execution/**/test_uca6c*.py` | 20 |
| T1-C `tests/unit/runtime/nexus/**` + `tests/unit/tools/**` | 4 (+ 0 tools) |
| T1-D remainder | 5 |

## 11. Test results

| Phase | Files / scope | Passed | Failed | Skipped | Result |
| ----- | ------------- | -----: | -----: | ------: | ------ |
| T1-A | 19 files | 122 | 0 | 1 | PASS |
| T1-B | 20 files | 114 | 0 | 0 | PASS |
| T1-C | 4 files | 26 | 0 | 0 | PASS |
| T1-D | 5 files | 29 | 0 | 0 | PASS |
| T2 | — | — | — | — | **NO TRACKED `test_uca6c*` INTEGRATION FILES** |
| T3 | 4 gate modules | 49 | 0 | 0 | PASS |
| T4 | 4 SQLite modules | 18 | 0 | 0 | PASS |
| T5-a | `test_runtime_import_boundary.py` | 5 | 0 | 0 | PASS |
| T5-b | `tests/unit/tools/registry/` | 25 | 0 | 0 | PASS |
| T6 | `test_persistence_provider_binding.py` | 24 | 0 | 0 | PASS |
| T8 | 2 fresh `uv run python -c` imports | 2 | 0 | 0 | PASS |

**Skip classification (T1-A):** PostgreSQL Autonomous Work store unavailable — optional backend; not sole evidence for any critical invariant (§31).

**Repro commands (sequential, from repo root):**

```powershell
$t1a = git ls-files ":(glob)tests/unit/autonomous_work/**/test_uca6c*.py"
uv run pytest $t1a -q
# … analogous lists for T1-B, T1-C, T1-D per §18
```

Session logs: `.tmp/session/uca-6c-r6-cert-rerun/T1-A.log` … `T6.log`, `T3.log`, `T4.log`, `T5-a.log`, `T5-b.log`.

## 12. Former blocker closure

| Blocker | Status | Evidence |
| ------- | ------ | -------- |
| B1 Clock / stale datetime patch | **CLOSED** | `test_uca6c_r6_r5_9_r2_r1_r2_caller_held_resume_authority.py` in **T1-B** (PASS) |
| B2 ToolRegistry circular import | **CLOSED** | T5 + T8; T1-C Nexus UCA tests collect+PASS |
| B3 SQLite/CW typing regression | **CLOSED** | T4 + T6; CW binder contracts |
| B4 Permissive SQLite path coercion | **CLOSED** | T4; no `Path(str(` in sqlite provider |
| B5 Weak Tool Registry denylist | **CLOSED** | `test_runtime_import_boundary.py` + full `tests/unit/tools/registry/` |

## 13. Static quality

**Ruff** (`ruff check` + `ruff format --check` on §14 surfaces only):

| Observation | Classification |
| ----------- | -------------- |
| E402 late imports in `worker_recovery_governed_fulfillment_composition.py` (intentional deferred import block) | NON-BLOCKING OBSERVATION |
| Format drift would affect 3 certified-surface files (no semantic change) | NON-BLOCKING OBSERVATION |

**Pyright** (same §14 file set):

| Observation | Classification |
| ----------- | -------------- |
| Protocol stub bodies in `collaborative_work/materialization_factory.py` / `persistence_provider.py` (`...` ellipsis — reportReturnType) | NON-BLOCKING OBSERVATION |
| Internal sqlite `bundle.py` client protocol variance (`Path` vs `str`) | NON-BLOCKING OBSERVATION |
| `RootExecutionLaunchRequest` generic arity in `qualified_capability_execution_dispatch_service.py` | NON-BLOCKING OBSERVATION |

No static finding classified as certified-seam **BLOCKER** (behavior proven by T3/T4/T6 gates).

## 14. Findings

### BLOCKER

NONE

### NON-BLOCKING OBSERVATION

- Ruff E402/format on deferred-import composition module and minor format drift on two other §14 files (pre-existing on baseline).
- Targeted Pyright reportReturnType/reportArgumentType on CW protocol stubs and sqlite internal typing (pre-existing; gates green).

### OUT-OF-SCOPE PARALLEL CHANGE

NONE (no commits between certified baseline and audit HEAD).

## 15. Exit criteria EC-01–EC-36

| ID | Result |
| -- | ------ |
| EC-01 Baseline drift clean | PASS |
| EC-02 Single Discovery owner | PASS |
| EC-03 Acquisition coordination-only | PASS |
| EC-04 Qualification boundary | PASS |
| EC-05 Binding != Execution | PASS |
| EC-06 Single EE lifecycle owner | PASS |
| EC-07 Single ExecutionIdentityAuthority | PASS |
| EC-08 Worker recovery != execution resume | PASS |
| EC-09 No AW/UCA pre-approval | PASS |
| EC-10 Agent / Declarative / MSE separate | PASS |
| EC-11 No universal approval grant | PASS |
| EC-12 Single ExecutionContinuationPort | PASS |
| EC-13 Single SuspendedExecutionOperationStore | PASS |
| EC-14 ToolRuntime mandatory | PASS |
| EC-15 Nexus internal | PASS |
| EC-16 Durable restart proof | PASS |
| EC-17 Multi-host/fencing proof | PASS |
| EC-18 Crash-window proof | PASS |
| EC-19 Canonical idempotency proof | PASS |
| EC-20 SQLite runtime-composition boundary | PASS |
| EC-21 Strong public semantic typing | PASS |
| EC-22 No reflection bypass on certified seams | PASS |
| EC-23 Contract-first pluginability | PASS |
| EC-24 Mandatory test matrix | PASS |
| EC-25 Targeted static quality acceptable | PASS |
| EC-26 Production diff during CERT = 0 | PASS |
| EC-27 Test diff during CERT = 0 | PASS |
| EC-28 Reproducible evidence | PASS |
| EC-29 Tool Registry lightweight invariant | PASS |
| EC-30 No wildcard/non-lightweight root imports | PASS |
| EC-31 SQLite factory/materializer CW contracts | PASS |
| EC-32 Invalid SQLite path fail-closed | PASS |
| EC-33 No ToolRegistry/Nexus collection blocker | PASS |
| EC-34 No clock test blocker | PASS |
| EC-35 Sole tracked CERT change is this artifact | PASS (pending commit) |
| EC-36 Final status eligible for freeze track | PASS |

## 16. Final verdict

```text
UCA-6C R6 ENTERPRISE CERTIFICATION = PASS
R6-FREEZE ELIGIBILITY = YES
ARCHITECTURE FROZEN = NO
BLOCKER COUNT = 0
```

Architecture checkpoint (§29): all items affirmed **YES** with bounded evidence above; none **UNKNOWN**.
