# TRACE-X-P5-R1 — Policy & Effective Profile Execution Attribution (candidate certification)

## Revision record

| Field | Value |
|---|---|
| R1 audited SHA | `65f7e1ef4832d99a19be7953734a42d5bd5cbb4f` |
| R1-R1 implementation SHA | `a452de39a721cd357be3ba5ecd0c3a6d41b630bd` |
| P5-R1 START_HEAD | `98c0d9d7ae9763bce931c60e19b6a91af3a2f4e9` |
| P5-R1-R1 START_HEAD | `65f7e1ef4832d99a19be7953734a42d5bd5cbb4f` |
| P5-R1-R1-Q1 START_HEAD | `a452de39a721cd357be3ba5ecd0c3a6d41b630bd` |
| P5-R1-R1-Q2 START_HEAD | `538af9ef51a6ca483f607988481ef2794deb87b9` |
| P5-R1-R1-Q3 START_HEAD | `c51d04b7849d44d1adee78ad6d4e7eedd8b5ab68` |
| P5-R1-R1-Q4 START_HEAD | `ea1e4947fe66c120bb502885dc912e5f98698339` |
| P5-R1-R1-Q5 START_HEAD | `274c5ff40e0c4f30764d13be65a00e3b463ec5e4` |
| Q2 FINAL_COMMIT | _(placeholder until qualification commit)_ |
| Q3 FINAL_COMMIT | `0146f6a862339e5c6eefbf53b5622bf3e8491db5` |
| Q4 FINAL_COMMIT | `41c36b4ad8983fd0c90a4c9513d28de124a07538` |
| Q5 FINAL_COMMIT | `a6e04f2c1959674c477d591b5a9d8012d16eb80a` |
| FINAL_COMMIT | _(see git push output)_ |
| TRACE-X-P5-R1-R1-Q1 | **READY FOR INDEPENDENT RE-AUDIT** (qualification remediation; not CLOSED) |
| TRACE-X-P5-R1-R1-Q2 | **READY FOR INDEPENDENT RE-AUDIT** (derived resume comparison; not CLOSED) |
| TRACE-X-P5-R1-R1-Q3 | **READY FOR INDEPENDENT RE-AUDIT** (scoped import-provenance closure; not CLOSED) |
| TRACE-X-P5-R1-R1-Q4 | **READY FOR INDEPENDENT RE-AUDIT** (canonical constructor alias-escape closure; not CLOSED) |
| TRACE-X-P5-R1-R1-Q5 | **READY FOR AUDIT** (full-dotted canonical constructor reference closure; not CLOSED) |
| TRACE-X-P5-R1-R1 | **READY FOR INDEPENDENT RE-AUDIT** (not CLOSED) |
| TRACE-X-P5-R1 | **READY FOR INDEPENDENT RE-AUDIT** (not CLOSED) |

## Independent audit findings (R1 @ `65f7e1ef…`) — R1-R1 remediation

| ID | Resolution |
|---|---|
| P5-R1-POLICY-CANONICAL-READ-01 | `policy_provenance_projection` uses typed envelope via `validate_payload_envelope` when `payload_schema_id` is present; legacy path only when absent. E2E: governance fact → `ValidatingEvidencePersistencePort` → reconstruction. |
| P5-R1-CHILD-PROFILE-PINNING-02 | Neutral `ChildExecutionContextInheritancePort` + `ProfileResolutionChildContextInheritanceAdapter` wired through `ChildExecutionRunner` → `GraphExecutor` → `NexusLoop` → profile-aware host/scenario spec. |
| P5-R1-REVISION-REF-OWNERSHIP-03 | `EffectiveProfileRevisionProvenanceRef` is opaque (non-empty str only); no `effprof_rev_` / suffix grammar in neutral contract. |
| P5-R1-RESUME-BASELINE-04 | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED / PRE-EXISTING** — committed machine evidence `docs/project/maintainers/qualification/TRACE_X_P5_R1_R1_Q1_RESUME_BASELINE_EVIDENCE.json` (baseline `98c0d9d7…`, Q1 head `a452de39…`, conclusion `PRE_EXISTING_NON_R1_REGRESSION`). Owner: Profile Resolution / checkpoint-resume qualification. |

## Child execution inventory (scope-aware import-provenance AST discovery @ Q3)

**Q3 blocker (START_HEAD `c51d04b7…`):** discovery collected import provenance only from module-level `tree.body`, so constructor calls after function-local / nested-scope imports could escape closed-world inventory.

**Q3 remediation:** lexical traversal applies `visit_Import` / `visit_ImportFrom` in statement order per scope stack (module, function, async function, nested function); sibling scopes do not leak bindings; nested functions inherit enclosing provenance; local reassignment/shadowing of canonical constructor names fails closed; class-body canonical imports are mechanically forbidden (methods do not receive class-body aliases as unqualified locals); unaliased `import intergrax.runtime.execution.child` binds `intergrax` and resolves `intergrax.runtime.execution.child.ChildExecutionRunner()` only (not bogus `child.ChildExecutionRunner()`).

Supported forms: direct `from … child import ChildExecutionRunner`, `as` alias, `import … child as …`, `from intergrax.runtime.execution import child`, and full dotted attribute call from `intergrax` after unaliased child-module import.

**Q3 qualification:** production discovery parity unchanged (`discovered = 4`, `unknown = 0`, `orphan = 0`, `duplicate = 0`, rebind/shadow/class-body violations = 0 in `intergrax/`). Sentinels `test_txp5r1_q45`–`q54`; Q3 lineage `test_txp5r1_q55`. Q3 implementation evidence SHA `0146f6a862339e5c6eefbf53b5622bf3e8491db5` (not superseded by Q4 bookkeeping).

**Q4 blocker (START_HEAD `ea1e4947…`):** `Q3-CANONICAL-CONSTRUCTOR-ALIAS-ESCAPE-01` — compound assignment targets and RHS containers/conditionals could propagate canonical `ChildExecutionRunner` without simple `ast.Name` rebind detection.

**Q4 remediation:** fail closed on any canonical constructor reference outside sanctioned direct constructor calls (including generic subscript callee); recursive assignment-target unpacking (`Name`, `Tuple`, `List`, `Starred`); RHS alias-escape scan for containers, conditionals, returns, yields, and non-constructor call arguments; type annotations and parameter annotations remain excluded from escape enforcement.

**Q4 qualification:** production parity unchanged (`discovered = 4`, `classified = 4`, `unknown = 0`, `orphan = 0`, `duplicate = 0`, canonical constructor alias escapes = 0, rebind/shadow/class-body violations = 0). Production non-call inventory: import bindings, direct/generic constructor calls, and typed parameter annotations only (`delegated_subtask_child_port.py`). Sentinels `test_txp5r1_q56`–`q64`; Q4 lineage `test_txp5r1_q65`.

Registry parity: `tests/qualification/trace_x/_trace_x_p5_r1_child_discovery.py`, `tests/qualification/trace_x/_trace_x_p5_r1_child_registry.py`, gates `test_txp5r1_q12`–`q24`, `q31`–`q37`, `q45`–`q55`, `q56`–`q65`.

**Q5 blocker (START_HEAD `274c5ff4…`):** `Q4-DOTTED-CONSTRUCTOR-ESCAPE-01` — direct-call discovery recognized full dotted `intergrax.runtime.execution.child.ChildExecutionRunner` via `_dotted_expr_canonical_constructor`, but alias-escape used `_expr_is_canonical_constructor_ref` limited to `ast.Name` and single-level module-alias attributes, so full-dotted assignment/container/return/argument escapes could evade qualification.

**Q5 remediation:** one authoritative `_expr_is_canonical_constructor_ref` (including generic subscript roots) shared with `_is_canonical_constructor_call`; full dotted paths require provenance `intergrax_roots` (suffix-only paths such as `other.runtime.execution.child.ChildExecutionRunner` remain false).

**Q5 qualification:** production parity unchanged (`discovered = 4`, `classified = 4`, `unknown = 0`, `orphan = 0`, `duplicate = 0`, alias escapes = 0, rebind/shadow/class-body violations = 0). Sentinels `test_txp5r1_q66`–`q72`; Q5 lineage `test_txp5r1_q73`. Q5 implementation evidence SHA `a6e04f2c1959674c477d591b5a9d8012d16eb80a`.

Registry parity (Q5): gates `test_txp5r1_q12`–`q24`, `q31`–`q37`, `q45`–`q55`, `q56`–`q65`, `q66`–`q73`.

## Resume baseline evidence (@ Q2)

Committed artifact `TRACE_X_P5_R1_R1_Q1_RESUME_BASELINE_EVIDENCE.json` schema `trace_x_p5_r1_r1_q2_resume_baseline_v2`: structured `failures[]` per test node; comparison fields are derived from baseline/current and must match the serialized `comparison` projection (gates `test_txp5r1_q26`, `q38`–`q44`). R1-R1 implementation comparison SHA remains `a452de39…`; baseline SHA remains `98c0d9d7…`.

| Surface key | Classification | Inheritance |
|---|---|---|
| `intergrax/runtime/nexus/execution/graph_executor.py::GraphExecutor.__init__` | PROFILE_AWARE_CAPABLE_PRODUCTION | forwards `child_context_inheritance` |
| `intergrax/runtime/execution/execution_work_port.py::ChildExecutionWorkPort.__init__` | GENERIC_PRODUCTION | optional injection |
| `intergrax/runtime/execution/execution_work_port.py::DelegatedSubtaskChildExecutionWorkPort.__init__` | GENERIC_PRODUCTION | optional injection |
| `intergrax/runtime/execution/execution_work_port.py::DelegatedProviderChildExecutionEngine.__init__` | GENERIC_PRODUCTION | optional injection |

Profile-aware `wire_host_effective_profile_execution` root: `scenario_runtime_baseline.py::build_scenario_runtime_from_environment` (harness uses direct adapter wiring; gate `test_txp5r1_q09`).

`ProductionAgentCapabilityRuntime` → `build_production_delegated_subtask_child_execution_port` → `DelegatedSubtaskChildExecutionWorkPort` → `ChildExecutionRunner`: **GENERIC_PRODUCTION** (no effective-profile host ownership; drift watch `test_txp5r1_q22`).

Child identity owner unchanged: `ChildExecutionRunner` → `default_execution_identity_authority.mint_child_execution_identity()` (gate `test_txp5r1_q24`).

## Scope delivered

- Neutral read-only contracts: `ExecutionEffectiveProfileProvenance`, `ExecutionEffectiveProfileProvenanceReader`, `EffectiveProfileRevisionProvenanceRef`.
- Neutral child context seam: `ChildExecutionContextInheritancePort` / `ChildExecutionContextInheritanceRequest`.
- Profile Resolution adapter: `PinningStoreExecutionEffectiveProfileProvenanceReader` → `EffectiveProfileExecutionPinningStore.get` only.
- `ExecutionReconstructor` typed `policy_decision_provenance` + per-`ExecutionId` `execution_effective_profile_provenance` with explicit `effective_profile_provenance_read_status`.
- Mandatory `revision_admission` on `build_environment_host_task_execution` and `compose_application_host_orchestration_session`.
- Scenario runtime profile-aware wiring via `wire_host_effective_profile_execution`.
- Diagnostic composition optional profile provenance reader injection (harness + scenario).
- Mechanical gates: `tests/qualification/trace_x/test_trace_x_p5_r1_qualification_gates.py`.

## Ownership matrix

| Concern | Semantic owner | Contract | Implementation | Composition |
|---|---|---|---|---|
| Policy permission | Governance | `PolicyDecisionSpinePayloadV1` / GEP | Governance persistence | Orchestration spine |
| Policy reconstruction | Evidence Plane | `ReconstructedPolicyDecisionProvenance` | `policy_provenance_projection` | `ExecutionReconstructor` |
| Profile revision truth | Profile Resolution | `EffectiveProfileExecutionPinningStore` | Application profile_resolution | Host admission |
| Profile reconstruction read | Evidence Plane (projection) | `ExecutionEffectiveProfileProvenanceReader` | Pinning store adapter | Diagnostic / reconstructor wiring |

## FRZ disposition (no self-closure)

| Criterion | Evidence produced | Remaining | Proposed disposition |
|---|---|---|---|
| FRZ-TRC-07 | Typed policy fields in `ExecutionReconstruction` | Independent audit of all production reconstruction entrypoints | **CANDIDATE / OPEN** |
| FRZ-TRC-08 | Mandatory admission + neutral reader + fail-closed reconstruction | Scenario/harness durability audit at scale | **CANDIDATE / OPEN** |
| FRZ-TRC-11 | — | Configured→effective provenance (P5-R2) | **OPEN** |
| FRZ-TEN-02/05/07 | Tenant-scoped reader lookup + negative tests | Full TENANT-X program | **OPEN** |

## Tenant isolation audit (roadmap §2.0.1)

**Verdict: PASS (candidate)** — reconstruction tenant, runtime event tenant, and profile reader tenant are aligned; cross-tenant binding lookup fails closed when reader is configured.

## Unresolved findings

| ID | Class |
|---|---|
| P5-R2 configured→effective join | TRACKED FREEZE DEBT (P5-R2 / FRZ-TRC-11) |
| Resume adoption tests (`test_missing_binding_on_resume_fails_closed`, `test_resume_preserves_pinned_revision_not_current_host_revision`) | ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED / PRE-EXISTING |

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
