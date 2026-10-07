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
| P5-R1-R1-Q6 START_HEAD | `4aba36e50a474a0d6915a60f73f5a3b31e6f71c2` |
| P5-R1-R1-Q7 START_HEAD | `5afc518a78fe5777d5f67934c1ecbd1ec6740e0a` |
| P5-R1-R1-Q8 START_HEAD | `4bd430ae660dbc9f826c3b5ede0b92aefea36b07` |
| Q2 FINAL_COMMIT | _(placeholder until qualification commit)_ |
| Q3 FINAL_COMMIT | `0146f6a862339e5c6eefbf53b5622bf3e8491db5` |
| Q4 FINAL_COMMIT | `41c36b4ad8983fd0c90a4c9513d28de124a07538` |
| Q5 FINAL_COMMIT | `a6e04f2c1959674c477d591b5a9d8012d16eb80a` |
| Q6 FINAL_COMMIT | `41b158ccd2a80453269a7026d3d8e12a11c0088d` |
| Q6 implementation SHA | `41b158ccd2a80453269a7026d3d8e12a11c0088d` |
| Q7 FINAL_COMMIT | `306ec1d9661a9d979028b34ac23b7ac1fd98ecd1` |
| Q7 implementation SHA | `5bb6699948b7281e95e03cc3521c568db7899ab5` |
| Q8 FINAL_COMMIT | `8b9cdbcd5281176bb19528e7c9789f2b2e7d4063` |
| Q8 implementation SHA | `8b9cdbcd5281176bb19528e7c9789f2b2e7d4063` |
| FINAL_COMMIT | `8b9cdbcd5281176bb19528e7c9789f2b2e7d4063` |
| TRACE-X-P5-R1-R1-Q1 | **READY FOR INDEPENDENT RE-AUDIT** (qualification remediation; not CLOSED) |
| TRACE-X-P5-R1-R1-Q2 | **READY FOR INDEPENDENT RE-AUDIT** (derived resume comparison; not CLOSED) |
| TRACE-X-P5-R1-R1-Q3 | **READY FOR INDEPENDENT RE-AUDIT** (scoped import-provenance closure; not CLOSED) |
| TRACE-X-P5-R1-R1-Q4 | **READY FOR INDEPENDENT RE-AUDIT** (canonical constructor alias-escape closure; not CLOSED) |
| TRACE-X-P5-R1-R1-Q5 | **READY FOR INDEPENDENT RE-AUDIT** (full-dotted canonical constructor reference closure; not CLOSED) |
| TRACE-X-P5-R1-R1-Q6 | **READY FOR INDEPENDENT RE-AUDIT** (canonical reference usage-context closure; not CLOSED) |
| TRACE-X-P5-R1-R1-Q7 | **READY FOR INDEPENDENT RE-AUDIT** (lexical binding authority closure; not CLOSED) |
| TRACE-X-P5-R1-R1-Q8 | **READY FOR AUDIT** (definition-time scope evaluation closure; not CLOSED) |
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

**Q6 blocker (START_HEAD `4aba36e5…`):** `Q5-CANONICAL-REFERENCE-CONTEXT-COVERAGE-01` — forbidden-constructor detection remained syntax-handler-driven (assign/return/yield/call arguments), so walrus bindings, function defaults, lambda defaults, boolean/conditional runtime use, comprehension/container propagation, attribute/subscript storage, and wrapper-call arguments could evade closed-world classification.

**Q6 remediation:** reference-centric pass with one authoritative `classify_canonical_constructor_usage` (`CanonicalConstructorUsageKind`: `DIRECT_CONSTRUCTOR_CALL`, `TYPE_ANNOTATION`, `IMPORT_BINDING`, `FORBIDDEN_RUNTIME_ESCAPE`, `UNKNOWN`); every provenance-resolved canonical reference is parent-context classified; direct callee allowance requires canonical root of `ast.Call` (optional generic subscript); annotation allowance is AST annotation-context only; `UNKNOWN` fails closed.

**Q6 qualification:** production inventory unchanged (`legal constructor surfaces = 4`, `forbidden canonical runtime-reference usages = 0`, `unknown usage contexts = 0`; discovery parity `discovered = classified`, `unknown surfaces = 0`, `orphan = 0`, `duplicate = 0`). Sentinels `test_txp5r1_q74`–`q86`; Q6 lineage `test_txp5r1_q87`.

Registry parity (Q6): gates `test_txp5r1_q12`–`q24`, `q31`–`q37`, `q45`–`q55`, `q56`–`q65`, `q66`–`q73`, `q74`–`q87`.

**Q7 blocker (START_HEAD `5afc518a…`):** `Q6-LEXICAL-BINDING-SHADOW-01` — import provenance treated module-level constructor names as canonical at every expression site; Python lexical binders (parameters, `for`/`with`/`except as`, walrus, nested `def`/`class`, pattern captures, destructuring) could shadow/rebind without updating effective provenance, yielding false-positive canonical constructor surfaces (e.g. parameter named `ChildExecutionRunner`).

**Q7 remediation:** one authoritative `lexical_bound_names` / `lexical_bound_names_from_target` / `lexical_bound_names_from_match_pattern` layer applied in statement order on the scope stack before Q6 `classify_canonical_constructor_usage`; canonical `from … import ChildExecutionRunner` restores provenance after shadow (`shadowed_constructor_names.discard`); unsupported binding shapes fail closed via `lexical_binding_violations`. Pipeline: import provenance → lexical binding resolution → canonical reference identification → `CanonicalConstructorUsageKind` → fail-closed usage policy.

**Q7 qualification:** production parity unchanged (`legal constructor surfaces = 4`, `classified registry surfaces = 4`, `unknown surfaces = 0`, `orphan = 0`, `duplicate = 0`, `forbidden canonical runtime usages = 0`, `unknown usage contexts = 0`, `canonical lexical-shadow violations = 0` in `intergrax/`). Production shadow inventory: none (gate `test_txp5r1_q100`). Sentinels `test_txp5r1_q88`–`q101`; Q7 lineage `test_txp5r1_q102`.

Registry parity (Q7): gates `test_txp5r1_q12`–`q24`, `q31`–`q37`, `q45`–`q55`, `q56`–`q65`, `q66`–`q73`, `q74`–`q87`, `q88`–`q102`.

**Q8 blocker (START_HEAD `4bd430ae…`):** `Q7-DEFINITION-TIME-SCOPE-01` — Q7 applied parameter lexical bindings before visiting function/lambda default expressions, so same-name defaults (e.g. `def f(ChildExecutionRunner=ChildExecutionRunner)`) incorrectly treated the RHS as parameter-shadowed instead of outer-scope canonical escape.

**Q8 remediation:** one authoritative callable pipeline — `_visit_callable_definition_time_expressions` (decorators + defaults in enclosing provenance) → `_enter_callable_lexical_scope` (parameter binds) → annotations/body under child provenance → `_exit_callable_lexical_scope`; shared for `FunctionDef`, `AsyncFunctionDef`, and `Lambda`. Parameter names do not affect default RHS classification; body retains Q7 shadow semantics; local canonical import after parameter default shadow unchanged.

**Q8 qualification:** production parity unchanged (`legal constructor surfaces = 4`, `classified registry surfaces = 4`, `unknown surfaces = 0`, `orphan = 0`, `duplicate = 0`, `forbidden canonical runtime usages = 0`, `unknown usage contexts = 0`, `canonical lexical-shadow violations = 0` in `intergrax/`). Sentinels `test_txp5r1_q103`–`q109`; Q8 lineage `test_txp5r1_q110`.

Registry parity (Q8): gates `test_txp5r1_q12`–`q24`, `q31`–`q37`, `q45`–`q55`, `q56`–`q65`, `q66`–`q73`, `q74`–`q87`, `q88`–`q110`.

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
