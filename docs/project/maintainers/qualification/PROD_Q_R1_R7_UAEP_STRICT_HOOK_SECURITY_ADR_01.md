# PROD-Q-R1-R7 — UAEP Strict Hook Security (Gate B1 + B2)

| Field | Value |
|-------|-------|
| **Status** | **ACCEPTED — OPERATOR RATIFIED A+E** |
| **Revision** | **R0** |
| **Decision date** | **2026-10-17** |
| **Decision** | **B1 = Alternative A** · **B2 = Alternative E** |
| **Approved implementation** | **IMPL-01** (`PROD-Q-R1-R7-UAEP-STRICT-HOOK-SECURITY-IMPL-01`) |
| **Ratification authority** | Explicit operator decision (`ACCEPT A+E`) |
| **Evidence baseline (ratification)** | `84c1e550fd847ba9e19bcc87742ca086e888bf2a` |
| **Branch** | `development` |
| **START_HEAD / evidence inventory SHA** | `be21ef1a9016d21f67ecc2af53fcff8555606cde` (historical proposal inventory) |
| **Parent program** | **PROD-Q-R1-R7** — **NOT CLOSED** |
| **Related stage** | **PROD-Q-R1-R7-D3** — durable projection/restart **READY FOR AUDIT**; strict UAEP security path **NOT QUALIFIED** |
| **Qualification predecessor** | [`PROD_Q_R1_R7_D3_DURABLE_PROJECTION_RESTART_RECONCILIATION.md`](PROD_Q_R1_R7_D3_DURABLE_PROJECTION_RESTART_RECONCILIATION.md) § D3-R1 UAEP event and hook boundary (**Gate B FAIL**) |
| **Baseline middleware ADR** | **ADR-CTRL-X-001** (Accepted) — typed middleware hook ownership; this ADR **refines producer obligations** for UAEP `BEFORE_CONTEXT_BUILD`; does **not** reopen HookPoint / pipeline / Tier-0 payload union ownership |
| **Does not close** | **PROD-Q-R1-R7** · **PROD-Q-R1-R7-D3** (production security slice) · **PROD-Q-R1-R7-Q** · **PROD-Q** · any global **FRZ-*** PASS |
| **Production delta (IMPL-01)** | **B1 + B2** — UAEP typed hook payload; hook violation catalog phase (see §23) |
| **Historical proposal status** | **PROPOSED — ARCHITECTURE DECISION REQUIRED** (§1–§22 preserved) |

**Competing ADR check:** No accepted ADR already resolves UAEP `BEFORE_CONTEXT_BUILD` payload vs `PromptDefenseMiddleware` or `hook_runtime_guard` platform-signal phase override. **ADR-CTRL-X-001** mandates typed payloads at hook boundaries but does not specify UAEP reflex producer shape. **No conflict — proceed as bounded child ADR.**

---

## 1. Decision context

Under strict product defaults (`prompt_defense_enabled=True`), UAEP reflex execution invokes `BEFORE_CONTEXT_BUILD` **before** `agent.build_context(request)` while supplying hook input that does not satisfy the existing Prompt Defense typed contract. When middleware returns `BLOCK`, `hook_runtime_guard` attempts to publish `platform.hook.hook_blocked` with `phase=ExecutionPhase.CONTEXT_BUILDING`, which **overrides** the platform kind catalog phase (`STEP_EXECUTION`) and fails durable runtime event schema validation.

Canonical lifecycle proof temporarily **disables** prompt defense for stage-15 UAEP only (`testing_support/canonical_agent_lifecycle_composition.py`). That is harness debt, not production equivalence.

**Mandatory gate:** **STOP — ARCHITECTURE DECISION REQUIRED** until operator ratifies this ADR. **Proposal ≠ permission ≠ execution.**

---

## 2. Current HEAD and evidence SHAs

| Artifact | SHA / pointer |
|----------|----------------|
| **Inventory / ADR authoring HEAD** | `be21ef1a9016d21f67ecc2af53fcff8555606cde` |
| **D3-R1 UAEP hook boundary qualification** | `7400c345127a54b03b411de6044d290ad70d911a` (`AUDITED_HEAD` in D3 reconciliation) |
| **D3-R1 pytest evidence** | 10 passed — log `.tmp/session/PROD-Q-R1-R7-D3-R1-UAEP-EVENT-AND-HOOK-BOUNDARY-QUALIFICATION/pytest-final.log` |
| **Roadmap rollup** | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) — **PROD-Q-R1-R7** **NOT CLOSED** / **BLOCKED** |
| **Freeze checklist** | [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) — scoped FRZ rows **OPEN** |

---

## 3. Precise defects

### 3.1 Gate B1 — UAEP `BEFORE_CONTEXT_BUILD` vs Prompt Defense contract

**Observed (@ inventory SHA):**

- `UaepExecutor` builds `HookContext` with `phase=CONTEXT_BUILDING`, `runtime_state={"prompt": request.message, ...}`, and **default** `payload=EmptyMiddlewareHookPayload` (no `LlmInferenceHookPayload`).
- `PromptDefenseMiddleware.before` accepts only `HookPoint.BEFORE_CONTEXT_BUILD` and requires `isinstance(payload, LlmInferenceHookPayload)`; otherwise **fail-closed `BLOCK`** (`application_security_wiring.py`).
- Graph reference producer `ContextManager` sets `payload=LlmInferenceHookPayload(prompt=task.message)` at the same hook point (`context_manager.py`).
- `ApplicationGuardrailMiddleware` applies the same `LlmInferenceHookPayload` requirement for input scan at `BEFORE_CONTEXT_BUILD` / `BEFORE_LLM_INFERENCE`.

**Security meaning gap:** UAEP carries the pre-build user/task text in `runtime_state["prompt"]` but not in the Tier-0 typed field Prompt Defense reads. Middleware does **not** fall back to `runtime_state` (by design — ADR-CTRL-X-001 typed ABI).

**Hook timing:** UAEP runs the hook **before** `build_context`; inspectable text at this point is the **intake message** (`request.message`), not yet retrieved/engine-assembled context. Graph engine path is equivalent at the same hook ordering (hook before engine assembly).

### 3.2 Gate B2 — `hook_runtime_guard` vs platform event catalog phase

**Observed:**

- `_emit_hook_violation` calls `build_platform_signal_event(..., phase=ctx.phase, ...)`.
- `build_platform_signal_event` sets `phase=phase or entry.phase` (`spine_consolidation.py`).
- Catalog assigns `platform.hook.hook_blocked` (and `hook_error`, `hook_timeout`) → `ExecutionPhase.STEP_EXECUTION`.
- When `ctx.phase` is `CONTEXT_BUILDING`, emitted `DOMAIN_SIGNAL` fails `assert_runtime_event_schema` with phase mismatch (`test_hook_runtime_guard.py::test_hook_violation_domain_signal_with_context_building_phase_fails_schema_guard`).

**Effect:** Execution is still **denied** when middleware blocks, but **security/ops telemetry may not persist** on validating bus/store paths — observability must not become permission, and denial must not depend on telemetry success.

---

## 4. Closed-world owner / contract / caller inventory

| Responsibility | Semantic owner | Sanctioned composition owner | Key artifacts |
|----------------|----------------|------------------------------|---------------|
| Hook payload types | Platform Tier-0 contracts | — | `intergrax/contracts/middleware_hook_semantics.py` |
| UAEP `BEFORE_CONTEXT_BUILD` producer | UAEP executor (runtime nexus) | `UaepExecutor.execute` | `intergrax/runtime/nexus/uaep/uaep_executor.py` |
| Graph `BEFORE_CONTEXT_BUILD` producer | Context manager | `ContextManager` engine path | `intergrax/runtime/nexus/context/context_manager.py` |
| Prompt injection decision | `inspect_prompt_for_injection` + profile | `PromptDefenseMiddleware` | `intergrax/applications/_shared/application_security_wiring.py` |
| Middleware orchestration | Pipeline ordering / timeouts | `MiddlewarePipeline` + `invoke_guarded_hook` | `intergrax/runtime/middleware/` |
| Platform signal kind + phase | Platform kind catalog | `PLATFORM_KIND_CATALOG` | `intergrax/runtime/events/spine_consolidation.py` |
| Event schema enforcement | Runtime event guard | `assert_runtime_event_schema` / validating store | durable event write boundary |
| Security evidence recording | Observability / event bus | **Non-authoritative** — must not grant execution | `RuntimeEventBus`, durable store |
| Application security profile | Application environment contract | `register_application_security_hooks` | `ApplicationSecurityProfile.prompt_defense_enabled` |

**Forbidden:** second Prompt Defense implementation; untyped `runtime_state` consumption by security middleware; UAEP exemption; downstream widening of middleware authority.

---

## 5. Before dependency graph

```text
[UAEP reflex execute]
  UaepExecutor.execute
    → HookContext(phase=CONTEXT_BUILDING, runtime_state.prompt=request.message, payload=Empty*)
    → MiddlewarePipeline.run_before(BEFORE_CONTEXT_BUILD)
         → PromptDefenseMiddleware → BLOCK (wrong payload type)
         → invoke_guarded_hook → hook_runtime_guard._emit_hook_violation(phase=ctx.phase)
              → build_platform_signal_event(hook_blocked, phase=CONTEXT_BUILDING)
              → assert_runtime_event_schema → FAIL (expected STEP_EXECUTION)
    → (if reached) agent.build_context(request)

[Graph context engine path — reference]
  ContextManager
    → HookContext(payload=LlmInferenceHookPayload(prompt=task.message), phase=CONTEXT_BUILDING)
    → MiddlewarePipeline.run_before(BEFORE_CONTEXT_BUILD)
         → PromptDefenseMiddleware → inspect prompt or ALLOW
```

---

## 6. Gate B1 — alternatives A–D

| Alt | Summary | Security / compatibility |
|-----|---------|---------------------------|
| **A** | UAEP sets `LlmInferenceHookPayload(prompt=request.message)` (and `subject` parity with graph where applicable) | **Preferred** if pre-build inspectable text equals graph `task.message` semantics — **proven @ inventory SHA** for intake message before `build_context`. Reuses existing Prompt Defense + guardrail contract. No middleware widening. |
| **B** | Extend `PromptDefenseMiddleware` (and guardrails) to accept `ContextBuildHookPayload.message` | Smaller producer change if message field were authoritative; **requires middleware contract expansion** and explicit policy for `capability` field (not scanned today). Duplicates overlapping fields with `LlmInferenceHookPayload`. |
| **C** | New shared `ContextInspectionHookPayload` in Tier-0 | Only if A/B cannot represent protected content truthfully; **largest contract churn**; requires ownership assignment and consumer updates. |
| **D** | Exempt UAEP reflex from Prompt Defense | **Rejected by default** — weakens strict product semantics; requires explicit security-policy exception. **Not recommended.** |

### B1 analysis (closed-world answers)

1. **Trusted context about to be built:** intake task message driving context assembly (`request.message` / `task.message`).
2. **Origin:** user/task intake on the runtime request — not yet RAG/engine output at this hook instant on UAEP path.
3. **Content class:** user/task input (not mixed retrieved bundle at hook time).
4. **Available before protected operation:** yes — in UAEP `runtime_state["prompt"]` today; not in typed `payload`.
5. **Authoritative representation for middleware:** Tier-0 `HookContext.payload` per ADR-CTRL-X-001 — **not** `runtime_state`.
6. **`LlmInferenceHookPayload.prompt` accuracy:** **yes** for the pre-build intake message when set from `request.message` (matches graph).
7. **Skip risk if using prompt only:** retrieved/hostile **post-retrieval** instructions embedded later in assembled context are **out of scope** for `BEFORE_CONTEXT_BUILD` on both paths today; do not conflate with this gate.
8. **Graph equivalence:** **yes** for intake message at hook invocation ordering.
9. **`ContextBuildHookPayload`:** can carry `message` but is **not** the contract Prompt Defense implements; would require Alt B.
10. **Middleware expansion:** Alt A requires **none**; B/C require explicit ADR-approved widening.

---

## 7. Gate B2 — alternatives E–G

| Alt | Summary | Notes |
|-----|---------|-------|
| **E** | Emit platform hook violation signals with **catalog canonical phase** (`STEP_EXECUTION`); omit `phase=ctx.phase` override | Preserves catalog authority; hook point + `payload` diagnostics retain originating context. **Preferred.** |
| **F** | Extend phase catalog to allow `CONTEXT_BUILDING` for hook violation kinds | Requires schema-policy change, producer/consumer inventory, and justification — **not needed** if E satisfies observability. |
| **G** | New distinct event kind | **Reject** unless E proves semantically insufficient — duplicates signal mechanism. |

**Diagnostic source-phase attribution (E):** retain `ctx.phase` in signal **payload** (e.g. `source_execution_phase` or existing fields `point` + hook metadata) without setting top-level `RuntimeEvent.phase` off-catalog.

---

## 8. Security and compatibility trade-offs

| Topic | A + E | B | D |
|-------|-------|---|---|
| Strict product prompt defense on UAEP | Restores parity with graph | Parity with broader payload | **Breaks parity** |
| Tier-0 contract stability | **High** | Medium (middleware change) | High but unsafe |
| Event catalog stability | **High** | High | High |
| Harness stage-15 proof | Can re-enable `prompt_defense_enabled` after impl | Same | Leaves security disabled |
| Retrieved-content injection | Unchanged limitation on both paths at this hook | Same | N/A |

---

## 9. Recommended combination

**Ratify: B1 = Alternative A** + **B2 = Alternative E**.

- UAEP producer aligns with graph: truthful `LlmInferenceHookPayload` at `BEFORE_CONTEXT_BUILD`.
- Hook violation publisher uses catalog phase for `platform.hook.hook_*` signals; optional payload field records originating `ctx.phase`.

**Reject:** D (exemption). **Defer:** B unless implementation discovers intake message cannot be represented in `LlmInferenceHookPayload.prompt` without omitting security-relevant pre-build text (not shown @ inventory SHA).

---

## 10. Expected solution shape (post-ratification implementation)

1. **`uaep_executor.py`:** Before `run_before(BEFORE_CONTEXT_BUILD)`, set `payload=LlmInferenceHookPayload(prompt=request.message or None)` and `subject` facet consistent with graph (tenant from canonical runtime request).
2. **`hook_runtime_guard.py`:** Stop passing `phase=ctx.phase` into `build_platform_signal_event` for platform hook violations; use catalog default. Add non-authoritative `source_execution_phase` (or equivalent) into event **payload** when `ctx.phase` differs from catalog phase.
3. **Harness:** Remove stage-15-only `prompt_defense_enabled=False` workaround once qualification passes.
4. **Tests:** Extend UAEP + lifecycle tests per §14 and §15 below.
5. **No change** to `PromptDefenseMiddleware` decision logic unless operator later ratifies Alt B.

---

## 11. Files authorized for future implementation

| File | Allowed change |
|------|----------------|
| `intergrax/runtime/nexus/uaep/uaep_executor.py` | Typed hook payload + subject at `BEFORE_CONTEXT_BUILD` |
| `intergrax/runtime/middleware/hook_runtime_guard.py` | Canonical platform signal phase + diagnostic payload |
| `testing_support/canonical_agent_lifecycle_composition.py` | Re-enable prompt defense when qualified |
| `tests/unit/runtime/nexus/test_uaep_context_assembled_event_contract.py` | UAEP hook payload contract |
| `tests/unit/applications/test_application_security_wiring.py` | UAEP-shaped contexts with typed payload |
| `tests/unit/runtime/middleware/test_hook_runtime_guard.py` | Phase + publish success under schema guard |
| `tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py` | Strict prompt defense on UAEP path |
| `tests/unit/applications/test_prod_q_r1_r7_d3_*` | Regression-only if touched by shared fixtures |
| New qualification module (optional) | `tests/qualification/prod_q/test_prod_q_r1_r7_uaep_strict_hook_security.py` |

---

## 12. Explicitly forbidden modifications

- Disabling or bypassing `PromptDefenseMiddleware` for UAEP reflex agents in production composition.
- Reading `runtime_state` in security middleware as a substitute for typed payload.
- Changing `PLATFORM_KIND_CATALOG` hook phases without a separate ratified schema-policy ADR (Alt F).
- Adding a second prompt-injection engine or duplicate middleware with divergent rules.
- Treating observability publish failure as `ALLOW`.
- Inventing retry/compensation that re-admits execution after `BLOCK` without existing sanctioned recovery owners.
- Marking this ADR **ACCEPTED** or global **FRZ-*** **PASS** in implementation commits.

---

## 13. Threat model (summary)

**Assets:** tenant-scoped execution integrity; pre-context user/task text; security audit trail.

**Adversaries:** prompt injection via task message; hostile retrieved content (bounded scope note); middleware confusion via malformed hook input.

**Trust boundaries:** Tier-0 payload at middleware boundary; platform catalog at event emission; Prompt Defense policy in application security profile.

**In-scope controls:** `BEFORE_CONTEXT_BUILD` typed payload + Prompt Defense + hook violation telemetry with valid schema.

**Known limitation (unchanged):** content merged **after** this hook (retrieval/engine assembly) requires separate controls (`retrieval_poisoning_defense`, post-build policies) — not expanded in this ADR.

---

## 14. Negative qualification matrix

| Case | Expected outcome | Semantic owner | Execution denied? | Evidence | Regression test |
|------|------------------|----------------|-------------------|----------|-----------------|
| Benign UAEP prompt | ALLOW | PromptDefenseMiddleware | No | middleware result ALLOW | New: UAEP hook + defense enabled |
| Prompt injection in UAEP input | BLOCK | PromptDefenseMiddleware | Yes | BLOCK reason + optional defense event | `test_prompt_defense_middleware_blocks_injection_pattern` pattern on UAEP ctx |
| Hostile instructions only in post-build context | Not blocked at this hook (documented gap) | Context/policy plane | — | N/A — track separately | — |
| Missing typed payload | BLOCK | PromptDefenseMiddleware | Yes | BLOCK reason | Existing `test_prompt_defense_blocks_before_context_build_without_llm_payload` |
| Wrong payload type | BLOCK | PromptDefenseMiddleware | Yes | BLOCK reason | Same |
| Empty prompt, nonempty runtime_state only | BLOCK today; **after A** empty prompt ALLOW if typed prompt empty | PromptDefenseMiddleware | If injection N/A | middleware result | UAEP: typed empty → ALLOW |
| Invalid tenant / execution identity | Fail-closed per existing tenant middleware / identity guards | Tenant security / execution identity | Yes | existing tests | unchanged graph tests |
| Middleware BLOCK | No `build_context` / step abort | UAEP guard_hook / pipeline | Yes | hook result | UAEP unit/integration |
| Middleware ERROR | BLOCK via guard | hook_runtime_guard | Yes | hook_error event | `test_pipeline_exception_blocks_with_hook_error` |
| Middleware timeout | BLOCK via guard | hook_runtime_guard | Yes | hook_timeout event | pipeline timeout tests |
| BLOCK + telemetry publish failure | **Still DENY** | Pipeline guard | Yes | execution blocked even if bus raises | New: bus failure does not ALLOW |
| Phase mismatch on violation event | **After E:** schema PASS | hook_runtime_guard + catalog | N/A (deny already) | durable store accepts event | extend `test_hook_runtime_guard` |
| No event bus | DENY unchanged; no event | hook_runtime_guard | Yes | no publish | existing `event_bus=None` paths |
| Validating durable store | Violation event persists with `STEP_EXECUTION` | schema guard | N/A | store write | integration with validating store |
| Recovery after BLOCK | No automatic re-admit without sanctioned workflow | Execution policy | Yes | task/run failure | lifecycle test |
| Graph pathway regression | Unchanged ALLOW/BLOCK semantics | ContextManager | Per case | graph unit tests | `context_manager` + security wiring |

**Invariant:** telemetry failure **never** converts DENY → ALLOW.

---

## 15. FRZ mapping (scoped evidence only — globals remain OPEN)

| FRZ | Contribution of this ADR / future impl | Status |
|-----|----------------------------------------|--------|
| **FRZ-SEC-01** | Typed tenant-scoped hook input + fail-closed UAEP/graph parity | **OPEN** — ADR prepares policy only |
| **FRZ-SEC-04** | Closes UAEP prompt-defense bypass class under strict profile | **OPEN** |
| **FRZ-SEC-07** | Bounded threat matrix §13–§14 | **OPEN** |
| **FRZ-REG-02** | Regression gates for B1/B2 fixes | **OPEN** |
| **FRZ-REG-03** | Negative tests (phase mismatch, missing payload) | **OPEN** |
| **FRZ-PRD-12** | Canonical lifecycle with prompt defense enabled on UAEP | **OPEN** |
| **FRZ-CTR-01 / FRZ-TYP-01** | Preserve Tier-0 typed middleware ABI | Evidence contribution |
| **FRZ-BND-*** / **FRZ-OWN-*** | Single owner per row in §4 | Evidence contribution |

**new global FRZ PASS = 0**

---

## 16. Parent / blocker implications

| Parent | Effect |
|--------|--------|
| **PROD-Q-R1-R7-D3** | Durable projection scope remains audit-ready; **strict UAEP + prompt defense** blocked on this ADR + implementation |
| **PROD-Q-R1-R7** | Remains **NOT CLOSED** until Gate B qualified |
| **PROD-Q-R1-R7-Q** | §C security dimensions gain regression target; no automatic Package D closure |

---

## 17. Open operator decisions

1. **Confirm B1=A** (UAEP `LlmInferenceHookPayload` alignment) vs escalate to B/C with written justification.
2. **Confirm B2=E** (catalog phase + payload diagnostic for source phase).
3. **Approve** post-ratification task `PROD-Q-R1-R7-UAEP-STRICT-HOOK-SECURITY-IMPL-01` and pytest budget.
4. **Accept documented limitation** that `BEFORE_CONTEXT_BUILD` scans intake message only (retrieval poisoning remains separate controls).

---

## 18. Ratification conditions

Operator ratification **ACCEPTED** requires:

1. Written approval of **A + E** (or documented alternative with threat-model delta).
2. Acknowledgment that implementation is a **separate** task with `production delta > 0`.
3. Independent audit plan for implementation commit on GitHub (not Cursor report alone).
4. No simultaneous promotion of global FRZ rows to PASS.

Until then: **PROPOSED — ARCHITECTURE DECISION REQUIRED**.

---

## 19. Post-ratification implementation definition (not executed)

**Task ID:** `PROD-Q-R1-R7-UAEP-STRICT-HOOK-SECURITY-IMPL-01`

**Scope:** Implement §10; qualify §14; re-enable stage-15 prompt defense; graph + D3 regressions.

**Targeted verification (single pytest invocation preferred):**

```text
uv run pytest -p no:xdist -q \
  tests/unit/runtime/nexus/test_uaep_context_assembled_event_contract.py \
  tests/unit/applications/test_application_security_wiring.py \
  tests/unit/runtime/middleware/test_hook_runtime_guard.py \
  tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py::test_canonical_agent_lifecycle_reaches_serving_and_execution \
  tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py \
  tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py
```

**Success criteria:**

1. Benign UAEP input permitted through active Prompt Defense.
2. Malicious intake blocked.
3. Missing/invalid payload denied (fail-closed).
4. Hook BLOCK recorded with catalog-valid phase.
5. No weakening or loss of security evidence on validating store path.
6. Graph `BEFORE_CONTEXT_BUILD` pathway unaffected.
7. Canonical lifecycle with Prompt Defense enabled on UAEP.
8. D3 corruption/recovery tests remain green.
9. No second authority or alternate security path.

---

## 20. Intended after dependency graph

```text
[UAEP reflex execute — target]
  UaepExecutor.execute
    → HookContext(
         phase=CONTEXT_BUILDING,
         payload=LlmInferenceHookPayload(prompt=request.message),
         subject=…,
       )
    → MiddlewarePipeline.run_before(BEFORE_CONTEXT_BUILD)
         → PromptDefenseMiddleware → ALLOW | BLOCK
         → invoke_guarded_hook
              → _emit_hook_violation → build_platform_signal_event(
                   kind=hook_blocked,
                   phase omitted → catalog STEP_EXECUTION,
                   payload={…, source_execution_phase: context_building},
                 )
              → assert_runtime_event_schema → PASS
    → guard_hook → build_context on ALLOW only
```

---

## 21. Enterprise architecture audit (@ proposal HEAD)

| Criterion | Result | Evidence |
|-----------|--------|----------|
| Layer boundaries | **PASS** | Apps middleware consumes Tier-0 contracts; UAEP in runtime nexus; no `applications/` import in `intergrax/` core violation in proposed shape |
| Communication direction | **PASS** | Producers → pipeline → middleware; events → bus/store |
| Contracts over implementations | **PASS** | Decision anchored on `middleware_hook_semantics` + catalog |
| Exactly-one semantic owner | **PASS** | §4 matrix |
| Exactly-one composition owner | **PASS** | Pipeline + sanctioned producers |
| Strong typing | **PASS** | Alt A preserves typed payload |
| Modularity | **PASS** | Middleware unchanged under A |
| Pluginability | **NOT PROVEN** | Prompt Defense profile plugins exist; UAEP qualification pending impl |
| Structural replaceability | **PASS** | Middleware/port pattern unchanged |
| No reflection / dynamic probing | **PASS** | `isinstance` on typed models only |
| No pseudo-contracts | **PASS** | Rejects `runtime_state`-only security input |
| No duplicated security mechanism | **PASS** | Single PromptDefenseMiddleware |
| No bypasses | **FAIL (current code)** / **PASS (target)** | UAEP empty payload today; harness disables defense |
| Governance ≠ Execution | **PASS** | ADR does not execute |
| Proposal ≠ permission | **PASS** | Status PROPOSED |
| No downstream authority expansion | **PASS** | Producers supply truth; middleware decides |
| Fail-closed behavior | **PARTIAL** | BLOCK without payload proven; telemetry path broken (B2) |
| Correct event identity/phase | **FAIL (current)** / **PASS (target E)** | Phase mismatch test |
| Security evidence continuity | **PARTIAL** | Deny without durable event today on B2 failure |
| Observability non-authoritative | **PASS** | Deny independent of bus |
| Regression protection | **PARTIAL** | D3-R1 tests document failure; impl tests pending |

---

## 22. Operator ratification decision (single question)

> **Ratify PROD-Q-R1-R7-UAEP-STRICT-HOOK-SECURITY-ADR-01 R0 as ACCEPTED with recommended combination B1=Alternative A (UAEP `LlmInferenceHookPayload` at `BEFORE_CONTEXT_BUILD`) and B2=Alternative E (catalog-canonical platform hook violation phase with diagnostic source phase in payload), authorizing implementation task `PROD-Q-R1-R7-UAEP-STRICT-HOOK-SECURITY-IMPL-01`?**

**Responses:** `ACCEPT A+E` · `ACCEPT WITH AMENDMENT (specify B/C/F)` · `REJECT`

**Operator response (recorded):** `ACCEPT A+E` — authorizes IMPL-01 only; does **not** close PROD-Q-R1-R7, D3 production security slice, PROD-Q, or global **FRZ-***.

---

## 23. IMPL-01 implementation reference (Cursor)

| Item | Value |
|------|-------|
| **START_HEAD** | `84c1e550fd847ba9e19bcc87742ca086e888bf2a` |
| **FINAL_COMMIT** | `dbf1c3596c90146e3819ad17e0dc9b0c3c4664d7` |
| **B1** | `uaep_executor.py` — `LlmInferenceHookPayload` + `MiddlewareExecutionSubjectFacet` at `BEFORE_CONTEXT_BUILD` |
| **B2** | `hook_runtime_guard.py` — omit `phase=ctx.phase`; `source_execution_phase` in payload |
| **Harness** | `canonical_agent_lifecycle_composition.py` — removed stage-15 `prompt_defense_enabled=False` workaround |
| **Maximum status** | **READY FOR AUDIT** — independent acceptance not granted by this section |

### Before / after owner graph (implementation)

**Before:** UAEP `EmptyMiddlewareHookPayload` + harness prompt-defense off; hook violations with off-catalog top-level `phase`.

**After:** UAEP aligns with graph typed payload; platform hook violations use catalog `STEP_EXECUTION` with diagnostic `source_execution_phase`; stage-15 exercises product prompt defense default.

### Test evidence (mandatory core regression)

Command in §19; log under `.tmp/session/PROD-Q-R1-R7-UAEP-STRICT-HOOK-SECURITY-IMPL-01/`.

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
