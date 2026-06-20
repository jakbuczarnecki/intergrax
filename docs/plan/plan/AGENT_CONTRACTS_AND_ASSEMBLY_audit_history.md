# AGENT_CONTRACTS_AND_ASSEMBLY — audit history + LC closeout

**Parent hub:** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../AGENT_CONTRACTS_AND_ASSEMBLY.md)

## Phase ACP — Agent Cognitive Patterns (ACP)

**Status:** **Done** (2026-06-11) — Waves **0–8** delivered; master register **80/80** ACP-* rows **Done**; fleet migration **100%** Runtime dimension  
**Architecture:** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §13–§40 (incl. **§32.0** readability & typed-only contracts)  
**ADR:** [ADR-AGENT-001](../adr/entries/2026-06-11/ADR-AGENT-001.md) · [ADR-AGENT-002](../adr/entries/2026-06-11/ADR-AGENT-002.md) · [ADR-AGENT-003](../adr/entries/2026-06-11/ADR-AGENT-003.md)  
**Author guide:** [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix AC (sync with §32.0)  
**Audit:** [`audit/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../audit/AGENT_CONTRACTS_AND_ASSEMBLY.md) · domain audit **2026-06-11**  
**Priority ladder:** **Band 2aw** — **closed** · **Band 2bb (ACP-CLOSE)** — **closed** · **Band 2bc (ACP-FINISH)** — **closed** (2026-06-13)

**Strategic outcome (delivered):** Tier-2 authors use **`agent.run(AgentRunRequest)`** + typed **`on_next_step` → `StepOutcome`**; environment merges per-agent memory/tools/RAG/LLM from Tier-3 profile; Nexus remains `Task` entry for multi-agent prod.

**Remaining (ACP-FINISH):** none — §25.4–§25.5 **Done** via **ACP-TOK-*** (2026-06-11 implementation · 2026-06-13 doc sync).

**Explicit production gate:** mutating agents — **ACP-CLOSE-PROD-*** **Done**; token budget depth **Done** (**ACP-TOK-2** · **ACP-TOK-3**) for hosts using `AgentBinding.budget_slice`.

**Doc canon status (2026-06-13):** architecture §13–§40 **accepted and implementation-complete**; GAP-ACP-36/37 **Closed** via **ACP-TOK-*** + **ACP-FINISH-DOC-1**.

**Explicitly excluded:** Nexus refactor; moving `GraphExecutor`/`PolicyEngine` into agents; Phase K business agents; new Tier-0 execution engine.

**Full-domain scope:** Phase **ACP** (§13–§40) **implemented** at platform level; **§12–§20** normative — maintain via gate + **AUDIT-IDEAL** residuals. See [§12–§20 scope mapping](#acp-scope-mapping-12-20-vs-acp-waves).

---

### ACP scope mapping §12–§20 vs ACP waves

**Purpose:** Prevent treating registry, contract, prompt, and lifecycle canon as “closed trivia” while only shipping ACP runtime. Every new agent MUST satisfy **§12 contract** *and* **§13+ runtime** after Wave 0–2.

| Arch § | Topic | Baseline in code | Plan status | Verified by |
|--------|-------|------------------|-------------|-------------|
| **§12** | Agent contract (capabilities, schemas, tools, risk, validation, failure modes) | **Done** — ACP-CON-4 register gate | ACP-CON-4 · ACP-0b · ACP-DX-5 **Done** | `test_agent_assembly_resolver.py` + register rejection |
| **§14** | Agent execution result | **Done** — typed `AgentRunResult` + trace | ACP-DX-1 · ACP-OBS-1 **Done** | Typed `AgentRunResult` + trace |
| **§15** | Agent registry | **Done** (REG-*) | ACP-CON-6 · AUDIT-IDEAL-19.1 **Done** | `check_harness_registry_resolution.py` · `registry_snapshot_store.py` |
| **§16** | Capability model | **Done** (CG-*) | ACP-CON-6 **Done** | Capability routing integration test |
| **§17** | Prompt registry | **Done** (PE-*) | PE-* **Done** | PE wiring tests |
| **§18** | Registry architecture | **Done** (REG-*) | REG-* **Done** | `test_registry_wiring.py` |
| **§19** | Capability graph | **Done** (CG-*) | CG-* · AUDIT-IDEAL-20.1 **Done** | `phase_v_capability_graph_guard.py` |
| **§20** | Lifecycle governance | **Done** (V-ALG, AS-2) | ACP-PROD-9 · AUDIT-IDEAL-31.1 **Done** | `check_agents_lifecycle_metadata.py` · `check_on_call_ownership_model.py` |
| **§13–§40** | ACP runtime (run, step loop, env, prod) | **Done** — incl. §25.4–§25.5 | ACP · ACP-CLOSE · **ACP-FINISH Done** | ACP-TOK-* · `test_acp_token_*` |

**Rule:** An agent PR MUST pass **§12 assembly validation** (**ACP-CON-4**) and applicable **ACP-CLOSE** rows when touching mutating prod paths or legacy surfaces.

---

### ACP implementation principles (normative for every PR)

| # | Principle | Architecture | Enforcement |
|---|-----------|--------------|-------------|
| P1 | **Readability first** | §32.0 | Every `on_next_step` ends with one `StepOutcome.*` factory; reviewer understands continue/complete/fail without running app |
| P2 | **Typed-only author surface** | §32.0 · §37.1 | Pydantic `extra=forbid`; no `dict` state keys in `agents/`; CI `check_agent_typed_state.py` |
| P3 | **READ → UPDATE → DECIDE** | §32.0.1 | `load_session_state` → `state_delta` → `StepOutcome` — no in-place mutation |
| P4 | **One engine, two entries** | §29 · §38 | Direct `run()` and `Task→Nexus` share `AgentRunRequest` merge + step loop — no divergent code paths |
| P5 | **Harness executes, agent decides** | §38 | Domain planning in `on_next_step`; `HarnessKernel` owns policy/trace/budget/state — no planning in kernel |
| P5b | **Runtime is glue only** | §38 · §32.4 | `AgentRuntime.advance_step` calls `on_next_step` then `HarnessKernel.execute_step` — **no** policy logic, trace append, or state merge in runtime |
| P6 | **Legacy removal is deliverable** | §13.5 · ACP-CLOSE-LEG | DEBT register → zero; no new author-visible UAEP/AgentEngine after ACP-CLOSE |
| P9 | **§12 contract is not optional** | §12 · §45 | Register-time gate: schemas, risk, validation_rules, failure_modes — **ACP-CON-4** |
| P10 | **Fleet migration is a program** | Wave 8 · §40.15 | Tiered batches `ACP-MIG-*`; not one-off ACP-LEG-2 PR |
| P11 | **Prod decision = scoreboard** | §40.15 · ACP-PROD-12 | Single report; thresholds binding for roster promotion |
| P7 | **Cross-layer contracts** | §30 · matrix below | Agent PRs that touch tools/memory/RAG/policy MUST cite paired domain plan row |
| P8 | **Architecture = acceptance spec** | §45 · waves | Wave DoD = architecture behaviors demonstrable in tests — not “types exist” alone |

---

### ACP legacy & technical debt register (must shrink to zero)

**Audit 2026-06-11:** **18/18 Closed** in code — DEBT register fully closed; **ACP-CLOSE complete** (CI-2 Done).

| Debt ID | Legacy surface | Replacement | Status | Closed by / Open row |
|---------|----------------|-------------|--------|----------------------|
| DEBT-ACP-01 | `Agent.run()` without `AgentRunRequest` | §29 typed `run()` | **Closed** | ACP-DX-3 |
| DEBT-ACP-02 | `RuntimeRequest` opaque metadata I/O | `AgentRunRequest` §30.9 | **Closed** | ACP-DX-1 |
| DEBT-ACP-03 | Raw `dict` agent state | `AcpSessionState` §32.0 | **Closed** | ACP-0 · ACP-DX-6 |
| DEBT-ACP-04 | `decide_after_step` + `AgentDecision` author API | `StepOutcome` factories §32.0.4 | **Closed** | ACP-CLOSE-LEG-2 · CognitiveAgent UAEP shim only |
| DEBT-ACP-05 | `get_steps` / `run_step` as primary author API | `on_next_step` §32.5 | **Closed** | ACP-STEP-3 · ACP-8 |
| DEBT-ACP-06 | `AgentEngine.run` fallback in `AgentEngine` | `advance_step` + kernel only | **Closed** | ACP-CLOSE-LEG-1 |
| DEBT-ACP-07 | `build_context` duplicating `RuntimeConfig` | `merge_environment` §30 | **Closed** | ACP-DX-2 · ACP-CFG |
| DEBT-ACP-08 | No `AgentRunTrace` on result | Plane B §31 | **Closed** | ACP-OBS-1 |
| DEBT-ACP-09 | No `ApplicationRunSummary` | Plane A §31 | **Closed** | ACP-OBS-2 |
| DEBT-ACP-10 | Single LLM model per run | `StepLLMRouter` §33 | **Closed** | ACP-LLM-1 |
| DEBT-ACP-11 | Ad-hoc graph handoff via metadata | `SharedContextView` §34 | **Closed** | ACP-STATE-1 |
| DEBT-ACP-12 | Capability routing by class name | Registry token §37.6 | **Closed** | ACP-CON-6 |
| DEBT-ACP-13 | Free-text errors / terminal reasons | Enums §37.4–§37.5 | **Closed** | ACP-CON-1 |
| DEBT-ACP-14 | Full state replace / in-place mutation | `state_delta` §37.2 | **Closed** | ACP-CON-2 |
| DEBT-ACP-15 | Scaffold UAEP-first only | Typed scaffold `--pattern` | **Closed** | ACP-8 |
| DEBT-ACP-16 | Roster on legacy patterns | Typed loop fleet-wide | **Closed** | ACP-MIG-* · ACP-LEG-2 |
| DEBT-ACP-17 | No prod checkpoint / idempotency (platform) | §40 persistence modules | **Closed** — platform + host depth + compensation queue + cross-run idempotency | ACP-PROD-1..3 · **ACP-CLOSE-PROD-1..8 Done** |
| DEBT-ACP-18 | ReAct loop split from TOOL-ENG-6 | Unified budget §25.2 | **Closed** | **ACP-CLOSE-PAT-1** + **TOOL-ENG-6** |

**Removal policy:** ACP + ACP-CLOSE + **ACP-FINISH closed** (2026-06-13). Extend via ADR if scope expands.

---

### ACP cross-domain coupling matrix

Agent layer is **not isolated**. Each ACP wave may require coordinated delivery or read-only integration with:

| Intergrax domain | Architecture | Plan | ACP touchpoints | Sync rule |
|------------------|--------------|------|-----------------|-----------|
| **Unified execution / policy** | `UNIFIED_EXECUTION_RUNTIME` §42 | `plan/UNIFIED_EXECUTION_RUNTIME.md` | Policy pre/post inside **`HarnessKernel.execute_step`**; `PolicyVerdictRecord` | ACP-STEP-2b · ACP-ORG-3 before STRICT prod |
| **Orchestration / Nexus** | `ORCHESTRATION` · `NEXUS_EXECUTION_FLOW` | `plan/ORCHESTRATION.md` · `plan/NEXUS_EXECUTION_FLOW.md` | Graph node → `AgentRunRequest`; HITL pause | ACP-DX-4 must not fork Task lifecycle |
| **Tools** | `TOOLS` §gateway | `plan/TOOLS.md` **TOOL-ENG-6** | `tool_gateway`; declarative `StepActionRequest`; idempotency | ACP-3 + TOOL-ENG-6 same sprint; ACP-PROD-2..3 |
| **Skills** | `SKILLS` | `plan/SKILLS.md` | `skill_ids` → allowed_tools on contract | AS-3 CI already; verify on binding slices |
| **LLM adapters** | `LLM_ADAPTERS` | `plan/LLM_ADAPTERS.md` | `StepLLMRouter`; per-step `LlmStepResult` on trace | ACP-LLM-1 + LLM profile merge |
| **Memory** | `MEMORY` | `plan/MEMORY.md` | `memory_view`; `memory_scope` user vs org §30.9 | ACP-DX-2 identity namespace tests |
| **RAG** | `RAG` | `plan/RAG.md` | `rag_gateway`; collection binding | ACP-DX-5 host slice tests |
| **Observability** | `OBSERVABILITY` | `plan/OBSERVABILITY.md` | `AgentRunTrace`; redaction §40.8 | ACP-OBS-1 + OBS trace spine |
| **Reliability / HITL** | `RELIABILITY_FAILURE_AND_HITL` | `plan/RELIABILITY_FAILURE_AND_HITL.md` | `pause_hitl`; checkpoint resume | ACP-PROD-1 + acceptance 04/05 |
| **Tier-3 applications** | `TIER3_APPLICATION_ENVIRONMENT` | `plan/TIER3_APPLICATION_ENVIRONMENT.md` | `AgentBinding`; profile merge; intake `RequestIdentity` | ACP-DX-5 · ACP-ORG-1 |
| **Critic / verification** | `CRITIC_VERIFICATION` | `plan/CRITIC_VERIFICATION.md` | `ReflectionAgent` critic hooks | ACP-6 — no critic SDK in Tier-2 |
| **DX / scaffold / CI** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | `plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | Scaffold; conformance scripts | ACP-8 · ACP-PROD-10 |
| **Integrations** | `INTEGRATIONS` | `plan/INTEGRATIONS.md` | `required_integration_slugs`; gateway-only | ACP-CON-7 |
| **Elastic / scale** | `ELASTIC_CAPACITY_AND_SCALING` | `plan/ELASTIC_CAPACITY_AND_SCALING.md` | Budget guards §32.6 | **ACP-STEP-2b** kernel enforcement |

**PR rule:** If an ACP PR imports or changes Tier-1 Nexus semantics beyond the bridge listed in the wave, **stop** and split — agent layer composes; it does not fork Nexus.

---

### ACP architecture traceability (§ → deliverables → verification)

| Architecture § | Capability delivered | Primary plan IDs | Verification |
|----------------|---------------------|------------------|--------------|
| §12 | Full `AgentContract` at register | **ACP-CON-4** · AS-1 | Incomplete contract → `AgentAssemblyError` |
| §14 | Typed execution result | ACP-DX-1 · ACP-OBS-1 | `AgentRunResult` fields + trace |
| §15–§16 | Registry + capabilities | REG/CG **Done** · ACP-CON-6 · AUDIT-IDEAL-19.1 **Done** | Capability token routing |
| §17–§19 | Prompt + registry + graph | PE/REG/CG **Done** · AUDIT-IDEAL-20.1 **Done** | `phase_v_capability_graph_guard.py` |
| §20 | Lifecycle governance | V-ALG/AS **Done** · ACP-PROD-9 · AUDIT-IDEAL-31.1 **Done** | `check_on_call_ownership_model.py` |
| §13 | `run()` + `on_next_step` author API | ACP-DX-3 · ACP-STEP-1 | Direct `run()` test + agent_os 01 |
| §21–§28 | Cognitive patterns + gaps closed | ACP-1..6 · ACP-9 | Pattern unit + reference agents |
| §29 | `AgentRunRequest` / `Result` | ACP-DX-1 · ACP-CON-1 | Round-trip JSON; enum tests |
| §30 · §30.9 | Environment merge + identity | ACP-DX-2 · ACP-DX-5 | Merge order; user_id gate |
| §25.4 | Invocation token metering (agent + environment) | **ACP-TOK-1** · GAP-ACP-36 | `invocation_usage` + `budget.tokens_*` live in kernel |
| §25.5 | Per-agent limits + exceed reactions | **ACP-TOK-2** · **ACP-TOK-3** · GAP-ACP-37 | Hard cap blocks LLM; `BudgetReactionProfile` paths |
| §31 | Dual observability planes | ACP-OBS-1 · ACP-OBS-2 | Trace on result; multi-agent summary |
| §32 · **§32.0** | Step loop + readability | ACP-STEP-* · ACP-DX-6 · ACP-0 | Factory tests; typed-state CI |
| §33 | Per-step LLM | ACP-LLM-1 | Model hint within profile |
| §34 | Shared graph state | ACP-STATE-1 · ACP-PROD-5 | Two-agent handoff |
| §35 | UC-1..11 without rewrite | Waves 2–6 integration | UC mapping acceptance |
| §37 | Operational contracts | ACP-CON-* | Merge, enums, routing, security CI |
| §38 | NexusLoop vs HarnessKernel split | ACP-STEP-2 · ACP-STEP-2b | Runtime glue-only test; kernel owns policy/trace/budget/state |
| §39 | Org policy envelope | ACP-ORG-* | UC-11 fixture |
| §40 | Production reliability + scoreboard | ACP-PROD-* **Done** · **ACP-CLOSE-PROD-1..8 Done** | Per-roster `production_mode` via §40.15 scoreboard thresholds |
| §45 | New agent checklist | ACP-8 · ACP-11..13 | Scaffold + conformance CI |
| **Fleet** | Roster migration | **ACP-MIG-*** Wave 8 | Tracker 100% Runtime dimension |

### ACP — Master register (ACP-DOC · ACP-DX · ACP-PROD · …)

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| ACP-DOC.1 | ACP0 | **Architecture canon §21–§28** — ACP spec, flows, patterns, gaps | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | This document + ADR-AGENT-001 |
| ACP-DOC.2 | ACP0 | **Appendix AC** — cognitive patterns author guide in `AGENT_CREATION_GUIDE.md` | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + pattern selection table + skeleton |
| ACP-DOC.3 | ACP0 | **Audit prompt** — ACP dimensions in domain audit | **Done** | `audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Regenerated via `generate_domain_audit_prompts.py` |
| ACP-ADR.1 | ACP0 | **ADR-AGENT-001** accepted | **Done** | `docs/adr/entries/2026-06-11/ADR-AGENT-001.md` | Linked from architecture §21 |
| ACP-ADR.2 | ACP0 | **ADR-AGENT-002** accepted — `run()` facade | **Done** | `docs/adr/entries/2026-06-11/ADR-AGENT-002.md` | Linked from architecture §29 |
| ACP-DOC.4 | ACP0 | **Architecture §29–§30** — run facade + per-agent environment binding | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §29–§30 + ADR-AGENT-002 |
| ACP-DOC.5 | ACP0 | **Architecture §31–§36** — dual observability, step loop, LLM routing, UC catalog | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §31–§36 + ADR-AGENT-003 |
| ACP-ADR.3 | ACP0 | **ADR-AGENT-003** accepted — `on_next_step` + dual observability | **Done** | `docs/adr/entries/2026-06-11/ADR-AGENT-003.md` | Linked from architecture §31–§32 |
| ACP-DX-1 | ACP-DX | **`AgentRunRequest` / `AgentRunResult` / `RequestIdentity` / `AgentEnvironmentOverrides`** Pydantic contracts | **Done** | `intergrax/contracts/agent_run.py` | Round-trip + user_id required when memory_scope=user |
| ACP-DX-2 | ACP-DX | **`merge_environment`** + `EffectiveAgentRunEnvironment` + **memory_scope resolution** §30.9 | **Done** | `intergrax/agents/run_environment.py` | Unit test merge order + user vs org namespace |
| ACP-DX-3 | ACP-DX | **`IntergraxAgent.run` upgrade** — uses merge + typed result; hooks `configure_run`, `on_run_start/end` | **Done** | `intergrax/agents/authoring/base.py`, `acp_run.py` | Test direct run without Nexus |
| ACP-DX-4 | ACP-DX | **Nexus node bridge** — Task metadata → AgentRunRequest same merge path | **Done** | `runtime_request_bridge.py`, `agent_engine.py`, `acp_checkpoint_task_enricher.py` | Harness hosts set `acp.session.v1` via task enricher |
| ACP-DX-5 | ACP-DX | **`AgentBinding` profile slices** — tool/memory/integration per roster entry | **Done** | `applications/contracts/manifest.py` | Binding slice merge tests |
| ACP-DX-6 | ACP-DX | **Author readability kit** — `StepOutcome` factories, `load_session_state` / `session_state_delta`, `check_agent_typed_state.py` | **Done** | `intergrax/agents/authoring/step_outcome.py`, `state_access.py`, `scripts/` | Factories set consistent enums; CI fails raw dict state in agents |
| ACP-DOC.10 | ACP0 | **Architecture §32.0** — author readability & typed-only contracts (READ/UPDATE/DECIDE) | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §32.0 + ACP-AP-11..15 + checklist §45 |
| ACP-STEP-1 | ACP-STEP | **`AgentStepContext` / `StepOutcome` / author `on_next_step`** on `IntergraxAgent` | **Done** | `intergrax/agents/authoring/step_loop.py`, `agent_step_context.py`, `base.py` | Unit: terminal + continue via factories §32.0; no dict author surface |
| ACP-STEP-2 | ACP-STEP | **`AgentRuntime.advance_step`** — glue only: `on_next_step` → `HarnessKernel.execute_step`; **no policy/trace/state logic** | **Done** | `intergrax/agents/authoring/step_loop.py` | Unit: advance_step contains no policy imports; delegates 100% to kernel |
| ACP-STEP-2b | ACP-STEP | **`HarnessKernel.execute_step`** — L1 harness cycle: policy pre/post, state merge §37.2, gateways, budgets §32.6, trace/`AgentStepRecord`, declarative actions §32.8 | **Done** | `intergrax/runtime/kernel/step_kernel.py` | Integration: policy deny + trace record + budget exceeded from kernel only |
| ACP-CON-4 | ACP-CON | **§12 full contract gate at register** — `input_schema`, `output_schema`, `risk_level`, `validation_rules`, `failure_modes`, budgets; reject incomplete contracts | **Done** | `agent_assembly_resolver.py` | Register with stub contract → `AgentAssemblyError`; roster agents pass |
| ACP-STEP-3 | ACP-STEP | **UAEP legacy bridge** — `run_step` → advance_step + kernel | **Done** | `uaep_step_bridge.py`, `uaep.py` | UAEP unit tests green; kernel trace on session |
| ACP-OBS-1 | ACP-OBS | **`AgentRunTrace` / `AgentStepRecord`** on `AgentRunResult` | **Done** | `intergrax/contracts/agent_run_trace.py` | `test_acp_wave3_trace`: `steps[0].llm_calls` populated |
| ACP-OBS-2 | ACP-OBS | **`ApplicationRunSummary`** from Nexus task completion | **Done** | `application_run_summary_builder.py`, `task_finisher.py` | `test_application_run_summary_builder` multi-agent |
| ACP-LLM-1 | ACP-LLM | **`StepLLMRouter`** on step context | **Done** | `intergrax/agents/authoring/llm_router.py` | Per-step model hint + trace record |
| ACP-STATE-1 | ACP-STATE | **`SharedContextView`** for graph handoffs | **Done** | `intergrax/contracts/shared_context.py`, `shared_context_bridge.py` | Two-node handoff unit test |
| ACP-CON-1 | ACP-CON | **`AgentRunErrorCode` / `TerminalReason` enums** + Pydantic on run contracts | **Done** | `intergrax/contracts/agent_run.py`, `agent_run_enums.py` | extra=forbid; enum round-trip |
| ACP-CON-2 | ACP-CON | **`state_delta` merge-patch** + `_version` + checkpoint/resume | **Done** | `intergrax/agents/authoring/state_merge.py` | Unit: merge, delete null, conflict |
| ACP-CON-3 | ACP-CON | **Side-effect mode** immediate vs declarative enforcement | **Done** | `intergrax/agents/authoring/side_effect_validation.py` | Reject mixed mode per step |
| ACP-CON-6 | ACP-CON | **Capability routing** — registry query by token not class | **Done** | Nexus selection path + test | Integration: two impls same capability |
| ACP-CON-7 | ACP-CON | **Security CI guards** — gateway-only I/O, STRICT widen deny | **Done** | `scripts/check_agent_step_security.py` | CI green on roster |
| ACP-DOC.6 | ACP0 | **Architecture §37** — pre-implementation operational contracts | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Audit gaps A–G closed in canon |
| ACP-DOC.7 | ACP0 | **Architecture §38** — NexusLoop vs HarnessKernel execution stack | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §38 + ACP-INV-11 |
| ACP-DOC.8 | ACP0 | **Architecture §39** — organizational policy envelope & virtual workforce | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §39 + UC-11 |
| ACP-ORG-1 | ACP-ORG | **`OrganizationalPolicyEnvelope`** on ApplicationEnvironmentProfile | **Done** | `intergrax/applications/contracts/org_policy.py` | Pydantic extra=forbid |
| ACP-ORG-2 | ACP-ORG | **`OrganizationalPolicyContext`** in merge_environment | **Done** | `intergrax/agents/run_environment.py` | Role + envelope merge test |
| ACP-ORG-3 | ACP-ORG | **Kernel org enforcement** — channel/tool/playbook overlays | **Done** | `intergrax/runtime/kernel/step_kernel.py` | Block denied channel tool |
| ACP-ORG-4 | ACP-ORG | **`PolicyVerdictRecord` + compliance_summary** on trace/result | **Done** | `intergrax/contracts/agent_run_trace.py` | Step trace assertion |
| ACP-ORG-5 | ACP-ORG | **Reference org fixture + golden compliance eval** | **Done** | `lab_org_virtual_workforce_defaults` + gate tests | Zero POLICY_DENIED on happy path |
| ACP-DOC.9 | ACP0 | **Architecture §40** — production reliability, safety, persistence, release gates | **Done** | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §40 canon accepted; ACP-PROD delivered; depth = ACP-CLOSE |
| ACP-PROD-1 | ACP-PROD | **Checkpoint / resume / replay** — step store + crash recovery | **Done** | `checkpoint_store.py`, acceptance `05c`/`05d` | Resume smoke; no double mutating tool |
| ACP-PROD-2 | ACP-PROD | **Side-effect idempotency ledger** — dedupe + declarative execute/commit in kernel | **Done** | `side_effect_ledger.py`, `declarative_tool_executor.py`, `step_kernel.py` | Resume replay skip + commit on invoke |
| ACP-PROD-3 | ACP-PROD | **ToolExecutionProfile + compensation enqueue** | **Done** | `tool_execution_profile.py`, `compensation_enqueue.py`, `step_kernel.py` | Policy deny after commit triggers recall |
| ACP-PROD-4 | ACP-PROD | **ReliabilityProfile in HarnessKernel** — retry/CB/timeout | **Done** | `intergrax/runtime/kernel/session_reliability.py` | REL profile wired |
| ACP-PROD-5 | ACP-PROD | **SharedContextView CAS + conflict policy** | **Done** | `intergrax/contracts/shared_context.py` | Parallel graph conflict test |
| ACP-PROD-6 | ACP-PROD | **`ArtifactRef` contract** on result/step | **Done** | `intergrax/contracts/artifact_ref.py` | Typed artifacts in test |
| ACP-PROD-7 | ACP-PROD | **Agent threat model CI** — matrix §40.7 | **Done** | `scripts/check_agent_threat_model.py` | CI-02..03 + threat cases |
| ACP-PROD-8 | ACP-PROD | **Privacy/redaction on trace/memory** | **Done** | `intergrax/contracts/privacy_redaction.py` | PII redaction test |
| ACP-PROD-9 | ACP-PROD | **Release eval gates** — golden/regression/policy suites | **Done** | `scripts/check_agent_release_gates.py` | Staging gate green |
| ACP-PROD-10 | ACP-PROD | **CI conformance matrix §40.10** | **Done** | `scripts/check_acp_ci_conformance_matrix.py` | CI-01..16 applicable rows |
| ACP-PROD-11 | ACP-PROD | **Schema version registry + migration adapters** | **Done** | `intergrax/contracts/migrations/` | check_contract_schema_versions |
| ACP-0 | ACP1 | **`AcpSessionState` / `acp.state.v1` schema** — Pydantic envelope + agent subclass pattern §32.0 | **Done** | `intergrax/contracts/acp_state.py` | Unit test round-trip; extra=forbid |
| ACP-0b | ACP1 | **`cognitive_pattern` on AgentContract** — optional field + validation | **Done** | `agent_contract_meta.py`, `agent_assembly_resolver.py` | `test_cognitive_patterns` validation |
| ACP-1 | ACP1 | **`CognitiveAgent` ABC** — perceive/reason/act/evaluate + `on_next_step` | **Done** | `patterns/base.py` | Pattern probe runs |
| ACP-2 | ACP2 | **`ReflexAgent`** | **Done** | `patterns/reflex.py` | `PatternReflexProbe` |
| ACP-3 | ACP2 | **`ReActAgent`** — bounded loop, budget in `acp.state.v1` | **Done** | `patterns/react.py`, `patterns/states.py` | `PatternReActProbe` |
| ACP-4 | ACP2 | **`PlanExecuteAgent`** — multi-step + phase machine | **Done** | `patterns/plan_execute.py` | `PatternPlanExecuteProbe` |
| ACP-5 | ACP2 | **`DecompositionAgent`** — sub-question queue + convergence | **Done** | `patterns/decomposition.py` | `PatternDecompositionProbe` |
| ACP-6 | ACP2 | **`ReflectionAgent`** — draft/critique/revise phases | **Done** | `patterns/reflection.py` | `PatternReflectionProbe` (CVL hook Wave 6+) |
| ACP-7 | ACP3 | **Decision helpers** — legacy UAEP bridge; new code uses `StepOutcome` factories §32.0 (ACP-DX-6) | **Done** | `intergrax/agents/authoring/decisions.py` | Primary `finish`/`continue_with`/…; UAEP helpers deprecated; `to_step_outcome` bridge |
| ACP-8 | ACP3 | **Scaffold `--pattern`** flag on `new-agent` | **Done** | `scaffold/new_agent.py`, `scaffold/cli.py` | `test_acp_pattern_scaffold` |
| ACP-9 | ACP4 | **Harness reference probes** — one per pattern | **Done** | `patterns/reference.py` | Pattern probe unit tests |
| ACP-10 | ACP4 | **Unit test package** `tests/unit/agents/authoring/patterns/` | **Done** | tests | 32 gate tests — one probe run per pattern + contracts + phase machines |
| ACP-11 | ACP5 | **Gate: ACP pattern scaffold** — no UAEP boilerplate | **Done** | `scripts/check_scaffold_acp_pattern.py` | Scaffold smoke script |
| ACP-13 | ACP5 | **Pattern conformance** — contract vs class | **Done** | `scripts/check_agent_pattern_conformance.py` | AST check on `agents/*/contract.py` |
| ACP-12 | ACP5 | **Acceptance: pattern agent in agent_os suite** | **Done** | `tests/acceptance/agent_os/test_acp_pattern_agents.py` | NexusLoop + `acp.session.v1` per pattern (mock LLM) |
| ACP-CFG | ACP6 | **`build_context` profile injection** — reduce per-agent `RuntimeConfig` duplication | **Done** | `intergrax/agents/reference_harness.py` | `build_lab_agent_runtime_config_from_merged` |
| ACP-LEG-1 | ACP-LEG | **Deprecate AgentEngine path** — `DeprecationWarning` in `AgentEngine` fallback | **Done** | `intergrax/agents/agent_engine.py` | `test_agent_engine_legacy_deprecation` |
| ACP-LEG-2 | ACP-LEG | **Fleet migration complete** — superseded by **Wave 8** `ACP-MIG-*` program (not ad-hoc per-agent) | **Done** | `agents/*` | Scoreboard Runtime ≥100% roster-wide; typed-state CI allowlist empty |
| ACP-MIG-1 | ACP-MIG | **Fleet inventory auditor** — legacy surface per agent (`uaep`/`runtime_engine`/`dict state`) | **Done** | `scripts/audit_agent_fleet_legacy.py` | JSON report for all `agents/*` packages |
| ACP-MIG-2 | ACP-MIG | **Migration tiers + batch order** — harness → staging read-only → staging mutating → prod-eligible | **Done** | plan §6.1aw Wave 8 · `agents/README.md` | Documented tiers match roster table |
| ACP-MIG-3 | ACP-MIG | **Pilot batch (3 agents)** — echo, signoff_probe, research → typed `on_next_step` | **Done** | `agents/echo`, `signoff_probe`, `research` | Direct `run()` + agent_os green per agent |
| ACP-MIG-4 | ACP-MIG | **Product batch** — legal, summary, LKW trio, DSW quartet | **Done** | product `agents/*` | Host wiring tests unchanged; scoreboard Runtime ≥80% each |
| ACP-MIG-5 | ACP-MIG | **Remaining roster** — org_worker, assistant, K-path agents; lab mocks excluded | **Done** | `agents/*` | Zero UAEP-only new code; bridge allowlist shrinking |
| ACP-MIG-6 | ACP-MIG | **Fleet migration CI gate** — `check_agent_fleet_migration.py` blocks regression | **Done** | `scripts/` | CI fails if migrated agent reintroduces legacy surface |
| ACP-MIG-7 | ACP-MIG | **Per-host binding verification** after each batch | **Done** | `applications/*/manifest.py` tests | AgentBinding slices + capability routing per host |
| ACP-PROD-12 | ACP-PROD | **`AgentProductionReadinessReport`** scoreboard — 10 dimensions 0–100% per agent | **Done** | `intergrax/contracts/agent_readiness.py`, `scripts/report_agent_production_readiness.py` | Report generated for roster; prod promotion uses thresholds §6.1az |
| ACP-LEG-3 | ACP-LEG | **Document AgentEngine internal-only** | **Done** | `runtime.py` module docstring + architecture §13 | INTERNAL ONLY banner |
| ACP-LEG-4 | ACP-LEG | **Remove author UAEP from `--pattern` scaffold** — typed hooks only | **Done** | `scaffold/new_agent.py` | `--pattern` agents have no `get_steps` |
| ACP-DOC.11 | ACP0 | **Detailed implementation waves §6.1aw** + debt/coupling matrix | **Done** | `plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | §6.1aw |
| ACP-DOC.12 | ACP0 | **Plan correction** — §12–§20 scope map, runtime/kernel split, ACP-CON-4 | **Done** | `plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Wave 1 + scope mapping |
| ACP-DOC.13 | ACP0 | **Wave 8 fleet migration** + **§6.1az production readiness scoreboard** | **Done** | `plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Operational closure |

---

## Phase ACP-CLOSE — Architecture compliance closeout

**Status:** **Done** (2026-06-11) — post-ACP domain audit · **Band 2bb** · queue **[§6.1bb](#61bb-harness-implementation-queue--acp-close-done)**  
**Source:** Architecture ↔ plan ↔ code audit (2026-06-11)  
**Goal (achieved):** **DEBT-ACP register → zero**; §40 host depth; §40.12 evidenced; CI-1..3 wired.

**Explicitly excluded:** Nexus graph/orchestration refactor; Phase K agents; new Tier-0 engines.

### ACP-CLOSE — Master register

| ID | Area | Deliverable | Status | Arch § | Modules / scripts | Acceptance |
|----|------|-------------|--------|--------|-------------------|------------|
| ACP-CLOSE-DOC-1 | DOC | Plan Phase ACP header + scope map sync | **Done** | plan | `plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` | This update |
| ACP-CLOSE-DOC-2 | DOC | Architecture §28.3 GAP register — Closed/Open truth table | **Done** | §28.3 | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | 32 Closed · 3 Open (03/04/07) |
| ACP-CLOSE-DOC-3 | DOC | Architecture §36.4 · §40.13 · implementation status tables | **Done** | §36.4 · §40.13 | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | Code maps Done; §40 platform implemented |
| ACP-CLOSE-DOC-4 | DOC | Regenerate domain audit prompt | **Done** | audit | `audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` | `generate_domain_audit_prompts.py` |
| ACP-CLOSE-LEG-1 | LEG | **Remove** `AgentEngine` fallback from `AgentEngine` | **Done** | §13.5 · §38 | `agent_engine.py` | `ValueError` on non-UAEP/non-ACP agents; DEBT-ACP-06 **Closed** |
| ACP-CLOSE-LEG-2 | LEG | **Remove** author-visible UAEP (`decide_after_step` on `IntergraxAgent`) | **Done** | §13.3–13.4 | `uaep_linear_bridge.py`, `uaep.py` | `linear_agent_decide_after_step`; DEBT-ACP-04 **Closed** for linear agents |
| ACP-CLOSE-LEG-3 | LEG | Retire public `uaep_pipeline.py` bridge | **Done** | §13.5 | ADR-FLOW-005 | Public module removed; superseded by ACP-only `AgentEngine` |
| ACP-CLOSE-LEG-4 | LEG | §45 checklist — UAEP internal-only wording | **Done** | §45 | `AGENT_CREATION_GUIDE.md`, `check_agent_creation_guide_acp_canon.py` | No author UAEP-first path; CI grep gate |
| ACP-CLOSE-LEG-5 | LEG | **Delete** Tier-1 `RuntimeEngine` pipeline stack (`pipelines/`, `runtime_steps/`, engine planner) | **Done** | §13.5 · ADR-FLOW-005 | `nexus/tools/tool_loop.py`, `plan_context_invocation.py` | ACP-only scaffold; `RuntimeConfig.pipeline` removed; docs scrubbed |
| ACP-CLOSE-PROD-1 | PROD | `AgentCheckpointStore` on **all mutating product hosts** | **Done** | §40.1 | `acp_checkpoint_host_wiring.py`, `harness_host_runtime.py` | Auto-resolve store on harness hosts; exposed on `HarnessHostRuntime` |
| ACP-CLOSE-PROD-2 | PROD | `acp_checkpoint_task_enricher` on product hosts (lab pattern) | **Done** | §40.1.4 | `task_control_wiring.py`, `applications/*/host/factory.py` | `build_reliability_task_enricher(..., agent_checkpoint_store=)` |
| ACP-CLOSE-PROD-3 | PROD | `CatalogDeclarativeToolInvoker` — real execution context (no `MagicMock` shim) | **Done** | §32.8 · §40.3 | `catalog_declarative_invoker.py` | `_CatalogDispatchLLMStub` + direct `RuntimeContext`; preserves host invoker |
| ACP-CLOSE-PROD-4 | PROD | Nexus **E2E** acceptance — catalog declarative invoker (not callable mock) | **Done** | §27 · §40.12 | `test_acp_nexus_catalog_declarative_resume.py` | `build_harness_host_runtime` + `NexusLoop` resume green |
| ACP-CLOSE-PROD-5 | PROD | **Durable compensation queue** for `enqueued` requests | **Done** | §40.3.3 | `compensation_queue_store.py`, `compensation_queue_worker.py` | Host + kernel persist `enqueued` jobs |
| ACP-CLOSE-PROD-6 | PROD | `ReliabilityProfile.idempotency_store` ↔ ledger replay §40.2.2 | **Done** | §40.2 | `idempotency_ledger_bridge.py`, `idempotency_store_wiring.py`, kernel replay | `test_acp_declarative_mutating_cross_run_dedupe.py` |
| ACP-CLOSE-PROD-7 | PROD | §40.12 checklist **green** — reference mutating agent artifact | **Done** | §40.12 | `section_40_12_checklist.py`, `check_acp_section_40_12_checklist.py` | `build/acp_section_40_12_reference.json` |
| ACP-CLOSE-PROD-8 | PROD | Scoreboard mutating agents **100%** checkpoint + idempotency dimensions | **Done** | §40.15 | `readiness/scoreboard.py` | `--require-mutating-checkpoint-idempotency-100` green |
| ACP-CLOSE-PAT-1 | PAT | ReAct ↔ **TOOL-ENG-6** unified tool loop + budget keys | **Done** | §26.3 · §25.2 | `patterns/react.py`, `react_budget.py`, `tool_loop_step.py` | 2-iteration integration; DEBT-ACP-18 **Closed** |
| ACP-CLOSE-PAT-2 | PAT | `ReflectionAgent` → CVL critic hooks (gateway only) | **Done** | §26.6 | `critic_gateway.py`, `patterns/reflection.py`, `acp_session_host_wiring.py` | Gateway-only CVL; host `critic_graph_hooks` on ACP run |
| ACP-CLOSE-PAT-3 | PAT | Author terminology — single canonical §29 entry | **Done** | §28.3 GAP-07 · §29.0 | `AGENT_CREATION_GUIDE.md`, architecture §22–§23 · §27 | §29 single entry; GAP-ACP-07 **Closed** |
| ACP-CLOSE-ORG-1 | ORG | STRICT **configure_run widen deny** per-agent | **Done** | §39.4 | `configure_run_strict.py`, `merge_environment`, `acp_run` | `test_configure_run_strict.py` |
| ACP-CLOSE-ORG-2 | ORG | UC-11 compliance golden per **product host** | **Done** | §39.5 | `product_host_org_envelope`, `uc11_compliance_golden.py` | `test_uc11_product_host_compliance.py` (6 hosts) |
| ACP-CLOSE-CI-1 | CI | Post-LEG grep + fleet migration gate — zero Tier-2 `AgentEngine` | **Done** | §40.10 CI-04 | `check_agent_fleet_migration.py`, `check_agent_acp_close_ci.py` | `.github/workflows/unit-tests.yml` |
| ACP-CLOSE-CI-2 | CI | Anti-pattern ACP-AP-02 after TOOL-ENG-6 | **Done** | §28.4 | `check_agent_acp_ap02_tool_loop_boundary.py` | CI-17 · Nexus does not schedule tool iterations |
| ACP-CLOSE-CI-3 | CI | `check_agent_production_readiness.py --fail-on-blockers` in gate workflow | **Done** | §40.15 | `check_agent_acp_close_ci.py` | CI-16 + gate workflow |

**Cross-plan (not ACP-CLOSE IDs — deliver in owning domain):**

| ID | Domain plan | Deliverable | Status |
|----|-------------|-------------|--------|
| AUDIT-IDEAL-19.1 | this file §AUDIT-IDEAL | Durable cross-host registry snapshot | **Done** |
| AUDIT-IDEAL-20.1 | this file §AUDIT-IDEAL | Product CI blast-radius on tool/skill changes | **Done** |
| AUDIT-IDEAL-31.1 | this file §AUDIT-IDEAL | Owner/on-call mandatory on certified agents | **Done** |
| TOOL-ENG-6 | `plan/TOOLS.md` | Tool loop step — sync with ACP-CLOSE-PAT-1 | **Done** |
| ACP-TOK-1..3 | this file §ACP-FINISH | Token metering, limits, reactions | **Done** |
| ACP-TOK-CI | this file §ACP-FINISH | Token budget CI gate | **Done** |
| ACP-FINISH-DOC-1 | this file §ACP-FINISH | GAP-ACP-36/37 Closed + §40.13 sync | **Done** (2026-06-13) |

**ACP-CLOSE DoD:** DEBT-ACP **3/3 Open → Closed**; architecture §28.3 synced; mutating scoreboard dimensions **≥100%**; §40.12 evidenced; `pytest -m gate` green.

**Delivery rule:** One **ACP-CLOSE-\*** ID per PR → update this register → journal on phase completion.

---

## Phase ACP-FINISH — Agent architecture completion

**Status:** **Done** (2026-06-13) — GAP-ACP-36/37 **Closed** · architecture §28.3 **37 Closed · 0 Open** · **Band 2bc closed**  
**Goal:** Close **GAP-ACP-36** (invocation token rollups) and **GAP-ACP-37** (per-agent limits + application reaction policies). After this phase, agent architecture canon is **decision-complete and implementation-complete** for §13–§40 (AUDIT-IDEAL §12–§20 residuals remain parallel).

**Explicitly excluded:** New cognitive patterns; Nexus orchestration refactor; Phase K business agents.

### ACP-FINISH — Master register

| ID | Area | Deliverable | Status | Arch § | Modules / scripts | Acceptance |
|----|------|-------------|--------|--------|-------------------|------------|
| ACP-TOK-1 | TOK | **Metering** — agent + environment token rollups in invocation state | **Done** | §25.4 · §33.4 | `HarnessKernel`, `acp_run.py`, `merge_environment` | `test_acp_token_usage_metering.py` |
| ACP-TOK-2 | TOK | **Limits** — per-agent caps from application + hard/advisory enforcement | **Done** | §25.5.1–§25.5.2 | `AgentBinding.budget_slice`, `AgentExecutionOptions.max_total_tokens`, kernel pre-LLM check | `test_acp_token_budget_enforcement.py` |
| ACP-TOK-3 | TOK | **Reactions** — environment policies on threshold/exceed | **Done** | §25.5.3 · §30.8 | `CostProfile.budget_reaction`, host hooks, notify wiring | `abort` · `hitl` · `degrade_model` · `notify_only` · `custom_hook` paths tested |
| ACP-TOK-CI | CI | Token budget contract gate | **Done** | §25.4–§25.5 · §40.10 | `check_agent_token_budget_contract.py` | CI-18 row; fails if kernel bypasses metering or agents increment budget in state_delta |
| ACP-FINISH-DOC-1 | DOC | Architecture status tables + GAP-ACP-36/37 → Closed | **Done** (2026-06-13) | §28.3 · §36.4 · §40.13 | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`, audit prompt | 37 Closed · 0 Open; §40.13 declares architecture **complete** |

### ACP-FINISH — Sub-task breakdown (implementation guide)

| Parent | Sub | Work package | Depends on |
|--------|-----|--------------|------------|
| ACP-TOK-1 | 1a | After each LLM call: increment `AcpBudgetState.tokens_in/out/total` from `LlmCallRecord` / router drain | — |
| ACP-TOK-1 | 1b | Build `AcpInvocationUsageView` (agent mirror + environment rollup from task / `ApplicationRunSummary`) | 1a |
| ACP-TOK-1 | 1c | `merge_environment`: materialize `tokens_limit` / `tokens_remaining` when limits resolved (no enforcement yet) | 1a |
| ACP-TOK-1 | 1d | Tests: `test_acp_token_usage_metering.py`, adaptive `model_hint` downgrade fixture (§33.4) | 1a–1c |
| ACP-TOK-2 | 2a | Resolve limits: `CostProfile` → `AgentBinding.budget_slice` → `execution_options` (merge §25.5.1) | 1c |
| ACP-TOK-2 | 2b | `HarnessKernel` pre-LLM hard enforcement when `enforcement=hard` | 2a |
| ACP-TOK-3 | 2c | Wire `BudgetReactionProfile` reactions + `RuntimeEvent` `BUDGET_THRESHOLD` / `BUDGET_EXCEEDED` | 2b |
| ACP-TOK-3 | 2d | Tier-3 reference: one product host with `budget_slice` + `budget_reaction` + notify hook | 2c |
| ACP-TOK-CI | — | Static + smoke gate; wire `check_agent_acp_close_ci.py` or matrix CI-18 | 1d + 2d |
| ACP-FINISH-DOC-1 | — | Close GAP register; refresh §40.13.1 audit acceptance | all above |

**Cross-domain:** [`plan/LLM_ADAPTERS.md`](LLM_ADAPTERS.md) (metering source) · [`plan/UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) (`RunBudget`, `BudgetEnforcer`) · [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) (COST-1 · `cost_runtime_bridge.py` baseline).

**ACP-FINISH DoD:** ACP-TOK-1..3 + ACP-TOK-CI + ACP-FINISH-DOC-1 **Done**; `pytest -m gate` green; architecture §28.3 **0 Open** GAPs for ACP.

**Delivery rule:** One **ACP-TOK-\*** or **ACP-FINISH-DOC-1** per PR → update register → journal on phase completion.

---
