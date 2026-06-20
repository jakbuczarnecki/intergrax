# AGENT_CONTRACTS_AND_ASSEMBLY — audit history + LC closeout

**Parent hub:** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../AGENT_CONTRACTS_AND_ASSEMBLY.md)

## Phase ACP-DEPTH — (merged into ACP-FINISH)

Historical alias for token-depth work. **Closed register:** [Phase ACP-FINISH](#phase-acp-finish--agent-architecture-completion) **Done** (2026-06-13).

---

### 6.1av Harness implementation queue — Agent Cognitive Patterns (ACP) — closed

**Status:** **Done** (2026-06-11) · **Follow-on:** [§6.1bc ACP-FINISH](#61bc-harness-implementation-queue--acp-finish-closed) **Done** (2026-06-13)

**Purpose:** Historical wave order (Band 2aw). **Detailed steps:** [§6.1aw](#61aw-acp-detailed-implementation-waves).

| Wave | IDs | Closes architecture | Legacy removed |
|------|-----|---------------------|----------------|
| **0** | ACP-DX-1 · ACP-CON-1 · ACP-CON-4 · ACP-0 · ACP-DX-6 · ACP-CON-2 | **§12** gate · §29 · §37.1–§37.2 · **§32.0** types | DEBT-ACP-02/03/13/14 ✓ |
| **1** | ACP-STEP-1 · ACP-STEP-2 · ACP-STEP-2b · ACP-CON-3 | §32 · §38 · §32.8 | DEBT-ACP-04 partial → **ACP-CLOSE-LEG-2** |
| **2** | ACP-DX-2 · ACP-DX-3 · ACP-DX-4 · ACP-DX-5 · ACP-CFG | §29–§30 · §36 | DEBT-ACP-01/07 ✓ |
| **3** | ACP-OBS-1 · ACP-OBS-2 · ACP-LLM-1 · ACP-STATE-1 | §31–§34 | DEBT-ACP-08..11 ✓ |
| **4** | ACP-STEP-3 · ACP-LEG-1 · ACP-LEG-3 | §13.4 UAEP bridge | DEBT-ACP-05 ✓ · 06 → **ACP-CLOSE-LEG-1** |
| **5** | ACP-0b · ACP-1..13 · ACP-8 · ACP-LEG-4 | §21–§28 patterns + scaffold | DEBT-ACP-15 ✓ · 18 → **ACP-CLOSE-PAT-1** |
| **8** | **ACP-MIG-1..7** · **ACP-LEG-2** | Fleet migration | DEBT-ACP-16 ✓ |
| **6** | ACP-CON-6 · ACP-CON-7 · ACP-ORG-1..5 | §37.6–§37.7 · §39 | DEBT-ACP-12 ✓ |
| **7** | ACP-PROD-1..12 | §40 platform + scoreboard | DEBT-ACP-17 platform ✓ · depth → **ACP-CLOSE-PROD-*** |

**Continuous:** §6.1 gate maintenance · `pytest -m gate` green every PR.

---

### 6.1aw ACP detailed implementation waves

Each wave lists **PR-sized steps** in order. A step is **Done** only when acceptance tests pass and listed debt IDs are closed or explicitly bridged with deprecation.

#### Wave 0 — Typed contracts foundation (architecture §29 · §37 · §32.0)

**Goal:** All run/step/state types exist before loop wiring. **No** author-facing `dict` in new code after this wave.

| Step | ID | Files / modules | Tasks | Acceptance | Debt closed |
|------|-----|-----------------|-------|------------|-------------|
| 0.1 | ACP-DX-1 | `intergrax/contracts/agent_run.py` | Define `AgentRunRequest`, `AgentRunResult`, `RequestIdentity`, `AgentEnvironmentOverrides`, `AgentExecutionOptions`, `GovernanceSnapshot`, `AgentRunCost` — all `extra=forbid` | `tests/unit/contracts/test_agent_run_roundtrip.py` | DEBT-ACP-02 |
| 0.2 | ACP-CON-1 | same + enums module | `AgentRunErrorCode`, `TerminalReason`, `StepNextAction`, `AgentRunError`; wire to result/outcome fields | Enum round-trip; reject free-text in validation | DEBT-ACP-13 |
| 0.2b | ACP-CON-4 | `agent_assembly_resolver.py` | Extend `validate_contract_metadata` for §12 required fields: `input_schema`, `output_schema`, `risk_level`, `validation_rules` (≥1), `failure_modes` (≥1), `max_steps` or contract budgets; wire `AgentRegistry.register` | `test_agent_assembly_resolver.py`: incomplete contract raises `AgentAssemblyError`; reference agents pass | §12 |
| 0.3 | ACP-0 | `intergrax/contracts/acp_state.py` | `AcpSessionState`, `AcpBudgetState`, `ACP_STATE_KEY` constant; document subclass pattern §32.0.2 | Serialize/deserialize; `_version` field | DEBT-ACP-03 |
| 0.4 | ACP-DX-6a | `intergrax/agents/authoring/step_outcome.py` | `StepOutcome` model + factories: `continue_with`, `complete`, `fail`, `pause_hitl`, `replan` — set enums consistently | Factory unit tests per §32.0.4 | DEBT-ACP-04 |
| 0.5 | ACP-DX-6b | `intergrax/agents/authoring/state_access.py` | `load_session_state(agent, step_ctx)`, `session_state_delta(model, *, include=...)` on `IntergraxAgent` | Typed load + delta from Pydantic dump | DEBT-ACP-03 |
| 0.6 | ACP-CON-2 | `intergrax/agents/authoring/state_merge.py` | RFC 7396 shallow merge; `null` delete; `_version` increment; resume conflict → `VALIDATION_FAILED` | Merge unit matrix §37.2 | DEBT-ACP-14 |
| 0.7 | ACP-DX-6c | `scripts/check_agent_typed_state.py` | Fail CI on `state.get(` / `state[` in `agents/` (allowlist bridge files until Wave 4) | Script in CI workflow | — |

**Wave 0 DoD:** `uv run pytest tests/unit/contracts/ tests/unit/runtime/registry/test_agent_assembly_resolver.py -q` green; **incomplete `AgentContract` cannot register**; architecture §37.1 + §12 gate provable without Nexus.

---

#### Wave 1 — Step loop & kernel (architecture §32 · §38)

**Goal:** One iteration = `advance_step` (glue) → `on_next_step` (domain) → `HarnessKernel.execute_step` (harness).  
**Invariant (normative):** `AgentRuntime.advance_step` has **no** policy engine calls, trace writers, budget counters, or state-merge logic — those live **only** in `HarnessKernel.execute_step` (architecture §13 table · §38.1 L1 · §38.3).

```text
AgentRuntime.advance_step(agent, step_ctx):
    outcome = await agent.on_next_step(step_ctx)     # L2 — domain only
    await HarnessKernel.execute_step(outcome, step_ctx)  # L1 — all harness work
    return outcome

HarnessKernel.execute_step(outcome, step_ctx) -> StepExecutionRecord:
    1. policy pre-check (tools, budget, autonomy, org overlays when §39 wired)
    2. validate + apply state_delta §37.2 (_version bump)
    3. run declarative requested_actions if mode=declarative §32.8
    4. policy post-check on outcome + side effects
    5. enforce step/session budgets §32.6
    6. emit RuntimeEvents; append AgentStepRecord to run trace (Plane B)
    7. optional checkpoint hook when enabled
    DOES NOT: call on_next_step • domain replan • choose next graph agent
```

| Step | ID | Files / modules | Tasks | Acceptance | Cross-domain |
|------|-----|-----------------|-------|------------|--------------|
| 1.1 | ACP-STEP-1a | `intergrax/contracts/agent_step.py` (or `agent_run.py`) | `AgentStepContext` typed: gateways as protocols, `state_snapshot` internal, `load_session_state` path; gateways policy-bound at context build | Context construction test | TOOLS gateway protocol |
| 1.2 | ACP-STEP-1b | `intergrax/agents/authoring/step_loop.py` | `IntergraxAgent.on_next_step` default; `@step` driver mapping; forbid override `advance_step` | Continue + terminal factory tests | — |
| 1.3 | ACP-STEP-2 | `step_loop.py` | `AgentRuntime.advance_step`: **exactly two awaits** — `on_next_step` then `kernel.execute_step`; static check / test that module imports no policy or trace sink | `test_advance_step_is_glue_only.py`: no `PolicyEngine` / `TraceWriter` in advance_step body | — |
| 1.4 | ACP-STEP-2b | `intergrax/runtime/kernel/step_kernel.py` | `HarnessKernel.execute_step` implements full L1 cycle above; **zero** imports from `agents/` domain packages | Integration: policy deny, budget exceeded, trace step record — all attributed to kernel | UAEP · OBSERVABILITY |
| 1.5 | ACP-CON-3 | `step_kernel.py` or kernel helper | Enforce immediate vs declarative mutual exclusion §32.8 at kernel validation | Mixed-mode step rejected before actions run | TOOLS trace |

**Wave 1 DoD:** Glue-only test green; kernel integration test proves policy + trace + state merge without `advance_step` containing harness logic; reference `on_next_step` agent runs 3-step loop.

**Anti-pattern (reject in review):** policy pre/post or `trace.append` inside `AgentRuntime.advance_step` — violates §38 and duplicates L1.

---

#### Wave 2 — Run facade, environment merge, Nexus bridge (architecture §29–§30 · §36)

**Goal:** Same path for `agent.run()` and graph node execution.

| Step | ID | Files / modules | Tasks | Acceptance | Cross-domain |
|------|-----|-----------------|-------|------------|--------------|
| 2.1 | ACP-DX-2 | `intergrax/agents/run_environment.py` | `merge_environment(platform, app_profile, org_envelope, binding, request)` → `EffectiveAgentRunEnvironment`; `memory_scope` resolution §30.9 | Merge order unit test | MEMORY · TIER3 |
| 2.2 | ACP-DX-3 | `intergrax/agents/authoring/base.py` | `run(request: AgentRunRequest)` loop using Wave 1; hooks `configure_run`, `on_run_start/end`; typed `AgentRunResult` | Direct run without Nexus | — |
| 2.3 | ACP-DX-4 | `intergrax/agents/agent_engine.py`, graph executor bridge | Task metadata → `AgentRunRequest`; same `merge_environment` + `run()` | `agent_os` test 01 parity | NEXUS_EXECUTION_FLOW |
| 2.4 | ACP-DX-5 | `applications/contracts/`, host wiring | `AgentBinding` tool/memory/RAG/LLM slices per roster entry | Legal or research host test | TIER3 |
| 2.5 | ACP-CFG | `reference_harness.py`, migrate 1–2 reference agents | Remove duplicated `RuntimeConfig` from `build_context`; profile injection only | Reference agent diff shrinks | INTEGRATIONS |

**Wave 2 DoD:** `await agent.run(AgentRunRequest(...))` in pytest; Nexus single-agent test unchanged behavior; DEBT-ACP-01/07 closed.

---

#### Wave 3 — Observability, LLM routing, shared state (architecture §31–§34)

**Goal:** Plane A + Plane B journals; per-step model; graph handoffs typed.

| Step | ID | Files / modules | Tasks | Acceptance | Cross-domain |
|------|-----|-----------------|-------|------------|--------------|
| 3.1 | ACP-OBS-1 | `intergrax/contracts/agent_run_trace.py` | `AgentRunTrace`, `AgentStepRecord` (tool/RAG/LLM/decision/error codes); attach to `AgentRunResult` | Trace assertion on 2-step run | OBSERVABILITY |
| 3.2 | ACP-LLM-1 | `intergrax/agents/authoring/llm_router.py` | `StepLLMRouter` on context; policy-bound `model_hint`; record in step trace | Per-step model within profile | LLM_ADAPTERS |
| 3.3 | ACP-STATE-1 | `intergrax/contracts/shared_context.py` | `SharedContextView` read/write for graph handoffs | Two-node handoff unit test | ORCHESTRATION |
| 3.4 | ACP-OBS-2 | Nexus task completion path | `ApplicationRunSummary` on Task terminal; `trace_id` join | `agent_os` 02 multi-agent | OBSERVABILITY |

**Wave 3 DoD:** **Met** — `result.trace.steps[0].llm_calls` populated (`test_acp_wave3_trace`); `ApplicationRunSummary` on `TaskResult.metadata` via `build_nexus_task_result` + builder unit test.

---

#### Wave 4 — Legacy bridge & deprecation (architecture §13.4–§13.5)

**Goal:** Existing UAEP agents keep working **through bridge**; new code uses typed loop only.

| Step | ID | Files / modules | Tasks | Acceptance | Debt closed |
|------|-----|-----------------|-------|------------|-------------|
| 4.1 | ACP-STEP-3 | `intergrax/agents/uaep.py` | Map `run_step`/`decide_after_step` → `advance_step` + typed `StepOutcome` translation | Existing UAEP unit tests green | DEBT-ACP-05 bridge |
| 4.2 | ACP-LEG-1 | `agent_engine.py` | `DeprecationWarning` on `AgentEngine` fallback path | Warning in test | DEBT-ACP-06 |
| 4.3 | ACP-LEG-3 | docs + `runtime.py` docstring | Mark `AgentEngine` internal-only in canon | No author guide references | — |

**Wave 4 DoD:** **Met** — UAEP steps route through `HarnessKernel` (`uaep_step_bridge`); `DeprecationWarning` on `AgentEngine` fallback; `AgentEngine` marked internal-only. Fleet body migration = Wave 8.

---

#### Wave 5 — Cognitive patterns & scaffold (architecture §21–§28 · §32.0)

**Goal:** Pattern library demonstrates **readable** typed agents; scaffold emits correct skeleton.

| Step | ID | Files / modules | Tasks | Acceptance | Cross-domain |
|------|-----|-----------------|-------|------------|--------------|
| 5.1 | ACP-0b | `agent_contract_meta.py` | `cognitive_pattern` enum on contract; assembly validation | Resolver test | AS |
| 5.2 | ACP-1 | `patterns/base.py` | `CognitiveAgent` ABC → `on_next_step` delegates perceive/reason/act/evaluate | Base unit test | — |
| 5.3 | ACP-2 | `patterns/reflex.py` | Single-shot `StepOutcome.complete` | Unit test | — |
| 5.4 | ACP-3 | `patterns/react.py` | Budget in `AcpBudgetState`; tool loop | Integration + **TOOL-ENG-6** sync | TOOLS |
| 5.5 | ACP-4..6 | plan_execute, decomposition, reflection | Each uses typed state + factories; reflection uses CVL hook | Pattern tests | CRITIC_VERIFICATION |
| 5.6 | ACP-8 · ACP-LEG-4 | `scaffold/new_agent.py` | `--pattern`; emit state subclass + `on_next_step` skeleton §32.0.5; **no** UAEP boilerplate | Scaffold smoke | DX |
| 5.7 | ACP-9..10 | `pattern_reference_*.py`, tests package | One harness reference per pattern | Lab wiring smoke | — |
| 5.8 | ACP-11..13 | CI scripts | UAEP-only gate; pattern conformance; extend `agent_os` | CI green | DX |

**Wave 5 DoD:** **Met** — `new-agent` defaults to typed `reflex`; `--pattern` selects other patterns; `--uaep` legacy only; pattern library + probes + CI scripts; DEBT-ACP-05/15 closed.

**Prerequisite for Wave 8:** Waves 0–5 Done (typed loop, `run()`, patterns, scaffold target exist).

---

#### Wave 8 — Fleet migration program (operational — full roster)

**Goal:** Migrate **all** Tier-2 agents in `agents/` from legacy UAEP/`AgentEngine`/dict-state surfaces to typed **`on_next_step` + `AcpSessionState` + `agent.run(AgentRunRequest)`** — in controlled batches, without breaking hosts.

**Scope (~16 product agents + harness probes; excludes `lab/mock_agents.py` fixtures unless listed):**

| Tier | Agents (initial roster) | Risk | Migration target | Scoreboard min before next tier |
|------|-------------------------|------|------------------|--------------------------------|
| **T0 Harness** | echo, signoff_probe | Low | Reflex / typed loop | Runtime ≥80% |
| **T1 Staging read-only** | research, summary, local_search | Low–med | Pattern base + typed state | Runtime ≥80% |
| **T2 Staging mutating** | legal, local_indexer, local_synthesizer, DSW×4 | Med | Full §32.0 + host tests | Runtime ≥90%; Checkpointing N/A until Wave 7 |
| **T3 Prod-eligible** | echo (prod), future promoted | High | Scoreboard **overall ≥90%**, no dimension below 80% | Per §6.1az |
| **T4 Experimental** | problem_radar, vendor_discovery, org_worker, assistant | Variable | Best-effort; may stay bridge longer with ADR | Documented waiver |

**Per-agent migration checklist (every agent in a batch):**

```text
1. Inventory  — ACP-MIG-1 report row: legacy flags, host bindings, mutating tools
2. Contract   — §12 complete (ACP-CON-4); cognitive_pattern set (ACP-0b)
3. State      — typed AcpSessionState subclass; remove dict keys
4. Runtime    — on_next_step + StepOutcome factories; remove author UAEP unless bridge-only shim
5. Tests      — agents/<slug>/tests: await agent.run(AgentRunRequest); agent_os if applicable
6. Host       — manifest AgentBinding unchanged or updated; ACP-MIG-7 host test
7. Scoreboard — generate report; tier gate before merge
8. CI         — check_agent_fleet_migration.py + typed-state (allowlist -= agent)
```

| Step | ID | Tasks | Acceptance |
|------|-----|-------|------------|
| 8.1 | ACP-MIG-1 | `audit_agent_fleet_legacy.py` → `build/agent_fleet_inventory.json` | All packages listed; legacy flags accurate |
| 8.2 | ACP-MIG-2 | Migration tiers in plan + `agents/README.md` migration table | Operator can pick next batch from table |
| 8.3 | ACP-MIG-3 | **Pilot PR batch** (≤3 agents): echo, signoff_probe, research | 3 scoreboard reports; Runtime ≥80% each |
| 8.4 | ACP-MIG-4 | **Product PR batch**: legal, summary, LKW×3, DSW×4 (may split 2 PRs) | Host tests green; Runtime ≥80% |
| 8.5 | ACP-MIG-5 | Remaining agents + shrink typed-state allowlist to zero | `check_agent_typed_state.py` full roster |
| 8.6 | ACP-MIG-6 | CI regression gate on fleet | Re-introducing `get_steps`-only agent fails CI |
| 8.7 | ACP-MIG-7 | Post-batch host binding verification | legal + research + lab smoke per batch |
| 8.8 | ACP-LEG-2 | Close fleet migration — DEBT-ACP-16 | **Done** — 100% roster Runtime; `check_agent_production_readiness.py --require-fleet-migration-closure` |

**Wave 8 DoD:** No production agent on UAEP-only author path; fleet inventory clean; **ACP-LEG-2 Done**; scoreboard generated for every roster agent.

**Delivery rule:** One **batch PR** (ACP-MIG-3 or MIG-4) may migrate ≤5 agents; each agent row updated in [fleet tracker](#acp-fleet-migration-tracker) below.

##### ACP fleet migration tracker

| Agent | Tier | Host(s) | Status | Batch | Runtime % | Blocker |
|-------|------|---------|--------|-------|-----------|---------|
| echo | T0/T3 | lab, poc | **Done** | MIG-3 | — | — |
| signoff_probe | T0 | lab | **Done** | MIG-3 | — | — |
| research | T1 | research, lab | **Done** | MIG-3 | — | — |
| summary | T1 | research | **Done** | MIG-4 | — | — |
| legal | T2 | legal, lab | **Done** | MIG-4 | — | — |
| local_indexer | T2 | LKW | **Done** | MIG-4 | — | — |
| local_search | T1 | LKW | **Done** | MIG-4 | — | — |
| local_synthesizer | T2 | LKW | **Done** | MIG-4 | — | — |
| dispute_* (×4) | T2 | DSW | **Done** | MIG-4 | — | — |
| organization_worker | T4 | lab | **Done** | MIG-5 | — | — |
| intergrax_assistant | T4 | assistant | **Done** | MIG-5 | — | — |
| problem_radar | T4 | K.1 path | **Done** | MIG-5 | — | — |
| vendor_discovery | T4 | K.2 path | **Done** | MIG-5 | — | — |

*Update **Status** → In progress / Done per PR; **Runtime %** from ACP-PROD-12 report.*

---

#### Wave 6 — Routing, security, organizational policy (architecture §37.6–§37.7 · §39)

| Step | ID | Tasks | Acceptance | Cross-domain |
|------|-----|-------|------------|--------------|
| 6.1 | ACP-CON-6 | Nexus resolves `required_capability` → registry token; ban class name in task payload | **Done** — `capability_routing.py` + gate test | ORCHESTRATION · REG |
| 6.2 | ACP-CON-7 | `check_agent_step_security.py` — gateway-only I/O, STRICT widen deny | **Done** — static roster gate | UNIFIED_EXECUTION_RUNTIME |
| 6.3 | ACP-ORG-1..2 | `OrganizationalPolicyEnvelope` + merge context | **Done** — merge + host profile preset | TIER3 |
| 6.4 | ACP-ORG-3..4 | Kernel org overlays; `PolicyVerdictRecord` on trace | **Done** — channel deny + compliance_summary | UAEP policy |
| 6.5 | ACP-ORG-5 | Lab org fixture + compliance eval | **Done** — happy-path kernel gate | V-EVAL |

**Wave 6 DoD:** UC-11 path demonstrable; capability routing test passes.

---

#### Wave 7 — Production reliability gate (architecture §40)

**Blocks:** mutating / customer-facing agents until minimum **ACP-PROD-1..3** + **ACP-PROD-9..10** Done.

| Step | ID | Delivers §40 capability | Acceptance | Cross-domain |
|------|-----|-------------------------|------------|--------------|
| 7.1 | ACP-PROD-1 | Checkpoint / resume / replay | **Done** — store + host wiring + `test_acceptance_05c`/`05d` resume smoke | RELIABILITY |
| 7.2 | ACP-PROD-2 | Idempotency ledger | **Done** — ledger dedupe + replay skip + kernel execute/commit + host catalog invoker wiring | TOOLS |
| 7.3 | ACP-PROD-3 | `ToolExecutionProfile` + compensation | **Done** — mutating tool gate + step-failure compensation enqueue | TOOLS |
| 7.4 | ACP-PROD-4 | ReliabilityProfile in kernel | **Done** — circuit breaker + checkpoint interval | RELIABILITY |
| 7.5 | ACP-PROD-5 | SharedContext CAS | **Done** — per-key publish/CAS + graph test | ORCHESTRATION |
| 7.6 | ACP-PROD-6 | `ArtifactRef` | **Done** — `artifact_refs` on `AgentRunResult` | OBSERVABILITY |
| 7.7 | ACP-PROD-7..8 | Threat CI + privacy redaction | **Done** — threat gate + PII redaction | OBSERVABILITY · MEMORY |
| 7.8 | ACP-PROD-9..11 | Release gates + CI matrix + schema registry | **Done** — aggregate CI scripts | DX |
| 7.9 | ACP-PROD-12 | Production readiness scoreboard — aggregate §6.1az gates into one report | **Done** — `report_agent_production_readiness.py` · `check_agent_production_readiness.py` | DX · V-EVAL |

**Wave 7 DoD:** §40.12 checklist green for reference mutating agent; scoreboard emitted for roster; architecture §40 maturity gate **unblocks** prod roster promotion via scoreboard thresholds.

---

### 6.1az Agent Production Readiness Scoreboard (ACP-PROD-12)

**Purpose:** Single **operator-facing artifact** — replaces hunting across scattered CI scripts when deciding if an agent may enter **production roster** (`production_mode`, `production_eligible`).

**Artifact:** `AgentProductionReadinessReport` (typed, `extra=forbid`) — per agent, per generation run.

```text
AgentProductionReadinessReport:
    agent_id: str
    contract_id: str
    generated_at: datetime
    overall_pct: float                    # 0–100 weighted mean (see weights)
    production_eligible_recommendation: bool
    dimensions: list[AgentReadinessDimensionScore]

AgentReadinessDimensionScore:
    dimension: AgentReadinessDimension    # enum — 10 values below
    pct: float                            # 0–100
    status: pass | partial | fail | not_applicable
    weight: float                         # for overall_pct
    evidence: list[str]                   # test names, CI script ids, plan rows
    blockers: list[str]                   # human-readable gaps
```

| # | Dimension | Architecture / plan source | Scoring inputs (automated where possible) | Default weight |
|---|-----------|---------------------------|-------------------------------------------|----------------|
| 1 | **Contract** | §12 · ACP-CON-4 | Assembly resolver; schemas; validation_rules; failure_modes | 10% |
| 2 | **Runtime** | §13 · §32 · §32.0 · Wave 8 | `on_next_step`; StepOutcome factories; typed state; no UAEP author surface | 15% |
| 3 | **Policy** | §37.7 · §39 · ACP-ORG | PolicyVerdictRecord in trace; org envelope test; STRICT deny cases | 10% |
| 4 | **Observability** | §31 · ACP-OBS | AgentRunTrace on result; step records; trace_id join | 10% |
| 5 | **Checkpointing** | §40.1 · ACP-PROD-1 | Resume smoke; checkpoint store wired | 10% |
| 6 | **Idempotency** | §40.2 · ACP-PROD-2 | Mutating tools have idempotency_key; ledger dedupe test | 10% |
| 7 | **Security** | §40.7 · ACP-CON-7 · ACP-PROD-7 | Gateway-only I/O; threat matrix rows; vendor import CI | 10% |
| 8 | **Evaluation** | §40.9 · ACP-PROD-9 · V-EVAL | Golden/regression suites registered; staging green | 10% |
| 9 | **Lifecycle** | §20 · V-ALG · AS-2 | owner_team; runbook_ref; promotion evidence when prod-eligible | 5% |
| 10 | **Capability routing** | §37.6 · ACP-CON-6 | Task routes by capability token; binding resolves impl | 10% |

**Production roster promotion thresholds (normative — no compromise for mutating/customer-facing):**

| Profile | `overall_pct` | Per-dimension floor | Extra |
|---------|---------------|---------------------|-------|
| **Read-only staging** | ≥70% | Runtime ≥80% | Checkpointing/Idempotency may be `not_applicable` |
| **Mutating staging** | ≥80% | Checkpointing + Idempotency **100%** | ACP-CLOSE-PROD-1..4 · §40.12 reference |
| **Production roster** | **≥90%** | **No dimension below 80%** (except N/A) | ACP-PROD-9..10 green · §40.12 checklist |
| **Waiver** | — | — | ADR + operator sign-off only |

**Commands (target):**

```bash
uv run python scripts/report_agent_production_readiness.py --agent legal
uv run python scripts/report_agent_production_readiness.py --roster --format markdown
uv run python scripts/check_agent_production_readiness.py --min-overall 90 --fail-on-blockers
```

**Integration:** `check_agent_release_gates.py` (ACP-PROD-9) **consumes** scoreboard output — not duplicate logic. CI matrix §40.10 row **CI-16**: scoreboard generation on roster in gate workflow.

**Architecture canon:** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §40.15.

---

### 6.1ax Suggested PR sequence (single-ID commits)

```text
Wave 0:  ACP-DX-1 → ACP-CON-1 → ACP-CON-4 → ACP-0 → ACP-DX-6 (0.4+0.5+0.7) → ACP-CON-2
Wave 1:  ACP-STEP-1 → ACP-STEP-2b → ACP-STEP-2 → ACP-CON-3   # kernel first, then glue wiring
Wave 2:  ACP-DX-2 → ACP-DX-3 → ACP-DX-4 → ACP-DX-5 → ACP-CFG
Wave 3:  ACP-OBS-1 → ACP-LLM-1 → ACP-STATE-1 → ACP-OBS-2
Wave 4:  ACP-STEP-3 → ACP-LEG-1 → ACP-LEG-3
Wave 5:  ACP-0b → ACP-1 → … → ACP-13 → ACP-8+LEG-4
Wave 8:  ACP-MIG-1 → ACP-MIG-2 → ACP-MIG-3 → ACP-MIG-4 → ACP-MIG-5 → ACP-MIG-6 → ACP-MIG-7 → ACP-LEG-2
Wave 6:  ACP-CON-6 → ACP-CON-7 → ACP-ORG-1 → … → ACP-ORG-5
Wave 7:  ACP-PROD-1 → … → ACP-PROD-11 → ACP-PROD-12
ACP-CLOSE:  DOC-2..4 → LEG-1 → LEG-2 → LEG-3 → PROD-1 → PROD-2 → PROD-3 → PROD-4 → PROD-5 → PROD-6 → PAT-1 (+ TOOL-ENG-6) → ORG-1 → PROD-7 → PROD-8 → PAT-2 → CI-1..3
ACP-FINISH:  TOK-1 → TOK-2 → TOK-3 → TOK-CI → FINISH-DOC-1
```

**Note:** ACP + ACP-CLOSE + **ACP-FINISH complete** (2026-06-13) — §25.4–§25.5 token depth closed.

**Journal:** ACP waves journaled; one entry on **ACP-CLOSE** phase completion, per [`implementation-journal/README.md`](../implementation-journal/README.md).

---

### 6.1bb Harness implementation queue — ACP-CLOSE (done)

**Purpose:** Single ordered backlog for **full architecture ↔ implementation compliance** after ACP waves. **Band 2bb**. **Closed 2026-06-11.**

| Order | ID | Priority | Arch § | Status |
|-------|-----|----------|--------|--------|
| 1 | ACP-CLOSE-DOC-2 | P0 | §28.3 | **Done** |
| 2 | ACP-CLOSE-DOC-3 | P0 | §36.4 · §40.13 | **Done** |
| 3 | ACP-CLOSE-DOC-4 | P1 | audit | **Done** |
| 4 | ACP-CLOSE-LEG-1 | **P0** | §13.5 | **Done** |
| 5 | ACP-CLOSE-LEG-2 | **P0** | §13.4 | **Done** |
| 6 | ACP-CLOSE-LEG-3 | P1 | §13.5 | **Done** |
| 7 | ACP-CLOSE-LEG-4 | P2 | §45 | **Done** |
| 8 | ACP-CLOSE-PROD-1 | **P0** | §40.1 | **Done** |
| 9 | ACP-CLOSE-PROD-2 | **P0** | §40.1.4 | **Done** |
| 10 | ACP-CLOSE-PROD-3 | P1 | §32.8 | **Done** |
| 11 | ACP-CLOSE-PROD-4 | **P0** | §27 · §40.12 | **Done** |
| 12 | ACP-CLOSE-PROD-5 | P1 | §40.3.3 | **Done** |
| 13 | ACP-CLOSE-PROD-6 | P1 | §40.2.2 | **Done** |
| 14 | ACP-CLOSE-PAT-1 + TOOL-ENG-6 | P1 | §26.3 | **Done** |
| 15 | ACP-CLOSE-ORG-1 | P1 | §39.4 | **Done** |
| 16 | ACP-CLOSE-PROD-7 | **P0** | §40.12 | **Done** |
| 17 | ACP-CLOSE-PROD-8 | **P0** | §40.15 | **Done** |
| 18 | ACP-CLOSE-PAT-2 | P2 | §26.6 | **Done** |
| 19 | ACP-CLOSE-PAT-3 | P2 | §28.3 | **Done** |
| 20 | ACP-CLOSE-ORG-2 | P2 | §39.5 | **Done** |
| 21 | ACP-CLOSE-CI-1 | P1 | §40.10 | **Done** |
| 22 | ACP-CLOSE-CI-2 | P2 | §28.4 | **Done** |
| 23 | ACP-CLOSE-CI-3 | P1 | §40.15 | **Done** |

**Parallel (owning plan):** AUDIT-IDEAL-19.1 · 20.1 · 31.1 — **Done** (2026-06-13); see [Phase AUDIT-IDEAL](#phase-audit-ideal--ideal-architecture-gap-register-2026-06-09).

**Minimum viable close (P0 only):** DOC-1..4 · LEG-1 · LEG-2 · PROD-1 · PROD-2 · PROD-4 · PROD-7 · PROD-8 · CI-2 → **Done** — ACP-CLOSE wave complete.

---

### 6.1bc Harness implementation queue — ACP-FINISH (closed)

**Purpose:** Final tasks to declare **agent architecture (§13–§40) implementation-complete**. **Band 2bc**.  
**Status:** **Done** (2026-06-13) — GAP-ACP-36 · GAP-ACP-37 **Closed** · architecture §28.3 open count → **0**.

| Order | ID | Priority | Arch § | Status | Depends |
|-------|-----|----------|--------|--------|---------|
| 1 | ACP-TOK-1 | **P1** | §25.4 · §33.4 | **Done** | — |
| 2 | ACP-TOK-2 | **P1** | §25.5.1–§25.5.2 | **Done** | TOK-1 |
| 3 | ACP-TOK-3 | **P1** | §25.5.3 · §30.8 | **Done** | TOK-2 |
| 4 | ACP-TOK-CI | P2 | §40.10 CI-18 | **Done** | TOK-1..3 |
| 5 | ACP-FINISH-DOC-1 | P1 | §28.3 · §40.13 | **Done** (2026-06-13) | TOK-CI |

**Parallel (ideal-architecture depth — not blocking ACP-FINISH DoD):**

| ID | Priority | Arch § | Status | Notes |
|----|----------|--------|--------|-------|
| AUDIT-IDEAL-19.1 | **P0** | §15 Registry | **Done** | `registry_snapshot_store.py` · `check_registry_snapshot_diff.py` |
| AUDIT-IDEAL-20.1 | P1 | §19 Cap. graph | **Done** | `phase_v_capability_graph_guard.py` · `check_capability_graph_strict_deploy.py` |
| AUDIT-IDEAL-31.1 | P1 | §20 Lifecycle | **Done** | `check_agents_lifecycle_metadata.py` · `check_on_call_ownership_model.py` |

**Suggested PR order:**

```text
ACP-TOK-1 (metering) → ACP-TOK-2 (limits) → ACP-TOK-3 (reactions + reference host)
  → ACP-TOK-CI → ACP-FINISH-DOC-1
```

**Journal:** [`entries/2026-06-13/acp-finish-doc-1-gap-register-closeout.md`](../implementation-journal/entries/2026-06-13/acp-finish-doc-1-gap-register-closeout.md) (ACP-FINISH phase completion).

---

### 6.1bd Layer backlog — post-maturity (P2–P4, non-blocking)

**Status:** Active maintenance — does **not** block layer completion (ACP + AUDIT-IDEAL **Done**).

| ID | Priority | Topic | Status | Notes |
|----|----------|-------|--------|-------|
| COST-1 | P2 | Nexus `RunBudget` graph env cap | Partial | Per-agent ACP-TOK enforcement **Done**; graph-level cap deferred |
| ROSTER-PROD | P2 | Individual agent `production_mode` promotion | Ongoing | §40.15 thresholds per agent; platform gates **Done** |
| FAUDIT-REG.1 | P2 | Extend `HarnessRegistrySnapshot` with eval registry depth | Planned | `PLATFORM_FOUNDATION` master register |

---

### 6.1l Harness implementation queue — registry architecture closeout (closed)

**Purpose:** Single ordered list for **Phase REG** (Band 2r). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REG-DOC.1** | Docs | **Done** | Appendix O + cross-refs | Author map complete |
| 2 | **REG-1** | Code | **Done** | `HarnessRegistrySnapshot` + `registry_wiring` | `test_registry_wiring.py` |
| 3 | **REG-2** | Code | **Done** | `registry_assembly_resolver` wire | `test_registry_wiring.py` |
| 4 | **REG-3** | CI | **Done** | `check_harness_registry_resolution.py` | CI green |

**Suggested PR order (complete):** REG-DOC.1 → REG-1 → REG-2 → REG-3.### 6.1m Harness implementation queue — capability graph closeout (closed)

**Purpose:** Single ordered list for **Phase CG** (Band 2s). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CG-DOC.1** | Docs | **Done** | Appendix P + cross-refs | Author map complete |
| 2 | **CG-1** | Code | **Done** | `capability_graph_wiring` | `test_capability_graph_wiring.py` |
| 3 | **CG-2** | Code | **Done** | `capability_graph_assembly_resolver` | wire-time validation tests |
| 4 | **CG-3** | CI | **Done** | `check_harness_capability_graph_wiring.py` | CI green |

**Suggested PR order (complete):** CG-DOC.1 → CG-1 → CG-2 → CG-3.

---

### 6.2bj Phase CG execution order (Band 2s — closed 2026-06-02)

**Status:** **Done** · register: [Phase CG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · queue: [§6.1m](#61m-harness-implementation-queue--capability-graph-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CG-DOC.1 | Appendix P + plan sync | High |
| 2 | CG-1 | `capability_graph_wiring` | Critical |
| 3 | CG-2 | `capability_graph_assembly_resolver` | High |
| 4 | CG-3 | `check_harness_capability_graph_wiring.py` | Medium |### 6.2bi Phase REG execution order (Band 2r — closed 2026-06-02)

**Status:** **Done** · register: [Phase REG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · queue: [§6.1l](#61l-harness-implementation-queue--registry-architecture-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REG-DOC.1 | Appendix O + plan sync | High |
| 2 | REG-1 | `HarnessRegistrySnapshot` + `registry_wiring` | Critical |
| 3 | REG-2 | `registry_assembly_resolver` | High |
| 4 | REG-3 | `check_harness_registry_resolution.py` | Medium |

---

### 6.2bg Phase AS execution order (Band 2q — closed 2026-06-02)

**Status:** **Done** · register: [Phase AS](plan/ORCHESTRATION.md) · queue: [§6.1k](#61k-harness-implementation-queue--agent-assembly-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | AS-DOC.1 | Appendix N + plan sync | High |
| 2 | AS-1 | `agent_assembly_resolver` | Critical |
| 3 | AS-2 | Lifecycle state on `AgentContract` | High |
| 4 | AS-3 | `skill_ids` resolution audit script | Medium |### 6.2bh Phase CLEAN execution order (closed 2026-06-02)

**Status:** **Done** · register: [Phase CLEAN](plan/ORCHESTRATION.md) · queue: [§6.1j](#61j-harness-implementation-queue--legacy-module-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CLEAN-1 | Remove `chat_router.py` | Critical |
| 2 | CLEAN-2 | Remove `tools_agent.py` | Critical |
| 3 | CLEAN-3 | `check_legacy_modules_removed.py` in CI | High |
| 4 | CLEAN-4 | Docs sync | Low |

---

## Phase AS — Agent assembly control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (AS-DOC.1 + AS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §18; ideal model §17 in [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix N**.

**Priority ladder:** **Band 2q** (§4.0) — closed; default queue = **§6.1** maintenance.

### AS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| AS-DOC.1 | AS0 | **Appendix N** — agent assembly control plane (contract, capabilities, skills, lifecycle) | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| AS-1 | AS1 | **`agent_assembly_resolver`** — contract metadata validation at register time | **Done** | `runtime/registry/agent_assembly_resolver.py`, `agent_registry.py` | `test_agent_assembly_resolver.py` |
| AS-2 | AS2 | **Lifecycle metadata enforcement** — `production_eligible` owner/runbook requirements | **Done** | `agent_assembly_resolver.py`, `agent_routing_policy.py` | resolver + routing tests |
| AS-3 | AS3 | **`skill_ids` → `allowed_tools` resolution audit** — CI script + docs cross-ref | **Done** | `scripts/check_agent_skill_resolution.py`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), Legal domain steps, product-only contract variants — [§6.3a](#63a-business-backlog-register-consolidated).

---

---

## Phase PE — Prompt registry control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (PE-DOC.* + PE-1–3); gate **623 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §17; V-REM-PE.1/PE.2 governance schema (**Done**); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix M**.

**Priority ladder:** **Band 2p** (§4.0) — closed; default queue = **§6.1** maintenance.

### PE — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| PE-1 | PE1 | **`PromptProfile`** + `prompt_runtime_bridge` — `catalog_path` → `RuntimeConfig.prompt_catalog_path` | **Done** | `environment_profile.py`, `prompt_runtime_bridge.py`, `config.py` | `test_prompt_runtime_bridge.py` |
| PE-2 | PE2 | **`prompt_wiring`** — `resolve_prompt_registry()`, `PromptRegistryProtocol` | **Done** | `prompt_wiring.py`, `prompt_registry_protocol.py` | `test_prompt_wiring.py` |
| PE-3 | PE3 | **Environment wire** — `materialize_runtime_config`, `build_runtime_context_from_environment`, `ApplicationBuildContext.prompt_registry` | **Done** | `runtime_config_bridge.py`, `environment_wiring.py`, `runtime_context.py` | wiring tests + gate |
| PE-4 | PE4 | **Nexus injection** — `prompt_registry_resolver`; `tools_step`, `tool_planning_prompts`, `engine_plan_models`, `nexus_llm_plan_builder` use `RuntimeContext.prompt_registry` | **Done** | `prompt_registry_resolver.py`, nexus/tools + nexus_llm_plan_builder | `test_tools_step_prompt_registry.py` |
| PE-DOC.1 | PE0 | **Appendix M** — prompt registry control plane (§M.1–M.6) | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |

**Residual:** none on Tier-3 host build path. Legacy YAML prompt assets (`chat_router*`, `tools_agent_*`) remain as catalog files only.

---

---

## Phase REG — Registry architecture control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REG-DOC.1 + REG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §19; capability graph V-CG **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix O**.

**Priority ladder:** **Band 2r** (§4.0) — closed; default queue = **§6.1** maintenance.

### REG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REG-DOC.1 | REG0 | **Appendix O** — registry architecture control plane | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REG-1 | REG1 | **`HarnessRegistrySnapshot`** + `registry_wiring` + `RegistrySnapshotProtocol` | **Done** | `registry_snapshot.py`, `registry_wiring.py` | `test_registry_wiring.py` |
| REG-2 | REG2 | **`registry_assembly_resolver`** — profile ↔ registry conformance at wire time | **Done** | `registry_assembly_resolver.py`, `environment_wiring.py` | `test_registry_wiring.py` |
| REG-3 | REG3 | **Host registry resolution CI** — `check_harness_registry_resolution.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), marketplace UI, Band 3 product hosts — [§6.3a](#63a-business-backlog-register-consolidated).

---

---

## Phase CG — Capability graph control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CG-DOC.1 + CG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §20; Phase V-CG **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix P**.

**Priority ladder:** **Band 2s** (§4.0) — closed; default queue = **§6.1** maintenance.

### CG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CG-DOC.1 | CG0 | **Appendix P** — capability graph control plane | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CG-1 | CG1 | **`capability_graph_wiring`** — environment subgraph from catalog + registry snapshot | **Done** | `capability_graph_wiring.py`, `capability_graph_protocol.py` | `test_capability_graph_wiring.py` |
| CG-2 | CG2 | **`capability_graph_assembly_resolver`** — wire-time catalog node validation | **Done** | `capability_graph_assembly_resolver.py`, `environment_wiring.py` | `test_capability_graph_wiring.py` |
| CG-3 | CG3 | **Host capability graph CI** — `check_harness_capability_graph_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only graph nodes — [§6.3a](#63a-business-backlog-register-consolidated).

---

---

### Phase L — Agent OS Certification

**Directive:** L1 certification recorded in Appendix A. K.1/K.2 are **Phase K product work** — **last** in the plan (§6.3), not concurrent with harness bands 1–2.  
**Agent workflow:** [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md)

| # | Deliverable | Status | Req | Notes |
|---|-------------|--------|-----|-------|
| L.1 | UAEP-first agent scaffold | **Done** | R2 | `python -m intergrax.scaffold new-agent` |
| L.2 | Agent creation guide | **Done** | R2 | Single canonical how-to |
| L.3 | Lab application (Tier-3) | **Done** | R1 | `applications/lab_application/` |
| L.4 | Reference technical agents | **Done** | R5 | Echo + `agents/lab/mock_agents.py` |
| L.5 | Agent OS acceptance suite | **Done** | R1 | `tests/acceptance/agent_os/` (+ `05b` mid-step UAEP) |
| L.6 | Runtime independence verification | **Done** | R5 | Register + run without Nexus edits |
| L.7 | Application composition verification | **Done** | R5 | Agents ≠ applications |
| L.8 | Certification checklist | **Done** | R1 | Appendix A (this file) |
| L.9 | **Sign-off exercise** | **Done** | — | `agents/signoff_probe/` — Appendix A record |

**Acceptance tests (L.5):**

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
```

| # | Scenario | Test |
|---|----------|------|
| 1 | Single agent | `test_acceptance_01_single_agent_execution` |
| 2 | Sequential multi-agent | `test_acceptance_02_sequential_multi_agent` |
| 3 | Parallel multi-agent | `test_acceptance_03_parallel_multi_agent` |
| 4 | HITL approve/resume | `test_acceptance_04_human_approval_flow` |
| 5 | Checkpoint recovery | `test_acceptance_05_checkpoint_recovery` · `test_acceptance_05c` · `test_acceptance_05d` |
| 6 | Retry / alternate agent | `test_acceptance_06_retry_flow` |
| 7 | Partial results | `test_acceptance_07_partial_results` |
| 8 | Memory / shared context | `test_acceptance_08_memory_handoff` |
| 9 | Sandbox tools | `test_acceptance_09_sandbox_tool_execution` |
| 10 | Shadow workspace | `test_acceptance_10_shadow_workspace` |

---

---

#### V-ALG — Agent Lifecycle Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-ALG.1 | Agent certification gate contract (quality/policy/security) | **Done** | **Critical** | Certification criteria codified + tested |
| V-ALG.2 | Promotion flow (dev -> staging -> production) with evidence | **Done** | High | Promotion requires evidence bundle |
| V-ALG.3 | Deprecation + retirement workflow and migration window policy | **Done** | High | `AgentRegistry` / `AgentRouter` filter retired/deprecated via `agent_routing_policy.py` |
| V-ALG.4 | Owner/on-call metadata required for production-eligible agents | **Done** | High | Production-mode ownership gate enforced at selection |#### V-CE — Context Quality and Regression Hardening

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-CE.1 | Relevance/freshness/confidence scoring in context assembly | **Done** | High | Scores emitted in trace/runtime events |
| V-CE.2 | Duplicate suppression + context quality thresholds | **Done** | Medium | Threshold policy test coverage |
| V-CE.3 | Context regression benchmark suite | **Done** | High | CI regression baseline stored and compared |
| V-CE.4 | Retrieval effectiveness evaluation (precision/recall@k style) | **Done** | Medium | Bench report in evaluation registry |

---

#### V-PE — Prompt Engineering Architecture

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-PE.1 | Prompt registry governance contract (owner/version/risk metadata) | **Done** | High | `PromptMeta` extended; `harness_capability_summary` reference prompt; registry governance validation |
| V-PE.2 | Prompt composition model (system/task/policy/context layers) | **Done** | High | Canon + reference implementation path |
| V-PE.3 | Deterministic policy injection overlays | **Done** | High | Prompt build trace shows overlays |
| V-PE.4 | Prompt regression/adversarial test suite | **Done** | Medium | Gate includes prompt regression subset |#### V-EVAL — Evaluation and Benchmarking Operations

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-EVAL.1 | Unified evaluation modes: offline/online/shadow/human | **Done** | **Critical** | Mode contracts documented + wired |
| V-EVAL.2 | Golden datasets + scenario libraries + regression suites | **Done** (typed asset bundle contracts) | High | Versioned benchmark assets |
| V-EVAL.3 | Automated evaluators (rule-based + LLM judge) | **Done** | High | Evaluator outputs persisted |
| V-EVAL.4 | Evaluation registry trend/comparison reports | **Done** | High | Report artifact required for major releases |

---

## Phase ACP-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — formal Full Harness LC closeout; no open P0/P1 in domain scope  
**Prerequisites:** Phase ACP + ACP-CLOSE + ACP-FINISH + AUDIT-IDEAL **Done**  
**Goal:** Reconcile audit prompt + ACP-INV-02 canon; confirm gates green; journal + progress tracker  
**ADR:** **No ADR needed** — documentation and process closeout only

| ID | Deliverable | Status | Priority | Module | Acceptance |
|----|-------------|--------|----------|--------|------------|
| ACP-LC-S1 | **Audit prompt sync** — AUDIT-IDEAL rows Done in known gaps | **Done** | High | `docs/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` | No Planned drift |
| ACP-LC-S2 | **ACP-INV-02 canon** — remove stale „until ACP-LEG” wording | **Done** | High | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §21 | Matches ACP-CLOSE-LEG-5 |
| ACP-LC-S3 | **Gate verification** — `check_agent_acp_close_ci.py` green | **Done** | High | `scripts/check_agent_acp_close_ci.py` | Fleet 17/17 · migration complete |
| ACP-LC-S4 | **Full Harness LC journal** + `layer_completion_progress.json` | **Done** | High | implementation-journal | mature status |

**Deferred P2 (not blocking LC):** `boundary_demo` ReflexAgent migration · COST-1 graph RunBudget cap · FAUDIT-REG.1

### 6.1av Harness implementation queue — Agent contracts audit maintenance (closed)

**Source:** Layer 6 audit (2026-06-18) — `AGENT_CONTRACTS_AND_ASSEMBLY` layers 17–20, 31 · [`../audit_results/2026-06-18/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../audit_results/2026-06-18/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Priority ladder:** **Band 1** (§6.1) — fleet hygiene + CI bundle alignment; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **ACP-MAINT-01** | Code | P2 | **Done** | Migrate `boundary_demo` off author-time `allowed_tools` — `skill_ids`/`extra_tools` + ReflexAgent path per AS-3 | `check_agent_skill_resolution.py` green; partner PoC behavior preserved |
| 2 | **ACP-MAINT-02** | CI | P2 | **Done** | Include `check_agent_skill_resolution.py` in `check_agent_acp_close_ci.py` umbrella | ACP close CI fails on AS-3 violations fleet-wide |
| 3 | **ACP-MAINT-03** | Docs | P3 | **Done** | Sync audit prompt known gaps — AUDIT-IDEAL-19.1/20.1/31.1 **Done** vs stale Planned wording | `docs/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` matches plan LC closeout |

**Suggested PR order:** none — §6.1av queue closed (2026-06-18).

**Cross-domain (not ACP-owned):** COST-1 graph `RunBudget` cap — [`plan/UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) · FAUDIT-REG.1 — [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

### 6.1ay Harness implementation queue — Agent contracts audit maintenance (2026-06-19)

**Source:** Interactive layer audit (2026-06-19) — `AGENT_CONTRACTS_AND_ASSEMBLY` layers 17–20, 31 · [`../audit_results/2026-06-19/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../audit_results/2026-06-19/AGENT_CONTRACTS_AND_ASSEMBLY.md) · prior: [`../audit_results/2026-06-18/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../audit_results/2026-06-18/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Priority ladder:** **Band 1** (§6.1) — doc sync + audit artifact; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **ACP-MAINT-DOC-01** | Docs | P3 | **Done** | Close §6.1av header; add architecture §28.3 revalidation note (fleet 17/17, ACP close CI) | Canon matches 2026-06-19 audit |
| 2 | **ACP-MAINT-DOC-02** | Docs | P3 | **Done** | Sync `docs/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` known gaps — AUDIT-IDEAL-19.1/20.1/31.1 **Done** | Lines 54/58 without stale „Planned” |
| 3 | **ACP-MAINT-AUDIT-01** | Docs | P3 | **Done** | Persist Mode A2 audit result under `docs/audit_results/2026-06-19/` | `AGENT_CONTRACTS_AND_ASSEMBLY.md` + `progress.json`; L3+ verdict |

**Suggested PR order:** none — §6.1ay queue closed (2026-06-19).

**Cross-domain (not ACP-owned):** COST-1 graph `RunBudget` cap · FAUDIT-REG.1 — unchanged deferred.

---

*End of Agent Contracts and Assembly Implementation Plan.*
