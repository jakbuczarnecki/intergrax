# REASONING_PLANNING — Platform Audit

## Metadata

- **campaign_id:** `2026-08-18`
- **campaign_started_at:** `2026-08-18`
- **Layer code:** REASONING_PLANNING
- **Tier(s):** Tier-1 Nexus planning · Tier-1 tool planning · Tier-2 cognitive patterns
- **layer_audited_at:** 2026-08-19
- **audited_sha:** `fe876d301df07ce22e438b0a55167275ccec32b5`
- **Status:** COMPLETE
- **Auditor:** independent ChatGPT platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-19
- **Architecture doc(s):**
  - `docs/project/architecture/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/architecture/REASONING_AND_COGNITION.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/maintainers/plans/REASONING_AND_COGNITION.md`
- **Scope in:**
  - Nexus plan structural integrity before PLAN_CREATED
  - planning vs execution agent eligibility parity
  - replan semantic closure
  - cognitive verdict integrity
  - tool-planning outcome semantics
  - product-shaped logic in generic planner core
- **Scope out:**
  - remediation implementation
  - overstatement of duplicate step-id production reachability without exact evidence
- **Prior audit reference(s):** [`NEXUS_EXECUTION_FLOW`](NEXUS_EXECUTION_FLOW.md) planning narrative; [`REASONING_AND_COGNITION`](../../project/architecture/REASONING_AND_COGNITION.md) cognition canon
- **architecture_sync:** COMPLETE after Commit A
- **plan_sync:** COMPLETE after Commit A
- **post_sync_sha:** `d7988045cfa550c4338eedc326b54933c4058541`

## Executive summary

**Verdict: FAIL.** Six accepted findings (4 HIGH, 2 MEDIUM) show insufficient pre-PLAN_CREATED graph validation, planning without production eligibility filters, broken Nexus replan closure, cognitive patterns that can erase FAIL/HUMAN/REPLAN verdicts, collapsed tool-planning outcomes, and product-specific research decomposition in generic `TaskPlanner`. Positive control: execution plane enforces production routability even when planning does not. No new vendor leak — research hardcoding is product leakage, not vendor leakage.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-REASONING_PLANNING-01

**Nexus accepts structurally invalid plans too early**

- **Severity:** HIGH
- **Category:** RELIABILITY
- **Related classification:** TEST GAP
- **Status at publication:** ACCEPTED
- **Remediation block:** RPL-FIX-A
- **Claim falsified:** Impossible execution graphs are rejected at the planning boundary before PLAN_CREATED/PLANNED.
- **Observation:** `validate_nexus_plan()` checks unknown `depends_on` and unknown `agent_id`; it does not fully validate graph integrity before PLAN_CREATED. Cycle detection occurs later in `ExecutionGraph`. Graph cycle failure is transformed into an `AgentExecutionResult` with empty `agent_id` without marking graph node failed. GraphRunner can follow a path expecting a real final agent. Confirmed cycle case is sufficient evidence.
- **Location:**
  - `intergrax/runtime/nexus/planning/plan_validator.py:L11-L35` — partial validation @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/runtime/nexus/execution/execution_graph.py:L65-L71` — cycle detection @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/runtime/nexus/execution/graph_executor.py:L197-L207` — cycle → failed result with empty agent_id @ `fe876d301df07ce22e438b0a55167275ccec32b5`
- **Reproduction:**
  1. `git show fe876d301df07ce22e438b0a55167275ccec32b5:intergrax/runtime/nexus/planning/plan_validator.py` — no cycle/integrity checks.
  2. `git show fe876d301df07ce22e438b0a55167275ccec32b5:intergrax/runtime/nexus/execution/graph_executor.py` — cycle handled at execution time.
- **Impact:** Invalid plans can reach RUNNING before structural failure surfaces.
- **Confidence:** CONFIRMED

### AUDIT-20260818-REASONING_PLANNING-02

**Planning ignores production agent eligibility**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Related classification:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** RPL-FIX-B
- **Claim falsified:** Planning publishes only agents that are production-routable under the same semantics enforced at execution.
- **Observation:** Registry supports production-aware routability. `AgentRouter`/`AgentEngine` enforce `production_mode`. Deterministic planner uses capability/all-agent lookup without equivalent production filter. LLM planner explicitly uses routable agents with `production_mode=False`. Plan validator checks registered agents, not production-routable agents. Planner can publish plan with agent later rejected by execution plane — execution layer is positive safety counterexample.
- **Location:**
  - `intergrax/runtime/nexus/planning/nexus_plan_bridge.py:L101-L104` — `production_mode=False` @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/runtime/nexus/planning/nexus_llm_plan_builder.py:L31` — `production_mode=False` @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/runtime/nexus/planning/plan_validator.py:L22-L33` — registered agents only @ `fe876d301df07ce22e438b0a55167275ccec32b5`
- **Reproduction:**
  1. `git grep -n "production_mode=False" fe876d301df07ce22e438b0a55167275ccec32b5 -- intergrax/runtime/nexus/planning/`
  2. `git show fe876d301df07ce22e438b0a55167275ccec32b5:intergrax/runtime/nexus/planning/plan_validator.py` — no routability filter.
- **Impact:** Plans can include agents that execution will reject in production mode.
- **Confidence:** CONFIRMED

### AUDIT-20260818-REASONING_PLANNING-03

**`request_replan()` does not result in Nexus replanning**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT
- **Related classification:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** RPL-FIX-C
- **Claim falsified:** Task-level replan intent from agent runtime closes into a typed Nexus replan transition.
- **Observation:** Author helper says Nexus may schedule replanned run. `StepOutcome.replan` is terminal with `next_action REPLAN`. ACP loop stops on terminal. `_terminal_status()` maps terminal non-HITL outcome to SUCCEEDED. `AgentRunResult` → `RuntimeAnswer` bridge does not preserve REPLAN intent. No inspected production consumer closes path back to Nexus planner. Target distinction: LOCAL_REPLAN vs NEXUS_REPLAN_REQUEST.
- **Location:**
  - `intergrax/agents/authoring/decisions.py:L88-L94` — `request_replan()` @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/agents/authoring/acp_run.py:L140-L145` — `_terminal_status` → SUCCEEDED @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/agents/authoring/acp_uaep_shim.py:L199-L203` — REPLAN → MODIFY_PLAN only @ `fe876d301df07ce22e438b0a55167275ccec32b5`
- **Reproduction:**
  1. `git show fe876d301df07ce22e438b0a55167275ccec32b5:intergrax/agents/authoring/decisions.py` — replan outcome helper.
  2. `git show fe876d301df07ce22e438b0a55167275ccec32b5:intergrax/agents/authoring/acp_run.py` — terminal status mapping.
  3. `git grep -n "REPLAN" fe876d301df07ce22e438b0a55167275ccec32b5 -- intergrax/runtime/nexus` — no Nexus replan consumer path established.
- **Impact:** Agents can signal replan intent that never reaches Nexus planning.
- **Confidence:** CONFIRMED

### AUDIT-20260818-REASONING_PLANNING-04

**Framework cognitive patterns can ignore FAIL/HUMAN/REPLAN verdicts**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT
- **Related classification:** RELIABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** RPL-FIX-D
- **Claim falsified:** Framework cognitive patterns preserve authoritative `CognitiveEvaluation` verdicts (COMPLETE/CONTINUE/FAIL/HUMAN/REPLAN).
- **Observation:** Base `CognitiveAgent` has correct canonical verdict mapping. `PlanExecuteAgent` locally handles only COMPLETE vs continue/phase-done. Reflection/plan-execute paths may complete by phase progression instead of preserving non-complete evaluation verdict. Pattern can turn failure/human/replan intent into CONTINUE or COMPLETE.
- **Location:**
  - `intergrax/agents/authoring/patterns/base.py:L89-L115` — canonical `_evaluation_to_outcome` @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/agents/authoring/patterns/plan_execute.py:L27-L46` — phase progression overrides evaluation @ `fe876d301df07ce22e438b0a55167275ccec32b5`
- **Reproduction:**
  1. `git show fe876d301df07ce22e438b0a55167275ccec32b5:intergrax/agents/authoring/patterns/plan_execute.py` — ignores FAIL/HUMAN/REPLAN branches.
  2. Compare with `base.py` canonical mapping.
- **Impact:** Cognitive failure/human/replan intent can be silently erased.
- **Confidence:** CONFIRMED

### AUDIT-20260818-REASONING_PLANNING-05

**Tool planner collapses planning failure and legitimate no-tool choice into the same empty-plan outcome**

- **Severity:** MEDIUM
- **Category:** RELIABILITY
- **Related classification:** OPERABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** RPL-FIX-E
- **Claim falsified:** Tool planning distinguishes parse failure, forbidden tool request, schema failure, legitimate no-tool, and successful tool plan via typed outcomes.
- **Observation:** Prompt-based parse failure → `plan_obj None` → empty `ToolCallPlan`. Forbidden tool request can also → empty `ToolCallPlan`. Consumer maps empty plan to `empty_tool_calls`. Malformed reasoning, forbidden selection, and legitimate no-tool lose distinct semantics. Forbidden tool is not executed — not a policy bypass claim.
- **Location:**
  - `intergrax/runtime/nexus/tools/tool_planning_service.py:L262-L297` — parse/forbidden collapse @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/runtime/nexus/tools/patterns/bounded_react.py:L82` — `empty_tool_calls` stop reason @ `fe876d301df07ce22e438b0a55167275ccec32b5`
- **Reproduction:**
  1. `git show fe876d301df07ce22e438b0a55167275ccec32b5:intergrax/runtime/nexus/tools/tool_planning_service.py` — forbidden tool returns empty plan same as parse failure.
- **Impact:** Operators and downstream recovery cannot distinguish failure modes from intentional no-tool.
- **Confidence:** CONFIRMED

### AUDIT-20260818-REASONING_PLANNING-06

**Generic TaskPlanner contains product-specific research workflow logic**

- **Severity:** MEDIUM
- **Category:** BOUNDARY VIOLATION
- **Related classification:** ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** RPL-FIX-F
- **Claim falsified:** Generic core planning contains no product-specific workflow decomposition; product logic lives in graph/profile/Tier-3 configuration.
- **Observation:** Core `TaskPlanner` knows `research.pipeline`, `research_summarize`, `research.web_search`, `research.summarize`, etc. Architecture already has generic graph/profile/rule mechanisms. Product decomposition belongs in GraphSpec/planner rule/Tier-3 configuration, not generic planner core.
- **Location:**
  - `intergrax/runtime/nexus/planning/task_planner.py:L66-L67` — research pipeline branch @ `fe876d301df07ce22e438b0a55167275ccec32b5`
  - `intergrax/runtime/nexus/planning/task_planner.py:L130-L162` — `_is_research_pipeline` / `_research_pipeline_plan` @ `fe876d301df07ce22e438b0a55167275ccec32b5`
- **Reproduction:**
  1. `git show fe876d301df07ce22e438b0a55167275ccec32b5:intergrax/runtime/nexus/planning/task_planner.py` — hardcoded research capabilities.
- **Impact:** Product workflow leakage into universal planner core increases coupling and drift risk.
- **Confidence:** CONFIRMED

## Provider / backend abstraction

No new vendor leak. Nexus planner, `NexusPlan`, cognitive patterns, and `ToolPlannerProtocol` remain provider-neutral abstractions. Research hardcoding is **PRODUCT LEAKAGE**, not vendor leakage.

## Falsification log

1. **Forbidden tool executes despite denial** — not observed; empty plan only.
2. **Execution plane also ignores production_mode** — execution enforces routability (positive counterexample to RPL-02 impact, not falsification of finding).
3. **Duplicate step-id reachability** — not promoted without exact production evidence.

## Prior-audit comparison

Extends Nexus flow and cognition canon with planning-integrity, eligibility, replan, verdict, and tool-outcome claims. First canonical Protocol v2.2 `REASONING_PLANNING` snapshot.

## Open questions / blocked items

- LOCAL_REPLAN vs NEXUS_REPLAN_REQUEST ownership — planning only (**RPL-FIX-C**).
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-19
- **Accepted findings:** all 6 (`AUDIT-20260818-REASONING_PLANNING-01` … `AUDIT-20260818-REASONING_PLANNING-06`)
- **Remediation blocks:** RPL-FIX-A … RPL-FIX-F — all **ACCEPTED / PLANNED** only
