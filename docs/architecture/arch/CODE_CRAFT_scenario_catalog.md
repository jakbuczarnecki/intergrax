# CODE_CRAFT — §11+ scenarios & control

**Parent hub:** [`CODE_CRAFT.md`](../CODE_CRAFT.md)

## 11. Cross-domain integration matrix

| Domain | Relationship |
|--------|--------------|
| **TOOLS** | Registers `codecraft.*`; retains `code.exec` as substrate |
| **SKILLS** | `codecraft.ephemeral_builder` bundle composes tool_ids |
| **INTEGRATIONS** | `sandbox_host`, `security_scanner` |
| **LLM_ADAPTERS** | Separate codegen profile |
| **UNIFIED_EXECUTION_RUNTIME** | Policy bundle, UAEP hooks, §42.12 ToolRuntime |
| **CRITIC_VERIFICATION** | L0/L1 verdicts, evaluator-loop pattern reuse |
| **RELIABILITY_FAILURE_AND_HITL** | Sandbox/shadow substrate, cleanup, HITL |
| **OBSERVABILITY** | Event spine |
| **NEXUS_EXECUTION_FLOW** | Optional `CodeCraftNode` in graph specs |
| **ORCHESTRATION** | Subtask / branch delegation |
| **TIER3_APPLICATION_ENVIRONMENT** | `CodeCraftProfile` wiring |
| **ADAPTIVE_HARNESS_INTELLIGENCE** | L4 trigger: when to invoke craft (ECC-6) |

---

## 12. Current state vs target (summary)

| Capability | Status (2026-06-13) | Notes |
|------------|----------------------|-------|
| Isolated exec | **Done** — `runtime/sandbox/`, `code.exec` | Substrate reused |
| Cloud sandbox | **Done** — `HostedSandboxSession`, e2b/modal/daytona | ECC-4 default for regulated hosts |
| Tool policy | **Done** — `SANDBOX_REQUIRED_TOOLS`, UAEP gateway | Extended for `codecraft.*` |
| Harness craft loop | **Done** — `CodeCraftOrchestrator` | ECC-2 |
| Ephemeral tool registry | **Done** — task-scoped registry | ECC-5 |
| CodeCraftProfile | **Done** — Tier-3 profile | ECC-3 |
| Static code gate | **Done** — `StaticCodeGate` | ECC-1 |
| Promotion contract | **Done** — `CraftResult` | ECC-3 |
| CODECRAFT trace events | **Done** — full §10.1 taxonomy | S8 |
| Graph node | **Done** — optional `CodeCraftNode` | ECC-5 |
| AHI adaptive trigger | **Done** — `adaptive_trigger.py` | ECC-6 |
| Metrics dashboards | **Done** — §10.2 trace panel via `codecraft.metrics_snapshot` | ECC-MAINT-04 |

**Maturity:** **L3+** — ECC-0…ECC-6 + post-closeout S7–S11 (2026-06-13). Depth backlog GAP-ECC-20…23 **closed** (ECC-MAINT-01..04, 2026-06-18).

**Audit revalidation (2026-06-19, ECC-MAINT-DOC-01):** §6.1av confirmed · 31 unit tests + `check_codecraft_layer.py` green · no open P0/P1.

---

## 13. Anti-patterns

| Anti-pattern | Why forbidden |
|--------------|---------------|
| Craft loop in Tier-2 agent | Bypasses policy, untestable, duplicates Harness |
| Register ephemeral tools in global ToolRegistry | Catalog pollution |
| New parallel sandbox runtime | Violates §5.2 reuse |
| Direct `subprocess` in agents | Violates UAEP §42.12 |
| ECC without sandbox session | Fail-closed — no host exec fallback |
| Treating local sandbox as production security | Misleading isolation guarantee |

---

## 14. Related documentation

| Document | Role |
|----------|------|
| [`plan/CODE_CRAFT.md`](../plan/CODE_CRAFT.md) | Full audit register + ECC-0…ECC-6 implementation phases |
| [`TOOLS.md`](TOOLS.md) | Catalog primitives `code.exec`, `sandbox.exec` |
| [`SKILLS.md`](SKILLS.md) | `sandbox.code_exec`, `sandbox.refactor_loop` compositions |
| [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) | Sandbox/shadow isolation |
| [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) | Verification loop reuse |
| [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix B | Shadow workspace + sandbox task flags |

---

## 15. Implementation status

| Phase | Scope | Status |
|-------|-------|--------|
| ECC-0 | ADR + domain pair docs | **Done** (2026-06-10) |
| ECC-1 | `codecraft.run` + static gate + trace | **Done** (2026-06-13) |
| ECC-2 | Session loop `start/iterate/dispose` | **Done** (2026-06-13) |
| ECC-3 | Modes + HITL + promotion | **Done** (2026-06-13) |
| ECC-4 | Cloud sandbox default + security.scan | **Done** (2026-06-13) |
| ECC-5 | Ephemeral tool registry + graph node | **Done** (2026-06-13) |
| ECC-6 | AHI adaptive trigger | **Done** (2026-06-13) |
| ECC-7 | Trace taxonomy parity (generation/test/verdict/HITL/promote) | **Done** (2026-06-13, S8) |
| ECC-8 | Single-shot sandbox routing parity on `codecraft.run` | **Done** (2026-06-13, S9) |
| ECC-9 | `health.check_codecraft` probe + `check_codecraft_layer.py` gate | **Done** (2026-06-13, S10) |

Detail: [`plan/CODE_CRAFT.md`](../plan/CODE_CRAFT.md).

---

## 16. Implementation module map

Tier-0 contracts and gate (ECC-1+):

| Module | Responsibility |
|--------|----------------|
| `intergrax/codecraft/contracts.py` | `CraftResult`, `StaticGateResult`, session I/O models |
| `intergrax/codecraft/profile.py` | `CodeCraftProfile`, `CraftMode` |
| `intergrax/codecraft/static_gate.py` | `StaticCodeGate` — AST, imports, size, forbidden patterns |
| `intergrax/codecraft/codegen_adapter.py` | LLM code generation (ECC-2) |
| `intergrax/codecraft/test_runner.py` | `CraftTestRunner` (ECC-2) |
| `intergrax/codecraft/promoter.py` | `CraftResultPromoter` (ECC-3) |

Tier-1 runtime orchestration:

| Module | Responsibility |
|--------|----------------|
| `intergrax/runtime/codecraft/orchestrator.py` | `CodeCraftOrchestrator` (ECC-2) |
| `intergrax/runtime/codecraft/session_manager.py` | `craft_id` lifecycle (ECC-2) |
| `intergrax/runtime/codecraft/sandbox_resolver.py` | `isolation_tier` routing (ECC-4) |
| `intergrax/runtime/codecraft/cv_bridge.py` | CVL iteration verdict (ECC-2) |
| `intergrax/runtime/codecraft/adaptive_trigger.py` | AHI catalog-miss trigger (ECC-6) |
| `intergrax/runtime/codecraft/trace.py` | `CodeCraftTraceEmitter`, `CODECRAFT_*` diagnostic steps |
| `intergrax/runtime/codecraft/ephemeral_registry.py` | Task-scoped tools (ECC-5) |

Tool surface (`ToolRuntime`):

| Module | Responsibility |
|--------|----------------|
| `intergrax/tools/providers/codecraft/bundle.py` | Register `codecraft.*` catalog tools |
| `intergrax/tools/providers/codecraft/service.py` | Handler services — gate + sandbox delegate |

Tier-3 wiring (ECC-3):

| Module | Responsibility |
|--------|----------------|
| `intergrax/applications/_shared/codecraft_wiring.py` | `wire_application_codecraft()` |
| `ApplicationEnvironmentProfile.codecraft_profile` | Host profile field |

**Trace integration:** `TraceComponent.CODECRAFT` diagnostic steps (`codecraft.session_opened`, `codecraft.static_gate`, …) — correlated via `craft_id`, `sandbox_session_id`, `run_id`. Full `RuntimeEventType` enum extension deferred until observability spine unifies tool-domain events.

**Policy:** `codecraft.*` tools added to `SANDBOX_REQUIRED_TOOLS` (same routing as `code.exec`). Fail-closed when `codecraft_profile` missing, `mode=disabled`, or sandbox session absent.

**CI gate:** `scripts/check_codecraft_layer.py` — catalog tools, wiring, trace steps, health probe registration (ECC-9).
