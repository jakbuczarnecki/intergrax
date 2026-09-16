# INTEGRAX-EXECUTION-R5-APPLICATIONS-CONTRACT-BOUNDARY-COUPLING-AUDIT-AND-REMEDIATION

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-R5-APPLICATIONS-CONTRACT-BOUNDARY-COUPLING-AUDIT-AND-REMEDIATION` |
| Domain | Tier-0/1 layer boundary · contract ownership |
| Change class | Class B — compatible internal ownership correction |

## Scope

Audit and remediate forbidden imports of `intergrax.applications` / `intergrax.applications.contracts` from platform scan roots (`intergrax/runtime`, `intergrax/agents`, `agents/`, `intergrax/contracts`, `intergrax/rag`, `intergrax/tools`, `intergrax/skills`, `intergrax/llm_adapters`) without changing Execution Engine semantics.

## Repository State

Remediation on branch `development`.

## Baseline SHA

`ea6def4f7ba9ae36cb575f79e0c029b0328c8723`

## Original Gate Failure

`test_intergrax_no_applications_import_gate` (ARCH-F17) — script `scripts/maintenance/check_intergrax_no_applications_imports.py` reported six violating modules importing Tier-3 application surfaces from lower layers.

## Full Applications Import Inventory

| Consumer | Imported symbol | Source module | Runtime use | Ownership candidate | Verdict |
| -------- | --------------- | ------------- | ----------- | ------------------- | ------- |
| `intergrax/runtime/nexus/execution/graph_builder.py` | `EvaluatorLoopGraphBinding` | `applications.contracts.graph_spec` | Plan metadata validation for evaluator-loop | `runtime/nexus/execution` | **MOVE** (platform orchestration DTO) |
| `intergrax/runtime/sandbox/resolver.py` | `ApplicationEnvironmentProfile`, `SandboxProfile`, `EffectiveProfileRevision` | `applications.contracts.*` | Sandbox isolation authority | `intergrax/contracts` protocols + `SandboxProfile` | **INVERT** (structural port) |
| `intergrax/runtime/sandbox/enforcement.py` | same | same | Tool wiring profile resolution | same | **INVERT** |
| `agents/model_routing_qualifier/*` | `RoutingEvaluatingLLMAdapter`, observers | `applications._shared.routing_evaluating_adapter` | Qualification routing observation | `llm_adapters/routing` | **MOVE** |
| `intergrax/agents/authoring/acp_uaep_shim.py` | `apply_rag_from_tool_wiring_context` | `applications._shared.rag_runtime_bridge` | Copy RAG managers from tool wiring | `rag/profiles` | **MOVE** |

No imports from `intergrax/contracts` or `intergrax/core` to `intergrax.applications` in production code (scan clean aside from remediated rows).

## Symbol Ownership Classification

| Symbol | Classification | Decision |
| --- | --- | --- |
| `EvaluatorLoopGraphBinding` | D — composition / orchestration model tied to `EvaluatorLoopSpec` | **MOVE** → `intergrax.runtime.nexus.execution.evaluator_loop_graph_binding` |
| `SandboxProfile` | E — shared DTO (session root / exec flag) | **MOVE** → `intergrax.contracts.sandbox_profile` |
| `ApplicationEnvironmentProfile` (runtime sandbox) | B — application contract | **INVERT** — runtime uses `ProfileSandboxIsolationSource` protocol only |
| `EffectiveProfileRevision` (runtime sandbox) | B — application contract | **INVERT** — runtime uses `EffectiveProfileRevisionIsolationView` protocol |
| `RoutingEvaluatingLLMAdapter` | A — platform LLM routing port | **MOVE** → `intergrax.llm_adapters.routing.evaluating_adapter` |
| `apply_rag_from_tool_wiring_context` | C — transport/wiring sync | **MOVE** → `intergrax.rag.profiles.tool_wiring_runtime_sync` |

## Runtime Consumers

Runtime sandbox and graph builder no longer import `intergrax.applications`. Structural typing via `intergrax.contracts.execution_environment_isolation` preserves pinned-revision and legacy profile extras in `ToolWiringContext`.

## Agent Consumers

Tier-2 qualification agents import platform routing adapter from `llm_adapters`. `intergrax/agents/authoring/acp_uaep_shim` imports neutral RAG wiring sync from `intergrax.rag`.

## Public API Analysis

- `intergrax.applications.contracts.graph_spec` re-exports `EvaluatorLoopGraphBinding` from runtime (shim).
- `intergrax.applications.contracts.environment_profile` still exports `SandboxProfile` via `sub_profiles` re-export from `intergrax.contracts.sandbox_profile`.
- `intergrax.applications._shared.routing_evaluating_adapter` remains Tier-3 composition shim (`env=` + default adapter factory).
- `intergrax.applications._shared.rag_runtime_bridge` re-exports `apply_rag_from_tool_wiring_context` from `intergrax.rag`.

## Plugin Compatibility Analysis

No external plugin API paths removed. Old import paths for graph binding, sandbox profile, routing adapter, and RAG wiring remain available through thin re-export shims where previously public.

## Current Dependency Graph

```text
runtime/sandbox ──→ applications.contracts (environment_profile, profile_resolution)
runtime/graph_builder ──→ applications.contracts.graph_spec
agents (qualifier) ──→ applications._shared.routing_evaluating_adapter
intergrax/agents ──→ applications._shared.rag_runtime_bridge
```

## Target Dependency Graph

```text
runtime/sandbox ──→ intergrax.contracts (isolation protocols, SandboxProfile)
runtime/graph_builder ──→ runtime.nexus.execution.evaluator_loop_graph_binding
agents ──→ llm_adapters.routing.evaluating_adapter
intergrax/agents ──→ rag.profiles.tool_wiring_runtime_sync
applications ──→ (shim re-exports only) ──→ canonical platform owners
```

## Selected Remediation

Move misplaced neutral symbols to runtime / contracts / llm_adapters / rag; replace concrete application profile types in runtime sandbox with structural protocols; keep Tier-3 shims for compatibility.

## Contracts / Ports

- `ProfileSandboxIsolationSource` — `.sandbox` fragment for isolation authority.
- `EffectiveProfileRevisionIsolationView` — `.effective_profile` for pinned revisions.
- `RoutingProfileSource` — `.llm_routing_profile` for evaluating adapter.

## Compatibility Shims

| Path | Canonical owner |
| --- | --- |
| `applications.contracts.graph_spec.EvaluatorLoopGraphBinding` | `runtime.nexus.execution.evaluator_loop_graph_binding` |
| `applications.contracts.environment_profile.SandboxProfile` | `intergrax.contracts.sandbox_profile` |
| `applications._shared.routing_evaluating_adapter` | wraps `llm_adapters.routing.evaluating_adapter` |
| `applications._shared.rag_runtime_bridge.apply_rag_from_tool_wiring_context` | `rag.profiles.tool_wiring_runtime_sync` |

## Layer Boundary Assessment

Scan roots: **0** `intergrax.applications` imports. Direction restored: applications → platform contracts/runtime; not inverse.

## Pluginability Assessment

Runtime sandbox resolution accepts any object satisfying isolation protocols; application adapters continue binding concrete `EffectiveProfileRevision` / `ApplicationEnvironmentProfile` into tool wiring extras at composition time.

## Execution Engine Impact

**None** — no changes to `ExecutionRuntime`, `ExecutionBoundary`, `StrategyExecutionRouter`, governance, identity, recovery, or evidence control paths.

## Architecture Gate Result

`test_intergrax_no_applications_import_gate`: **PASS** (`check_intergrax_no_applications_imports.py` exit 0).

## Known-Good Gates

| Gate | Result |
| --- | --- |
| `test_ee_final_arch_*` (10 modules) | PASS |
| U5 zero-bypass | PASS |
| UE-10R4.1 | PASS |
| OBS-DIAG conformance architecture | PASS |
| F-01 (EE-B2 fault matrix registry) | unchanged / PASS via EE-B2 suite membership |

## Qualification Regression

`tests/unit/testing_support/execution_qualification/`: **216 passed**, 7 skipped (live perf env).

## Static Quality

Ruff format/check on changed production files; pyright on sandbox + evaluating adapter + `sandbox_profile`: **0 errors**.

## Changed Files

Production / agents (R5 scope):

- `intergrax/contracts/sandbox_profile.py` (new)
- `intergrax/contracts/execution_environment_isolation.py` (new)
- `intergrax/runtime/nexus/execution/evaluator_loop_graph_binding.py` (new)
- `intergrax/rag/profiles/tool_wiring_runtime_sync.py` (new)
- `intergrax/llm_adapters/routing/evaluating_adapter.py` (new)
- `intergrax/runtime/sandbox/enforcement.py`, `resolver.py`
- `intergrax/runtime/nexus/execution/graph_builder.py`
- `intergrax/applications/contracts/graph_spec.py`
- `intergrax/applications/contracts/environment_profile/sub_profiles.py`
- `intergrax/applications/_shared/routing_evaluating_adapter.py`
- `intergrax/applications/_shared/rag_runtime_bridge.py`
- `intergrax/agents/authoring/acp_uaep_shim.py`
- `agents/model_routing_qualifier/routing_observation.py`
- `agents/model_routing_qualifier/steps/model_routing_job.py`

## Remaining Debt

- `ApplicationEnvironmentProfile` and full profile-resolution graph remain under `intergrax.applications.contracts` by design (Tier-3 composition); only runtime-facing fragments were neutralized.
- Broader architecture suite debt (R1–R14) outside R5 scope unchanged.

## Decision

Coupling was a **layer violation** (misplaced platform-neutral surfaces under applications). Remediation applied without allowlist changes.

## Commit SHA

_(filled at commit)_

## Final Verdict

**R5 APPLICATIONS CONTRACT BOUNDARY = PASS — COUPLING REMEDIATED**

## Required inventory table

| Consumer | Symbol | Current owner | Correct owner | Decision |
| -------- | ------ | ------------- | ------------- | -------- |
| runtime/graph_builder | `EvaluatorLoopGraphBinding` | applications.contracts.graph_spec | runtime.nexus.execution | MOVE + shim |
| runtime/sandbox | `SandboxProfile` | applications.contracts | intergrax.contracts.sandbox_profile | MOVE + re-export |
| runtime/sandbox | profile / revision types | applications.contracts | contracts isolation protocols | INVERT |
| agents qualifier | `RoutingEvaluatingLLMAdapter` | applications._shared | llm_adapters.routing | MOVE + Tier-3 shim |
| intergrax/agents | `apply_rag_from_tool_wiring_context` | applications._shared | rag.profiles | MOVE + re-export |
