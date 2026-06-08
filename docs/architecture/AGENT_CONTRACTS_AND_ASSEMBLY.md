# Agent Contracts, Registry, and Capability Model

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 17–20, 31  
---

---

# 12. Agent Contract

Every agent MUST implement a clear contract.

The contract should be easy for humans and LLMs to understand.

Minimum required fields:

```text
AgentContract:
    id
    name
    description
    version
    capabilities
    input_schema
    output_schema
    allowed_tools
    required_adapters
    execution_mode
    max_steps
    max_duration
    max_cost
    risk_level
    validation_rules
    failure_modes
```

---


---

# 13. Suggested Agent Interface

This is conceptual pseudocode, not a required programming language implementation.

```text
interface Agent:

    get_contract() -> AgentContract

    can_handle(task_context) -> CapabilityMatchResult

    execute(agent_input, execution_context) -> AgentExecutionResult

    validate(agent_output, execution_context) -> ValidationResult
```

Agent implementations should be simple.

The goal is to let developers focus on domain logic, not infrastructure.

All `execute()` implementations MUST delegate to `AgentEngine` and the Unified Agent Execution Protocol (§42.5). Agents MUST NOT implement private runtime lifecycles.

---


---

# 14. Agent Execution Result

Every agent should return a structured result.

Recommended structure:

```text
AgentExecutionResult:
    agent_id
    run_id
    status
    summary
    artifacts
    structured_data
    evidence
    confidence
    warnings
    errors
    used_tools
    cost
    duration
    next_recommendations
```

The result must be inspectable by Nexus and by humans.

---


---

# 15. Agent Registry

Nexus discovers agents through the Agent Registry.

The registry stores:

- agent id
- name
- description
- version
- capabilities
- required adapters
- allowed tools
- execution modes
- cost profile
- risk profile
- status

Nexus MUST use the registry for agent selection.

Agents MUST NOT be hardcoded into Nexus logic unless explicitly needed for a minimal prototype.

Even in prototypes, hardcoded agents should be treated as temporary.

---


---

# 16. Capability Model

A capability describes what an agent can do.

Examples:

```text
capability: vendor.discovery
capability: vendor.scoring
capability: legal.contract_review
capability: research.web_search
capability: problem_radar.source_monitoring
capability: problem_radar.clustering
capability: onboarding.daily_guidance
```

Nexus should route tasks to capabilities, not only to specific class names.

This allows agents to be replaced later.

---


---

# 45. Checklist For New Agent Implementation

Before implementing a new agent, answer:

```text
1. What hypothesis does this agent test?
2. What capability does it provide?
3. What input does it require?
4. What structured output does it produce?
5. What tools/adapters does it need?
6. What is the validation rule?
7. What are failure modes?
8. What is the maximum acceptable cost/time?
9. How will success be evaluated?
10. How will Nexus route tasks to this agent?
11. Which AgentSteps does the agent declare (§42.6)?
12. Which AgentDecision types can the agent emit (§42.7)?
13. Does the agent conform to UAEP via AgentEngine (§42.5)?
14. Are all tool calls routed through ToolRuntime (§42.12)?
15. Are forbidden runtime patterns avoided (§42.41)?
```

If these questions cannot be answered, do not implement the agent yet.

---

---

# 17. Prompt Registry Architecture

Prompt artifacts are **governed platform assets**, not ad-hoc strings in agents.

## 17.1 Requirements

- ownership and versioning on every prompt id (`PromptMeta`),
- composable layers: system / task / policy / context,
- deterministic policy injection overlays,
- regression suites on golden prompt catalogs,
- Tier-3 `PromptProfile` selects YAML catalog path per host.

## 17.2 Code map

| Module | Role |
|--------|------|
| `intergrax/prompts/registry/` | YamlPromptRegistry, governance validation |
| `intergrax/runtime/architecture/prompt_registry_governance.py` | Ownership / risk tier gates |
| `intergrax/runtime/architecture/prompt_composition.py` | Layer composition |
| `intergrax/runtime/architecture/prompt_policy_overlay.py` | Policy overlays |
| `intergrax/runtime/architecture/prompt_regression_suite.py` | Golden regression |
| `intergrax/applications/_shared/prompt_wiring.py` | Environment → Nexus prompt registry |

**Authoring:** [`guides/AGENT_CREATION_GUIDE.md` Appendix M](../guides/AGENT_CREATION_GUIDE.md) · **Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase PE.

---

# 18. Registry Architecture

Registries are versioned, snapshot-capable catalogs — not mutable globals.

## 18.1 Registry types

| Registry | Tier | Consumed by |
|----------|------|-------------|
| Agent registry | 1 | Nexus agent selection |
| Tool registry | 0 | `ToolRuntime` |
| Skill registry | 0 | Skill resolver |
| Integration registry | 0 | Provider hosts |
| Prompt registry | 0/1 | Nexus steps, eval |
| Evaluation registry | 1 | EvalRunner, release gates |

## 18.2 Assembly pattern

Tier-3 `wire_application_environment()` materializes registries from `ApplicationEnvironmentProfile` tool/skill/integration/prompt profiles → `RuntimeConfig` via `runtime_config_bridge.py` and domain `*_assembly_resolver.py` modules.

Snapshots and conformance CI validate registry shape before release (`scripts/check_agents_lifecycle_metadata.py`, harness registry guards).

**Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase REG.

---

# 19. Capability Graph Architecture

Registries and capability layers MUST be represented as a typed dependency graph:

```text
Integration -> Tool -> Skill -> Policy -> Agent -> Application -> Product
```

## 19.1 Minimum requirements

- typed node and edge taxonomy,
- dependency lineage and provenance,
- blast-radius impact analysis for version/policy/runtime changes,
- compatibility validation on graph edges before release.

## 19.2 Code map

| Module | Role |
|--------|------|
| `runtime/architecture/capability_graph.py` | Core graph model |
| `capability_graph_lineage.py` | Lineage / provenance |
| `capability_graph_compatibility.py` | Edge compatibility |
| `capability_graph_applications.py` | Application slice |
| `scripts/phase_v_capability_graph_guard.py` | CI guard |

Nexus routes to **capabilities** (§16), not hardcoded class names. Graph edges MUST reflect manifest roster per application — not global cross-product shortcuts.

**Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase CG.

---

# 20. Agent Lifecycle Governance

Beyond contract shape (§12) and registry metadata (§15):

| Stage | Requirement |
|-------|-------------|
| Certification | quality + policy + security gates before production |
| Promotion | dev → staging → production with evidence |
| Deprecation | migration windows, runtime filters for retired agents |
| Retirement | rollback/archive semantics |
| Ownership | explicit owner + escalation path |

**Code:** `runtime/architecture/agent_lifecycle_governance.py`, `agent_certification.py`, `agent_promotion.py`, `production_ownership.py`.

Runtime MUST reject or reroute retired/deprecated agents in production mode (V-REM-ALG.*). **Plan:** Phase AS + V-REM in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md).

---
