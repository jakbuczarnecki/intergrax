# RELIABILITY_FAILURE_AND_HITL — production gates (§40+)

**Parent hub:** [`RELIABILITY_FAILURE_AND_HITL.md`](../RELIABILITY_FAILURE_AND_HITL.md)

# 35. Autonomy Control Model (Autonomy Slider)

Users and operators MUST be able to **steer how much the system acts without asking** — at session, task, or step granularity. This is distinct from host **execution posture** (`ExecutionMode`: STRICT | BALANCED | EXPLORATORY) and agent **dispatch mode** (`AgentExecutionMode`: SYNC | ASYNC).

## 35.1 Autonomy levels

| Level | User experience | Harness behaviour |
|-------|-----------------|-------------------|
| **MANUAL** | User drives each meaningful action | Tools with side effects blocked unless explicitly approved; planner may suggest but not execute; HITL default-on for external writes |
| **ASK** | Agent proposes; user confirms high-impact steps | Policy routes risky tools and low-confidence outputs to approval queue; auto-continue for read-only / safe tools per allowlist |
| **AUTONOMOUS** | Agent executes within policy envelope | Full tool policy + cost caps; HITL only on policy triggers (risk class, confidence threshold, regulated pathways) |

```text
AutonomyLevel:
    MANUAL
    ASK
    AUTONOMOUS
```

**Mid-run changes:** autonomy MAY change at any time via `TaskExecutionOptions.autonomy_level` or operator API — PolicyEngine re-evaluates **before the next UAEP step** and before each tool invocation (UAEP §42.11).

## 35.2 Resolution order

```text
effective_autonomy = min(
    user_requested_autonomy,      # slider / task option
    tenant_policy_ceiling,        # org governance
    execution_mode_ceiling,       # STRICT caps at ASK for destructive tools
    agent_contract.risk_level     # high-risk agents never fully AUTONOMOUS without override
)
```

| Host `execution_mode` | Typical autonomy ceiling |
|-----------------------|--------------------------|
| `EXPLORATORY` | Up to `AUTONOMOUS` (lab) |
| `BALANCED` | Up to `ASK` for destructive tools; `AUTONOMOUS` for read-only |
| `STRICT` | Default `ASK`; `AUTONOMOUS` only with explicit policy exception |

## 35.3 Mapping to runtime primitives

| Autonomy | Tool execution | Planning | HITL |
|----------|----------------|----------|------|
| MANUAL | `PolicyDecision.DENY` except allowlisted reads; `REQUEST_HUMAN` before writes | Plan visible; execute only after approval | Default for most steps |
| ASK | Risk-scored: auto vs queue | Auto plan; confirm on `risk >= threshold` | Queue for gated tools |
| AUTONOMOUS | Policy + budget only | Auto plan and execute | On interrupt types only (§42.8) |

**Implementation anchors:** `PolicyEngine.evaluate_tool_call`, `AgentDecision.REQUEST_HUMAN`, `HumanDecisionStore`, `hitl.*` tools — UAEP §42.10.

## 35.4 UX contract (platform)

- Slider state MUST be **persisted** on `Task` / session metadata and echoed in trace (`AUTONOMY_LEVEL_SET`, `AUTONOMY_LEVEL_CHANGED`).
- Downgrade (AUTONOMOUS → MANUAL) MUST be **immediate** for new steps; in-flight tool calls follow cancel-or-complete policy per `CancellationCoordinator`.
- Upgrade (MANUAL → AUTONOMOUS) MUST NOT bypass unresolved HITL items.

**As-built (2026-06-09):** `AutonomyLevel` on `TaskExecutionOptions`; effective level via `autonomy_resolver` + `AutonomyGovernanceMiddleware`; trace events `AUTONOMY_LEVEL_SET` / `AUTONOMY_LEVEL_CHANGED`. **Mid-run HTTP API** (`POST …/tasks/{id}/autonomy`) mounted on **lab host only** — runtime downgrade/upgrade works on all paths when set on task envelope.

**Tier-3 debt:** product hosts without `mount_harness_task_routes` require client to set `autonomy_level` on task create or resume payload.

**Plan:** [`plan/RELIABILITY_FAILURE_AND_HITL.md`](../plan/RELIABILITY_FAILURE_AND_HITL.md) Phase REL-ADV (**Done**); surface parity → H-APP-WIRING.1.

---
