# Intergrax Runtime Architecture

**Hub only** — domain architecture and implementation are paired 1:1 under `architecture/` and `plan/`; multi-layer features are paired 1:1 under `features/architecture/` and `features/plan/`.
**Target:** [`guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**Features:** [`features/README.md`](features/README.md) — cross-layer capability docs that coordinate multiple domain pairs without replacing domain ownership.
**Invariants:** [`guides/SYSTEM_INVARIANTS.md`](guides/SYSTEM_INVARIANTS.md) — cross-layer MUST/MUST NOT rules + `SYS-INV-*` index (P2-ARCH-01)
**Maturity:** [`guides/MATURITY_TAXONOMY.md`](guides/MATURITY_TAXONOMY.md) — four-axis A/I/P/E vocabulary; legacy L3/L4/L5 mapping (P2-ARCH-02). Maturity labels elsewhere in this hub are summaries only; authoritative production readiness claims require four-axis A/I/P/E statements in the owning architecture/plan pair.
**Layer completion:** [`guides/LAYER_COMPLETION_MODE.md`](guides/LAYER_COMPLETION_MODE.md) — deep domain layer closeout workflow
**Doc boundaries (Experimentation/DX):** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md#architecture-vs-implementation-rules-boundary) — architecture vs Cursor/workflow rules placement (P2-ARCH-13)
**Audit:** [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) · **Idea intake (Mode I):** [`bootstrap/idea_audit.txt`](bootstrap/idea_audit.txt) · **Cursor bootstrap:** [`bootstrap/`](bootstrap/) · **Domain audit prompts:** [`audit/`](audit/) · **Architecture audit results:** [`audit_results/`](audit_results/README.md) · **Implementation journal:** [`implementation-journal/`](implementation-journal/README.md)
**Authoring:** [`guides/`](guides/)

---

## Documentation topology

```text
docs/architecture/<DOMAIN>.md       ↔ docs/plan/<DOMAIN>.md
docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
```

Domain pairs own layer architecture and implementation truth. Feature pairs coordinate capabilities that cut across multiple domain pairs. Feature implementation still lands in the owning domain plan rows.

Current feature pairs:

| Feature | Architecture | Plan |
|---------|--------------|------|
| `TOKEN_OPTIMIZATION` | [`features/architecture/TOKEN_OPTIMIZATION.md`](features/architecture/TOKEN_OPTIMIZATION.md) | [`features/plan/TOKEN_OPTIMIZATION.md`](features/plan/TOKEN_OPTIMIZATION.md) |

---

## Four tiers

```text
Tier-0  intergrax/          integrations · tools · skills · LLM · RAG · memory · codecraft
Tier-1  intergrax/runtime/    Nexus · AgentEngine · UAEP · policy
Tier-2  agents/             domain capabilities
Tier-3  applications/       deployable hosts
```

Stack: Integration → Tool → Skill → Agent
Execution: [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md)

---

## Implementer quick start

**Default queue:** [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) **§4.0** priority ladder — Band 1 gate maintenance on every PR; Band 3 product work is **frozen** unless leadership reprioritizes (§6.3).

| Goal | Read first | Command |
|------|------------|---------|
| New agent | [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) | `python -m intergrax.scaffold new-agent <name> --capability <cap>.<action>` |
| New application host | [`guides/APPLICATION_CREATION_GUIDE.md`](guides/APPLICATION_CREATION_GUIDE.md) | `python -m intergrax.scaffold new-application <name>_application` |
| Agent + app bundle | [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](plan/TIER3_APPLICATION_ENVIRONMENT.md) | `python -m intergrax.scaffold new-stack <name>` |
| Extension / plugin | [`guides/EXTENSION_AUTHOR_GUIDE.md`](guides/EXTENSION_AUTHOR_GUIDE.md) | `bootstrap_catalogs()` + entry points `intergrax.tools` / `intergrax.skills` / `intergrax.integrations` |
| Multi-layer feature | [`features/README.md`](features/README.md) | feature architecture → feature plan → affected domain pairs |
| Harness health | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) §6.1 | `uv run intergrax doctor --ci` · `uv run pytest -m gate -q` |

**Work cycle:** strategy → architecture pair or feature pair → smallest domain-owned plan item → implement → gate green → update paired docs + journal if significant.

---

## Agent in the harness environment

**Hub summary for architects, researchers, and AI crawlers** — full canon in [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §13–§40 · plan [Phase ACP](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md).

Intergrax is **not** “one Python class that is also the OS.” The **agent** is a **domain decision unit** inside a **typed, governed environment**. Responsibility is split by design:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ L4  Application + NexusLoop.handle_task()                                 │
│     Environment: profiles, AgentBinding, RequestIdentity, org envelope  │
│     Orchestration: Task graph, capability routing, HITL, Plane A log    │
│     DOES NOT: plan inside one agent's cognitive loop                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ graph node → one Agent.run() per role
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L3  Agent.run() — session decision loop (many steps, one user-facing run) │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ each iteration
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L2  Agent.on_next_step() — author domain hook                             │
│     READ typed state · UPDATE state_delta · DECIDE StepOutcome §32.0      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ StepOutcome
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L1  HarnessKernel.execute_step() — deterministic harness primitive        │
│     policy · gateways · trace · budgets · state merge · checkpoint hook   │
│     DOES NOT: domain replan · choose next graph agent                     │
└─────────────────────────────────────────────────────────────────────────┘
```

| Question | Owner | Canon |
|----------|-------|-------|
| Who acts (tenant, user, org agent)? | Application intake → `RequestIdentity` | §30.9 |
| Which agents run on this Task? | **NexusLoop** + capability registry | §37.6 |
| What is the next domain move? | **`on_next_step`** → `StepOutcome` | §32 · §32.0 |
| Is policy/trace/state safe? | **`HarnessKernel`** | §38 |
| Lab vs prod same agent code? | `merge_environment` + `AgentBinding` | §30 |
| Can this agent ship to production? | Production Readiness Scoreboard | §40.15 · ACP-PROD-12 |

**Strategic invariants (ADR-AGENT-001..003):**

- **Nexus is not the agent** — it orchestrates; it does not replace `on_next_step`.
- **HarnessKernel does not plan** — it executes one harness cycle per step.
- **AgentRuntime.advance_step is glue only** — `on_next_step` then kernel; no policy logic in runtime.
- **Agents are replaceable; the harness is the product.**

**Author entry points:** [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) Appendix AC · roster [`agents/README.md`](../agents/README.md).

**Implementation:** architecture **decision-complete**; code delivery [ACP waves](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md#61aw-acp-detailed-implementation-waves) (typed contracts → step loop → fleet migration Wave 8 → prod gates → **ACP-CLOSE-LEG-5** pipeline retirement). Product agents control the loop via **`on_next_step`** only; Tier-1 `RuntimeEngine` pipeline stack removed ([ADR-FLOW-005](adr/entries/2026-06-12/ADR-FLOW-005.md)).

---

## Application in the harness environment

**Hub summary** — full canon in [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) §24–§51 (APP-CON / APP-EVOL / APP-OPS) · **freeze audit:** [`guides/GOVERNANCE_CONSISTENCY_AUDIT.md`](guides/GOVERNANCE_CONSISTENCY_AUDIT.md) · plan [H-APP-CON](plan/TIER3_APPLICATION_ENVIRONMENT.md#phase-h-app-con--application-environment-architecture-canon-app-con) · [H-APP-FREEZE](plan/TIER3_APPLICATION_ENVIRONMENT.md#phase-h-app-freeze--cross-document-governance-consistency-audit).

The **application** is a **deployable composition shell** — not a cognitive agent. It normalizes intake → `Task`, declares roster and harness profiles, and returns product output. Tier-3 authors control environment through **three modes** (§30): declarative profile, rules envelope, imperative `ApplicationHost` hooks.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ L4  Application host (Tier-3)                                           │
│     ApplicationManifest · ApplicationEnvironmentProfile · surfaces      │
│     ApplicationHost.on_hook (optional) · ApplicationRunSummary (Plane A) │
│     DOES NOT: on_next_step · domain tool loops · private Nexus fork     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ UnifiedTaskRunner.run_task()
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L3  NexusLoop.handle_task() — Agent OS (Tier-1)                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ graph node → Agent.run() — see Agent section above
```

| Question | Owner | Canon |
|----------|-------|-------|
| What agents are active in this product? | **`ApplicationManifest`** roster | §24 · §27 |
| What harness slices are enabled? | **`ApplicationEnvironmentProfile`** (§22.1 flat · §22.6 bundles) | §22 · [ADR-APP-003](adr/entries/2026-06-17/ADR-APP-003.md) |
| Reactive vs daemon vs batch? | Posture + host factory | §23 |
| Who sets routing capability? | L1–L4 matrix | §23.3 |
| Virtual org / simulation rules? | **`OrganizationalPolicyEnvelope`** | §39 |
| Dynamic block at intake / selection? | **`ApplicationHost`** + `HookPoint` | §32 |
| Multi-agent orchestration summary? | **`ApplicationRunSummary`** | §26 · §33 |

**Strategic invariants (APP-CON §28.1):**

- **Applications compose; they do not cognate** — business logic stays in Tier-2 agents.
- **One Task lifecycle** — all surfaces converge on `UnifiedTaskRunner` → `NexusLoop`.
