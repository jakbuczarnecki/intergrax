# Domain Layer Audit Instructions

**Purpose:** Copy-paste audit prompts for each of the 21 Harness domain pairs — no ad-hoc instructions per iteration.  
**Procedure:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md) · **Output format:** [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8  
**Why separate files (not architecture/plan canon):** Architecture docs describe *what* the layer is; plan docs track *implementation status*. Audit prompts are *operator tooling* — they belong in `guides/` alongside the audit map.

---

## How to run a domain audit

1. Pick the domain from the table below.
2. Open the matching `audit/<DOMAIN>.md` file.
3. Copy from `---BEGIN PROMPT---` through `---END PROMPT---` into a new agent chat.
4. Edit only the **USER CONFIG** block (`mode`, optional `focus`).
5. The agent reads the domain pair + code + tests; it does **not** implement unless you request a separate pass.

**Modes**

| `mode` | Behaviour |
|--------|-----------|
| `audit-only` | Report + maturity score. No file edits. |
| `audit-and-fix` | Report + update `plan/<DOMAIN>.md` gap rows and architecture audit register if present. No code unless explicitly requested. |

**Depth:** Every domain prompt requests **engine-depth** review — architecture canon, plan status, source code, tests, CI gates, and production-system comparison (not documentation-only survey).

---

## Domain index (21 pairs)

| Domain | Audit prompt | Primary code | Audit map layers |
|--------|--------------|--------------|------------------|
| `PLATFORM_FOUNDATION` | [PLATFORM_FOUNDATION.md](PLATFORM_FOUNDATION.md) | tiers, `AGENTS.md`, doc governance | 1–2, 32 |
| `UNIFIED_EXECUTION_RUNTIME` | [UNIFIED_EXECUTION_RUNTIME.md](UNIFIED_EXECUTION_RUNTIME.md) | `intergrax/runtime/nexus/`, UAEP, policy | 4–5, 8, 23–24 |
| `ORCHESTRATION` | [ORCHESTRATION.md](ORCHESTRATION.md) | `intergrax/runtime/nexus/orchestration/` | 3, 9 |
| `NEXUS_EXECUTION_FLOW` | [NEXUS_EXECUTION_FLOW.md](NEXUS_EXECUTION_FLOW.md) | `runtime_steps/`, flow narrative | 8–10 |
| `AGENT_CONTRACTS_AND_ASSEMBLY` | [AGENT_CONTRACTS_AND_ASSEMBLY.md](AGENT_CONTRACTS_AND_ASSEMBLY.md) | `intergrax/agents/`, registry, prompts | 17–20, 31 |
| `INTEGRATIONS` | [INTEGRATIONS.md](INTEGRATIONS.md) | `intergrax/integrations/` | 13 |
| `RAG` | [RAG.md](RAG.md) | `intergrax/rag/` | 14 |
| `TOOLS` | [TOOLS.md](TOOLS.md) | `intergrax/tools/`, `ToolRuntime` | 11 |
| `CODE_CRAFT` | [CODE_CRAFT.md](CODE_CRAFT.md) | `intergrax/codecraft/` | 11b |
| `SKILLS` | [SKILLS.md](SKILLS.md) | `intergrax/skills/` | 12 |
| `LLM_ADAPTERS` | [LLM_ADAPTERS.md](LLM_ADAPTERS.md) | `intergrax/llm_adapters/` | 6 |
| `MEMORY` | [MEMORY.md](MEMORY.md) | `intergrax/memory/`, context | 15–16 |
| `MODALITY` | [MODALITY.md](MODALITY.md) | `intergrax/modality/` | 29 |
| `OBSERVABILITY` | [OBSERVABILITY.md](OBSERVABILITY.md) | `intergrax/observability/` | 21, 30 |
| `RELIABILITY_FAILURE_AND_HITL` | [RELIABILITY_FAILURE_AND_HITL.md](RELIABILITY_FAILURE_AND_HITL.md) | retry, HITL, failure taxonomy | 22 |
| `TIER3_APPLICATION_ENVIRONMENT` | [TIER3_APPLICATION_ENVIRONMENT.md](TIER3_APPLICATION_ENVIRONMENT.md) | `applications/`, profiles | 3, 28 |
| `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | [EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | `intergrax/scaffold/`, CI gates | 25–27, 30 |
| `ADAPTIVE_HARNESS_INTELLIGENCE` | [ADAPTIVE_HARNESS_INTELLIGENCE.md](ADAPTIVE_HARNESS_INTELLIGENCE.md) | adaptive loops, routing | L4 AHI |
| `CRITIC_VERIFICATION` | [CRITIC_VERIFICATION.md](CRITIC_VERIFICATION.md) | critic, eval harness | 25 depth |
| `REASONING_AND_COGNITION` | [REASONING_AND_COGNITION.md](REASONING_AND_COGNITION.md) | planners, classifiers | 7 |
| `ELASTIC_CAPACITY_AND_SCALING` | [ELASTIC_CAPACITY_AND_SCALING.md](ELASTIC_CAPACITY_AND_SCALING.md) | capacity, backpressure | 30 |

---

## Shared production Harness checklist

**Apply to every domain audit.** For each item: **Yes / Partial / No / Unknown** + file/symbol/test evidence.

### Architecture & modularity

- Single canonical execution path — no parallel legacy shortcuts for the same capability.
- Tier boundaries respected (`intergrax/` ↔ `agents/` ↔ `applications/` import rules).
- Typed contracts (`Protocol`, Pydantic, enums) — no untyped dict bridges where canon defines types.
- Composition over duplication — Tier-0 mechanism reused, not reimplemented in Nexus or agents.
- Extension via registries/plugins — not forked runtime branches.

### Configuration & strategy selection

- Formal profile types exist and are documented (`*Profile` in Tier-3 `ApplicationEnvironmentProfile`).
- Environment variables and profile fields **actually wired** to runtime behaviour (no dead config).
- Strategy/mode selection explicit where the domain supports multiple algorithms (retrieval, chunking, planning, tool selection, model routing).
- Defaults are safe; advanced behaviour requires explicit Tier-3 opt-in.
- Feature flags and experimental paths labelled (beta) in canon and code.

### Override & customization surfaces

- Tier-3 hosts can override backends, profiles, and wiring without modifying `intergrax/runtime/`.
- Tier-2 agents consume capabilities through Nexus policy + runtime bridges — no vendor SDK bypass.
- `Protocol` / hook / bridge extension points documented and tested.
- Agent-specific or product-specific logic stays in `agents/` or `applications/` — not in universal Tier-0.

### Observability, tracing & logging

- Every decision and invocation emits trace/journal events with `trace_id`, `run_id`, `tenant_id` where applicable.
- Hot paths have structured logs — not print/debug-only.
- OpenTelemetry or platform spine integration on critical paths (or documented gap).
- Redaction policy for sensitive payloads in production traces.
- Metrics/counters for latency, errors, saturation where production SLOs require them.

### Security & governance

- Policy checks before side effects (tools, memory writes, retrieval, LLM calls, delegation).
- Tenant isolation enforced and tested — not only documented.
- Secrets never in agent code or static config committed to repo.
- Risk classification drives HITL and permission boundaries.
- Untrusted input surfaces (user docs, web, tool output) have explicit defence (poisoning, injection, sandbox).

### Reliability & error handling

- Failures typed and classified — not bare exceptions swallowed.
- Retries with budgets; circuit breakers or fallbacks where canon requires.
- Idempotency for mutating operations where applicable.
- Graceful degradation paths documented and tested.
- HITL escalation wired for high-risk failure classes.

### Performance & scale

- Behaviour documented for **small** inputs (single request) and **large** inputs (corpora, long context, burst traffic).
- Streaming/async paths where blocking would violate SLOs.
- Backpressure, concurrency limits, and resource budgets enforced at runtime — not only in docs.
- No unbounded in-memory growth on default paths.

### Testing & verification

- Unit tests for core contracts and edge cases — not only happy path.
- Integration/acceptance tests for runtime wiring.
- CI gate scripts (`scripts/check_harness_*.py`) relevant to the domain are green.
- Golden/eval harness where quality measurement matters (RAG, critic, tools).
- Plan row status matches code reality — "Done" requires evidence.

### Documentation alignment

- `architecture/<DOMAIN>.md` ↔ `plan/<DOMAIN>.md` ↔ code agree.
- Gaps map to plan phase IDs (not orphan findings).
- Maturity score L0–L4 with evidence per [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md) §5.

---

## Relationship to other audit docs

| Document | Role |
|----------|------|
| [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md) | 32-layer map, scoring, output skeleton |
| [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) | Generic multi-layer / full-platform prompt |
| `audit/<DOMAIN>.md` (this folder) | **Deep single-domain** prompt — use this for RAG, Tools, Memory, etc. |
| `architecture/<DOMAIN>.md` | Canon + audit **results** register (updated after audits) |
| `plan/<DOMAIN>.md` | Remediation queue — updated in `audit-and-fix` mode |

---

## Adding or updating prompts

When architecture or plan contracts change:

1. Update the domain's `audit/<DOMAIN>.md` prompt (code paths, dimensions).
2. Update the index table above if primary code paths shift.
3. Do **not** duplicate the full shared checklist — reference this README.
