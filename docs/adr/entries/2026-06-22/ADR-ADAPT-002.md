# ADR-ADAPT-002: ADAS (Agent Design Search) inside AHI — not a separate layer

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-22 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md#adas--agent-design-search-sub-capability) · [`satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](../../../architecture/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md) · [ADR-ADAPT-001](../2026-06-05/ADR-ADAPT-001.md) · Phase **AHI-ADAS** |

## Context

Industry trends toward **meta-agent building** (agents that propose other agents) create pressure to add a parallel "agent factory" layer, a Tier-3-only lab workflow, or unconstrained code mutation paths.

Intergrax already ships **Adaptive Harness Intelligence (AHI)** — a governed Tier-1 control plane with observe → propose → validate → shadow → canary → apply → verify semantics, policy gates, archive/version stores, and verification loops ([ADR-ADAPT-001](../2026-06-05/ADR-ADAPT-001.md)).

**ADAS (Agent Design Search)** extends AHI to search, evaluate, archive, and promote **agent candidates** instead of runtime profile versions. Without an explicit ADR, implementers may:

- Fork a second governance/evaluation/archive stack
- Place ADAS logic in Tier-3 applications
- Allow MAS strategy agents to mutate production `agents/` or Nexus runtime directly

## Decision

Implement **ADAS as a governed AHI sub-capability** in Tier-1:

```text
ADAPTIVE_HARNESS_INTELLIGENCE
  └── ADAS / Agent Design Search
        └── MAS / Meta Agent Search (replaceable AgentDesignStrategy)
```

Concrete rules:

1. **Tier-1 ownership** — control plane lives at `intergrax/runtime/adaptive/agent_design_search/`; reuses AHI signal, governance, verification, and promotion lifecycle patterns.
2. **Not a separate top-level harness layer** — no parallel PolicyEngine, evaluation registry, trace system, or promotion stack.
3. **Not Tier-3-only** — optional `applications/adas_lab/` is operator UI/wiring only; archive, evaluation, governance, and promotion semantics remain Tier-1.
4. **Candidate materialization via scaffold only** — `AgentCandidateDraft` → existing scaffold bridge → sandbox/archive; no direct production source overwrite.
5. **MAS is a strategy, not the architecture** — Tier-2 agents may propose drafts; they must not self-approve promotion or bypass static/eval/archive gates.
6. **Evidence over declaration** — promotion requires auditable `AgentCandidateEvidenceBundle` and governed active registration (registry pointer + tenant binding in v1).

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| ADAS as new Tier-1 domain parallel to AHI | Duplicates governance, evaluation, and lifecycle; drift risk |
| ADAS as Tier-3-only application | Core archive/eval/governance logic would leak into applications tier |
| MAS agents write directly to `agents/` | Breaks tier boundaries; untyped mutations; no rollback |
| External AutoML / agent-builder SaaS | Vendor lock-in; breaks offline/air-gapped lab requirements |

## Consequences

### Positive

- Single adaptive canon: profile tuning and agent design search share gates, utility model, and verification semantics
- Reuse of scaffold, AgentContract, agent promotion patterns, observability spine, and cost governance
- Clear audit boundary for enterprise customers ("governed design loop", not "agent that writes agents")

### Negative

- ADAS implementation must track AHI envelope changes (shared governance contracts)
- Agent candidate promotion adds active-registration semantics beyond profile version apply

## Compliance

- Tier boundaries preserved (`intergrax/` ↛ `agents/` direct mutation by MAS; agents ↛ applications)
- PolicyEngine and human approval remain mandatory for production promotion by default
- Linked architecture hub § ADAS, satellite canon, and plan Phase AHI-ADAS updated

## Implementation notes

- Architecture: [`ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](../../../architecture/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md)
- Plan phases: **AHI-ADAS-00** (docs/ADR) through **AHI-ADAS-90** (enterprise hardening)
- No runtime code implied by this ADR alone
