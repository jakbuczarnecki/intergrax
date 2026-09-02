# ADR-DISPUTE_SIM-001: Four-agent split for Dispute Simulation Workspace

**Status:** Accepted  
**Date:** 2026-06-07  
**Domain:** Tier-3 application (`dispute_sim_application`)

## Context

DSW must organize dispute materials, analyze arguments, propose strategy, and simulate court paths - while keeping correspondence drafts behind HITL. A single monolithic agent would mix intake, analysis, strategy, and simulation concerns, complicating testing, policy, and CVL critic attachment.

Existing `legal` agent (`legal.review`) covers single-contract review - different bounded context. DSW is a new product environment per explicit reprioritization.

## Decision

Split DSW into **four Tier-2 agents** mounted in one Tier-3 product host:

| Agent | Capability |
|-------|------------|
| `dispute_intake` | `dispute.intake` |
| `dispute_analyst` | `dispute.analyze` |
| `dispute_strategist` | `dispute.strategy` |
| `dispute_scenario` | `dispute.scenario` |

Orchestration via Nexus capability graph (`dispute.pipeline` - DSW.2), not custom runtime loops.

Correspondence pitfall review lives in `dispute_scenario` (mode `correspondence_review`) to keep outbound-risk logic with process simulation.

Reuse `IntegrationProfile.legal_product()` and `legal` skill bundle; optional subgraph to `legal.review` in DSW.5.

## Consequences

**Positive**

- Clear tier hygiene and per-agent test surfaces
- CVL critics attach per milestone (matrix, strategy, drafts)
- Parallel implementation waves (DSW.1 intake unblock DSW.3)

**Negative**

- More manifest wiring than single-agent SKU
- Graph spec required before full pipeline UX (DSW.2)

**Follow-up**

- DSW.2 graph registration
- DSW.4 HITL on correspondence outputs
- No ADR needed for individual UAEP steps unless cross-agent contract changes
