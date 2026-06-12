# ADR-ADAPT-001: Adaptive Harness Intelligence over classical RL

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-05 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · canon [§54](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md#54-adaptive-harness-intelligence-ahi--l4-runtime-addendum) · [Phase W-ADAPT](../../plan/CRITIC_VERIFICATION.md) |

## Context

Intergrax needs **L4 Adaptive Harness Intelligence (AHI)**: a closed loop that observes Nexus runs, proposes bounded profile changes, gates them through existing Phase V governance, and applies them via versioned profiles with verify/rollback.

The platform already ships **governance L4** contracts (`adaptive_governance.py`, `phase_v_closeout_gate.py --enforce-l4`) but not a **runtime L4** loop that measurably improves utility on golden scenarios.

Alternatives include end-to-end reinforcement learning, per-agent self-modifying code, and external AutoML services.

## Decision

Implement a **governed Adaptive Control Plane** in `intergrax/runtime/adaptive/` with:

- Evidence-driven signals (`HarnessOutcomeSignal`, utility function U)
- Rule-based and contextual-bandit sub-engines (Thompson sampling v1)
- Reuse of Phase V envelopes (`AdaptiveLoopProposal`, `evaluate_bounded_adaptive_loop`)
- Immutable `ProfileVersionStore` with shadow → canary → apply → verify → rollback
- **Async** adaptation (never block Nexus hot path)
- **No classical RL** (no policy-gradient training, no unconstrained reward maximization, no black-box model updates)

PolicyEngine and human gates remain mandatory for `POLICY_LEARNING` and high-risk changes.

## Consequences

### Positive

- Full auditability and alignment with existing Phase V / W-OPS artifacts
- Reuse of evaluation registry, cost budget, capability graph, and promotion patterns
- Tier-3 hosts configure adaptation via typed profiles without forking Nexus

### Negative

- No claims of neural policy optimality or global convergence
- Additional operational surface (scheduler, stores, runbooks)

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| End-to-end RL fine-tuning | Opaque, hard to audit, violates harness governance model |
| Per-agent self-modifying code | Breaks Tier-2/Tier-3 separation; untyped mutations |
| External AutoML SaaS | Vendor lock-in; breaks offline/air-gapped lab requirements |

## Compliance

- Tier-1 package: `intergrax/runtime/adaptive/`
- Governance source of truth remains `intergrax/runtime/architecture/adaptive_governance.py`
- Closeout gate target: `phase_w_adapt_closeout_gate.py --enforce-l4-runtime`
