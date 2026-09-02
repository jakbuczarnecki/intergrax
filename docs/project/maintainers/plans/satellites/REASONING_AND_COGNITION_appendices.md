# REASONING_AND_COGNITION - appendices

**Parent hub:** [`REASONING_AND_COGNITION.md`](../REASONING_AND_COGNITION.md)

## Appendix A - Reasoning and Cognition traceability (Phase COG-DEPTH)

| Architecture § | Topic | Task IDs |
|----------------|--------|----------|
| §5 Three planes | Plane boundaries | COG-DOC.* |
| §9 Classification | Classifier extensions | COG-3.* · ORCH-CONFIG.1 **Done** |
| §9.4 Routing modes | Authoring canon (docs) | COG-DOC.6 **Done** |
| §10 Nexus planning | Planner unification | COG-1.* |
| §10.4 LLM planner | Prompt Registry | COG-2.1 |
| §12 Engine planner | Orchestrator bridge | COG-1.1 |
| §14 DecisionRecord | Planning phase emit | COG-4.* |
| §15 Prompt compilation | Registry on all planners | COG-2.* |
| §16 Model selection | ReasoningProfile | COG-5.* |
| §17 Failure taxonomy | ReasoningFailureKind | COG-6.* |
| §18 Observability | SLO metrics | COG-OBS.* |
| §21 Gap register | Maturity uplift | All COG-DEPTH |

### Historical closeout traceability (pre-RCL domain)

These items implemented under FLOW/ORCH phases - **Done**; canon now owned by RCL:

| Legacy ID | Deliverable | RCL architecture § |
|-----------|-------------|-------------------|
| FLOW-1 | `EngineBackedNexusPlanner` | §10.4 |
| FLOW-11 | Pre-plan policy hooks | §10.5 |
| FLOW-12 | `DecisionRecord` UAEP gate | §14 |
| FLOW-17 | `multi_agent_order` | §10.3 |
| ORCH-1 | Planner strategies explicit | §10 |
| ORCH-2 | Declarative `graph_spec` | §11 |
| FAUDIT-COG.1 | DecisionRecord contract | §14 |

---

## Appendix B - FAUDIT-32 §7 scorecard (baseline)

| Audit question | Pre-RCL | Post COG-DOC | Post COG-DEPTH target |
|----------------|---------|--------------|----------------------|
| Structured plan contract? | Yes (`NexusPlan`) | Yes - canon §10 | Maintain |
| DecisionRecord per step? | UAEP only | Documented §14 | Planning + UAEP |
| Reasoning separated from execution? | Yes (UAEP) | Canon §4, §8 | Maintain |
| Planning strategies explicit? | Yes | Canon §10, Appendix B | Maintain |
| Prompt compilation layered? | Partial | Cross-ref §15 | COG-2.* Done |
| Reasoning failures classified? | No | Taxonomy §17 doc | COG-6.* code |
| **Layer score** | **L2** | **L2** (plan accurate) | **L3** (COG-DEPTH **Done**) |

---

## Appendix C - Operator reading order

1. [`architecture/REASONING_AND_COGNITION.md`](../architecture/REASONING_AND_COGNITION.md) - RCL canon
2. This plan - COG-DEPTH register when implementing
3. [`architecture/NEXUS_EXECUTION_FLOW.md`](../architecture/NEXUS_EXECUTION_FLOW.md) - end-to-end flow only
4. [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix I §I.4 - host planner configuration

---

### COG-DEPTH - Paydown log

| Date | COG ID | Summary |
|------|--------|---------|
| 2026-06-09 | COG-1.*–COG-OBS.* | Phase COG-DEPTH **22/22 Done**; reference host engine planner presets |

---
