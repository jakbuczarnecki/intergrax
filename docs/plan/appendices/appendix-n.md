**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix N — Nexus execution flow traceability (Phase FLOW)

**Source:** [`NEXUS_EXECUTION_FLOW_REFERENCE.md`](NEXUS_EXECUTION_FLOW_REFERENCE.md) §23–§25 · [ADR-FLOW-001](adr/ADR-FLOW-001.md)

**Phase register:** [Phase FLOW](#phase-flow--nexus-execution-depth) · **Band 2aj** · queue [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed) · execution [§6.2aj](#62aj-phase-flow-execution-order-band-2aj--closed-2026-06-07)

**Status:** **Done** (2026-06-07) · **17/18** deliverables Done (**FLOW-8 Deferred**)

> **Note:** Distinct from `AGENT_CREATION_GUIDE.md` Appendix N (agent assembly). This appendix maps **orchestration runtime depth** gaps only.

### N.1 FLOW-GAP → FLOW ID matrix (complete)

| Gap ID | Category | Severity | FLOW ID | Deliverable | AUDIT_MAP § |
|--------|----------|----------|---------|-------------|-------------|
| FLOW-GAP-01 | Runtime-core | High | FLOW-1 | Real `EngineBackedNexusPlanner` | §7 |
| FLOW-GAP-02 | Runtime-core | **Critical** | FLOW-2 | ADR-FLOW-001 delegation expansion | §10 |
| FLOW-GAP-03 | Runtime-core | Medium | FLOW-3 | `max_delegation_depth` enforcement | §10 |
| FLOW-GAP-04 | Runtime-core | Medium | FLOW-4 | Opt-in run-level retry | §9, §22 |
| FLOW-GAP-05 | DX | Low | FLOW-5 | `AgentGraph.on_error` wire | §9 |
| FLOW-GAP-06 | Runtime-core | Medium | FLOW-6 | Strict cycle detection | §9 |
| FLOW-GAP-07 | Production-hardening | Medium | FLOW-7 | `MergePolicy` / composer profile | §9 |
| FLOW-GAP-08 | DX / lifecycle | Low | FLOW-10 | Reserved lifecycle states ADR | §8 |
| FLOW-GAP-09 | Production-hardening | Medium | FLOW-11 | Pre-plan policy hooks | §5 |
| FLOW-GAP-10 | Product-proof | Product | FLOW-8 | §42.43 reference Tier-3 app (**Deferred**) | §28 |
| FLOW-GAP-11 | Production-hardening | Medium | FLOW-9 | Multi-agent eval hooks | §25 |
| FLOW-GAP-12 | Runtime-core | Medium | FLOW-13 | `max_inflight_nodes` profile + factory wire | §9 |
| FLOW-GAP-13 | Runtime-core | Medium | FLOW-14 | `SubtaskContract` in delegation expansion | §10 |
| FLOW-GAP-14 | Production-hardening | Medium | FLOW-15 | Subagent budget envelope enforcement | §10 |
| FLOW-GAP-15 | DX | Low | FLOW-16 | `MODIFY_PLAN` reserved semantics ADR | §9 |
| FLOW-GAP-16 | DX | Low | FLOW-17 | `MULTI_AGENT` deterministic ordering policy | §9 |
| §24 / FAUDIT-COG-1 | Cognition | Medium | FLOW-12 | `DecisionRecord` regression gate | §7 |
| — | Docs | Low | FLOW-DOC.* | Flow reference + plan sync | — |

### N.2 Maturity uplift targets

| AUDIT_MAP § | Baseline (FAUDIT-32) | Target | Closing FLOW IDs |
|-------------|----------------------|--------|------------------|
| §5 Policy | L2 partial | **L3** | FLOW-11 |
| §7 Reasoning / planning | L2 | **L3** | FLOW-1, FLOW-12 |
| §8 Execution runtime | L3 | **L3** | FLOW-10 (maintain) |
| §9 Orchestration / graph | L3 partial | **L3+** | FLOW-4–7, FLOW-6, FLOW-13, FLOW-16, FLOW-17 |
| §10 Subagents | L2 | **L3** | FLOW-2, FLOW-3, FLOW-14, FLOW-15 |
| §25 Evaluation | L2 | **L3** | FLOW-9 |

### N.3 Paydown log

| Date | FLOW ID | Summary |
|------|---------|---------|
| 2026-06-07 | — | Phase FLOW scheduled; Appendix N (FLOW) created; §6.1aj + §6.2aj active |
| 2026-06-07 | — | FLOW-GAP-12–16 + FLOW-13–17 added; orchestration plan complete vs flow reference |
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Full Phase FLOW closeout; ADR-FLOW-001/002/003 accepted; gate green |

---

*Plan synced (2026-06-07). **Harness platform** bands 1–2aj **Done** (FAUDIT-32 **23/23** + Phase FLOW **17/18**). **Default active queue:** [§6.1](#61-harness-implementation-queue--continuous-gate) maintenance. Product: [§6.3](#63-end-of-plan--deferred-product-work-only) incl. **FLOW-8**. **Every PR:** §6.1 gate green.*
