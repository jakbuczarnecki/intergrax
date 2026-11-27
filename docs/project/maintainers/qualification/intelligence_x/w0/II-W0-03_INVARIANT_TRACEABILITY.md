# II-W0-03 — Intelligence Invariant Traceability Matrix

**Package:** II-W0-03  
**Primary obligations:** Architecture §4 (twelve II-INV rows)  
**Evidence commit:** `79e978feb280614e35daf2c8f09d83c9634ed348`

| II-INV | Exact normative obligation | Canonical source ID | Repository path | Section | Source commit | Source blob | Authority preserved | Future evidence owner | Positive oracle | Negative oracle | Conflict disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|
| II-INV-01 Goal First | Accepted objective governs execution; plans are revisable means | Architecture §4 | `docs/project/capabilities/architecture/INTEGRAX_INTELLIGENCE.md` | §4 row II-INV-01; complement map | `79e978fe…` | `c89c4ccb…` | Goal/acceptance owners unchanged | W1 goal contract mapping | Plans subordinate to accepted goal | Weakening goal immutability without owner | No conflict |
| II-INV-02 Single Execution Authority | All effectful ops pass canonical admission | UEA-INV-005, UEA-INV-007, UEA-INV-022 | `docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md` | Invariant table ~§UEA | `79e978fe…` | `dc99950e…` | Execution sole tree | W1 admission proofs | Single admission path | Second execution root | No conflict |
| II-INV-03 Adaptive Planning | Typed validated plan revisions | UEA-INV-004, UEA-INV-006 | `UNIFIED_EXECUTION_ARCHITECTURE.md` | UEA-INV-004/006 | `79e978fe…` | `dc99950e…` | Node ≠ execution identity | W1 plan revision tests | Topology ≠ execution tree | Plan as lifecycle owner | No conflict |
| II-INV-04 Governed Autonomy | No inferred auth or HITL bypass | UEA-INV-009, UEA-INV-021 | `UNIFIED_EXECUTION_ARCHITECTURE.md` | §12 governed execution | `79e978fe…` | `dc99950e…` | Governance permission owner | W1 negative auth suite | Explicit permission only | Implicit grant from plan | No conflict |
| II-INV-05 Dynamic Collaboration | Canonical topology/handoff | ORCHESTRATION + Nexus | `INTEGRAX_INTELLIGENCE.md` | §4 complement II-INV-05 | `79e978fe…` | `c89c4ccb…` | Orchestration topology owner | W1 handoff mapping | Legal handoff paths | Second collaboration bus | **Unresolved:** message extension owner deferred to W1 (§7.1) |
| II-INV-06 Evidence-Based Decisions | Structured evidence; untrusted text not authority | UEA-INV-015 | `UNIFIED_EXECUTION_ARCHITECTURE.md` + OBS docs | UEA-INV-015 cross-ref | `79e978fe…` | `dc99950e…` | Observability records truth | W1 evidence tests | Evidence does not grant authority | Log text as permission | No conflict |
| II-INV-07 Outcome Verification | Acceptance beyond executor declaration | Decision Verification | `INTEGRAX_INTELLIGENCE.md` | §4 complement; §7.1 | `79e978fe…` | `c89c4ccb…` | Verification owner separate from executor | W1 benchmark oracles | Independent verification | Self-certify success | No conflict |
| II-INV-08 Recovery Before Escalation | Authorized alternatives before repeat | UEA-INV-012, UEA-INV-013; ERL | `UNIFIED_EXECUTION_ARCHITECTURE.md`; `UNIFIED_EXECUTION_RUNTIME.md` | UEA-INV-012/013; ERL uncertainty | `79e978fe…` | `cb8a28d2…` | Execution/ERL recovery | W1 REF-II-04 | Preserve identity on retry | Blind uncertain retry | No conflict |
| II-INV-09 Bounded Intelligence | Child budgets within parent | UEA-INV-010 | `UNIFIED_EXECUTION_ARCHITECTURE.md` | UEA-INV-010 | `79e978fe…` | `dc99950e…` | Budget single ledger | W1 budget tests | No second ledger | Delegated budget reset | No conflict |
| II-INV-10 Experience Reuse | Scoped validated experience | Memory / AHI / Self-Healing | `INTEGRAX_INTELLIGENCE.md` | §4 complement | `79e978fe…` | `c89c4ccb…` | Existing strategy owners | W1 experience gates | Reuse within scope | Cross-tenant replay | No conflict |
| II-INV-11 Provider Neutrality | Profile-based routing | Model routing / AHI | `INTEGRAX_INTELLIGENCE.md` | §4 complement; II-OPR-06 | `79e978fe…` | `c89c4ccb…` | Routing owners | W1 routing proofs | Replaceable providers | Vendor branch in core | No conflict |
| II-INV-12 Single Semantic Ownership | Extend owners; no duplicate planes | SYS-INV-01 | `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | SYS-INV-01 | `79e978fe…` | `e45b9a3e…` | Tier import canon | W1 owner audit | No new Tier edge | Duplicate store/validator | No conflict |

**FRZ/ECAP:** Applicable where existing campaigns already bind Foundation freeze evidence; no FRZ promotion claimed in W0-A. No synthetic identifiers.

**Primary row count:** 12/12.

**Positive oracle (aggregate):** Each invariant preserves existing authority and names a future W1 evidence owner. **PASS (documentary).**

**Negative oracle (aggregate):** Reject weakened frozen controls and unbound normative claims; unsubstantiated mappings remain explicit (II-INV-05 message extension).

**Original acceptance criterion:** Traceability to existing invariants; no conflicting authority — **met with one documented W1 deferral.**