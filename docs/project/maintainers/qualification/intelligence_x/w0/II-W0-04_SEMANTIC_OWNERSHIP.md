# II-W0-04 — Semantic Ownership and Overlap Matrix

**Package:** II-W0-04  
**Binding source:** Architecture §7.1  
**Evidence commit:** `79e978feb280614e35daf2c8f09d83c9634ed348`

## Ownership matrix (§7.1 coverage)

| Concern | Accountable semantic owner | Supporting mechanism | Proposal authority | Decision authority | Execution authority | State / evidence ownership | Prohibited duplication | Source ref | Commit / blob | Provisional W1 | Unresolved decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Admission, execution identity, lifecycle | Execution / UER | Execution Engine | Cognition proposes plans | Governance permission; Execution admits | Execution | Execution records | Second execution root | §7.1 row 1 | `79e978fe…` / `c89c4ccb…` | Map ports | — |
| Plan proposal, validation, revision | Reasoning & Cognition + Orchestration topology | NexusPlan | Cognition | Orchestration validates topology | Execution admits work | Plan lineage owner | Duplicate validators | §7.1 rows 4–5 | same | Schema reconcile | — |
| Scheduling and interpretation | Nexus (private to Execution) | Nexus | Nexus | Execution | Execution | Execution/Nexus | Nexus as lifecycle owner | §7.1 row 6; UEA-INV-007 | `dc99950e…` | Call-order proof | — |
| Principal, policy, permissions | Governance | Security boundaries | Host requests | Governance | Execution applies checkpoint | Policy store | Duplicate permission authority | §7.1 row 10 | `c89c4ccb…` | Child authority tests | — |
| RunBudget accounting | Budget | Single ledger | Host/Cognition estimates | Budget enforcement | Execution coordinates | Budget subsystem | Duplicate RunBudget ledger | §7.1 row 11 | `c89c4ccb…` | Ledger integration | — |
| Capacity admission | Capacity | Capacity subsystem | Host | Capacity | Execution | Capacity projections | Capacity owning budget | §7.1 row 12 | `c89c4ccb…` | LOAD profiles | — |
| Persistent goals / Worker envelope | Autonomous Work | AW runtime | AW host | AW + Governance | Execution per work item | AW durable state | AW as admission owner | §7.1 rows 3, 12 | `c89c4ccb…` | AW port map | — |
| Input-context budget | Context Engineering | UCL / Memory | Cognition | Context Engineering | Execution | Context owners | Duplicate context ledger | §7.1 row 12 | `c89c4ccb…` | Token accounting | — |
| Retry, recovery, uncertainty | Execution / ERL / Self-Healing | ERL, SH strategies | Diagnostics read-only proposals | Execution / AW (worker) | Execution | Recovery records | Duplicate recovery controller | §7.1 row 14 | `cb8a28d2…` | REF-II-04 | — |
| Durable configuration adaptation | AHI | AHI | AHI evidence proposals | Governance + owners | Execution for admitted changes | AHI store | Unauthorized adaptation | §7.1 row 15 | `c89c4ccb…` | AHI overlap review | — |
| Evidence interpretation | ERI / Diagnostics | Read-only diagnostics | ERI | Domain verification | Execution | ERI projections | Diagnostics mutating state | §7.1 row 16 | `c89c4ccb…` | Read-only proofs | — |
| Experience retention | Memory / AHI / Self-Healing | Within scopes | Strategy owners | Memory governance | Execution | Memory | Duplicate experience truth | §7.1 row 17 | `c89c4ccb…` | Poisoned exp. tests | — |
| Cross-domain capability acceptance | Platform Architecture (proposed) | — | Architecture | Governance adoption | N/A | Evidence only | Runtime takeover | §7.1 row 19 | `c89c4ccb…` | W0-05/06 | No runtime owner until adoption |

## Mandatory overlap areas

| Overlap domain | Boundary rule | Prohibited duplication |
|---|---|---|
| AHI / AHE | AHI proposes configuration; Governance and owners decide; Execution admits effects | Second adaptation control plane |
| ERI / Execution | ERI read-only interpretation | ERI as admission authority |
| Autonomous Work / Execution | AW owns durable responsibility; Execution owns per-attempt lifecycle | AW bypassing admission |
| Self-Healing / Execution | SH strategies within authorized recovery | SH minting execution identity |
| Governance / Execution | Governance decides permission; Execution applies UEA-INV-022 | Governance as lifecycle owner |
| Budget / Capacity | Budget accounts; Capacity allocates; no merged ledger | Duplicate RunBudget |
| Memory / Context Engineering | Scoped retention vs input assembly | Cross-tenant experience bleed |

## Positive oracle

Every concern has one accountable owner or an explicit blocking decision (cross-domain acceptance; inter-agent message extension from II-W0-03). **PASS (documentary).**

## Negative oracles (expected rejection)

Duplicate execution identity; duplicate authoritative store; duplicate recovery controller; duplicate RunBudget ledger; duplicate permission authority; duplicate validation authority; second control plane; unauthorized adaptation; unsubstantiated ownership resolution without W1 evidence.

**Original acceptance criterion:** Single owner per concern; overlap and conflict matrix — **met at documentation level with explicit W1 deferrals.**