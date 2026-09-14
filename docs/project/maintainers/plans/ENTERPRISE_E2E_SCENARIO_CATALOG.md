# Enterprise E2E Scenario Catalog

**Document type:** Maintainer-level frozen scenario portfolio record  
**Catalog:** Enterprise E2E Scenario Catalog **v1**  
**Selection status:** **FROZEN**  
**Scenario count:** **30** (7 previously frozen · 23 newly frozen in v1 portfolio freeze)  
**Last frozen:** 2026-09-14  
**Task:** SCENARIO-CATALOG-FREEZE-30-DOCS-R1

---

## Authority and boundaries

| Role | Document |
|------|----------|
| **Frozen problem portfolio (this file)** | Selected enterprise E2E problems for future design, quality gates, and proof work |
| Framework / filesystem inventory | [`E2E_SCENARIO_FRAMEWORK_AUDIT.md`](../../architecture/E2E_SCENARIO_FRAMEWORK_AUDIT.md) |
| Public in-development scenario presentation | [`PROOF_LIBRARY.md`](../../proofs/PROOF_LIBRARY.md) |
| Per-package lifecycle truth | `platform_proofs/scenarios/<slug>/SCENARIO_SPEC.md` frontmatter |

**This catalog is not:**

- proof acceptance, EXECUTABLE, VERIFIED, or public Scenario Proof status;
- an implementation priority list;
- a mandate to create `platform_proofs/scenarios/<slug>/` for every row.

Adding, removing, or replacing a scenario after v1 freeze requires an explicit **portfolio governance** decision—not casual edits to this table.

---

## What FROZEN means here

| Term | Meaning in this catalog |
|------|-------------------------|
| **Selection status: FROZEN** | The **problem** is approved and frozen into portfolio v1 |
| **Lifecycle** (separate column) | Platform Proof package stage from `SCENARIO_SPEC.md` when a package exists |
| **No package** | No design package yet; frozen selection does not imply `create_scenario_proof.py` has been run |

Frozen **does not** mean: implemented, initialized, executable, verified, production-ready, or proof-accepted.

---

## Catalog order vs implementation order

Numbers **1–30** are stable **catalog identity and display order** only. They are not implementation priority unless a separate program record says otherwise.

Do not conflate:

- catalog number;
- implementation priority;
- current lifecycle or proof status.

---

## Portfolio selection philosophy

A frozen scenario **must**:

1. Represent a real business or technical problem.
2. Be an enterprise-class problem.
3. Have material financial, operational, legal, compliance, or security consequences.
4. Be a full E2E problem—not a single mechanism demo.
5. Exist independently of current Intergrax capabilities.
6. Be hard enough that it cannot honestly reduce to one or two trivial agent graphs.
7. Be an opportunity to discover platform gaps.
8. Later exercise the platform rather than bypass it.
9. Not be a demonstration of a single feature.
10. Support falsification—the scenario must be able to show what the platform **cannot** do.

**INTERGRAX FIT** is intentionally **out of scope** for this catalog. Problem first; platform mapping belongs in scenario design and implementation preparation.

---

## Frozen catalog (30)

| # | Scenario | Problem (summary) | Slug | Package | Lifecycle (`SCENARIO_SPEC`) | Frozen in v1 |
|---|----------|-------------------|------|---------|----------------------------|--------------|
| 1 | AI Incident Investigation | Operational investigation with incomplete, conflicting, or misleading evidence; competing hypotheses, falsification, and **UNRESOLVED** as a valid outcome | `ai_incident_investigation` | Yes | `EXECUTABLE` | Previously frozen |
| 2 | Strategic Decision Council / Decision Engine | Complex strategic decision requiring multiple perspectives, conflicting assessments, evidence, and an auditable final disposition | `strategic_decision_council` | Yes | `DESIGN` | Previously frozen |
| 3 | Verified Product Identification | Establish true product identity in a huge, noisy catalog from incomplete natural-language input | `verified_product_identification` | Yes | `IMPLEMENTATION_INITIALIZED` | Previously frozen |
| 4 | Enterprise Payment Uncertainty Recovery | A financial operation ran but actual outcome is unknown; establish truth before retry, compensation, or further execution | `enterprise_payment_uncertainty_recovery` | Yes | `IMPLEMENTATION_INITIALIZED` | Previously frozen |
| 5 | Shared RAG Wrong-Documents Blast Radius | Shared retrieval/RAG delivers wrong documents to many apps; detect contamination, blast radius, and limit impact without global shutdown | `shared_rag_wrong_documents_blast_radius` | No | — | Previously frozen |
| 6 | Critical Payment Path with Unhealthy AI Dependency | Critical payment flow depends on unstable AI; business must survive AI degradation without unsafe double effects | `critical_payment_path_unhealthy_ai_dependency` | No | — | Previously frozen |
| 7 | Mid-flight Business Policy Change | Business rules change during a long process; decide which executions stay on old policy snapshot vs apply new rules | `mid_flight_business_policy_change` | No | — | Previously frozen |
| 8 | Indirect Prompt Injection → Privileged Action | Untrusted content tries to drive a privileged external operation via the agent | `indirect_prompt_injection` | Yes | `EXECUTABLE` | Newly frozen |
| 9 | Cross-Tenant Isolation Failure | One tenant’s agent reaches another tenant’s data, memory, RAG, cache, tools, or evidence | `cross_tenant_isolation_failure` | No | — | Newly frozen |
| 10 | Delegated Authority / Confused Deputy | Agent has broader technical rights than the requesting user; must not use them for user-forbidden actions | `delegated_authority_confused_deputy` | No | — | Newly frozen |
| 11 | Persistent Memory Poisoning | Malicious or wrong information persists in memory and affects future runs, users, or decisions | `persistent_memory_poisoning` | Yes | `DESIGN` | Newly frozen |
| 12 | Multi-Agent Cascading Failure | One agent’s wrong output is treated as truth downstream, collapsing the whole process | `multi_agent_cascade_containment` | Yes | `DESIGN` | Newly frozen |
| 13 | Deceptive HITL Approval | Human approval exists but the operator sees manipulated or incomplete description and approves the wrong action | `deceptive_hitl_approval` | No | — | Newly frozen |
| 14 | Agentic Cost & Resource Runaway | Agent(s) enter costly loops, multiplying model/tool/subagent use without control | `agentic_cost_runaway` | Yes | `DESIGN` | Newly frozen |
| 15 | Emergency Agent Fleet Containment | Dangerous model, tool, strategy, or plugin defect requires fast partial fleet stop/limit without killing all work | `emergency_agent_fleet_containment` | No | — | Newly frozen |
| 16 | Compromised Plugin / Software Supply Chain | Trusted plugin, MCP server, dependency, or adapter is compromised or dangerously changed | `software_supply_chain_compromise` | Yes | `DESIGN` | Newly frozen |
| 17 | Privileged Infrastructure Change with Safe Rollback | Privileged production infra change may partially succeed; verify, stop, or rollback/compensate safely | `privileged_infrastructure_change` | Yes | `DESIGN` | Newly frozen |
| 18 | Identity Revocation During a Long-Running Execution | User or agent loses authorization mid multi-hour process | `identity_revocation_during_execution` | No | — | Newly frozen |
| 19 | Model / Agent Release Regression & Safe Rollback | New model/agent/prompt/retriever/policy passes tests but harms production subsets | `model_agent_release_regression` | No | — | Newly frozen |
| 20 | Cross-Border Data Residency & Sovereignty Enforcement | Customer or data class must not leave permitted geography despite dynamic provider/model/storage/tool choice | `cross_border_data_residency` | No | — | Newly frozen |
| 21 | Right-to-Erasure vs Legal Hold & Backups | Erasure request conflicts with legal hold, retention, backups, embeddings, cache, and memory | `governed_data_erasure` | Yes | `DESIGN` | Newly frozen |
| 22 | Regulated Automated Decision Appeal & Reconstruction | Months later, reconstruct regulated AI decision basis: data, model, policy, evidence, versions, human involvement | `regulated_decision_appeal_reconstruction` | No | — | Newly frozen |
| 23 | Regional Outage — Resume Thousands of In-Flight Executions | Region fails during thousands of runs; after recovery, know real effects and what may resume safely | `regional_outage_inflight_recovery` | No | — | Newly frozen |
| 24 | External API / Schema Drift Mid-Execution | External provider changes contract or semantics during a long process | `external_api_schema_drift` | No | — | Newly frozen |
| 25 | Upstream Data Corruption → Correction Propagation & Selective Replay | Source data was wrong; find dependents, invalidate poisoned results, replay selectively | `upstream_data_corruption_selective_replay` | No | — | Newly frozen |
| 26 | Disconnected Edge Operation & Later Reconciliation | Edge operates offline with local decisions; reconnect requires safe reconciliation of divergent worlds | `disconnected_edge_reconciliation` | No | — | Newly frozen |
| 27 | Duplicate / Out-of-Order Event Storm | After failure/retry/failover, duplicate and misordered events must not repeat business effects | `duplicate_out_of_order_event_storm` | No | — | Newly frozen |
| 28 | Separation of Duties Under Delegated Automation | One person or agent must not prepare, approve, and execute a critical operation when policy requires independent roles | `separation_of_duties_automation` | No | — | Newly frozen |
| 29 | Shadow / Rogue Agent Lifecycle | Many team-created agents lose owners, stale credentials, old models, or operate outside governance | `shadow_rogue_agent_lifecycle` | No | — | Newly frozen |
| 30 | Emergency Product Recall / Stop-Ship | Defective batch requires tracing dependents, stopping sale/ship, and auditable safe state across systems | `emergency_product_recall_stop_ship` | No | — | Newly frozen |

**Lifecycle column** reflects `SCENARIO_SPEC.md` YAML at v1 freeze time. Reconcile from package frontmatter after any lifecycle change; do not infer lifecycle from selection status.

---

## Slug and identity notes

| Catalog # | Note |
|-----------|------|
| 4 | **Not** the same problem as `payment_exception_recovery` (template DESIGN scaffold: “Autonomous Payment Exception Resolution Under Partial Failure”). Do not merge identities without portfolio decision. |
| 12 | Package title: “Multi-Agent **Incident Response** Cascade Containment”; frozen problem is cascading failure propagation. Keep slug `multi_agent_cascade_containment`. |
| 21 | Slug `governed_data_erasure`; scenario title: “Governed Personal Data Erasure Under Conflicting Retention Obligations”. |

Rows **5–7** and **9–10, 13, 15, 18–20, 22–30** have **no** `platform_proofs/scenarios/<slug>/` package at freeze time. Creating packages is a separate gated process (`create_scenario_proof.py` / quality gate).

---

## EXISTS IN REPO / NOT PART OF FROZEN-30

Design or template packages that are **not** portfolio v1 selections (do not delete; out of scope for this catalog):

| Slug | Lifecycle (at freeze) |
|------|------------------------|
| `vendor_payment_fraud` | `DESIGN` |
| `autonomous_production_remediation` | `DESIGN` |
| `payment_exception_recovery` | `DESIGN` |

---

## Related maintainer records

| Record | Relationship |
|--------|----------------|
| [`PRODUCT_PORTFOLIO_SELECTION.md`](../product-portfolio/PRODUCT_PORTFOLIO_SELECTION.md) | Commercial **product** portfolio—orthogonal to this **scenario** portfolio |
| [`ENTERPRISE_RELIABILITY_LAYER_IMPLEMENTATION_PLAN.md`](ENTERPRISE_RELIABILITY_LAYER_IMPLEMENTATION_PLAN.md) | ERL program; catalog #4 is the flagship ERL qualification scenario instance (ERL-QUAL-004) |
| [`platform_proofs/README.md`](../../../../platform_proofs/README.md) | Proof Library gateway and authoring workflow |

---

## Change control

After v1 freeze:

1. Open an explicit portfolio change (addendum or v2) with rationale.
2. Update this table and counts; preserve historical rows or dated addenda per [`PRODUCT_PORTFOLIO_SELECTION.md`](../product-portfolio/PRODUCT_PORTFOLIO_SELECTION.md) integrity pattern.
3. Do **not** silently alter lifecycle in this file—read from `SCENARIO_SPEC.md` or mark package absent.
