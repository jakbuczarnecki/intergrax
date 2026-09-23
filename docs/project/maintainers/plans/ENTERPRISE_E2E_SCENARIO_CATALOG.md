# Enterprise E2E Scenario Catalog — v1 + v2 Addendum

**Document type:** Maintainer-level scenario portfolio record  
**Canonical SSOT:** Enterprise E2E Scenario Portfolio (this file)

| Portfolio lineage | Count | Status |
|-------------------|-------|--------|
| **v1 — frozen membership** | **30** (7 previously frozen · 23 newly frozen in v1 freeze) | **FROZEN** · **Last frozen:** 2026-09-14 |
| **v2 addendum** | **+4** (#31–#34) | **FROZEN** · **Addendum dated:** 2026-09-23 |
| **Current portfolio membership** | **34** | v1 historical selection **unchanged** + v2 addendum |

**v1 tasks:** SCENARIO-CATALOG-FREEZE-30-DOCS-R1 · **Authority ratification:** SCENARIO-CATALOG-FREEZE-30-DOCS-R2  
**v2 addendum task:** SCENARIO-PORTFOLIO-V2-ADDENDUM-34

**Catalog identity/order does not imply implementation priority.**

---

## Authority and boundaries

This file is the **Canonical Single Source of Truth (SSOT)** for the **Enterprise E2E Scenario Portfolio**: catalog numbers **1–34** (current membership), stable slug identity, selection-frozen problem identities, and dated portfolio lineage (**v1 frozen 30** + **v2 addendum +4**).

Historical **v1** remains the authoritative record of the **2026-09-14** freeze: problems **1–30**, **Selection status: FROZEN** at v1, and v1 portfolio membership (7 previously frozen · 23 newly frozen). **v2 addendum** appends **#31–#34** without renumbering or altering v1 rows.

| Role | Document |
|------|----------|
| **Frozen problem portfolio (this file)** | Selection-frozen enterprise E2E problems for future design, quality gates, and proof work |
| Framework / filesystem inventory | [`E2E_SCENARIO_FRAMEWORK_AUDIT.md`](../../architecture/E2E_SCENARIO_FRAMEWORK_AUDIT.md) |
| Public in-development scenario presentation | [`PROOF_LIBRARY.md`](../../proofs/PROOF_LIBRARY.md) |
| **Current** per-package lifecycle | `platform_proofs/scenarios/<slug>/SCENARIO_SPEC.md` YAML frontmatter |

**Current scenario lifecycle is never sourced from this catalog.** The authoritative current lifecycle is always the YAML frontmatter in `platform_proofs/scenarios/<slug>/SCENARIO_SPEC.md`.

**This catalog is not authority for:** current scenario-package lifecycle, implementation status, executable status, verified status, or public proof acceptance.

**This catalog is not:**

- proof acceptance, EXECUTABLE, VERIFIED, or public Scenario Proof status;
- an implementation priority list;
- a mandate to create `platform_proofs/scenarios/<slug>/` for every row.

Adding, removing, or replacing a scenario after a portfolio freeze requires an explicit **portfolio governance** decision—not casual edits to this table.

---

## What FROZEN means here

| Term | Meaning in this catalog |
|------|-------------------------|
| **Selection status: FROZEN** | The **problem** is approved and frozen into portfolio v1 |
| **Lifecycle at v1 freeze** (separate column) | Historical snapshot: `lifecycle` from `SCENARIO_SPEC.md` frontmatter **at portfolio v1 freeze** when a package existed; not updated when packages advance |
| **No package** | No design package yet; frozen selection does not imply `create_scenario_proof.py` has been run |

Frozen **does not** mean: implemented, initialized, executable, verified, production-ready, or proof-accepted.

---

## Catalog order vs implementation order

Numbers **1–34** are stable **catalog identity and display order** only (v1: **1–30** frozen 2026-09-14; v2 addendum: **31–34** frozen 2026-09-23). They are not implementation priority unless a separate program record says otherwise.

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

## Frozen catalog v1 (30) — historical membership (2026-09-14)

| # | Scenario | Problem (summary) | Slug | Package | Lifecycle at v1 freeze | Frozen in v1 |
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

**Lifecycle at v1 freeze** is a **historical snapshot** only: values were taken from `SCENARIO_SPEC.md` YAML when portfolio v1 was frozen (**Last frozen:** 2026-09-14). Do **not** update this column when package lifecycle changes later. For **current** lifecycle, read `platform_proofs/scenarios/<slug>/SCENARIO_SPEC.md` frontmatter. Do not infer lifecycle from selection status or from this snapshot.

---

## v2 addendum — portfolio extension (+4) — 2026-09-23

### Addendum rationale (selection only)

Portfolio **v1** (Frozen-30, 2026-09-14) remains historically correct and is **not** rewritten. Later external market research on consequential agent execution identified four additional problem identities that are **material and distinct** from the v1 set. They were added because the problems exist **independently of Intergrax**, not because of platform fit or proof results.

**Addendum selection does not mean:** commercial validation, production readiness, implementation, proof acceptance, willingness-to-pay, or implementation priority. Each new row means **problem selected for future falsification** only.

### v2 addendum catalog

| # | Scenario | Problem (summary) | Slug | Package | Added in v2 addendum |
|---|----------|-------------------|------|---------|----------------------|
| 31 | Unauthorized Commercial Commitment | Agent can invoke a business API, but the **business consequence** (refund, discount, credit, SLA/pricing/contract/purchase commitment, guarantee, commercial term change) exceeds the **organizational/commercial mandate** to bind the org — technical capability ≠ business authority | `unauthorized_commercial_commitment` | No | 2026-09-23 |
| 32 | Stale Evidence Before Consequential Action | Decision/plan relied on evidence that was **valid when assessed** but **no longer valid** before a consequential action executes (temporal validity / freshness of material evidence — not corruption or policy version alone) | `stale_evidence_before_consequence` | No | 2026-09-23 |
| 33 | Correct Action, Wrong Principal | Operation is business- and technically correct, but the **consequence is attributed** to the wrong principal, legal entity, tenant, account, or organizational context (`correct action + wrong principal = invalid consequence`) | `correct_action_wrong_principal` | No | 2026-09-23 |
| 34 | Purpose-Bound Data Egress / Trusted Tool Exfiltration | Individually legal **read** and **send/transfer** capabilities compose into a **forbidden information flow** (wrong purpose, destination, or data class) — not reducible to blocking a whole tool or DLP checkbox | `purpose_bound_data_egress` | No | 2026-09-23 |

No `platform_proofs/scenarios/<slug>/` packages exist for #31–#34 at addendum time. Creating packages is a separate gated process.

### Per-scenario addendum notes

#### #31 — `unauthorized_commercial_commitment`

| Field | Content |
|-------|---------|
| **Real problem** | Financial or contractual obligation is incurred at a value or under terms beyond the actor’s **business mandate**, while the underlying API call may be technically permitted. |
| **Why distinct** | **#10** (`delegated_authority_confused_deputy`): agent’s technical rights exceed the **requesting principal’s** rights. **#31**: operation may be within tool/IAM scope, but **binding commercial authority** for that specific outcome is missing. **#13** / **#28** address deceptive approval and role separation, not mandate-to-bind for commercial effects. |
| **Material consequence** | Financial loss, contractual liability, unauthorized commercial concessions, audit/regulatory exposure. |
| **Selection caveat** | Portfolio membership only — not validation, implementation, or proof acceptance. |

#### #32 — `stale_evidence_before_consequence`

| Field | Content |
|-------|---------|
| **Real problem** | Consequential action proceeds on a **stale evidence snapshot** after the world changed (e.g. verified supplier status invalidated before payment). |
| **Why distinct** | **#7**: **policy/rule** changes mid-flight. **#32**: **truth or validity of material evidence** changes. **#25**: data was **wrong/corrupted** upstream. **#32**: earlier evidence was **correct**, then became insufficient. **#11**: **poisoned** persistent memory — not required here. |
| **Material consequence** | Fraudulent or mistaken payments, wrong authorization, operational and financial harm. |
| **Selection caveat** | Portfolio membership only — not validation, implementation, or proof acceptance. |

#### #33 — `correct_action_wrong_principal`

| Field | Content |
|-------|---------|
| **Real problem** | Valid operation applied under the **wrong organizational principal** (legal entity, tenant, account, authority chain) without requiring a data leak or confused deputy. |
| **Why distinct** | **#9**: **cross-tenant isolation** failure (reach others’ data/memory/RAG). **#33**: may occur with correct data access and valid API — **wrong attribution of consequence**. **#10**: wrong **user vs agent credentials**; **#33**: wrong **entity on whose behalf** the effect occurs. |
| **Material consequence** | Cross-entity accounting/legal error, misallocated payments, compliance and contractual breach. |
| **Selection caveat** | Portfolio membership only — not validation, implementation, or proof acceptance. |

#### #34 — `purpose_bound_data_egress`

| Field | Content |
|-------|---------|
| **Real problem** | Composed **purpose-bound information flow violation**: sensitive data leaves through an allowed channel to a **non-allowed destination or purpose**. |
| **Why distinct** | **#8**: **untrusted content** drives privileged behavior. **#34**: **flow semantics** across allowed capabilities (trigger may vary). **#9**: tenant **data isolation** breach. **#16**: **compromised** supply chain. **#20**: **geography/residency** constraint — orthogonal to purpose/destination pairing of tools. |
| **Material consequence** | Confidentiality breach, regulatory exposure, loss of customer trust. |
| **Selection caveat** | Portfolio membership only — not validation, implementation, or proof acceptance. |

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

**v1 freeze (2026-09-14):** problems **1–30** — do not renumber, alter problem identity, or rewrite **Lifecycle at v1 freeze** / **Frozen in v1** cells to reflect later package state.

**v2 addendum (2026-09-23):** problems **#31–#34** appended per task SCENARIO-PORTFOLIO-V2-ADDENDUM-34; v1 table preserved as historical membership.

After any portfolio freeze:

1. Open an explicit portfolio change (addendum or new version) with rationale.
2. Update counts and lineage; preserve historical rows or dated addenda per [`PRODUCT_PORTFOLIO_SELECTION.md`](../product-portfolio/PRODUCT_PORTFOLIO_SELECTION.md) integrity pattern.
3. Do **not** rewrite v1 lifecycle snapshot cells; addenda use separate provenance columns/tables. For live lifecycle, use `SCENARIO_SPEC.md` only.
