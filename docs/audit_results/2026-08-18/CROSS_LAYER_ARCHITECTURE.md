# CROSS_LAYER_ARCHITECTURE - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Audit unit:** CROSS_LAYER_ARCHITECTURE
- **Owning architecture/program:** PLATFORM_FOUNDATION · INTERGRAX_ARCHITECTURE_PRINCIPLES · intergrax_runtime_architecture hub · SYSTEM_INVARIANTS · MATURITY_TAXONOMY · TIER3_APPLICATION_ENVIRONMENT · GOVERNED_EXECUTION (meta-architecture / documentation topology)
- **Tier(s):** cross-layer - documentation topology, invariant index, composition qualification concept, control-plane governance taxonomy, maturity requalification semantics, remediation DAG requirement
- **audited_sha:** `ac2a7107393cbf63953c1cfbc9757891aed91de0`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/architecture/intergrax_runtime_architecture.md`
  - `docs/project/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`
  - `docs/project/architecture/PLATFORM_FOUNDATION.md`
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/architecture/GOVERNED_EXECUTION.md`
  - `docs/project/technical/guides/SYSTEM_INVARIANTS.md`
  - `docs/project/technical/guides/MATURITY_TAXONOMY.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md`
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/maintainers/plans/GOVERNED_EXECUTION.md`
- **Scope in:**
  - authoritative platform ownership topology vs runtime hub index
  - cross-layer invariant index freshness vs post-June Protocol-v2 findings
  - composition-level production qualification closure concept
  - control-plane mutation governance taxonomy gap
  - maturity invalidation / requalification semantics for accepted findings
  - cross-layer remediation dependency graph requirement before campaign implementation
  - positive controls preserving four-tier model and domain ownership
- **Scope out:**
  - runtime remediation implementation
  - source/test/CI/script changes
  - building the final detailed remediation DAG in this persistence task
  - promoting every architecture markdown file to a domain
  - monolithic ProductionEngine or GovernanceEngine
- **Prior audit reference(s):** all thirty-five completed Protocol-v2 layers at campaign baseline; this layer synthesizes meta-architecture gaps only - does not re-open per-domain findings
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `5111162eb34477c8f507d427377c114214b9f7ac`

## Scope / ownership mapping

| Concept | Canonical ownership |
|---------|---------------------|
| Audit unit (Protocol v2 layer code) | **CROSS_LAYER_ARCHITECTURE** |
| Architecture artifact classification / owner register | **PLATFORM_FOUNDATION** + runtime hub |
| Cross-layer MUST/MUST NOT index | **SYSTEM_INVARIANTS** - compact index; domains remain semantic owners |
| Composition qualification closure | **TIER3_APPLICATION_ENVIRONMENT** evaluates closure; domains own domain qualification |
| Control-plane mutation boundary taxonomy | **GOVERNED_EXECUTION** - domain executors remain specialized |
| Maturity requalification semantics | **MATURITY_TAXONOMY** + platform architecture governance |
| Cross-layer remediation DAG | **campaign rollup** / Platform Foundation coordination - **CLA-REMEDIATION-DAG-INTEGRITY** |
| Per-layer report | `docs/audit_results/2026-08-18/CROSS_LAYER_ARCHITECTURE.md` |

## Architecture hierarchy under audit

```text
INTERGRAX_ARCHITECTURE_PRINCIPLES (META_ARCHITECTURE)
        │
        ▼
intergrax_runtime_architecture.md (META_ARCHITECTURE hub + indexes)
        │
        ├── DOMAIN pairs (24 primary index + additional canonical pairs)
        ├── FEATURE pairs (capabilities/README coordination)
        └── SUPPORTING_MODEL / SATELLITE (runtime graph, dependency model, domain satellites)
        │
        ▼
SYSTEM_INVARIANTS (cross-layer index - not second canon)
        │
        ▼
MATURITY_TAXONOMY (four-axis vocabulary + requalification semantics)
        │
        ▼
Per-domain Protocol-v2 target invariants (semantic owners)
```

## Executive summary

**Verdict: FAIL.** Five accepted HIGH and one accepted MEDIUM finding show that the runtime hub falsely advertises a complete 24-domain register while canonical owners such as **GOVERNED_EXECUTION**, **AGENT_DISTRIBUTION**, **PLATFORM_PLUGINS**, and **PROOF_RECEIPTS** are absent; **SYSTEM_INVARIANTS** still references a stale domain count and omits durable post-June cross-layer rules; no architecture defines composition-level production qualification closure; **Governed Execution** lacks a first-class **CONTROL_PLANE_MUTATION** evaluation boundary; maturity claims can remain advertised while accepted CRITICAL security findings invalidate prior posture without explicit requalification; and the campaign lacks one cross-layer remediation DAG before implementation. Positive controls: four-tier topology, Nexus/UER/Agent split, domain runtime semantics, single-owner principles, feature coordination without domain replacement, and consolidation-over-parallel-subsystems posture all remain sound. Remediation is **ACCEPTED / PLANNED** meta-architecture sync only - not implemented.

## Verdict

**FAIL** - 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-01 (CLA-01)

- **Severity:** HIGH
- **Category:** ARCHITECTURE TOPOLOGY / OWNERSHIP AUTHORITY
- **Status at publication:** ACCEPTED
- **Remediation block:** CLA-CANON-TOPOLOGY-INTEGRITY
- **Claim falsified:** `intergrax_runtime_architecture.md` is the complete current architecture ↔ plan owner register.
- **Observation:** The hub calls itself the "complete 24-domain architecture ↔ plan register" and lists exactly twenty-four domain pairs. The repository also contains canonical platform architecture / implementation-plan owners not represented in that index, including **PLATFORM_PLUGINS**, **PROOF_RECEIPTS**, **AGENT_DISTRIBUTION**, and **GOVERNED_EXECUTION**, plus supporting canonical models requiring explicit classification. [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../project/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) requires exactly one architectural owner per reusable capability.
- **Location:**
  - `docs/project/architecture/intergrax_runtime_architecture.md` - header claim; § Domain pair index (24)
  - `docs/project/architecture/PLATFORM_PLUGINS.md`, `PROOF_RECEIPTS.md`, `AGENT_DISTRIBUTION.md`, `GOVERNED_EXECUTION.md` - canonical pairs absent from hub index
- **Impact:** Operators and agents cannot trust the hub as authoritative topology; ownership drift hides duplicate-authority risk.
- **Confidence:** CONFIRMED

### AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-02 (CLA-02)

- **Severity:** HIGH
- **Category:** CROSS-LAYER CANON / ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** CLA-CANON-TOPOLOGY-INTEGRITY
- **Claim falsified:** `SYSTEM_INVARIANTS.md` is the current compact cross-layer authority indexed to the live platform topology.
- **Observation:** The guide states rules are spread across twenty-two domain pairs and serves as the single cross-layer authority, yet the platform topology is larger and many post-June cross-layer invariants live only in domain Protocol-v2 sections or audit reports. The same document requires new cross-layer rules to be reflected there; [`MATURITY_TAXONOMY.md`](../../project/technical/guides/MATURITY_TAXONOMY.md) A4 explicitly relies on cross-layer mapping through **SYSTEM_INVARIANTS**.
- **Location:**
  - `docs/project/technical/guides/SYSTEM_INVARIANTS.md` - §1 Purpose (22 domain pairs); cross-layer sections vs current campaign findings
- **Impact:** Cross-layer audits and A4 maturity claims may reference stale or incomplete invariant coverage.
- **Confidence:** CONFIRMED

### AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-03 (CLA-03)

- **Severity:** HIGH
- **Category:** PRODUCTION AUTHORITY / COMPOSITION QUALIFICATION
- **Status at publication:** ACCEPTED
- **Remediation block:** CLA-PRODUCTION-QUALIFICATION-INTEGRITY
- **Claim falsified:** A fully materialized application composition has a canonical architecture model for proving all mandatory components are simultaneously qualified for one target environment.
- **Observation:** The platform has multiple legitimate local production/maturity mechanisms (A/I/P/E, STRICT, PRODUCT profile, agent production gates, plugin/provider qualification, hosting maturity, Tier-3 production gates, proof evidence). No canonical architecture defines composition-level closure: given **TargetEnvironment** + materialized runtime identity + exact application/environment revision + component qualification references + mandatory evidence freshness → **QUALIFIED | NOT_QUALIFIED | STALE | INCOMPLETE**.
- **Location:**
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` - local gates/maturity without composition closure model
  - `docs/project/technical/guides/MATURITY_TAXONOMY.md` - axis vocabulary without composition evaluator
- **Impact:** Product hosts may advertise partial qualification while mandatory components remain unqualified or stale.
- **Confidence:** CONFIRMED

### AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-04 (CLA-04)

- **Severity:** HIGH
- **Category:** GOVERNANCE TOPOLOGY / CONTROL PLANE
- **Status at publication:** ACCEPTED
- **Remediation block:** CLA-CONTROL-PLANE-GOVERNANCE-INTEGRITY
- **Claim falsified:** Governed Execution taxonomy covers state-changing control-plane mutations with a first-class evaluation boundary.
- **Observation:** Governed Execution defines evaluation boundaries for model/agent/tool/side-effect/output/post-run execution but not control-plane mutation. The platform contains multiple governed state-changing control planes (Agent Distribution activation/rollback, AHI apply/rollback, ECP capacity mutations, live task autonomy changes, plugin/config activation/admission). Domains define their own authorization/gate semantics without a shared **CONTROL_PLANE_MUTATION** authority context.
- **Location:**
  - `docs/project/architecture/GOVERNED_EXECUTION.md` - Governance Evaluation Point taxonomy and G3B coverage table
- **Impact:** Control-plane mutations may bypass consistent authority/evidence semantics across domains.
- **Confidence:** CONFIRMED

### AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-05 (CLA-05)

- **Severity:** HIGH
- **Category:** MATURITY AUTHORITY / REQUALIFICATION
- **Status at publication:** ACCEPTED
- **Remediation block:** CLA-PRODUCTION-QUALIFICATION-INTEGRITY
- **Claim falsified:** Accepted audit findings trigger explicit maturity impact evaluation before prior claims remain valid.
- **Observation:** [`MATURITY_TAXONOMY.md`](../../project/technical/guides/MATURITY_TAXONOMY.md) defines P0 risk but no architecture-level rule that accepted findings must evaluate impact on existing maturity claims. The Tier-3 hub still advertises A4/I3/P3/E3 while accepted campaign CRITICAL security defects affect that composition path. Severity alone must not auto-downgrade, but production/evidence-safety findings require explicit requalification decisions.
- **Location:**
  - `docs/project/technical/guides/MATURITY_TAXONOMY.md` - maintenance without finding-impact semantics
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` - Current maturity table
- **Impact:** Stale maturity labels may overstate production posture after new accepted evidence.
- **Confidence:** CONFIRMED

### AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-06 (CLA-06)

- **Severity:** MEDIUM
- **Category:** REMEDIATION ARCHITECTURE / DEPENDENCY MANAGEMENT
- **Status at publication:** ACCEPTED
- **Remediation block:** CLA-REMEDIATION-DAG-INTEGRITY
- **Claim falsified:** The campaign has one cross-layer remediation dependency graph before normal implementation starts.
- **Observation:** Two hundred eleven accepted findings precede this layer with many remediation blocks. Per-layer recommended orders and cross-references exist, but no single DAG describes prerequisites, shared canonical primitives, superseded/merged blocks, safe parallelism, and final recertification ordering with vocabulary `depends_on`, `shares_authority_with`, `merge_into`, `supersedes`, `can_parallelize_with`, `verified_by`.
- **Location:**
  - `docs/audit_results/2026-08-18/README.md` - remediation rollup without unified DAG
- **Impact:** Parallel duplicate abstractions and unsafe remediation ordering risk during campaign implementation.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Four-tier architecture remains sound | NOT falsified |
| Nexus / UER / Agent responsibility split remains sound | NOT falsified |
| Domain runtime semantics remain domain-owned | NOT falsified |
| Architecture Principles single-owner rule remains correct | NOT falsified |
| Feature docs coordinate but do not replace domain owners | NOT falsified |
| Governed Execution remains one governance plane with specialized enforcement owners | NOT falsified |
| Observability remains canonical execution evidence owner | NOT falsified |
| Application Hosting remains lifecycle owner | NOT falsified |
| ECP remains capacity owner | NOT falsified |
| AHI remains adaptation owner | NOT falsified |
| LKW remains product adopter/proof, not platform owner | NOT falsified |
| No evidence supports a platform rewrite | NOT falsified |
| Cross-layer remediation should consolidate authorities, not introduce parallel subsystems | NOT falsified |

## Duplicate ownership / cross-links

| Existing finding / domain | Relationship |
|-----------------------------|--------------|
| **TIER_LAYER_BOUNDARIES / TL-FIX-*** | Tier-boundary enforcement - CLA-01/02 address documentation topology, not duplicate TL-FIX scope |
| **PLATFORM_FOUNDATION / PF-*** | Foundation proof gates - cross-link; CLA does not re-open PF proof-runner defects as new findings |
| **SECURITY_BOUNDARIES CRITICAL findings** | Evidence invalidating composition maturity - cross-link CLA-05; do not duplicate SEC remediation blocks |
| **POLICY_GOVERNANCE / PG-FIX-*** | Execution-time governance - CLA-04 extends taxonomy; PG-FIX remains execution spine |
| **END_TO_END_SYSTEM / E2E-CONTROL-AUTHORITY-INTEGRITY** | Live task autonomy bypass - consumer of CLA-04 target boundary |
| **Per-domain Protocol-v2 sections** | Semantic owners for rules indexed by CLA-02 - do not copy audit findings verbatim into SYSTEM_INVARIANTS |

## Systemic consolidation map

| Theme | Consolidation direction |
|-------|-------------------------|
| Ownership topology | One classification register in runtime hub + PLATFORM_FOUNDATION meta-invariants |
| Cross-layer rules | Compact SYSTEM_INVARIANTS index pointing to domain canon |
| Production posture | Composition qualification closure at Tier-3; domain qualification unchanged |
| Control-plane mutations | CONTROL_PLANE_MUTATION taxonomy in Governed Execution; domain executors unchanged |
| Maturity claims | Explicit requalification semantics; severity alone does not auto-downgrade |
| Remediation ordering | Single campaign DAG before implementation - built in CAMPAIGN_ROLLUP, not this task |

## Root-cause remediation grouping

### CLA-CANON-TOPOLOGY-INTEGRITY - ownership register + invariant index

**Priority:** P0  
**Findings:** CLA-01, CLA-02  
**Owner:** PLATFORM_FOUNDATION / meta architecture  

One complete current ownership topology and one current cross-layer invariant index.

### CLA-PRODUCTION-QUALIFICATION-INTEGRITY - composition closure + maturity requalification

**Priority:** P0  
**Findings:** CLA-03, CLA-05  
**Owners:** TIER3_APPLICATION_ENVIRONMENT · MATURITY_TAXONOMY / platform architecture governance  

Production posture is composition-qualified; maturity is requalified when accepted evidence invalidates prior assumptions.

### CLA-CONTROL-PLANE-GOVERNANCE-INTEGRITY - control-plane mutation boundary

**Priority:** P0  
**Finding:** CLA-04  
**Owner:** GOVERNED_EXECUTION  
**Consumers:** AGENT_DISTRIBUTION · AHI · ECP · task-control/runtime · Platform Plugins activation/admission  

One control-plane permission/evidence semantic boundary; specialized domain executors - no universal mutation executor.

### CLA-REMEDIATION-DAG-INTEGRITY - campaign dependency graph

**Priority:** P0 before campaign remediation  
**Finding:** CLA-06  
**Owner:** campaign rollup / Platform Foundation coordination  

Build the final cross-layer remediation DAG before implementation. **Not built in this persistence task.**

## Architecture / plan sync state

| Doc | Section | Status |
|-----|---------|--------|
| `docs/project/architecture/intergrax_runtime_architecture.md` | Architecture artifact classification register | SYNCED |
| `docs/project/technical/guides/SYSTEM_INVARIANTS.md` | Cross-layer index refresh | SYNCED |
| `docs/project/architecture/PLATFORM_FOUNDATION.md` | Protocol v2 cross-layer meta-architecture target invariants | SYNCED |
| `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` | Composition qualification closure semantics | SYNCED |
| `docs/project/technical/guides/MATURITY_TAXONOMY.md` | Finding/evidence-driven requalification semantics | SYNCED |
| `docs/project/architecture/GOVERNED_EXECUTION.md` | CONTROL_PLANE_MUTATION evaluation point class | SYNCED |
| `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` | CLA-CANON-TOPOLOGY-INTEGRITY · CLA-REMEDIATION-DAG-INTEGRITY | SYNCED |
| `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` | CLA-PRODUCTION-QUALIFICATION-INTEGRITY | SYNCED |
| `docs/project/maintainers/plans/GOVERNED_EXECUTION.md` | CLA-CONTROL-PLANE-GOVERNANCE-INTEGRITY | SYNCED |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `ac2a7107393cbf63953c1cfbc9757891aed91de0`; documentation topology and meta-architecture claims only - no source re-audit beyond pinned SHA context.
- Final detailed remediation DAG intentionally deferred to CAMPAIGN_ROLLUP (**CLA-06**).
- Remediation not performed in this task.
- No source, test, CI, or script changes.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-CROSS_LAYER_ARCHITECTURE-01` … `06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted meta-architecture audit observations, target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED. The cross-layer remediation DAG is **PLANNED** here and **not** constructed in this task.
