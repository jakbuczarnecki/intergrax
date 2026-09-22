# ADR-GR-12-MEMORY-SPECIALIZED-GOVERNANCE-BOUNDARY

| Field | Value |
| ----- | ----- |
| **Status** | Accepted — GR-12-A4-R3 |
| **Date** | 2026-09-22 |
| **Task** | `GR-12-A4-R3 — Specialized Memory Governance ADR` |

---

## Context

GR-12 residual path `CP-MEM-SPECIALIZED-MUTATION` was inventoried as using **memory-native** governance (`MemoryGovernanceEvaluationRequest` → `MemorySecurityGovernanceService`) rather than **CLA-04** (`ControlPlaneMutationAuthorizationBoundary`). The platform must decide whether this is a second control-plane permission engine, a specialized domain authority accepted by GR-12, or not GR-12 control-plane at all.

**Hard constraint:** for one consequential memory mutation there must not be two independent, peer permission authorities (`MemoryGovernanceDecision` **and** `ControlPlaneMutationDecision` both granting mutation).

Memory already implements a **contract-first**, **pluginable** policy stack (`MemorySecurityStrategySet`) with **fail-closed** merge semantics. CLA-04 is not a goal in itself; GR-12 requires consistent authority semantics.

---

## Existing Memory governance contracts

| Item | Module / type |
| ---- | ------------- |
| Request | `intergrax.memory.contracts.memory_security_governance.MemoryGovernanceEvaluationRequest` |
| Context | `MemorySecurityContext` (`RequestIdentity`, `MemoryControlScopeRef`, `MemoryGovernanceOperation`, optional `reference_time`) |
| Decision | `MemoryGovernanceDecision` (`outcome`, `reason_code`, `policy_id`, `policy_version`, `operation`, optional trust/classification/retention/constraints/`subject_memory_id`) |
| Denial | `MemoryGovernanceDenied` (carries `decision`; zero mutation) |
| Service | `intergrax.memory.memory_security_governance_service.MemorySecurityGovernanceService.evaluate` |
| Enforcement helper | `intergrax.memory.memory_specialized_mutation_governance.enforce_specialized_memory_mutation` |
| Policy ports | `MemoryAuthorizationPolicy`, `MemoryTrustEvaluationPolicy`, `MemoryAdmissionPolicy`, `MemoryGovernancePolicy`, `MemoryRetentionPolicy` bundled in `MemorySecurityStrategySet` |
| Canonical revision | `CanonicalMemoryGovernanceSourceAuthority` + `validate_canonical_governance_source_snapshot` |

Operations enum includes `REMEMBER`, `RECALL`, `PROMOTE`, `SUPERSEDE`, `DELETE`, `COMPACT`, `PROJECT`, `UPDATE`. Outcomes include `ALLOW`, `DENY`, `ALLOW_WITH_CONSTRAINTS`, `REQUIRE_REVIEW`.

**No** `ControlPlaneMutation*` types appear under `intergrax/memory/`.

---

## Production mutation inventory

| Mutation | Entry point | Context | Governance | GR-12 CP applicability |
| -------- | ----------- | ------- | ---------- | ------------------------ |
| User profile remember / supersede / delete | `default_memory_control_plane` | A — execution | `MemorySecurityGovernanceService` via `_enforce_governance` | **Not applicable** |
| Long-horizon compact / promote | `long_horizon_memory_service` | B — background lifecycle (execution-triggered) | `enforce_specialized_memory_mutation` | **Not applicable** |
| Procedure remember / supersede / deprecate / delete | `procedural_memory_service` | A — execution | `enforce_specialized_memory_mutation` | **Not applicable** |
| Entity projection / relation index writes | `entity_memory_indexing` | A — execution | `enforce_specialized_memory_mutation` | **Not applicable** |
| Procedure projection indexing | `procedural_memory_indexing` | A — execution | `enforce_specialized_memory_mutation` | **Not applicable** |
| Recall / disclosure filters | `entity_temporal_memory_service`, `procedural_memory_service`, `default_memory_reference_reader` | A — execution (read) | `filter_recall_candidates` / disclosure helpers | **Not applicable** (not mutation) |
| Live operator admin memory API | *(none in `intergrax.applications` production)* | C — would be operator | *Not implemented* | **Applicable only when introduced** |

**Bypass note (in scope):** `summary_compressor` compacts text without store mutation. `session_turn_index_service` may delete vector index entries — vector data-plane, not memory governance authority. No production **live operator** memory mutation endpoint/CLI was found.

---

## Caller classification

| Caller | Execution | Background | Operator | Bootstrap | Notes |
| ------ | --------- | ---------- | -------- | --------- | ----- |
| `default_memory_control_plane` | yes | — | — | — | Execution `RequestIdentity` |
| `long_horizon_memory_service` | yes | yes | — | — | Compaction during agent runs |
| `procedural_memory_service` | yes | — | — | — | |
| `entity_memory_indexing` | yes | — | — | — | Projection from LTM entries |
| `procedural_memory_indexing` | yes | — | — | — | |
| Applications wiring | — | — | — | compose | Injects `MemorySecurityGovernanceService`, no admin API |

---

## Authority ownership

| Responsibility | Current owner |
| -------------- | ------------- |
| Request construction | Domain services (`*_memory_service`, `default_memory_control_plane`, indexing helpers) |
| Tenant/scope authority | `MemorySecurityContext.scope` (`MemoryControlScopeRef` / entity scope mapping) |
| Policy evaluation | `MemorySecurityGovernanceService` → injected `MemorySecurityStrategySet` |
| Final permission decision | Merged `MemoryGovernanceDecision` (strictest outcome across policies) |
| Evidence | `MemoryGovernanceDecision` + memory diagnostics emitters (**not** GR-8 fact in R3) |
| Mutation execution | Domain store ports (`upsert_*`, `delete_*`, control plane capabilities) |

**Canonical authority (production today):** `MemorySecurityGovernanceService` — **not** CLA-04.

---

## Identity / scope model

- **Actor:** `RequestIdentity` on execution paths (not a synthetic default in governance service).
- **Scope:** `MemoryControlScopeRef` with `MemoryControlPlaneScope` (tenant + user binding); entity scopes require `user_id` for mutation (`memory_control_scope_from_entity_scope`).
- **Resource:** `MemoryGovernanceTarget` / record snapshots (`memory_id`, `revision`, `kind`, optional scope).
- **Subject vs actor:** Memory records carry provenance/trust/governance metadata; policies may distinguish trust escalation and cross-scope (`CROSS_SCOPE` reason).
- **Shared / agent memory:** Scoped via control scope refs and entity scope — not flattened to “tenant only”.

---

## Policy pluginability

**Yes.** External implementations can replace evaluation by supplying a custom `MemorySecurityStrategySet` (five `Protocol` policies) via composition (`specialized_memory_wiring`, `procedural_memory_wiring`, `resolve_memory_security_governance_service`). Default strategies live in `intergrax.memory.strategies.defaults.memory_security_governance`.

---

## Evidence model

`MemoryGovernanceDecision` records **policy_id**, **policy_version**, **reason_code**, **outcome**, **operation**, and optional trust/classification/retention fields. There is **no** platform-wide `decision_id` field in the contract. This differs from `ControlPlaneMutationAuthorizationEvidence` by design; equivalence is not required for execution-domain writes.

---

## GR-12 applicability

**Decision: split classification (Option D), closed for current production.**

| Surface class | GR-12 control-plane |
| ------------- | ------------------- |
| Execution / data-plane memory writes | **NOT_APPLICABLE** |
| Background lifecycle (compaction/promotion during runs) | **NOT_APPLICABLE** |
| Live operator administrative memory mutation | **No surface today**; when added → **APPLICABLE** with **single** authority (memory validation → CLA-04 adapter pattern — **not** dual ALLOW) |

Catalog path `CP-MEM-SPECIALIZED-MUTATION` is reclassified to **NOT_APPLICABLE** for GR-12 (consequential **data** mutations ≠ live control-plane configuration/operator-visible platform authority changes).

Memory-native governance remains **authoritative** for all current mutation paths (**Option B** semantics within the execution domain, without CLA-04).

---

## Architecture options

| Option | Decision |
| ------ | -------- |
| Memory-native authority (execution/background) | **ACCEPT** |
| CLA-04 adapter for all memory writes today | **REJECT** |
| Split: execution memory-native / future operator CLA-04 | **ACCEPT** |
| Dual authorization (memory ALLOW + CLA-04 ALLOW) | **REJECT** |

---

## Decision

1. **Canonical permission authority (production):** `MemorySecurityGovernanceService.evaluate` → `MemoryGovernanceDecision` → domain owner executes store mutation.
2. **CLA-04:** **not** used on current production memory mutation paths; **not** required to “qualify” execution writes.
3. **GR-12 CP-MEM:** **NOT_APPLICABLE** for current inventory; **revisit** when a live operator memory mutation API is introduced.
4. **Next bounded task:** `GR-12-A4-R3-R1 — Memory Specialized Governance Qualification` (prove enterprise invariants; no runtime adapter in R3).
5. **GR-12 overall** remains **IN PROGRESS** until R3-R1 and final GR-12 certification.

---

## Rejected alternatives

- Dual independent `MemoryGovernanceDecision` and `ControlPlaneMutationDecision` for the same mutation.
- Mechanical rewrite of all memory writes to CLA-04 (fake unification; loses specialized trust/retention/admission semantics).
- Global memory mutation executor in governance/runtime.
- Default ALLOW or synthetic principal when governance is missing.
- Provider/storage-coupled policy (Qdrant/Postgres/Redis-specific authority).

---

## TOCTOU / revision semantics

**Applicable** for specialized projections via `CanonicalMemoryGovernanceSourceAuthority` and revision checks on snapshots. User-profile control plane uses entry **revision** on supersede/delete paths. There is **no** CLA-04-style post-authorization CAS for memory domain today. Stale authorization for **hypothetical** live operator APIs would require a future bounded design (not invented in R3).

---

## Fail-closed semantics

| Condition | Mutation allowed? |
| --------- | ----------------: |
| `ALLOW` / `ALLOW_WITH_CONSTRAINTS` | yes (subject to `permits_mutation()`) |
| `DENY` / `REQUIRE_REVIEW` on mutation ops | **no** |
| Evaluator exception | **no** (`POLICY_FAILURE` deny) |
| Missing `strategies` | **no** (`POLICY_MISSING`) |

---

## Layer boundaries

Memory policy contracts remain in `intergrax.memory.contracts`. Runtime CLA-04 boundary is not imported by memory domain code. Applications compose memory governance via wiring modules only.

---

## Non-goals (R3)

- Memory governance migration to CLA-04 on execution paths.
- Changing `MemorySecurityGovernanceService` merge logic.
- Memory persistence / vector / catalog / GR-10 changes.
- GR-8 fact emission for memory decisions.

---

## Next bounded task

**GR-12-A4-R3-R1 — Memory Specialized Governance Qualification** — prove pluginability, fail-closed, scope binding, and single-authority invariants on production mutation surfaces documented above.

Qualification SSOT: `tests/qualification/governance/gr12/gr12_a4_r3_memory_architecture_decision.py`.
