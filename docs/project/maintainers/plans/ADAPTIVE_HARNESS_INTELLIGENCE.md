# Adaptive Harness Intelligence - Implementation Plan

**Architecture (1:1):** [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Last updated:** 2026-06-22 - **AHI-ADAS-00** ADAS top-level + satellite implementation plans; **P2-ARCH-10** AHI governance boundary.

**Cross-feature - Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md). AHI may later consume token optimization telemetry to recommend budgets/profiles, but production auto-apply remains forbidden until governance and quality gates explicitly allow it.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (ADAPTIVE_HARNESS_INTELLIGENCE plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_implementation_history.md`](plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_implementation_history.md), [`plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md). Phase AUDIT-IDEAL - **Planned** / open rows only. §6.1 maintenance queues - open P0/P1 only
- **Token Optimization:** read feature pair + row `TOKEN-AHI-1`; do not implement adaptive token policy until TOKEN-6 telemetry/regression gates exist.
- **Use** `Read` with offset/limit - open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Architecture doc alignment (P2-ARCH)

| ID | Task | Status |
|----|------|--------|
| **P2-ARCH-10** | Clarify AHI governance boundary and production auto-apply rule | **Done** (2026-06-20) |

Architecture: [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md#governance-boundary).

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_implementation_history.md`](plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_implementation_history.md) | implementation history |
| [`plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md) | ADAS / Agent Design Search detailed implementation plan |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Phase TOKEN-AHI - Adaptive token optimization recommendations (Frozen/Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md)
**Priority:** P3 / Frozen until TOKEN-6 telemetry and regression gates ship  
**Delivery rule:** recommendation-only first; no autonomous production auto-apply.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-AHI-1** | Code | P3 | Partial / Token Optimization side Done | AHI consumes Token Optimization telemetry to recommend compact/full profiles, budget adjustments, and escalation rules by task/step/source type | Recommendations are observable and reversible; no automatic compression or budget reduction without policy; quality-drop escalation to fuller context supported; uses existing token telemetry, no duplicate token accounting; `uv run pytest tests/unit/runtime/adaptive/ -q` |

**TOKEN-7A update:** TOKEN-7A now provides the Token Optimization-side advisory recommendation contract and policy-only helper (`recommend_token_optimization_action`). Production AHI adaptive consumption / auto-apply remains deferred. Recommendation-only remains the first allowed mode. No autonomous production auto-apply is introduced. TOKEN-AHI-1 remains the broader AHI integration target unless fully implemented in a future task.

**Explicit exclusions:** no autonomous production auto-apply, no tenant-wide learned compression policy without governance review, no adaptive strategy before token-vs-quality regression gates exist.

---

## Phase AUDIT-IDEAL - Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §25 · baseline **32/32 L3** · W-ADAPT **70/70 Done**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** - L4 runtime exists; gaps = production evidence + marketplace readiness

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-6.2 | §6 LLM | Live cost/latency/quality routing (shared LLM_ADAPTERS) | P2 | **Partial** - runtime adapter swap: [M-LLM-X.5](plan/LLM_ADAPTERS.md) |
| AUDIT-IDEAL-9.3 | §9 Orchestration | Dynamic execution strategy selection (shared ORCHESTRATION) | P2 | **Done** |
| AUDIT-IDEAL-12.2 | §12 Skills | Dynamic skill selection L4 hook (shared SKILLS) | P2 | **Done** |
| AUDIT-IDEAL-24.2 | §24 Cost | Automated cost optimization recommendations (shared UAEP) | P2 | **Done** |
| AUDIT-IDEAL-AHI.1 | §25 AHI | 30-day L4 closed-loop evidence on ≥3 golden scenarios (real deploy) | P1 | **Planned** - requires real deploy evidence; no production 30-day loop yet |
| AUDIT-IDEAL-AHI.2 | §25 AHI | Bounded policy learning without governance drift | P2 | **Partial** - governance contracts done (`adaptive_governance.py`); runtime `AdaptationExecutor` loop pending |
| AUDIT-IDEAL-AHI.3 | §25 AHI | Capability marketplace readiness (trust, certification, billing) | P3 | **Planned** - marketplace readiness not productized |

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

---

## Phase AHI-ADAS - Agent Design Search (Proposed)

**Architecture:** [`architecture/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](../../architecture/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md)
**Implementation plan:** [`ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md)  
**Detailed plan satellite:** [`plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](plan/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md)  
**ADR:** [ADR-ADAPT-002](../../technical/adr/entries/2026-06-22/ADR-ADAPT-002.md)
**Hub canon:** [ADAS sub-capability](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md#adas--agent-design-search-sub-capability)

ADAS extends AHI with a governed agent-candidate design loop (scaffold → static gate → evaluation → archive → shadow/canary → promotion → verify). Full task breakdown lives in the implementation plan and detailed plan satellite.

| Phase | Purpose | Status |
|-------|---------|--------|
| **AHI-ADAS-00** | Documentation canon + ADR + top-level and satellite implementation plans | **Done** (2026-06-22) |
| **AHI-ADAS-10** | Core contracts + candidate archive | Planned |
| **AHI-ADAS-20** | Scaffold bridge + static gate | Planned |
| **AHI-ADAS-30** | Candidate evaluation + utility scoring | Planned |
| **AHI-ADAS-40** | Search controller + strategies | Planned |
| **AHI-ADAS-50** | Hooks and lifecycle events | Planned |
| **AHI-ADAS-60** | Optional Tier-2 MAS agents | Planned |
| **AHI-ADAS-70** | Shadow / canary / promotion bridge | Planned |
| **AHI-ADAS-80** | Optional Tier-3 ADAS Lab application | Planned |
| **AHI-ADAS-90** | Enterprise hardening (retention, tenant isolation, evidence bundles) | Planned |

**Delivery rule:** One **AHI-ADAS-*** phase (or sub-phase row) per PR → update this table → link evidence bundle when eval gates apply.

---

## Protocol v2 remediation - Adaptive Harness Intelligence audit (2026-08-18)

**Source:** Protocol v2 audit [`ADAPTIVE_HARNESS_INTELLIGENCE`](../../audit_results/2026-08-18/ADAPTIVE_HARNESS_INTELLIGENCE.md) - **FAIL**, 6 ACCEPTED findings (2026-08-20). Historical W-ADAPT / AUDIT-IDEAL **Done** rows above are **not** reopened.

<a id="ahi-promotion-authority-integrity-2026-08-18"></a>

### AHI-PROMOTION-AUTHORITY-INTEGRITY - scope-bound authoritative promotion decision

**Priority:** P0
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-01`](../../audit_results/2026-08-18/ADAPTIVE_HARNESS_INTELLIGENCE.md), [`AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-02`](../../audit_results/2026-08-18/ADAPTIVE_HARNESS_INTELLIGENCE.md), [`AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-03`](../../audit_results/2026-08-18/ADAPTIVE_HARNESS_INTELLIGENCE.md)

One scope-bound authoritative promotion decision binds proposal, gates, approval, tenant, and exact profile version. `apply()` enforces `passed_all_gates` and version lineage from the governing package. Every lifecycle mutation verifies `tenant_id` + `task_class` + `artifact_type` + `version_id` scope. Human-required promotion fails closed when approval evidence authority is unavailable. Cross-link [`GOVERNED_EXECUTION`](../../architecture/GOVERNED_EXECUTION.md) and [`IDENTITY_TRUST`](../../architecture/IDENTITY_TRUST.md) - do not duplicate approval/identity infrastructure.

<a id="ahi-evidence-qualification-integrity-2026-08-18"></a>

### AHI-EVIDENCE-QUALIFICATION-INTEGRITY - action/stage-aware gate completeness

**Priority:** P0/P1
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-04`](../../audit_results/2026-08-18/ADAPTIVE_HARNESS_INTELLIGENCE.md)

Gate completeness requirements are action/stage aware. Distinguish optional recommendation evidence from mandatory production promotion evidence. Missing mandatory production evidence never silently passes - `passed_all_gates` means all gates required for the intended action were evaluated and passed.

<a id="ahi-activation-consistency-integrity-2026-08-18"></a>

### AHI-ACTIVATION-CONSISTENCY-INTEGRITY - recoverable transactional activation and CAS fencing

**Priority:** P0/P1
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-05`](../../audit_results/2026-08-18/ADAPTIVE_HARNESS_INTELLIGENCE.md), [`AUDIT-20260818-ADAPTIVE_HARNESS_INTELLIGENCE-06`](../../audit_results/2026-08-18/ADAPTIVE_HARNESS_INTELLIGENCE.md)

Active configuration promotion/rollback has recoverable transactional semantics (`operation_id`, expected/new version, idempotency, reconciliation). Active pointer swap uses expected-version CAS/fencing. Reuse existing platform CAS/revision mechanisms where appropriate.

---
