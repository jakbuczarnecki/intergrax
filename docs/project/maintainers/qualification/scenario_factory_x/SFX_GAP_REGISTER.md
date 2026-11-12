# SCENARIO-FACTORY-X — gap and conflicts register

**Source:** master plan R1 §6 (gaps), §3.4, §15.1 and §16 (conflicts) — [`SFX-MASTER-PLAN.md`](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md).  
**Index:** [`SFX_INDEX.md`](./SFX_INDEX.md) · **Evidence pin:** `eadb1c2116feb0a2090b234a36fd4182bd6ace8d`  
**Consolidation:** final dispositions are reconciled with the W1 research findings in `docs/project/maintainers/qualification/scenario_factory_x/SFX_GAP_ANALYSIS.md` (T-IGX-037-C). Until then this register is authoritative.

Status vocabulary: `VERIFIED` (gap confirmed at the pin) · `MISSING_EVIDENCE` (not yet established; scheduled task). R1 changes: ▲ revised, ▼ new in R1.

## Gaps SFX-GAP-01..17

| Gap | Description | Status | Disposition / task | Owner |
|---|---|---|---|---|
| SFX-GAP-01 | No web toolchain, browser auth or SPA serving | VERIFIED | Master plan §7.8; tasks SFX-P10-04..07; OD-SFX-05 | SFX |
| SFX-GAP-02 | No scenario catalog | VERIFIED | Tier-2 sidecar `sfx.scenario_catalog_entry.v1` (§7.6); no manifest change; SFX-P11-03 | SFX |
| SFX-GAP-03 | Discovery agent absent | VERIFIED | SFX-P6-01 | SFX |
| SFX-GAP-04 | ROI component absent | VERIFIED | SFX-P11-02 | SFX |
| SFX-GAP-05 | SEF contracts PROPOSED_ONLY | VERIFIED | GATE-SEF-CONSUME (§3.3) | SEF-X track |
| SFX-GAP-06 | `sync_agent_env.py` generates no agents | VERIFIED | SFX-P4-B01 (post-freeze by default; §15.1) | SFX (W3) |
| SFX-GAP-07 | MCP client compatibility | MISSING_EVIDENCE | SFX-P1-02, SFX-P10-02 | SFX |
| SFX-GAP-08 ▲ | `ApprovalStorePort` is in-memory with no tenant | VERIFIED | Avoided by using decision bindings (SFX-CAP-08) | SFX (avoidance) |
| SFX-GAP-09 | Dual `ToolRegistry` | MISSING_EVIDENCE | SFX-P2-01 ownership review | Platform (review in SFX-P2-01) |
| SFX-GAP-10 | Business-scenario workspace location | VERIFIED | ADR-SFX-002 | SFX |
| SFX-GAP-11 | Research facts | MISSING_EVIDENCE | SFX-P1 (T-IGX-037-B) | SFX |
| SFX-GAP-12 | No SFX qualification suite | VERIFIED | SFX-P9, SFX-P13 | SFX |
| SFX-GAP-13 ▼ | No PR-level SAST or dependency audit; ruff has no `S` rules; no pip-audit or npm audit | VERIFIED | SFX-P9-03; OD-SFX-14 | SFX + operator |
| SFX-GAP-14 ▼ | No conformance rule forbids generated code from mutating environment profile, policy mode, governance or certification records, contract lifecycle, or exposure policy | VERIFIED | SFX-P9-02 | SFX |
| SFX-GAP-15 ▼ | Platform self-promotion seams: `AgentContract.lifecycle_state` defaults to PRODUCTION (`agent_contract_meta.py:71`); environment profile not frozen (`root.py:85`); `PRODUCTION_ATTACHED` does not require STRICT+PRODUCT; `evaluate_agent_promotion` not wired; `apply_roster_agent_governance` self-certifies | VERIFIED | **Platform-owned** (Agent Runtime Governance / PDC-X D1/D6), not fixed by SFX. SFX compensates inside its own scope (§13), records the seams and hands them to the platform owner. They block **production admission**, not SFX closure. | Platform |
| SFX-GAP-16 ▼ | No owner for per-scenario production admission | VERIFIED | OD-SFX-13 (§14.2) | Operator |
| SFX-GAP-17 ▼ | MCP composition widens `allowed_binding_ids` (`mcp_application_server.py:83-91`) | VERIFIED | SFX MCP uses an explicit allowlist with a no-widening test (SFX-P10-02) | SFX (compensation); seam platform-owned |

R1 lists GAP-03 and GAP-04 in one row ("Discovery agent and ROI component absent"); they are split here, one row per gap.

## Conflicts register

Reported conflicts; none is resolved unilaterally by SFX.

| ID | Conflict | Source | Status | Disposition / owner |
|---|---|---|---|---|
| C-SEF-01 | SEF names no external intake consumer, and its CON-N-01 is "human submit only". SFX conforms through HO-2, but SEF must acknowledge SFX as a consumer. | §3.4 | OPEN | SEF track, under OD-SEF-07 or a new SEF OD; part of GATE-SEF-CONSUME and OD-SFX-10 |
| C-SEF-02 | SEF T-P5-004 (network egress) is OPTIONAL and draft; SFX closure requires network restriction on agentic engineering. | §3.4, §13 | OPEN | SFX asks SEF to make T-P5-004 mandatory for GATE-SEF-CONSUME; otherwise GATE-SFX-FULL stays BLOCKED |
| C-SEF-03 | CON-E-04 `VerificationRecord` has no SEF WBS task; SFX consumes it. | §3.4 | OPEN (BLOCKED) | SEF track to assign a task |
| C-SEF-04 | GAP-HIGH-07 (SEF ↔ READINESS kit coupling) is OPEN. | §3.4 | OPEN | SFX scaffold templates stay on the existing scaffold tools until the READINESS kit is CLOSED; SFX never defines kit templates |
| C-OD-01 | OD-SFX-04 says "agent-env only", but the SSOT mechanism requires committing the regenerated `.claude/` and `.cursor/` adapters in the same change. | §15.1 | **RESOLVED** by [OD-SFX-16](./operator_decisions/OD-SFX-16.md) | Adapters ship with their agent-env sources in SFX-P4-A01 (W2); `sync_agent_env.py` and `role_registry.yaml` stay untouched |
| C-RDY-01 | SFX must not define Delivery Kit templates; REF-2 must not duplicate the READINESS settlement reference. | §16 | OPEN (standing constraint) | READINESS-X owns the kit; enforced in P8-01 and P12-02 scope |
| C-AF-01 | `scripts/` changes interact with the ARCH-FREEZE inventory and SHA lock. | §15.1 | OPEN (standing constraint) | P4-B01 moved to W3; optional OD-SFX-12 only before `T-B7-LOCK-FREEZE-SHA` |
| C-IIAC-01 | Every post-freeze product wave needs an ACT-N11 audited re-bind. | §16 | OPEN (standing constraint) | REBIND-Wn at the end of W3..W8; OD-SFX-09 |
| C-ECAP-01 | Applicability of ECAP §16 ("update ECAP when a new domain is added") to a Tier-2 product module. | §16 | OPEN | Decided in the ADR-SFX-001 review (T-IGX-037-C) |
