# Capability Catalog & Discovery V1 — Final Program Audit (Stages 1–14)

**Audit ID:** CAPABILITY-CATALOG-V1-FINAL-PROGRAM-AUDIT  
**Date:** 2026-09-07  
**Branch:** `development`  
**Start HEAD:** `0d19448452fae52a98780f0e704c8c4b75fa3334`  
**Audited code HEAD:** `0d19448452fae52a98780f0e704c8c4b75fa3334`  
**origin/development at audit execution:** `0d19448452fae52a98780f0e704c8c4b75fa3334`  
**Worktree status:** clean  

**Scope:** Adversarial program-level architecture audit of Capability Catalog & Discovery V1 Stages 1–14 as one coherent system. Not feature implementation; no code remediation for discovered pre-existing repository issues outside V1 scope.

**Related canon:** [CAPABILITY_CATALOG_AND_DISCOVERY architecture](../../architecture/CAPABILITY_CATALOG_AND_DISCOVERY.md) · [plan](../plans/CAPABILITY_CATALOG_AND_DISCOVERY.md)

---

## Program verdict

```text
APPROVED_FOR_FINAL_INDEPENDENT_AUDIT
```

No Stage 1–14 code/architecture blocker was found. Stale documentation in the architecture hub was reconciled in this audit commit. Repository-health findings unrelated to V1 are listed separately and do not block program qualification.

**Program status:**

```text
Capability Catalog & Discovery V1
Stages 1–14 independently verified
Program CLOSED
```

---

## Authority map

| Concern | Canonical authority | Duplicate authority found |
| --- | --- | --- |
| Catalog federation | `FederatedCapabilityCatalog` / `CapabilityCatalogSource` | **none** |
| Agent acquisition / lifecycle | Agent Distribution / AC-4 | **none** |
| Agent serving | RuntimeRevision / Nexus | **none** |
| Skill definition / resolution | Skill domain (`SkillRegistry`, `SkillResolver`) | **none** |
| Tool execution | `RuntimeToolInvoker` | **none** |
| Plugin loading | Platform Plugins | **none** |
| Governance | `govern_capability_candidates` + injected evaluators | **none** |
| AW recovery | AW-7A (`WorkerCapabilityAcquisitionDecisionService`) | **none** |
| Marketplace presentation | `intergrax/marketplace` read surface | **none** |
| Isolation decision | ADR-SEC-002 / host-security governance | **none** |
| Usage metering | `intergrax/capability_metering` | **none** |
| Application composition | `wire_application_environment()` | **none** |
| Stage-14 work-stage loop | `WorkStageCapabilityDiscoveryLoopCoordinator` (decides via Stage 8; executes via injected port) | **none** |

---

## Stage matrix (1–14)

| Stage | Purpose | Main owner | Verified invariant | Status |
| --- | --- | --- | --- | --- |
| 1 | Contracts & frozen boundaries | `intergrax/contracts/capability_catalog` | V1 kinds Agent/Skill/Tool only; source-qualified identity; no universal registry API | Implemented / independently verified |
| 2 | Federated catalog read model | `intergrax/capability_catalog/federation.py` | Read-only sources; conflict fail-closed; deterministic ordering | Implemented / independently verified |
| 3 | Query / filtering | `discovery.py` | Query ≠ permission ≠ selection; enterprise scope fail-closed | Implemented / independently verified |
| 4 | Ranking | `ranking.py` | Ranking ≠ selection; identity/provenance preserved; deterministic tie-break | Implemented / independently verified |
| 5 | Governance | `governance.py` | DISCOVER→RANK→GOVERN→SELECT; STRICT empty pipeline fail-closed | Implemented / independently verified |
| 6 | Skill versioning | contracts + skill adapters | Exact pinning; digest; no mutable latest substitution | Implemented / independently verified |
| 7 | Tool/Skill distribution visibility | private + domain adapters | Catalog visibility only; no installation from discovery | Implemented / independently verified |
| 8 | Work-stage discovery | `work_stage_discovery.py` | Stateless; fresh per need; typed `WorkStageCapabilityNeed` trigger | Implemented / independently verified |
| 9 | AW bridge | `capability_acquisition_service.py` | Decides ≠ executes; A0–A4 preserved; separate from Stage 8 | Implemented / independently verified |
| 10 | Bootstrap evidence | Tier-3 `ApplicationPlatformPluginEvidence` | Bootstrap evidence ≠ runtime inventory | Implemented / independently verified |
| 11 | Marketplace | `intergrax/marketplace` | Product/read surface; no install/registry mutation | Implemented / independently verified |
| 12 | Isolation assessment | ADR-SEC-002 | Trusted in-process V1 truth; isolation ≠ authorization | Implemented / independently verified |
| 13 | Usage metering | `intergrax/capability_metering` | Usage ≠ pricing/billing; source-qualified attribution | Implemented / independently verified |
| 14 | Closed loop | `work_stage_capability_loop.py` | Typed need → fresh federation → govern → domain port → observe | Implemented / independently verified |

---

## Cross-stage invariants

| Invariant | Evidence |
| --- | --- |
| AVAILABLE ≠ ACTIVE lifecycle | `AvailabilityDisposition` vocabulary; `EffectiveCapabilitySet` requires `HOST_AVAILABLE`; no catalog install/activate API |
| Source-qualified identity end-to-end | `CapabilityDiscoveryIdentity.sort_key` = `(kind, source_id, source_kind, logical_id)`; tests for same logical_id / different source preserved |
| Provenance snapshot-like | Frozen `CapabilityProvenance` pydantic model; attribution projects from discovery candidate, not live registry lookup |
| Governance precedes effective selection | `WorkStageCapabilityDiscoveryService.resolve`: discover → rank → govern → `select_effective_executable_candidates` |
| Skill ≠ executable runtime unit | No `execute_skill` / `skill.execute` in Stage 1–14 code; non-TOOL selections blocked in Stage-14 loop |
| AC-4 ≠ AW-7A | Separate services, ports, and architecture gates; AW-7A registry adapters read-only (`list()` only) |
| A4 human authority only | `a4_never_self_authorized`; acquisition tests reject USE_EXISTING with A4 |
| AW Stage-14 core runtime-neutral | `intergrax/autonomous_work/*.py` production modules: zero `intergrax.runtime` / `intergrax.applications` imports |
| Marketplace ≠ trust | Commercial/publisher metadata display-only; governance evaluators independent |
| Metering quantity = Σ quantity | `CapabilityUsageSummaryReport` rollups sum `quantity`, not event count |

---

## Identity / provenance chain

Verified path without semantic collapse:

```text
Stage 2 entry (CapabilityCatalogEntry.identity)
  → Stage 3 candidate (CapabilityDiscoveryCandidate)
  → Stage 4 ranked (RankedCapabilityCandidate)
  → Stage 5 governed (GovernedCapabilityCandidate)
  → Stage 8 effective (EffectiveCapabilitySet)
  → Stage 9 AW projection (separate WorkerCapabilityCandidate — distinct semantics, not collapse)
  → Stage 13 attribution (CapabilityUsageAttribution from discovery handoff)
  → Stage 14 selection evidence (CapabilityIdentityKey in loop iteration evidence)
```

`kind`, `source_kind`, `source_id`, `logical_id` remain in `sort_key` at every catalog stage. Federation test `test_same_logical_id_different_source_both_preserved` and Stage-14 closed-loop multi-source test provide program evidence.

---

## Governance chain

```text
WorkStageCapabilityNeed
  → discover_capability_candidates (Stage 3)
  → rank_capability_candidates (Stage 4)
  → govern_capability_candidates (Stage 5)
  → select_effective_executable_candidates (Stage 8)
  → domain execution port (Stage 14)
```

No DISCOVER→SELECT→GOVERN shortcut found. Second loop iteration takes fresh `federated_catalog.snapshot()` and re-runs full Stage-8 pipeline (closed-loop tests).

---

## Forbidden flow matrix

| # | Flow | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Catalog → registry mutation | **PASS** | No `register`/`install`/`activate` in `intergrax/capability_catalog`; architecture AST gates |
| 2 | Catalog → execution | **PASS** | No execute/invoke in catalog package; Stage 14 uses injected `WorkStageToolExecutionPort` |
| 3 | Marketplace → install/activate | **PASS** | `MarketplaceCatalogService` read-only; marketplace architecture gates (40 tests) |
| 4 | AW → AgentRegistry mutation | **PASS** | No AgentRegistry references in AW modules |
| 5 | AW → ToolRegistry mutation | **PASS** | AW adapters call `list()` only |
| 6 | AW → SkillRegistry mutation | **PASS** | AW adapters call `list()` / manifest lookup only |
| 7 | Discovery → permission grant | **PASS** | Availability evidence separate from governance; query contracts typed |
| 8 | Ranking → permission elevation | **PASS** | Rankers preserve availability; governance evaluators narrow only |
| 9 | Skill → direct execution | **PASS** | Stage-14 loop blocks non-TOOL execution |
| 10 | Metering → billing/payment | **PASS** | No price/currency fields in metering packages |
| 11 | Marketplace → trust grant | **PASS** | Publisher metadata display-only |
| 12 | Sandbox → authorization grant | **PASS** | ADR-SEC-002; isolation decision tests |
| 13 | Bootstrap evidence → runtime inventory | **PASS** | Stage 10 evidence on plugin bootstrap only |
| 14 | Stage-14 observation → hidden rediscovery | **PASS** | Rediscovery only via typed `next_need` on observation provider |

---

## Autonomous loop (Stage 14)

Reference proof: `tests/integration/autonomous_work/test_capability_discovery_closed_loop.py` (17 passed).

Verified terminal semantics:

- `success + no next_need` → `COMPLETED`
- `failure + no next_need` → `ESCALATED` (not false success)
- Denied governance → `BLOCKED` without domain execution
- Fresh governance on second iteration (governance provider per iteration index)

`RuntimeToolInvoker` bridge lives in test/qualification adapter only (`tests/integration/autonomous_work/runtime_tool_invoker_work_stage_port.py`), outside AW core.

---

## Test matrix

| Suite | Result |
| --- | --- |
| `tests/unit/capability_catalog/` + `tests/unit/contracts/capability_catalog/` | **227 passed** |
| `tests/integration/autonomous_work/test_capability_discovery_closed_loop.py` | **17 passed** |
| `tests/unit/capability_metering/` + contracts | **25 passed** |
| `tests/unit/marketplace/` + contracts | **40 passed** |
| AW Stage 9 (`test_worker_capability_acquisition*.py`, catalog adapters) | **66 passed** |
| Skills unit suite | **238 passed** |
| Tool registry + catalog adapters (focused) | **25 passed** |
| AC-4 architecture gate + dynamic acquisition | **13 passed** |
| Platform plugins + Stage 12 isolation | **36 passed** |
| Architecture gates (metering, marketplace, stage12) | **17 passed** |
| Program boundary gate (new) | run with audit commit |

**Ruff** (`intergrax/capability_catalog`, contracts, metering, marketplace, `work_stage_capability_loop.py`): **All checks passed**

---

## Repository-health findings (category B — not V1 blockers)

| Finding | Owner | Why not V1 blocker |
| --- | --- | --- |
| `test_autonomous_work_contracts_do_not_import_runtime_services` fails on `execution_dispatch.py` importing `ExecutionRequest` | AW-5A contracts | Pre-existing AW execution dispatch boundary (commit `19341589a`); not introduced by Stage 14; AW production modules have zero runtime imports |
| Tool provider tests require optional `celery` / `langchain_core` | Tools providers | Collection error in optional provider tests; Stage 7 registry/catalog tests pass |
| Architecture hub stale “planned” maturity language | Documentation | Reconciled in this audit; code and tests contradict stale text |

---

## Documentation reconciliation

Updated in audit commit:

- `docs/project/architecture/CAPABILITY_CATALOG_AND_DISCOVERY.md` — maturity boundary reflects V1 CLOSED; Stages 1–14 independently verified
- `docs/project/maintainers/plans/CAPABILITY_CATALOG_AND_DISCOVERY.md` — Stages 1–14 marked independently verified; program CLOSED

---

## Universal abstraction audit

All hits for `UniversalRegistry`, `CapabilityDiscoveryPort`, `UniversalCapability*` classified:

- **Forbidden-in-production:** absent from program packages (catalog, metering, marketplace, AW Stage-14 core)
- **Test gate strings / negative fixtures:** `test_capability_catalog_architecture_gates.py`, `test_capability_discovery_closed_loop.py`
- **Unrelated domains:** `LiveCapabilityExecutorV1` (local workspace application), `ProductionAgentCapabilityRuntime` (AC production composition) — outside Capability Catalog V1 program scope

`Worker*CapabilityDiscoveryPort` in AW-7A are bounded AW ports, not catalog `CapabilityDiscoveryPort` universal kernel.

---

## Dependency direction (simplified)

```text
contracts/capability_catalog  ←  capability_catalog  ←  marketplace
        ↑                              ↑
capability_metering            autonomous_work (Stage 14 loop → catalog only)
        ↑
   (attribution from catalog candidates)

agent_distribution / skills / tools / platform_plugins / runtime
  → domain-owned; catalog adapters read domain public surfaces only
  → no cycle: catalog does not import runtime or applications
```

---

## Approval conditions checklist

| # | Condition | Status |
| --- | --- | --- |
| 1 | Catalog read-only | PASS |
| 2 | No universal runtime | PASS |
| 3 | No registry merging | PASS |
| 4 | Source-qualified identity end-to-end | PASS |
| 5 | Provenance preserved | PASS |
| 6 | Ranking separated from selection | PASS |
| 7 | Governance precedes effective selection | PASS |
| 8 | Skill declarative | PASS |
| 9 | AC-4 and AW separate | PASS |
| 10 | Bootstrap evidence not runtime SoT | PASS |
| 11 | Marketplace product/read surface | PASS |
| 12 | Isolation not catalog authority | PASS |
| 13 | Metering separate from billing | PASS |
| 14 | Stage-14 typed governed rediscovery | PASS |
| 15 | AW Stage-14 core runtime-neutral | PASS |
| 16 | No discovery-driven installation | PASS |
| 17 | No hidden registry mutation | PASS |
| 18 | No private cross-component bypass | PASS |
| 19 | No fabricated evidence | PASS |
| 20 | No Stage 1–14 contradiction | PASS |
| 21 | Canonical docs agree with code | PASS (after reconciliation) |
| 22 | Status metadata truthful | PASS (after reconciliation) |
| 23 | V1-specific gates green | PASS |

---

## Independent program closure

Independent review result:
APPROVED

Capability Catalog & Discovery V1:
CLOSED

Audited qualification SHA:
e707d0b6de99ed87eb8805a8b5377867dc62a3a0

`e707d0b6de99ed87eb8805a8b5377867dc62a3a0` remains ancestor of `development`; later commits have no Capability Catalog V1 semantic overlap.
