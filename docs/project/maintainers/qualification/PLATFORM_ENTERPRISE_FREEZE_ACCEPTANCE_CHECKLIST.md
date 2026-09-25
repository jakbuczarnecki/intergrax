# Platform Enterprise Freeze Acceptance Checklist

> Ten dokument jest globalnym acceptance-control artefaktem dla architecture freeze. Nie zastępuje domain semantic authorities ani PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md.

**Branch:** development

## Document roles

| Artifact | Role |
|---|---|
| [Platform Enterprise Completion Roadmap](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) | What / when / current program status (closure SSOT) |
| This checklist | What must eventually be proven for freeze (FRZ-*) |
| Roadmap evidence ledger (roadmap §5) | What exact evidence closed a stage |
| Domain architecture / qualification records | What the mechanism semantically means |

**FRZ criterion references semantic authorities. It does not redefine them.** If a freeze criterion conflicts with domain semantic authority: **STOP** and reconcile before proceeding.

Companion program tracker: [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md).

---

## Allowed statuses

Only:

- OPEN
- PASS
- BLOCKED
- N/A — WITH EVIDENCE

Forbidden as final acceptance verdict: PARTIAL, MOSTLY, PROBABLY, GOOD ENOUGH, DEFERRED, LATER.

## PASS evidence rule

FRZ-* = PASS only when **all** exist:

1. exact GitHub SHA / HEAD;
2. code and/or contract evidence at that SHA;
3. qualification / gate / test evidence;
4. independent audit evidence.

A Cursor implementation report alone is **not** sufficient evidence.

## Initial checklist policy

Criteria default to OPEN. Historical closure may be noted under **Evidence / historical evidence** without upgrading to PASS until independently recertified under the current program. Do not expand PASS coverage without exact-SHA independent evidence.

---

## Freeze acceptance criteria

| ID | Domain | Freeze acceptance criterion | Primary closing stage(s) | Status | Evidence / historical evidence |
| --- | --- | --- | --- | --- | --- |
| FRZ-BND-01 | Layer boundaries | canonical layer model documented | EBH-2*, EBH-3, EBH-4 | OPEN | — |
| FRZ-BND-02 | Layer boundaries | allowed dependency directions explicit | EBH-2*, EBH-3, EBH-4 | OPEN | — |
| FRZ-BND-03 | Layer boundaries | reverse dependencies = 0 | EBH-2*, EBH-3, EBH-4 | OPEN | — |
| FRZ-BND-04 | Layer boundaries | private implementation leakage = 0 | EBH-2*, EBH-3, EBH-4 | OPEN | **EBH-2G-R1** (`94eb6ae…`): package root / leaf contract import boundary regression gate on RAG subsystem (R1 scope; not global layer-boundary PASS). **EBH-2G-R2-R1** (`be7e817…`): RAG graph composition no longer imports concrete Integration graph-provider implementations; RAG operates against `intergrax.integrations.contracts.graph_store.GraphStore` (R2-R1 scope; not global PASS). |
| FRZ-BND-05 | Layer boundaries | cross-layer construction only sanctioned composition | EBH-2*, EBH-3, EBH-4 | OPEN | **EBH-2G-R1** (`94eb6ae…`): canonical RAG retrieval composition path qualified with mechanical gates (R1 scope). **EBH-2G-R2-R1** (`be7e817…`): sanctioned Graph Store composition path — `IntegrationProfile` → `resolve_from_profile(GRAPH_STORE)` → Integration `GraphStore` → typed RAG adapter → RAG `GraphStore` (R2-R1 scope; not global PASS). |
| FRZ-BND-06 | Layer boundaries | transport adapters do not own runtime semantics | EBH-2*, EBH-3, EBH-4 | OPEN | — |
| FRZ-OWN-01 | Ownership | every major concern has exactly one semantic owner | EBH-2*, EBH-3 | OPEN | **EBH-2G-R2-R1** (`be7e817…`): Integrations owns graph provider selection/materialization; RAG owns GraphRAG semantics/adaptation (R2-R1 scope; not global PASS). |
| FRZ-OWN-02 | Ownership | one canonical contract per responsibility | EBH-2*, EBH-3 | OPEN | **EBH-2G-R2-R1** (`be7e817…`): Integration `GraphStore` and RAG `GraphStore` have distinct responsibilities; typed adaptation separates their semantic ownership (R2-R1 scope; not duplicate-contract claim). |
| FRZ-OWN-03 | Ownership | one sanctioned composition owner | EBH-2*, EBH-3 | OPEN | **EBH-2G-R2-R1** (`be7e817…`): IntegrationProfile / Integration resolver is sanctioned Graph Store composition owner; RAG adapts via typed seam only (R2-R1 scope). |
| FRZ-OWN-04 | Ownership | no shadow authority | EBH-2*, EBH-3 | OPEN | — |
| FRZ-OWN-05 | Ownership | no duplicated semantic mechanism | EBH-2*, EBH-3 | OPEN | **EBH-2G-R2-R1** (`be7e817…`): RAG-local backend registry removed; no duplicate provider-selection mechanism in audited Graph Store flow (R2-R1 scope). |
| FRZ-CTR-01 | Contracts / abstraction | consumers depend on contracts | EBH-2*, EBH-3, EBH-4 | OPEN | EBH-2F independently **CLOSED** (`765fabe…`); R2 (`f7fabab…`); **EBH-2G-R1** independently **CLOSED** at `94eb6ae87532057f96cb1512a1487b1f6194ae28` — canonical retrieval contracts, no concrete integration-provider/vendor coupling in `RetrievalService`, typed manager/composition boundaries. **EBH-2G-R2-R1** (`be7e817…`): RAG graph composition consumes Integration `GraphStore` contract, not concrete provider (R2-R1 scope). **EBH-2G** parent **BLOCKED** by **EBH-2G-R2** (final R2 recertification CURRENT after R2-R2 closure). |
| FRZ-CTR-02 | Contracts / abstraction | concrete provider leakage = 0 | EBH-2*, EBH-3, EBH-4 | OPEN | **EBH-2G-R1** (`94eb6ae…`): `RetrievalService` has no direct integration-provider/vendor coupling on audited composition boundary (R1 scope). **EBH-2G-R2-R1** (`be7e817…`): zero concrete Integration provider leakage from audited RAG graph composition (R2-R1 scope). |
| FRZ-CTR-03 | Contracts / abstraction | no pseudo-contract dicts | EBH-2*, EBH-3, EBH-4 | OPEN | Historical `EBH-2F-R2-BLOCKER` superseded; `IntegrationContractFactory = IntegrationFactory`; explicit `contract_specs` / `integration_contract_specs()` only (gate). |
| FRZ-CTR-04 | Contracts / abstraction | public/internal contracts classified | EBH-2*, EBH-3, EBH-4 | OPEN | R2-R2 (`311064f…`); R2 closed (`f7fabab…`); parent recert `765fabe…` revalidates on current HEAD. **EBH-2G-R2-R1** (`be7e817…`): Infrastructure provider contract and GraphRAG domain contract remain distinct (R2-R1 scoped evidence). |
| FRZ-CTR-05 | Contracts / abstraction | contract responsibility narrow/cohesive | EBH-2*, EBH-3, EBH-4 | OPEN | `IntegrationPlugin` / `IntegrationFactory` → `catalog_factory.py`; `resolve_typed` delegates to `resolve_from_profile` (no parallel resolver owner). **EBH-2G-R2-R1** (`be7e817…`): Graph Store adapter responsibility is narrow and explicit (R2-R1 scope). |
| FRZ-CTR-06 | Contracts / abstraction | contracts do not encode concrete vendor/runtime implementation | EBH-2*, EBH-3, EBH-4 | OPEN | **EBH-2G-R2-R1** (`be7e817…`): RAG semantic contract no longer encodes vendor/provider choice through `graph_store_backend` (R2-R1 scope). |
| FRZ-TYP-01 | Strong typing | semantic boundaries strongly typed | EBH-2* | OPEN | R2 closed (`f7fabab…`); parent recert `765fabe…` confirms `integrations/registry/*` semantic surfaces without `-> Any`. **EBH-2G-R1** CLOSED (`94eb6ae…`): typed retrieval composition, typed `MetadataFilter`, typed query-embedding boundary; targeted pyright = 0 relevant errors on audited surfaces. **EBH-2G-R2-R1** (`be7e817…`): typed Integration→RAG graph seam accepted at closure SHA (R2-R1 scope). |
| FRZ-TYP-02 | Strong typing | Any at semantic boundaries = 0 unless evidence-backed transport reason | EBH-2* | OPEN | R2 chain closed at child level; provider `_open_*` transport `Any` outside R2 registration/resolution semantic owner. **EBH-2G-R1** (`94eb6ae…`): zero semantic `Any` on audited retrieval composition boundary; no `Any` workaround introduced for R1 remediation. **EBH-2G-R2-R1** (`be7e817…`): semantic `Any` removed from `CypherRagGraphStore.integration_store` (R2-R1 scope; transport/property metadata may remain dynamic elsewhere). |
| FRZ-TYP-03 | Strong typing | generic object semantic seams = 0 | EBH-2* | OPEN | R2-R2: no-expected `resolve_contract` → `CategoryIntegrationInstance` (AST gate + pyright). **EBH-2G-R1** (`94eb6ae…`): retrieval query/candidate semantic boundaries use canonical typed contracts, not generic `object` as substitute ABI (R1 scope only). |
| FRZ-TYP-04 | Strong typing | reflection-based semantic dispatch = 0 | EBH-2* | OPEN | Historical `EBH-2F-R2-BLOCKER` superseded; `plugin_register` has no `getattr`/`CONTRACT_SPECS` (architecture gate). **EBH-2G-R1** (`94eb6ae…`): no reflection/dynamic capability probing on audited retrieval boundary; typed properties replace `getattr`/`hasattr` capability discovery (R1 scope). **EBH-2G-R2-R1** (`be7e817…`): provider identity/materialization no longer uses dynamic probing in audited graph composition path (R2-R1 scope). |
| FRZ-TYP-05 | Strong typing | string-dispatch substitute for typed contract = 0 | EBH-2* | OPEN | — |
| FRZ-TYP-06 | Strong typing | type-ignore/cast cannot mask architecture mismatch | EBH-2* | OPEN | R2-R2 accepted (`311064f…`): `resolve_typed.py` AST gate — no `cast(` / `# type: ignore` on resolution helper. **EBH-2G-R1** (`94eb6ae…`): no `cast`, no `type: ignore`, no `.tolist()`/conversion workaround to hide query-embedding mismatch on audited retrieval surfaces. **EBH-2G-R2-R1** (`be7e817…`): R2-R1 introduced no cast/type-ignore workaround to hide Graph Store architecture mismatch (R2-R1 scope). |
| FRZ-PLG-01 | Pluginability | extensible mechanisms expose platform contracts | EBH-2*, EBH-5 | OPEN | Parent recert: `register_integration_plugin` → `integration_contract_specs()` / explicit `contract_specs` (`test_p2_003_explicit_contract_specs.py`). **EBH-2G-R2-R1** (`be7e817…`): graph providers consumed through canonical Integration contract (R2-R1 scope). |
| FRZ-PLG-02 | Pluginability | external structural implementation works without core patch | EBH-2*, EBH-5 | OPEN | Parent recert `765fabe…`: `test_external_plugin.py`, R1 replaceability gate, registry projection. |
| FRZ-PLG-03 | Pluginability | provider discovery has explicit owner | EBH-2*, EBH-5 | OPEN | **EBH-2G-R2-R1** (`be7e817…`): provider discovery owner = Integration catalog (R2-R1 scope). |
| FRZ-PLG-04 | Pluginability | provider selection has explicit owner | EBH-2*, EBH-5 | OPEN | **EBH-2G-R2-R1** (`be7e817…`): provider selection owner = IntegrationProfile / Integration resolver (R2-R1 scope). |
| FRZ-PLG-05 | Pluginability | activation/admission explicit | EBH-2*, EBH-5 | OPEN | **EBH-2G-R2-R1** (`be7e817…`): provider activation/materialization through canonical Integration resolution, not RAG-local vendor registry (R2-R1 scope). |
| FRZ-PLG-06 | Pluginability | plugin package != capability authority | EBH-2*, EBH-5 | OPEN | — |
| FRZ-PLG-07 | Pluginability | runtime extensions cannot self-expand authority | EBH-2*, EBH-5 | OPEN | — |
| FRZ-PLG-08 | Pluginability | dynamic registration reversible/scoped when supported | EBH-2*, EBH-5 | OPEN | — |
| FRZ-RPL-01 | Replaceability | representative default provider replaceable | EBH-5 | OPEN | — |
| FRZ-RPL-02 | Replaceability | custom implementation proof exists | EBH-5 | OPEN | — |
| FRZ-RPL-03 | Replaceability | replaceability not monkeypatch-only | EBH-5 | OPEN | — |
| FRZ-RPL-04 | Replaceability | vendor/backend switch does not modify consumers | EBH-5 | OPEN | — |
| FRZ-EXE-01 | Execution | exactly one legal platform execution authority | HARNESS-*, EBH-4, GOV-X* | OPEN | — |
| FRZ-EXE-02 | Execution | all meaningful executable work enters canonical boundary | HARNESS-*, EBH-4, GOV-X* | OPEN | — |
| FRZ-EXE-03 | Execution | Nexus remains private implementation | HARNESS-*, EBH-4, GOV-X* | OPEN | — |
| FRZ-EXE-04 | Execution | Agent != Execution | HARNESS-*, EBH-4, GOV-X* | OPEN | — |
| FRZ-EXE-05 | Execution | child work admitted correctly | HARNESS-*, EBH-4, GOV-X* | OPEN | — |
| FRZ-EXE-06 | Execution | execution identity canonical | HARNESS-*, EBH-4, GOV-X* | OPEN | — |
| FRZ-EXE-07 | Execution | no alternate scheduler/worker execution authority | HARNESS-*, EBH-4, GOV-X* | OPEN | — |
| FRZ-GOV-01 | Governance | Governance != Execution | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-02 | Governance | proposal != permission != execution | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-03 | Governance | child_authority subset parent_authority | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-04 | Governance | downstream scope narrowing only | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-05 | Governance | fresh authorization before meaningful effects | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-06 | Governance | missing HITL never approval | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-07 | Governance | required evidence before work | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-08 | Governance | policy/profile revision attributable | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-09 | Governance | runtime extensions cannot expand authority | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-GOV-10 | Governance | control-plane mutation governed | HARNESS-*, GOV-X* | OPEN | — |
| FRZ-OBS-01 | Observability / diagnostics | canonical event/evidence spine | HARNESS-*, TRACE-X, CTRL-X | OPEN | — |
| FRZ-OBS-02 | Observability / diagnostics | Observability does not mint execution truth | HARNESS-*, TRACE-X, CTRL-X | OPEN | — |
| FRZ-OBS-03 | Observability / diagnostics | Diagnostics interprets facts only | HARNESS-*, TRACE-X, CTRL-X | OPEN | — |
| FRZ-OBS-04 | Observability / diagnostics | correlation IDs propagated | HARNESS-*, TRACE-X, CTRL-X | OPEN | — |
| FRZ-OBS-05 | Observability / diagnostics | terminal state observable | HARNESS-*, TRACE-X, CTRL-X | OPEN | — |
| FRZ-OBS-06 | Observability / diagnostics | side-effect paths observable | HARNESS-*, TRACE-X, CTRL-X | OPEN | — |
| FRZ-OBS-07 | Observability / diagnostics | diagnostic provenance reconstructable | HARNESS-*, TRACE-X, CTRL-X | OPEN | — |
| FRZ-TRC-01 | Traceability | execution causal chain reconstructable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-02 | Traceability | parent-child causality reconstructable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-03 | Traceability | tool invocation attributable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-04 | Traceability | provider invocation attributable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-05 | Traceability | model call/context decision attributable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-06 | Traceability | side effect attributable to authorization | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-07 | Traceability | policy revision attributable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-08 | Traceability | profile revision attributable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-09 | Traceability | restart/resume preserves trace continuity | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-10 | Traceability | terminal outcome linked to causal evidence | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-11 | Traceability | configured/effective provenance reconstructable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-TRC-12 | Traceability | transport identity != runtime identity and mapping attributable | TRACE-X, GOV-X* | OPEN | — |
| FRZ-STA-01 | Persistence / state | exactly one truth owner per state family | STATE-X | OPEN | — |
| FRZ-STA-02 | Persistence / state | no duplicate semantic state stores | STATE-X | OPEN | — |
| FRZ-STA-03 | Persistence / state | atomic/transaction boundaries explicit | STATE-X | OPEN | — |
| FRZ-STA-04 | Persistence / state | tenant isolation | STATE-X | OPEN | — |
| FRZ-STA-05 | Persistence / state | stale state rejected | STATE-X | OPEN | — |
| FRZ-STA-06 | Persistence / state | configured/effective/persisted state distinction | STATE-X | OPEN | — |
| FRZ-STA-07 | Persistence / state | policy artifacts durable where required | STATE-X | OPEN | — |
| FRZ-STA-08 | Persistence / state | checkpoints are state not identity authority | STATE-X | OPEN | — |
| FRZ-REC-01 | Recovery / durability | crash recovery deterministic | STATE-X | OPEN | — |
| FRZ-REC-02 | Recovery / durability | restart recovery preserves authority | STATE-X | OPEN | — |
| FRZ-REC-03 | Recovery / durability | resume does not mint new identity incorrectly | STATE-X | OPEN | — |
| FRZ-REC-04 | Recovery / durability | replay cannot create divergent truth | STATE-X | OPEN | — |
| FRZ-REC-05 | Recovery / durability | fork semantics explicit | STATE-X | OPEN | — |
| FRZ-REC-06 | Recovery / durability | partial persistence fails closed where required | STATE-X | OPEN | — |
| FRZ-REC-07 | Recovery / durability | external operation uncertainty modeled | STATE-X | OPEN | — |
| FRZ-CMP-01 | Compatibility / evolution | frozen contracts inventory exists | COMPAT-X | OPEN | — |
| FRZ-CMP-02 | Compatibility / evolution | versioning policy exists | COMPAT-X | OPEN | — |
| FRZ-CMP-03 | Compatibility / evolution | persisted schemas versioned | COMPAT-X | OPEN | — |
| FRZ-CMP-04 | Compatibility / evolution | events evolution policy exists | COMPAT-X | OPEN | — |
| FRZ-CMP-05 | Compatibility / evolution | plugin/provider compatibility policy exists | COMPAT-X | OPEN | — |
| FRZ-CMP-06 | Compatibility / evolution | migration policy exists | COMPAT-X | OPEN | — |
| FRZ-CMP-07 | Compatibility / evolution | deprecation/removal policy exists | COMPAT-X | OPEN | — |
| FRZ-CMP-08 | Compatibility / evolution | compatibility shims not parallel authorities | COMPAT-X | OPEN | — |
| FRZ-SEC-01 | Security / isolation | tenant isolation proven | CTRL-X, PROD-Q | OPEN | — |
| FRZ-SEC-02 | Security / isolation | secrets references / late resolution | CTRL-X, PROD-Q | OPEN | — |
| FRZ-SEC-03 | Security / isolation | raw secrets do not cross unrelated layers | CTRL-X, PROD-Q | OPEN | — |
| FRZ-SEC-04 | Security / isolation | production auth bypass = 0 | CTRL-X, PROD-Q | OPEN | — |
| FRZ-SEC-05 | Security / isolation | sandbox/isolation boundaries qualified | CTRL-X, PROD-Q | OPEN | — |
| FRZ-SEC-06 | Security / isolation | provider/plugin trust qualified | CTRL-X, PROD-Q | OPEN | — |
| FRZ-SEC-07 | Security / isolation | security claim bounded by explicit threat model/evidence | CTRL-X, PROD-Q | OPEN | — |
| FRZ-REL-01 | Reliability | timeout ownership explicit | CTRL-X, PROD-Q | OPEN | — |
| FRZ-REL-02 | Reliability | retry ownership explicit | CTRL-X, PROD-Q | OPEN | — |
| FRZ-REL-03 | Reliability | idempotency ownership explicit | CTRL-X, PROD-Q | OPEN | — |
| FRZ-REL-04 | Reliability | cancellation propagates correctly | CTRL-X, PROD-Q | OPEN | — |
| FRZ-REL-05 | Reliability | external operation termination semantics explicit | CTRL-X, PROD-Q | OPEN | — |
| FRZ-REL-06 | Reliability | provider failure typed | CTRL-X, PROD-Q | OPEN | — |
| FRZ-REL-07 | Reliability | no hidden infinite retry/fallback | CTRL-X, PROD-Q | OPEN | — |
| FRZ-CTL-01 | Cross-cutting control planes | Security: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-02 | Cross-cutting control planes | Reliability: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-03 | Cross-cutting control planes | Cost/Budget: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-04 | Cross-cutting control planes | Evaluation: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-05 | Cross-cutting control planes | Critic/Verification: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-06 | Cross-cutting control planes | Observability: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-07 | Cross-cutting control planes | Diagnostics: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-08 | Cross-cutting control planes | Tools: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-09 | Cross-cutting control planes | Skills: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-10 | Cross-cutting control planes | Agent Registry/Assembly: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-11 | Cross-cutting control planes | Capability Graph: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-CTL-12 | Cross-cutting control planes | Context Engineering: one owner; one canonical boundary; no peer execution/governance authority; typed contracts; regression protection | CTRL-X | OPEN | — |
| FRZ-PRD-01 | Production qualification | production qualification independent of lab/harness | PROD-Q | OPEN | — |
| FRZ-PRD-02 | Production qualification | provider/plugin qualification enforced | PROD-Q | OPEN | — |
| FRZ-PRD-03 | Production qualification | unsupported production configuration fails closed | PROD-Q | OPEN | — |
| FRZ-PRD-04 | Production qualification | startup lifecycle qualified | PROD-Q | OPEN | — |
| FRZ-PRD-05 | Production qualification | shutdown/resource cleanup qualified | PROD-Q | OPEN | — |
| FRZ-PRD-06 | Production qualification | degraded behavior explicit | PROD-Q | OPEN | — |
| FRZ-PRD-07 | Production qualification | strict vs lab semantics separated | PROD-Q | OPEN | — |
| FRZ-PRD-08 | Production qualification | production-only bypass count = 0 | PROD-Q | OPEN | — |
| FRZ-REG-01 | Regression / qualification infrastructure | every frozen invariant has evidence protection | QUAL-X | OPEN | — |
| FRZ-REG-02 | Regression / qualification infrastructure | every corrected blocker has regression gate | QUAL-X | OPEN | `test_ebh_2f_r2_integration_plugin_contract_typing_gate.py` — 14 historical R2 invariants + `resolve_typed` overload/impl AST + embedded pyright. |
| FRZ-REG-03 | Regression / qualification infrastructure | negative tests prove gates detect violations | QUAL-X | OPEN | — |
| FRZ-REG-04 | Regression / qualification infrastructure | allowlists minimal and justified | QUAL-X | OPEN | — |
| FRZ-REG-05 | Regression / qualification infrastructure | stale inventories = 0 | QUAL-X | OPEN | — |
| FRZ-REG-06 | Regression / qualification infrastructure | deterministic qualification tests | QUAL-X | OPEN | — |
| FRZ-REG-07 | Regression / qualification infrastructure | clean checkout reproducibility | QUAL-X | OPEN | — |
| FRZ-REG-08 | Regression / qualification infrastructure | environment failures cannot become false PASS | QUAL-X | OPEN | — |
| FRZ-REG-09 | Regression / qualification infrastructure | critical invariant protected by code/test, not docs only | QUAL-X | OPEN | EBH-2F CLOSED (`765fabe…`): integration gates + 559 tests. **EBH-2G-R1** CLOSED (`94eb6ae…`, paths reconciled in `c65832a2…`): canonical `rag-guard` RAG trigger set = `vector_store/**`, `embedding_provider/**`, `document_parser/**`, `rerank_provider/**`, `graph_store/**`; `search_provider/**` audited and intentionally excluded (canonical RAG retrieval does not depend on Integration `SEARCH_PROVIDER` category); collection hygiene (`d6eba2c6…`); mechanical RAG regression gates. Global FRZ-REG closure remains **QUAL-X**. |
| FRZ-REG-10 | Regression / qualification infrastructure | mandatory freeze qualification suite defined | QUAL-X | OPEN | — |
| FRZ-HRN-01 | Harness / Top-Tier | INV-1..INV-34 recertified current HEAD | HARNESS-FINAL | OPEN | — |
| FRZ-HRN-02 | Harness / Top-Tier | A-Z Top-Tier audit repeated current HEAD | HARNESS-FINAL | OPEN | — |
| FRZ-HRN-03 | Harness / Top-Tier | pluginability matrix reconciled | HARNESS-FINAL | OPEN | — |
| FRZ-HRN-04 | Harness / Top-Tier | governance matrix reconciled | HARNESS-FINAL | OPEN | — |
| FRZ-HRN-05 | Harness / Top-Tier | durability matrix reconciled | HARNESS-FINAL | OPEN | — |
| FRZ-HRN-06 | Harness / Top-Tier | recovery matrix reconciled | HARNESS-FINAL | OPEN | — |
| FRZ-HRN-07 | Harness / Top-Tier | historical DeepSeek findings closed/superseded/non-blocking | HARNESS-FINAL | OPEN | — |
| FRZ-HRN-08 | Harness / Top-Tier | no PARTIAL/GAP/TARGET remains in frozen scope | HARNESS-FINAL | OPEN | — |
| FRZ-DOC-01 | Documentation / canon | canonical docs match code | EBH-6, ARCH-FREEZE | OPEN | — |
| FRZ-DOC-02 | Documentation / canon | no conflicting semantic authority | EBH-6, ARCH-FREEZE | OPEN | — |
| FRZ-DOC-03 | Documentation / canon | ownership docs current | EBH-6, ARCH-FREEZE | OPEN | — |
| FRZ-DOC-04 | Documentation / canon | dependency/layer model current | EBH-6, ARCH-FREEZE | OPEN | — |
| FRZ-DOC-05 | Documentation / canon | public contracts manifest current | EBH-6, ARCH-FREEZE | OPEN | — |
| FRZ-DOC-06 | Documentation / canon | composition manifest current | EBH-6, ARCH-FREEZE | OPEN | — |
| FRZ-DEBT-01 | Architecture debt | unresolved architecture blocker = 0 | ARCH-FREEZE | OPEN | — |
| FRZ-DEBT-02 | Architecture debt | unresolved transitional authority = 0 | ARCH-FREEZE | OPEN | — |
| FRZ-DEBT-03 | Architecture debt | temporary compatibility seam in frozen scope = 0 | ARCH-FREEZE | OPEN | — |
| FRZ-DEBT-04 | Architecture debt | PARTIAL inside frozen scope = 0 | ARCH-FREEZE | OPEN | — |
| FRZ-DEBT-05 | Architecture debt | DEFERRED inside frozen scope = 0 | ARCH-FREEZE | OPEN | — |
| FRZ-DEBT-06 | Architecture debt | every N/A/outside-scope item evidence-backed | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-01 | Freeze mechanics | all mandatory roadmap stages CLOSED | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-02 | Freeze mechanics | all FRZ criteria PASS or evidence-backed N/A | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-03 | Freeze mechanics | mandatory qualification suite green | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-04 | Freeze mechanics | exact freeze SHA recorded | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-05 | Freeze mechanics | canonical contract manifest frozen | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-06 | Freeze mechanics | canonical layer manifest frozen | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-07 | Freeze mechanics | ownership manifest frozen | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-08 | Freeze mechanics | composition manifest frozen | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-09 | Freeze mechanics | non-blocking debt register frozen | ARCH-FREEZE | OPEN | — |
| FRZ-FRZ-10 | Freeze mechanics | post-freeze change policy active | ARCH-FREEZE | OPEN | — |

---

## Accumulated evidence log

This log records independently audited evidence contributions without upgrading global freeze criteria prematurely. `OPEN` remains `OPEN` until the criterion's designated closing stage proves the criterion at platform scope.

| Evidence ID | Exact SHA / HEAD | Scope | FRZ criteria informed | Effect |
| --- | --- | --- | --- | --- |
| EBH-2F-R1 | `324ad07d60211cb9c72bcff668e988f180e6ff5b` | Final host-execution-boundary recertification. Tier-3 production hosts consume `runtime.execution` / `HostTaskExecutionPort`; shared harness execution wiring is governance-only and contains no `NexusLoop`; Nexus-backed materialization remains in sanctioned runtime/composition owners; revision admission is strongly typed; structural custom execution-port replaceability and anti-regression gates were independently verified. | FRZ-BND-04, FRZ-BND-05, FRZ-OWN-03, FRZ-CTR-01, FRZ-CTR-02, FRZ-CTR-06, FRZ-TYP-01, FRZ-TYP-03, FRZ-PLG-02, FRZ-RPL-02, FRZ-RPL-03, FRZ-EXE-01, FRZ-EXE-03, FRZ-REG-02, FRZ-REG-09 | Evidence contribution only; criteria remain `OPEN` until their platform-wide closing stages. |
| R1-SQLITE-ENV-01 | `324ad07d60211cb9c72bcff668e988f180e6ff5b` | `test_governed_contractor_http_root_uses_canonical_execution_facade` still fails before reaching the execution-facade spy with `sqlite3.DatabaseError: file is not a database` during collaborative-work persistence bootstrap. Independent R1 audit found no causal relation to the host-execution boundary. | FRZ-STA-03, FRZ-PRD-04, FRZ-REG-08 | Open evidence item. Must be resolved or independently classified with reproducible environment/test-isolation evidence before the applicable `STATE-X` / `PROD-Q` / `QUAL-X` criteria can close. It is not an R1 blocker. |
| EBH-2F-R2-BLOCKER | `8c89bdfa046bbf124ed838b64834bdc825bb8f57` | Independent audit rejected EBH-2F parent closure: weak `IntegrationPlugin` / `IntegrationFactory` typing and `CONTRACT_SPECS` reflection on `register_integration_plugin`. **Superseded** by R2 remediation chain + parent recert on HEAD `26f529e8cac295a9791f2b5763ae9d43da7633c5` (mechanical gates); EBH-2F-R2 awaits independent parent closure. | FRZ-CTR-01, FRZ-CTR-03, FRZ-CTR-05, FRZ-TYP-01, FRZ-TYP-02, FRZ-TYP-04, FRZ-PLG-01, FRZ-PLG-02, FRZ-REG-02, FRZ-REG-09 | Historical blocker record retained; FRZ rows remain OPEN until platform-wide closure stages. |
| EBH-2F-R2-R1-BLOCKER | `8570c8440ddf75d1256f58ecb5182d9a1a441ec5` | Independent audit rejected EBH-2F-R2-R1 pre-built closure: `instance_for_category` accepted generic `PlatformIntegrationContract`, wrong category, and plain objects. Tracked as EBH-2F-R2-R1-R1 (CLOSED at audit scope) with follow-on EBH-2F-R2-R1-R1-R1. | FRZ-CTR-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-06, FRZ-PLG-02, FRZ-REG-02, FRZ-REG-09 | Blocker evidence; FRZ rows remain OPEN until independent closure at platform scope. |
| EBH-2F-R2-R1-R1-BLOCKER | `3c6f84b5ca8c5478ec3908f17fabfaf467567146` | Independent audit rejected EBH-2F-R2-R1-R1 closure: registry-only `contract_for_category` broke DI-only `external_work` pre-built binding. Tracked as EBH-2F-R2-R1-R1-R1 with follow-on EBH-2F-R2-R1-R1-R1-R1. | FRZ-CTR-01, FRZ-OWN-02, FRZ-OWN-05, FRZ-TYP-02, FRZ-TYP-03, FRZ-PLG-02, FRZ-REG-02, FRZ-REG-09 | Blocker evidence; FRZ rows remain OPEN until independent closure at platform scope. |
| EBH-2F-R2-R1-R1-R1-BLOCKER | `fd7a12f773406d1c7babb79388aba30526a17aa6` | Independent audit rejected EBH-2F-R2-R1-R1-R1 closure: category contract resolution unified, but public result APIs still annotated as `PlatformIntegrationContract` while `external_work` legally returns `ExternalWorkIntegration`. Tracked as EBH-2F-R2-R1-R1-R1-R1 with follow-on EBH-2F-R2-R1-R1-R1-R1-R1. | FRZ-CTR-01, FRZ-CTR-04, FRZ-CTR-05, FRZ-CTR-06, FRZ-TYP-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-06, FRZ-PLG-01, FRZ-PLG-02, FRZ-PLG-04, FRZ-REG-02, FRZ-REG-09 | Blocker evidence; FRZ rows remain OPEN until independent closure at platform scope. |
| EBH-2F-R2-R1-R1-R1-R1-BLOCKER | `77054d545143b86a2fb3082448919adfcc1e3a5a` | Independent audit rejected EBH-2F-R2-R1-R1-R1-R1 closure: catalog-only `resolve()` over-declared `CategoryIntegrationInstance`. **Superseded** by child closure `be8c6aaec48431d74dc93572fa27e34433e7c4c5`; EBH-2F-R2 parent recert CURRENT on HEAD `26f529e8cac295a9791f2b5763ae9d43da7633c5`. | FRZ-BND-02, FRZ-BND-03, FRZ-OWN-01, FRZ-OWN-02, FRZ-OWN-05, FRZ-CTR-01, FRZ-CTR-04, FRZ-CTR-05, FRZ-CTR-06, FRZ-TYP-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-06, FRZ-PLG-01, FRZ-PLG-02, FRZ-PLG-04, FRZ-REG-02, FRZ-REG-09 | Historical blocker record retained; FRZ rows remain OPEN until platform-wide closure stages. |
| EBH-2F-R2-R1-R1-R1-R1-R1 | `be8c6aaec48431d74dc93572fa27e34433e7c4c5` | Independent audit accepted catalog vs profile resolution result precision (`resolve` → `PlatformIntegrationContract`; profile surfaces → `CategoryIntegrationInstance`). | FRZ-CTR-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-REG-02, FRZ-REG-09 | Child CLOSED; FRZ criteria remain OPEN at platform scope. |
| EBH-2F-R2-PARENT-RECERT | `26f529e8cac295a9791f2b5763ae9d43da7633c5` | Independent parent recertification **rejected** closure: public `resolve_typed.resolve_contract()` generic overload + implementation declared semantic `Any`. EBH-2F-R2 **BLOCKED**; remediation **EBH-2F-R2-R2** CURRENT. | FRZ-CTR-01, FRZ-CTR-04, FRZ-CTR-05, FRZ-TYP-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-06, FRZ-REG-02, FRZ-REG-09 | Blocker evidence; FRZ rows remain OPEN. |
| EBH-2F-R2-R2 | `311064ffc9534e053498f5ae1d70f1c398cb319b` | Independent audit accepted generic integration resolution helper strong typing (`resolve_contract` → `CategoryIntegrationInstance` / `T`; semantic `Any` = 0; cast/type-ignore = 0). **Historical accepted evidence.** | FRZ-CTR-01, FRZ-CTR-04, FRZ-CTR-05, FRZ-TYP-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-06, FRZ-OWN-02, FRZ-OWN-05, FRZ-PLG-01, FRZ-PLG-02, FRZ-REG-02, FRZ-REG-09 | Child CLOSED; FRZ criteria remain OPEN at platform scope. |
| EBH-2F-R2 | `f7fabab880595c6baf941847219701e466f67c39` | Independent audit accepted R2 parent closure on exact GitHub SHA (typed plugin/factory, explicit specs, no reflection, catalog/profile precision, DI-only `external_work`, `resolve_typed` strong typing, structural external plugin, fail-closed negatives, regression gate). | FRZ-CTR-01, FRZ-CTR-03..06, FRZ-TYP-01..06, FRZ-OWN-02, FRZ-OWN-05, FRZ-PLG-01..02, FRZ-REG-02, FRZ-REG-09 | Child CLOSED at platform stage scope; FRZ rows remain OPEN until platform-wide closure stages. |
| EBH-2F-PARENT-RECERT-FINAL | `765fabe38d0f35fe9087ae811887382b41aa9323` | Historical Cursor AI parent recertification (R1+R2+cross-boundary) — superseded by independent **EBH-2F CLOSED** on same SHA. | FRZ-BND-01..06, FRZ-OWN-01..05, FRZ-CTR-01..06, FRZ-TYP-01..06, FRZ-PLG-01..08, FRZ-REG-02, FRZ-REG-09, FRZ-RPL-02..04 | Evidence contribution only; all FRZ rows remain OPEN. |
| EBH-2F | `765fabe38d0f35fe9087ae811887382b41aa9323` | Independent audit accepted Integrations & Hosting parent closure (R1 host execution + R2 integration/plugin + cross-boundary). Post-closure HEAD delta to `ba41ef759…`: suspended-operation lease/authority only — EBH-2F invariants preserved. | FRZ-BND-01..06, FRZ-OWN-01..05, FRZ-CTR-01..06, FRZ-TYP-01..06, FRZ-PLG-01..08, FRZ-REG-02, FRZ-REG-09, FRZ-RPL-02..04 | Stage CLOSED; FRZ criteria remain OPEN at platform scope. |
| EBH-2G-AUDIT | `ba41ef7592c07df04d75a15cd2efc36fe86bd041` | Cursor AI closed-world RAG boundary audit: single `RetrievalService` semantic owner; integration backends via catalog/`integration_vectorstore`; tool seam delegates to RAG; no agent direct vector SDK. **BLOCKED** — semantic typing gaps (`resolve_retrieval_service` `Any`, `_candidates_to_chunks` duck-typing), targeted pyright 49 errors on retrieval/tool surfaces, `rag-guard` path coverage gap. Child **EBH-2G-R1** required. | FRZ-BND-01..06, FRZ-OWN-01..05, FRZ-CTR-01..06, FRZ-TYP-01..06, FRZ-PLG-01..08, FRZ-REG-02, FRZ-REG-09, FRZ-RPL-01..04 | Evidence contribution only; all FRZ rows remain OPEN. |
| EBH-2G-R1 | `94eb6ae87532057f96cb1512a1487b1f6194ae28` | Independent exact-SHA audit accepted retrieval typing / ABI / qualification closure (remediation chain `d6eba2c6…`, `c65832a2…`, `bf0b5f02…`, anchor `94eb6ae…`): strong typing on composition seam, canonical `RetrievalHit` ABI, typed capability contract, `MetadataFilter` and query-embedding boundaries, package/leaf import gate, reconciled `rag-guard` paths, targeted pyright = 0, mechanical regression gates. | FRZ-BND-04, FRZ-BND-05, FRZ-CTR-01, FRZ-CTR-02, FRZ-TYP-01..04, FRZ-TYP-06, FRZ-REG-09 | Child CLOSED; FRZ criteria remain OPEN at platform scope; **EBH-2G** parent **BLOCKED** by **EBH-2G-R2**. |
| EBH-2G-R2-R1 | `be7e81755fe720bc02e02736a9aa20007083e7e5` | Independent exact-SHA audit accepted Graph Store ownership remediation: Integration-owned provider selection; RAG-local registry and `graph_store_backend` authority removed; typed Integration→RAG graph adaptation; fail-closed resolution; preserved RAG DI and local/harness `InMemoryGraphStore`; ownership regression gate; targeted pyright = 0; RAG gates green per audited implementation. Post-closure HEAD `7fe4e00…` delta unrelated to R2-R1 scope. | FRZ-BND-04, FRZ-BND-05, FRZ-OWN-01..03, FRZ-OWN-05, FRZ-CTR-01..02, FRZ-CTR-04..06, FRZ-TYP-01..02, FRZ-TYP-04, FRZ-TYP-06, FRZ-PLG-01, FRZ-PLG-03..05 | Child CLOSED; FRZ criteria remain OPEN at platform scope; **EBH-2G-R2** **BLOCKED** by **EBH-2G-R2-R2** (**CURRENT**). |
| EBH-2G-R2-R2 | `29a27007204c93f13fec6a3c70ecb0c96f17fc0f` | Independent exact-SHA audit accepted RAG composition typing / contract-purity closure: S1–S4; canonical `IntegrationProfile` / `IntegrationContractSpec` / `RerankProvider` seams; no compatibility profile import, duplicate mode authority, legacy `rerank_scores`, dynamic rerank probing, or concrete rerank-provider leakage; typed `Sequence[RerankerResult]` result boundary; provider manifest/maturity metadata path remains lightweight; targeted pyright and RAG regression gates green. `.env.example` UTF-8 qualification residual was normalized without semantic configuration change. Direct `_shared.health` import cycle remains tracked freeze debt for EBH-3. | FRZ-BND-04, FRZ-BND-05, FRZ-OWN-01..05, FRZ-CTR-01..02, FRZ-CTR-04..06, FRZ-TYP-01..04, FRZ-TYP-06, FRZ-PLG-01, FRZ-PLG-03..05, FRZ-REG-08, FRZ-REG-09 | Stage CLOSED at R2-R2 scope; evidence contribution only. All listed FRZ criteria remain `OPEN` until their designated platform-wide closing stages. |

---

## Freeze Criteria Coverage Matrix

Completeness detector: every FRZ family must have at least one primary closing stage. If a family lacks a stage owner, the program is incomplete.

| Stage | Primary FRZ families |
| --- | --- |
| EBH-2* | BND, OWN, CTR, TYP, PLG |
| HARNESS-* | HRN, EXE, GOV, OBS |
| GOV-X1 / GOV-X2 | GOV, EXE, TRC |
| EBH-3 | BND, OWN, CTR |
| EBH-4 | BND, CTR, EXE, GOV |
| CTRL-X | CTL, SEC, REL, OBS |
| STATE-X | STA, REC |
| TRACE-X | TRC, OBS, GOV |
| COMPAT-X | CMP |
| PROD-Q | PRD, SEC, REL |
| QUAL-X | REG |
| EBH-5 | PLG, RPL |
| EBH-6 | all applicable architecture families |
| EBH-7 | all enterprise families |
| ARCH-FREEZE | FRZ, DEBT, DOC + all remaining |

### FRZ family → stage owners

| FRZ family | Primary closing stage(s) |
| --- | --- |
| BND | EBH-2*, EBH-3, EBH-4 |
| OWN | EBH-2*, EBH-3 |
| CTR | EBH-2*, EBH-3, EBH-4 |
| TYP | EBH-2* |
| PLG | EBH-2*, EBH-5 |
| RPL | EBH-5 |
| EXE | HARNESS-*, EBH-4, GOV-X* |
| GOV | HARNESS-*, GOV-X* |
| OBS | HARNESS-*, TRACE-X, CTRL-X |
| TRC | TRACE-X, GOV-X* |
| STA | STATE-X |
| REC | STATE-X |
| CMP | COMPAT-X |
| SEC | CTRL-X, PROD-Q |
| REL | CTRL-X, PROD-Q |
| CTL | CTRL-X |
| PRD | PROD-Q |
| REG | QUAL-X |
| HRN | HARNESS-FINAL |
| DOC | EBH-6, ARCH-FREEZE |
| DEBT | ARCH-FREEZE |
| FRZ | ARCH-FREEZE |

---

## Update protocol

### Before task

1. Read [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md).
2. Read this checklist.
3. Resolve current task, parent, next mandatory stage, applicable FRZ-* IDs, blockers, and domain semantic authorities.

### After Cursor implementation

READY FOR AUDIT only → independent exact-SHA audit required before PASS or stage CLOSED.

### After independent closure

Atomically update:

1. roadmap stage/status and §5 evidence ledger;
2. checklist Status and Evidence for every FRZ-* advanced by that closure.

---

## Stable IDs

FRZ-* IDs in this document are stable. Renumbering requires an explicit migration note in both this file and the roadmap.
