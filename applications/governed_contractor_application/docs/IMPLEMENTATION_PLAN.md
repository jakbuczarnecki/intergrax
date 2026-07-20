# Governed Contractor — Implementation Plan

**The implementation map** for the Governed External Contractor (GEC) vertical — phases, status, and verification.

**Status:** Working draft (2026-07-20) — **GEC-0…GEC-6 Done**; GEC-7…GEC-11 Planned  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Application ADRs:** [`adr/README.md`](adr/README.md)  
**Agent tracker:** [`agents/external_contractor_adapter/docs/IMPLEMENTATION_PLAN.md`](../../../agents/external_contractor_adapter/docs/IMPLEMENTATION_PLAN.md)  
**Partner handoff:** [`PARTNER_HANDOFF.md`](PARTNER_HANDOFF.md)

Principle: **compose Tier-0** · **no business logic in Nexus** · **adapter is not an orchestrator** · **no unsupported maturity claims**

---

## Documentation model

| Topic | Where |
|-------|--------|
| Host purpose, trust, lifecycle, non-goals | **ARCHITECTURE.md** |
| Task status, phases, gates | **This file** |
| Application architecture decisions | **`adr/`** — [`adr/README.md`](adr/README.md) |
| Deploy runbook | **BUILD_AND_DEPLOY.md** |
| Partner quickstart | **PARTNER_HANDOFF.md** |
| Adapter contracts / prohibited duties | `agents/external_contractor_adapter/docs/` |

---

## 0. Scope at a glance

| Field | Value |
|-------|-------|
| Package | `governed_contractor_application` |
| Profile | `product` |
| Route prefix | `/v1/governed_contractor` |
| Default port | `8000` |
| Mounted agents | `external_contractor_adapter` |
| Default capability | `external_contractor.adapt` |
| Vertical phases | GEC-0 … GEC-11 |

---

## 1. Phase queue

| ID | Phase | Status | Priority |
|----|-------|--------|----------|
| **GEC-0** | Bootstrap and canonical documentation | **Done** | High |
| **GEC-1** | Contractor domain contracts | **Done** | High |
| **GEC-2** | Canonical external-work model + provider-neutral integration boundary | **Done** | High |
| **GEC-3** | Tier-2 adapter agent | **Done** | High |
| **GEC-4** | Governed Continuation composition (External Work first consumer) | **Done** | High |
| **GEC-5** | Meaningful side-effect policy | **Done** | High |
| **GEC-6** | Governed proof profile (descriptive) | **Done** | High |
| **GEC-7** | Tier-3 API and proof workflow | Planned | High |
| **GEC-8** | Partner handoff and mapping | Planned | Medium |
| **GEC-9** | Deterministic stub integration | Planned | High |
| **GEC-10** | Live external partner integration | Planned | Medium |
| **GEC-11** | Public end-to-end proof and PASS matrix | Planned | High |

---

## GEC-0 — Bootstrap and canonical documentation

| Field | Content |
|-------|---------|
| **Goal** | Create Tier-3 product host and Tier-2 adapter via canonical scaffolds; establish architecture and phase plan |
| **Architecture impact** | Establishes package layout, roster mount, ADR indexes, trust/lifecycle narrative; no domain runtime yet |
| **Implementation tasks** | Run `new-agent` + `new-application --profile product`; align capability ids; write ARCHITECTURE / IMPLEMENTATION_PLAN / PARTNER_HANDOFF; ADR placeholders |
| **Files / packages** | `applications/governed_contractor_application/**`, `agents/external_contractor_adapter/**` |
| **Tests** | Scaffold smoke tests for agent and host; deploy-triad / doc-pair / ADR scaffold checks |
| **Acceptance gates** | Both trees exist from scaffold; docs cover required sections; no GEC-1 runtime contracts committed as “Done” |
| **Non-goals** | Domain contracts, live partner calls, HITL wiring, receipts, public PASS matrix |
| **Dependencies** | Scaffold CLI (`intergrax.scaffold`), product profile templates |
| **Closeout evidence** | This phase marked **Done**; verification commands in §2 |

---

## GEC-1 — Contractor domain contracts

| Field | Content |
|-------|---------|
| **Goal** | Define reusable, provider-neutral contractor/quote/status/deliverable contracts in platform space |
| **Architecture impact** | Shared types for quote-first lifecycle; consumed by adapter and host; **not** owned by Tier-3 |
| **Implementation tasks** | Reuse audit; platform contracts in `intergrax/contracts/`; unit tests; ADR-EXTWORK-001 |
| **Files / packages** | `intergrax/contracts/money.py`, `intergrax/contracts/external_work.py` |
| **Tests** | `tests/unit/contracts/test_money.py`, `tests/unit/contracts/test_external_work.py` |
| **Acceptance gates** | Contracts importable without `applications/` or partner URLs; docs updated |
| **Non-goals** | HTTP client, partner mapping, HITL UI, adapter lifecycle logic |
| **Dependencies** | GEC-0 |
| **Closeout evidence** | Contract tests green; reuse audit below; ADR-EXTWORK-001 accepted |

### GEC-1 reuse audit

| Required GEC concept | Existing Intergrax mechanism | Decision |
|----------------------|------------------------------|----------|
| Intergrax task identity | `Task.task_id`, `AgentRunRequest` / result `run_id` strings | **reuse** |
| External task identity | No external-task foreign-key contract | **new** — `ExternalTaskCorrelation.external_task_id` (foreign key only) |
| Correlation | `AgentRunRequest.correlation_id` field pattern | **reuse** (optional string on correlation model) |
| Idempotency | Tool/runtime `idempotency_key` string + `IdempotencyStore` | **reuse** (optional key field; no new store) |
| Contractor identity | Integration catalog `provider_id` / slug; no Agent Card type | **compose/new** — `ExternalContractorIdentity` wraps `provider_id` + external ids + optional descriptor ref/digest (no URL) |
| Contractor status | Nexus `TaskState` (orchestration); no quote commercial stages | **new** — `ExternalWorkStatus` (Nexus `TaskState` unchanged) |
| Quote / money | `AgentRunCost` float USD token proxy; budgets are token/limit scopes | **new** — `MoneyAmount` (`Decimal`) + `CommercialQuote` |
| Quote acceptance | `HumanDecisionRecord.decision_id`, `ExecutionInterrupt.interrupt_id`, `ActorIdentity`, `PolicyDecision` | **compose/new** — `QuoteAcceptanceEvidence` refs only; no authz/payment |
| Deliverable reference | `ArtifactRef` / `ApplicationArtifactRef` / workspace refs (require harness provenance) | **compose/new** — `ExternalDeliverableRef` (workspace-safe URI + digest/size conventions) |
| Content digest | Hosted `sha256:<64 hex>` convention | **reuse** — `validate_content_digest` |
| Expiration | Per-model datetime validators elsewhere | **reuse** pattern — aware UTC + `expires_at > created_at` |
| Acceptance matching | `ValidationResult` | **reuse** — `validate_quote_acceptance_match` |

**Why new abstractions are platform-level**

- `MoneyAmount`: commercial exact money is reusable beyond GEC; float LLM cost rollups are a different domain.
- `ExternalWorkStatus`: quote/commercial stages are reusable for any external-work integration; polluting Nexus `TaskState` would couple orchestration to commerce.
- Correlation / quote / acceptance / deliverable models: multiple future apps need the same Intergrax↔external join vocabulary; Tier-2/3 packages must remain consumers.

**Deferred (from GEC-1 closeout):** GEC-2…GEC-6 (**Done**); HITL UX product surfaces; ProofReceipt persistence (later).

---

## GEC-2 — Canonical External Work model and provider-neutral integration boundary

| Field | Content |
|-------|---------|
| **Goal** | Platform-owned interaction model (GEC-2A) + sync `ExternalWorkIntegration` Protocol (GEC-2B) for any external-work provider |
| **Architecture impact** | Reusable beyond contractors; Tier-2/3 consume only; no transport implementation |
| **Implementation tasks** | Domain models + Protocol + `IntegrationCategory.EXTERNAL_WORK` profile binding + contract tests + ADR |
| **Files / packages** | `intergrax/contracts/external_work.py`, `intergrax/integrations/contracts/external_work.py`, profile/category wiring |
| **Tests** | `tests/unit/integrations/test_external_work_integration.py`, GEC-1 contract tests extended |
| **Acceptance gates** | Deterministic in-memory fake implements Protocol; no HTTP/A2A; no Tier-2/3 ownership of boundary |
| **Non-goals** | Live provider; A2A/REST clients; GEC-8 stub provider; HITL/policy/receipts |
| **Dependencies** | GEC-1 |
| **Closeout evidence** | Contract tests green; [`ADR-EXTWORK-002`](../../../docs/adr/entries/2026-07-20/ADR-EXTWORK-002.md) |

### GEC-2 reuse audit (summary)

| Concern | Existing mechanism | Decision |
|---------|-------------------|----------|
| Provider / work identity | `ExternalContractorIdentity`, `ExternalTaskCorrelation` | **reuse** |
| Quote / acceptance / status / deliverables | GEC-1 models | **reuse** |
| Discovery metadata | No external-work descriptor | **new** — `ExternalWorkProviderDescriptor` (+ `ExternalWorkCapability`) |
| Integration interface style | Sync `Protocol` under `integrations/contracts/` | **extend** — `ExternalWorkIntegration` |
| Result/error model | `IntegrationError` family | **extend** — `ExternalWorkError` + `ExternalWorkErrorCode` |
| Cancellation / idempotency | Idempotency keys on correlation + tool ledger (tools only) | **reuse** keys on mutating ops; document retry rules (no new middleware) |
| Timeline | Runtime traces / events (different domain) | **new** — `ExternalWorkTimelineEvent` (provider-observed facts) |
| Provider evidence | ProofReceipt (later) / `GovernedProofProfile` (GEC-6) | **new** refs only — `ExternalProviderEvidenceRef` (distinct from proof) |
| Registry/binding | `IntegrationProfile` / `IntegrationCategory` | **reuse** — `external_work` slot; catalog slug deferred to provider phase |

**Why “External Work” not “Contractor”:** same boundary serves AI contractors, human services, SaaS jobs, and future protocols. GEC remains the first consumer.

**Deferred:** provider packages (A2A/REST), `PROVIDER_CATEGORY_CONTRACT_REGISTRY` slug, HITL UX product, ProofReceipt persistence.

---

## GEC-3 — Tier-2 adapter agent

| Field | Content |
|-------|---------|
| **Goal** | Implement `ExternalContractorAdapterAgent` as domain adapter (map external lifecycle → Intergrax contracts) |
| **Architecture impact** | Typed steps/hooks; correlation; idempotency; **no** orchestration ownership |
| **Implementation tasks** | `ExternalWorkAdapter` mapping; DI for `ExternalWorkIntegration`; deterministic fake tests |
| **Files / packages** | `agents/external_contractor_adapter/` (+ host builder injection hook) |
| **Tests** | Agent unit tests with `DeterministicExternalWorkFake`; capability `external_contractor.adapt` |
| **Acceptance gates** | Adapter has no HITL accept/reject; no policy decisions; no `applications/` imports; no transport |
| **Non-goals** | Public API; ProofReceipt store; partner-specific URLs; polling/resume; real providers |
| **Dependencies** | GEC-1, GEC-2 |
| **Status** | **Done** (2026-07-20) |
| **Closeout evidence** | `uv run pytest agents/external_contractor_adapter/tests -q`; agent ARCHITECTURE “GEC-3 implemented baseline” |

---

## GEC-4 — Governed Continuation composition

| Field | Content |
|-------|---------|
| **Goal** | Compose reusable Governed Continuation over existing Nexus interrupt/HITL/resume; External Work supplies `reason=QUOTE` only |
| **Architecture impact** | Platform `governed_continuation` helpers; Tier-2 surfaces/forwards; Nexus remains sole orchestration runtime |
| **Implementation tasks** | `ContinuationReason` + composition helpers; adapter surface/forward; deterministic composition tests; docs + ADR |
| **Files / packages** | `intergrax/contracts/governed_continuation.py`; Tier-2 adapter; ADR-GOVERNED-CONTINUATION-001 |
| **Tests** | Interrupt composition, resume evidence refs, correlation, Tier-2 non-governance, no duplicate runtime |
| **Acceptance gates** | No new interruption framework; quote is first consumer only; evidence propagated without interpretation |
| **Non-goals** | HITL UX product; quote-specific lifecycle engine; payment/wallet; resume engines; transport |
| **Dependencies** | GEC-3 |
| **Closeout evidence** | `test_governed_continuation*.py` green; architecture gates; ADR accepted |

**Reuse audit:** Nexus interrupt + HITL + resume already sufficient; only generic reason + composition helpers added.

---

## GEC-5 — Meaningful side-effect policy

| Field | Content |
|-------|---------|
| **Goal** | Authorize meaningful external side effects via platform policy before provider execution |
| **Architecture impact** | Platform `MeaningfulSideEffectRequest` + `evaluate_meaningful_side_effect`; host injects evaluator; adapter remains rule-free |
| **Implementation tasks** | Platform contract + PolicyEngine method; Tier-2 gates CREATE/ACCEPT/CANCEL; host DI; docs/ADR |
| **Files / packages** | `intergrax/contracts/meaningful_side_effect.py`; `intergrax/runtime/policy/`; adapter; `host/agent_builders.py` |
| **Tests** | Ordering ALLOW/DENY/REQUIRE_HUMAN; fail-closed; observational quote receipt; host smoke |
| **Acceptance gates** | Policy before mutation; evidence ≠ allow; no Tier-2 rules; boundaries intact |
| **Non-goals** | Payment/wallet; product policy packs; policy admin UX; provider transport |
| **Dependencies** | GEC-4 |
| **Closeout evidence** | `test_meaningful_side_effect_policy.py` + ARCHITECTURE §7 + ADR-POLICY-SIDE-EFFECT-001 |

---

## GEC-6 — Governed proof profile

| Field | Content |
|-------|---------|
| **Goal** | Define reusable descriptive `GovernedProofProfile` for governed external side effects |
| **Architecture impact** | Platform contract + Tier-2 composition; not a receipt/persistence product |
| **Implementation tasks** | Reuse audit; contract + ADR; compose after ALLOW; deterministic tests; docs |
| **Files / packages** | `intergrax/contracts/governed_proof.py`; Tier-2 adapter; ADR-GOVERNED-PROOF-001 |
| **Tests** | Profile composition; identity/policy/evidence refs; no transport/persistence/signing |
| **Acceptance gates** | Descriptive only; provider-neutral; GEC-5 flow unchanged |
| **Non-goals** | Persistence, signatures, ProofReceipt store, audit DB, verification engine |
| **Dependencies** | GEC-3…GEC-5 |
| **Closeout evidence** | Contract + adapter tests green; architecture docs updated |

---

## GEC-7 — Tier-3 API and proof workflow

| Field | Content |
|-------|---------|
| **Goal** | Product API for intake, quote presentation, status, deliverables, trace/receipt exposure |
| **Architecture impact** | Serving routes beyond scaffold `/run`; proof workflow entrypoints |
| **Implementation tasks** | Extend `serving/`; schemas; host smoke + API contract tests |
| **Files / packages** | `applications/governed_contractor_application/serving/`, `host/` |
| **Tests** | Host API tests; happy-path stub workflow |
| **Acceptance gates** | Public task API does not embed partner URLs; auth/tenant settings respected |
| **Non-goals** | Tray frontend; Slack Socket Mode |
| **Dependencies** | GEC-4…GEC-6 |
| **Closeout evidence** | API tests green; BUILD_AND_DEPLOY updated |

---

## GEC-8 — Partner handoff and mapping

| Field | Content |
|-------|---------|
| **Goal** | Document partner field maps and operator handoff without contaminating core |
| **Architecture impact** | Handoff package / docs only; mapping tables for design partner(s) |
| **Implementation tasks** | Expand PARTNER_HANDOFF; sample fixtures; mapping ADR if needed |
| **Files / packages** | `docs/PARTNER_HANDOFF.md`, optional `partner_handoff/` samples |
| **Tests** | Fixture/schema assertion tests (as appropriate) |
| **Acceptance gates** | No partner identity in `intergrax/` |
| **Non-goals** | Live production partner SLA |
| **Dependencies** | GEC-7 |
| **Closeout evidence** | Handoff checklist complete |

---

## GEC-9 — Deterministic stub integration

| Field | Content |
|-------|---------|
| **Goal** | Offline, deterministic external contractor stub for CI proof |
| **Architecture impact** | Stub provider behind GEC-2 contract; enables GEC-11 without live partner |
| **Implementation tasks** | Stub Agent Card, quote, status, deliverable; end-to-end host test |
| **Files / packages** | Integration stub + tests under platform and/or app tests |
| **Tests** | Deterministic E2E with stub |
| **Acceptance gates** | CI-stable; no network to real partner |
| **Non-goals** | Live partner credentials in CI |
| **Dependencies** | GEC-2…GEC-7 |
| **Closeout evidence** | Stub E2E green in CI |

---

## GEC-10 — Live external partner integration

| Field | Content |
|-------|---------|
| **Goal** | Operator-gated live run against a real external contractor agent |
| **Architecture impact** | Env-based partner endpoint config; mapping from GEC-8 |
| **Implementation tasks** | Live proof script/runbook; secrets via env; record evidence |
| **Files / packages** | Scripts/docs under application; config only |
| **Tests** | Manual/operator proof (not required flaky in default CI) |
| **Acceptance gates** | Live path documented; fail-closed without credentials |
| **Non-goals** | Claiming production readiness |
| **Dependencies** | GEC-8, GEC-9 |
| **Closeout evidence** | Operator proof notes / receipt ids |

---

## GEC-11 — Public end-to-end proof and PASS matrix

| Field | Content |
|-------|---------|
| **Goal** | Public-adoptable proof path with explicit PASS matrix |
| **Architecture impact** | Ties GEC into public adoption docs without overstating maturity |
| **Implementation tasks** | PASS matrix; link from public-adoption docs; finalize wording (source-available) |
| **Files / packages** | App docs + optional `docs/public-adoption/` entry |
| **Tests** | Automated stub matrix + documented live optional row |
| **Acceptance gates** | Matrix rows evidence-backed; no “production-ready” claim |
| **Non-goals** | Marketplace listing; open-source relicensing |
| **Dependencies** | GEC-9 (required), GEC-10 (optional live row) |
| **Closeout evidence** | Published PASS matrix with stub PASS |

---

## 2. Verification

```bash
# GEC-1 / GEC-2 / GEC-4 platform contracts + integration boundary
uv run pytest tests/unit/contracts/test_money.py tests/unit/contracts/test_external_work.py -q
uv run pytest tests/unit/contracts/test_governed_continuation.py -q
uv run pytest tests/unit/integrations/test_external_work_integration.py -q
uv run pytest tests/unit/contracts/test_agent_run_roundtrip.py -q

# GEC host / adapter smoke (+ GEC-4 continuation composition)
uv run pytest agents/external_contractor_adapter/tests -q
uv run pytest applications/governed_contractor_application/tests -q
uv run pytest tests/unit/applications/test_application_deploy_triad.py -q -k governed_contractor
uv run pytest tests/unit/scaffold/test_adr_scaffold.py -q -k governed_contractor
```

Local run (scaffold smoke only — not GEC domain proof):

```bash
cp applications/governed_contractor_application/.env.example applications/governed_contractor_application/.env
uv run uvicorn governed_contractor_application.host.main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health
```

---

## 3. Recommended first task after GEC-6

**GEC-7:** Tier-3 API and proof workflow — expose intake/quote/status and proof-profile surfaces (still no wallet/payment product; ProofReceipt persistence remains later).
