# Governed Contractor — Implementation Plan

**The implementation map** for the Governed External Contractor (GEC) vertical — phases, status, and verification.

**Status:** Working draft (2026-07-20) — **GEC-0 Done**; GEC-1…GEC-11 Planned  
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
| **GEC-1** | Contractor domain contracts | Planned | High |
| **GEC-2** | External contractor integration contract | Planned | High |
| **GEC-3** | Tier-2 adapter agent | Planned | High |
| **GEC-4** | Quote-first HITL lifecycle | Planned | High |
| **GEC-5** | Meaningful side-effect policy | Planned | High |
| **GEC-6** | Governed contractor receipt | Planned | High |
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
| **Implementation tasks** | Draft pydantic/dataclass contracts; version schema ids; unit tests for validation and serialization |
| **Files / packages** | Likely `intergrax/contracts/` (or agreed platform package) — exact path decided in GEC-1 |
| **Tests** | Contract unit tests; JSON round-trip; reject invalid quote/acceptance payloads |
| **Acceptance gates** | Contracts importable without `applications/` or partner URLs; docs updated |
| **Non-goals** | HTTP client, partner mapping, HITL UI, adapter lifecycle logic |
| **Dependencies** | GEC-0 |
| **Closeout evidence** | Contract tests green; ARCHITECTURE §5/§9/§11 cross-links updated |

---

## GEC-2 — External contractor integration contract

| Field | Content |
|-------|---------|
| **Goal** | Platform integration surface for external contractor agents (discover, create task, quote, status, deliverables) |
| **Architecture impact** | Provider-neutral integration category/protocol; stub + future partner providers |
| **Implementation tasks** | Integration contract + registry wiring pattern; no partner hardcoding in core |
| **Files / packages** | `intergrax/integrations/` (contract + optional stub provider) |
| **Tests** | Integration contract tests; fail-closed on missing config |
| **Acceptance gates** | Adapter can depend on integration API without importing applications |
| **Non-goals** | Live partner SDK; Tier-3 serving routes; HITL |
| **Dependencies** | GEC-1 |
| **Closeout evidence** | Integration unit tests; ADR if category/shape is non-obvious |

---

## GEC-3 — Tier-2 adapter agent

| Field | Content |
|-------|---------|
| **Goal** | Implement `ExternalContractorAdapterAgent` as domain adapter (map external lifecycle → Intergrax contracts) |
| **Architecture impact** | Typed steps/hooks; correlation; idempotency; **no** orchestration ownership |
| **Implementation tasks** | Replace scaffold stubs in `steps/` / agent hooks; wire integration dependency; agent tests |
| **Files / packages** | `agents/external_contractor_adapter/` |
| **Tests** | Agent unit tests with stub integration; capability `external_contractor.adapt` |
| **Acceptance gates** | Adapter has no HITL accept/reject; no policy decisions; no `applications/` imports |
| **Non-goals** | Public API; ProofReceipt store; partner-specific URLs in agent code |
| **Dependencies** | GEC-1, GEC-2 |
| **Closeout evidence** | Agent tests green; agent ARCHITECTURE updated to “implemented baseline” |

---

## GEC-4 — Quote-first HITL lifecycle

| Field | Content |
|-------|---------|
| **Goal** | Pause for quote acceptance via existing runtime HITL; continue only after accept |
| **Architecture impact** | Nexus + HITL own gate; Tier-3 presents quote; adapter resumes on decision |
| **Implementation tasks** | Wire HITL decision points; map accept/reject to adapter continue/stop; tests for both paths |
| **Files / packages** | Runtime HITL usage from host/adapter boundary; Tier-3 presentation hooks |
| **Tests** | Unit/integration: reject stops side effects; accept allows continue |
| **Acceptance gates** | No continue-after-quote without HITL accept record |
| **Non-goals** | Wallet/payment product; Slack/tray UX |
| **Dependencies** | GEC-3 |
| **Closeout evidence** | HITL path tests; trace shows decision correlation |

---

## GEC-5 — Meaningful side-effect policy

| Field | Content |
|-------|---------|
| **Goal** | Enforce policy on post-acceptance external mutations and deliverable writes |
| **Architecture impact** | Host policy bundles; runtime enforcement; adapter remains decision-free |
| **Implementation tasks** | Policy pack for GEC side effects; deny/allow tests; document rule ownership |
| **Files / packages** | `host/policy/` + platform policy wiring |
| **Tests** | Policy unit tests; denied path does not call external mutate |
| **Acceptance gates** | Side effects covered; bypass impossible from adapter alone |
| **Non-goals** | Full enterprise policy authoring UX |
| **Dependencies** | GEC-4 |
| **Closeout evidence** | Policy tests + ARCHITECTURE §7 confirmation |

---

## GEC-6 — Governed contractor receipt

| Field | Content |
|-------|---------|
| **Goal** | Emit/store governed receipt via existing ProofReceipt / DocumentStore path |
| **Architecture impact** | Receipt schema for GEC evidence; Tier-3 exposure later in GEC-7 |
| **Implementation tasks** | Map normalized evidence → receipt; persist; unit tests |
| **Files / packages** | Platform proof receipt usage; thin host glue only |
| **Tests** | Receipt create/load; correlation to `run_id` / external task id |
| **Acceptance gates** | No parallel receipt stack inside the application package |
| **Non-goals** | Partner cryptographic attestation product |
| **Dependencies** | GEC-3…GEC-5 |
| **Closeout evidence** | Receipt tests; schema id documented |

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

## 2. Verification (GEC-0)

```bash
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

## 3. Recommended first task after GEC-0

**GEC-1:** Introduce provider-neutral contractor domain contracts (quote, acceptance decision ref, external task correlation, status, deliverable refs) under `intergrax/` — not under this application package.
