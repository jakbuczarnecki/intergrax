# external_contractor_adapter - architecture

**Status:** GEC-3…GEC-6 baseline (2026-07-20) - mapping + governed continuation + side-effect policy + descriptive proof profile composition; no transport / partner SDK / receipt persistence  
**Vertical:** Governed External Contractor (GEC)  
**Platform reference:** [`docs/project/technical/platform/governed_external_execution.md`](../../../docs/project/technical/platform/governed_external_execution.md) - ownership · lifecycle · invariants
**Implementation tracker:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
**Agent ADRs:** [`adr/README.md`](adr/README.md)  
**Host architecture:** [`applications/governed_contractor_application/docs/ARCHITECTURE.md`](../../../applications/governed_contractor_application/docs/ARCHITECTURE.md)

This agent is a **Tier-2 domain adapter**, not a second orchestration system. Nexus owns multi-step task orchestration; this package maps external work into Intergrax contracts via the GEC-2 Protocol and may **surface** a governed continuation blocker / **forward** continuation evidence (GEC-4).

---

## Purpose

Prove that GEC-1/GEC-2 abstractions are sufficient for a reusable Tier-2 adapter:

```text
Tier-3 Application
  → ExternalContractorAdapterAgent / ExternalWorkAdapter
  → ExternalWorkIntegration  (injected)
  → Deterministic fake (tests) | future provider implementation
```

The adapter owns discovery, create/correlate, quote/timeline/deliverable/evidence **normalization**, and correlation preservation only. It does **not** own governance.

---

## Capabilities

| Capability | Role |
|------------|------|
| `external_contractor.adapt` | Primary adapter capability (host default) |

Provider feature tokens (`ExternalWorkCapability`) are discovered at runtime - never assumed.

---

## Typed inputs and outputs (GEC-1 platform; consumed in GEC-3)

| Direction | Platform type | Notes |
|-----------|---------------|-------|
| In | Governed `task_id` / `run_id` + `ExternalTaskCorrelation` | Intergrax identity remains primary |
| In | `QuoteAcceptanceEvidence` (from runtime HITL refs) | Adapter consumes; does not create |
| Out | `CommercialQuote` / `MoneyAmount` | For Tier-3 presentation |
| Out | `ExternalWorkStatus` timeline | Not Nexus `TaskState` |
| Out | `ExternalDeliverableRef` | Workspace-safe resource URI |
| Out | Normalized tool/evidence facts | For receipts - not partner hardcoding |

Canonical modules: `intergrax.contracts.external_work`, `intergrax.contracts.money` ([ADR-EXTWORK-001](../../../docs/project/technical/adr/entries/2026-07-20/ADR-EXTWORK-001.md)).

Integration boundary (GEC-2 Done): `intergrax.integrations.contracts.external_work.ExternalWorkIntegration` ([ADR-EXTWORK-002](../../../docs/project/technical/adr/entries/2026-07-20/ADR-EXTWORK-002.md)).

---

## External integration dependency

Depends only on the **provider-neutral `ExternalWorkIntegration`** Protocol (GEC-2).

| Rule | Detail |
|------|--------|
| Injection | Constructor `external_work=` on the agent; host may supply via `settings.external_work_integration` |
| Mapping entry | `ExternalWorkAdapter` + `steps/domain_job.py` |
| Provider neutrality | No `if provider == …` / transport / partner SDK / A2A / HTTP |
| Fake | `tests/fakes/DeterministicExternalWorkFake` - GEC-3 proof only, not GEC-8/9 stub |

---

## Ownership (GEC-3)

| Owns (mapping) | Does **not** own (governance / lifecycle) |
|----------------|-------------------------------------------|
| Request → `ExternalWorkCreateRequest` | Quote accept/reject decisions |
| Snapshot / quote / timeline / deliverables / evidence normalization | Policy, wallet, payment, spend auth |
| Correlation + idempotency **forwarding** | Retry/poll/resume engines |
| Forwarding `QuoteAcceptanceEvidence` / continuation evidence | Creating acceptance or HITL decisions |
| Surfacing `GovernedContinuationRequest` (`reason=QUOTE`) | Evaluating approvals / resuming Nexus |
| Describing proposed side effects + composing policy boundary | Implementing policy rules / spend limits |
| Forwarding evidence **only after** policy ALLOW | Inferring allow from evidence / resume |
| Composing descriptive `GovernedProofProfile` | ProofReceipt signing / persistence / publication |
| Structured `ExternalWorkError` surfacing | Audit storage / cryptographic attestation |
| `ExternalWorkStatus` as adapter state | Extending Nexus `TaskState` with commercial stages |

### Governed Continuation (GEC-4 consumer)

External Work is the **first specialization** of platform Governed Continuation ([ADR-GOVERNED-CONTINUATION-001](../../../docs/project/technical/adr/entries/2026-07-20/ADR-GOVERNED-CONTINUATION-001.md)):

```text
map quote → surface continuation (QUOTE) → Nexus interrupt (existing)
  → human/policy decision → QuoteAcceptanceEvidence
  → side-effect policy (ACCEPT_QUOTE) → ALLOW → forward via Protocol
```

**Identity rule:** task identity and run identity are distinct. Governed continuation is correlated to a real Nexus `run_id` forwarded from the execution context. Consumers must never synthesize run identity from `task_id`. Missing run identity fails closed (structured correlation error) - Tier-2 does not invent Nexus execution identity.

Adapter APIs: `surface_continuation_blocker` / `with_continuation_surface` / `forward_continuation_evidence`. No `ContinuationRuntime` or quote lifecycle engine here.

### Meaningful side-effect policy (GEC-5 consumer)

Platform contract: `MeaningfulSideEffectRequest` + existing `PolicyDecision` / `PolicyAction` ([ADR-POLICY-SIDE-EFFECT-001](../../../docs/project/technical/adr/entries/2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md)).

| Concept | Rule |
|---------|------|
| Meaningful side effect | External action that may create commitment, mutation, disclosure, access change, or irreversible consequence |
| Quote receipt | **Observational** - surfaces governed continuation; does not gate `get_quote` as a mutation |
| Quote acceptance | **Meaningful** (`ACCEPT_QUOTE`) - policy before `submit_quote_acceptance` |
| Evidence vs policy | `QuoteAcceptanceEvidence` is continuation evidence, not an allow decision |
| Fail closed | Missing evaluator / principal / run identity / indeterminate → no provider call |
| Ownership | Rules in platform/host policy; Tier-2 only describes + composes |

Missing execution identity is represented explicitly and never encoded as a synthetic identifier or placeholder value.

Domain actions (not platform enums): `CREATE_EXTERNAL_WORK`, `ACCEPT_QUOTE`, `CANCEL_EXTERNAL_WORK`.

Provider-bound method classification (`PROVIDER_METHOD_SIDE_EFFECT_CLASS`):

| Method | Class |
|--------|-------|
| `create_work`, `submit_quote_acceptance`, `cancel_work` | meaningful side effect |
| `discover`, `get_work`, `get_quote`, `get_timeline`, `get_deliverables`, `get_evidence` | observational |

### Governed proof profile (GEC-6 consumer)

> A proof profile is a description of governed execution, not a receipt, not an audit log, and not an authorization mechanism.

**Invariant:** Every successful policy-authorized meaningful side effect produces a `GovernedProofProfile`. Proof composition is mandatory, not best-effort. Proof-required identities (principal / task / run) are validated before the provider-bound call and reused for policy, execution correlation, and proof - a successful side effect must never return without proof.

After a meaningful side effect succeeds under policy ALLOW, the adapter composes `GovernedProofProfile` ([ADR-GOVERNED-PROOF-001](../../../docs/project/technical/adr/entries/2026-07-20/ADR-GOVERNED-PROOF-001.md)):

| Included | Rule |
|----------|------|
| principal / tenant / task_id / run_id | Preserved - never invented |
| action / resource / provider_id | Canonical platform identifiers |
| `PolicyAction` + rule/reason | Referenced from the ALLOW decision - not recomputed |
| `GovernanceEvidenceRef` | Points at artifacts (e.g. quote acceptance id) - no payload embed |
| `correlation_id` / `idempotency_key` | Preserved from the request |
| Optional `ContinuationReason` | When continuation evidence was involved |

Tier-2 does **not** sign, hash, store, or publish proofs. `ProofReceipt` persistence remains a later platform capability.

Deferred: HITL UX, product policy packs / business rules, ProofReceipt persistence, workspace publication, polling engines, real providers.

---

## Synchronous mapping flow (GEC-3 + GEC-5 gate + GEC-6 proof)

```text
discover → policy(CREATE) → create_work (idempotent) → compose GovernedProofProfile
  → [no acceptance] enrich reads (observational) → optional QUOTE continuation surface
  → [acceptance] policy(ACCEPT_QUOTE) → submit_quote_acceptance → compose proof (evidence refs)
```

No poll loops, sleep, background workers, or retry engines in this package.

---

## `Agent.run()` / typed `on_next_step` alignment

| Item | GEC-3 baseline |
|------|----------------|
| Pattern | ACP **reflex** (`CognitivePattern.REFLEX`) |
| Entry | `perceive` / `act` → `run_domain_job` → `ExternalWorkAdapter` |
| LLM | Stub adapter for offline smoke; mapping is deterministic Protocol calls |

Do **not** build an internal orchestration graph that duplicates Nexus.

---

## Ownership matrix

| Concern | Owner |
|---------|-------|
| Nexus task graph / scheduling | Runtime Nexus |
| Policy allow/deny | Runtime policy + Tier-3 bundles |
| HITL quote accept/reject | Runtime HITL + Tier-3 surfaces |
| Agent Card discovery, quote fetch, status sync, deliverable fetch | **This adapter** |
| Wallet / payment approval | Tier-3 / runtime - **prohibited here** |
| Workspace escape / external publication approval | Tier-3 / runtime - **prohibited here** |
| Reusable contracts | `intergrax` - **not** this package as long-term home |

---

## Idempotency and retries

| Requirement | Rule |
|-------------|------|
| Idempotency | External create/continue must use deterministic keys from Intergrax identity + stage |
| Retries | Safe for reads/status; mutating calls only with idempotency |
| Crash recovery | Correlate existing external task; do not double-create when key present |

---

## External task correlation

Maintain a stable mapping:

```text
intergrax run_id / task_id  ↔  external_task_id  ↔  quote_id
```

Surface correlation in adapter outputs so Tier-3 receipts and traces can join facts.

---

## Status and evidence normalization

| External concept | Adapter duty |
|------------------|--------------|
| Partner status enums/strings | Map to governed status model (GEC-1) |
| Tool calls / work evidence | Normalize to platform evidence shapes |
| Opaque partner blobs | Do not require core to understand partner JSON |

---

## Prohibited responsibilities

- Quote acceptance or rejection decisions
- Wallet or payment approval
- Policy decisions or governance bypasses
- Workspace allowlist escape
- External publication approval
- Competing domain contractor implementation (for example local code review)
- Importing `applications.*`
- Owning Nexus orchestration

---

## Layout

| Path | Role |
|------|------|
| `external_contractor_adapter_agent.py` | Reflex agent + DI for `ExternalWorkIntegration` |
| `external_work_adapter.py` | Provider-neutral mapping translator |
| `contract.py` | `AgentContract` + `cognitive_pattern` |
| `capabilities.py` | Capability ids |
| `steps/domain_job.py` | Domain step entry |
| `schemas/adapt_result.py` | Composed adapter result (platform contracts) |
| `tests` + `tests/fakes` | Agent tests + deterministic fake |
| `docs/project/technical/adr` | Agent ADRs |

---

## Tier hygiene

- Imports only `intergrax.*` and `external_contractor_adapter.*`
- Tools / integrations resolved by Tier-3 host profiles
- Registration: `AgentBinding.mount(...)` in `applications/governed_contractor_application/manifest.py`
