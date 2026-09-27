# S24-GAP-04-P0 — External Contract Semantic Compatibility Architecture Lock

## 1. Metadata and baseline

| Field | Value |
| ----- | ----- |
| **Task** | `S24-GAP-04-P0` (architecture lock only) |
| **Pre-audit baseline** | `41bfa6c40ea3b4a15e2c8ef3d432ef578bb82f1c` |
| **Session start HEAD** | `0be469bc020bf4f22590420e1d53427665b28f76` (`development` = `origin/development`) |
| **Relevant-path drift** `41bfa6c..HEAD` | **None** (`intergrax/integrations/**`, `intergrax/runtime/integrations/**`, `intergrax/contracts/**`, `platform_proofs/scenarios/external_api_schema_drift/**`, `docs/project/architecture/INTEGRATIONS.md`) |
| **Branch** | `development` |
| **Artifact role** | Closed-world design record before `S24-GAP-04-P1` |
| **Production / tests / scenario spec** | **0 mutation in P0** |
| **Status** | **ARCHITECTURE LOCKED — Class A** · **UCA REOPEN: NO** · **EE REOPEN: NO** |

---

## 2. Problem

Integrations already own typed provider/backend binding, catalog registration, config, security posture, and health metadata. Scenario **#24** (`external_api_schema_drift`) requires distinguishing **schema**, **protocol**, and **semantic** drift from true compatibility — including **variant H**: HTTP success and deserialization are **not** compatibility proof.

Today the dominant failure mode is implicit:

```text
HTTP 2xx + payload deserializes → assumed COMPATIBLE
```

There is **no** reusable, Integrations-owned mechanism that aggregates **typed expected operation contract** plus **typed observed evidence** into one of exactly five factual outcomes without LLM-only verdicts, schema-only shortcuts, vendor branches in platform core, or recovery/acquisition/authority decisions.

---

## 3. Current repository facts

| Fact | Evidence |
| ---- | -------- |
| Integration identity | `integration_id` = `{provider_id}:{integration_kind}` — [`docs/project/architecture/INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) |
| Catalog authority | Provider-owned `IntegrationContractSpec` via `declare_integration_contract` → canonical Integration Catalog — **single** discovery source |
| `IntegrationContractSpec` scope | Registration metadata: category, provider, contract/integration class, config, capabilities, security, runtime binding — **not** per-operation business semantics — `intergrax/integrations/contracts/contract_spec.py` |
| `PlatformIntegrationContract` | Category runtime contract — unchanged by this lock |
| Plugin parity | `IntegrationPlugin.integration_contract_specs()` + factory — `intergrax/integrations/contracts/plugin.py` |
| Public Integrations contracts | `intergrax/integrations/contracts/**` |
| Runtime integration surface | `intergrax/runtime/integrations/**` (category contracts, metadata) |
| Cross-execution tenant scope | `tenant_id: str` on `ExecutionScopeIdentity`, `TaskEnvelope`, ERL evidence types — **no** new canonical `TenantId` type required |
| Scenario #24 | `platform_proofs/scenarios/external_api_schema_drift/SCENARIO_SPEC.md` — `gap_decision: NOT_COMPLETED`; variant H explicitly rejects deserialize-as-compatible |

**Distinction (frozen):** `IntegrationContractSpec` ≠ **external operation contract** (request/response meaning, protocol invariants, semantic invariants for one provider operation).

---

## 4. Root cause

Typed Integration boundaries exist; **fact-based compatibility assessment** across schema / protocol / semantic dimensions does not. Compatibility conclusions leak into HTTP/deserialize success paths and informal reasoning instead of authoritative typed evidence aggregated deterministically.

---

## 5. Owner matrix

| Concern | Canonical owner |
| ------- | ----------------- |
| Compatibility mechanism (assessment semantics) | **Integrations** |
| Integration/provider catalog registration | Provider package + **Integration Catalog** (unchanged authority) |
| Expected provider-operation technical semantics | Provider Integration package |
| Business/domain semantic invariants | Application/domain via **typed extension** supplied into assessment composition |
| Observed external facts | Integration/provider boundary (adapter); collection via evidence provider SPI |
| Compatibility aggregation | **Integrations** (`ExternalContractCompatibilityService` — P1) |
| Evidence redaction | Provider boundary + existing security posture |
| Recovery disposition A–G, `CONFIGURE_EXISTING`, adaptation | Autonomous Work / recovery (GAP-01, GAP-03) |
| Acquisition, capability gap minting | Capability Acquisition / canonical discovery |
| Authority / principal / OAuth scope changes | Governance / existing authority owners |
| Tool execution | EE + ToolRuntime |
| Proof expected result | Scenario proof layer only |

Compatibility delivers **facts** (`ExternalContractCompatibilityAssessment`). It does **not** decide reuse, adapt, acquire, or change authority.

---

## 6. Non-goals (P0 / mechanism boundary)

- LLM-only or LLM-primary compatibility verdict
- Schema-only shortcut (skip protocol/semantic when required)
- Treating HTTP 2xx or successful deserialization as `COMPATIBLE`
- `IntegrationContractSpec.metadata["schema"]` / arbitrary metadata bags for semantics
- Second Integration Catalog, `VendorCompatibilityMap`, central vendor `if provider_id == …` in generic core
- `dict[str, Any]`, reflection, `getattr`/`setattr` dispatch for verdict paths
- Minting `CapabilityGap`, recovery disposition, Tool invoke, provider I/O inside assessment service
- New event bus or proof-local observability framework
- Semantic modification of `PlatformIntegrationContract`, `IntegrationContractSpec`, or Catalog semantics in P0/P1
- Generic JSON-logic / expression interpreter for business semantics in P1
- Python callables stored inside frozen public contract models

---

## 7. Architecture principle

```text
ExternalContractCompatibilityExpectation
        +
ExternalContractCompatibilityEvidence (authoritative, scoped)
        ↓
ExternalContractCompatibilityService  (pure, no I/O)
        ↓
per-dimension ExternalContractCompatibilityFinding
        ↓
deterministic aggregation
        ↓
ExternalContractCompatibilityAssessment  (five outcomes only)
        ↓
consumers: AW / recovery, proof projection, diagnostics (caller-owned)
```

---

## 8. Dimension model

Frozen enum `ExternalContractCompatibilityDimension`:

```text
SCHEMA
PROTOCOL
SEMANTIC
```

Each required dimension has an independent dimension-level status (see §18). Dimensions are **not** collapsed into one boolean.

---

## 9. Outcome model (exactly five)

```text
ExternalContractCompatibilityOutcome.COMPATIBLE
ExternalContractCompatibilityOutcome.SCHEMA_INCOMPATIBLE
ExternalContractCompatibilityOutcome.PROTOCOL_INCOMPATIBLE
ExternalContractCompatibilityOutcome.SEMANTIC_INCOMPATIBLE
ExternalContractCompatibilityOutcome.INSUFFICIENT_EVIDENCE
```

No `AUTHORITY_CHANGE_REQUIRED`, `RETRY`, `ACQUIRE`, `ADAPT`, or AW disposition codes on this surface.

---

## 10. Scope identity (`ExternalContractCompatibilityScope`)

Immutable scope carried on **expectation**, **evidence**, and **assessment request**. Prevents cross-tenant / cross-operation mixing without inventing a new tenant type.

| Field | Type | Rule |
| ----- | ---- | ---- |
| `tenant_id` | `str` (non-empty) | Aligns with `ExecutionScopeIdentity.tenant_id` / platform envelope conventions |
| `integration_id` | `str` | Canonical `{provider_id}:{integration_kind}` |
| `provider_id` | `str` | Must match parsed identity of `integration_id` |
| `integration_kind` | `str` | Category slug; must match `integration_id` |
| `external_operation_id` | `str` | Provider-owned stable operation key (not URL guess) |
| `contract_ref` | `str` | Typed contract identity (URI or stable ref id — no raw OpenAPI blob) |
| `contract_version` | `str` | Expected or observed version label |
| `host_binding_ref` | `str \| None` | Optional Tier-3 host binding discriminator when multiple profiles share tenant |
| `execution_task_id` | `TaskId \| None` | Optional; when assessment is execution-scoped |
| `execution_run_id` | `RunId \| None` | Optional |
| `execution_id` | `ExecutionId \| None` | Optional |

**Identity continuity:** evidence scope must **exactly** match expectation scope on all non-optional fields and on optional execution fields when expectation sets them. Any mismatch → aggregation never yields `COMPATIBLE` (see §19).

---

## 11. Expected contract model (Q1)

Frozen type: **`ExternalContractCompatibilityExpectation`**.

| Field | Purpose |
| ----- | ------- |
| `expectation_id` | Stable id for assessment correlation |
| `scope` | `ExternalContractCompatibilityScope` (full identity) |
| `required_dimensions` | `frozenset[ExternalContractCompatibilityDimension]` — caller declares which dimensions must be evidenced |
| `contract_ref` | Duplicate of `scope.contract_ref` for explicit contract pinning |
| `contract_version` | Expected version pin |
| `technical_expectation_ref` | Provider-owned ref to schema/protocol spec artifact (fingerprint id, not raw document) |
| `semantic_expectation_refs` | `tuple[str, ...]` — ids of registered semantic assertion specs (not rule bodies in core) |
| `domain_extension_ref` | `str \| None` — pointer to application-supplied typed semantic extension contract (when business invariants apply) |

**Does not store:** raw request/response payload, secrets, or arbitrary JSON schema document bodies.

Provider package owns **technical** truth for schema/protocol expectations and **provider-semantic** assertion catalogs. Application/domain owns **business** invariants via `domain_extension_ref` + domain evaluator (P2), not generic core interpretation.

---

## 12. Expectation resolution (Q2)

**Primary path (P1):** full `ExternalContractCompatibilityExpectation` is supplied on **`ExternalContractCompatibilityAssessmentRequest.expectation`** by the composition root (host wiring, provider module, or proof harness). No Catalog lookup required for assessment.

**Optional SPI (same plugin/composition path as built-in):**

```text
ExternalContractCompatibilityExpectationResolver (Protocol)
  resolver_id: str
  resolve(key: ExternalContractExpectationKey) -> ExternalContractCompatibilityExpectation | None
```

- `ExternalContractExpectationKey` = `(integration_id, external_operation_id, contract_ref, contract_version, tenant_id)` — typed tuple/dataclass, not a global vendor map.
- Resolvers are **injected** into composition as an explicit `tuple[...]` or small **evaluator-style registry keyed only by `resolver_id`**, not provider slug.
- **Forbidden:** modifying `IntegrationContractSpec.metadata`, second Integration Catalog, or generic core resolver that branches on vendor name.

External plugins implement resolver/evaluator/evidence provider alongside `IntegrationPlugin` registration; generic core stays vendor-agnostic.

---

## 13. Observed evidence model (Q3)

Base immutable record: **`ExternalContractCompatibilityEvidence`**.

| Field | Purpose |
| ----- | ------- |
| `evidence_id` | Stable fact id |
| `scope` | Same shape as expectation scope |
| `dimension` | `ExternalContractCompatibilityDimension` |
| `observed_at` | `datetime` (timezone-aware, from collector — not hidden service clock) |
| `contract_ref` / `contract_version` | Observed contract pin |
| `authority` | `ExternalContractEvidenceAuthority` (§14) |
| `evidence_refs` | `tuple[str, ...]` — redacted handles, fingerprints, invariant result refs |
| `fact` | Dimension-specific **sealed** fact type (no raw body); see below |

**Dimension fact carriers (typed, no `Any`):**

| Dimension | Example fact fields (illustrative) |
| --------- | ----------------------------------- |
| SCHEMA | `schema_fingerprint`, `field_presence_results`, `enum_mismatch_codes`, `shape_delta_ref` |
| PROTOCOL | `protocol_version`, `method`, `content_type`, `status_semantics_code`, `header_invariant_results` |
| SEMANTIC | `tuple[ExternalContractSemanticAssertionResult, ...]` |

**`ExternalContractSemanticAssertionResult` (Q12 building block):**

| Field | Type |
| ----- | ---- |
| `assertion_id` | `str` |
| `status` | `SemanticAssertionStatus` = `PASS` \| `FAIL` \| `UNKNOWN` |
| `authority` | Authoritative class only for verdict materiality |
| `evidence_refs` | `tuple[str, ...]` |

---

## 14. Evidence authority (Q4)

```text
ExternalContractEvidenceAuthority.PROVIDER_ADAPTER
ExternalContractEvidenceAuthority.APPLICATION_INVARIANT
ExternalContractEvidenceAuthority.CONTRACT_SPECIFICATION
ExternalContractEvidenceAuthority.LLM_ADVISORY
```

| Class | May materially support dimension finding? |
| ----- | ---------------------------------------- |
| `PROVIDER_ADAPTER` | Yes |
| `APPLICATION_INVARIANT` | Yes |
| `CONTRACT_SPECIFICATION` | Yes |
| `LLM_ADVISORY` | **No** |

**Hard invariant:** `LLM_ADVISORY` may attach hypotheses, explanation candidates, or “needs more evidence” hints via separate advisory attachment — it **cannot** be the sole basis for `COMPATIBLE` or any `*_INCOMPATIBLE` dimension finding.

If only LLM (or other non-authoritative) evidence exists for a required dimension → top-level `INSUFFICIENT_EVIDENCE`.

---

## 15. Raw payload policy

Evidence contracts **must not** store: raw HTTP body, tokens, credentials, PII payloads, arbitrary request/response blobs.

Allowed: redacted evidence refs, schema/protocol fingerprints, sanitized status codes, typed invariant results, provider version refs.

---

## 16. Dimension-level finding (Q5)

**`ExternalContractCompatibilityFinding`**

| Field | Type |
| ----- | ---- |
| `dimension` | `ExternalContractCompatibilityDimension` |
| `status` | `DimensionCompatibilityStatus` = `COMPATIBLE` \| `INCOMPATIBLE` \| `INSUFFICIENT_EVIDENCE` |
| `reason_code` | `ExternalContractCompatibilityReasonCode` (subset) |
| `evidence_refs` | `tuple[str, ...]` |
| `evaluator_id` | `str` |
| `source_authority` | `ExternalContractEvidenceAuthority` |

No `bool` compatibility flag.

---

## 17. Assessment result contract

**`ExternalContractCompatibilityAssessment`** (frozen)

| Field | Purpose |
| ----- | ------- |
| `assessment_id` | Correlation id |
| `expectation_id` | From expectation |
| `scope` | Copy for observability |
| `outcome` | `ExternalContractCompatibilityOutcome` |
| `findings` | `tuple[ExternalContractCompatibilityFinding, ...]` |
| `reason_code` | Primary `ExternalContractCompatibilityReasonCode` |
| `evidence_refs` | Aggregated refs used |
| `assessed_at` | From request (explicit; service does not call wall clock) |

---

## 18. Reason codes (P0 minimum)

```text
NONE
SCHEMA_MISMATCH
PROTOCOL_MISMATCH
SEMANTIC_MISMATCH
MISSING_REQUIRED_EVIDENCE
EVIDENCE_CONFLICT
STALE_EVIDENCE
IDENTITY_MISMATCH
UNSUPPORTED_EVALUATOR
EVALUATOR_AMBIGUITY
```

Free-text is diagnostic only; primary semantics = `reason_code` + `outcome`.

---

## 19. Deterministic aggregation (Q6)

**Input:** `ExternalContractCompatibilityAssessmentRequest`

- `expectation: ExternalContractCompatibilityExpectation`
- `evidence: tuple[ExternalContractCompatibilityEvidence, ...]`
- `assessed_at: datetime`
- `freshness_policy: ExternalContractCompatibilityEvidencePolicy`
- `evaluators: tuple[ExternalContractCompatibilityEvaluator, ...]` (injected; see §21)

**Ordered algorithm (exact precedence: SCHEMA → PROTOCOL → SEMANTIC):**

```text
1. If any evidence.scope conflicts with expectation.scope on required identity fields
     → outcome INSUFFICIENT_EVIDENCE, reason IDENTITY_MISMATCH

2. Filter evidence to matching scope; apply freshness_policy(evidence, assessed_at, request.window)
     → stale required-dimension evidence treated as absent (reason STALE_EVIDENCE when relevant)

3. For each required dimension, select evaluator(s) (§21); if ambiguous → INSUFFICIENT_EVIDENCE / EVALUATOR_AMBIGUITY

4. Run selected evaluator → dimension finding(s); detect authoritative conflict (§20)

5. If any required dimension has authoritative INCOMPATIBLE for SCHEMA
     → SCHEMA_INCOMPATIBLE (reason SCHEMA_MISMATCH)

6. Else if any required dimension has authoritative INCOMPATIBLE for PROTOCOL
     → PROTOCOL_INCOMPATIBLE (reason PROTOCOL_MISMATCH)

7. Else if any required dimension has authoritative INCOMPATIBLE for SEMANTIC
     → SEMANTIC_INCOMPATIBLE (reason SEMANTIC_MISMATCH)

8. Else if any required dimension lacks sufficient authoritative evidence
     or has INSUFFICIENT_EVIDENCE finding
     → INSUFFICIENT_EVIDENCE (reason MISSING_REQUIRED_EVIDENCE or EVIDENCE_CONFLICT)

9. Else all required dimensions authoritative COMPATIBLE
     → COMPATIBLE (reason NONE)
```

**Why order matters:** schema/protocol failure can invalidate semantic interpretation; semantic failure after schema/protocol pass is still **`SEMANTIC_INCOMPATIBLE`** (variant H). Steps 5–7 only fire on **authoritative** `INCOMPATIBLE` for that dimension — not on missing schema blocking semantic (missing → step 8).

No weighted scores; no confidence thresholds.

---

## 20. Conflicting evidence (Q7)

Within the same `(scope, dimension, assessment window)`:

```text
authoritative COMPATIBLE + authoritative INCOMPATIBLE
  → dimension treated as INSUFFICIENT_EVIDENCE
  → top-level INSUFFICIENT_EVIDENCE, reason EVIDENCE_CONFLICT
```

Forbidden: newest-wins silent, majority vote, LLM tie-break.

Optional **explicit supersession policy** (P2+ only): separate typed `ExternalContractEvidenceSupersessionPolicy` injected on request — not default; without it, conflicts fail closed.

---

## 21. Freshness (Q8)

**No hidden clock** in the evaluator: `assessed_at` is **required on the request**; collectors stamp `observed_at` on evidence.

```text
ExternalContractCompatibilityEvidencePolicy (Protocol)
  def accepts(
      evidence: ExternalContractCompatibilityEvidence,
      assessed_at: datetime,
      window: ExternalContractAssessmentWindow,
  ) -> bool
```

- `ExternalContractAssessmentWindow` = explicit `(valid_from, valid_until) | max_age: timedelta | evidence_ttl_ref` — supplied by **caller/host**, not hardcoded in core (no baked-in 5m/1h/24h).
- Stale required-dimension evidence → treated as missing → `INSUFFICIENT_EVIDENCE` / `STALE_EVIDENCE`.

Tests pass fixed `assessed_at` and window.

---

## 22. Evaluator SPI (Q10)

```text
ExternalContractCompatibilityEvaluator (Protocol)
  evaluator_id: str
  supported_dimensions: frozenset[ExternalContractCompatibilityDimension]
  can_evaluate(expectation, evidence_bundle) -> bool
  evaluate(
      expectation: ExternalContractCompatibilityExpectation,
      evidence: tuple[ExternalContractCompatibilityEvidence, ...],
      context: ExternalContractCompatibilityEvaluationContext,
  ) -> tuple[ExternalContractCompatibilityFinding, ...]
```

- Generic service depends on **Protocol only**.
- Provider/domain implementations live in provider packages or application modules; registered at composition.
- **Registry (if used):** `ExternalContractCompatibilityEvaluatorRegistry` keyed by **`evaluator_id` only** — lists evaluators, not vendors. Selection = explicit `evaluator_id` on request **or** unique `can_evaluate` match.
- **Ambiguity:** more than one evaluator claims the same material assessment without explicit `evaluator_id` → **FAIL CLOSED** (`EVALUATOR_AMBIGUITY` / `UNSUPPORTED_EVALUATOR`).
- **Forbidden in generic service:** `match provider_id`, `if category == "jira"`, central vendor map.

---

## 23. Evidence collection vs assessment (Q11)

```text
ExternalContractEvidenceProvider (Protocol)  → I/O at provider boundary
        ↓
tuple[ExternalContractCompatibilityEvidence]
        ↓
ExternalContractCompatibilityService.assess(...)  → pure classification
```

Assessment service **must not**: call provider HTTP, mutate config, retry, switch integration, invoke Tools, or create capability gaps.

Probing belongs in evidence providers invoked **before** `assess`.

---

## 24. Semantic invariants without expression language (Q12)

- Business meaning (customs value, acting principal, units, idempotency semantics) is **not** interpreted in generic Integrations core.
- Provider/domain evaluators emit **`ExternalContractSemanticAssertionResult`** (`assertion_id` + `PASS`/`FAIL`/`UNKNOWN` + authoritative source + refs).
- Generic service **aggregates** semantic dimension: any authoritative `FAIL` → semantic `INCOMPATIBLE`; all required assertions authoritative `PASS` → semantic `COMPATIBLE`; any required `UNKNOWN` or missing assertion coverage → `INSUFFICIENT_EVIDENCE`.
- No arbitrary JSON logic, no stored Python callables in public models, no LLM-only semantic pass.

---

## 25. Consumer boundary to Autonomous Work

AW/recovery may consume `ExternalContractCompatibilityAssessment` as **input facts** when choosing configure / adapt / acquire / authority escalation. GAP-04 does **not** emit AW dispositions, does not route `AUTHORITY_CHANGE_REQUIRED`, and does not declare true capability gaps.

Authority observations (e.g. wrong OAuth scope) may appear as **protocol/semantic incompatible facts**; granting scope remains governance/recovery.

---

## 26. Observability

Assessment exposes stable ids: `assessment_id`, `expectation_id`, `outcome`, `reason_code`, `evidence_refs`, `evaluator_id`s. Callers project into existing `TraceEvent` / diagnostics / proof evidence — **no** new event bus.

---

## 27. External plugin parity

Same contracts and SPI paths for built-in and external `IntegrationPlugin` packages:

- optional `ExternalContractCompatibilityExpectationResolver`
- `ExternalContractEvidenceProvider`
- `ExternalContractCompatibilityEvaluator`

No edit to generic platform core per vendor.

---

## 28. Failure model

| Condition | Outcome |
| --------- | ------- |
| Identity mismatch | `INSUFFICIENT_EVIDENCE` / `IDENTITY_MISMATCH` |
| Missing/stale required evidence | `INSUFFICIENT_EVIDENCE` |
| Authoritative conflict | `INSUFFICIENT_EVIDENCE` / `EVIDENCE_CONFLICT` |
| LLM-only support | `INSUFFICIENT_EVIDENCE` |
| Evaluator ambiguity | `INSUFFICIENT_EVIDENCE` / `EVALUATOR_AMBIGUITY` |
| Schema authoritative fail | `SCHEMA_INCOMPATIBLE` |
| Protocol authoritative fail (schema pass) | `PROTOCOL_INCOMPATIBLE` |
| Semantic authoritative fail (schema+protocol pass) | `SEMANTIC_INCOMPATIBLE` |

Never infer contract version from error strings, operation id from URL, or semantic rules from field names — insufficient typed facts → `INSUFFICIENT_EVIDENCE`.

---

## 29. Frozen surfaces — NO semantic change

| Surface | P0/P1 |
| ------- | ----- |
| `PlatformIntegrationContract` | **Unchanged** |
| `IntegrationContractSpec` | **Unchanged** |
| Integration Catalog authority | **Unchanged** |
| UCA / EE contracts | **Unchanged** |

Compatibility is an **additive** Integrations extension under `intergrax/integrations/contracts/` + `intergrax/integrations/`.

**Dependency direction:** public contracts → `intergrax/integrations/contracts/`; service implementation → `intergrax/integrations/` (not runtime-only hiding of public semantics).

---

## 30. P1 proposed implementation scope

| Artifact | Path |
| -------- | ---- |
| Public frozen types, enums, Protocols | `intergrax/integrations/contracts/external_contract_compatibility.py` |
| Pure assessment service | `intergrax/integrations/external_contract_compatibility_service.py` |
| Evaluator list / ambiguity guard (only if needed) | `intergrax/integrations/external_contract_compatibility_evaluator_registry.py` |

**P1 gates (enterprise static):** no `Any`, no `dict[str, Any]`, no reflection dispatch, no vendor branches in service, no second catalog, no LLM-only verdict, no provider I/O in service, deterministic tests with injected `assessed_at` / policy.

**P1 test modules (planned):** `tests/unit/integrations/test_external_contract_compatibility_service.py` (and contract shape tests adjacent to existing integrations unit layout).

---

## 31. P2 proposed scope

- Provider `ExternalContractEvidenceProvider` + schema/protocol/semantic evaluators in provider packages
- Application domain semantic extension evaluators
- Composition registration wiring (alongside existing plugin registration)
- Reference evaluator patterns — **not** Asterion-specific core branches or scenario hardcodes

---

## 32. Future test plan (certification gates)

| Case | Expected outcome |
| ---- | ---------------- |
| schema+protocol+semantic PASS | `COMPATIBLE` |
| schema FAIL | `SCHEMA_INCOMPATIBLE` |
| schema PASS, protocol FAIL | `PROTOCOL_INCOMPATIBLE` |
| variant H: HTTP/deserialize OK, semantic FAIL | `SEMANTIC_INCOMPATIBLE` |
| semantic UNKNOWN | `INSUFFICIENT_EVIDENCE` |
| LLM-only compatible claim | `INSUFFICIENT_EVIDENCE` |
| authoritative conflict | `INSUFFICIENT_EVIDENCE` / `EVIDENCE_CONFLICT` |
| wrong tenant/provider/operation | never `COMPATIBLE` |
| external plugin evaluator via Protocol | passes without subclassing generic service |

---

## 33. Classification

```text
CLASS A — additive Integrations contracts + pure service
UCA REOPEN = NO
EE REOPEN = NO
```

---

## 34. Stop conditions (not triggered in P0)

P0 design does **not** require: UCA/EE semantic changes, capability gap minting, recovery ownership, second catalog, metadata-only semantic model, generic expression framework, LLM-only semantic verdict, or provider I/O inside assessment service.

---

## 35. Q1–Q12 closure table

| Q | Verdict | Exact solution | Owner | Planned contract / file |
| - | ------- | -------------- | ----- | ------------------------ |
| Q1 | **LOCKED** | `ExternalContractCompatibilityExpectation` + `ExternalContractCompatibilityScope` fields in §10–11 | Integrations (types); provider/app supply values | `intergrax/integrations/contracts/external_contract_compatibility.py` |
| Q2 | **LOCKED** | Primary: expectation on request; optional `ExternalContractCompatibilityExpectationResolver` injected at composition; no Catalog/metadata | Composition / provider package | Same + host wiring |
| Q3 | **LOCKED** | `ExternalContractCompatibilityEvidence` + sealed dimension facts + semantic assertion results §13 | Provider boundary collectors | Same contracts module |
| Q4 | **LOCKED** | Four-way `ExternalContractEvidenceAuthority`; LLM non-authoritative §14 | Integrations aggregation rules | Service + contracts |
| Q5 | **LOCKED** | `ExternalContractCompatibilityFinding` + `DimensionCompatibilityStatus` §16 | Integrations | Contracts |
| Q6 | **LOCKED** | Nine-step algorithm §19 (SCHEMA → PROTOCOL → SEMANTIC precedence) | Integrations service | `external_contract_compatibility_service.py` |
| Q7 | **LOCKED** | Authoritative conflict → `INSUFFICIENT_EVIDENCE` / `EVIDENCE_CONFLICT` §20 | Integrations | Service |
| Q8 | **LOCKED** | Request `assessed_at` + `ExternalContractCompatibilityEvidencePolicy` + caller window §21 | Caller/host policy | Contracts Protocol |
| Q9 | **LOCKED** | Scope fields §10; `tenant_id` + `integration_id` + operation + contract pin + optional execution ids | Integrations | Contracts |
| Q10 | **LOCKED** | `ExternalContractCompatibilityEvaluator` Protocol; optional id-keyed registry; ambiguity fail-closed §22 | Provider/domain plugins | Contracts + optional registry |
| Q11 | **LOCKED** | `ExternalContractEvidenceProvider` vs pure `assess` §23 | Integrations / provider | Contracts + service |
| Q12 | **LOCKED** | `ExternalContractSemanticAssertionResult` (`assertion_id`, PASS/FAIL/UNKNOWN); core aggregates only §24 | Provider + application evaluators | Contracts |

---

## 36. Blockers

```text
0
```

P0 architecture lock **PASS** pending independent audit of the committing revision.

---

> **Independent audit reminder:** Wprowadzony architecture lock S24-GAP-04-P0 musi zostać niezależnie zaudytowany na podstawie rzeczywistego commitu i aktualnego kodu przechowywanego na GitHub. Raport Cursor AI nie stanowi samodzielnej zgody na rozpoczęcie implementacji S24-GAP-04-P1.
