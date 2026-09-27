# S24-GAP-04-P0-R1 — External Contract Semantic Compatibility Architecture Lock

## 1. Metadata and baseline

| Field | Value |
| ----- | ----- |
| **Task** | `S24-GAP-04-P0-R1` (identity & determinism closure on P0 lock) |
| **Pre-audit baseline** | `e0069dd3de52bca7106bc6dc6602163a8506ba73` |
| **Session start HEAD** | `e0069dd3de52bca7106bc6dc6602163a8506ba73` (`development` = `origin/development`) |
| **Relevant-path drift** | **None** (`intergrax/integrations/**`, `intergrax/runtime/integrations/**`, `intergrax/contracts/**`, `platform_proofs/scenarios/external_api_schema_drift/**`, `docs/project/architecture/INTEGRATIONS.md`) |
| **Branch** | `development` |
| **Artifact role** | Closed-world design record before `S24-GAP-04-P1` |
| **Production / tests / scenario spec** | **0 mutation in P0-R1** |
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

## 10. Assessment subject identity (`ExternalContractCompatibilitySubject`)

Immutable subject carried on **expectation**, **evidence**, and **assessment**. Identifies **what is being assessed** — not which contract version is expected or observed.

| Field | Type | Rule |
| ----- | ---- | ---- |
| `tenant_id` | `str` (non-empty) | Aligns with `ExecutionScopeIdentity.tenant_id` / platform envelope conventions |
| `integration_id` | `str` | Canonical `{provider_id}:{integration_kind}` |
| `provider_id` | `str` | Must match parsed identity of `integration_id` |
| `integration_kind` | `str` | Category slug; must match `integration_id` |
| `external_operation_id` | `str` | Provider-owned stable operation key (not URL guess) |
| `host_binding_ref` | `str \| None` | Optional Tier-3 host binding discriminator when multiple profiles share tenant |
| `execution_task_id` | `TaskId \| None` | Optional; when assessment is execution-scoped |
| `execution_run_id` | `RunId \| None` | Optional |
| `execution_id` | `ExecutionId \| None` | Optional |

**Not in subject (frozen):** `contract_ref`, `contract_version`, schema/protocol fingerprints — those belong to **expected** or **observed** contract pins (§10a).

**Identity continuity:** evidence `subject` must **exactly** match expectation `subject` on all non-optional fields and on optional execution fields when expectation sets them. Mismatch on tenant, provider, integration, `integration_kind`, `external_operation_id`, `host_binding_ref`, or explicit execution scope → `INSUFFICIENT_EVIDENCE` / `IDENTITY_MISMATCH` (see §19). **Expected contract version ≠ observed contract version is not identity mismatch** — it is normal assessment material for evaluators.

### 10a. Contract pin (`ExternalContractPin`)

Immutable pin for **which contract revision** is expected or observed (separate from subject).

| Field | Type |
| ----- | ---- |
| `contract_ref` | `str` | Typed contract identity (URI or stable ref id — no raw OpenAPI blob) |
| `contract_version` | `str` | Version label for that ref |

Schema/protocol technical refs (fingerprints, spec artifact ids) live on **expectation** or inside **dimension facts** — not on subject identity.

### 10b. Architecture examples (identity vs version)

**Same subject, different contract version (valid assessment input):**

```text
subject = freight-x / submit_manifest  (tenant + integration + operation aligned)

expected_contract = { ref: freight-manifest-api, version: v1 }
observed_contract = { ref: freight-manifest-api, version: v2 }

→ VALID assessment input
→ evaluators decide COMPATIBLE vs *_INCOMPATIBLE vs INSUFFICIENT_EVIDENCE
→ never IDENTITY_MISMATCH solely due to version delta
```

**Different provider (identity mismatch):**

```text
expectation.subject.provider_id = freight-x
evidence.subject.provider_id       = freight-y

→ IDENTITY_MISMATCH → INSUFFICIENT_EVIDENCE
```

**Different operation (identity mismatch):**

```text
external_operation_id = submit_manifest  vs  cancel_manifest

→ IDENTITY_MISMATCH → INSUFFICIENT_EVIDENCE
```

**Observed version unknown (not identity mismatch):**

```text
evidence.subject matches expectation.subject
observed_contract = None  (provider cannot authoritatively pin version)

→ not IDENTITY_MISMATCH
→ evaluator / missing version facts → INSUFFICIENT_EVIDENCE when version proof required
```

---

## 11. Expected contract model (Q1)

Frozen type: **`ExternalContractCompatibilityExpectation`**.

| Field | Purpose |
| ----- | ------- |
| `expectation_id` | Stable id for assessment correlation |
| `subject` | `ExternalContractCompatibilitySubject` — **what** is assessed |
| `expected_contract` | `ExternalContractPin` — **which revision** is expected (single canonical pin; no duplicate `contract_ref`/`contract_version` elsewhere on expectation) |
| `required_dimensions` | `frozenset[ExternalContractCompatibilityDimension]` — **must not be empty**; caller declares which dimensions the sequential gate applies (no automatic SCHEMA+PROTOCOL injection when only SEMANTIC is listed — Scenario #24 callers set all three explicitly when needed) |
| `schema_expectation_ref` | Provider-owned ref to schema spec artifact (fingerprint id, not raw document) |
| `protocol_expectation_ref` | Provider-owned ref to protocol spec artifact (version/method/content-type invariants, not raw document) |
| `semantic_expectation_refs` | `tuple[str, ...]` — ids of registered semantic assertion specs (not rule bodies in core) |
| `domain_extension_ref` | `str \| None` — pointer to application-supplied typed semantic extension contract (when business invariants apply) |

**Does not store:** raw request/response payload, secrets, arbitrary JSON schema bodies, or duplicate contract pins on `subject`.

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

- `ExternalContractExpectationKey` = subject fields + `expected_contract: ExternalContractPin` + `tenant_id` — typed dataclass, not a global vendor map. Contract pin is expectation lookup material, not subject-equality material for evidence pairing.
- Resolvers are **injected** into composition as an explicit `tuple[...]` or small **evaluator-style registry keyed only by `resolver_id`**, not provider slug.
- **Forbidden:** modifying `IntegrationContractSpec.metadata`, second Integration Catalog, or generic core resolver that branches on vendor name.

External plugins implement resolver/evaluator/evidence provider alongside `IntegrationPlugin` registration; generic core stays vendor-agnostic.

---

## 13. Observed evidence model (Q3)

Base immutable record: **`ExternalContractCompatibilityEvidence`**.

| Field | Purpose |
| ----- | ------- |
| `evidence_id` | Stable fact id |
| `subject` | `ExternalContractCompatibilitySubject` — must match expectation `subject` for material evidence (§10) |
| `observed_contract` | `ExternalContractPin \| None` — observed revision when provider can pin; `None` when unknown (not identity mismatch) |
| `dimension` | `ExternalContractCompatibilityDimension` |
| `observed_at` | `datetime` (timezone-aware, from collector — not hidden service clock) |
| `authority` | `ExternalContractEvidenceAuthority` (§14) |
| `evidence_refs` | `tuple[str, ...]` — redacted handles, fingerprints, invariant result refs |
| `fact` | **Closed discriminated union** (exact carriers below); contract validation rejects dimension/fact mismatch |

**`fact` union (frozen public names, no `Any`, no arbitrary mapping):**

```text
ExternalContractSchemaEvidenceFact
  | ExternalContractProtocolEvidenceFact
  | ExternalContractSemanticEvidenceFact
```

**Dimension ↔ fact invariant (hard):**

| `dimension` | `fact` type required |
| ----------- | -------------------- |
| `SCHEMA` | `ExternalContractSchemaEvidenceFact` |
| `PROTOCOL` | `ExternalContractProtocolEvidenceFact` |
| `SEMANTIC` | `ExternalContractSemanticEvidenceFact` |

Mismatch at construction/validation → **contract validation failure** (not runtime duck typing).

**`ExternalContractSchemaEvidenceFact` (P1 minimum):**

| Field | Type |
| ----- | ---- |
| `schema_ref` | `str \| None` |
| `schema_fingerprint` | `str \| None` |
| `validation_status` | `SchemaValidationStatus` = `PASS` \| `FAIL` \| `UNKNOWN` |
| `violation_codes` | `tuple[str, ...]` |
| `shape_delta_ref` | `str \| None` |

**`ExternalContractProtocolEvidenceFact` (P1 minimum):**

| Field | Type |
| ----- | ---- |
| `protocol_ref` | `str \| None` |
| `protocol_version` | `str \| None` |
| `method` | `str \| None` |
| `content_type` | `str \| None` |
| `validation_status` | `ProtocolValidationStatus` = `PASS` \| `FAIL` \| `UNKNOWN` |
| `violation_codes` | `tuple[str, ...]` |
| `header_invariant_results` | `tuple[ExternalContractProtocolHeaderInvariantResult, ...]` |

(`ExternalContractProtocolHeaderInvariantResult` = typed invariant id + `PASS`/`FAIL`/`UNKNOWN` — not a headers dict.)

**`ExternalContractSemanticEvidenceFact` (P1 minimum):**

| Field | Type |
| ----- | ---- |
| `assertions` | `tuple[ExternalContractSemanticAssertionResult, ...]` |

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
| `assessment_id` | **Caller-supplied** immutable id from request (no hidden `uuid4()` inside pure service — reproducibility) |
| `expectation_id` | From expectation |
| `subject` | Copy for observability |
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

## 19. Assessment request (data only) and service composition

**Frozen request:** **`ExternalContractCompatibilityAssessmentRequest`** — facts only; no callables, registries, policies, or evaluator instances.

| Field | Purpose |
| ----- | ------- |
| `assessment_id` | Caller-supplied immutable correlation id |
| `expectation` | `ExternalContractCompatibilityExpectation` |
| `evidence` | `tuple[ExternalContractCompatibilityEvidence, ...]` |
| `assessed_at` | Explicit assessment timestamp (service does not call wall clock) |
| `assessment_window` | `ExternalContractAssessmentWindow` — caller-supplied window data for freshness (§21) |
| `explicit_evaluator_ids` | `tuple[str, ...]` = `()` — optional evaluator **identifiers** only (§22) |

**Forbidden on request:** `evaluators`, evaluator registry objects, `ExternalContractCompatibilityEvidencePolicy` implementation, resolvers, evidence providers, callables, or any service/locator bag.

**Service composition (P1):** **`ExternalContractCompatibilityService`** receives dependencies via constructor:

```text
evaluators: tuple[ExternalContractCompatibilityEvaluator, ...]
evidence_policy: ExternalContractCompatibilityEvidencePolicy
```

Request = data. Service = behavior. Host/composition root wires policy implementation when building the service — not per-request injection.

External plugins inject evaluator / evidence provider / expectation resolver through the same composition/plugin extension path as built-in Integrations — **not** via `request.evaluators = (...)`.

---

## 19a. Deterministic aggregation (Q6)

**Per-dimension normalization (before sequential gate):** for each required dimension, evaluators produce findings; authoritative conflict within a dimension (§20) normalizes that dimension to **`INSUFFICIENT_EVIDENCE`** with `reason_code = EVIDENCE_CONFLICT` — never simultaneous decisive `COMPATIBLE` and `INCOMPATIBLE` for the same dimension.

Each required dimension ends as exactly one of: `COMPATIBLE` \| `INCOMPATIBLE` \| `INSUFFICIENT_EVIDENCE`.

**Final ordered algorithm** (canonical dimension order: `SCHEMA` → `PROTOCOL` → `SEMANTIC`; only dimensions in `required_dimensions` participate):

```text
validate evidence.subject vs expectation.subject
  → mismatch on tenant | provider | integration | integration_kind |
     external_operation_id | host_binding_ref | explicit execution scope
  → INSUFFICIENT_EVIDENCE / IDENTITY_MISMATCH

filter evidence to matching subject
apply service.evidence_policy(evidence, assessed_at, request.assessment_window)
  → stale required-dimension evidence treated as absent (STALE_EVIDENCE when relevant)

resolve evaluator per required dimension (§22)
  → ambiguity → INSUFFICIENT_EVIDENCE / EVALUATOR_AMBIGUITY

evaluate + normalize per-dimension conflicts (§20)

IF SCHEMA in required_dimensions:
    if SCHEMA == INCOMPATIBLE → SCHEMA_INCOMPATIBLE / SCHEMA_MISMATCH
    if SCHEMA == INSUFFICIENT_EVIDENCE → INSUFFICIENT_EVIDENCE
    else continue

IF PROTOCOL in required_dimensions:
    if PROTOCOL == INCOMPATIBLE → PROTOCOL_INCOMPATIBLE / PROTOCOL_MISMATCH
    if PROTOCOL == INSUFFICIENT_EVIDENCE → INSUFFICIENT_EVIDENCE
    else continue

IF SEMANTIC in required_dimensions:
    if SEMANTIC == INCOMPATIBLE → SEMANTIC_INCOMPATIBLE / SEMANTIC_MISMATCH
    if SEMANTIC == INSUFFICIENT_EVIDENCE → INSUFFICIENT_EVIDENCE
    else continue

all required dimensions COMPATIBLE → COMPATIBLE / NONE
```

**Precedence tests (locked):**

| SCHEMA | PROTOCOL | SEMANTIC | Outcome |
| ------ | -------- | -------- | ------- |
| `INSUFFICIENT_EVIDENCE` | (any) | `INCOMPATIBLE` | `INSUFFICIENT_EVIDENCE` (not semantic incompatible) |
| `COMPATIBLE` | `INSUFFICIENT_EVIDENCE` | `INCOMPATIBLE` | `INSUFFICIENT_EVIDENCE` |
| `COMPATIBLE` | `COMPATIBLE` | `INCOMPATIBLE` | `SEMANTIC_INCOMPATIBLE` |
| `INCOMPATIBLE` | `INCOMPATIBLE` | (any) | `SCHEMA_INCOMPATIBLE` |
| `COMPATIBLE` | `INCOMPATIBLE` | `INCOMPATIBLE` | `PROTOCOL_INCOMPATIBLE` |

**Conflict example (locked):** authoritative SCHEMA compatible + authoritative SCHEMA incompatible → dimension SCHEMA = `INSUFFICIENT_EVIDENCE` / `EVIDENCE_CONFLICT` → top-level **`INSUFFICIENT_EVIDENCE`**, not `SCHEMA_INCOMPATIBLE`.

No weighted scores; no confidence thresholds.

---

## 20. Conflicting evidence (Q7)

Within the same `(subject, dimension, assessment window)`:

```text
authoritative COMPATIBLE + authoritative INCOMPATIBLE
  → dimension treated as INSUFFICIENT_EVIDENCE
  → top-level INSUFFICIENT_EVIDENCE, reason EVIDENCE_CONFLICT
```

Forbidden: newest-wins silent, majority vote, LLM tie-break.

Optional **explicit supersession policy** (P2+ only): separate typed `ExternalContractEvidenceSupersessionPolicy` injected at **service composition** — not on assessment request; without it, conflicts fail closed.

---

## 21. Freshness (Q8)

**No hidden clock** in the evaluator: `assessed_at` and `assessment_window` are **request data**; collectors stamp `observed_at` on evidence.

**Request data:**

- `assessed_at: datetime`
- `assessment_window: ExternalContractAssessmentWindow` — explicit `(valid_from, valid_until) | max_age: timedelta | evidence_ttl_ref` from caller/host, not hardcoded in core (no baked-in 5m/1h/24h).

**Service dependency (not on request):**

```text
ExternalContractCompatibilityEvidencePolicy (Protocol)
  def accepts(
      evidence: ExternalContractCompatibilityEvidence,
      assessed_at: datetime,
      window: ExternalContractAssessmentWindow,
  ) -> bool
```

Injected when constructing `ExternalContractCompatibilityService`. Stale required-dimension evidence → treated as missing → `INSUFFICIENT_EVIDENCE` / `STALE_EVIDENCE`.

Tests pass fixed `assessed_at`, window, and inject a concrete policy on the service.

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
- **Registry (if used):** `ExternalContractCompatibilityEvaluatorRegistry` keyed by **`evaluator_id` only** — lists evaluators held by the service, not vendors.
- **Selection:** `request.explicit_evaluator_ids` (identifiers only) **or**, when empty, service `evaluators` list → unique `can_evaluate` match per required dimension.
- **Ambiguity:** more than one service-registered evaluator claims the same material assessment without a matching explicit id → **FAIL CLOSED** (`EVALUATOR_AMBIGUITY` / `UNSUPPORTED_EVALUATOR`).
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
| Subject identity mismatch (not contract version delta) | `INSUFFICIENT_EVIDENCE` / `IDENTITY_MISMATCH` |
| Expected v1 vs observed v2, same subject | Valid input; evaluator decides — never `IDENTITY_MISMATCH` |
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

**P1 gates (enterprise static):** no `Any`, no `dict[str, Any]`, no reflection dispatch, no vendor branches in service, no second catalog, no LLM-only verdict, no provider I/O in service, frozen request without executable dependencies, deterministic tests with caller `assessment_id`, injected `assessed_at` / window on request, policy on service ctor.

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
| SCHEMA insufficient + SEMANTIC incompatible | `INSUFFICIENT_EVIDENCE` (gate stops at SCHEMA) |
| SCHEMA+PROTOCOL pass, SEMANTIC incompatible | `SEMANTIC_INCOMPATIBLE` |
| SCHEMA+PROTOCOL incompatible + SEMANTIC incompatible | `SCHEMA_INCOMPATIBLE` |
| SCHEMA pass, PROTOCOL insufficient, SEMANTIC incompatible | `INSUFFICIENT_EVIDENCE` |
| semantic UNKNOWN | `INSUFFICIENT_EVIDENCE` |
| LLM-only compatible claim | `INSUFFICIENT_EVIDENCE` |
| authoritative conflict on dimension | dimension `INSUFFICIENT_EVIDENCE` / top-level `EVIDENCE_CONFLICT` |
| wrong tenant/provider/operation (subject) | `IDENTITY_MISMATCH`; never `COMPATIBLE` |
| same subject, expected v1 / observed v2 | valid assessment; not `IDENTITY_MISMATCH` |
| external plugin evaluator via composition | passes without subclassing generic service |

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
| Q1 | **LOCKED** | `ExternalContractCompatibilitySubject` + `ExternalContractPin` + `ExternalContractCompatibilityExpectation` (`subject`, `expected_contract`, non-empty `required_dimensions`) §10–11 | Integrations (types); provider/app supply values | `intergrax/integrations/contracts/external_contract_compatibility.py` |
| Q2 | **LOCKED** | Primary: expectation on request; optional `ExternalContractCompatibilityExpectationResolver` at composition; key = subject + expected pin; no Catalog/metadata | Composition / provider package | Same + host wiring |
| Q3 | **LOCKED** | `ExternalContractCompatibilityEvidence` with `subject`, `observed_contract`, closed `fact` union + exact schema/protocol/semantic fact types §13 | Provider boundary collectors | Same contracts module |
| Q4 | **LOCKED** | Four-way `ExternalContractEvidenceAuthority`; LLM non-authoritative §14 | Integrations aggregation rules | Service + contracts |
| Q5 | **LOCKED** | `ExternalContractCompatibilityFinding` + `DimensionCompatibilityStatus` §16 | Integrations | Contracts |
| Q6 | **LOCKED** | Sequential dimension gate §19a (SCHEMA → PROTOCOL → SEMANTIC); insufficient stops before later incompatible | Integrations service | `external_contract_compatibility_service.py` |
| Q7 | **LOCKED** | Authoritative conflict normalizes dimension to `INSUFFICIENT_EVIDENCE` / `EVIDENCE_CONFLICT` §20 | Integrations | Service |
| Q8 | **LOCKED** | Request: `assessed_at` + `assessment_window`; policy: service dependency `ExternalContractCompatibilityEvidencePolicy` §21 | Caller data + host-composed policy | Contracts Protocol + service ctor |
| Q9 | **LOCKED** | Subject identity without contract version §10; `IDENTITY_MISMATCH` only on subject fields | Integrations | Contracts |
| Q10 | **LOCKED** | Evaluators on service ctor; request `explicit_evaluator_ids` only; ambiguity fail-closed §22 | Provider/domain plugins | Contracts + optional registry |
| Q11 | **LOCKED** | `ExternalContractEvidenceProvider` vs pure `assess`; request data-only §19, §23 | Integrations / provider | Contracts + service |
| Q12 | **LOCKED** | `ExternalContractSemanticEvidenceFact.assertions` + `ExternalContractSemanticAssertionResult`; dimension/fact invariant §13, §24 | Provider + application evaluators | Contracts |

---

## 36. Blockers

```text
0
```

P0-R1 architecture lock **PASS** pending independent audit of the committing revision.

---

> **Independent audit reminder:** Wprowadzona korekta S24-GAP-04-P0-R1 musi zostać niezależnie zaudytowana na podstawie rzeczywistego commitu i aktualnego kodu na GitHub. Raport Cursor AI nie stanowi samodzielnej zgody na rozpoczęcie S24-GAP-04-P1.
