# Hybrid Ask — unified evidence, query orchestration and read-only live execution

**Status:** READY_FOR_REVIEW  
**Task:** LKW-HYBRID-ASK-ARCH-1-REVIEW-FIX-2-CROSS-VERSION-RUN-ACCESS-DECISION
**Classification:** docs-only architecture and implementation contract

**Canonical references:** [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md) · [`KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md`](KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md) · [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md) · [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md)

---

## 1. One-sentence outcome

One workspace question can combine indexed RAG evidence with authorized read-only live provider evidence in a single grounded response with unified provenance, strict policy enforcement and no automatic persistence of live result bodies.

---

## 2. Current-state truth

### 2.1 Implemented Ask pipeline (indexed RAG only)

```text
WorkspaceAskService
→ workspace authorization (ManagedWorkspaceService)
→ local.workspace.search (LocalWorkspaceTaskExecutor)
→ map_search_hits → WorkspaceSearchHitV1
→ AskAnswerAssembler (one bounded LLM call)
→ project_ask_citations (document/source citations)
→ durable WorkspaceAskRun (DocumentStore via WorkspaceAskRepository)
```

**Facts from the repository today:**

| Area | Current state |
|------|---------------|
| Ask mode | **Indexed RAG only** — `WorkspaceAskService` always invokes `local.workspace.search`; no live capability execution |
| Evidence model | `WorkspaceAskRun.evidence` is `list[WorkspaceSearchHitV1]` |
| Citations | `AskCitation` / `WorkspaceAskCitationV1` assume durable documents and sources (`document_id`, `source_id`, `source_path`, `file_name`) |
| Assembler input | `AskAnswerAssembler.assemble(question, evidence: list[WorkspaceSearchHitV1])` |
| Evidence indexing | Positional `E1`, `E2`, … indexes over verified hits only |
| Live Access Bindings | Durable configuration via `WorkspaceLiveAccessBinding`; `WorkspaceLiveAccessBindingService` authorizes bindings but **does not execute** them during Ask |
| Query Policy V1 | `QueryPolicyModeV1` contains only `indexed_only` and `live_only` |
| Hybrid / automatic | **Not implemented** — `hybrid` and `automatic` do not exist in models or runtime |
| Live query executor | **No production executor** — no `WorkspaceLiveCapabilityExecutorPort`, no Knowledge Query Orchestrator |
| Unified evidence / citation ABI | **Absent** — no discriminated indexed/live union |
| Hybrid Ask | **Not implemented** |

Absent Query Policy continues to behave as **indexed-only** unless a separately accepted migration states otherwise.

---

## 3. Product modes (frozen implementation sequence)

### 3.1 `indexed_only`

- Execute indexed workspace retrieval only.
- Never invoke a live capability.
- Preserve existing `WorkspaceAskService` behavior and public HTTP compatibility.
- Absent Query Policy defaults to indexed-only.

### 3.2 `live_only`

- Execute only explicitly authorized Live Access Bindings.
- Do not run indexed RAG retrieval.
- Require at least one validated live call for a completed answer.
- No arbitrary provider access; no implicit access through Connection Attachment alone.

### 3.3 `hybrid`

- Execute both indexed retrieval **and** at least one validated live capability call (V1 Hybrid Ask proof minimum).
- Normalize both result families into one evidence collection.
- Allow **one** synthesis call over that collection.
- Citations must retain indexed/live distinction.
- A failed required live call must **not** silently downgrade to indexed-only success.
- No unrestricted model autonomy over provider execution.

### 3.4 `automatic` — deferred

Orchestration may not auto-select modes in V2. `automatic` is excluded from `QueryPolicyModeV2`.

### 3.5 Query Policy V2 vocabulary

```text
QueryPolicyModeV2:
  indexed_only
  live_only
  hybrid
```

**V1 backward compatibility:** persisted `WorkspaceQueryPolicy` records with `QueryPolicyModeV1` (`indexed_only` | `live_only`) remain valid and readable. `hybrid` requires V2 mode enum and acceptance of this architecture. Migration adds V2 fields without mutating historical revision payloads in place.

### 3.6 Absent Query Policy defaults (frozen)

| Request context | Effective mode | Outcome |
|-----------------|----------------|---------|
| No persisted Query Policy + `indexed_only` V1 or V2 request | `indexed_only` | **No error** — indexed retrieval proceeds |
| No persisted Query Policy + `live_only` or `hybrid` request | — | `query_policy_required` → **HTTP 409** |

Absent Query Policy does **not** authorize live or hybrid execution.

---

## 4. Query input and authority

### 4.1 Frontend-neutral query command (conceptual)

```text
KnowledgeQueryCommandV1
├── tenant_id
├── principal_id (or authenticated-principal reference)
├── workspace_id
├── question
├── audience_context
├── requested_evidence_mode (when public contract allows)
├── configuration_revision (committed Workspace Knowledge Configuration)
└── request/run identity
```

Effective configuration is read from **one committed revision**. Child records from different revisions must not be mixed.

### 4.2 Non-authoritative frontend values

Frontend-provided values must **not** supply authoritative:

```text
provider_id
integration_kind
connection credentials
provider endpoint
capability implementation
remote authorization state
```

These are derived from committed configuration, binding metadata and registered capability contracts.

---

## 5. Audience context

Typed audience context (minimum):

```text
AudienceContextV1
├── audience: personal | shared
└── eligibility inputs for source/binding validation
```

**Rules** (canonical detail: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md) §5, §10.1):

| Rule | Enforcement |
|------|-------------|
| Personal evidence in shared Ask | **Rejected** before synthesis |
| `PERSONAL_ONLY` Indexed Sources / Live Access Bindings in shared audience | **Rejected** |
| Mixed personal/shared evidence in one run | **Rejected deterministically** |
| Prompt instructions as authorization | **Forbidden** — validation is code, not prompt |

The core orchestrator contract is HTTP- and Slack-neutral. V1 HTTP proof may use a personal audience adapter; shared audience follows Conversation Context guards.

---

## 6. Evidence Plan

### 6.1 `EvidencePlanV1` (immutable, typed)

```text
EvidencePlanV1
├── plan_id
├── tenant_id
├── workspace_id
├── configuration_revision
├── mode: indexed_only | live_only | hybrid
├── indexed_retrieval_directive (optional per mode)
├── ordered_live_call_proposals: list[LiveCallProposalV1]
├── budget_snapshot
└── audience_context
```

**Indexed retrieval directive** (when present):

```text
IndexedRetrievalDirectiveV1
├── max_results (bounded)
└── retrieval_limits aligned with Query Policy
```

**Live call proposal** (safe logical values only):

```text
LiveCallProposalV1
├── call_id
├── live_access_binding_id
├── capability_id
└── typed_capability_request (schema-validated payload)
```

### 6.2 Model may propose; validator must approve

The model may propose a typed plan. A **deterministic validator** must approve every executable directive before execution.

### 6.3 Forbidden model-controlled values

The plan must not authoritatively contain:

```text
connection_ref
provider_id
integration_kind
credential_ref
provider client
provider endpoint
arbitrary URL
raw HTTP method / headers
arbitrary Graph request
arbitrary JQL
arbitrary SQL / DAX
arbitrary MCP tool
```

These are resolved from committed configuration and registered capability descriptors.

---

## 7. Plan validation (frozen order)

```text
 1. tenant/workspace authorization
 2. committed configuration revision
 3. requested mode vs Query Policy
 4. active Workspace Connection Attachment
 5. active Live Access Binding
 6. audience eligibility
 7. Connection administrative availability
 8. capability present on binding allowlist
 9. capability present in Query Policy allowlist
10. connection present in Query Policy allowlist
11. read-only capability descriptor (runtime catalog)
12. request schema validation
13. resource-scope validation
14. live-call count budget
15. result-item and byte budgets
16. total-duration budget
```

**Executable capability set** = intersection of:

```text
active binding capabilities
∩ Query Policy capabilities
∩ runtime capability catalog (TenantLiveCapabilityCatalog)
∩ audience eligibility
```

No layer may widen another layer's authorization.

---

## 8. Live capability execution boundary (frozen)

Four separate runtime concepts must not be conflated:

### 8.1 `TenantLiveCapabilityCatalog` (descriptor catalog only)

Platform-owned descriptor catalog. **Not** an integration resolver and **not** an executable handler registry.

**Responsibilities:**

- Advertise typed capability metadata (`LiveCapabilityDescriptorV1`)
- Provide request/result schema references
- Provide read-only / effect / resource / limit metadata
- Perform **no** provider invocation

### 8.2 `TenantConnectionCapabilityReadService` (safe projections only)

Exposes safe Connection projections and descriptor listing. It does **not** resolve a runtime integration instance. Do not describe it as an integration resolver.

### 8.3 `TenantConnectionIntegrationResolverPort` (runtime integration resolution)

Narrow runtime port:

```text
resolve(
    tenant_id,
    connection_ref,
    provider_id,
    integration_kind,
) -> existing integration instance
```

**Production adapter:** `KnowledgeConnectionRegistry` (or an explicitly named adapter over it).

**Rules:**

- Return only an **already rehydrated** integration instance
- Never construct a second client
- Never read credentials
- Fail safely when the runtime integration is unavailable

Alternative resolution path: `ConnectionAwareVendorResolver` when wired to the same registry-backed instance — still **one** rehydrated integration, never a duplicate client.

### 8.4 `LiveCapabilityHandlerV1` (provider capability implementation)

```text
execute(
    integration,
    validated_request,
    resolved_resource_scope,
    effective_budget,
) -> normalized provider result
```

A handler must:

- Be read-only
- Receive the **already resolved** integration instance
- Accept a capability-specific validated request
- Never resolve credentials
- Never persist result bodies
- Return a typed result for normalization

### 8.5 `LiveCapabilityHandlerRegistry` (executable handlers)

Runtime registry keyed by:

```text
provider_id
integration_kind
capability_id
```

Separate from `TenantLiveCapabilityCatalog`. Validation must require:

```text
descriptor identity
=
handler registry identity
=
validated Live Access Binding identity
```

### 8.6 `WorkspaceLiveCapabilityExecutorPort` / `WorkspaceLiveCapabilityExecutor`

Provider-neutral application port. One validated executable live call in; transient normalized live evidence out.

**Frozen dependencies:**

```text
TenantConnectionPort
TenantLiveCapabilityCatalogPort
TenantConnectionIntegrationResolverPort
LiveCapabilityHandlerRegistry
live result normalizer
clock / timeout boundary
```

**Execution flow:**

```text
validated ExecutableLiveCallV1
→ safe Connection projection (TenantConnectionPort)
→ descriptor lookup (TenantLiveCapabilityCatalog)
→ handler lookup (LiveCapabilityHandlerRegistry)
→ existing integration resolution (TenantConnectionIntegrationResolverPort)
→ bounded handler execution (LiveCapabilityHandlerV1)
→ normalized transient live evidence (LiveWorkspaceEvidenceV1)
→ optional safe receipt
```

**Responsibilities:**

- Enforce timeout, item and byte limits (strictest effective budget)
- Normalize provider errors to stable domain codes
- Never expose credentials or raw provider clients
- Never persist result bodies by itself

**Rejected:** direct Jira, Confluence, Microsoft Graph or Slack branches in the executor; orchestrator calling `JiraIssueTrackerIntegration` / `ConfluenceWikiKnowledgeIntegration` directly; provider branches in `WorkspaceAskService`; separate provider-specific Ask services; describing `TenantConnectionCapabilityReadService` as an integration resolver.

---

## 9. First provider decision

### 9.1 Selected: **Jira** (`jira` / `issue_tracker`) — one initial capability only

**First bounded live capability proof:** exactly one capability:

```text
capability_id: jira.issue.read
```

| Field | Value |
|-------|-------|
| Production integration method | `JiraIssueTrackerIntegration.get_knowledge_issue(issue_key)` |
| Typed request | `JiraIssueReadRequestV1` → `issue_key` |
| Typed result | `JiraKnowledgeIssue` |

**Required authorization (frozen):**

1. Live Access Binding identifies the allowed Jira project remote resource.
2. Server-side remote-resource resolution derives the authoritative Jira project key.
3. Request schema validates `issue_key`.
4. Deterministic scope validation proves the issue key belongs to the bound project.
5. The model cannot provide or override the authoritative project scope.
6. The same rehydrated Jira integration instance is used (no second client).
7. No arbitrary JQL is accepted.

**Deferred to a later capability task:**

```text
jira.issues.list_bounded
jira.issues.search
```

Note: `search_knowledge_issues(project_key, cursor, limit)` on `JiraIssueTrackerIntegration` is **bounded project inventory/listing**, not semantic or free-form search. It is **not** part of the first Hybrid Ask proof.

**Selection rationale:**

- Production integration method exists (`JiraIssueTrackerIntegration`)
- Read-only, typed input/output (`knowledge_read.py`)
- Project scope enforced server-side — **no LLM-generated JQL**
- Connection-aware instance reuse through `KnowledgeConnectionRegistry` / integration wiring
- Deterministic fixture proof without network access (strict Pydantic parsers)

### 9.2 Deferred: **Confluence**

Confluence exposes `list_knowledge_pages(space_id, cursor, limit)` (inventory listing) and `get_knowledge_page(page_id, version_number)`. The legacy `search_pages(query, limit)` path accepts arbitrary query strings and is unsuitable for V1 Hybrid Ask policy. Exact read requires explicit `version_number`, increasing plan complexity. Confluence live proof follows after Jira bounded capability acceptance.

Microsoft Graph live search is **not** selected and must not be simulated through delta, reconciliation or full inventory.

---

## 10. Unified evidence ABI

Two separate model families: **transient synthesis evidence** (in-memory during execution) and **durable evidence projection** (persisted Ask record).

### 10.1 Transient synthesis evidence (in-memory only)

Exists during query execution and synthesis. Contains bounded `content`. Passed to the assembler. **Not** the durable Ask record contract. Discarded after finalization according to retention policy.

```text
WorkspaceEvidenceV1
├── IndexedWorkspaceEvidenceV1
└── LiveWorkspaceEvidenceV1
```

**Common transient fields:**

```text
evidence_id
evidence_type          # indexed | live
tenant_id
workspace_id
safe_display_name
retrieved_at
content                # excerpt or bounded structured content (transient only)
content_hash
audience
```

### 10.2 Indexed provenance (transient)

```text
source_id
document_id
chunk_id
location
score
safe_source_label / path projection
indexed_source_binding_id (when available)
```

### 10.3 Live provenance (transient)

```text
live_access_binding_id
connection_ref
capability_id
remote_resource_id
remote_item_id
provider_id
integration_kind
call_id
remote_updated_at
safe_locator (when approved)
truncation_state
execution_receipt_id (when retained)
```

Provider identity is permitted in safe provenance. Credentials, private endpoints and raw authorization material are forbidden.

### 10.4 Durable evidence projection (persisted Ask record)

Separate name family — **not** `WorkspaceEvidenceV1`:

```text
PersistedAskEvidenceV2
├── PersistedIndexedEvidenceV2
└── PersistedLiveEvidenceProvenanceV2
```

`WorkspaceAskRunV2` stores `list[PersistedAskEvidenceV2]` — provenance projections only, never transient live bodies.

### 10.5 Evidence identity rules

Evidence IDs must:

- Be unique within one Ask run
- Remain stable from plan validation through synthesis
- Not embed secrets or raw provider payloads
- Support deterministic citation resolution
- Distinguish indexed and live namespaces

**Recommended encoding:**

```text
indexed:  idx:{workspace_id}:{document_id}:{chunk_id_or_digest}
live:     live:{call_id}:{remote_item_id_digest}
```

List position (`E1`, `E2`) may remain a **synthesis index** but must not be the sole durable evidence identity.

---

## 11. Unified citation ABI

### 11.1 Discriminated union

```text
WorkspaceCitationV1
├── IndexedWorkspaceCitationV1
└── LiveWorkspaceCitationV1
```

### 11.2 Common fields

```text
evidence_id
evidence_type
safe_display_name
excerpt (when retention permits)
retrieved_at
```

### 11.3 Indexed citation (backward compatible)

Retains existing `WorkspaceAskCitationV1` fields where practical: `document_id`, `source_id`, `workspace_id`, `source_path`, `file_name`, `chunk_id`, `score`, `location`.

### 11.4 Live citation

```text
provider_id
connection_safe_label
capability_id
remote_resource_id / remote_item_id (when safe)
freshness (remote_updated_at)
call_id / receipt_id
```

A live citation must **not** pretend to be a durable LKW Document.

Safe final citations may contain only explicitly frozen minimum fields. Under `EPHEMERAL`, an excerpt is **not** retained unless the architecture explicitly classifies it as retained content (see §13).

### 11.5 Public HTTP versioning (frozen path versioning)

Explicit path versioning — **no** content negotiation required to distinguish V1 and V2.

| Route | Contract | Behavior |
|-------|----------|----------|
| `POST /v1/local_workspace/workspaces/{workspace_id}/ask` | `WorkspaceAskRequestV1` / `WorkspaceAskResponseV1` | **Indexed-only** — never invokes live capabilities; response shape unchanged |
| `GET /v1/local_workspace/asks/{run_id}` | V1 run projection | Indexed-only durable shape for V1 runs |
| `POST /v2/local_workspace/workspaces/{workspace_id}/ask` | `WorkspaceAskRequestV2` / `WorkspaceAskResponseV2` | Supports `indexed_only`, `live_only`, `hybrid` |
| `GET /v2/local_workspace/asks/{run_id}` | `WorkspaceAskRunV2` durable projection | Returns durable V2 projection — **not** transient live bodies |

**Rejected alternatives:** `Accept` header versioning, `schema-version` header, `api_version` request field.

**Cross-version run access (frozen):**

```text
GET /v1/local_workspace/asks/{run_id}
+ run is WorkspaceAskRunV2
→ HTTP 409
→ ask_run_version_mismatch

GET /v2/local_workspace/asks/{run_id}
+ run is WorkspaceAskRun V1
→ HTTP 409
→ ask_run_version_mismatch
```

- V1 GET returns only V1 records.
- V2 GET returns only V2 records.
- No automatic V2-to-V1 projection.
- No automatic V1-to-V2 projection.
- No document-only representation of a hybrid run.
- The repository may identify the stored schema version, but the route must fail deterministically when the version does not match.
- V1 clients that only use V1 routes continue to receive unchanged indexed-only behavior for V1 runs.

---

## 12. Synthesis contract

One synthesis step over unified `WorkspaceEvidenceV1` collection.

| Rule | Requirement |
|------|-------------|
| Assembler input | Provider-neutral evidence list |
| Assembler executes live calls | **No** |
| Invent evidence IDs | **No** |
| Used evidence IDs | Must resolve to validated evidence |
| Completed answer | At least one citation |
| Hybrid acceptance | At least one used indexed **and** one used live evidence item |
| Answer text | May reference both modes; provenance remains distinguishable |
| Insufficient evidence | First-class `insufficient_evidence` status |
| Model-only trust | No evidence body trusted solely because the model references it |

**Rejected:** separate indexed-answer and live-answer LLM calls with concatenated prose.

`AskAnswerAssembler` evolves to accept `list[WorkspaceEvidenceV1]`; indexed-only path projects indexed members to preserve behavior.

---

## 13. Live result retention

Query Policy `live_result_retention` (`LiveResultRetentionV1`) governs post-synthesis persistence of live evidence.

### 13.1 Transient evidence destruction point

Transient `WorkspaceEvidenceV1` (including `content`) exists only from plan validation through synthesis finalization. After the run is finalized and persisted:

- Transient live evidence bodies are destroyed per retention policy.
- Only durable projections (`PersistedAskEvidenceV2`, citations, optional receipts) remain in `WorkspaceAskRunV2`.
- `GET /v2/local_workspace/asks/{run_id}` returns the durable projection only.

### 13.2 `EPHEMERAL` (default)

- Raw provider result never durably persisted
- Normalized live evidence body/excerpt **not** stored in Ask repository after synthesis
- Final answer may remain part of durable Ask run
- Safe citations and minimum audit provenance may persist per public contract
- No LKW Document, Chunk or Vector created from live results

**`PersistedLiveEvidenceProvenanceV2` under `EPHEMERAL` may contain only:**

```text
evidence_id
evidence_type
safe_display_name
provider_id
live_access_binding_id
connection_ref or safe connection label
capability_id
safe remote resource/item references when approved
retrieved_at
remote_updated_at
content_hash
truncated
call_id
```

**Must not contain:**

```text
content
excerpt copied from live result
structured live result
raw provider body
```

### 13.3 `RECEIPT_ONLY`

May additionally persist `LiveExecutionReceiptV1`:

```text
receipt_id
run_id
call_id
live_access_binding_id
capability_id
started_at
completed_at
item_count
byte_count
content_hash / result_hash
truncated
normalized_outcome (safe enum / code only)
```

Must **not** persist: raw provider body, credentials, tokens, private headers, provider client, unapproved private locator. Live evidence bodies and excerpts remain forbidden.

### 13.4 `WorkspaceAskRunV2` (durable model)

Frozen contents:

```text
run metadata
answer
citations
persisted evidence provenance (list[PersistedAskEvidenceV2])
optional safe receipts
execution status
```

Must **not** contain transient live evidence bodies or `WorkspaceEvidenceV1.content`.

- **V1 runs** (`WorkspaceAskRun`): indexed evidence as `WorkspaceSearchHitV1`; unchanged for historical reads.
- In-flight execution state (plan, partial calls) is separate from durable run record.

---

## 14. Ask Run persistence and compatibility

| Question | Decision |
|----------|----------|
| V1 vs V2 model | `WorkspaceAskRun` remains V1; introduce `WorkspaceAskRunV2` for hybrid/live |
| Indexed run readability | V1 records unchanged; repository returns V1 shape for `run_id` without V2 marker |
| Transient vs durable evidence | `WorkspaceEvidenceV1` transient during execution; `PersistedAskEvidenceV2` durable in run |
| In-flight vs durable | `EvidencePlanV1` and execution receipts are separate types from persisted run evidence |
| Citation versioning | `citation_schema_version: 1 \| 2` on run; V2 uses discriminated citations |
| Live fields persisted | Provenance + citations + optional receipt only — never raw body by default |
| Run metadata | `query_mode`, `configuration_revision`, `plan_id`, `indexed_retrieval_status`, `live_execution_status`, `truncation`, `partial_failure` |
| GET Run V1 | `GET /v1/local_workspace/asks/{run_id}` — V1 indexed projection only; V2 run → `ask_run_version_mismatch` (409) |
| GET Run V2 | `GET /v2/local_workspace/asks/{run_id}` — durable `WorkspaceAskRunV2` projection without transient bodies; V1 run → `ask_run_version_mismatch` (409) |
| Cross-version access | No projection; version mismatch fails with HTTP 409 |

No in-place mutation of historical records without explicit migration task.

---

## 15. Failure semantics

Stable domain outcomes (fail-closed). Each code has **one** unambiguous HTTP status for V2. Where V1 behavior differs, implementation documents it explicitly (`LKW-HYBRID-ASK-1E`) — do not mix statuses in one cell.

| Code | Meaning | V2 HTTP |
|------|---------|---------|
| `workspace_not_found` | Workspace missing or unauthorized | 404 |
| `query_policy_required` | No persisted policy; `live_only` or `hybrid` requested | 409 |
| `query_policy_invalid` | Committed server projection is invalid | 503 |
| `query_mode_not_allowed` | Mode not permitted by policy | 403 |
| `configuration_revision_mismatch` | Request revision does not match committed head | 409 |
| `ask_run_version_mismatch` | Run schema version does not match route version | 409 |
| `configuration_projection_unstable` | Head revision changed during plan/commit | 503 |
| `configuration_projection_invalid` | Committed projection failed validation | 503 |
| `indexed_retrieval_failed` | Search/RAG path failed | 502 |
| `live_binding_not_found` | Binding ID unknown | 404 |
| `live_binding_unavailable` | Binding disabled/unavailable | 409 |
| `live_capability_not_allowed` | Not on binding/policy intersection | 403 |
| `live_capability_unavailable` | Catalog/connection unavailable | 503 |
| `live_request_invalid` | Typed request failed schema/scope | 400 |
| `live_execution_timeout` | Exceeded duration budget | 504 |
| `live_execution_failed` | Provider error after normalization | 502 |
| `live_result_invalid` | Normalization/validation failed | 502 |
| `live_result_too_large` | Exceeded byte/item budget | 413 |
| `audience_mismatch` | Personal/shared violation | 403 |
| `insufficient_evidence` | Cannot ground answer | 200 (product status) |
| `assembly_failed` | Synthesis/parse failure | 502 |
| `unknown_evidence_id` | Model cited unknown ID | 502 |
| `persistence_failed` | Run save failed | 500 |

**Query Policy default (no error):** absent persisted Query Policy + `indexed_only` V1/V2 request → effective `indexed_only`, no error.

Configuration projection failures (`configuration_projection_unstable`, `configuration_projection_invalid`, `query_policy_invalid`) map to **503** — aligned with accepted Knowledge Access HTTP behavior. Do **not** map them to 409.

**Hybrid fail-closed rules:**

- Required planned live call cannot silently disappear
- Authorization failure ≠ `insufficient_evidence`
- Provider unavailability ≠ no matching result
- Partial provider payloads not synthesized unless normalized and marked complete enough
- Unavailable live dependency does not auto-disable durable bindings
- No raw provider exception text in public responses

---

## 16. Budgets and truncation

Enforced fields (Query Policy upper bound; capability descriptor may lower):

```text
max_live_calls
max_total_duration_ms
max_result_items
max_result_bytes
```

- Strictest effective value wins across policy, descriptor and remaining run budget
- Budget calculated before and during execution
- Normalization cannot exceed remaining budget
- Truncation explicit in evidence and receipt; provenance identity preserved
- Oversized unsafe payloads fail closed
- No automatic second live call after budget exhaustion

---

## 17. Observability

**Safe fields:**

```text
run_id, plan_id, call_id
tenant_hash / workspace_hash (or approved opaque refs)
configuration_revision, mode, capability_id, live_access_binding_id
duration_ms, item_count, byte_count, truncated, normalized_outcome, error_code
```

**Never log by default:** question content, raw live results, credentials, tokens, private URLs, authorization headers, provider client repr.

Identifiers that may contain PII or tenant-identifying material use hashing or redaction per deployment observability policy.

---

## 18. Explicitly rejected designs

| Rejected | Reason |
|----------|--------|
| Extend `WorkspaceSearchHitV1` with optional live fields | Breaks indexed contract; prevents clean union |
| Pretend live evidence is an indexed Document | Violates persistence boundary |
| Persist live result bodies by default | Violates EPHEMERAL contract |
| LLM chooses provider IDs, credentials, endpoints, URLs | Authorization bypass |
| LLM executes arbitrary JQL/SQL/DAX/Graph/MCP | Unbounded provider access |
| Authorize live calls from Connection Attachment alone | Missing binding allowlist |
| Authorize live calls from Query Policy alone | Missing binding scope |
| Bypass Live Access Binding | Policy intersection violated |
| Provider branches in `WorkspaceAskService` | Orchestrator owns mode logic |
| Separate provider-specific Ask services | Duplicates policy enforcement |
| Direct provider calls from answer assembler | Separation of concerns |
| One client for indexed + another for live access | Connection reuse violated |
| Silent fallback from failed hybrid to indexed-only | Fail-closed hybrid semantics |
| Citation IDs from list position only | Unstable provenance |
| Merge personal and shared evidence | Audience isolation |
| Convert live results to Documents without Knowledge Intake | Intake owns durability |

---

## 19. Implementation decomposition

Parent block: **`LKW-HYBRID-ASK-1`** — **BLOCKED_ON_ARCH_ACCEPTANCE** until this document is accepted.

### `LKW-HYBRID-ASK-1A` — Unified evidence, provenance, citation and Ask Run V2 contracts

| | |
|---|---|
| **Purpose** | Freeze Pydantic/domain types for transient `WorkspaceEvidenceV1`, `PersistedAskEvidenceV2`, citations, `WorkspaceAskRunV2` |
| **Depends on** | `LKW-KNOWLEDGE-ACCESS-1` (accepted), this architecture |
| **Production scope** | Models, schemas, repository serialization contract |
| **Non-goals** | Live execution, orchestrator, HTTP |
| **Tests** | Model validation, evidence ID rules, V1/V2 run round-trip |
| **Acceptance gate** | Types compile; V1 indexed run still deserializes |
| **User-visible** | None (contract only) |

### `LKW-HYBRID-ASK-1B` — Query Policy V2, effective policy and Evidence Plan validation

| | |
|---|---|
| **Purpose** | `QueryPolicyModeV2` with `hybrid`; `EvidencePlanV1` validator |
| **Depends on** | `1A`, Knowledge Configuration service |
| **Production scope** | Policy resolution, plan builder/validator, mode intersection |
| **Non-goals** | Provider execution |
| **Tests** | Validation order, intersection, audience rejection, V1 policy compatibility |
| **Acceptance gate** | Valid hybrid plan approved; invalid plans rejected with stable codes |
| **User-visible** | None |

### `LKW-HYBRID-ASK-1C` — Live Capability Executor + first Jira `jira.issue.read`

| | |
|---|---|
| **Purpose** | `WorkspaceLiveCapabilityExecutorPort` + `jira.issue.read` capability handler |
| **Depends on** | `1B`, `TenantLiveCapabilityCatalog`, `LiveCapabilityHandlerRegistry`, `KnowledgeConnectionRegistry`, Jira integration |
| **Production scope** | Executor with frozen port dependencies, Jira `jira.issue.read` handler, budgets, error normalization |
| **Non-goals** | Ask synthesis, HTTP |
| **Tests** | Fixture-based Jira call, budget truncation, connection reuse, no persistence |
| **Acceptance gate** | One validated call returns `LiveWorkspaceEvidenceV1`; no raw body stored |
| **User-visible** | None (internal port) |

### `LKW-HYBRID-ASK-1D` — Knowledge Query Orchestrator

| | |
|---|---|
| **Purpose** | Mode-aware orchestration for `indexed_only`, `live_only`, `hybrid` |
| **Depends on** | `1B`, `1C`, indexed retrieval port (existing search path) |
| **Production scope** | `KnowledgeQueryOrchestrator`, plan execution, unified evidence collection |
| **Non-goals** | HTTP, assembler changes beyond interface |
| **Tests** | Per-mode execution, hybrid fail-closed, partial failure semantics |
| **Acceptance gate** | Hybrid run produces ≥1 indexed + ≥1 live evidence in memory |
| **User-visible** | None |

### `LKW-HYBRID-ASK-1E` — WorkspaceAskService, repository, HTTP V1/V2 integration

| | |
|---|---|
| **Purpose** | Wire orchestrator into Ask; unified citations; retention; stable errors; frozen V1/V2 path versioning |
| **Depends on** | `1A`, `1D` |
| **Production scope** | `WorkspaceAskService`, `ask_repository`, routes/schemas V1 (indexed-only) and V2 (hybrid/live) |
| **Non-goals** | Conversational frontend, Slack adapter |
| **Tests** | Indexed-only regression; hybrid HTTP proof; GET Run V1/V2 |
| **Acceptance gate** | Indexed-only behavior unchanged; hybrid errors stable |
| **User-visible** | Hybrid Ask via HTTP when policy allows |

### `LKW-HYBRID-ASK-1F` — Bounded acceptance proof

| | |
|---|---|
| **Purpose** | End-to-end hybrid proof |
| **Depends on** | `1E`, `LKW-KNOWLEDGE-ACCESS-1` configuration |
| **Proof scenario** | One Connection, one Indexed Source, one Live Access Binding, hybrid Query Policy, one question → ≥1 indexed + ≥1 live evidence, grounded answer, indexed + live citation, same rehydrated integration instance, no durable live body, no new Document/Chunk/Vector |
| **Tests** | Integration/acceptance test with deterministic fixtures |
| **Acceptance gate** | Proof checklist green |
| **User-visible** | Demonstrable hybrid Ask |

**Dependency chain:** `1A → 1B → 1C → 1D → 1E → 1F`. `1C` and indexed retrieval port may proceed in parallel after `1B` where ports are defined.

---

## 20. Relationship to adjacent blocks

| Block | Relationship |
|-------|--------------|
| `LKW-KNOWLEDGE-ACCESS-1` | **Prerequisite (accepted)** — configuration, bindings, Query Policy V1 |
| `LKW-CONVERSATION-CONTEXT-1` | Audience guards feed orchestrator; Hybrid Ask does not reimplement Conversation Context |
| `LKW-CONVERSATIONAL-FRONTEND-1` | Planner invokes `workspace.ask`; follows this contract |
| Vendor Knowledge | Connection catalog, capability catalog, integration reuse — platform-owned |
| `KNOWLEDGE_ACCESS_ARCHITECTURE.md` | Target vocabulary; this document freezes Hybrid Ask implementation contract |

---

## 21. References

| Document | Role |
|----------|------|
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Roadmap status and slice tracking |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | LKW product architecture index |
| [`KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md`](KNOWLEDGE_ACCESS_IMPLEMENTATION_CONTRACT.md) | Query Policy V1, Live Access Binding HTTP contract |
