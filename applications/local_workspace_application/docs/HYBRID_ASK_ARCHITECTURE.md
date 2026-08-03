# Hybrid Ask — unified evidence, query orchestration and read-only live execution

**Status:** READY_FOR_REVIEW  
**Task:** LKW-HYBRID-ASK-ARCH-1-UNIFIED-EVIDENCE-QUERY-ORCHESTRATION-AND-READ-ONLY-LIVE-EXECUTION-CONTRACT  
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

## 8. Live capability execution port

### 8.1 `WorkspaceLiveCapabilityExecutorPort`

Provider-neutral application port. One validated executable live call in; normalized live evidence out.

**Responsibilities:**

- Receive one already validated `ExecutableLiveCallV1`
- Resolve Connection through connection-aware runtime (`TenantConnectionCapabilityReadService`, existing integration instance reuse)
- Invoke one registered read-only capability implementation
- Enforce timeout, item and byte limits (strictest effective budget)
- Normalize provider errors to stable domain codes
- Return `LiveWorkspaceEvidenceV1` and optional safe execution receipt
- Never expose credentials or raw provider clients
- Never persist result bodies by itself

**Rejected:** orchestrator calling `JiraIssueTrackerIntegration` / `ConfluenceWikiKnowledgeIntegration` directly; provider branches in `WorkspaceAskService`; separate provider-specific Ask services.

---

## 9. First provider decision

### 9.1 Selected: **Jira** (`jira` / issue tracker)

**First bounded live capability proof:** project-scoped issue read via existing `JiraKnowledgeReadClient`:

| Capability | Integration method | Typed contract |
|------------|-------------------|----------------|
| Bounded project issue listing | `search_knowledge_issues(project_key, next_page_token, limit)` | `JiraKnowledgeIssuePage` |
| Exact issue read | `get_knowledge_issue(issue_key)` | `JiraKnowledgeIssue` |

**Selection rationale:**

- Production integration methods exist (`JiraIssueTrackerIntegration`)
- Read-only, typed input/output (`knowledge_read.py`)
- Bounded via `limit`; project scope via validated `project_key` — **no LLM-generated JQL**
- Connection-aware instance reuse through Vendor Knowledge / integration wiring
- Vendor Knowledge adapter (`jira_issues.py`) already maps sync surfaces to the same client
- Deterministic fixture proof without network access (strict Pydantic parsers)

### 9.2 Deferred: **Confluence**

Confluence exposes `list_knowledge_pages(space_id, cursor, limit)` (inventory listing) and `get_knowledge_page(page_id, version_number)`. The legacy `search_pages(query, limit)` path accepts arbitrary query strings and is unsuitable for V1 Hybrid Ask policy. Exact read requires explicit `version_number`, increasing plan complexity. Confluence live proof follows after Jira bounded capability acceptance.

Microsoft Graph live search is **not** selected and must not be simulated through delta, reconciliation or full inventory.

---

## 10. Unified evidence ABI

### 10.1 Discriminated union

```text
WorkspaceEvidenceV1
├── IndexedWorkspaceEvidenceV1
└── LiveWorkspaceEvidenceV1
```

### 10.2 Common fields

```text
evidence_id
evidence_type          # indexed | live
tenant_id
workspace_id
safe_display_name
retrieved_at
content                # excerpt or bounded structured content
content_hash
audience
```

### 10.3 Indexed provenance

```text
source_id
document_id
chunk_id
location
score
safe_source_label / path projection
indexed_source_binding_id (when available)
```

### 10.4 Live provenance

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

### 11.5 Public API versioning

- **V1 HTTP responses** remain valid for indexed-only runs (`WorkspaceAskResponseV1` shape).
- **V2** introduces optional `evidence_type` on citations, `query_mode`, `plan_id`, and discriminated evidence on GET Run when hybrid/live participated.
- Clients that do not send `Accept` / schema version continue to receive V1-compatible indexed citations for indexed-only runs.
- Hybrid runs require V2 response contract or explicit `api_version=2` request field (exact HTTP field frozen in `LKW-HYBRID-ASK-1E`).

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

Query Policy `live_result_retention` (`LiveResultRetentionV1`) governs post-synthesis persistence.

### 13.1 `EPHEMERAL` (default)

- Raw provider result never durably persisted
- Normalized live evidence body/excerpt **not** stored in Ask repository after synthesis
- Final answer may remain part of durable Ask run
- Safe citations and minimum audit metadata may persist per public contract
- No LKW Document, Chunk or Vector created from live results

### 13.2 `RECEIPT_ONLY`

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

Must **not** persist: raw provider body, credentials, tokens, private headers, provider client, unapproved private locator.

### 13.3 `WorkspaceAskRun` evolution

- **V1 runs** (`WorkspaceAskRun`): indexed evidence as `WorkspaceSearchHitV1`; unchanged for historical reads.
- **V2 runs** (`WorkspaceAskRunV2`): `evidence` as `list[WorkspaceEvidenceV1]`; live bodies omitted under `EPHEMERAL`; receipts optional under `RECEIPT_ONLY`.
- In-flight execution state (plan, partial calls) is separate from durable run record.

---

## 14. Ask Run persistence and compatibility

| Question | Decision |
|----------|----------|
| V1 vs V2 model | `WorkspaceAskRun` remains V1; introduce `WorkspaceAskRunV2` for hybrid/live |
| Indexed run readability | V1 records unchanged; repository returns V1 shape for `run_id` without V2 marker |
| In-flight vs durable | `EvidencePlanV1` and execution receipts are separate types from persisted run evidence |
| Citation versioning | `citation_schema_version: 1 \| 2` on run; V2 uses discriminated citations |
| Live fields persisted | Provenance + citations + optional receipt only — never raw body by default |
| Run metadata | `query_mode`, `configuration_revision`, `plan_id`, `indexed_retrieval_status`, `live_execution_status`, `truncation`, `partial_failure` |
| GET Run compatibility | V1 clients: indexed fields only; V2 clients: full union when `schema_version=2` |

No in-place mutation of historical records without explicit migration task.

---

## 15. Failure semantics

Stable domain outcomes (fail-closed):

| Code | Meaning | HTTP family |
|------|---------|-------------|
| `workspace_not_found` | Workspace missing or unauthorized | 404 |
| `query_policy_missing_or_invalid` | No policy or revision mismatch | 400 / 409 |
| `query_mode_not_allowed` | Mode not permitted by policy | 403 |
| `configuration_projection_unstable` | Head revision changed during plan | 409 |
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
| **Purpose** | Freeze Pydantic/domain types for `WorkspaceEvidenceV1`, citations, `WorkspaceAskRunV2` |
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

### `LKW-HYBRID-ASK-1C` — Live Capability Executor + first Jira capability

| | |
|---|---|
| **Purpose** | `WorkspaceLiveCapabilityExecutorPort` + Jira bounded read capability |
| **Depends on** | `1B`, `TenantLiveCapabilityCatalog`, Jira integration |
| **Production scope** | Executor, Jira capability handler, budgets, error normalization |
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

### `LKW-HYBRID-ASK-1E` — WorkspaceAskService, repository, HTTP integration

| | |
|---|---|
| **Purpose** | Wire orchestrator into Ask; unified citations; retention; stable errors |
| **Depends on** | `1A`, `1D` |
| **Production scope** | `WorkspaceAskService`, `ask_repository`, routes/schemas V2 |
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
