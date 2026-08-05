# Vendor Knowledge Live Capability Rollout Architecture

**Status:** `READY_FOR_REVIEW`
**Task:** `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1`
**Parent plan:** `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1` — `ACCEPTED / CLOSED`
**Accepted plan ancestor:** `c21f9a35e1719177b40a1ba33d58baf0e6df41f3`
**Branch:** `development`

This document freezes one provider-neutral Vendor Knowledge live capability
boundary. It is the implementation contract for later provider/source-kind
tasks. It does not activate any provider task.

## 1. Scope and non-goals

The boundary covers exactly these fifteen source kinds:

| provider family | source kinds |
|---|---|
| Microsoft Graph (`ms365_graph`) | `drive`, `mail`, `teams_channel`, `teams_chat`, `calendar` |
| Slack (`slack`) | `slack_conversation` |
| Jira (`jira`) | `issues` |
| Confluence (`confluence`) | `pages` |
| Google Workspace (`google_workspace`) | `drive`, `docs`, `sheets`, `calendar`, `slides`, `mail`, `chat` |

Databricks has no selected source kind and is excluded. Power BI and Atlan are
outside this rollout.

Live means one authorized, read-only, bounded call that returns normalized
ephemeral evidence and optionally a receipt. It is not:

- durable synchronization or materialization;
- background ingestion, automatic indexing, or corpus refresh;
- write-back, provider mutation, subscription, or webhook processing;
- a provider SDK boundary exposed to an application;
- a durable continuation or synchronization job.

This task creates no handlers, descriptors, registrations, runtime models,
provider wiring, application orchestration, tests, or Hybrid Ask changes.

## 2. Existing foundation

The architecture reuses the current production foundation:

- `LiveCapabilityDescriptorV1` and the tenant-safe capability catalog provide
  read-only descriptor discovery.
- `WorkspaceLiveAccessBindingService` creates a durable binding from a tenant
  connection attachment, selected capabilities, optional remote resource,
  audience eligibility, and configuration revision.
- `validate_evidence_plan` verifies tenant/workspace and configuration
  revision, active attachment and binding, policy allowlists, audience
  eligibility, descriptor identity, request validation, resource scope, and
  restrictive budgets before execution.
- `LiveCapabilityExecutorV1` is the authoritative one-call executor. It
  resolves one exact handler, receives one already resolved integration,
  applies the deadline with `asyncio.wait_for`, validates normalized results,
  enforces item/byte limits, and creates receipt-only receipts.
- `KnowledgeConnectionRegistry` is the instance-local registry of already
  constructed integration instances. It does not look up secrets or construct
  provider clients.
- `LiveCapabilityResultItemV1`, `LiveCapabilityExecutionResultV1`,
  `LiveWorkspaceEvidenceV1`, and `LiveExecutionReceiptV1` are the normalized
  result/evidence/receipt foundation. Their semantics are tightened by this
  document where the current fields are not yet sufficient for the rollout.

The existing application orchestration remains the application boundary. The
architecture does not move it into `intergrax/` and does not create a second
executor, registry, or receipt system.

### 2.1 Review finding: production foundation versus frozen shared delta

The current production foundation already provides:

```text
LiveCapabilityDescriptorV1
tenant-safe capability catalog
durable Live Access Binding lifecycle
evidence-plan validation
LiveCapabilityHandlerV1 protocol
exact handler registry
provider-neutral executor
connection integration resolver
basic item/byte budgets
normalized result/evidence models
receipt-only retention
```

`ARCH-1` nevertheless freezes an additional provider-neutral shared delta that
does not yet exist as one accepted runtime boundary:

```text
canonical source_kind assertion and validation
contract_version across descriptor/handler/request/result/binding
strict capability-specific request models
request/result schema resolution
ValidatedLiveCapabilityCallV1 or equivalent typed call
atomic descriptor-handler-schema registration
missing-pair and duplicate-pair validation
provider page/request/upstream/content budgets
expanded provider-neutral error taxonomy
source-kind-aware result/evidence provenance
ordered item-identity-aware receipt hashing
safe-locator validation/filtering
shared registration bootstrap
shared contract test suite
```

These are provider-neutral runtime changes. They must be implemented by
`VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1` before any provider-specific
live handler is activated; the first Microsoft Graph task must not implement
them privately.

## 3. Ownership boundaries

### Provider/core workstreams own

- provider clients, authentication, credentials, token handling and transport;
- pagination and provider cursor semantics;
- typed provider read primitives and provider-specific read models;
- stable provider references and provider error boundaries;
- durable Vendor Knowledge adapters;
- source-specific core and adapter tests;
- provider-family core closeout.

### Vendor Knowledge live rollout owns

- live capability descriptors and canonical capability identities;
- the shared request, validation, result, evidence, locator, receipt, and
  retention contracts;
- provider-specific live handlers, descriptor/handler registration, and
  binding enforcement;
- effective budgets, deadline enforcement, error normalization, evidence
  mapping, safe locators, receipts, retention, cross-provider contracts, and
  live family closeouts.

### Application owns

- workspace configuration and Live Access Binding lifecycle;
- query policy and audience context;
- application plan construction and invocation of the shared executor;
- presentation of transient evidence/citations and application-level error
  handling.

Applications never import provider SDKs or call provider integrations directly.
Provider handlers are the only live boundary consumers of the resolved
integration object.

## 4. Component model and one-call lifecycle

The frozen component flow is:

```text
tenant connection
  → workspace connection attachment
  → active Live Access Binding
  → allowed canonical capability
  → optional binding-fixed remote resource
  → validated live call
  → exact descriptor/handler resolution
  → already resolved provider integration
  → bounded provider read
  → normalized result validation
  → executor-enforced item/byte truncation
  → normalized live evidence
  → optional receipt-only receipt
```

The single authoritative executor owns the boundary around provider execution.
The single authoritative descriptor catalog and handler registry own exact
capability routing. The single authoritative receipt contract owns receipts.
No provider family may introduce a parallel implementation of any of these.

## 5. Capability identity and source-kind enforcement

The conceptual identity tuple is exactly:

```text
(provider_id, integration_kind, source_kind, capability_id, contract_version)
```

The authoritative source-kind model is **A: `source_kind` is encoded
canonically in `capability_id`**. The existing exact handler lookup remains
keyed by:

```text
(provider_id, integration_kind, capability_id, contract_version)
```

The source kind is not a second optional lookup dimension. It is parsed from
the canonical capability ID and must be validated equal to:

1. the descriptor's declared source kind;
2. the handler's declared source kind;
3. the source kind of the bound remote resource, when a resource is bound;
4. the provider read surface selected by the handler.

Any disagreement makes the descriptor/handler pair invalid and unavailable.
There is no fallback lookup by provider alone, capability suffix, resource
type, or application label. This prevents one capability from serving two
source kinds, wrong adapter selection, cross-source resource reuse, and
provider-family ambiguity.

A descriptor or handler may carry `source_kind` as an explicit validated
assertion for diagnostics and registration checks, but `capability_id` is the
canonical source-kind authority. A provider task must not register the same
capability identity for another source kind.

## 6. Capability naming grammar

Every live identity uses this grammar:

```text
vendor.<provider_id>.<source_kind>.<operation>[.<suboperation>]
```

The provider and source-kind segments use lowercase ASCII slugs with
underscore-separated words. The operation vocabulary is restricted to:

```text
search
list
read
thread.read
child.read
content.read
```

Examples of valid shapes are:

```text
vendor.ms365_graph.drive.search
vendor.ms365_graph.mail.read
vendor.ms365_graph.teams_channel.thread.read
vendor.slack.slack_conversation.read
vendor.jira.issues.search
vendor.confluence.pages.read
vendor.google_workspace.calendar.list
vendor.google_workspace.docs.content.read
```

The final operation set for a source kind is selected by its provider task;
these examples do not claim provider availability. `search`, `list`, exact
`read`, thread/child reads, and bounded `content.read` are separate identities
when their semantics differ. No name contains an application name, tenant,
workspace, connection, resource, query, cursor, credential, or write/admin
verb. Contract version is never encoded only by renaming a capability.

Values such as `PLANNED:vk...search/read` in earlier rollout matrices are
planning placeholders, not canonical production capability IDs.

## 7. Operation classes

Provider tasks must declare each applicable class as exactly one of:
`SUPPORTED`, `UNSUPPORTED_BY_PROVIDER`, `DEFERRED`, or `NOT_APPLICABLE`.

- **Bounded search:** provider-native search/filter primitive with validated
  query, fixed scope, and upstream limits. A local filter over an unbounded
  download is forbidden.
- **Bounded list/query:** provider-native collection/list or bounded time-window
  query. It has its own capability identity even when it resembles search.
- **Exact item read:** one stable remote identity, with one bounded provider
  read surface. It is not inferred from a list response.
- **Thread/child collection read:** a bounded child collection under an item or
  parent resource. It cannot widen the parent binding.
- **Bounded content read:** provider-native content bytes/text read with an
  explicit per-item and total content ceiling. Metadata or a hosted reference
  does not imply content bytes.

An operation marked unsupported, deferred, or not applicable has no descriptor
or handler. A provider task must not synthesize an operation from a less
truthful primitive.

## 8. Shared request contract

The shared immutable request envelope is the only handler call envelope:

```text
call_id
capability_id
contract_version
connection_ref
live_access_binding_id
remote_resource_id (nullable, binding-derived)
validated operation-specific request model
requested/effective budget snapshot
audience context reference when required
```

The envelope contains no tenant-supplied credentials, tokens, clients,
authorization headers, provider endpoint, raw SDK model, raw durable cursor,
or arbitrary connection/resource identity. `tenant_id`, `workspace_id`, and
configuration revision are execution context and are copied from authoritative
application state, not accepted as provider operation arguments.

The public proposal may arrive as JSON-like data at the application boundary,
but the handler contract is not a dictionary. The executor receives an
immutable, typed `ValidatedLiveCapabilityCallV1`; its operation field is a
capability-specific strict Pydantic model. That model is the only place where
operation-specific arguments are represented.

## 9. Request model ownership and validation

Shared envelope and result contracts belong to the platform Vendor Knowledge
runtime, not application code. Source-specific live request models belong under
the Vendor Knowledge live boundary, preferably:

```text
intergrax/runtime/vendor_knowledge/live/<provider_id>/
```

or an equivalent existing Vendor Knowledge-owned package with the same
ownership. They must not be placed in provider SDK modules or application
orchestration.

Each descriptor declares a resolvable, versioned `request_schema_ref`, for
example:

```text
schema://vendor-knowledge/live/<provider>/<source_kind>/<operation>/request/v1
```

Registration validates that the schema reference resolves to the model
expected by the handler and that its version equals the descriptor contract
version. Models are immutable, strict, and `extra="forbid"`; coercion of
unknown or ambiguous types is not allowed.

Validation order is mandatory:

```text
binding/configuration/audience checks
→ exact descriptor lookup
→ request schema resolution and strict validation
→ resource-scope validation
→ effective budget calculation
→ provider integration resolution
→ handler/provider invocation
```

Malformed input, unknown fields, schema mismatch, or invalid types become
`live_request_invalid` without invoking the provider. Provider handlers receive
the validated model and cannot replace validation with a private convention.

## 10. Result contract

The normalized result is provider-neutral and immutable. It contains the
validated call identity, timestamps, `normalized_outcome`, `items`,
`item_count`, `byte_count`, `truncated`, an optional stable `error_code`, and
an optional receipt. The executor recalculates item and byte counts from the
accepted bounded items; handler-supplied counts are not authoritative.

Every result item has:

```text
remote_item_id       stable provider/source identity, never a display label
safe_display_name    bounded, secret-free name
content              bounded normalized text; empty is allowed when contract says metadata-only
content_hash         SHA-256 of the returned content
retrieved_at         timezone-aware retrieval time
remote_updated_at    optional timezone-aware provider update time
safe_locator         optional safe reference
truncated            item-level truncation marker
```

The result also carries or is bound to:

```text
provider_id
integration_kind
source_kind
capability_id
contract_version
```

These system-owned fields override conflicting provider metadata. Raw provider
SDK payloads, response bodies, headers, tokens, cursors, and provider exception
types never cross the boundary.

Provider-specific metadata is allowed only in a separate `metadata` field that
is strictly typed or JSON-safe, bounded, secret-free, and non-authoritative
for tenant, workspace, audience, connection, binding, or resource ownership.
It cannot replace or override system-owned fields.

## 11. Scope and authorization

Every live call follows this exact authorization chain:

```text
tenant connection
→ workspace connection attachment
→ active Live Access Binding
→ allowed capability
→ optional binding-fixed remote resource
→ validated live call
→ provider integration resolution
```

The authoritative values are checked as follows:

| identity | enforcement |
|---|---|
| `tenant_id` | configuration, binding, attachment, connection and execution context must match |
| `workspace_id` | configuration, binding and execution context must match |
| `connection_ref` | active attachment and binding value; never operation-supplied |
| `live_access_binding_id` | active binding in the same tenant/workspace and current configuration revision |
| `remote_resource_id` | binding value; request cannot replace or widen it |
| `audience` | query context is checked against binding `audience_eligibility` before provider invocation |
| `configuration_revision` | plan and configuration revision must match exactly |

The binding is authoritative for provider, integration kind, connection, scope,
and allowed capability set. Provider-returned metadata never grants access.
Cross-tenant, cross-workspace, cross-attachment, and cross-binding reuse fails
closed.

Scope behavior is fixed:

- A **connection-wide** capability requires a binding with no remote resource;
  operation arguments cannot add one.
- A **resource-required** capability requires a binding with one remote
  resource of the matching source/resource type.
- A **resource-optional** capability may use the binding's fixed resource, or
  operate connection-wide only when its descriptor explicitly allows that
  form; an operation cannot expand a resource-optional binding.
- A **child-resource** read requires a bound parent. The child identifier is
  validated as belonging to that parent by the provider read surface; it is not
  an alternate parent or connection identity.
- A **search within a bound parent** uses only the binding-fixed parent and
  bounded query arguments. It cannot accept a second parent or broaden scope.

## 12. Effective budget model

The final budget is the minimum of every applicable limit:

```text
descriptor limits
∩ query-policy limits
∩ binding limits, if the binding contract later adds them
∩ application execution limits
∩ hard platform ceilings
∩ remaining absolute deadline
```

No provider task may raise an effective limit. A handler may return fewer
results. The executor enforces the final output limits even if a handler fails
to do so.

The budget contract includes:

```text
max_live_calls                 run-level
max_total_duration_ms          run-level
deadline                       absolute executor deadline
max_result_items               per call
max_result_bytes               per call
max_provider_pages             per call
max_provider_requests          per call
max_upstream_items             per call
max_provider_page_size         per request
max_content_bytes_per_item     per item
```

Provider tasks may add a validated fixed time window or an operation-specific
bound, but not an unbounded default. `max_live_calls` and duration constrain a
run; item, byte, request, page, and content ceilings constrain each call.

## 13. Pagination and provider request bounds

Search/list handlers enforce bounds at the provider boundary. Before invoking
the provider, the handler must have a finite maximum page count, request count,
upstream item count, page size, and—where applicable—a fixed validated time
window. It must stop on the first bound and mark a valid partial result
`truncated=True`.

Provider continuation tokens may be used internally only to traverse pages
within the same bounded call. Raw durable cursors are not a public live
continuation contract. A live call cannot silently become a synchronization
run, persist a checkpoint, or schedule continuation work.

Exact reads and bounded content reads use one exact provider primitive or a
provider-defined bounded sequence; they do not download a collection and filter
locally.

## 14. Timeout and cancellation

The executor owns one absolute deadline. It calculates remaining time and
enforces the provider handler with the existing `asyncio.wait_for` boundary.
Provider handlers must propagate cancellation to provider transport and must
pass a timeout no greater than the remaining deadline to the provider client.

Handlers may not shield work, create background jobs, or add unbounded retries.
Provider resources are cleaned up in the normal cancellation/finally path.

If the deadline expires, the normalized outcome is `FAILED` with
`live_execution_timeout`; no uncertain partial items are emitted. A provider
may return a valid bounded partial result before the deadline, in which case
the result is retained with `TRUNCATED` and its truncation markers. The executor
performs a final deadline check after handler completion.

## 15. Error taxonomy

The following codes are the complete stable public live boundary taxonomy:

```text
live_binding_unavailable
live_capability_unavailable
live_request_invalid
live_resource_scope_invalid
live_execution_timeout
live_execution_failed
live_result_invalid
live_result_too_large
live_provider_unauthorized
live_provider_forbidden
live_provider_not_found
live_provider_throttled
live_provider_temporarily_unavailable
live_provider_contract_violation
```

All provider exceptions, SDK classes, URLs, tokens, raw cursors, and response
bodies are mapped to these safe codes. Internal diagnostics may retain only
redacted structured details keyed by `call_id`; they are never returned as
provider payloads.

Retryability is a caller-facing classification, not an in-call retry policy:

| code group | retryability |
|---|---|
| request, scope, capability, binding, forbidden, not-found, result-invalid, result-too-large, contract-violation | not retryable without a changed contract, binding, or request |
| unauthorized | not automatically retryable; caller must re-authorize/reconnect |
| timeout, execution-failed | retryable only as a new bounded call when policy permits; no automatic retry |
| throttled, temporarily-unavailable | retryable as a new bounded call subject to provider guidance and application policy |

Provider authorization and HTTP/SDK details never determine the public error
shape. `live_provider_contract_violation` is used when the provider or handler
breaks an accepted typed read/result contract.

## 16. Normalized evidence mapping

A successful bounded result maps one item to one transient live evidence record:

```text
provider/source kind identity → provider_id + integration_kind + source_kind
capability identity           → capability_id + contract_version
binding identity              → tenant/workspace/binding/connection provenance
remote item identity          → remote_item_id
display/content               → safe_display_name + content + content_hash
times                         → retrieved_at + remote_updated_at
location                      → safe_locator when safe
limits                        → truncated
audience                      → executor-verified audience
```

System-owned provenance is copied from the validated call and binding, never
from provider metadata. An evidence ID is:

```text
live:<sha256(canonical(
  provider_id, integration_kind, source_kind, capability_id,
  contract_version, live_access_binding_id, connection_ref,
  remote_resource_id, call_id, remote_item_id
))>
```

The canonical encoding is length-delimited or canonical JSON with explicit
field names and null values. Duplicate remote item IDs in one result are a
contract error. This makes evidence collision-safe across providers, source
kinds, bindings, and calls while deterministic within an execution.

## 17. Safe locator

`safe_locator` is an optional inspectable reference, not an authorization
mechanism. It may be:

- a provider-approved HTTPS web URL with no credentials or sensitive grants;
- an opaque provider item reference;
- a bounded human-readable location.

It must not contain access tokens, signed URLs carrying sensitive grants,
authorization headers, credential references, raw continuation tokens, private
receipt data, or unbounded query text. The executor/result validator rejects
or removes unsafe locators. A result is valid with `safe_locator=None`; raw
URLs are never mandatory.

## 18. Receipt and retention

A receipt is created only when the requested retention mode is
`RECEIPT_ONLY` (including normalized failures and timeouts). `EPHEMERAL` keeps
the result in the execution path and does not persist a receipt. A receipt is
opaque, unique, and contains no raw content or provider payload:

```text
receipt_id
run_id
call_id
live_access_binding_id
capability_id + contract_version
provider_id + source_kind
started_at + completed_at
item_count + byte_count
normalized_outcome
error_code, when applicable
truncated
result_hash
```

`result_hash` is SHA-256 of canonical ordered item entries containing position,
`remote_item_id`, `content_hash`, item truncation, outcome, error class,
`item_count`, and `byte_count`. It is ordered and item-identity-aware:
reordering or substituting items cannot produce the same hash merely because
the content hashes happen to match. The hash never requires storing raw
content.

Retention is limited to:

```text
EPHEMERAL     transient result/evidence only
RECEIPT_ONLY  receipt metadata only, no content
```

Durable live content retention is explicitly deferred to the durable
materialization path. Live results do not automatically enter the indexed
corpus. Application provenance may be retained only as a separate,
content-free, explicitly governed application record; that is not a second
receipt system or automatic indexing.

## 19. Descriptor and handler registration

Every provider family contributes one Vendor Knowledge live registration bundle
at the Vendor Knowledge boundary, preferably under:

```text
intergrax/runtime/vendor_knowledge/live/<provider_id>/registration.py
```

The bundle declares descriptors and constructs stateless handlers. Provider
core modules expose typed integration read primitives but do not register live
handlers. LKW-specific modules do not contain production provider handlers.

Application startup passes all bundles through one registration boundary that:

1. validates every descriptor and handler identity;
2. derives and checks the canonical source kind from `capability_id`;
3. resolves request/result schema references;
4. checks descriptor/handler contract-version equality;
5. validates that every descriptor has exactly one handler and every handler
   has exactly one descriptor;
6. rejects duplicates and invalid read-only/effect declarations;
7. atomically publishes one immutable descriptor catalog and one immutable
   handler registry.

A descriptor without a handler is unavailable for execution. A handler without
a descriptor fails startup/registration validation. Duplicate identities fail
closed; no last-write-wins behavior is permitted.

## 20. Integration reuse

Handlers receive an already resolved integration instance:

```text
KnowledgeConnectionRegistry
→ tenant/provider/integration identity check
→ existing integration instance
→ live handler
```

Handlers must not load credentials, resolve secrets, construct or persist
provider clients, own a connection registry, or replace the provider
integration. The same integration instance and transport/read primitives remain
usable by durable adapters and other approved consumers.

This applies equally to families with shared clients. For example, the seven
Google Workspace source kinds reuse one
`GoogleWorkspaceCollaborationSuiteIntegration`; a Google live handler never
constructs a second client family. The same rule applies to the five Microsoft
Graph source kinds and to Slack, Jira, and Confluence.

## 21. Contract versioning and compatibility

The initial live contract version is:

```text
contract_version = "1"
request schema version = 1
result schema version = 1
```

The descriptor, handler, request schema, result schema, executable call, and
receipt must agree on this version. Version is part of the exact conceptual
identity and handler lookup. A binding created for version 1 cannot silently
execute version 2.

Breaking changes require a new contract version and a new independently
registered descriptor/handler identity. During migration, parallel versions
may coexist only when both are fully registered and validated. A compatibility
adapter may translate an old request/result only at an explicitly versioned
boundary; it cannot silently upgrade a binding. Deprecation requires an
application-visible policy window and removal only after no active binding can
select the deprecated version. Renaming a capability without version checks is
not a compatibility strategy.

## 22. Microsoft Graph Mail attachment boundary

The Mail source kind must keep these levels independent:

```text
hasAttachments
attachment inventory
attachment metadata
attachment content bytes
embedded/inline attachment content
provider-hosted attachment reference
```

`hasAttachments` is a message-level boolean and does not imply inventory.
Inventory proves only bounded attachment descriptors. Metadata proves fields
such as stable attachment ID, type, name, size, inline flag, and update
metadata. Content bytes require an exact provider primitive, a bounded content
policy, and an independently declared capability. Inline/embedded content and
provider-hosted references require separate provider declarations; neither
may be inferred from ordinary attachment metadata.

No live Mail capability may imply attachment bytes. The later Mail task must
declare each level independently as `SUPPORTED`, `UNSUPPORTED_BY_PROVIDER`,
`DEFERRED`, or `NOT_APPLICABLE`, including its own byte and item bounds.

## 23. Provider readiness gate

The generic gate is evaluated per source kind, not by family inference. A
source kind may enter live implementation only when it has:

```text
stable source_kind identity
shared provider integration
bounded typed read primitive
stable remote identity
safe provider references
provider error boundary
secret-free public models
focused core tests
sufficient review status
```

The gate outcome is exactly one of:

```text
READY_FOR_LIVE_ROLLOUT
BLOCKED_BY_CORE_READ_SURFACE
BLOCKED_BY_ADAPTER
BLOCKED_BY_PROVIDER_SEMANTICS
BLOCKED_BY_REVIEW
```

A durable adapter is required only when the live handler needs its accepted
identity, safe normalization, or provider-specific provenance contract.
Incomplete durable materialization alone does not block a valid independent
live read primitive. `BLOCKED_BY_ADAPTER` must identify the missing identity or
normalization dependency; it cannot mean “durable sync is unfinished” by
itself.

Google Workspace is the first explicit family using this gate. Its seven
source kinds are evaluated independently: ready `drive`, `docs`, `sheets`, or
`calendar` work may proceed while `slides`, `mail`, or `chat` remain blocked or
deferred. Google readiness is not evaluated by this architecture task.

## 24. Provider-family rollout requirements

Each later provider/source-kind task must provide:

1. an operation matrix with the four availability statuses for search/list,
   exact read, child/thread read, and content read where applicable;
2. one canonical descriptor/handler identity per supported operation;
3. a strict request model and schema reference;
4. reuse of the already resolved shared integration;
5. bounded upstream page/request/item/time-window behavior;
6. stable remote identity and safe locator behavior;
7. mapping of provider exceptions to the stable taxonomy;
8. normalized result/evidence and no raw provider payload leakage;
9. source-specific tests without real external credentials;
10. a family closeout proving shared integration, source-kind isolation,
    registration, limits, errors, evidence, receipts, and no duplicate clients.

The final cross-provider audit retains stable columns:

```text
provider_id
integration_kind
source_kind
capability_id
contract_version
operation_class
availability_status
request_schema_ref
result_schema_ref
resource_scope
provider_bound
max_pages
max_requests
max_items
max_bytes
timeout_policy
error_mapping
safe_locator_policy
receipt_policy
retention_policy
integration_reuse_evidence
focused_test_evidence
family_closeout_evidence
defer/block reason
```

Every one of the fifteen source kinds must appear as implemented, explicitly
deferred, or blocked with evidence. No family-level “ready” value may replace
source-kind rows.

### 24.1 `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1`

**Status:** `READY_FOR_REVIEW`

**Technical outcome:** implement the shared runtime delta frozen by `ARCH-1`
before any production provider-specific live handler is activated.

**User outcome:** every provider enters live access through the same validated
and tested boundary instead of defining private rules in its first handler.

**Provider-neutral scope:**

```text
descriptor identity/version/source-kind support
strict schema registry/resolution
typed validated call envelope
descriptor-handler-schema registration bundle
atomic registration validation
effective provider-call budgets
shared error taxonomy
normalized provenance additions
safe-locator enforcement
receipt contract and ordered identity-aware hashing
provider-neutral execution context, outcome, item/result and handler contracts
canonical runtime ownership independent of the LKW application tier
shared contract tests
bootstrap integration needed by all provider families
```

**Explicitly out of scope:**

```text
Microsoft Graph Drive/Mail/Teams handlers
Slack, Jira, Confluence or Google Workspace handlers
provider SDK calls or provider-specific request models beyond test fixtures
application UI, LKW behavior, durable synchronization or indexing
provider credentials or provider clients in handlers
```

**Acceptance gates:**

```text
one canonical capability identity parser/validator
source_kind cannot disagree across ID, descriptor and handler
contract version cannot silently change
strict request validation occurs before integration/provider invocation
unknown request fields are rejected
descriptor without handler is unavailable
handler without descriptor fails registration
schema mismatch fails registration
duplicate identities fail closed
registry/catalog publication is atomic
effective budget is the minimum of all applicable limits
provider pages/requests/upstream items/content bytes are finite
executor remains the authoritative timeout/output-limit boundary
provider-neutral errors are stable and secret-free
evidence provenance includes provider/source/capability/version/binding identity
unsafe locators are removed or rejected
receipt hash is ordered and remote-item-identity-aware
EPHEMERAL and RECEIPT_ONLY remain the only live retention modes
live results never enter the indexed corpus automatically
no provider clients or credentials enter handlers
focused shared contract tests pass without external credentials
```

**Review-fix ownership resolution:**

The first production-handler review exposed an invalid dependency direction:
the runtime handler contract previously required application-owned execution
models from `local_workspace_application`. The canonical outcome enum,
execution context, result item, execution result, receipt and async handler
protocol now live under `intergrax/runtime/vendor_knowledge/live/`. The LKW
executor remains the application-owned orchestrator and creates receipts after
validating provider results; it consumes the exact runtime classes and passes
the canonical validated call subtype without exposing application policy or
configuration to handlers.

## 25. Test architecture

No tests are added by `ARCH-1`; the following pyramid is mandatory for later
implementation tasks.

### Shared contract tests

The shared suite proves descriptor/handler identity, duplicate rejection,
missing pair rejection, strict request validation before provider invocation,
scope non-widening, tenant/workspace/binding/audience enforcement, restrictive
budget calculation, timeout/cancellation, error normalization, result
validation, executor truncation, evidence identity, safe locator filtering,
receipt privacy, ordered identity-aware receipt hashing, and retention.

### Provider/source-kind tests

Each source-kind suite proves reuse of the existing integration, the exact
provider primitive, bounded upstream requests, source-specific identity and
mapping, provider error normalization, safe locators, and no credential or raw
SDK payload leakage. Normal unit tests do not require external credentials.

### Family closeout

The family suite proves one shared integration, no duplicate clients, exact
source-kind/resource isolation, consistent registration, limits, errors,
evidence, receipts, retention, and all source-kind statuses.

### Final cross-provider audit

The audit verifies all fifteen accepted rows, every operation availability
status, readiness evidence or block reason, and the stable columns above. It
must explicitly record Google independent gates and Databricks exclusion.

## 26. Mandatory architecture decisions

| decision | chosen rule | rejected alternative | reason | implementation consequence |
|---|---|---|---|---|
| capability identity and source kind | Canonical source kind in capability ID; conceptual identity has five fields | Independent optional source-kind lookup | Preserves exact routing without ambiguous fallback | Validate descriptor/handler source kind against ID |
| capability naming grammar | `vendor.<provider>.<source_kind>.<operation>[.<suboperation>]` | Matrix placeholders or application names | Stable, provider/source explicit, read-only | Separate IDs per semantic operation |
| search versus exact read | Separate bounded `search`/`list`/`read`/child/content identities | One universal `search/read` capability | Prevents false provider semantics | Provider task declares each operation status |
| request model ownership | Shared envelope in Vendor Knowledge; strict source model under live provider package | Raw arbitrary dict or application model | One validation convention and reusable platform contract | Handler accepts validated immutable model |
| result metadata policy | Optional bounded JSON-safe, secret-free, non-authoritative metadata | Raw SDK payload | Protects boundary and ownership | Validate metadata separately |
| resource-scope enforcement | Binding is authoritative; operation cannot widen | Handler trusts request resource IDs | Prevents cross-resource reuse | Scope validator runs before provider call |
| effective budget calculation | Minimum of descriptor, policy, binding, application, hard ceiling, deadline | Handler-selected limits | Caller cannot raise limits | Executor re-enforces output |
| pagination/request bounds | Finite pages, requests, upstream items, page size, and windows | Unbounded local filtering or durable cursor exposure | Keeps one call bounded | Provider boundary stops and marks truncation |
| timeout ownership | One executor deadline and `asyncio.wait_for` | Provider-specific timeout loops/jobs | One cancellation boundary | Handler propagates cancellation |
| error taxonomy | Listed stable public codes; provider details internal/redacted | SDK exceptions or raw responses | Provider-neutral caller contract | Map all provider failures |
| evidence identity | Canonical SHA-256 over full provider/source/binding/call/item tuple | Raw URL or provider item ID alone | Collision-safe provenance | Deterministic `live:` IDs |
| safe locator rules | Optional secret-free URL/opaque/bounded location | Mandatory raw provider URL | URLs can contain grants/secrets | Filter or set null |
| receipt hashing | Ordered, item-identity-aware canonical hash | Content-only or unordered hash | Detects reorder/substitution | Receipt contains no content |
| retention | `EPHEMERAL` or `RECEIPT_ONLY`; durable content deferred | Automatic live indexing | Separates live and durable lifecycles | No corpus write from live result |
| descriptor registration | One atomic Vendor Knowledge bundle/catalog | App-local or LKW module registration | Consistent startup validation | Descriptor without handler unavailable |
| handler registration | One immutable exact registry with version | Per-provider registries/fallbacks | One routing authority | Duplicates and missing pairs fail |
| integration reuse | Inject already resolved integration instance | Handler credentials/client construction | Avoids duplicate clients and secrets | Handler is transport-agnostic |
| contract versioning | Initial v1; version in descriptor/handler/request/result; parallel only explicit | Capability rename or silent upgrade | Safe compatibility | Binding version must match exactly |
| readiness gate | Per-source-kind gate with five allowed outcomes | Family inferred readiness | Google siblings remain independent | Durable adapter required only for identity/normalization dependency |

No mandatory decision in this table is `TBD`. Provider-specific operation
availability remains the documented responsibility of each later source-kind
task.

## 27. Frozen status and acceptance gates

The status after this task is:

```text
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1
  ACCEPTED / CLOSED

VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1
  READY_FOR_REVIEW

VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1
  READY_FOR_REVIEW

VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1-REVIEW-FIX-2
  READY_FOR_REVIEW

MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE
  PLANNED / BLOCKED_BY_SHARED_FOUNDATION

all other provider live tasks
  PLANNED

GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-READINESS-GATE-1
  PLANNED

VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FAMILY-AUDIT-1
  PLANNED
```

Acceptance requires proof that later tasks use:

```text
one shared executor
one shared descriptor/handler registration boundary
one shared receipt contract
provider credentials outside handlers
request validation before provider invocation
non-widening resource scope
restrictive effective budgets
no automatic live-content indexing
independent Google source-kind gates
all fifteen source kinds in the final audit
Databricks excluded until source-kind selection
```

This architecture is ready for review. It is not accepted and it does not
activate any provider implementation.
