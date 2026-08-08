# LKW Conversation Context and Audience Isolation Architecture

```text
Task: LKW-CONVERSATION-CONTEXT-ARCH-1 — ACCEPTED
Classification: docs-only architecture and roadmap contract
Status: ACCEPTED
```

**Canonical source for:** provider-observed conversation audience; conversation-to-workspace resolution; personal versus shared conversation isolation; binding semantic identity; activation policies; conversation-level state versus thread-level memory; evidence audience restrictions; shared capability boundaries; binding administration; shared source eligibility; deterministic guards; relationship between conversational frontend events and knowledge sources; provider-neutral frontend adapter requirements.

**Related:** [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md) · [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md) · [`SLACK_MVP_DISCOVERY.md`](SLACK_MVP_DISCOVERY.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

---

## 1. Core security invariant

**The audience of the outbound answer determines the maximum knowledge scope.**

A shared-conversation answer must never be expanded by the personal permissions, private workspace selection, private memory or private sources of the individual user who invoked the assistant.

**Rejected model:**

```text
caller has access
→ answer may use caller's private knowledge
```

**Required model:**

```text
answer is visible to a shared audience
→ use only knowledge explicitly approved for that shared conversation context
```

This invariant must eventually be enforced deterministically before retrieval and model invocation — not through prompt instructions alone.

---

## 2. Canonical vocabulary

### 2.1 Conversation Surface

A provider-backed conversational environment through which a human interacts with LKW.

**Examples:** Slack workspace installation; Microsoft Teams tenant/application installation; web-chat deployment; mobile application deployment.

A Conversation Surface is **not** a knowledge source by itself.

### 2.2 Conversation Address

Reuse the existing provider-neutral concept represented by:

```text
installation_id
conversation_id
thread_id
```

Do not create a Slack-only replacement. Future durable configuration may store a canonical opaque reference derived from the provider address, but never credentials or raw provider payloads.

### 2.3 ConversationIngressContext

A conceptual provider-neutral ingress observation produced by a frontend adapter when a conversational event arrives. This is a contract only — not a persistence model.

```text
ConversationIngressContext
├── conversation_connection_ref
├── opaque_conversation_ref
├── opaque_thread_ref
├── actor_principal_ref
├── observed_audience
├── activation_signal
└── provider_event_ref
```

**Observed audience** (`observed_audience`) is an ingress observation state:

```text
PERSONAL
SHARED
UNKNOWN
```

| Observed value | Meaning |
|---|---|
| `PERSONAL` | Provider adapter positively confirms that the outbound answer is visible only to the authenticated principal/application conversation |
| `SHARED` | Answer may be visible to multiple human participants |
| `UNKNOWN` | Ingress audience cannot be determined safely |

**Rules:**

```text
observed PERSONAL  → private principal/application conversation only
observed SHARED    → multi-participant conversation
observed UNKNOWN   → fail closed
```

`UNKNOWN` is an ingress observation state — **not** a valid durable binding audience.

The durable binding must match the observed audience:

```text
binding.audience_mode == ingress.observed_audience
```

A mismatch must fail closed **before:**

```text
workspace resolution
memory lookup
planner invocation
retrieval
tool execution
attachment intake
Ask
```

A provider-observed group conversation can **never** resolve a `PERSONAL` binding.

Provider adapters map native events to `observed_audience` and normalized `activation_signal`. The core LKW contract must remain free of Slack event names.

### 2.4 Conversation Context Binding

A durable LKW-owned binding that determines which product context is active when a message arrives from a particular external conversation.

**Conceptual minimum fields:**

```text
conversation_context_binding_id
tenant_id
conversation_connection_ref
frontend_provider_id
opaque_conversation_ref
audience_mode
workspace_resolution_policy
workspace_id                    # required only for FIXED_WORKSPACE
owner_principal_ref
activation_policy
thread_context_policy
administrative_status
configuration_version
created_at
updated_at
```

**Field rules:**

| Field | Rule |
|---|---|
| `owner_principal_ref` | Required for `PERSONAL` audience; absent for `SHARED` audience |
| `opaque_conversation_ref` | Provider-scoped; canonical; secret-free; not interpreted by LKW business logic |
| `workspace_resolution_policy` | `FIXED_WORKSPACE` or `PERSONAL_SELECTION` — see §4 |
| `workspace_id` | Required when `workspace_resolution_policy = FIXED_WORKSPACE`; resolved before Ask, planning, retrieval or tool execution |
| `audience_mode` | Immutable after creation; `PERSONAL` ↔ `SHARED` change requires disable/replace |

This task freezes semantics only. It does **not** implement the Pydantic model or persistence.

**Persistence owner:** LKW application domain — not `slack_companion`, not `SlackConversationChannelIntegration`, not `vendor_knowledge`, not a Slack knowledge adapter.

#### 2.4.1 Binding semantic identity and cardinality

**V1 semantic identity:**

```text
tenant_id
+ conversation_connection_ref
+ opaque_conversation_ref
```

**Required invariant:**

```text
At most one ACTIVE Conversation Context Binding exists for one semantic identity.
```

| Cardinality | Behavior |
|---|---|
| zero active bindings | fail closed |
| more than one active binding | corrupt/ambiguous configuration — fail closed |
| exactly one active binding | proceed to ingress/binding validation |

Additional rules:

- a thread inherits the parent conversation binding;
- no separate thread binding in V1;
- `audience_mode` is immutable after creation;
- changing `PERSONAL` to `SHARED` or `SHARED` to `PERSONAL` requires disabling/replacing the binding;
- changing the bound workspace is a versioned administrative mutation;
- binding reads and mutations are tenant-scoped;
- the referenced conversation connection must be active, tenant-owned and resolvable as a conversational frontend connection;
- disabled/revoked connection or binding fails closed.

Do **not** define last-write-wins behavior.

### 2.5 Conversation Audience Mode

V1 durable binding modes:

```text
PERSONAL
SHARED
```

(`UNKNOWN` exists only on ingress — see §2.3.)

**PERSONAL** — answer visible only inside a private user-to-application conversation.

Typical examples: Slack app DM; private web-chat session; private mobile conversation.

**SHARED** — answer visible to multiple participants of the external conversation.

Typical examples: public channel; private team channel; group conversation; shared project room.

An MPIM or another group DM is always `SHARED`. It must never be treated as several combined personal conversations.

### 2.6 Principal rules

#### PERSONAL binding

Require:

```text
owner_principal_ref is present
ingress.actor_principal_ref == binding.owner_principal_ref
```

A mismatch fails closed.

#### SHARED binding

Require:

```text
owner_principal_ref is absent
actor_principal_ref is still required for authorization, audit and rate limits
```

Caller authorization may narrow the shared capability boundary but may **never** expand:

```text
workspace evidence
source eligibility
live capabilities
conversation audience
```

Bot/system-authored events must not resolve a human Ask context.

### 2.7 Activation Policy and normalized signals

Provider-neutral activation policies:

```text
ALWAYS
MENTION_ONLY
EXPLICIT_COMMAND
```

Normalized activation signals (adapter-mapped):

```text
ORDINARY_MESSAGE
MENTION
EXPLICIT_COMMAND
THREAD_CONTINUATION
```

LKW core evaluates:

```text
binding.activation_policy
against
ingress.activation_signal
```

before any product side effect.

**Recommended initial mapping:**

```text
PERSONAL conversation:  ALWAYS
SHARED conversation:      MENTION_ONLY
```

| Policy | Semantics |
|---|---|
| `ALWAYS` | Accept an authorized ordinary message |
| `MENTION_ONLY` | `MENTION` may open a new assistant thread/session; `THREAD_CONTINUATION` may continue an already established, non-expired assistant thread without another mention; ordinary messages outside an active assistant thread are ignored; continuation must match the same binding and canonical thread; session bounded by time and context limits |
| `EXPLICIT_COMMAND` | Accept only a validated explicit command signal |

Provider adapters map native events to normalized signals. They do **not** independently choose the workspace or evidence scope.

Transport acknowledgement may occur before activation evaluation, but no Ask, planner, intake, retrieval or mutation may occur before the policy is satisfied.

The core LKW contract must not contain Slack-specific terms such as `app_mention`.

### 2.8 Thread Context Policy

V1 policy:

```text
CURRENT_THREAD_BOUNDED
```

Meaning:

- the external conversation binding selects the workspace;
- the thread inherits the parent conversation binding;
- a thread does not independently change the workspace in V1;
- only a bounded amount of current-thread conversation context may be supplied;
- the whole channel history is not automatically copied into the model prompt;
- channel history is available as knowledge only through a separately approved Indexed Source or Live Access Binding.

---

## 3. Workspace resolution policy

Replace ambiguous single behavior with a provider-neutral workspace resolution policy:

```text
FIXED_WORKSPACE
PERSONAL_SELECTION
```

### 3.1 SHARED binding

Required:

```text
workspace_resolution_policy = FIXED_WORKSPACE
workspace_id is required
```

The caller cannot replace the shared workspace through personal selection or a natural-language reference.

### 3.2 PERSONAL binding

Allow:

```text
workspace_resolution_policy = PERSONAL_SELECTION
```

and optionally:

```text
FIXED_WORKSPACE
```

For `PERSONAL_SELECTION`:

- the selected workspace state is durable;
- it is keyed by `tenant_id` + `conversation_context_binding_id` + `owner_principal_ref`;
- it must reference an authorized workspace;
- missing selection may use an explicitly configured safe default or require user selection;
- no process-global or deployment-global active workspace fallback;
- changing selection must be an explicit authorized product operation;
- selection does not change another personal conversation automatically.

**Current implementation honesty:** the legacy exact-command Slack selection
state (`LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID` / companion selection state)
remains temporary. The conversational interaction path resolves its authoritative
workspace context before loading durable thread memory; migrating the legacy
selection state is outside `LKW-CONVERSATION-CONTEXT-1C`.

---

## 4. Effective context resolution

### 4.1 Ingress-to-binding validation (all conversations)

```text
inbound ConversationIngressContext
→ resolve active unique binding for semantic identity
→ validate active conversational connection
→ validate binding.audience_mode == ingress.observed_audience
→ validate principal rules
→ evaluate activation_policy against activation_signal
→ resolve workspace per workspace_resolution_policy
→ select conversation-level state and thread-level memory partition
→ construct allowed_product_capabilities
→ permitted indexed/live evidence for audience
→ Ask or validated execution
```

### 4.2 PERSONAL conversation

```text
inbound conversation address
→ observed PERSONAL + active PERSONAL binding (principal match)
→ authorized principal
→ personal workspace (FIXED_WORKSPACE or PERSONAL_SELECTION)
→ personal/shared sources allowed for that principal and audience eligibility
→ personal thread memory partition
→ Ask or validated execution
```

### 4.3 SHARED conversation

```text
inbound conversation address
→ observed SHARED + active SHARED binding
→ fixed shared workspace (FIXED_WORKSPACE only)
→ workspace sources explicitly approved for shared consumption (SHARED_ALLOWED)
→ shared thread memory partition
→ READ_ONLY_ASK capability boundary
→ Ask or validated execution
```

For a shared conversation, **ignore:**

```text
caller's private active workspace
caller's DM workspace selection
caller's personal conversation history
caller's private memory
caller's private attachments
caller's private source bindings
caller's personal connector grants
```

There must be **no fallback** from a missing shared binding to the caller's personal workspace. Missing, disabled or invalid shared binding must **fail closed**.

---

## 5. Workspace audience policy

Not every workspace can safely be exposed in a shared conversation. LKW requires an explicit workspace audience policy or equivalent classification.

**Conceptual V1 modes:**

```text
PERSONAL
SHARED
```

**Rules:**

```text
PERSONAL workspace:
- may be used by an authorized personal context;
- must not be bound to a shared conversation.

SHARED workspace:
- may be bound to an approved shared conversation;
- all sources exposed to shared Ask must be intentionally approved for the shared audience.
```

Do not solve this with a prompt instruction. The restriction must eventually be enforced before retrieval and model invocation.

---

## 6. Conversation-level state versus thread-level memory

Freeze two different stores/concepts.

### 6.1 Conversation-level state

**Conceptual identity:**

```text
tenant_id
+ conversation_context_binding_id
+ principal_ref when PERSONAL
```

May contain:

```text
personal workspace selection
binding-level preferences
safe rate-limit or session metadata
```

Must **not** contain unbounded conversation transcripts.

### 6.2 Thread-level memory

**Conceptual identity:**

```text
tenant_id
+ conversation_context_binding_id
+ canonical_thread_ref
```

Required for both `PERSONAL` and `SHARED` conversational memory.

**Rules:**

- two threads never share short-term conversational memory;
- a provider thread inherits the conversation binding;
- provider adapter creates or maps a canonical opaque thread reference;
- for a provider without native threads, the adapter supplies an explicit conversation-session reference;
- missing/invalid thread identity fails closed when thread memory is requested;
- bounded limits must be configurable by maximum messages, bytes and age;
- no automatic copying between threads;
- no automatic copying between `PERSONAL` and `SHARED` partitions.

**Invariant:**

```text
PERSONAL context cannot be resolved from a SHARED conversation.
```

### 6.3 Runtime integration status

`LKW-CONVERSATION-CONTEXT-1B2` is **ACCEPTED / CLOSED**.
`LKW-CONVERSATION-CONTEXT-1C` is **READY_FOR_REVIEW**.

The accepted personal Slack conversational flow now resolves the authoritative
Conversation Execution Context before loading thread memory. The application
passes only bounded `user`/`assistant` recent turns to the planner and persists
one safe user/assistant exchange after rendering. The durable lifecycle uses the
shared conditional `DocumentStore`, with a bounded exchange-identity marker and
the existing event receipt coordinating recovery and delivery retries.

Default limits are conservative and configuration-driven:

```text
max_messages: 20
max_bytes: 16 KiB
max_age_seconds: 24 hours
```

Absent, expired or over-budget history becomes an empty `recent_turns` tuple.
Malformed or unavailable durable memory fails closed before planning. The
generic partition remains valid for shared execution contexts, but shared Slack
routing and activation are outside this slice.

---

## 7. Shared capability boundary (V1)

V1 shared conversations default to:

```text
READ_ONLY_ASK
```

Ordinary shared mention/continuation must **not** perform:

```text
workspace create/delete/archive
workspace activation or rebinding
Conversation Context Binding mutation
Connection attachment or administration
Indexed Source creation/removal
Live Access Binding creation/removal
query-policy mutation
source approval mutation
automatic attachment acceptance
credential or provider administration
```

Shared administrative changes require a separately authorized administration path.

```text
shared participant identity
→ may invoke only capabilities allowed by the binding/context policy
→ may further be restricted by principal authorization
→ never gains administrative capability merely by being a channel member
```

`allowed_product_capabilities` in `ConversationExecutionContext` must be deterministically constructed — not generated by the model.

For V1, planner/executor in shared context must fail closed on actions outside the allowed capability set.

---

## 8. Binding administration

Conversation Context Binding creation, workspace rebinding, audience replacement and shared-source approval must require an authorized tenant/workspace administrator or equivalent explicit product role.

An ordinary frontend message must **not** mutate its own authorization boundary.

Administrative mutations require:

- administrative actor identity;
- idempotency;
- expected configuration version / CAS;
- audit receipt;
- safe disable/revoke behavior.

The binding must contain:

- no credentials;
- no provider payload.

Exact persistence schemas remain deferred to `LKW-CONVERSATION-CONTEXT-1`.

---

## 9. Shared source eligibility

Freeze an explicit shared eligibility concept for both durable and live knowledge:

```text
PERSONAL_ONLY
SHARED_ALLOWED
```

The exact future model may live on the workspace Indexed Source Binding and Live Access Binding.

**Required rules:**

```text
workspace audience = SHARED
AND
source/binding eligibility = SHARED_ALLOWED
→ eligible for shared evidence
```

Otherwise excluded.

| Rule | Value |
|---|---|
| default | `PERSONAL_ONLY` |
| existing sources | not silently promoted |
| approval scope | tenant- and workspace-scoped |
| Indexed Source vs Live Access | independent approvals |
| caller private entitlement | cannot promote eligibility |
| eligibility change | administrative mutation |
| citations and live evidence | retain source/binding eligibility used for authorization |

**Consequence:**

```text
When multiple SHARED conversations are bound to the same workspace,
every SHARED_ALLOWED source of that workspace is potentially available
to all of those conversations.
```

A team needing different knowledge boundaries must use separate workspaces or a future finer-grained audience policy.

---

## 10. Evidence isolation and deterministic guards

### 10.1 Before planner / retrieval / model

Validate:

```text
active unique binding
active conversational connection
observed audience matches binding
principal rules
activation policy satisfied
workspace resolution succeeded
workspace audience compatible
allowed product capability
thread partition identity
every indexed/live/memory item matches tenant + workspace + audience eligibility
```

The guard applies to all model inputs:

```text
retrieved evidence
live tool results
thread memory
planner context
attachment-derived content
system-added context
```

not only citations.

For a shared conversation, every selected evidence item must satisfy at least:

```text
evidence.tenant_id == binding.tenant_id
evidence.workspace_id == binding.workspace_id
evidence source is active
evidence source eligibility = SHARED_ALLOWED
evidence does not originate from a PERSONAL memory partition
```

The retrieval or query orchestrator must reject mixed personal/shared evidence before the model receives it.

### 10.2 Before outbound delivery

Validate:

```text
outbound conversation matches binding
outbound thread matches current canonical thread
answer audience remains unchanged
all citations and evidence receipts match tenant/workspace
all sources are eligible for the audience
no PERSONAL memory or connector result entered a SHARED answer
```

A guard failure must suppress the outbound answer and produce a safe internal failure result.

This is a deterministic guard, not an LLM instruction.

---

## 11. Conversation events are not knowledge sources

```text
conversational frontend event
!=
indexed knowledge source
!=
live knowledge permission
```

**Independent grants:**

| Grant | Controls |
|---|---|
| **Conversation Context Binding** | Where and under which audience the assistant may respond |
| **Indexed Source Binding** | Which provider content may be durably synchronized into RAG |
| **Live Access Binding** | Which provider capabilities may be used at request time |

None implies another.

**Examples:**

```text
bot enabled in a project channel
does not imply
channel history is indexed

channel history indexed
does not imply
bot may answer in that channel

bot may answer in channel
does not imply
live Slack history reads are allowed
```

This principle is provider-neutral. Binding detail: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md) · [`docs/project/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md).

---

## 12. Shared attachments

### 12.1 PERSONAL conversation

An explicitly accepted attachment may enter the personal conversation's effective workspace through the existing Knowledge Intake path.

### 12.2 SHARED conversation

An attachment may enter only the workspace bound to the shared conversation. It must never fall back to the uploader's private workspace.

**Supported future policies:**

```text
EXPLICIT_ACCEPTANCE
CONFIGURED_AUTO_ACCEPT
DISABLED
```

**Initial recommended policy:** `EXPLICIT_ACCEPTANCE`.

No provider event automatically becomes durable workspace knowledge unless an explicit accepted policy allows it. V1 shared `READ_ONLY_ASK` does not include automatic attachment acceptance.

---

## 13. Connections and private user credentials

```text
A user's private Connection or provider entitlement does not automatically
become available to a shared conversation.
```

For shared use, the workspace must reference a connection/resource grant explicitly approved for the shared audience.

**Example:**

```text
Artur can personally access an entire Figma organization
```

does **not** imply:

```text
the UX shared conversation can query the entire Figma organization
```

The shared workspace may use only an explicitly approved Figma project, resource or capability.

Individual caller permissions may restrict a shared action further, but may **never** expand the shared evidence boundary.

```text
effective shared authorization
≠ shared permissions UNION caller private permissions

effective shared authorization
= shared approved permissions
  optionally narrowed by caller authorization
```

---

## 14. Provider-neutral adapter boundary

### 14.1 Provider adapter owns

```text
provider event mapping
provider conversation identity mapping
observed_audience determination
mention/activation signal mapping
canonical thread reference mapping
outbound delivery
provider acknowledgement and retries
```

### 14.2 LKW core owns

```text
Conversation Context Binding
ingress/binding audience validation
audience classification
workspace resolution
conversation-level state and thread memory partition selection
evidence isolation
allowed_product_capabilities construction
Ask/execution authorization
outbound evidence guard
```

### 14.3 Provider adapter must not

```text
choose a workspace independently
read personal workspace selection for shared channels
merge personal and shared memory
select RAG sources
authorize live provider access
store knowledge configuration
expand shared capability boundary
```

---

## 15. Slack as the first adapter and proof

Slack is documented here only as the first concrete adapter. Future adapters may include Microsoft Teams, web chat, mobile chat and other ConversationChannel integrations.

**Observed audience mapping (adapter examples only):**

| Slack surface | Observed audience | Activation |
|---|---|---|
| Slack IM / app DM | `PERSONAL` | `ALWAYS` |
| Slack public channel | `SHARED` | `MENTION_ONLY` |
| Slack private channel | `SHARED` | `MENTION_ONLY` |
| Slack MPIM / group DM | `SHARED` | `MENTION_ONLY` (initial product support may remain deferred) |

Slack implementation terms (`app_mention`, `message.im`, `message.channels`, `message.groups`, `message.mpim`) belong in Slack adapter documentation — not in the LKW core contract.

### 15.1 Current implementation honesty

**Implemented today:**

```text
one approved Slack user
DM-only workflow
authoritative Conversation Context resolution before interaction planning
bounded durable thread memory and receipt-coordinated exchange persistence
legacy exact-command selection remains temporary
existing Ask and command behavior
```

**Not implemented:**

```text
observed-audience validation
durable personal selection
shared-channel runtime
shared Slack thread-memory transport
shared capability enforcement
shared source eligibility
mention/thread-continuation runtime
Slack history indexing
live Slack Ask
```

Do not rewrite the existing DM implementation as though it already supports the target architecture.

---

## 16. Relationship with conversational planning

Future planner, resolver and executor requests receive a deterministic pre-resolved context envelope.

**Conceptual input:**

```text
ConversationExecutionContext
├── tenant_id
├── audience_mode
├── workspace_id
├── principal_ref
├── conversation_context_binding_id
├── activation_policy
├── canonical_thread_ref
└── allowed_product_capabilities
```

The LLM planner must **not** choose: audience mode; shared versus personal memory; conversation workspace binding; source visibility; private-to-shared data elevation; allowed product capabilities.

Those decisions occur deterministically before or after planning according to the operation — never through model discretion.

A natural-language workspace reference may target an operation, but it must not silently replace the workspace bound to a shared conversation. Potential future workspace switching inside a shared channel requires an explicit authorized reconfiguration operation and is outside V1.

Binding detail: [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md).

---

## 17. Required diagrams

### 17.1 Personal conversation

```text
personal conversation + observed PERSONAL
→ active unique PERSONAL binding (principal match)
→ authorized principal
→ workspace (FIXED_WORKSPACE or PERSONAL_SELECTION)
→ conversation-level state + thread memory partition
→ permitted indexed/live evidence
→ answer visible only in personal conversation
```

### 17.2 Shared conversation

```text
shared conversation + observed SHARED + activation signal
→ active unique SHARED binding
→ fixed shared workspace
→ shared thread memory partition
→ SHARED_ALLOWED indexed/live evidence only
→ READ_ONLY_ASK capability boundary
→ deterministic evidence guard
→ answer visible in shared conversation
```

### 17.3 Forbidden crossover

```text
shared conversation
-X→ caller's personal workspace
-X→ caller's DM memory
-X→ caller's private attachments
-X→ caller's private connector grants
-X→ UNKNOWN or mismatched observed audience
```

### 17.4 Independent grants

```text
Conversation Context Binding
    controls where and for whom LKW responds

Indexed Source Binding
    controls durable knowledge ingestion

Live Access Binding
    controls request-time provider reads
```

### 17.5 State versus memory

```text
conversation-level state
  tenant + binding + principal (PERSONAL)
  → workspace selection, preferences, session metadata

thread-level memory
  tenant + binding + canonical_thread_ref
  → bounded conversational memory (PERSONAL and SHARED)
```

---

## 18. Roadmap placement

```text
DONE:
SLACK-CONVERSATION-RUNTIME-1
LKW Slack DM frontend foundations
SLACK-KNOWLEDGE-THREE-MODE-ARCH-1
SLACK-KNOWLEDGE-FOUNDATION-1

CURRENT ARCHITECTURE PREREQUISITE:
LKW-CONVERSATION-CONTEXT-ARCH-1 — ACCEPTED

THEN (independent implementation tracks):
LKW-SLACK-CONNECTED-SOURCE-1 — IN_PROGRESS / CHANGES_REQUIRED (REVIEW-FIX-2 — CHANGES_REQUIRED; REVIEW-FIX-3 not accepted)
→ an approved Slack conversation becomes an Indexed Source of a workspace
  (independent from conversational activation)

LKW-CONVERSATION-CONTEXT-1 — NEXT
→ provider-neutral durable Conversation Context Bindings,
   workspace audience policy, memory partitioning and evidence guards
  (LKW-wide prerequisite for shared adapters — not a competing planner/executor)

LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
→ Slack channel/private-channel mention handling over the generic LKW context layer

SLACK-LIVE-CAPABILITY-1
→ bounded request-time Slack reads

JOIN (final Slack proof — all prerequisites):
LKW-SLACK-CONNECTED-SOURCE-1
+ LKW-CONVERSATION-CONTEXT-1
+ LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
+ SLACK-LIVE-CAPABILITY-1
+ LKW-HYBRID-ASK-1
→ LKW-SLACK-KNOWLEDGE-PROOF-1
→ private DM and shared-channel proof with strict audience isolation
   (cannot claim indexed + live combined evidence before Hybrid Ask exists)

AFTER COMPLETE SLACK USER VERTICAL:
GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1
→ Google proof-critical read surfaces and adapters
→ LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1
→ LKW-GOOGLE-WORKSPACE-PROOF-1
→ remaining Google surfaces (Slides, Mail, Chat)
→ MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR
```

`LKW-CONVERSATION-CONTEXT-ARCH-1` and `LKW-CONVERSATION-CONTEXT-1` are LKW-wide tasks — not platform Slack provider ownership. `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1` is only the first provider adapter.

---

## 19. Status honesty (post-freeze)

**Architecture frozen for review:**

- provider-neutral personal/shared conversation contexts
- provider-observed audience validation
- binding semantic identity and cardinality
- audience-first authorization
- workspace resolution policies
- conversation-level state versus thread-level memory isolation
- evidence isolation and deterministic guards
- activation policies and normalized signals
- shared `READ_ONLY_ASK` capability boundary
- binding administration and shared source eligibility
- Slack as first adapter

**Not implemented:**

- Conversation Context Binding persistence
- observed-audience validation runtime
- durable personal workspace selection
- workspace audience policy enforcement
- shared Slack memory routing
- shared capability enforcement
- shared source eligibility runtime
- channel mention / thread-continuation runtime
- Slack channel Ask
- MPIM Ask
- shared attachment intake
- shared evidence enforcement runtime
- Slack history indexing
- live Slack Ask

This architecture is ACCEPTED. The bounded personal thread-memory runtime
integration is `LKW-CONVERSATION-CONTEXT-1C`; shared Slack routing and
administrative Conversation Context runtime remain outside this slice.
