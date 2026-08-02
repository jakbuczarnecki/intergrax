# LKW Conversation Context and Audience Isolation Architecture

```text
Task: LKW-CONVERSATION-CONTEXT-ARCH-1
Classification: docs-only architecture and roadmap contract
Status: READY_FOR_REVIEW
```

**Canonical source for:** conversation-to-workspace resolution; personal versus shared conversation isolation; activation policies; memory partitioning; evidence audience restrictions; thread-context handling; relationship between conversational frontend events and knowledge sources; provider-neutral frontend adapter requirements.

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

### 2.3 Conversation Context Binding

A durable LKW-owned binding that determines which product context is active when a message arrives from a particular external conversation.

**Conceptual minimum fields:**

```text
conversation_context_binding_id
tenant_id
conversation_connection_ref
frontend_provider_id
opaque_conversation_ref
audience_mode
workspace_id
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
| `workspace_id` | Resolved before Ask, planning, retrieval or tool execution; never selected from the caller's personal state for a `SHARED` conversation |

This task freezes semantics only. It does **not** implement the Pydantic model or persistence.

**Persistence owner:** LKW application domain — not `slack_companion`, not `SlackConversationChannelIntegration`, not `vendor_knowledge`, not a Slack knowledge adapter.

### 2.4 Conversation Audience Mode

V1 modes:

```text
PERSONAL
SHARED
```

**PERSONAL** — answer visible only inside a private user-to-application conversation.

Typical examples: Slack app DM; private web-chat session; private mobile conversation.

**SHARED** — answer visible to multiple participants of the external conversation.

Typical examples: public channel; private team channel; group conversation; shared project room.

An MPIM or another group DM is always `SHARED`. It must never be treated as several combined personal conversations.

### 2.5 Activation Policy

Provider-neutral activation policies:

```text
ALWAYS
MENTION_ONLY
EXPLICIT_COMMAND
```

**Recommended initial mapping:**

```text
PERSONAL conversation:  ALWAYS
SHARED conversation:      MENTION_ONLY
```

The core LKW contract must not contain Slack-specific terms such as `app_mention`. A Slack adapter may later map Slack `app_mention` events to the provider-neutral `MENTION_ONLY` activation policy.

### 2.6 Thread Context Policy

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

## 3. Effective context resolution

### 3.1 PERSONAL conversation

```text
inbound conversation address
→ active PERSONAL Conversation Context Binding
→ authorized principal
→ personal workspace context
→ personal/shared sources allowed for that principal
→ personal memory partition
→ Ask or validated execution
```

### 3.2 SHARED conversation

```text
inbound conversation address
→ active SHARED Conversation Context Binding
→ fixed shared workspace
→ workspace sources explicitly approved for shared consumption
→ shared conversation/thread context
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

## 4. Workspace audience policy

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

## 5. Evidence isolation

### 5.1 Before model invocation

For a shared conversation, every selected evidence item must satisfy at least:

```text
evidence.tenant_id == binding.tenant_id
evidence.workspace_id == binding.workspace_id
evidence source is active
evidence source is approved for SHARED consumption
evidence does not originate from a PERSONAL memory partition
```

The retrieval or query orchestrator must reject mixed personal/shared evidence before the model receives it.

### 5.2 Before outbound delivery

Before posting a shared answer, validate:

```text
response conversation == inbound bound conversation
response thread == inbound/current thread policy
all citations belong to binding.workspace_id
no personal-memory evidence exists
no personal workspace source exists
```

This is a deterministic guard, not an LLM instruction.

---

## 6. Memory and conversational-state partitioning

### 6.1 PERSONAL partition

**Conceptual identity:**

```text
tenant_id
+ conversation_connection_ref
+ principal_ref
+ opaque_conversation_ref
```

May contain: private user-to-LKW conversation state; personal workspace selection; private short-term conversational memory; private clarification and pending-action state.

### 6.2 SHARED partition

**Conceptual identity:**

```text
tenant_id
+ conversation_connection_ref
+ opaque_conversation_ref
+ optional thread_ref
```

May contain: bounded shared thread context; shared clarification state; shared pending actions when explicitly supported later.

Must **not** contain or reference: personal workspace selection; user DM history; personal memory; private pending actions; private attachments; private connector results.

**Invariant:**

```text
PERSONAL context cannot be resolved from a SHARED conversation.
```

No automatic memory copying between partitions is allowed.

---

## 7. Conversation events are not knowledge sources

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

This principle is provider-neutral. Binding detail: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md) · [`docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md).

---

## 8. Shared attachments

### 8.1 PERSONAL conversation

An explicitly accepted attachment may enter the personal conversation's effective workspace through the existing Knowledge Intake path.

### 8.2 SHARED conversation

An attachment may enter only the workspace bound to the shared conversation. It must never fall back to the uploader's private workspace.

**Supported future policies:**

```text
EXPLICIT_ACCEPTANCE
CONFIGURED_AUTO_ACCEPT
DISABLED
```

**Initial recommended policy:** `EXPLICIT_ACCEPTANCE`.

No provider event automatically becomes durable workspace knowledge unless an explicit accepted policy allows it.

---

## 9. Connections and private user credentials

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

## 10. Provider-neutral adapter boundary

### 10.1 Provider adapter owns

```text
provider event mapping
provider conversation identity mapping
mention/activation signal mapping
outbound delivery
provider acknowledgement and retries
```

### 10.2 LKW core owns

```text
Conversation Context Binding
audience classification
workspace resolution
memory partition selection
evidence isolation
Ask/execution authorization
outbound evidence guard
```

### 10.3 Provider adapter must not

```text
choose a workspace independently
read personal workspace selection for shared channels
merge personal and shared memory
select RAG sources
authorize live provider access
store knowledge configuration
```

---

## 11. Slack as the first adapter and proof

Slack is documented here only as the first concrete adapter. Future adapters may include Microsoft Teams, web chat, mobile chat and other ConversationChannel integrations.

**Future mapping:**

| Slack surface | Audience | Activation |
|---|---|---|
| Slack app DM | `PERSONAL` | `ALWAYS` |
| Slack public channel | `SHARED` | `MENTION_ONLY` |
| Slack private channel | `SHARED` | `MENTION_ONLY` |
| Slack MPIM | `SHARED` | `MENTION_ONLY` (initial product support may remain deferred) |

Slack implementation terms (`app_mention`, `message.im`, `message.channels`, `message.groups`, `message.mpim`) belong in Slack adapter documentation — not in the LKW core contract.

### 11.1 Current implementation honesty

**Implemented today:**

```text
one approved Slack user
direct-message workflow
configured/default personal workspace behavior
DM Ask and existing commands
```

**Not implemented:**

```text
shared Conversation Context Bindings
channel mention activation
shared workspace resolution
shared memory partition
channel attachment routing
shared evidence guard
```

Do not rewrite the existing DM implementation as though it already supports the target architecture.

---

## 12. Relationship with conversational planning

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
├── thread_context_ref
└── allowed product capability boundary
```

The LLM planner must **not** choose: audience mode; shared versus personal memory; conversation workspace binding; source visibility; private-to-shared data elevation.

Those decisions occur deterministically before or after planning according to the operation — never through model discretion.

A natural-language workspace reference may target an operation, but it must not silently replace the workspace bound to a shared conversation. Potential future workspace switching inside a shared channel requires an explicit authorized reconfiguration operation and is outside V1.

Binding detail: [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md).

---

## 13. Required diagrams

### 13.1 Personal conversation

```text
personal conversation
→ PERSONAL Conversation Context Binding
→ authorized principal
→ personal workspace
→ personal memory partition
→ permitted indexed/live evidence
→ answer visible only in personal conversation
```

### 13.2 Shared conversation

```text
shared conversation + activation signal
→ SHARED Conversation Context Binding
→ fixed shared workspace
→ shared memory/thread partition
→ shared-approved indexed/live evidence only
→ deterministic evidence guard
→ answer visible in shared conversation
```

### 13.3 Forbidden crossover

```text
shared conversation
-X→ caller's personal workspace
-X→ caller's DM memory
-X→ caller's private attachments
-X→ caller's private connector grants
```

### 13.4 Independent grants

```text
Conversation Context Binding
    controls where and for whom LKW responds

Indexed Source Binding
    controls durable knowledge ingestion

Live Access Binding
    controls request-time provider reads
```

---

## 14. Roadmap placement

```text
DONE:
SLACK-CONVERSATION-RUNTIME-1
LKW Slack DM frontend foundations
SLACK-KNOWLEDGE-THREE-MODE-ARCH-1
SLACK-KNOWLEDGE-FOUNDATION-1

CURRENT ARCHITECTURE PREREQUISITE:
LKW-CONVERSATION-CONTEXT-ARCH-1
→ provider-neutral personal/shared context and audience-isolation contract

THEN:
LKW-SLACK-CONNECTED-SOURCE-1
→ an approved Slack conversation becomes an Indexed Source of a workspace

LKW-CONVERSATION-CONTEXT-1
→ provider-neutral durable Conversation Context Bindings,
   workspace audience policy, memory partitioning and evidence guards

LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
→ Slack channel/private-channel mention handling over the generic LKW context layer

SLACK-LIVE-CAPABILITY-1
→ bounded request-time Slack reads

LKW-SLACK-KNOWLEDGE-PROOF-1
→ private DM and shared-channel proof with strict audience isolation

AFTER COMPLETE SLACK USER VERTICAL:
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR
```

`LKW-CONVERSATION-CONTEXT-ARCH-1` and `LKW-CONVERSATION-CONTEXT-1` are LKW-wide tasks — not platform Slack provider ownership. `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1` is only the first provider adapter.

---

## 15. Status honesty (post-freeze)

**Architecture frozen for review:**

- provider-neutral personal/shared conversation contexts
- audience-first authorization
- workspace binding
- memory isolation
- evidence isolation
- activation policies
- Slack as first adapter

**Not implemented:**

- Conversation Context Binding persistence
- workspace audience policy
- shared memory store
- channel mention runtime
- Slack channel Ask
- MPIM Ask
- shared attachment intake
- shared evidence enforcement
