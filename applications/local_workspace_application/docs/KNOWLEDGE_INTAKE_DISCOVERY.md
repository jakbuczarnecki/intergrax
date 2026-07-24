# LKW Knowledge Intake and asynchronous ingestion discovery

**Status:** `DOCUMENTED / READY_FOR_REVIEW`  
**Task:** `LKW-WORKSPACE-CONTENTS-1B-0`  
**Classification:** docs-only architecture and product contract  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Implementation plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
**Slack adapter contract:** [`SLACK_MVP_DISCOVERY.md`](SLACK_MVP_DISCOVERY.md)

---

## 1. Decision summary

| Decision | Classification | Binding statement |
|----------|----------------|-------------------|
| LKW / Intergrax owns the complete knowledge intake and ingestion lifecycle | `FROZEN` | Upload acceptance, registration, durable operation state, queue dispatch, extract/parse/chunk/embed, Document/Vector/Blob persistence, retry classification, status events, idempotency, tenant/workspace isolation belong to the platform product boundary — not to any chat client. |
| Slack (and other channels) are replaceable frontends only | `FROZEN` | Frontends collect channel-native input, invoke the same public LKW capabilities, and display accepted/progress/completion states. They contain no ingestion, RAG, storage or provider logic. |
| Operation-based asynchronous contract for every ingestion | `FROZEN` | Every Knowledge Input yields a durable Ingestion Operation. Small files may finish quickly internally; the observable product contract remains operation-based. No sync-for-small / async-for-large product split. |
| Durable Ingestion Operation is the source of truth | `FROZEN` | Operation state must not rest solely on a Slack thread, in-memory future, pub/sub event, or open HTTP connection. |
| Queue / worker boundary required for ingestion execution | `FROZEN` | Acceptance creates durable state and queues work; a worker performs parse → documents → chunks → embeddings → persistence. |
| Reuse existing Intergrax queue/message-bus/outbox when verified sufficient | `REUSE` | Do not create an LKW-specific queue framework merely for this feature. Classify a concrete platform gap only after implementation audit. Do **not** claim the current platform queue is production-durable unless verified by code and tests. |
| Pub/sub (or event bus) for fan-out / notification only | `FROZEN` | Events notify adapters; they are not the operation store. |
| LKW core does not call Slack | `FROZEN` | Completion reaches Slack only through a channel-neutral lifecycle event and a notification/correlation adapter. |
| Managed file / managed file batch | `FROZEN` | Uploaded bytes land under LKW-managed storage policy; Slack attachments map to this path after the core capability exists. |
| Uploaded folder snapshot ≠ connected local folder | `FROZEN` | Channel-exposed folder/archive is a one-time copied snapshot with no live sync. Connected folder remains connector-backed and resynchronizable. |
| Connected source via safe Source Candidate | `FROZEN` | Remote chat clients select opaque `candidate_id` + safe label only; never a raw filesystem path. |
| Explicit web URL intake | `DIRECTION` | Allowed as a Knowledge Input kind; requires SSRF/egress/private-network policy before acceptance. Ordinary Ask messages containing URLs must not auto-ingest. |
| Raw local path typed into Slack (or equivalent remote chat) | `REJECTED` | Forbidden product contract (path disclosure, ambiguous host, no FS guarantee, deployment-neutrality violation). |
| Exact HTTP route / class / enum names | `DEFERRED` | Contract vocabulary only; not implementation claims. |
| Queue / Blob / pub-sub vendor selection | `DEFERRED` | No Kafka, Google Pub/Sub, RabbitMQ, or Blob provider frozen here. |
| Exact Slack command syntax and Block Kit design | `DEFERRED` | Adapter mapping rules only; commands not frozen. |
| Numeric size/batch/retry/progress limits | `DEFERRED` | Policy gates required at implementation; numbers not frozen here. |

---

## 2. Problem and product goal

Users need to introduce knowledge into a workspace from many surfaces: Slack attachments, web uploads, desktop folder pickers, connected drives, and explicit URLs. Historically, LKW’s first vertical slice used **local-folder** sources and a developer-oriented `metadata.source_paths` path. That remains a valid current development mechanism, but it is **not** the target channel-neutral product contract.

**Product goal (FROZEN ARCHITECTURAL CONTRACT):**

```text
channel-native user input
→ public LKW Knowledge Intake capability
→ durable accepted Ingestion Operation
→ queue/worker processing
→ Document Store + Vector Store (+ optional managed originals)
→ channel-neutral lifecycle event
→ response adapter (Slack thread / web / mobile / …)
→ Ask and grounded results over tenant/workspace-scoped knowledge
```

One-sentence result: LKW owns durable asynchronous ingestion; Slack is only one replaceable frontend adapter.

---

## 3. Platform vs frontend boundary

### Platform / LKW owns (`FROZEN`)

- upload acceptance and Upload Session lifecycle;
- source/reference registration;
- durable Ingestion Operation state;
- queue dispatch and worker execution;
- extraction, parsing, document creation, chunking, embeddings;
- Document Store, Vector Store, and managed-original (Blob/Object Store) persistence;
- retry and failure classification;
- channel-neutral status events;
- idempotency;
- tenant/workspace isolation.

### Frontend owns (`FROZEN`)

- collecting user intent;
- collecting artifacts available through that frontend;
- mapping them into a public LKW request;
- displaying safe status and results.

### Slack must not (`FROZEN` / REJECTED if claimed)

- parse documents; create chunks or embeddings;
- write directly to Document Store or Vector Store;
- select a storage provider; access Qdrant directly;
- execute filesystem operations against user paths;
- implement a second ingestion pipeline;
- call an LLM for ingestion;
- become the source of truth for operation state;
- receive or display full local paths for connected sources.

### Channel neutrality (`FROZEN`)

The same LKW Knowledge Intake capability must be invokable from Slack, web, desktop, mobile, Microsoft Teams, Telegram, MCP, CLI, HTTP clients, and future conversation adapters.

Do **not** define separate product APIs such as `/slack-upload`, `/teams-upload`, or `/telegram-source`. Channel-specific transport belongs to the adapter.

The LKW core must **not** contain branches such as `if channel == "slack"`.

```text
Slack / Web / Mobile / Desktop / Teams / Telegram / MCP
                         |
                         v
                 LKW Knowledge Intake
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       Upload         Connector       Web fetch
          |              |              |
          +--------------+--------------+
                         |
                         v
                Ingestion Operation
                         |
                         v
                  Queue / Worker
                         |
                         v
          Parse → Documents → Chunks → Embeddings
                         |
              +----------+----------+
              v                     v
        Document Store          Vector Store
```

---

## 4. Canonical vocabulary

Use these terms consistently. Do not treat them as synonyms.

| Term | Meaning |
|------|---------|
| **Knowledge Intake** | The LKW capability that accepts Knowledge Inputs and drives the ingestion lifecycle. |
| **Knowledge Input** | A channel-neutral request to introduce knowledge into a workspace. Not automatically a durable resynchronizable Source. |
| **Source** | A durable logical origin of knowledge associated with a workspace (connector-backed, managed-upload-backed, web-resource-backed, or future application-feed-backed). **Not** defined as “a local filesystem path”. |
| **Document** | A processed knowledge unit created from a Source or Knowledge Input. One Source may produce one or many Documents, including updates across later sync runs. |
| **Ingestion Operation** | Durable execution record for accepting/processing a Knowledge Input or synchronizing a Source. Source of truth for execution state. |
| **Upload Session** | Temporary transfer boundary for managed bytes before ingestion begins. |
| **Intake Batch** | Logical group of multiple inputs submitted together (per-item results, partial success, safe summary). |
| **Source Candidate** | Safe preconfigured source option exposed to a frontend (`opaque candidate identity`, safe label, kind/type, optional description, tenant scope, availability). Must not encode path, credentials, or provider locator. |
| **Conversation Correlation** | Channel delivery metadata linking an operation to a future response destination (Slack thread, Teams conversation, websocket session, mobile notification target). Must not become part of Source identity or ingestion domain behavior. |

Required Ingestion Operation lifecycle vocabulary (`FROZEN`):

```text
accepted → queued → processing → completed | failed
```

Cancellation is **DEFERRED** and must not be presented as implemented.

---

## 5. Supported Knowledge Input classes

Contract vocabulary only. Exact enum/class names are **not** implementation claims. Unsupported providers must not be described as implemented.

| Input kind | Meaning | Managed original | Resynchronizable | Slack support direction |
|------------|---------|-----------------:|-----------------:|-------------------------|
| `managed_file` | One uploaded file copied under LKW-managed storage policy | Yes | No, unless later replaced | Native attachment |
| `managed_file_batch` | Several uploaded files submitted as one user action | Yes | No | Multiple attachments |
| `uploaded_folder_snapshot` | One-time snapshot of files supplied as archive or channel-exposed collection | Yes | No | Folder/ZIP/multiple files when the channel exposes them |
| `source_candidate` | Reference to a preconfigured connector-backed source | Provider-dependent | Usually yes | Safe numbered selection |
| `web_url` | Explicit fetchable web resource | Policy-dependent | Policy-dependent | Explicit URL intake |

**local-folder** remains the first **implemented** source provider / connector slice. That does not redefine Source as a path.

---

## 6. File, batch, folder and URL semantics

### 6.1 File attachment (Slack direction — PLANNED after core)

```text
Slack adapter
→ acknowledges the Slack envelope immediately
→ authorizes the user and workspace
→ downloads the file using Slack integration credentials
→ streams/uploads bytes to the LKW upload capability
→ never sends the Slack token into LKW core
→ never stores the temporary Slack file URL as the durable Source locator
```

LKW receives only safe transfer data such as: byte stream or finalized upload reference; safe filename; media type; size; checksum; optional safe user label; tenant/workspace context; idempotency identity. Exact request fields are **DEFERRED**.

### 6.2 Multiple attachments

```text
one user submission
→ several item-level inputs/operations
→ independent success/failure
→ safe aggregate summary
```

Do not require a long-running synchronous Slack request.

### 6.3 Folder distinction (`FROZEN`)

```text
uploaded folder
→ snapshot copied into managed storage
→ no live synchronization

connected local folder
→ connector remains attached to original location
→ future synchronization possible
→ Slack sees only safe candidate identity and label
```

A folder dragged or uploaded through a conversation interface is a snapshot only. Do not claim every Slack client natively supports folder drag-and-drop; document the mapping only when the channel exposes the content.

Connected local folders are registered via local-capable surfaces (desktop/tray picker, local CLI, local MCP, local authenticated HTTP, host configuration). Slack may later select `candidate_id` + safe label only.

### 6.4 Raw local path in remote chat — REJECTED

Explicitly prohibited:

```text
source add C:\Users\...\Documents\Contracts
source add /home/user/docs
```

Reasons: path disclosure in channel history; ambiguous target host; no guarantee the LKW host can access the path; remote/mobile clients do not share the filesystem; provider-specific details would leak into the frontend contract; unsafe filesystem-access surface; deployment-neutrality violation.

A local path may be accepted only by a trusted local-capable interface and converted behind the LKW boundary into a safe candidate/reference.

### 6.5 URL intake (`DIRECTION`)

- URL ingestion must be an **explicit** user action.
- Ordinary Ask messages containing a URL must **not** automatically trigger ingestion.
- Exact Slack command syntax is **DEFERRED**.
- Public web URLs and connector-authenticated resources are different cases.
- Credentials must never be embedded in user-visible URLs.
- Authenticated Drive/SharePoint/private-system resources should use a connector/Source Candidate rather than raw credentials or credential-bearing URLs.
- Implementation requires SSRF, redirect, egress and private-network access policy before acceptance.

---

## 7. Upload and asynchronous ingestion lifecycle

### Always operation-based (`FROZEN`)

```text
Knowledge Input accepted
→ durable operation created
→ operation identity returned
→ work queued
→ processing continues asynchronously
```

Do **not** document: small file = synchronous endpoint; large file = asynchronous endpoint.

### Upload and ingestion are separate phases (`FROZEN`)

For managed bytes:

```text
create upload session
→ transfer bytes
→ finalize upload
→ create Knowledge Input
→ create Ingestion Operation
→ queued processing
```

Large transfers may require resumable upload, checksums, transfer retry, upload expiration, and size policy. Exact endpoints, chunk sizes and provider behavior are **DEFERRED**.

---

## 8. Durable operation, queue and worker roles

### Durable operation (`FROZEN`)

Conceptually scoped by:

```text
tenant_id
workspace_id
operation_id
input/source identity
status
timestamps
safe error classification
```

Progress fields are allowed directionally; exact progress granularity is **DEFERRED**.

### Queue and worker (`FROZEN` direction; implementation status not claimed)

```text
Knowledge Intake service
→ durable operation record
→ platform queue/message bus capability
→ ingestion worker
→ parser/chunker/embedder/storage
```

Suitable for large files, many files, batches, retries, backpressure, embedding throttling, process restarts, worker concurrency, and failure isolation.

**REUSE rule:** reuse an existing Intergrax queue/message-bus/outbox capability when verified sufficient; do not create an LKW-specific queue framework merely for this feature; classify a concrete platform gap only after implementation audit.

Do **not** claim a production durable worker already exists. Exact co-location vs separate-process deployment is **not** frozen. File watcher remains an **optional** producer of ingest work, not the definition of Knowledge Intake.

---

## 9. Events, pub/sub and notification correlation

Pub/sub or an event bus is used for **fan-out and notification**, not as the source of truth.

Conceptual events (names/schemas **DEFERRED**):

```text
ingestion accepted
ingestion started
ingestion progressed
ingestion completed
ingestion failed
```

Potential consumers: Slack notification adapter; websocket gateway; mobile push adapter; Teams adapter; audit/trace; metrics.

Events must be tenant/workspace scoped; idempotent for consumers; safe for retries and duplicate delivery; free of raw credentials and sensitive provider locators.

### Notification boundary (`FROZEN`)

```text
durable operation state
        |
        v
channel-neutral lifecycle event
        |
        v
notification/correlation adapter
        |
        +--> Slack thread
        +--> web socket
        +--> mobile push
        +--> Teams conversation
```

The LKW core must **not** call Slack directly. Slack thread is only a response destination — not a queue, operation store, retry mechanism, or ingestion state.

---

## 10. Slack adapter mapping

**Historical note (preserve):** The original MVP-3 / Ask-only Slack slice intentionally ignored file events. That exclusion is **not** a permanent prohibition on Slack attachments.

**Future attachment handling** is governed by this channel-neutral Knowledge Intake contract and must be implemented only after the LKW core capability exists (`1B-1`+). Exact Slack commands and Block Kit design remain **DEFERRED**.

| Channel-native input | Maps to | Notes |
|----------------------|---------|-------|
| Single file attachment | `managed_file` | Adapter downloads with Slack credentials; uploads into LKW; never passes Slack token into core |
| Multiple attachments | `managed_file_batch` / Intake Batch | Per-item success/failure; safe aggregate summary |
| Folder / ZIP / channel-exposed collection | `uploaded_folder_snapshot` | Snapshot only; no live sync |
| Safe numbered candidate selection | `source_candidate` | Opaque id + safe label; never full path |
| Explicit URL intake action | `web_url` | Not automatic from ordinary Ask text |
| Raw filesystem path command | — | `REJECTED` |

Slack acknowledges transport immediately; long transfer/ingestion does not keep the Slack request open. Slack adapter does not parse/embed/store. Completion returns through channel-neutral lifecycle event + Conversation Correlation.

---

## 11. Storage boundaries

| Concern | Owner/store |
|---------|-------------|
| External connected source | Source Connector / external system |
| Managed uploaded original | Blob/Object Store capability |
| Source, document and operation metadata | Document Store |
| Chunks/embeddings/search index | Vector Store |
| Temporary transfer data | Upload/session provider |
| Channel correlation | Conversation notification/correlation storage |

Do **not** claim Document Store automatically stores original file bytes. Do **not** claim Vector Store stores the source document.

A deployment may use local, cloud, private, or mixed providers. Domain and frontend contract remain unchanged (deployment-neutral).

Blob/Object Store for managed originals is an **architectural boundary**; provider and product behavior are **not yet implemented**.

---

## 12. Tenant isolation, idempotency and security

### Tenant/workspace isolation (implementation gate)

Every durable Knowledge Input, Source, Document, Ingestion Operation, batch relation and managed original must be scoped by `tenant_id` and `workspace_id`. Cross-tenant access must fail closed.

### Idempotency (implementation gate)

Repeated delivery of one Slack event, one Slack attachment, one upload-finalize request, one queue message, or one completion event must not create duplicate sources, documents or embeddings unintentionally. Exact key formats are **DEFERRED**.

### Sensitive data — never expose through frontend responses or generic events

- full local path;
- provider credentials;
- Slack bot token;
- temporary Slack download URL;
- storage connection string;
- private bucket/blob key;
- raw provider locator;
- internal exception details.

### Policy gates (required at implementation; numeric limits DEFERRED)

Accepted media types; file-size limits; batch limits; decompression/archive limits; malware scanning integration point; URL egress; redirect limits; private-network URL blocking; checksum verification; duplicate file policy; retention of managed originals; failure and retry classification.

---

## 13. Implementation slicing

| Slice | Status | Concern |
|-------|--------|---------|
| `LKW-WORKSPACE-CONTENTS-1A` | `OPERATOR_VERIFIED` | Inspect active workspace sources (safe summaries; no full path). Not `LIVE_VERIFIED`. |
| `LKW-WORKSPACE-CONTENTS-1B-0` | `DOCUMENTED / READY_FOR_REVIEW` | This document — freeze channel-neutral Knowledge Intake and async ingestion contract. |
| `LKW-WORKSPACE-CONTENTS-1B-1` | `NEXT` | Durable Knowledge Intake and Ingestion Operation foundation: channel-neutral intake submission → tenant/workspace-scoped durable operation → idempotent acceptance → queue/worker boundary → neutral lifecycle event boundary. Does **not** yet implement all file/URL/connector variants. |
| `LKW-WORKSPACE-CONTENTS-1B-2` | planned | Managed file upload capability |
| `LKW-WORKSPACE-CONTENTS-1B-3` | planned | Slack attachment and multi-attachment adapter |
| `LKW-WORKSPACE-CONTENTS-1B-4` | planned | Preconfigured source candidate registration |
| `LKW-WORKSPACE-CONTENTS-1B-5` | planned | Explicit web URL intake |
| `LKW-WORKSPACE-CONTENTS-1C` | planned | Synchronization, operation inspection and channel-neutral completion notification |
| `LKW-WORKSPACE-CONTENTS-1D` | planned | Inspect indexed documents |
| `LKW-WORKSPACE-CONTENTS-1E` | planned | Safely remove source-owned knowledge |

**Explicit exclusions for `1B-1`:** does not automatically mean Slack file support, URL fetching, folder picker, connector marketplace, Kafka, Google Pub/Sub, production Blob Store, all ingestion providers, or background notification UI.

Do not mark any planned implementation slice as implemented.

---

## 14. Deferred decisions

The following remain **DEFERRED** (do not invent details in later docs as if frozen):

- exact HTTP route names;
- exact Pydantic/domain class names;
- upload protocol;
- resumable-upload technology;
- Blob Store provider;
- queue provider;
- pub/sub provider;
- outbox implementation;
- worker process topology;
- exact retry count;
- exact file-size limits;
- exact batch-size limits;
- exact progress granularity;
- exact Slack commands;
- Block Kit vs text for intake UX;
- automatic URL detection;
- cancellation UX;
- duplicate-content policy;
- original-file retention duration;
- malware scanner provider;
- remote connector OAuth implementation;
- whether batch state is a dedicated entity or derived from child operations.

---

## 15. Explicitly rejected contracts

| Rejected claim | Why |
|----------------|-----|
| Slack is the ingestion engine | Frontend only |
| Slack stores files in the knowledge base / calls Qdrant / creates embeddings | Storage and RAG stay behind LKW |
| Slack path command is the primary source contract | Paths leak and break deployment neutrality |
| Every folder upload becomes a live connector | Snapshot ≠ connected folder |
| Every source is a filesystem path | Source is a durable logical origin |
| Pub/sub is the operation source of truth | Durable operation record is |
| Slack thread stores job state | Correlation destination only |
| LKW core sends directly to Slack | Notification adapter resolves destination |
| Large files async but small files sync | Always operation-based |
| Document Store stores all original file bytes | Blob/Object Store for managed originals |
| Kafka / Google Pub/Sub selected | Vendor deferred |
| Production Blob Store already implemented | Boundary only |
| File upload endpoints / URL ingestion / Slack attachments already work | Planned after `1B-0` / `1B-1`+ |

Google NotebookLM is **not** a formal architectural dependency or compatibility claim.

---

## Illustrative API shape — NOT FROZEN / NOT IMPLEMENTED

The examples below are **conceptual only**. Route names, schemas, class names and event payloads are **not** frozen and **do not** exist as product endpoints today.

Illustrative acceptance result (NOT IMPLEMENTED):

```text
# ILLUSTRATIVE ONLY — routes/schemas not frozen
KnowledgeIntakeAcceptance:
  operation_id: <opaque>
  status: accepted
  workspace_id: <opaque>
  batch_id: <optional opaque>
```

Illustrative upload session flow (NOT IMPLEMENTED):

```text
# ILLUSTRATIVE ONLY — not existing endpoints
create upload session → transfer bytes → finalize → submit Knowledge Input
```

Do not present `/knowledge-inputs`, upload-session routes or event schemas as existing endpoints.

---

## 16. Documentation acceptance checklist

- [x] LKW owns complete intake and ingestion lifecycle
- [x] Frontends are replaceable; Slack has no ingestion/RAG/storage logic
- [x] Knowledge Input, Source, Document, Ingestion Operation are distinct
- [x] Source is not defined only as connector-backed local path
- [x] managed upload, source candidate and URL represented
- [x] uploaded folder snapshot vs connected folder distinct
- [x] raw local path in Slack rejected
- [x] always operation-based ingestion
- [x] durable operation is source of truth
- [x] queue/worker vs pub/sub roles distinct
- [x] LKW core does not call Slack
- [x] storage responsibilities separated
- [x] tenant/workspace and idempotency documented
- [x] deferred decisions listed without invented vendors/routes
- [x] historical Ask MVP file-ignore scoped, not erased
- [x] no claim that uploads, URLs, workers or Slack attachments are implemented
