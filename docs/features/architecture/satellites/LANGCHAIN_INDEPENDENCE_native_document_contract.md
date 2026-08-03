<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Native Knowledge Document Contract — LCI-1A

**Status:** LCI-1A architecture: **APPROVED**; LCI-1B implementation: **READY_FOR_REVIEW**
**Owner:** RAG (functional) · Tier-0 `intergrax/knowledge/` (neutral shared core)  
**Feature hub:** [`../LANGCHAIN_INDEPENDENCE.md`](../LANGCHAIN_INDEPENDENCE.md)  
**Feature plan:** [`../../plan/LANGCHAIN_INDEPENDENCE.md`](../../plan/LANGCHAIN_INDEPENDENCE.md)  
**Anchor domain architecture:** [`../../../architecture/RAG.md`](../../../architecture/RAG.md)

---

## 1. Scope

This satellite is the **source of truth** for the canonical Intergrax knowledge document ABI (`KnowledgeDocument`). It replaces `langchain_core.documents.Document` as the platform contract for:

- RAG ingest, chunking, embedding, indexing, retrieval, reranking
- Memory knowledge surfaces
- Modality text normalization outputs
- Integration fetch → RAG-ready normalization

**In scope (LCI-1A):** architecture, field contract, invariants, mappings, acceptance checklist.  
**Out of scope:** Python implementation (`LCI-1B`), LangChain bridge (`LCI-1C`), conformance CI (`LCI-1D`), consumer migration (`LCI-2A` onward).

---

## 2. Location decision

| Aspect | Decision |
|--------|----------|
| **Canonical type** | `KnowledgeDocument` |
| **Target module (LCI-1B)** | `intergrax/knowledge/contracts/document.py` |
| **Public import** | `from intergrax.knowledge.contracts import KnowledgeDocument` |
| **Package tier** | Tier-0 shared core — neutral, not under `intergrax/rag/` |
| **Functional owner** | RAG domain owns contract semantics and evolution |
| **Shared consumers** | Memory, modality, integrations import from `intergrax.knowledge.contracts`; they must not define parallel document types |

RAG remains the **functional owner** of the contract. Memory, modality, and integrations are **consumers** of the neutral Tier-0 type; they must not be forced to import from RAG pipeline internals.

---

## 3. Reuse analysis (Vendor Knowledge)

### 3.1 Reviewed models

From `intergrax/runtime/vendor_knowledge/models.py`:

- `KnowledgeSourceRef`, `KnowledgeItemIdentity`, `KnowledgeItemRevision`, `KnowledgeItemProvenance`, `KnowledgeItemDescriptor`, `KnowledgeContent`
- Supporting types: `JsonObject`, secret-key validation, safe URL validation, immutable Pydantic v2 `ConfigDict(extra="forbid", frozen=True)`

### 3.2 Semantics reused (principles, not direct imports)

| Principle | Source inspiration |
|-----------|-------------------|
| Tenant scope | `KnowledgeSourceRef.tenant_id` |
| Stable remote/source identity | `KnowledgeItemIdentity.remote_id`, `parent_remote_id` |
| Source revision | `KnowledgeItemRevision.version`, `etag`, `content_hash` |
| Provenance | `KnowledgeItemProvenance` (provider, source kind, remote id, safe locator) |
| Safe metadata | `_assert_safe_mapping`, forbidden secret-like keys |
| Credential-free locator | `_validate_safe_url` |
| Immutable strict models | Pydantic v2 frozen + extra forbid |

### 3.3 Classes not imported directly

| Class | Reason |
|-------|--------|
| `KnowledgeSourceRef` | Integration-layer connection/scope descriptor; not RAG-ready document |
| `KnowledgeItemDescriptor` | Inventory/fetch stage aggregate |
| `KnowledgeContent` | Holds binary/rich_text/structured_record — pre-normalization |
| `IntegrationCategory` | Integration taxonomy; would couple shared core → runtime/integrations |

**Decision:** Vendor Knowledge models represent the **external source item** stage. `KnowledgeDocument` represents the **RAG-ready normalized text** stage. They are adjacent pipeline stages, not duplicates.

### 3.4 Vendor Knowledge → KnowledgeDocument mapping (future)

```text
KnowledgeItemDescriptor + KnowledgeContent
        ↓ adapter normalizes content to str, maps identity/provenance
KnowledgeDocument
```

| Vendor Knowledge | KnowledgeDocument | Rule |
|----------------|-------------------|------|
| `descriptor.identity.remote_id` | `provenance.source_id` | required |
| `descriptor.identity.parent_remote_id` | `provenance.source_parent_id` | optional; external source hierarchy only (e.g. parent Drive folder, Confluence page, mail folder, message thread/channel) |
| `descriptor.provenance.source_kind` | `provenance.source_kind` | required |
| `descriptor.provenance.provider_id` | `provenance.provider_id` | optional |
| `descriptor.revision.version` / `etag` | `provenance.source_revision` | prefer `version`; `etag` if version absent |
| `descriptor.provenance.web_url` / `safe_locator` | `provenance.source_uri` | credential-free only |
| `descriptor.revision.content_hash` | `provenance.content_hash` | optional; not a substitute for `document_id` |
| `KnowledgeSourceRef.tenant_id` | `scope.tenant_id` | required |
| `descriptor.metadata` (safe subset) | `metadata` | JSON-safe only; reserved keys excluded |
| `KnowledgeContent` payload | `content` | **must be normalized to non-empty `str` before document creation** |

`identity.parent_document_id` is set **only** by an Intergrax process that creates a derivative document (parser, normalizer, chunker, or bridge). Vendor Knowledge `parent_remote_id` describes external source hierarchy, not document lineage.

Normalization adapter implementation is **out of scope** for LCI-1A; belongs to integration ingest paths after LCI-1B.

### 3.5 Shared validation component decision

1. `KnowledgeDocument` must not import private functions from `intergrax/runtime/vendor_knowledge/models.py`.
2. LCI-1B must not copy implementations of secret-key detection, recursive JSON validation, or credential-free URI validation.
3. Before implementation, perform an exact search for existing neutral helpers.
4. If a neutral component already exists, reuse it.
5. If none exists, LCI-1B must, in the same functional block:
   - extract neutral, reusable validation primitives to Tier-0;
   - use them in `KnowledgeDocument`;
   - rewire Vendor Knowledge to the same component;
   - preserve current Vendor Knowledge behavior via regression tests.
6. Two independent implementations of the same security policy are not allowed. Do not choose a final module name without checking the existing `intergrax/` layout.

---

## 4. Data model

Implementation target (LCI-1B): Pydantic v2 `BaseModel` with `ConfigDict(extra="forbid", frozen=True)` on all types. No automatic UUID generation. No `langchain*` or provider SDK types.

### 4.1 Type hierarchy

```text
KnowledgeDocument
├── schema_version: int
├── identity: KnowledgeDocumentIdentity
├── scope: KnowledgeDocumentScope
├── content: str
├── metadata: dict[str, JsonValue]
└── provenance: KnowledgeDocumentProvenance

KnowledgeDocumentIdentity
├── document_id: str
├── root_document_id: str
└── parent_document_id: str | None

KnowledgeDocumentScope
├── tenant_id: str
└── namespace: str | None

KnowledgeDocumentProvenance
├── source_kind: str
├── source_id: str
├── source_parent_id: str | None
├── provider_id: str | None
├── source_revision: str | None
├── source_uri: str | None
└── content_hash: str | None
```

Illustrative target signatures (LCI-1B):

```python
# intergrax/knowledge/contracts/document.py — implementation in LCI-1B only
class KnowledgeDocumentIdentity(BaseModel): ...
class KnowledgeDocumentScope(BaseModel): ...
class KnowledgeDocumentProvenance(BaseModel): ...
class KnowledgeDocument(BaseModel): ...
```

### 4.2 KnowledgeDocument fields

| Field | Type | Rule |
|-------|------|------|
| `schema_version` | `int` | Required; `1` for this contract |
| `identity` | `KnowledgeDocumentIdentity` | Required |
| `scope` | `KnowledgeDocumentScope` | Required |
| `content` | `str` | Required; non-empty; not whitespace-only |
| `metadata` | `dict[str, JsonValue]` | Required; default `{}`; JSON-compatible values only |
| `provenance` | `KnowledgeDocumentProvenance` | Required |

### 4.3 KnowledgeDocumentIdentity

| Field | Rule |
|-------|------|
| `document_id` | Required, persistent, non-empty. **Supplied by caller** (loader, parser, chunker, bridge). Never auto-generated. |
| `root_document_id` | Required, non-empty. For source document: equals `document_id`. For chunk/derivative: points to originating source document. |
| `parent_document_id` | `None` for source document. Required non-empty for direct chunk or derived document. Must not equal `document_id`. Set only by Intergrax derivative processes (parser, normalizer, chunker, bridge); never from vendor source hierarchy. |

### 4.4 KnowledgeDocumentScope

| Field | Rule |
|-------|------|
| `tenant_id` | Required, non-empty. No implicit `"default"`. |
| `namespace` | Optional neutral partition id (collection, index partition, logical bucket). |

**Not canonical fields** (may appear in `metadata` only): `workspace_id`, `slack_workspace_id`, `application_id`.

### 4.5 KnowledgeDocumentProvenance

| Field | Rule |
|-------|------|
| `source_kind` | Required, non-empty (e.g. `file`, `web`, `drive`, `mail`) |
| `source_id` | Required, persistent id of source element |
| `source_parent_id` | Optional; id of parent element in the external source (e.g. parent Drive folder, parent Confluence page, message thread/channel, mail folder). Describes source hierarchy, not document chunking or processing lineage. |
| `provider_id` | Optional provider slug |
| `source_revision` | Optional version/etag/revision |
| `source_uri` | Optional credential-free locator |
| `content_hash` | Optional content digest; **not** a substitute for `document_id` |

---

## 5. Invariants

1. All models are immutable (`frozen=True`) and reject unknown fields (`extra="forbid"`).
2. `content` is always a `str` ready for RAG processing — never bytes, media handles, or provider objects.
3. `tenant_id` is always required; missing tenant context is a validation error.
4. Identity fields are caller-supplied; the contract never mints IDs.
5. Lineage invariants enforced by LCI-1B validators (see §6).
6. Reserved keys are never valid inside native `KnowledgeDocument.metadata`, regardless of whether their values equal typed fields.
7. Typed field vs `metadata` conflict → validation error (fail-closed); duplicate reserved key with identical value is also rejected.
8. `schema_version` mismatch on deserialize → error (no silent migration).

---

## 6. Identity and lineage

### 6.1 Canonical document identity key

```text
Canonical document identity key: (tenant_id, namespace, document_id)
```

Rules:

1. `document_id` is persistent and unique within `(tenant_id, namespace)`.
2. It does not need to be globally unique across tenants.
3. `namespace=None` is an explicit absence of an additional partition; it must not be silently coerced to the string `"default"`.
4. `root_document_id` and `parent_document_id` are resolved only within the same `(tenant_id, namespace)`.
5. Cross-tenant and cross-namespace lineage is forbidden.
6. The storage layer must either include scope in the physical key or guarantee equivalent isolation via partition/collection.
7. This contract does not generate a physical storage key. Do not add a `global_document_id` field.

### 6.2 Source hierarchy vs document lineage

- **`provenance.source_parent_id`:** parent element in the external source (folder, page, thread, mail folder). Set by ingest from vendor descriptors such as `parent_remote_id`. Not chunking or processing history.
- **`identity.parent_document_id`:** immediate parent in Intergrax document lineage. Set **only** by Intergrax processes that create derivative documents (parser, normalizer, chunker, bridge). Never mapped from vendor `parent_remote_id`.

### 6.3 Lineage examples

```text
Source document (file.pdf)
  document_id = "file:abc123"
  root_document_id = "file:abc123"
  parent_document_id = None

Chunk 3 of source
  document_id = "file:abc123:chunk:3"    ← deterministic per chunker contract
  root_document_id = "file:abc123"
  parent_document_id = "file:abc123"

Re-chunk of chunk 3
  document_id = "file:abc123:chunk:3:sub:1"
  root_document_id = "file:abc123"
  parent_document_id = "file:abc123:chunk:3"
```

- **Source document:** `root_document_id == document_id`, `parent_document_id is None`.
- **Direct fragment:** `parent_document_id` points to immediate parent (source or parent chunk). A direct chunk may have `parent_document_id == root_document_id`.
- **Subchunk:** `parent_document_id != root_document_id`.
- **Root lineage:** `root_document_id` always points to the original source document id within the same document scope.
- Chunkers, loaders, and bridges own ID generation policy; the contract validates shape, scope, and consistency.

### 6.4 Lineage validators (LCI-1B required)

**Source document:**

```text
parent_document_id is None
root_document_id == document_id
```

**Derivative document, chunk, or subchunk:**

```text
parent_document_id is not None
root_document_id != document_id
parent_document_id != document_id
```

Additionally, `root_document_id`, `parent_document_id`, and `document_id` must belong to the same document scope `(tenant_id, namespace)`.

The local model blocks self-reference (`parent_document_id == document_id`). Full detection of multi-node lineage cycles belongs to the pipeline or document repository and remains outside LCI-1B.

---

## 7. Tenant and security (fail-closed)

| Rule | Behavior |
|------|----------|
| Tenant required | `tenant_id` must be explicit non-empty string |
| No default tenant | `tenant_id="default"` is forbidden as implicit fallback |
| Safe URI | `source_uri` must not embed username, password, or secret query parameters |
| Metadata secrets | Keys matching token/password/secret/api_key/authorization patterns → validation error |
| No silent coercion | Non-JSON-compatible metadata values → error, not `str()` conversion |
| No provider leakage | Provider SDK objects cannot pass through the contract |
| URL in metadata | String values that look like URLs are validated credential-free (same policy as Vendor Knowledge) |

---

## 8. Metadata rules

### 8.1 Allowed value types

`str`, `int`, `float`, `bool`, `null`, `list`, `dict` with string keys — recursively JSON-compatible. `float` values must be finite; `NaN`, `Infinity`, and `-Infinity` are forbidden.

### 8.2 Forbidden in metadata

- Class instances, callables, `bytes`
- Credentials, tokens, passwords
- Duplicate of any canonical/reserved field (including identical values)

### 8.3 Reserved keys (must not appear in `metadata`)

```text
schema_version
document_id
root_document_id
parent_document_id
tenant_id
namespace
source_kind
source_id
source_parent_id
provider_id
source_revision
source_uri
content_hash
```

### 8.4 Reserved metadata policy

Reserved keys are never valid inside native `KnowledgeDocument.metadata`, regardless of whether their values equal typed fields.

- Native constructor: reserved key present → `ValidationError` (no duplicate allowed even with an identical value).
- LangChain bridge (LCI-1C):
  1. Read known typed fields from input metadata.
  2. Detect conflicts between `Document.id`, aliases, and metadata.
  3. Remove all processed reserved keys from remaining metadata.
  4. Pass only remaining JSON-safe data to `KnowledgeDocument.metadata`.

---

## 9. Content rules

`KnowledgeDocument` holds **text ready for RAG**, not raw media.

| Rule | Detail |
|------|--------|
| Type | `str` only |
| Non-empty | Must contain at least one non-whitespace character |
| No auto-trim | Contract does not strip or modify `content` |
| Serialization | Exact `content` value preserved in round-trip |
| Pre-normalization | Binary, image, audio, video, structured provider records must be parsed/normalized **before** document creation |

---

## 10. Provenance rules

- `source_kind` + `source_id` together identify the external source element.
- `provider_id` is optional integration/provider slug.
- `source_revision` tracks upstream version; optional.
- `source_uri` is display/navigation only; must be credential-free.
- `content_hash` is advisory integrity signal; must not replace `document_id`.

---

## 11. Serialization and versioning

| Aspect | Policy |
|--------|--------|
| `schema_version` | `1` (integer) |
| Dump format | Deterministic JSON-compatible dict (UTF-8, sorted keys, stable separators, `allow_nan=False` or equivalent pre-dump validation) |
| JSON numbers | `float` values must be finite; `NaN`, `Infinity`, and `-Infinity` forbidden in metadata and serialized output |
| Round-trip | Deserialize → serialize preserves all fields and exact `content` |
| Unknown fields | Rejected on input (`extra="forbid"`) |
| Unknown `schema_version` | Rejected; no automatic migration |
| Semantic change | Existing field meaning change → new `schema_version` |
| Optional field addition | Requires explicit backward-compatibility review |

Serializers and deserializers are **LCI-1B** deliverables. This document defines requirements only.

---

## 12. LangChain Document mapping

Bridge implementation: **LCI-1C** (`intergrax/compat/langchain/`). Conversion rules defined here for LCI-1B/LCI-1C alignment.

### 12.1 Field mapping table

| LangChain `Document` | `KnowledgeDocument` | Rule |
|----------------------|---------------------|------|
| `page_content` | `content` | Exact value preserved |
| `id` | `identity.document_id` | Required; missing `id` requires explicit ID passed to bridge |
| metadata `root_document_id` | `identity.root_document_id` | Absent for source → defaults to `document_id` |
| metadata `parent_document_id` | `identity.parent_document_id` | Optional; `None` for source |
| metadata `tenant_id` | `scope.tenant_id` | Required; missing → error |
| metadata `namespace` | `scope.namespace` | Optional |
| metadata `source_kind` | `provenance.source_kind` | Required |
| metadata `source_id` or legacy `source` | `provenance.source_id` | Both present with different values → error |
| metadata `provider_id` | `provenance.provider_id` | Optional |
| metadata `source_revision` | `provenance.source_revision` | Optional |
| metadata `source_uri` | `provenance.source_uri` | Optional; validated credential-free |
| metadata `content_hash` | `provenance.content_hash` | Optional |
| metadata `source_parent_id` | `provenance.source_parent_id` | Optional |
| Remaining JSON-safe metadata | `metadata` | Preserved without loss; reserved keys excluded |
| — | `schema_version` | Set to `1` on conversion to native |

### 12.2 Bridge error policy

| Condition | Result |
|-----------|--------|
| Missing `tenant_id` in metadata | Error |
| Missing `source_kind` | Error |
| Missing `document_id` (`id` and metadata) | Error |
| `Document.id` conflicts with metadata `document_id` | Error |
| `source_id` vs legacy `source` conflict | Error |
| Reserved key in metadata (any value, including match with typed field) | Error; bridge strips processed reserved keys before native construction |
| Non-JSON-safe metadata value | Error |
| Whitespace-only `page_content` | Error |
| Random/fallback ID generation | **Forbidden** |
| `tenant_id="default"` implicit fallback | **Forbidden** |
| Unknown but valid extra metadata | **Preserved** in `metadata` (bridge must not drop) |

---

## 13. Error policy (summary)

All validation is **fail-closed**:

- Missing required identity, scope, or provenance fields → `ValidationError`
- Security violations (secrets in metadata/URI) → `ValidationError`
- Identity inconsistency (lineage invariants in §6.4, including `parent == self`) → `ValidationError`
- Reserved key in native `metadata` → `ValidationError`
- Schema version unsupported → `ValidationError`
- LangChain bridge (LCI-1C) uses same rules; no silent data loss or default tenant

---

## 14. Task boundaries

| Task | Deliverable | Depends on |
|------|-------------|------------|
| **LCI-1A** (this doc) | Architecture + contract spec + mappings | LCI-0C |
| **LCI-1B** | `intergrax/knowledge/contracts/document.py`; Pydantic models; serializers/validators; unit tests | LCI-1A acceptance |
| **LCI-1C** | `from_langchain_document` / `to_langchain_document` in `intergrax/compat/langchain/` | LCI-1B |
| **LCI-1D** | Conformance gate: import without `langchain*`; round-trip; identity/metadata tests; CI wiring | LCI-1B |

**LCI-1A explicitly does not:** implement Python modules, migrate consumers, create bridge, create serializers, add tests, modify inventory or CI.

---

## 15. Migration impact (inventory confirmation)

Per [`LANGCHAIN_INDEPENDENCE_dependency_inventory.md`](LANGCHAIN_INDEPENDENCE_dependency_inventory.md) §D, **16 contract files** currently expose `langchain_core.documents.Document` as `CORE_CONTRACT_LEAK`. Native `KnowledgeDocument` is the declared replacement target across RAG, integrations, memory, and modality consumer migrations (`LCI-2A`–`LCI-4D`). Inventory rows are unchanged in LCI-1A.

---

## 16. Acceptance checklist (LCI-1A)

- [x] Canonical location: `intergrax/knowledge/contracts/document.py` with public import `from intergrax.knowledge.contracts import KnowledgeDocument`
- [x] Complete field contract for `KnowledgeDocument`, `KnowledgeDocumentIdentity`, `KnowledgeDocumentScope`, `KnowledgeDocumentProvenance`
- [x] Identity and lineage rules documented
- [x] Tenant fail-closed policy documented
- [x] Metadata reserved keys and JSON-safety rules documented
- [x] Provenance rules documented
- [x] Serialization/versioning requirements (`schema_version = 1`) documented
- [x] LangChain `Document` mapping table with error policy
- [x] Vendor Knowledge reuse decision and mapping documented
- [x] Explicit boundaries for LCI-1B, LCI-1C, LCI-1D
- [x] RAG architecture hub entry points to this satellite
- [x] Feature plan status updated to READY_FOR_REVIEW
- [x] Canonical scoped identity key documented
- [x] Source hierarchy (`source_parent_id`) separated from document lineage (`parent_document_id`)
- [x] Strict source/derivative lineage invariants defined for LCI-1B
- [x] Reserved metadata always rejected (including duplicate canonical values)
- [x] Finite JSON numbers and deterministic serialization requirements documented
- [x] Shared validation component reuse decision documented

---

## 17. Implementation evidence (LCI-1B)

| Item | Location |
|------|----------|
| Native models | `intergrax/knowledge/contracts/document.py` |
| Public import | `from intergrax.knowledge.contracts import KnowledgeDocument` |
| Serializer API | `dump_knowledge_document` / `load_knowledge_document` in `intergrax/knowledge/contracts/document.py` |
| Shared validation | `intergrax/knowledge/contracts/validation.py` (reused by Vendor Knowledge models) |
| Unit tests | `tests/unit/knowledge/contracts/test_document.py` |
| Consumer migration | Not started (remains LCI-2+ / bridge LCI-1C) |
