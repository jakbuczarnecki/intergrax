# MP-5F-B4 — Collaborative Work Read Boundary

## 1. Scope

Certification of the domain-owned, scoped, reference-only Collaborative Work read boundary (WorkItem, WorkArtifact, WorkArtifactVersion canonical references). Covers contract ownership, isolation, ordering/limit semantics, provider pluginability, layer boundaries, MP-5D/B5 relationship, static typing, and regression evidence.

Out of scope: B5 adapter redesign, MP-5G harness changes, new ContextView ports, repository lifecycle redesign, authority/membership enumeration.

## 2. Git provenance

| Field | Value |
| --- | --- |
| `MP5F_B4_SESSION_START_HEAD` | `57ee5d7eac6b70ba29c7eb8d84c42b194235d53e` |
| Branch | `development` |
| Session start `HEAD == origin/development` | **yes** |
| Implementation commit (prior) | `9b8b82681` — `feat(collaborative-work): add scoped reference read boundary` |
| `MP5F_B4_EVIDENCE_HEAD` (pre-closeout) | `7d71bae234a826176fa3015cbd29cecbe333a497` |
| Closeout commit | `db222ec4f6b27d1e239668da572330d2d000e404` |

## 3. Current-head reconnaissance

| Symbol / area | Classification |
| --- | --- |
| `CollaborativeWorkReferenceReadPort` | **EXISTING-CANONICAL** |
| `DefaultCollaborativeWorkReferenceReader` | **EXISTING-CANONICAL** |
| `CollaborativeWorkScopedReferenceCatalog` | **EXISTING-CANONICAL** |
| `RepositoryBackedCollaborativeWorkReferenceCatalog` | **EXISTING-CANONICAL** |
| `DefaultCollaborativeWorkContextSource` | **TEMPORARY-B5-WIRING** (correct: consumes `CollaborativeWorkReferenceReadPort`, no repository imports) |
| Duplicate MP-5D port | **none** |

**B4 functionality at session start:** **ALREADY IMPLEMENTED** on `development` (ancestor `9b8b82681`); this session adds qualification artifact and removes forbidden typing escapes in the B4 gate test.

## 4. Canonical semantic owner

| Role | Owner |
| --- | --- |
| CW reference-read semantics | `intergrax.collaborative_work` — `contracts/collaborative_work_reference_read.py`, `default_collaborative_work_reference_reader.py` |
| ContextView consumer source port | `intergrax.contracts.context_view_source_ports` — `CollaborativeWorkContextSourcePort` (MP-5D) |
| Persistence / scoped catalog projection | `WorkItemRepository`, `WorkArtifactRepository`, `WorkArtifactVersionRepository` + `CollaborativeWorkScopedReferenceCatalog` (`repository.py`, `repository_backed_reference_catalog.py`) |
| Artifact publication authority | Existing publication repository / commands (unchanged) |
| B5 mapping | `context_view_source_adapters.py`, `context_view_source_mapping.py` |

## 5. Existing MP-5D/B5 relationship

- MP-5D **`CollaborativeWorkContextSourcePort`** is **not duplicated**.
- **`DefaultCollaborativeWorkContextSource`** injects **`CollaborativeWorkReferenceReadPort`** only; maps canonical refs → `ContextViewCollaborativeWorkSourceRef`.
- **`ContextViewCollaborativeWorkSourceRef`** is **not** the CW domain authority (mapping is B5-only, downstream of B4).

## 6. Domain read contract

| Artifact | Location |
| --- | --- |
| Request | `CollaborativeWorkReferenceReadRequest` (`scope` + `query`) |
| Scope | `CollaborativeWorkReferenceReadScope` — `tenant_id`, `workspace_id`, optional `work_item_id`, `work_artifact_id`, `work_artifact_version_id` |
| Query | `CollaborativeWorkReferenceReadQuery` — `entity_kinds`, `limit`, `version_selection` |
| Result | `CollaborativeWorkReferenceReadResult` — `outcome`, `references`, `reason` |
| Port | `CollaborativeWorkReferenceReadPort.read_references(identity, request)` |
| Default impl | `DefaultCollaborativeWorkReferenceReader` |
| Validation helper | `validate_collaborative_work_reference_read_request` |

Public exports: `intergrax.collaborative_work.contracts` (`CollaborativeWorkReferenceReadPort` and DTOs).

## 7. Scope model

Mandatory: **`tenant_id`**, **`workspace_id`**.

Optional narrowing: **`work_item_id`**, **`work_artifact_id`**, **`work_artifact_version_id`** (resource filters within workspace; no workspace inference from work item id).

Identity tenant must match request scope tenant or **`SCOPE_REJECTED`**.

Configured **`CollaborativeWorkReferenceReadCapabilityBinding`** must match request tenant/workspace or **`SCOPE_REJECTED`**.

## 8. Reference model

Canonical CW-owned reference DTOs (reference-only, no payload):

- `CollaborativeWorkItemCanonicalRef` — work item locator + `WorkItemState`
- `CollaborativeWorkArtifactCanonicalRef` — includes authoritative **`current_version_id`**
- `CollaborativeWorkArtifactVersionCanonicalRef` — full version identity tuple (aligned with `WorkArtifactVersion` fields; not a second ContextView model)

Union: `CollaborativeWorkCanonicalRef`.

## 9. WorkItem semantics

Enumerated via catalog over `WorkItemRepository` semantics (`list_for_workspace` / scoped filters). Returns **`CollaborativeWorkItemCanonicalRef`** only.

## 10. WorkArtifact semantics

Enumerated under scope; **`current_version_id`** on artifact ref comes from aggregate pointer, not timestamp inference.

## 11. WorkArtifactVersion semantics

**`CollaborativeWorkVersionSelection.CURRENT_ONLY`** (default): only version matching artifact **`current_version_id`**.

**`INCLUDE_HISTORICAL`**: all versions for scoped artifact(s).

Proven in `test_current_only_hides_historical_versions`.

## 12. Repository/provider composition

```text
DefaultCollaborativeWorkReferenceReader
  → CollaborativeWorkScopedReferenceCatalog.list_scoped_references
  → (default) RepositoryBackedCollaborativeWorkReferenceCatalog
       → WorkItemRepository + WorkArtifactRepository + WorkArtifactVersionRepository
```

Backend-specific **`list_scoped_references`** also implemented on InMemory / SQLite / PostgreSQL repository surfaces (same query type); composition root wires protocols only.

## 13. Pluginability

- Custom **`CollaborativeWorkReferenceReadPort`** without default runtime: `test_pluginability_custom_port_without_default_runtime`.
- Custom **`CollaborativeWorkScopedReferenceCatalog`** without concrete DB: `test_external_catalog_without_concrete_repository`.
- Architecture gates: B4 contract/default reader/catalog modules must not import ContextView, Memory, RAG, UCL, Nexus, or concrete repository modules (except catalog → repository **protocols**).

## 14. Tenant isolation

Cross-tenant seed under same workspace id excluded (`test_wrong_tenant_scope_rejected`, workspace read suites).

## 15. Workspace isolation

`tenant-A / workspace-B` data absent from `tenant-A / workspace-A` queries (`test_wrong_workspace_empty_ok`, limit-after-scope test).

## 16. Work-item isolation

Scoped `work_item_id` excludes other items and forged artifact ids (`test_forged_work_item_excludes_artifact`, artifact parent forgery test).

## 17. Deterministic ordering

Default reader sorts by `(entity_kind, work_item_id, work_artifact_id, work_artifact_version_id)` after scope validation (`test_deterministic_ordering_and_limit_after_full_scope`).

## 18. Limit semantics

`limit` on query; max **`COLLABORATIVE_WORK_REFERENCE_READ_MAX_LIMIT` (200)**. Applied **after** full scope filtering in catalog + reader sort/limit path (T9: many off-scope rows cannot consume limit).

## 19. Payload/reference-only guarantee

| Check | Result |
| --- | --- |
| Payload bytes in public result | **NO** |
| Artifact content hydration | **NO** |
| Repository row leakage | **NO** (listings projected to canonical refs) |
| Backend-specific public types | **NO** |
| ContextView DTO in CW domain contract | **NO** (import audit + AST gate) |

## 20. Provider contract violation handling

If catalog returns listings outside query scope or listings that fail canonical projection, **`DefaultCollaborativeWorkReferenceReader`** returns **`UNAVAILABLE`** with reason **`catalog_contract_violation`** (fail-closed, explicit — not silent drop; no exception escape). Structurally invalid listings are rejected at **`CollaborativeWorkScopedReferenceListing`** construction (B4 gate DTO tests; **MP-5F-B4-R2-R1**). Reader/query violations: B4 gate custom-catalog matrix + **MP-5F-B4-R2** (`test_plugin_out_of_scope_listing_fail_closed`).

## 21. Backend parity

| Backend | Catalog path | B4 gate |
| --- | --- | --- |
| InMemory | `RepositoryBackedCollaborativeWorkReferenceCatalog` + in-memory repos | **primary behavioral matrix** |
| SQLite | Native `list_scoped_references` on SQLite repos + same catalog composition | **contract parity** (implementation present; B4 gate uses InMemory + custom catalog) |
| PostgreSQL | Native `list_scoped_references` | **integration / repo suites** (no live PG in unit gate) |
| Custom provider | Port or catalog only | **proven** |

## 22. Static typing

Command:

```bash
uv run pyright \
  intergrax/collaborative_work/contracts/collaborative_work_reference_read.py \
  intergrax/collaborative_work/default_collaborative_work_reference_reader.py \
  intergrax/collaborative_work/repository_backed_reference_catalog.py \
  tests/unit/collaborative_work/test_cw_mp5f_b4_collaborative_work_reference_read.py
```

Closeout: **0 errors, 0 warnings**. B4 gate helpers use explicit typed builders (no `dict[str, object]` + `type: ignore`).

## 23. Layer boundary audit

| Check | Result |
| --- | --- |
| CW domain imports ContextView | **NO** |
| Duplicate `CollaborativeWorkContextSourcePort` | **NO** |
| B5 bypasses B4 (direct repository in adapter) | **NO** |
| Consumer → concrete repository | **NO** (must use port/catalog) |

## 24. B5 readiness

**Can B5 map B4 references to `ContextViewCollaborativeWorkSourceRef` without repository imports?** **YES** — `DefaultCollaborativeWorkContextSource` already does via mapping helpers.

## 25. Findings

| ID | Class | Disposition |
| --- | --- | --- |
| F1 | CW-TYPE-SAFETY | B4 test used `type: ignore` + dict kwargs — **fixed** this session |
| — | CW-READ-DUPLICATE | **none** |
| — | CW-B5-BYPASS | **none** (adapter uses port) |
| — | ADR | **none required** |

## 26. Regression evidence

```bash
uv run pytest tests/unit/collaborative_work/test_cw_mp5f_b4_collaborative_work_reference_read.py -q
# 17 passed

uv run pytest tests/unit/contracts/test_context_view_source_ports.py -q
# (included in combined gate)

uv run pytest \
  tests/unit/collaborative_work/test_mp5f_b5_context_view_source_adapters.py \
  tests/unit/collaborative_work/test_context_view_source_ports_architecture_gates.py -q
# 21 passed
```

Session logs: `.tmp/session/mp5f-b4/pytest.log`, `pyright-r2.log`, `pytest-b5-regression.log`.

## 27. Final verdict

**MP-5F-B4 — CLOSED / CERTIFIED**

Collaborative Work owns scoped reference-read semantics; MP-5D remains the sole ContextView source port; B4 is reference-only, isolation-bounded, provider-pluggable, and B5-ready.

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z GitHuba.
