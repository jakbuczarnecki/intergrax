# MP-7A — LKW Multiplayer Adoption Architecture & Contract Gate

| Field | Value |
|-------|-------|
| **Status** | **CLOSED / CERTIFIED** (subject to independent audit before MP-7B) |
| **Audit HEAD** | `2ef3b49646602a4bb79c7b76dc4ba1cab0e965d1` (MP-6H closure; ancestry verified at gate close) |
| **ADR** | [ADR-MP-008](../../technical/adr/entries/2026-09-19/ADR-MP-008.md) — **Accepted** |
| **Production code** | **NONE** (MP-7A) |

## 1. Audit identity

```text
START_HEAD = 2ef3b49646602a4bb79c7b76dc4ba1cab0e965d1
MP6_CLOSURE_ANCESTRY = yes (closure commit is ancestor of START_HEAD)
WORKTREE_STATE = clean at gate authoring (untracked build/pytest artifacts excluded)
```

## 2. Current LKW state (HEAD inventory)

| Area | LKW ownership today | Multiplayer overlap |
|------|---------------------|---------------------|
| Managed workspace (`workspaces/models.py`, `ManagedWorkspaceRepository`) | Product workspace lifecycle, knowledge scope | **PLATFORM DUPLICATE risk** if treated as collaborative workspace authority — **ADAPTER/MAPPING** via `collaborative_workspace_ref` (ADR-MP-008 Option B) |
| `tenant_id` / `workspace_id` on product records | Product isolation for knowledge, Ask, sources | **Not authorization proof** for Multiplayer (**MP-INV-04**); must map to platform scope |
| Workspace Knowledge Configuration | LKW-owned mutation engine | **PRODUCT-LOCAL** |
| Conversation Context (`conversation_context_models.py`) | Durable thread memory, audience policy | **Not ContextView** — separate ownership (**KEEP LKW-LOCAL**) |
| Nexus `Task` / graph pipeline | Execution orchestration | **KEEP LKW-LOCAL**; WorkItem **DEFER** (**MP-INV-07**) |
| Shadow / synthesis outputs | Product write surface | **MAP** to future WorkArtifact; not canonical today |
| Slack/channel correlation | Adapter storage | **ADAPTER/MAPPING** only |
| HITL / approval UX | Governance/Nexus | **KEEP**; Decision binding **DEFER** (**MP-INV-09**) |
| Principal / membership / delegation | Documented as **FUTURE** in LKW ARCHITECTURE | Platform-owned when adopted — **no local bool membership** |
| Imports | **No** `intergrax.collaborative_work` or `intergrax.contracts.collaborative_*` in `applications/local_workspace_application` at audit HEAD | Consumer path not yet wired |

Reference consumer pattern (Tier-3): `governed_contractor_application` wires `intergrax.contracts.collaborative_work` at host boundary with composition-only repository bundles — LKW must follow the same class of boundary, not invent `LKWCollaborativeWorkStore`.

## 3. LKW ↔ Multiplayer adoption matrix

| Capability | Platform owner | Classification | Notes |
|------------|----------------|----------------|-------|
| Principal | MP-1 | **ADOPT** (MP-7B) | Typed `LkwPrincipalBinding` → `intergrax/contracts/collaborative_work.py` principal types |
| Workspace | MP-1 / CW plane | **MAP** + **ADOPT** binding (MP-7B) | Option B: product workspace + `collaborative_workspace_ref` |
| Membership | MP-1 | **DEFER** | LKW multi-principal sharing not in current Product Alpha journey |
| Delegation | MP-1 | **DEFER** | Until agent-acting-for-user scenarios are product-scheduled |
| WorkItem | MP-2 | **DEFER** | No collaborative work lifecycle in current LKW journeys; Nexus Task remains |
| Assignment | MP-2 | **DEFER** | No principal-assignment semantics in MVP |
| WorkArtifact / version | MP-3 | **DEFER** + **MAP** | Shadow/synthesis ≠ WorkArtifact; adopt when durable collaborative output path is scheduled |
| Decision binding / projection | MP-4R | **DEFER** | HITL covers current approval journey |
| ContextView | MP-5 | **DEFER** | Conversation Context covers product thread memory |
| CollaborativeActivity | MP-6 | **DEFER** (read **ADOPT** MP-7F) | No LKW append ownership; observability ≠ Activity |

**NOT APPLICABLE:** AgentDirectory (MP-8), activity feed UI / realtime (MP-9).

## 4. First adoption subset (MP-7B scope)

```text
Principal identity binding (public contracts)
Collaborative Workspace typed reference on LKW workspace (Option B)
CollaborativeWorkEnforcement / effective authority pattern for future mutations
Composition-root provider wiring only (no domain repository imports)
```

## 5. Deferred subset

```text
Membership, Delegation, WorkItem, Assignment, WorkArtifact, Decision binding,
ContextView, CollaborativeActivity read, MP-8, MP-9
```

## 6. Overlap classification (local mechanisms)

| LKW mechanism | Classification |
|---------------|----------------|
| `ManagedWorkspaceRepository` | **PRODUCT-LOCAL** + **ADAPTER/MAPPING** for collaborative ref |
| Product tenant/workspace auth checks | **PRODUCT-LOCAL** (must not substitute platform enforcement) |
| `WorkspaceKnowledgeConfigurationMutationEngine` | **PRODUCT-LOCAL** |
| `Conversation Context` stores | **PRODUCT-LOCAL** |
| Channel/thread correlation IDs | **ADAPTER/MAPPING** |
| Shadow artifact store | **PRODUCT-LOCAL** (future **MAP** to WorkArtifact) |

No **PLATFORM DUPLICATE** authority blessed without migration — binding + strangler only.

## 7. Conversation Context vs ContextView

| Dimension | Conversation Context (LKW) | ContextView (MP-5) |
|-----------|---------------------------|-------------------|
| Ownership | LKW product | Platform / Collaborative Work |
| Lifecycle | Thread/session product rules | Principal-scoped composition snapshots |
| Audience | Product audience policy (`WorkspaceConversationAudience`) | Principal-specific eligible visibility |
| Source of truth | LKW durable conversation stores | Platform ContextView composer + sources |
| Composition | LKW Conversation Interaction stack | `intergrax/contracts/context_view_composition.py` + MP-5D ports |

Relation: optional **adapter/composition** may feed knowledge sources into ContextView later; **no semantic equality**.

## 8. WorkItem vs LKW Task

Nexus `Task` = execution unit (**KEEP LKW-LOCAL**). WorkItem = collaborative lifecycle (**DEFER**). Future link: `WorkItemExecutionLink` / `ExecutionProvenanceRef` only — thread/channel ids must not become WorkItem identity.

## 9. Contract bindings (ADOPT primitives — MP-7B)

### Principal

| Field | Value |
|-------|-------|
| Owner | Collaborative Work / MP-1 |
| LKW role | Consumer |
| Public contract | `intergrax/contracts/collaborative_work.py` (`CollaborativePrincipal`, `PrincipalKind`, enforcement request types) |
| Input mapping | LKW authenticated identity → `LkwPrincipalBinding` → platform principal id |
| Output mapping | Platform principal → LKW presentation only |
| Identity rule | Canonical platform id; LKW holds reference |
| Authority rule | Platform resolver + membership when required |
| Persistence owner | Platform stores |
| Provider visibility | None in LKW domain |
| Failure semantics | Fail closed on missing binding |
| Compatibility | Bootstrap binding in MP-7B; existing workspaces unchanged in MP-7A |

### Collaborative Workspace scope (binding)

| Field | Value |
|-------|-------|
| Owner | MP-1 collaborative workspace scope |
| LKW role | Consumer + product workspace owner |
| Public contract | Scoped ids on `CollaborativeWorkEnforcementRequest` and related DTOs in `intergrax/contracts/collaborative_work.py` |
| Input mapping | `lkw_workspace_id` → lookup `collaborative_workspace_ref` |
| Output mapping | Platform scope ids used only at integration boundary |
| Identity rule | **Explicit typed reference** — not shared string id |
| Authority rule | Platform enforcement |
| Persistence owner | Platform for CW state; LKW for product workspace row + ref |
| Provider visibility | Composition only |
| Failure semantics | No mutation without ref + enforcement |
| Compatibility | Strangler: product-only paths without ref |

## 10. Authority model

```text
authentication source (LKW host)
  → LkwPrincipalBinding
  → platform Principal
  → Membership/Delegation (when adopted)
  → CollaborativeWorkEnforcementRequest
  → platform mutation/read ports
```

## 11. Migration & compatibility

- **MP-7A:** no data migration.
- **Strangler:** add ref → route new consumer paths → remove duplicate authority only with proof.
- **Dual-write:** **FORBIDDEN** (default).
- **Cross-store transactions:** not assumed; failure semantics per-slice in MP-7B+.

## 12. Reference-product E2E scenario (MP-7G)

**User-visible:** Użytkownik pracuje w swoim workspace wiedzy LKW, zadaje pytanie z dowodami (Hybrid Ask), a gdy powstaje trwały, współdzielony rezultat pracy, widzi go w produkcie — przy czym tożsamość użytkownika, uprawnienia i historia działań są utrzymywane przez platformę Multiplayer, nie przez równoległą lokalną bazę uprawnień.

**Technical path:** real LKW HTTP/MCP/Slack-capable route → LKW application service → public Multiplayer contracts → platform provider → projection back to LKW.

## 13. Architecture gaps

| Gap | Classification | Correction slice |
|-----|----------------|------------------|
| LKW host lacks Multiplayer composition wiring | **MISSING COMPOSITION FACADE** | MP-7B |
| No `LkwPrincipalBinding` / `collaborative_workspace_ref` types yet | **MISSING IDENTITY BINDING** | MP-7B |
| Application services in `intergrax/collaborative_work/*` not re-exported as single Tier-3 facade | **MISSING COMPOSITION FACADE** (non-blocking; inject at host like GR-6) | MP-7B docs + wiring |

**BLOCKING ARCHITECTURE GAPS: NONE** for MP-7A closure (gaps scheduled in MP-7B).

## 14. MP-7 decomposition

| Slice | Scope |
|-------|--------|
| **MP-7A** | Architecture / ownership / ADR — **CLOSED** |
| **MP-7B** | Identity + workspace binding + composition enforcement wiring |
| **MP-7C** | Shared Work (deferred until product journey) |
| **MP-7D** | WorkArtifact + Decision binding (deferred) |
| **MP-7E** | ContextView adoption (deferred) |
| **MP-7F** | Collaborative Activity read consumption (deferred) |
| **MP-7G** | E2E reference-product qualification |
| **MP-7H** | Final enterprise adoption certification |

## 15. LKW product roadmap relation

MP-7 is a **parallel platform reference track**. Current direct LKW task remains **`LKW-PLUGIN-CAPABILITY-CONFIGURATION-1`**. MP-7B should not preempt that row without operator decision; MP-7 slices insert after MP-7A audit.

## 16. Validation

```text
uv run pytest tests/unit/collaborative_work/test_mp7a_lkw_adoption_architecture_gates.py -q
git diff --check
```

## 17. Blocking findings

```text
BLOCKING FINDINGS: NONE
```
