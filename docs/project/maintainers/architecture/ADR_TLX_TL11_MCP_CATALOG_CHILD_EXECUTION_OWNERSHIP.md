# ADR — TLX-TL11 MCP Catalog Tool Invocation Child Execution Ownership

| Field | Value |
| --- | --- |
| **Document ID** | `ADR-TLX-TL11-CHILD-OWNERSHIP` |
| **Task** | `TLX-TL11-CHILD-OWNERSHIP-CONTRACT-DESIGN` · `TLX-TL11-CHILD-OWNERSHIP-B1-RATIFICATION-FEASIBILITY` |
| **Operator authorization** | Design: `OPERATOR APPROVED — TLX-TL11 CHILD OWNERSHIP CONTRACT DESIGN ONLY` · Ratification/feasibility: `OPERATOR APPROVED — TLX-TL11 CHILD OWNERSHIP OPTION B — CONDITIONAL ARCHITECTURE RATIFICATION ONLY` |
| **Repository** | `jakbuczarnecki/intergrax` |
| **Branch** | `development` |
| **START_HEAD (design)** | `acff4087123ad9bd1bce1e2a612f633afe222c5f` |
| **START_HEAD (B1 ratification/feasibility)** | `953855f8357ab6f90686e26c26996fbaf79e5b21` |
| **Status** | **OPTION B — CONDITIONALLY RATIFIED (ARCHITECTURE DIRECTION ONLY)** + **B1 FEASIBILITY QUALIFIED @ START_HEAD** |

**Formal result:** `TLX-TL11-CHILD-OWNERSHIP-B1-RATIFICATION-FEASIBILITY — READY FOR INDEPENDENT AUDIT`

**Explicit non-closure:** Not `CHILD OWNERSHIP IMPLEMENTED`; not `P07/CANCEL-C CLOSED`; not `TL11 CLOSED`; not `TOOL-LIFE-X CLOSED`; no Gate A; no TL8 positive; no global `FRZ-*` PASS; no production delta; **no implementation authorization**.

**Conditional ratification is not implementation authorization.**

---

## 1. Problem statement

MCP catalog Tool invocation (`execute_mcp_catalog_tool_invocation`) currently admits work through **root** transport supervision (`admit_protected_mcp_root_execution_supervision` + `run_host_execution_with_mcp_transport_cancel`) while the **canonical root Execution** for the same `task_id` may already be supervised under OWN-A2. Evidence records this as **no sanctioned child-ownership contract** — a second `host_asyncio_task` for the same `task_id` yields `DUPLICATE_HANDOFF_CONFLICT`, which must **not** be interpreted as permission to run catalog business work.

**Objective:** Define the minimal typed, Execution-owned child-work contract so MCP catalog invoke is **governed child work under an already-running root Execution**, reusing inherited deadline/cancellation, without a second root supervisor, shadow dispatcher, or MCP-local lifecycle authority.

---

## 2. Batch A — Closed-world owner inventory

Evidence SHA baseline for design read: **`acff4087123ad9bd1bce1e2a612f633afe222c5f`** (START_HEAD). Prior TL11 packages cite their own publication SHAs (OWN-A2 `75643a97…`, P07-CANCEL-C R2 record on same lineage).

| # | Concern | Semantic owner | Composition owner | Existing contract | Call sites (representative) | Evidence SHA | Gap | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Root Execution identity | `ExecutionIdentityAuthority` / active binding | Host `HostTaskExecutionPort` | `execution_identity` + registry binding | `execute_mcp_agent_task`, `ActiveTaskRegistry` | OWN-A2, U4 @ HEAD | — | **ALREADY PROVEN** (root path) |
| 2 | Root supervision admission | `HostRootExecutionTransportDetachmentSupervisor` | Harness / MCP lifespan coupling | `RootExecutionTransportDetachmentSupervisionPort` | `admit_protected_mcp_root_execution_supervision`, `execute_mcp_agent_task` | `75643a97…` | Catalog invoke must **not** reuse as child admission | **ALREADY PROVEN** (root only) |
| 3 | Root transport detachment | MCP transport waiter + supervisor handoff | `mcp_host_execution_transport_cancel` | OWN-A2 typed handoff | `run_host_execution_with_mcp_transport_cancel` | OWN-A2, CANCEL-A | Child invoke must not imply second detach | **ALREADY PROVEN** (root) |
| 4 | Child execution admission | `ChildExecutionRunner` | `DelegatedSubtaskChildExecutionWorkPort` / `ChildExecutionWorkPort` | Implicit via `ExecutionBoundary` + hooks | U4 production wiring | U4 closure @ HEAD | **No MCP catalog admission path** | **IMPLEMENTATION REQUIRED** |
| 5 | Parent–child lineage | Lineage persistence hooks on runner | Same as child admission | `build_child_lineage_admission_hook` | `ChildExecutionRunner.execute` | U4 tests | MCP must mint child under **live** parent context | **IMPLEMENTATION REQUIRED** |
| 6 | Execution scope | Active execution contextvars + boundary | Host port / Nexus loop | `ExecutionBoundary` | In-process tool paths | EBH-2F lineage | MCP HTTP thread lacks parent scope unless scheduled | **IMPLEMENTATION REQUIRED** |
| 7 | ToolRuntime dispatch | `ExecutionBoundCatalogToolInvoker` | Nexus / host composition | `ExecutionBoundCatalogToolInvokeRequest` | `_sync_invoke_catalog`, `NexusExecutionBoundCatalogToolInvoker` | Gate B L2 | Dispatch OK; **ownership** wrong layer for MCP | **VALIDATION REQUIRED** |
| 8 | Parent deadline | `ExecutionDeadlineAuthority` + projection | Host execution | `ExecutionDeadlineProjection` | `ChildExecutionRunner` narrows child | TL7 DEADLINE_ONLY | Child cannot extend parent | **ALREADY PROVEN** (in-process child) |
| 9 | Child remaining-budget projection | `ChildExecutionRunner` + ledger policy | Work port composition | Budget hooks in runner | `narrow_child_deadline_at_utc`, ledger grant | U4 | Catalog child must use **parent pool**, no new wall budget | **IMPLEMENTATION REQUIRED** |
| 10 | Governed cancellation | `CancellationCoordinator` + `governed_cancel_active_task` | Harness task control | Task metadata contract | MCP cancel, `_sync_invoke_catalog` pre/post checks | CANCEL-C R2 | Child must observe parent cancel; no MCP cancel authority | **ALREADY PROVEN** (cooperative L2) |
| 11 | Business side-effect authorization | Governance collaborative work + `MeaningfulSideEffectAuthorizationPort` | MCP composition inject | `meaningful_side_effect_authorization.py` | `_authorize_meaningful_side_effect_catalog_invoke` | P07 R2 | Revalidate at child admission + bind to child `execution_id` | **VALIDATION REQUIRED** |
| 12 | Child terminal outcome | `ExecutionBoundary` + tool outcome typing | Invoker delegate | `ToolExecutionResult` | Catalog invoker | W4 / tool contracts | Map to child terminal enum in new port | **IMPLEMENTATION REQUIRED** |
| 13 | Parent terminal precedence | Execution terminal CAS / binding state | Host port | `task_state_requires_authoritative_exposure` | `_assert_execution_invoke_admitted` | P07 | Child in-flight when parent terminals → late result rules | **VALIDATION REQUIRED** |
| 14 | Late result | Host + MCP suppression | MCP bridge | Post-invoke cancel check | `_sync_invoke_catalog`, transport waiter | P07 R2-F04 | Physical thread may continue; structured late success suppressed | **ALREADY PROVEN** (L2 cooperative) |
| 15 | Physical sync worker lifetime | Thread pool / `to_thread` | MCP bridge | Protected launch pattern | `create_mcp_protected_host_execution_launch` | P07 | Child design must separate **semantic child owner** from **physical thread** | **OPERATOR DECISION REQUIRED** (thread strategy) |
| 16 | Shutdown and recovery | Supervisor shutdown observe | Harness lifespan | OWN-A2 `shutdown_supervised_work` | MCP drain | OWN-A2 | No durable child registry today — restart semantics fail-closed | **BLOCKED** (crash recovery N/A) |
| 17 | Tenant and principal binding | Verified `RequestIdentity` + registry | MCP trust resolution | TRUST-A patterns | `resolve_mcp_catalog_execution_binding`, `_assert_caller_authorized_for_execution` | TRUST-A | Extend to child admission token | **VALIDATION REQUIRED** |
| 18 | Tracing and observability | Existing execution/trace bindings | Host OTEL wiring | Prior TL11 TRC rows | Nexus / host | TL11 gap recon | Child `execution_id` must appear on spans | **IMPLEMENTATION REQUIRED** |
| 19 | Idempotency and UNKNOWN | `ToolExecutionResult.effect_certainty` + external op store | Invoker / W4-C | Tool outcome contracts | Serialize in MCP bridge | W4 / P07 | No autonomous retry on UNKNOWN | **ALREADY PROVEN** (policy); **VALIDATION REQUIRED** (child port) |

### U4 delegated-subtask seam — reuse assessment

| Aspect | U4 `DelegatedSubtaskChildExecutionWorkPort` | MCP catalog Tool invoke |
| --- | --- | --- |
| Parent context | In-process async parent on same loop | Often separate MCP HTTP request while root runs |
| Work shape | `DelegatedSpecialistChildWorkEnvelope` + specialist delegate | Sync/heavy `ExecutionBoundCatalogToolInvoker.invoke` |
| Scheduling | Direct `ChildExecutionRunner` | Today: spurious second root `host_asyncio_task` |
| Lifecycle | Specialist completion = child terminal | Tool timeout/cancel/UNKNOWN semantics |

**Conclusion:** U4 is **reference architecture** for **who** owns child admission (`ChildExecutionRunner`), not a drop-in port. Reuse **pattern** via new typed envelope + delegate; **do not** assume delegated-subtask lifecycle identity.

**Root vs child:** OWN-A2 owns **one supervised root `host_asyncio_task` per admitted root Execution**. Catalog invoke must register as **child work under that root's Execution identity**, not a competing root supervision slot.

---

## 3. Batch B — Canonical architecture options

### Option A — Reuse existing `ChildExecutionWorkPort` / runner path directly

| Dimension | Assessment |
| --- | --- |
| Exactly-one owner | `ChildExecutionRunner` (semantic child) — **yes** |
| New contract delta | Thin adapter: `ExecutionRequest[CatalogToolChildInput, ToolExecutionResult]` |
| Compatibility | ToolRuntime unchanged; MCP must schedule onto parent's execution loop |
| Composition | New `build_production_catalog_tool_child_work_port` mirroring U4 wiring |
| Semantic risks | HTTP handler not on parent loop → admission without parent context fails or lies |
| Validation | Full matrix §9; loop hop mechanism must be proven |
| Advantages | Minimal new types; maximal reuse of deadline/budget/lineage |
| Disadvantages | Cross-loop scheduling is non-trivial; easy to get TOCTOU wrong |
| Current evidence | U4 @ HEAD — in-process only |
| **Recommendation** | **Subsumed by Option B** (A lacks explicit MCP admission contract) |

### Option B — Extend Execution-owned typed child contract (catalog-specific admission)

| Dimension | Assessment |
| --- | --- |
| Exactly-one owner | **`ChildExecutionRunner`** remains semantic child owner; **`CatalogToolChildExecutionPort`** (new contract) owns **MCP admission + loop routing** only |
| New contract delta | `CatalogToolChildAdmissionRequest` / `Result`, refusal reasons, `CatalogToolChildExecutionPort.admit_and_execute` |
| Compatibility | Keeps `ExecutionBoundCatalogToolInvoker`; removes OWN-A2 from catalog invoke path |
| Composition | Host wires port with parent-loop scheduler + existing invoker |
| Semantic risks | Mis-wiring port as root supervision reintroduces DUPLICATE_HANDOFF |
| Validation | L2 full matrix + L3 subprocess HTTP invoke/cancel under active root |
| Advantages | Explicit fail-closed admission; testable boundaries; FRZ-CTR alignment |
| Disadvantages | New contract surface + composition glue |
| Current evidence | None for catalog child — **design only** |
| **Recommended option** | **Yes — for implementation planning** (not ratified here) |

### Option C — Fail-closed catalog invocation

| Dimension | Assessment |
| --- | --- |
| Exactly-one owner | N/A — deny path |
| New contract delta | Feature flag or explicit `REFUSED` until Option B shipped |
| Compatibility | Discovery remains; invoke returns `execution_admission_denied` / `child_ownership_unavailable` |
| Composition | Remove or guard `admit_protected_mcp_root_execution_supervision` on catalog path |
| Semantic risks | Operational gap until B lands |
| Validation | Negative tests only |
| Advantages | Safest interim if B delayed |
| Disadvantages | Blocks Gate B invoke qualification |
| Current evidence | Partially reflected in DUPLICATE_HANDOFF behavior |
| **Recommendation** | **Interim acceptable** if implementation batch splits; not target end state |

**Not acceptable:** New MCP-local coordinator, second root supervisor, or automatic worker takeover.

---

## 4. Batch C — Child contract specification (recommended: Option B)

### 4.1 Conceptual types (contracts tier)

```text
CatalogToolChildInvocationId     # stable per logical invoke (idempotency_key or minted UUID)
CatalogToolChildAdmissionRequest
CatalogToolChildAdmissionRefusalReason (enum)
CatalogToolChildAdmissionResult    # ADMITTED | REFUSED
CatalogToolChildWorkRequest        # wraps ExecutionBoundCatalogToolInvokeRequest + admission evidence
CatalogToolChildWorkResult         # terminal + ToolExecutionResult + child_execution_id
CatalogToolChildExecutionPort      # admit_and_execute(...)
```

### 4.2 Admission identity (required fields)

| Field | Source | Rule |
| --- | --- | --- |
| Verified caller principal | MCP `RequestIdentity` | Must match task owner (`_assert_caller_authorized_for_execution`) |
| Tenant | Resolved harness identity | Must match `ActiveTaskBinding` |
| Root execution ID | Active parent execution context on root loop | **Not** minted by MCP bridge |
| Task / run / attempt | Registry binding + active governance peek | Stale run/attempt → REFUSED |
| Parent authority evidence | `require_active_execution_id()` on root loop | MCP thread must not admit without root-scheduled check |
| Child invocation ID | Client `idempotency_key` or port-minted | Deduplication registry (in-memory per parent) |
| Agent / Tool ID | Binding metadata + registry | Scope mismatch → REFUSED |
| Bound step | Metadata canonical step | Stale step → REFUSED |
| Tool scope | Registry + enablement | Unchanged from Gate B |
| Governed authorization | `MeaningfulSideEffectAuthorizationPort` | At admission; bind `execution_id` to **child** id before physical invoke |

### 4.3 Ownership rules

- **Admit child work:** `CatalogToolChildExecutionPort` only after root Execution is **live** and **terminal-free** on binding.
- **Child lifetime:** `ChildExecutionRunner` + ledger release in `finally` (existing).
- **Registration:** Lineage hook + optional durable child table (implementation — default in-memory dedupe).
- **Duplicate admission:** Same `CatalogToolChildInvocationId` → `DUPLICATE_CHILD` (idempotent return of terminal result if completed).
- **Inactive proof:** Ledger child release + lineage segment closed + dedupe map terminal entry.
- **Durability:** Not required for TL11; restart → **fail-closed** (no orphan authority).
- **Parent terminates:** Child admission REFUSED; in-flight child receives cooperative cancel; late results → UNKNOWN / suppressed per P07.

### 4.4 Deadline / cancellation

- Child deadline ⊆ parent (`narrow_child_deadline_at_utc` — reuse runner).
- Cancellation: only via parent `CancellationCoordinator` on task metadata — **no** MCP transport cancel as business cancel.
- Transport disconnect on **child MCP request** may end waiter only if child work is supervised on root loop; otherwise **fail-closed**.
- Late results: terminal precedence — parent terminal wins; structured success suppressed if cancel observed post-invoke.
- **No** autonomous retry on UNKNOWN.

### 4.5 Results

- Canonical terminals align with TOOL-LIFE-X: `SUCCEEDED`, `FAILED`, `TIMED_OUT`, `CANCELLED`, `ABANDONED`, `UNCERTAIN_EFFECT`.
- Child completion carries `child_execution_id` for audit/trace.
- Failure propagates as `ToolExecutionResult` inside child boundary.

---

## 5. Batch D — Atomicity and concurrency

### 5.1 Linearization points

| Event | Linearization owner | Notes |
| --- | --- | --- |
| Child admission | Root-loop mutex + `ChildExecutionRunner` admission hooks | MCP thread posts coroutine to root loop; admission runs **on root loop** |
| Parent cancel vs child admit | Task metadata write ordering under mutation boundary | Cancel observed → REFUSED at admit |
| Parent terminal vs child complete | Terminal CAS before child result commit | Parent terminal wins |
| Duplicate child requests | Per-parent dedupe map under same mutex | Idempotent terminal replay |
| Transport disconnect | MCP waiter | May return after scheduling child; **only if** root supervisor retains child |
| Supervision handoff | OWN-A2 — **root only** | Catalog path must not call handoff |
| Child termination | Runner `finally` + ledger release | Idempotent cleanup |
| Shutdown | Supervisor refuses new root; child port refuses new admits | Observe in-flight |

### 5.2 TOCTOU / fail-closed

- Registry binding read on MCP thread is **hint only**; authoritative admit on root loop re-reads binding + parent execution context.
- `DUPLICATE_HANDOFF_CONFLICT` **≠** `ACCEPTED` — never start catalog work on refused root handoff.
- If child cannot be scheduled onto supervised root loop → **REFUSED** (`child_supervision_unavailable`).

### 5.3 Transport waiter return while work continues

Allowed **only when** child coroutine is enqueued on the **already supervised** root `host_asyncio_task` loop and MCP response is decoupled (async job pattern) **or** cooperative wait with bounded timeout. Current synchronous MCP tool response path may block until child terminal — implementation must not return success after losing supervision.

---

## 6. Batch E — Governance and side-effect safety

| Phase | Port | Rule |
| --- | --- | --- |
| Child admission (side-effect tools) | `MeaningfulSideEffectAuthorizationPort` | Build `CollaborativeWorkEnforcementRequest` with **child** `execution_id` after mint |
| Physical invoke | Bound invoker inside child boundary | Re-check cancel + deadline immediately pre-effect |
| Stale authorization | Governance identity mismatch vs binding | Fail closed (`unauthorized_side_effect_tool`) |
| Denial | Propagate as tool error before thread launch | No partial MCP success |
| Post-cancel / post-terminal | Protected work admission + cancel coordinator | Deny new effects |
| UNKNOWN external effect | `ToolExecutionResult.effect_certainty` | No conflation with provider cancel rollback |

**No second Governance policy system.**

---

## 7. Batch F — Adversarial test matrix (pre-implementation)

Abbreviated catalog — full rows in implementation batch must expand to pytest modules listed in §8.

| Invariant | Test type | Owner | Injection | Expected | Existing evidence | Layer | FRZ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Valid child under active root | positive | Child port | MCP fixture + live root | ADMITTED + result | — | L2/L3 | FRZ-EXE, FRZ-BND-03 |
| Duplicate child | negative | Dedupe map | repeat idempotency_key | idempotent terminal | — | L2 | FRZ-REL |
| Missing parent | negative | No root task | invoke only | REFUSED | P07 partial | L2 | FRZ-OWN |
| Wrong principal same tenant | negative | forged user | TRUST | REFUSED | TRUST-A | L2 | FRZ-TEN, FRZ-SEC |
| Wrong tenant | negative | tenant skew | binding | REFUSED | P07 | L2 | FRZ-TEN |
| Stale run/attempt | negative | bad run_id | MCP fields | REFUSED | P07 | L2 | FRZ-EXE |
| Wrong agent/tool scope | negative | metadata | composition | REFUSED | Gate B | L2 | FRZ-GOV |
| Root terminal | negative | terminal state | registry | REFUSED | P07 | L2 | FRZ-EXE |
| Parent cancel before admit | race | cancel coordinator | ordering | REFUSED | CANCEL-C | L2 | FRZ-GOV |
| Parent cancel during admit | race | boundary | loop hop | REFUSED | — | L2 | FRZ-REL |
| Parent cancel during invoke | race | in-flight thread | cancel metadata | cancelled + UNKNOWN | P07 R2-F04 | L2/L3 | FRZ-REL-04 |
| Child vs root deadline | boundary | clock | narrow child | child ⊆ parent | U4 runner | L2 | FRZ-EXE |
| Child timeout vs tool success | negative | short budget | slow tool | TIMED_OUT | — | L2 | FRZ-EXE |
| Transport disconnect | negative | drop HTTP | waiter | bounded error | CANCEL-A | L3 | FRZ-SEC |
| Root supervisor conflict | negative | second host task | OWN-A2 | REFUSED — **no catalog work** | OWN-A2 | L2 | FRZ-OWN-04 |
| Refused handoff | negative | handoff REFUSED | — | no ACCEPT inference | OWN-A2 | L2 | FRZ-REL |
| Non-cooperative sync tool | physical | blocking thread | cancel | cooperative deny + UNKNOWN | P07 | L2/L3 | FRZ-REL-04 |
| Late tool result | negative | delay | post-cancel | suppress success | P07 | L2 | FRZ-REL |
| Effect committed ack lost | uncertainty | fault inj. | — | UNCERTAIN_EFFECT | W4 policy | L2 | FRZ-REL |
| Governance DENY | negative | policy | MSE port | denied | P07 | L2 | FRZ-GOV |
| Stale Governance auth | negative | identity drift | peek | denied | — | L2 | FRZ-GOV |
| Tenant replay | negative | cross-tenant | replay | REFUSED | TRUST-A | L2 | FRZ-TEN |
| Concurrent child finalization | race | parallel completes | dedupe | single terminal | — | L2 | FRZ-REL |
| Shutdown during handoff | negative | lifespan | drain | REFUSED | OWN-A2 | L2 | FRZ-REL |
| Restart incomplete child | negative | n/a | — | fail-closed | SA-21 | L2 | FRZ-REC |
| Resource exhaustion | negative | limits | — | REFUSED | — | L2 | FRZ-REL |
| No duplicate external effect | idempotency | same key | — | single effect | — | L2/L3 | FRZ-REL |
| No shadow dispatcher | arch | static | — | single owner | gap recon | L1 | FRZ-OWN-04 |
| No new authority | arch | static | — | no MCP cancel owner | CANCEL-A | L1 | FRZ-GOV |

**L3 mandate:** Real MCP subprocess HTTP catalog invoke + governed cancel with active root — extend `tests/fixtures/tool_life_x_tl11_catalog_subprocess_qualification.py` (not mock-only for physical cancel).

---

## 8. Batch G — Implementation plan (not authorized)

| File | Class | Depends on |
| --- | --- | --- |
| `intergrax/contracts/catalog_tool_child_execution.py` | **New typed contract** | execution_identity, tool invoke request |
| `intergrax/runtime/execution/catalog_tool_child_execution_port.py` | **Existing owner extension** | `ChildExecutionRunner`, scheduler |
| `intergrax/applications/_shared/production_catalog_tool_child_execution_wiring.py` | **Composition** | invoker + MSE port |
| `intergrax/applications/_shared/mcp_catalog_invocation_bridge.py` | **Adapter** | remove OWN-A2 admit on invoke; call child port |
| `intergrax/applications/_shared/mcp_nexus_server.py` | **Composition** | wire child port |
| `tests/unit/runtime/execution/test_catalog_tool_child_execution.py` | **Tests** | matrix §7 |
| `tests/unit/applications/test_mcp_catalog_child_ownership_qualification.py` | **Tests** | MCP + races |
| `tests/fixtures/tool_life_x_tl11_catalog_subprocess_qualification.py` | **Evidence** | L3 |
| `docs/.../TOOL_LIFE_X_TLX_TL11_*_EVIDENCE.md` | **Evidence** | post-impl only |

**Parallelism:** Contract + runner adapter first; MCP bridge second; L3 last.

**Qualification boundary:** One batch closes child ownership L2 matrix + L3 invoke/cancel; does not close Gate A / TL8 / TOOL-LIFE-X parent.

---

## 9. Closure matrix (gate before implementation)

| Exit criterion | Owner | Contract | Evidence @ START_HEAD | Missing action | Tests | FRZ | Class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exactly-one child semantic owner | `ChildExecutionRunner` | New catalog child port | U4 @ HEAD | Implement B | §7 | FRZ-OWN-01..05 | **IMPLEMENTATION REQUIRED** |
| Root/child separation | OWN-A2 vs child port | ADR §2 | P07, OWN-A2 | Remove root admit on catalog | §7 | FRZ-BND-03 | **IMPLEMENTATION REQUIRED** |
| Authorization continuity | MSE port | existing + child bind | P07 R2 | Rebind child id | §7 | FRZ-GOV | **VALIDATION REQUIRED** |
| Tenant isolation | TRUST-A + binding | registry | TRUST-A | child admit checks | §7 | FRZ-TEN | **VALIDATION REQUIRED** |
| Deadline inheritance | Runner | projection | U4 | wire catalog child | §7 | FRZ-EXE | **IMPLEMENTATION REQUIRED** |
| Governed cancellation | Coordinator | task metadata | CANCEL-C | root-loop observe | §7 | FRZ-GOV | **ALREADY PROVEN** (root) |
| Child lifetime supervision | Child port + root loop | new | — | scheduler | §7 | FRZ-REL | **IMPLEMENTATION REQUIRED** |
| No orphan threads | Protected launch policy | MCP | P07 | align with child owner | L3 | FRZ-REL-04 | **VALIDATION REQUIRED** |
| Admission linearization | Root-loop mutex | new port | — | implement | §7 | FRZ-REL | **IMPLEMENTATION REQUIRED** |
| Terminal race safety | Terminal CAS | host | gap recon | child tests | §7 | FRZ-EXE | **VALIDATION REQUIRED** |
| UNKNOWN / side-effect safety | Tool result + MSE | existing | P07 | child binding | §7 | FRZ-REL | **VALIDATION REQUIRED** |
| Restart semantics | — | — | SA-21 N/A | fail-closed doc | negative | FRZ-REC | **BLOCKED** (by policy) |
| L2 qualification | QA | — | P07 partial | full matrix | new suite | FRZ-REG | **IMPLEMENTATION REQUIRED** |
| L3 physical MCP | QA | — | NOT PROVEN | subprocess | fixture | FRZ-SEC | **IMPLEMENTATION REQUIRED** |
| ECAP | Process | ECAP-1 | protocol | independent audit | — | — | **VALIDATION REQUIRED** |
| Independent audit SHA | Operator | — | START_HEAD | publish + audit | — | — | **VALIDATION REQUIRED** |

---

## 10. ECAP / FRZ (applicable IDs — statuses unchanged)

Design completion does **not** promote any `FRZ-*` row. Applicable checklist families for future implementation evidence:

- **FRZ-BND-03** — tier boundaries; no MCP shadow Execution owner
- **FRZ-OWN-01..05** — exactly-one child owner; no duplicate mechanism
- **FRZ-CTR-01..06** — typed contracts; no dict pseudo-contracts at admission boundary
- **FRZ-TYP-***, **FRZ-EXE-*** — execution lifecycle typing
- **FRZ-GOV-*** — MSE authorization continuity
- **FRZ-REL-***, **FRZ-REC-*** — cancellation, late results, restart fail-closed
- **FRZ-SEC-***, **FRZ-TEN-*** — principal/tenant binding
- **FRZ-STA-***, **FRZ-REG-*** — qualification registration after tests exist

---

## 11. Operator decisions remaining

1. **Cross-loop scheduling strategy** — **OPERATOR DECISION REQUIRED** before implementation: no Execution-owned cross-loop child admission mechanism is proven @ `953855f8357…` (§14–§15). Default interim: **Option C fail-closed** for independent MCP HTTP catalog invoke; in-process child path only where parent Execution context is genuine (§15 B2).
2. **Child invocation durability** — operator-approved default: in-memory dedupe per live parent lifecycle only; restart **fail-closed** (no exactly-once external-effect replay).
3. **Interim Option C** — **ratified** until `CatalogToolChildExecutionPort` is implemented and L2/L3 qualified.
4. **Physical thread policy** — retain bounded `to_thread` only under child semantic owner + protected launch; transport disconnect ≠ physical termination (P07 R2-F04).

---

## 12. Batch A — Conditional Option B ratification record (@ `953855f8357ab6f90686e26c26996fbaf79e5b21`)

| Decision element | Disposition |
| --- | --- |
| **Ratified child semantic owner** | **`ChildExecutionRunner`** — sole canonical owner of child Execution identity, boundary admission, deadline narrowing, and ledger release (`intergrax/runtime/execution/child.py`). |
| **Ratified typed adapter direction** | **`CatalogToolChildExecutionPort`** (contract **not yet present**) — MCP/catalog **admission + loop routing only**; strict adapter to existing Execution authority; **not** a second lifecycle owner. |
| **Approved fallback policy** | In-process-only child work when active parent Execution context is genuine; otherwise **fail-closed**. No orphan threads, new scheduler, or new supervisor. In-memory dedupe scoped to live parent lifecycle; restart fail-closed. |
| **Unproven cross-loop mechanism** | No sanctioned callable @ HEAD posts catalog child admission onto the **already-supervised root** `host_asyncio_task` loop with atomic parent-context admission (§14). |
| **Unproven physical lifetime ownership** | MCP `asyncio.to_thread` + transport waiter (P07) proves cooperative L2 suppression only — not child-runner-owned physical supervision for catalog path. |
| **Unimplemented adapter contract** | `CatalogToolChildExecutionPort` / `CatalogToolChildAdmissionRequest` — design types in §4 only; **zero production modules**. |
| **Unqualified L2/L3 MCP integration** | External MCP HTTP catalog invoke remains **Option C fail-closed** until implementation batch + adversarial L2 + real subprocess L3. |

Prior §2–§10 design findings and Option A/B/C comparisons are **preserved** and are **not** relabeled as implementation PASS.

---

## 13. Batch B — Closed-world feasibility audit (@ `953855f8357…`)

Evidence baseline: **`953855f8357ab6f90686e26c26996fbaf79e5b21`**. Capability classes are **distinct** — context access ≠ context propagation ≠ task scheduling ≠ child admission ≠ Execution ownership transfer.

### 13.1 Candidate mechanisms

| # | Owner → Contract → Composition → Caller | Execution context | Transfer | Auth check | Linearization | Evidence SHA | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | `ChildExecutionRunner` → implicit boundary hooks → `ChildExecutionWorkPort` / U4 wiring → in-process specialist/Nexus delegate | **Active parent** via `require_active_execution_*` on **same asyncio task** | ContextVar chain on parent task | `DefaultStrictAuthorityPolicy` + governance peek | Runner `execute` body before `ExecutionBoundary` | `953855f8357…` `child.py` L153–156 | **In-process child admission PROVEN** — not cross-loop |
| C2 | `ExecutionRuntime` → `ActiveExecutionWorkPortBinding` → `bind_active_execution_work_port` during root `boundary.execute` | Active root only inside bound execute | ContextVar set/reset in `runtime.py` L552–555 | `require_active()` port identity match | Bind token lifetime = root execute | `953855f8357…` | **Context propagation only** — not MCP cross-loop admission |
| C3 | `HostRootExecutionTransportDetachmentSupervisor` → `RootExecutionTransportDetachmentSupervisionPort` → MCP bridge `admit_protected_mcp_root_execution_supervision` | **New** `host_asyncio_task` for catalog invoke | Spawns/competes with existing root task | Verified `RequestIdentity` + task binding | Supervisor mutex + admission enum (`ACCEPTED` only opens gate) | OWN-A2 @ lineage; bridge L504–510 | **Root admission only** — **not** child under live root; duplicate → `DUPLICATE_HANDOFF_CONFLICT` |
| C4 | `ActiveTaskRegistry` → `ActiveTaskBinding` → `resolve_mcp_catalog_execution_binding` | Registry metadata on MCP thread | None (lookup) | Tenant/run alignment | `asyncio.Lock` on registry mutations | `active_task_registry.py` L27–28 | **Hint only** — not admission |
| C5 | `ExecutionBoundCatalogToolInvoker` → `NexusExecutionBoundCatalogToolInvoker` → `_sync_invoke_catalog` / `to_thread` | Optional active identity for MSE (`peek_*`); often **absent** on MCP thread | Thread pool | MSE + scope asserts pre-launch | Pre-launch checks; post-invoke cancel check | P07 R2 | **Tool dispatch** — wrong ownership layer for MCP catalog today |
| C6 | Ad hoc `asyncio.run_coroutine_threadsafe` (vendor/export/resilience) | Unrelated domains | Foreign loops | N/A | Per-module | `953855f8357…` (not on catalog path) | **Not Execution-owned catalog child seam** |

### 13.2 Verified current MCP catalog call graph (@ HEAD)

```text
execute_mcp_catalog_tool_invocation (MCP HTTP handler thread / MCP server loop)
  → resolve_mcp_catalog_execution_binding (registry hint)
  → admit_protected_mcp_root_execution_supervision (OWN-A2 root — NOT child)
  → run_host_execution_with_mcp_transport_cancel
  → asyncio.to_thread(_sync_invoke_catalog) → ExecutionBoundCatalogToolInvoker.invoke
```

**No** call site reaches `ChildExecutionRunner` or `ChildExecutionWorkPort` from this graph.

---

## 14. Batch C — Feasibility disposition

| Outcome | Verdict @ `953855f8357…` |
| --- | --- |
| **B1 — Existing Execution-owned cross-loop mechanism** | **NOT PROVEN** — no callable entry with trusted parent-owned submission, root-context atomic child admission, and callback ownership without new scheduler/supervisor. |
| **B2 — In-process-only feasible** | **PROVEN** — U4 production path: `build_production_delegated_subtask_child_execution_port` → `DelegatedSubtaskChildExecutionWorkPort.execute` → `ChildExecutionRunner.execute` under active parent (`production_delegated_subtask_child_execution_wiring.py`, `execution_work_port.py` L128–188). |
| **B3 — Neither path** | **Rejected** — B2 satisfies in-process canonical child ownership. |

**Final disposition: B2 (in-process-only)** with **external independent MCP HTTP catalog invoke remaining Option C fail-closed** until `CatalogToolChildExecutionPort` and a **new, explicitly authorized** cross-loop submission contract are implemented and qualified.

**Smallest missing canonical artifact for MCP child invoke:** typed **`CatalogToolChildExecutionPort.admit_and_execute`** plus a **proven root-loop scheduler** (not present @ HEAD; not inferable from `ActiveExecutionWorkPortBinding` or registry lookup).

---

## 15. Batch D — Child admission and concurrency invariants (required contract)

| Invariant | Authoritative linearization (required) | Existing @ HEAD |
| --- | --- | --- |
| Parent verified identity | `require_active_execution_identity()` on admission coroutine | **Proven** in `ChildExecutionRunner` (in-process) |
| Parent live / non-terminal | `task_state_requires_authoritative_exposure` + binding re-read on admit | **Proven** on MCP precheck (`_assert_execution_invoke_admitted`); child port must re-assert on root loop |
| Root ownership continuity | Single supervised root `host_asyncio_task` per task (OWN-A2) | **Proven** for root; catalog must not add second root task |
| Child identity minting | `mint_child_execution_id()` inside runner | **Proven** in-process |
| Child admission under parent | `ExecutionBoundary` + lineage hook before delegate | **Proven** in-process; **not** from MCP HTTP |
| Duplicate submissions | Per-parent dedupe map (implementation) | **Not implemented** for catalog |
| Parent cancel vs admit | `CancellationCoordinator` + mutation boundary ordering | **Proven** cooperative checks on MCP path; child port must refuse if cancel set |
| Root terminal vs child complete | Terminal CAS / `task_state_requires_authoritative_exposure` | **Partial** — host terminal exposure proven; child terminal race tests **validation required** |
| Cross-loop submission | Root-loop mutex + posted coroutine | **Required by design** — **missing @ HEAD** |
| Shutdown during admit | Supervisor refuses new root; child port refuses new admits | **Partial** — root drain (OWN-A2); catalog child port N/A |
| Worker after transport disconnect | Protected launch + UNKNOWN / suppress late success | **L2 cooperative** (P07) — not child-owner physical proof |
| Restart / repeated request | No durable dedupe → **fail-closed** | **Policy ratified** — no code claims replay safety |

**Atomicity:** runner + boundary hooks provide in-process atomicity today; cross-loop admit requires **new** root-loop linearization (design §5.1) — **not claimed as existing**.

---

## 16. Batch E — Cancellation, deadlines, physical lifetime (audit)

| # | Question | @ HEAD (catalog path) | Under Option B target |
| --- | --- | --- | --- |
| 1 | Parent cancel reaches child? | Cooperative metadata via `CancellationCoordinator`; pre/post checks in bridge | Child must observe same coordinator on root loop inside runner |
| 2 | Child inherits parent deadline? | N/A (no catalog child) | **Proven** in-runner via `narrow_child_deadline_at_utc` (U4) |
| 3 | Owner after MCP waiter cancel? | `to_thread` worker may continue (R2-F04) | Child semantic owner must define suppression; physical stop **not proven** |
| 4 | Worker after coroutine end? | **Yes** possible | Classify **VALIDATION REQUIRED** / L3 |
| 5 | Who observes real completion? | Invoker + post-invoke cancel check | Child terminal + MCP bridge mapping |
| 6 | Safe transport response? | After waiter outcome; late success suppressed | Must not return success after losing child supervision |
| 7 | Cancel with external effects possible? | UNKNOWN preserved | `ToolExecutionResult.effect_certainty` |
| 8 | UNKNOWN preserved? | **Yes** (P07 policy) | Revalidate on child port |
| 9 | Late result promotion prevented? | `_sync_invoke_catalog` post-cancel | Terminal precedence + child boundary |
| 10 | Shutdown? | OWN-A2 drain | Child port refuse new admits |

**Classification:** Physical lifetime supervision for catalog sync tools remains **BLOCKED for PASS** without L3 + explicit child-owner alignment; **Option C preserved** for external invoke.

---

## 17. Batch F — Security, governance, tenant isolation

MSE for catalog invoke today requires **genuine** active execution governance identity on the admitting thread (`_build_catalog_side_effect_enforcement_request` — returns `None` → deny if `peek_active_execution_*` missing). **Metadata-only admission is insufficient.**

Under Option B, final pre-effect check must run under **child** `execution_id` minted by `ChildExecutionRunner`, with stale authorization detected via governance identity mismatch vs binding (fail-closed). Verified MCP principal → caller authorization (`_assert_caller_authorized_for_execution`) must chain into child admission token on root loop.

---

## 18. Batch G — Implementation closure matrix (@ `953855f8357…`)

| Criterion | Semantic owner | Existing contract | Evidence SHA | Missing change | Test | FRZ | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exactly-one child owner | `ChildExecutionRunner` | U4 work port | `953855f8357…` | Catalog child port | U4 + new | FRZ-OWN-01..05 | **ALREADY PROVEN** (in-process) |
| Parent-child separation | Runner vs OWN-A2 root | ADR §2 | OWN-A2, P07 | Remove root admit on catalog invoke | §7 | FRZ-BND-03 | **IMPLEMENTATION REQUIRED** |
| Trusted caller | MCP TRUST + binding | RequestIdentity | P07 R2-F05 | Child admit token | §7 | FRZ-SEC, FRZ-TEN | **VALIDATION REQUIRED** |
| Root authority continuity | OWN-A2 supervisor | Root port | `953855f8357…` | No second root task | OWN-A2 tests | FRZ-OWN-04 | **ALREADY PROVEN** (root) |
| Cross-loop feasibility | — | — | §14 | Scheduler + port | — | FRZ-EXE | **BLOCKED** (B1 not proven) |
| In-process fallback | `ChildExecutionWorkPort` | execution_work_port | U4 | Wire catalog delegate | U4 tests | FRZ-EXE | **ALREADY PROVEN** |
| Child admission linearization | Child port (new) | — | — | Root-loop mutex | new | FRZ-REL | **IMPLEMENTATION REQUIRED** |
| Deadline inheritance | Runner | projection | U4 | Catalog wiring | U4 | FRZ-EXE | **ALREADY PROVEN** (pattern) |
| Governed cancellation | Coordinator | task metadata | CANCEL-C | Child observe | §7 | FRZ-GOV | **VALIDATION REQUIRED** |
| Effect authorization | MSE port | meaningful_side_effect | P07 | Child execution_id bind | §7 | FRZ-GOV | **VALIDATION REQUIRED** |
| In-memory dedupe scope | Child port | — | — | Per-parent map | §7 | FRZ-REL | **IMPLEMENTATION REQUIRED** |
| Restart fail-closed | Policy | — | SA-21 | Document + refuse | negative | FRZ-REC | **OPERATOR DECISION REQUIRED** (ratified default) |
| Thread lifetime | Protected launch | MCP launch | P07 | Align with child owner | L3 | FRZ-REL-04 | **BLOCKED** until L3 |
| No orphan work | OWN-A2 + runner finally | — | U4/P07 | Catalog path | L3 | FRZ-REL | **VALIDATION REQUIRED** |
| Late result / UNKNOWN | Bridge + tool result | W4/P07 | P07 | Child mapping | §7 | FRZ-REL | **VALIDATION REQUIRED** |
| Real subprocess MCP invoke | QA | — | NOT PROVEN | L3 fixture | fixture | FRZ-SEC | **IMPLEMENTATION REQUIRED** |
| L3 cancel / disconnect | QA | CANCEL-A | NOT PROVEN | HTTP L3 | fixture | FRZ-SEC | **IMPLEMENTATION REQUIRED** |
| ECAP | Process | ECAP-1 | protocol | Independent audit | — | — | **VALIDATION REQUIRED** |
| Independent exact-SHA audit | Operator | — | `953855f8357…` | Publication | — | FRZ-REG | **VALIDATION REQUIRED** |

---

## 19. Batch H — Future implementation package (not authorized)

| File | Class | Why | Expected change | Preceding contract | FRZ | New operator auth? |
| --- | --- | --- | --- | --- | --- | --- |
| `intergrax/runtime/execution/child.py` | **Existing canonical owner** | Semantic child | None without impl approval | — | FRZ-OWN | No |
| `intergrax/contracts/catalog_tool_child_execution.py` | **New typed adapter contract** | Admission API | Add types + port protocol | B1 ratification accepted | FRZ-CTR | **Yes** (implementation batch) |
| `intergrax/runtime/execution/catalog_tool_child_execution_port.py` | **Authorized composition** | Loop routing + runner | Implement `admit_and_execute` | Contract + scheduler decision | FRZ-BND-03 | **Yes** |
| `intergrax/applications/_shared/production_catalog_tool_child_execution_wiring.py` | **Composition** | Host wiring | Mirror U4 pattern | Port | FRZ-OWN-03 | **Yes** |
| `mcp_catalog_invocation_bridge.py` | **MCP adapter** | Remove OWN-A2 on invoke | Call child port; fail-closed if unavailable | Port + L2 | FRZ-BND-03 | **Yes** |
| `mcp_nexus_server.py` | **Composition** | Inject port | Wire MSE + child port | — | — | **Yes** |
| `test_catalog_tool_child_execution.py` | **New adversarial tests** | Matrix §7 | Full L2 races | Port | FRZ-REG | No (within impl auth) |
| `test_mcp_catalog_child_ownership_qualification.py` | **New tests** | MCP integration | HTTP + root live | L3 fixture | FRZ-SEC | No |
| `tool_life_x_tl11_catalog_subprocess_qualification.py` | **Existing subprocess fixture** | L3 proof | Extend invoke/cancel | L3 plan | FRZ-SEC | No |
| P07 / ADR evidence docs | **Evidence** | Publication | Post-impl only | L2/L3 PASS | — | No |

**Prohibited:** editing contracts or production modules before independent acceptance of this feasibility package **and** separate operator implementation authorization.

---

## 20. Validation selection (L1/L2/L3)

| Layer | This task | Subsequent implementation |
| --- | --- | --- |
| **L1** | Documentation integrity + `git diff --check` @ publication SHA | Same + arch static gates |
| **L2** | N/A (no behavior delta) | Full §7 adversarial matrix + gate B extensions |
| **L3** | N/A | Mandatory real MCP subprocess HTTP catalog invoke + cancel under active root |

Historical P07 L2 counts are **not** relabeled as current child-ownership feasibility proof.

---

## 21. ECAP / FRZ (@ checklist HEAD — statuses unchanged)

Applicable IDs: **FRZ-BND-03**, **FRZ-OWN-01..05**, **FRZ-CTR-01..06**, **FRZ-TYP-***, **FRZ-EXE-***, **FRZ-GOV-***, **FRZ-REL-***, **FRZ-REC-***, **FRZ-SEC-***, **FRZ-TEN-***, **FRZ-STA-***, **FRZ-REG-*** — all remain **OPEN** at platform scope; this package contributes scoped evidence only.

**Tenant isolation:** single-owner assessment — no MCP shadow Execution owner ratified; child semantic owner remains `ChildExecutionRunner` only.

---

## 22. References

- `docs/project/maintainers/qualification/TOOL_LIFE_X_TLX_TL11_GATE_B_P07_CANCEL_C_QUALIFICATION_EVIDENCE.md` (reconciled §Child ownership)
- `docs/project/maintainers/qualification/PLATFORM_EXECUTION_UNIFICATION_U4_CHILD_EXECUTION_CLOSURE.md`
- `docs/project/maintainers/qualification/TOOL_LIFE_X_TLX_TL11_OWN_A2_SUPERVISED_EXECUTION_HANDOFF_QUALIFICATION_EVIDENCE.md`
- `intergrax/contracts/root_execution_transport_detachment_supervision.py`
- `intergrax/runtime/execution/child.py`
- `intergrax/applications/_shared/mcp_catalog_invocation_bridge.py`

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
