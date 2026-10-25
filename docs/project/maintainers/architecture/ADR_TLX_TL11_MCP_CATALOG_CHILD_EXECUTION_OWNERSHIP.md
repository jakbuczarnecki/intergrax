# ADR — TLX-TL11 MCP Catalog Tool Invocation Child Execution Ownership

| Field | Value |
| --- | --- |
| **Document ID** | `ADR-TLX-TL11-CHILD-OWNERSHIP` |
| **Task** | `TLX-TL11-CHILD-OWNERSHIP-CONTRACT-DESIGN` |
| **Operator authorization** | `OPERATOR APPROVED — TLX-TL11 CHILD OWNERSHIP CONTRACT DESIGN ONLY` |
| **Repository** | `jakbuczarnecki/intergrax` |
| **Branch** | `development` |
| **START_HEAD** | `acff4087123ad9bd1bce1e2a612f633afe222c5f` |
| **Status** | **DESIGN — READY FOR INDEPENDENT AUDIT** (not ratified; not implementation authorization) |

**Formal result:** `TLX-TL11-CHILD-OWNERSHIP-CONTRACT-DESIGN — READY FOR INDEPENDENT AUDIT`

**Explicit non-closure:** Not `CHILD OWNERSHIP RATIFIED`; not `P07/CANCEL-C CLOSED`; not `TOOL-LIFE-X CLOSED`; no global `FRZ-*` PASS; no production delta.

**Architecture design is not architecture ratification.**

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

1. **Cross-loop scheduling strategy** — queue to root `host_asyncio_task` vs require in-process-only catalog invoke until async job API exists.
2. **Child invocation durability** — in-memory dedupe only vs durable child registry (TL11 default: in-memory).
3. **Interim Option C** — deny catalog invoke until Option B lands vs partial invoke under fail-closed races.
4. **Physical thread policy** — retain `to_thread` under child runner vs operator-mandated async job pattern for long tools.

---

## 12. References

- `docs/project/maintainers/qualification/TOOL_LIFE_X_TLX_TL11_GATE_B_P07_CANCEL_C_QUALIFICATION_EVIDENCE.md` (reconciled §Child ownership)
- `docs/project/maintainers/qualification/PLATFORM_EXECUTION_UNIFICATION_U4_CHILD_EXECUTION_CLOSURE.md`
- `docs/project/maintainers/qualification/TOOL_LIFE_X_TLX_TL11_OWN_A2_SUPERVISED_EXECUTION_HANDOFF_QUALIFICATION_EVIDENCE.md`
- `intergrax/contracts/root_execution_transport_detachment_supervision.py`
- `intergrax/runtime/execution/child.py`
- `intergrax/applications/_shared/mcp_catalog_invocation_bridge.py`

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
