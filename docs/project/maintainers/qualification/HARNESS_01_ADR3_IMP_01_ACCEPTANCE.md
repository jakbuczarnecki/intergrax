# HARNESS-01-ADR3-IMP-01 — Catalog L2 Per-Call Identity Qualification Record

| Field | Value |
|-------|-------|
| **Status** | **QUALIFIED** (M2 implementation · closure evidence) |
| **ADR** | [`ADR-HARNESS-003`](../../technical/adr/entries/2026-09-22/ADR-HARNESS-003.md) |
| **ADR3_CONTENT_SHA** | `e9f66c923c8a259e5898756cb1d31ad78023adf4` |
| **ADR3_ACCEPTANCE_SHA** | `58e3056dede3f5c2292e7ca340bb35f79dd1abaa` |
| **ADR3_IMP01_IMPLEMENTATION_SHA** | `ff5daa088c6cc92e0288222654682b37b2132c9d` |
| **IMPLEMENTATION_PARENT** | `511580262cd60857db247ddf6e9b1de47eebbe01` |
| **Production code (M2)** | Changed only in implementation SHA (see scope table) |

## Verdict

```text
ADR3-IMP-01-Q1: PASS — exact-SHA M2 qualified on origin/development
ADR3-IMP-01 / M2: FORMALLY CLOSED (catalog L2 per-call immutable identity)
```

## Provenance

| Field | Value |
|-------|-------|
| **QUALIFIED_SHA** | `7aeae50132b201c3abf7098fcf611a4b73492af6` (pytest executed at this `development` HEAD; M2 files bitwise identical to `ADR3_IMP01_IMPLEMENTATION_SHA`) |
| **CURRENT_HEAD (pre-evidence)** | `7aeae50132b201c3abf7098fcf611a4b73492af6` |
| **REMOTE_VISIBLE** | YES (`origin/development` contains implementation SHA) |
| **REACHABLE_FROM_ORIGIN_DEVELOPMENT** | YES |
| **REMOTE_BRANCH** | `origin/development` |
| **Qualification session** | 2026-09-22 — local pytest (sequential); logs under `.tmp/session/ADR3-IMP-01-Q1/` |
| **Working tree at qualification** | **Dirty** (uncommitted GR12/catalog workstream files only; **not** staged for this evidence commit) |

## MIXED_SCOPE

```text
MIXED_SCOPE = NO
```

Implementation commit `ff5daa088…` touches only catalog L2 contract, Nexus L3 catalog adapter, CodeCraft wiring callsite, UCA-6C R5 test doubles, and ADR3-IMP-01 architecture gates.

### Implementation scope (`511580262…` → `ff5daa088…`)

| File | Change | M2 relevance |
|------|--------|--------------|
| `intergrax/contracts/execution_bound_catalog_tool_invocation.py` | Remove `bind_execution_identity` from L2 protocol | **Core M2** |
| `intergrax/runtime/nexus/tools/nexus_execution_bound_catalog_tool_invoker.py` | Per-call identity from request; remove bind/match helpers | **L3 projection** |
| `intergrax/runtime/codecraft/wiring_bound_capability_execution.py` | Drop pre-invoke bind | **Consumer** |
| `tests/unit/autonomous_work/test_uca6c_r5_*.py` | Remove bind from fakes/calls | **Regression** |
| `tests/unit/runtime/architecture/test_adr3_imp_01_catalog_l2_per_call_identity.py` | New EBCI-07/12 gates | **Qualification** |

```text
git diff --stat 511580262cd60857db247ddf6e9b1de47eebbe01..ff5daa088c6cc92e0288222654682b37b2132c9d
 6 files changed, 187 insertions(+), 107 deletions(-)
```

## Identity model (M2)

```text
authoritative source = ExecutionBoundCatalogToolInvokeRequest (frozen dataclass)
mutable execution binding = removed from ExecutionBoundCatalogToolInvoker / catalog L3 invoke path
per-call immutable = YES (invoke(request) only; sequential A→B isolation gated)
```

## L3 adapter

```text
RuntimeState projection = LEGAL INTERNAL PROJECTION via _runtime_state(request)
identity source = ExecutionBoundCatalogToolInvokeRequest fields only
shared mutable invocation identity = NO (binding not updated on invoke; gate asserts empty binding after invoke)
second lifecycle owner = NO
```

## EBCI status (M2 catalog path)

| Invariant | Result |
|-----------|--------|
| EBCI-01 | PASS — typed L2 port unchanged except bind removal |
| EBCI-02 | PASS — L2 contract Nexus-free (gate) |
| EBCI-03 | PASS — no second EE; adapter delegates to RuntimeToolInvoker |
| EBCI-06 | PASS — governance evidence validated via existing adapter helpers |
| EBCI-07 | PASS — per-call immutable scope (`test_adr3_imp_01_*` behavioral gates) |
| EBCI-08 | PASS — identity fields on request |
| EBCI-09 | PASS — fail-closed invalid run_id |
| EBCI-10 | PASS — no identity mint in adapter AST (gate) |
| EBCI-11 | PASS — plugin L2 without Nexus (protocol-only contract) |
| EBCI-12 | PASS — no shared mutable execution scope (sequential + empty binding) |

## Strong typing / reflection (implementation diff)

```text
new Any = 0 (no new public seam weakening in diff)
new object masking = 0
new generic semantic bags = 0
reflection bypass (getattr/setattr/hasattr as compat) = 0
dual-mode bind fallback = 0
```

## Nexus boundary metrics (static / dynamic / lazy / reexport)

| Area | Static | Dynamic | Lazy | Reexport |
|------|-------:|--------:|-----:|---------:|
| catalog L2 (`execution_bound_catalog_tool_invocation.py`) | 0 | 0 | 0 | 0 |
| CodeCraft L2 consumer (`contracts/codecraft`, `wiring_bound_capability_execution.py`) | 0 | 0 | 0 | 0 |
| Tools public ABI (`ToolInvocationInvokerPort` / `invocation_pattern`) | 0 | 0 | 0 | 0 |

`CatalogToolInvocationPort` remains deprecated alias of `ExecutionBoundCatalogToolInvoker` only.

## Closed-world inventory (Harness-01 gate)

```text
DISCOVERED = 171
CLASSIFIED = 171
UNCLASSIFIED = 0
STALE = 0
```

## Qualification tests (recorded)

```text
# ADR3-IMP-01 focused + ADR2/ADR3 docs + UCA/CodeCraft R5/R4
uv run pytest \
  tests/unit/runtime/architecture/test_adr3_imp_01_catalog_l2_per_call_identity.py \
  tests/unit/runtime/architecture/test_harness_01_adr2_documentation_regression_gates.py \
  tests/unit/runtime/architecture/test_harness_01_adr3_documentation_regression_gates.py \
  tests/unit/autonomous_work/test_uca6c_r5_canonical_tool_runtime.py \
  tests/unit/autonomous_work/test_uca6c_r5_r4_execution_bound_approval_evidence.py \
  tests/unit/autonomous_work/test_uca6c_r4_real_codecraft_execution.py \
  tests/unit/autonomous_work/test_codecraft_bound_capability_error_mapping.py \
  -q
# 38 passed (focused + docs + UCA/CodeCraft R5/R4)

# CodeCraft error mapping
# 8 passed

uv run pytest tests/qualification/harness_01/ -q
# 120 passed

uv run pytest tests/qualification/harness_02/ -q
# 18 passed

uv run pytest \
  tests/qualification/harness_01/test_harness_01_w2_r6_idempotency_wiring_boundary.py \
  tests/qualification/harness_01/test_harness_01_w2_r6_r1_runtime_tool_invoker_recomposition.py \
  tests/qualification/harness_01/test_harness_01_w2_r6_r2_runtime_tool_invoker_reconfiguration_lifecycle.py \
  tests/qualification/harness_01/test_harness_01_w2_r6_r3_runtime_tool_invoker_transactional_ownership.py \
  tests/qualification/harness_01/test_harness_01_agent_layer_zero_nexus_imports.py \
  tests/qualification/harness_01/test_harness_01_agent_layer_dynamic_nexus_imports.py \
  tests/qualification/harness_01/test_harness_01_w2_r4_agent_lazy_nexus_exports.py \
  -q
# 30 passed

uv run pytest tests/unit/runtime/architecture/test_hardening_6_execution_authority_gate.py -q
# 4 passed

uv run pytest \
  tests/integration/runtime/test_nexus_loop_hitl_hooks.py \
  tests/integration/runtime/test_harness_hitl_pagerduty.py \
  tests/integration/runtime/test_declarative_policy_hitl_nexus_e2e.py \
  -q
# 6 passed
```

## Remaining ADR3 migration

```text
M1 = CLOSED
M2 = CLOSED (this record)
M3 = PENDING (declarative/compensation bind still on L2)
M4 = PARTIAL / PENDING
M5 = PENDING
M6 = PENDING
M7 = PARTIAL / PENDING
```

## Production changes in Q1

```text
NONE (qualification + evidence only)
```

## Next

```text
ADR3-IMP-02 — Declarative L2 Per-Call Identity Reconciliation (not started in this session)
```

Independent audit of this record, `ADR3_IMP01_IMPLEMENTATION_SHA`, implementation diff, per-call immutable catalog identity, absence of stateful `bind_execution_identity` on catalog L2, L3 RuntimeState projection, EBCI-07/EBCI-12, CodeCraft/catalog Nexus boundaries, closed-world inventory, full H01/H02/W2/EE/HITL, and closure evidence on GitHub is required before ADR3-IMP-02 and before resuming HARNESS-01-R5-W3-R1-Q2.
