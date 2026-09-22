# HARNESS-01-ADR3 — Architecture Decision Acceptance Record

| Field | Value |
|-------|-------|
| **Status** | **ACCEPTED** (architecture freeze · qualification / publication only) |
| **ADR** | [`ADR-HARNESS-003`](../../technical/adr/entries/2026-09-22/ADR-HARNESS-003.md) |
| **ADR3_CONTENT_SHA** | `e9f66c923c8a259e5898756cb1d31ad78023adf4` |
| **ADR3_CONTENT_PARENT** | `f6a538f4c2291d83d5c5423ac50bf3e9da45a033` |
| **ADR3_AUDIT_SHA** | Git commit that adds this file (`docs(architecture): reconcile ADR3 publication evidence`) — remote-visible on `origin/development`; pins `ADR3_CONTENT_SHA` and mixed-scope lineage |
| **Historical mixed-scope commit (orphaned)** | `3c70485f02b5efcf887b19f431949d30bb2ec5fe` — **not** on `development` lineage; **not** on GitHub |
| **Production code** | **Unchanged** (ADR3 delivery is docs + doc regression tests only) |

## Verdict

```text
HARNESS-01-R5-ADR3-Q1: PASS — ADR GITHUB-AUDITABLE (ADR3_CONTENT_SHA + publication evidence commit on origin/development)
Resolved by HARNESS-01-R5-ADR3 / ADR-HARNESS-003.
```

## Mixed-scope lineage reconciliation (no history rewrite)

| Commit | On `development` / GitHub | ADR3 content | Unrelated content | Scope-clean |
|--------|---------------------------|--------------|-------------------|------------:|
| `3c70485f02b5efcf887b19f431949d30bb2ec5fe` | **No** (local orphan only) | ADR-HARNESS-003, ADR index, EE hub pointer, ADR3 doc test | `UCA_6C_CANONICAL_HITL_BOUNDARY_RECONCILIATION.md` | **No** |
| `f6a538f4c2291d83d5c5423ac50bf3e9da45a033` | **Yes** | — | UCA HITL reconciliation doc only | **Yes** (UCA) |
| `e9f66c923c8a259e5898756cb1d31ad78023adf4` | **Yes** | Full ADR3 documentation package (see below) | — | **Yes** (ADR3) |

**MIXED_SCOPE (historical):** `YES` for `3c70485f…` only. Canonical published lineage splits UCA and ADR3 into separate commits. Do not cherry-pick or rewrite `3c70485f…`.

### ADR3 files (canonical package — `e9f66c923…`)

- `docs/project/technical/adr/entries/2026-09-22/ADR-HARNESS-003.md`
- `docs/project/technical/adr/README.md` (index entry)
- `docs/project/maintainers/architecture/EXECUTION_ENGINE.md` (hub pointer)
- `tests/unit/runtime/architecture/test_harness_01_adr3_documentation_regression_gates.py`

### Unrelated files (historical mixed commit `3c70485f…` only)

- `docs/project/maintainers/architecture/UCA_6C_CANONICAL_HITL_BOUNDARY_RECONCILIATION.md` (delivered separately at `f6a538f4c…`)

## Frozen summary

- **L1:** `intergrax.tools.invocation_pattern` / `ToolInvocationInvokerPort` — sole public Tools invocation ABI (ADR-HARNESS-001 D5 retained)
- **L2:** `ExecutionBoundCatalogToolInvoker`, `ExecutionBoundDeclarativeToolInvoker`, `CompensationSideEffectExecutionPort` — domain/platform contracts; Nexus import forbidden for external implementers
- **L3:** `RuntimeToolInvoker`, `CatalogDeclarativeToolInvoker`, `NexusExecutionBoundCatalogToolInvoker` — EE/Nexus internal only
- **EBCI-01 … EBCI-12** frozen; **PER-CALL IMMUTABLE** identity model; **`bind_execution_identity` → DEPRECATE / remove from final L2**
- **RuntimeState:** **LEGAL INTERNAL PROJECTION** in Nexus/EE only
- **Migration M1–M7** and **Gate 1–8** documented (implementation: ADR3-IMP-01)
- **Rejected:** `UniversalExecutionBoundInvoker`, universal tool invokers, generic capability executors

## Qualification tests (documentation regression)

```text
uv run pytest tests/unit/runtime/architecture/test_harness_01_adr3_documentation_regression_gates.py tests/unit/runtime/architecture/test_harness_01_adr2_documentation_regression_gates.py -q
```

Recorded 2026-09-22 session: **8 passed** (ADR3 + ADR2 documentation regression gates).

## Next

```text
ADR3-IMP-01 — Execution-Bound Invocation Contract Reconciliation
```

Independent audit of this record, `ADR3_CONTENT_SHA`, `ADR3_AUDIT_SHA`, and ADR-HARNESS-003 against GitHub is required before ADR3-IMP-01 and before resuming HARNESS-01-R5-W3-R1-Q2.
