# TOOL-LIFE-X R4 — final root-cause remediation and parent qualification evidence

**Status:** IMPLEMENTED · TESTED (scoped L2) · PHYSICALLY QUALIFIED (HTTP Class B qual path) · **not** independently accepted (F8 EXT-L4)

| Field | Value |
|---|---|
| Baseline SHA (operator START_HEAD) | `d782800e622a9b8c27351bd3818a2e8abf5c5f0e` |
| Frozen implementation SHA | *(set at commit — see git log after push)* |
| Branch | `development` |
| Roadmap authority | `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` §3.0.5 (EC1–EC15 canonical) |
| ADR | `ADR_ENTERPRISE_TOOL_LIFE_X_CERTIFIED_OWNER_LOSS_AND_STALLED_POLICY.md` |

## 1. Plan traceability (F0–F8)

| Phase | Disposition |
|---|---|
| F0–F4 | Prior accepted bundles reused; this wave closes systemic RCA defects A–E |
| F5/F6/F7 | Implementation + consolidated L2 qualification complete on this SHA |
| F8 EXT-L4 | **READY FOR INDEPENDENT ACCEPTANCE** — operator/auditor closure not performed by Cursor |

## 2. Root-cause summary

Generic catalog `http.request` was bound to `tool_life.http_idempotent_post.qual.v1` (Class B) despite arbitrary methods/hosts and `SideEffectRetrySafety.NOT_RETRY_SAFE` on the ToolContract. Durable redrive identity lacked invocation fingerprint; invoker no-dependency success path skipped `mark_succeeded()`; HTTP transport loss after submit could record physical FAILED instead of UNKNOWN.

## 3. Defects A–E

| ID | Finding | Disposition |
|---|---|---|
| A | `http.request` → automatic Class B | **Fixed** — generic HTTP has no sanctioned redrive binding (Class C default) |
| B | Class B requires enforceable provider contract | **Fixed** — `http.idempotent_post.qual` + `http_rest`/`http_idempotent_post` + physical qual server |
| C | Contradictory retry surfaces | **Reconciled** — explicit `NOT_RETRY_SAFE` on generic HTTP; Class B only on qual contract/tool |
| D | Durable effect identity | **Extended** — `bound_physical_effect_fingerprint` from `compute_invocation_operation_identity` |
| E1 | No-dep boundary success terminal | **Fixed** — `mark_succeeded()` in `else` before return |
| E2 | HTTP lost-ack durable FAILED | **Fixed** — `http_transport_error` + `may_have_started` → `mark_unknown()` |
| E3 | Timeout race on completed future | **Fixed** — validate `future.result()` before `mark_succeeded()` |
| E4 | Single terminal writer | **Preserved** — existing store helpers only |

## 4. Authoritative effect classification (final)

| Surface | Class | Notes |
|---|---|---|
| Generic `http.request` | **C** | Executes under governance; no automated redrive binding |
| `http.idempotent_post.qual` | **B** | Sanctioned qual catalog tool + provider idempotency key enforcement |
| `http_blind_post` / `http.read_only` qual | **C** | Explicit contracts |
| `invoke_catalog_tool` (MCP) | **C** | Default qual contract |
| Platform durable physical lifecycle | **B** | Explicit catalog tool only |
| DB SQL adapter (platform) | **A/B per prior F5** | Unchanged; not equated with HTTP Class B |

## 5. Production execution graph (qual positive path)

`RuntimeToolInvoker` → durable external-operation record → `before_physical_submit` (sanctioned bind + fingerprint) → `AllowlistHttpClient` via `http.idempotent_post.qual` → local idempotent POST server → lost-ack → UNKNOWN → certified owner loss → reconciliation → CAS redrive authority → identical idempotency key → single effect.

## 6. Physical qualification commands (L2 consolidated)

```powershell
uv run pytest `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_systemic_rca_authoritative_classification.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_b1_authoritative_effect_class.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_b2_http_class_b_physical_qual.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_b2_mcp_class_c_fail_closed.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_ext_l4_binding_adversarial.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_ext_l4_integrated_http_redrive_physical.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_f6_scenario_24_recovery.py `
  tests/unit/applications/test_tool_life_x_r4_production_entry_e2e_qualification.py `
  tests/unit/runtime/nexus/tools/test_tool_life_x_r4_terminal_lifecycle_invoker.py `
  -q
```

**Result:** 42 passed (session log: `.tmp/session/TOOL-LIFE-X-R4/pytest-consolidated-l2.log`).

**L3 reuse:** SQLite cross-process CAS/revocation — prior `test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store` helpers reused in integrated redrive proof. PostgreSQL Docker L3 not re-run on this change-impact scope (no remote SQL fencing change).

## 7. EC1–EC15 (canonical §3.0.5 matrix)

| EC | Status on this SHA | Evidence |
|---|---|---|
| EC1–EC4 | TESTED (prior + invoker) | Batch B/A suites + invoker deadline tests |
| EC5 | TESTED | UNKNOWN / uncertain ledger + ext-op UNKNOWN |
| EC6 | TESTED | Retry safety + Class B/C separation |
| EC7–EC9 | TESTED | Scenario 24 + R8/R9 patterns |
| EC10 | IMPLEMENTED | Prior DB limits qual |
| EC11 | TESTED | MCP Class C fail-closed R4 test |
| EC12 | **PHYSICALLY QUALIFIED** (scoped) | Integrated HTTP lost-ack redrive |
| EC13 | TESTED (scoped) | R4 physical + reused L3 helpers |
| EC14 | TESTED | No bypass in changed owners |
| EC15 | **No new lifecycle blockers** in R4 scope | Residuals: independent F8 only |

## 8. G-A01–G-A09 (summary)

| Gate | Result |
|---|---|
| G-A02 effect class | Authoritative trusted contracts only; generic HTTP Class C |
| G-A07 default-off recovery | Unchanged; qual wiring isolated |
| Others | Reuse prior F3/R6–R10 evidence unless path touched — binding/classification touched; HTTP/MCP Class C tests green |

## 9. FRZ / ECAP

No global `FRZ-*` promotion. Scoped qualification only. Production activation not performed.

## 10. Operator decisions

None required for safe Class C default on generic HTTP. Independent EXT-L4 acceptance remains operator-owned.

## 11. Conclusion

**Zero implementation blockers** remain for F8 handoff within authorized scope. **Parent CLOSED not asserted.**
