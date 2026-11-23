# PLAN T-IGX-043-N — FC-1 residual: NT-FC1-26 case 2 (ToolWiringContext carrier)

Status: **draft plan (MEP W3 scaffold hardened, Runda 1 autonomy 2026-11-21)**. Parent **T-IGX-043-A** (PT-FC1) owns FC-1 Parts A–C; this plan covers **only** the follow-on **F-4** row: second sub-case of **NT-FC1-26** (human decision store propagated into `ToolWiringContext` on the harness path). No product edits before `plan_approved`. No PASS, FRZ promotion, or independent-verification claim.

## 1. Purpose

After FC-1 Part B lands on `development`, STRICT harness hosts that inject `human_decision_store` into `build_harness_host_runtime` must expose the **same** store instance on the resolved `ToolWiringContext` used for tool wiring — not a second authority and not `None` when HITL tools are wired (consistency rule in **PLAN_T-IGX-043-A** section B4 / B7).

## 2. Prerequisites

| Gate | Requirement |
|------|-------------|
| FC-1 authority | **T-IGX-043-A** reviewed/merged on `development` (PR **#680** lineage); do not re-implement FC-1 Part B loop/spec changes here |
| Operator (Runda 1) | **D3** (043-M Part E), **D4** (WF E2–E4), **D5** (051-A TENANT) remain **BLOCKED** — out of scope for this ticket |
| Serialization | No concurrent edit of `harness_host_runtime.py` with **T-IGX-051-A**; coordinate with **T-IGX-043-J** per [`WP12_W3_PT_TICKETS_PREP.md`](../../../docs/project/maintainers/qualification/arch_freeze_program/continuation/WP12_W3_PT_TICKETS_PREP.md) |

## 3. Write set (firm)

| Id | Path | Intent |
|----|------|--------|
| N-1 | `intergrax/applications/_shared/tool_wiring.py` | Accept optional `human_decision_store` on `build_application_tool_wiring` (or equivalent host entry) and set `ToolWiringContext.human_decision_store` from the harness-injected store (single authority) |
| N-2a | `intergrax/applications/_shared/environment_wiring.py` | Add `human_decision_store` parameter to `wire_application_environment`; pass through to `build_application_tool_wiring` before registry snapshot is fixed (no post-hoc rebuild) |
| N-2b | `intergrax/applications/_shared/harness_host_runtime.py` | Forward injected `human_decision_store` into `wire_application_environment(...)` only (one line; FC-1 Part B guards unchanged) |

**N-2 caller ceiling (frozen):** only N-2b on harness path. Non-harness Tier-3 hosts do not inject a store in this ticket.

| File | Owner | Collision risk | Test acceptance |
|------|-------|----------------|-----------------|
| `tool_wiring.py` | PT-FC1 / 043-N | Low vs 043-J workflow files (disjoint) | NT-FC1-26 case 2 |
| `environment_wiring.py` | PT-FC1 / 043-N | Serialize before **051-A** if same module edited | NT-FC1-26 case 2 |
| `harness_host_runtime.py` | PT-FC1 / 043-N | **043-J** may touch WF; not `harness_host_runtime` per ticket | NT-FC1-26 case 2 |
| `test_fc1_harness_strict_persistence_wiring.py` | 043-N | None | `ctx_store is store` |

**Out of write set:** `intergrax/runtime/nexus/nexus_loop.py`, FC-1 Part B composition logic in `harness_host_runtime.py` (043-A), TENANT / WF product slices.

## 4. Test contract

| Node | Pytest target (from 043-A) | Case |
|------|----------------------------|------|
| NT-FC1-26 case 2 | `::test_injected_human_decision_store_reaches_tool_wiring_context` | `runtime.env_wiring.tool_wiring.wiring_context.human_decision_store is` injected store |
| NT-FC1-26 case 1 | `::test_acknowledgement_is_refused_when_hitl_tools_are_wired` | **043-A** regression — must stay green |

Red-first at baseline SHA before product edit; then FC-1 regression slice R3 subset (harness persistence file + loop helpers per 043-A §6). `uv run python -m scripts.governance.igx_gov ticket` before push when product paths change.

## 5. Out of scope

Full FC-1 rewrite, **T-IGX-043-M** Part E manifest, **043-J** E2–E4, **051-A** TENANT, train merge, RW milestone, FCS, `ARCH_FREEZE_CLOSED`.

## 6. Next steps (operator / architect)

1. Re-read harness call graph @ implementation SHA; freeze caller list for N-2.
2. Record plan digest in `T-IGX-043-N.md`; lifecycle `plan_pending`.
3. Wave-packet `Approved:` after FC-1 review closes; then `plan_approved` and worktree batch (e.g. `af-dev-043n`).