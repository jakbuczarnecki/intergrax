# T-IGX-043-G — EQO-A remediation wave, W-A: boundary inversion of 12 `intergrax/runtime` modules and Tier-2 agent seams (A4-003, A4-004, A3-016, A7-021)

## Scope

Product-path ticket of the single integrated remediation wave (DP-11; master-plan node `PT-WA`; authorization `AR-39`), drafted from the Campaign A dossier findings EQOA-A4-003, EQOA-A4-004, EQOA-A3-016 and EQOA-A7-021 (baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589; re-read at `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42`, every candidate file unchanged since the baseline), with EQOA-A2-009 and EQOA-A5-001 as a separable follow-on part. Design: DP-13 (the 12 `intergrax/runtime` modules that import `intergrax.applications` are fixed by inversion: neutral Tier-0 contracts and ports, Tier-3 builds, Tier-1 applies; no allowlist for them), DP-14 (Tier-2 agents import only `integrations.contracts` leaves), DP-15 (the qualifier agents are qualification tooling: the resolver moves to `tests/system`, the agent requires an injected provider), DP-16, DP-21, DP-22.

- Part A (firm): 11 `intergrax/runtime/execution` modules lose every `intergrax.applications` import. File paths and function names stay (ten of the modules around them are path-pinned by DUP-X and ROADMAP-REPLAY-X); two unpinned files are deleted (`host_orchestration_wiring_bundle.py`, `scenario_host_diagnostic_wiring.py`). Four new neutral contracts under `intergrax/contracts/` (graph topology base model, delegation budget policy, the wiring port, the resolved-environment dataclasses) and two new Tier-3 modules build what Tier-1 applies.
- Part B (firm): `intergrax/runtime/architecture/obs_diag_x3_ast_gates.py` (the twelfth module) moves to `testing_support/architecture/`.
- Part C (firm): the Tier-2 agent seams. S2d: the environment-credential provider resolver leaves `agents/` for `tests/system`; the web-search qualifier requires an injected provider; the LKW builder resolves it from the `IntegrationProfile` and refuses when absent. S2b and S2c re-point two agent imports to two new neutral contracts (`rag_retrieval_source_scope.py`, `workspace_tool_ids.py`); the five `external_contractor_adapter` files and the contract-leaf `SearchProvider` import need no code change under DP-14.
- Part D (firm, read-only): the AST census of data-driven imports for A7-021 and its re-scope table; no product edit.
- Part E (firm, a separable last commit group; question Q-WA-05 is recorded as option A in `OPERATOR_PACKET_3.md`): two LKW production modules routed through registry facades (S8b), with the negative test `NT-RW-WA-01` (A2-009).
- Part K (firm): KR-18, the contract root `intergrax/applications/contracts/environment_profile/root.py` no longer imports `intergrax.applications._shared` and takes the tool ids from the new `intergrax/tools/contracts/tool_availability.py` (re-exported by `intergrax/skills/registry/tool_requirements.py`); the contract-to-implementation edge is moved by a lazy import, not removed, and is registered as a residual; tests `NT-WA-KR18-1` (full register rule) to `NT-WA-KR18-4`.
- Amendment 1 part A (firm): `applications/local_workspace_application/manifest.py` mounts the web-search qualifier only when the LKW `IntegrationProfile` has a `search_provider` slot, so the default roster still builds; one derivation function (`local_workspace_profile_has_search_provider` in `host/agent_factories.py`) is used by the manifest and by the builder, the roster is derived from the profile instance of the manifest's own environment, and an invariant test (`NT-WA-25`) fails if enablement and acceptance disagree; tests `NT-WA-23`, `NT-WA-24`, `NT-WA-25`; `host/environment_profile.py` is a conditional path. The Q3 qualification route (an LKW input that sets the slot) is not part of this write set.
- Decision records DR-1..DR-4 in plan section 7c: Q-WA-09 option P2 supersedes the PACKET_0 default and requires orchestrator acceptance of P2 versus P1 because the lazy import in `tool_availability.py` is invisible to the EBH-2A gate; Appendix A1 needs the five paths of Amendment 1.
- A4-007 (three dynamic-import seams, confirmed by Campaign B-1 alpha) is owned by `T-IGX-043-H`; the importlib allowlist gate is `PT-G01`. X-A4-004 (ten `agents/` files fail the registry-bypass check at the baseline) is the same census as A4-004 and is closed by the Part C tests, with NT-WA-17 added.

Write set (exact lists in the plan, section 9; it equals `RW_PLAN.md` Appendix A1 block `PT-WA` with the differences listed in `.tmp/session/arch-freeze/w3-packet-0/INPUT_043-G.md` and `INPUT_043-G_AMENDMENT_1.md`): 45 firm product paths (6 new contracts and `tool_availability.py`, 2 new Tier-3 modules, 30 edited with `root.py`, `manifest.py` and `tool_requirements.py`, 2 edited for Part E, 4 deleted) and 1 conditional product path, ten new and nine edited test or support paths (two of the new ones are relocated support modules), three conditional test paths, eight new test files (25 tests), no docs. No edit of the gate scripts, `.github/`, the F10 inputs, the roadmap, the FRZ checklist, IIAC or PDC-X records. Product commits go only to branch `arch-freeze/rw-train`. The write-set baseline is the 19-module tight set measured by the ARCH-FREEZE session (12 seeds plus 7 Tier-1 callers, `.tmp/session/arch-freeze/closed-world/RESULT.md` section 1, a session artefact; only partly re-verified, plan section 1 and FACT_CHECK F-110..F-113); of the 7 callers only `host_orchestration_loop_init_spec.py` is edited. The 19 directly affected test files and the 417 test files that import the 24 Tier-3 targets are the regression and rebase scope. ACT-N11: every product path above breaks the IIAC activation until the single RB-2 re-pin; this ticket does not perform it and must not merge outside the wave.

Dependencies and overlaps stated in the plan: `PT-G01` (registry-bypass gate rewrite and the allowlist gates; the old regex gate stays red on six contract-leaf agent files until it lands), W-B (merge order and the two builder files, question Q-WA-06), FC-1 (spec builder, `harness_host_runtime.py`, loop init spec; FC-1 rebases onto this ticket), FC-3 (shared LKW `host/factory.py` in Part E), `PT-Q04` (F10 C4 rules for the six `agents/` paths, which have no rule today), `PT-G01` (the demo/proof exception list read by `NT-RW-WA-01`, question Q-WA-10).

Prerequisites before implementation starts (not assumed): milestone `M1` (PDC P9 closed), node `SCOPE`, the product-path authorization `AR-39`, the answers to the open questions in `OPEN_QUESTIONS.md` (Q-WA-01..Q-WA-04, Q-WA-06, Q-WA-07, Q-WA-09, Q-WA-10; the draft is written on the recommended option of each; Q-WA-05 is option A). `M1` is recorded in `WP-12/WAVE_SCHEDULE.md` and `PDC_INDEX.md`; this ticket takes no position on either.

No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan. A4-003, A4-004, A3-016 and A7-021 are not closed by the draft; closure conditions are in plan section 15.

## Plan digest

Plan digest: 5b7d2827185ae3f06562d5e8d76a4cdec067f26b5283929c0e1ebcc4333ac8d1

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-043-G.md` (drafted at `.tmp/session/eqo-a-remediation/FC/WA/PLAN_T-IGX-043-G.md`; the plan is copied unchanged into `governance/tickets/plans/` when the ticket is committed). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T09:59:28+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_DELEGATION_RECORD_2.md@c482c4345da1873ec58505c5475150afaeafbb75
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

Nothing was run when this ticket was drafted. One process at a time, one `uv run pytest` per verification step with multiple paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock`; the exact nodes and files are in plan sections 10 and 11:

- Red-first (S0): the eight new test files (25 tests: 4 refusal tests of configurations that must now be refused, 12 structural, 1 preserved refusal over the new API, 8 controls) on a detached worktree at the pre-change commit; the log is kept for the auditor.
- After the change (S4): all 25 green (`NT-RW-WA-01` needs the exemption list of question Q-WA-10). The three refusal groups that change behaviour are the S2d ones: an agent constructed without a provider, ambient credentials in the environment that must not activate a provider, and an LKW STRICT profile without a `search_provider` slot.
- Regression (S1..S3): boundary, inventory and ownership gates (including the EBH-2A contract gate that classifies the six new contract files, the EBH-4 typing gate, the npsc4.2 and DUP-X gates, the harness_01 and QUAL-X inventories), host and scenario composition tests, agent and qualifier tests; about 130 test functions in 17 files whose names were read at the current head, plus whole-file runs of the other listed files.
- S1 adds the KR-18 register node and the contracts `Any` ratchet measurement; S3d lists five exact LKW host test files as regression scope without a pass claim (they run red at base and head in the development environment: no LLM provider, `ollama` absent).
- S2b and S7: the 12 further directly affected test files of the ARCH-FREEZE list, and one collection-only check over the 417 test files that import or name the 24 Tier-3 targets (authorization WA-A12).
- S5: `check_intergrax_no_applications_imports.py` (expected exit 0, the closure condition of A4-003 at an exact SHA, not a claim) and `check_agent_registry_bypass.py` (informational; six contract-leaf files remain until `PT-G01`). S6: `uv run ruff check` and `uv run pyright` on the changed Python files.
- No broad or diagnostic suite, no Docker, no vendor integration, no live Q3 diagnostics (C10/C15 freeze). The independent audit is a separate session at the exact wave head (node `PT-Q02`), with the mutation checks of plan section 16.

## Out of scope

- The 15 product files outside the gate's roots that import `intergrax.applications` (AST count; question Q-WA-01; a dependency on `PT-G01`'s upward-import gate), the two undisposed dynamic nodes of the B1 closed-world run (`mcp_import_guard.py:60`, `mcp_server_engine.py:159`), the other dynamic-import census items that B-1 alpha did not probe (`wiring.py:64`, `context_engine_resolver.py:29`, `intergrax/cli/doctor_*`), the CI wiring of the registry-bypass check that did not run at the baseline (`.github/`, `PT-G01`/`PT-G02`), other fail-closed selectors and wiring (FC-1..FC-4), the one-resolver, profile-projection and composition-root work of W-B (DP-16..DP-19), the rewrite of `check_agent_registry_bypass.py`, the seam ledger, the importlib allowlist gate and the EBH-3 scan extension (`PT-G01`), the canonical layer text and the CTRL-X relabel (`PT-D12`, `PT-D10`), the F10 C4 rules for `agents/` paths (`PT-Q04`), the duplicate orchestration enums in both tiers (Q-WA-07), the S2b authority question (Q-WA-03).
- The IIAC re-pin (RB-2), the EBH-7 pin, FRZ status promotion, stage closure, GitHub settings and rulesets, validator code, `scripts/`, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, PDC-X and IIAC record directories, `intergrax/runtime/nexus/nexus_loop.py`, Docker or vendor work.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_e87d11d4-fd56-4734-a8fd-46691cb7b3f7

(the PT-WA implementer session; recorded here, outside the plan text).
