# T-IGX-043-A — EQO-A remediation wave, FC-1: execution routing and Nexus construction (A4-044, A4-009, A4-037; A4-024, A4-015, A4-031 conditional)

## Scope

Product-path ticket of the single integrated remediation wave (DP-11; master-plan node `PT-FC1`; authorization AR-41), drafted from the Campaign A dossier findings EQOA-A4-044, EQOA-A4-009, EQOA-A4-037, EQOA-A4-024, EQOA-A4-015 and EQOA-A4-031 (baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589; re-read at `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42`, all candidate files unchanged since the baseline). Design: DP-24 (hybrid fail-closed), DP-26 (first-routable fallback removed in STRICT/PRODUCT, typed LAB opt-in with a `fallback_used` event), DP-27 (`NexusLoop(production_mode=)` required keyword), with DP-18 / DP-13 / DP-14 coordination where files overlap seam work.

- Part A (firm): the host-task router uses the loop's own router so production-only routing checks apply on the canonical host-task path (A4-044); `AgentRouter` and `NexusLoop` take an explicit `production_mode`; the first-routable fallback needs a typed LAB opt-in and is refused in STRICT/PRODUCT (A4-009); the spec flag default is removed (A4-015 defaults).
- Part B (firm; Operator decisions D-FC1-01 and D-FC1-03, recorded as the Operator's acceptance of the recommended answers on 2026-11-16 in a session artefact, not a repository record): in `production_mode=True` the Tier-1 store openers (which stay LAB-only roots) do not read ambient env, and a missing task-memory, runtime-event or HITL decision store needs an injected store or an explicit typed acknowledgement: hosts with HITL inject the SQLite decision store from the existing persistence composition, hosts and workers without HITL pass `disable_human_decision_persistence=True`, which is refused when HITL tools are wired (A4-024, A4-037); the Tier-3 wiring no longer lets env re-enable task memory that the profile disabled.
- Part C (conditional, only if probe V2 / V6 is positive): `production_mode` of the Tier-2 agent contexts is derived from the host posture (A4-015 b); a STRICT `NexusLoop` requires a context engine and LLM adapter (A4-031).
- Sweep: 215 `NexusLoop(` calls without `production_mode=` and 7 `AgentRouter(` test calls get an explicit `production_mode=False` (behavior-preserving), plus ten notebooks and two usage docs.

Write set (exact lists in the plan, section 4): 14 firm product source files, 12 product-path documents/notebooks, 4 test files under `applications/`, up to 21 conditional product paths (2 forwarders, 4 agent files, up to 14 `build_harness_host_runtime` call sites, `tool_wiring.py`), five new test files, a `tests/` and `testing_support/` sweep, four owner-document sentences. No new product file. ACT-N11: every product path above breaks the IIAC activation (`IN_FORCE`, rebind at `969cbd964`) until the single RB-2 re-pin; this ticket does not perform it and must not merge outside the wave.

Decisions: D-FC1-03 (option D) is confirmed, the ceiling of up to 21 conditional product paths is accepted, the merge order is confirmed (FC-2, FC-1, FC-3, CONFIG-X scan extension; W-A/W-B before this ticket on `host_orchestration_environment_spec_builder.py`, `harness_host_runtime.py`, `nexus_loop.py`), and the product-path go is recorded as granted but effective only when `M1` holds, per the Operator's answers of 2026-11-16 transcribed by the orchestrating session in `.tmp/session/eqo-a-remediation/FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a session artefact, not yet a repository record). This ticket waits for the approval line (to be written through a commissioned non-implementing session; the commission is still to be given) and for `M1` (not decided).

Prerequisites before implementation starts: milestone `M1` (PDC P9 closed) and the Operator's separate authorization for product-path work (AR-39, AR-41); nodes `SCOPE`, `DS-2`, `DS-3`, `PT-B01` (probes V1, V2, V4, V5, V6). The Operator's answers to the six design questions are recorded in `OPEN_QUESTIONS.md` (D-FC1-01..06). State of PDC at the current head, recorded without a decision: `PDC_INDEX.md` (last changed `2bce73842`) still shows the P9 artifacts as awaiting the Operator's final confirmation, later commits `de025fca0`, `feaa2a87d`, `399d194e5`, `7960ae7f6` record closure steps, and `WP-12/WAVE_SCHEDULE.md` records `M1` as reached on 2026-11-16.

No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan. Reachability of A4-044, A4-037, A4-024, A4-015 b and A4-031 from a production host is not executed (static reading only; probes V1, V2, V4, V5, V6 pending).

## Plan digest

Plan digest: a40d3ee3e5df9e6fdb55d8dc14e98302a53c2a8de3b4df54fe663ab0a0a83cf5

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-043-A.md` (drafted at `.tmp/session/eqo-a-remediation/FC/FC-1/PLAN_T-IGX-043-A.md`; the plan is copied unchanged into `governance/tickets/plans/` when the ticket is committed). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line goes here after plan review, written by the Operator or by a commissioned, dedicated, non-implementing session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

Nothing was run when this ticket was drafted. One process at a time, one `uv run pytest` per verification step with multiple paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock`; the exact nodes and files are in section 6 and 7 of the plan:

- Baseline state of the regression slices R1 (routing), R2 (host task) and R3 (loop, HITL, stores) before any edit.
- Red-first: the five new test files (24 firm tests plus 2 conditional); all firm tests except the controls NT-FC1-03, 11, 18 are expected to fail on the unmodified code, in particular the refusals of STRICT/PRODUCT configurations (16: NT-FC1-01, 02, 05..08, 10, 12, 13, 15..17, 19..21, 26), and to pass after the change.
- Sweep proof R5, bounded: the 101 sweep-edited test files listed in `R5_SWEEP_FILES.txt`, exactly 5 commands (one per group, explicit file lists, no globs), one at a time under the lock, run once after the sweep and once at the end (AR-FC1-03b).
- Regression: R1, R2, R3, the gate slice R4 (`test_npsc4_2_residual_compatibility_gate.py`, `test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py`, `test_dup_x_npsc_boundary_gates.py`, `test_ebh_4_r1_r3_b6_typing_gate.py`, `test_obs_diag_x3_universal_spine_adoption.py`, `test_audit_ideal_depth_gate.py`, STATE-X, HARNESS-01 and CONFIG-X gate files) and the sweep proof R5 above.
- `uv run ruff check` and `uv run pyright` on the changed Python files; `uv run python scripts/docs/check_docs_domain_pairs.py`; `scan_call_sites.py` must report no `NexusLoop(` or `AgentRouter(` call without `production_mode=`.
- No broad or diagnostic suite, no Docker, no vendor integration (C10/C15 freeze). The independent audit is a separate session at the exact wave head (node `PT-Q02`).

## Out of scope

- Every other fail-closed selector and wiring (A4-017, A4-019, A4-026, A4-030, A4-032, A4-035, A4-036, A4-002), seam relocations of W-A / W-B (DP-13..DP-19), the closed-list entry naming the two Tier-1 store openers as LAB-only (seam documents), planner and assembly `production_mode=False` literals outside the named findings (registered as a Campaign B observation, D-FC1-05), the freeze-document statement for the HITL acknowledgement.
- The IIAC re-pin (RB-2), the EBH-7 pin, FRZ status promotion, stage closure, GitHub settings and rulesets, validator code, `scripts/`, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, PDC-X and IIAC record directories, any new file under `intergrax/`, `agents/` or `applications/`, Docker or vendor work.

## Lifecycle state

draft
