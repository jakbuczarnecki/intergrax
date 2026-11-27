# T-IGX-043-C — EQO-A remediation FC-3: vendor/provider selector fail-closed

## Scope

Make four silent vendor/backend selectors fail closed in PRODUCT and STRICT. Findings of the Campaign A dossier (PR #589, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`): EQOA-A4-017 (vision stub fabricates a 0.99 `object.stub` detection), EQOA-A4-019 (embedding provider defaults to `ollama`, vector store to in-memory, notification backend to `log`), EQOA-A4-026 (inbound interaction verification defaults to a no-op verifier; unset or invalid mode resolves to `none`), EQOA-A4-032 part c (guardrail factory pass-through stub and silent degradation to the pattern engine). Master-plan node `PT-FC3`; decisions DP-24 (hybrid: wiring layers fail closed on STRICT or PRODUCT, leaf selectors default-deny with an explicit typed LAB opt-in), DP-25, DP-28 (vision stub only through an explicit LAB profile with a provenance flag), DP-18, DP-14, DP-11; conditions C3, C5, C6.

The CONFIG-X scan extension (former Part G) is **not** part of this ticket. It is the separate draft ticket `T-IGX-043-F`, a gate-meaning change that needs the Operator's personal ruling and is not approvable by a delegate. This ticket does not depend on it for its own correctness (the existing CONFIG-X gates stay unchanged and must stay green); `T-IGX-043-F` merges after this ticket or in the same wave behind it.

Write set, separated (details in the plan, section 4):

- Product paths, ACT-N11 sensitive (any change after the IIAC-bound product tree breaks the activation until the single RB-2 re-pin): 32 files in `intergrax/` named by the design plus 30 callers (seven Tier-3 host factories, `local_workspace` host runtime composition, bootstraps, LAB surfaces, two LAB scripts, two scaffold files, two long-running modules, five files in `applications/_shared`) = 62 files; the Operator's accepted bound was "about 59"; the three above it were accepted (O-09 = option 1, transcription of 2026-11-16). The posture helper `intergrax/applications/_shared/wiring_posture.py` is imported, not created: it belongs to `T-IGX-043-B`, which merges first.
- Tests, not ACT-N11 paths but outside the EBH-7 identity allowlist (they merge before the final pin): six new test files with 51 negative tests, and the inversion of six tests that pin today's silent behaviour (plus default-dependent tests found by a bounded census; `applications/*/tests` files are ACT-N11 sensitive by prefix).
- Documents: none in this ticket (plan section 4 D).

Notifications (restored Q-FC3-02 option 1): no silent `log` default for HITL escalation and long-running notifications in PRODUCT/STRICT. The explicit adapter reaches `NexusLoop` and the scheduler through existing keywords (`notification_adapter` of `build_harness_host_runtime` and of `wire_harness_host_long_running_scheduler`), so no file of FC-1 or W-A/W-B is edited for it; the coordinator refuses an escalation with no injected adapter and no explicit channel. The impact class of the leg stays BLOCKS. Paths without a composition-time refusal are named in the plan (section 3.3); the Operator accepted the plan as drafted for them (O-08 = option 1: late refusal by the coordinator guard, class BLOCKS kept, the unrefused paths named in the audit).

The Operator's answers of 2026-11-16 (O-08, O-09, L-1, merge order B, A, C, F) are recorded in `OPEN_QUESTIONS.md` from the transcription `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md`, a session artefact and not yet a repository record. Status of the approval: the plan was approved by the commissioned signing delegate (see the approval line in the `## Approved` section); that approves the plan text only and is NOT an implementation go. Implementation starts only when the ARCH-FREEZE gate `SCOPE` and the Operator's product-path go are given, milestone M1 holds (whether PDC P9 is formally closed is not decided), the prerequisites of the plan (section 2) are met and the authorizations A-01..A-10 are given. The Operator's acceptance of the design questions Q-FC3-01..11 (recommended options) was relayed by the orchestrating session on 2026-11-16; the decision record is a session artefact under `.tmp`, not a repository record (`OPEN_QUESTIONS.md` states exactly what was accepted). No closure, PASS, FRZ promotion or independent-verification claim is made by this ticket; an independent audit at the exact wave SHA is part of the test contract.

## Plan digest

Plan digest: 475a6642fe356f678c3619abd5551f4fcc55bc80f55a3876406978514b2e0da4

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-C.md`](plans/PLAN_T-IGX-043-C.md). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T17:17:28+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@743087d57c674100dc9119959dc875271f70e72c

## Test contract

One process at a time (under the pytest lock if it applies to the code wave); none was run when this ticket was drafted. Commands (details and node lists in the plan, sections 5 to 7):

- Red-first at the baseline, then green at the fix: `uv run pytest tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py tests/unit/runtime/notifications/test_fc3_notification_selector_fail_closed.py tests/unit/model_inference/test_fc3_vision_fail_closed.py tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py tests/unit/integrations/providers/llm_guardrail/test_fc3_guardrail_fail_closed.py -p no:cacheprovider` (51 negative tests: a PRODUCT or STRICT configuration that must now be refused).
- Existing CONFIG-X directory, gates unchanged: `uv run pytest tests/qualification/config_x -p no:cacheprovider`.
- Regression: one command over the 78 existing node IDs listed in `regression_nodes.txt` of the plan package (each verified to exist at the baseline and at the current head), after the six defect-pinning tests are inverted.
- `uv run python -m scripts.governance.igx_gov ticket` before push.
- Independent audit (G6) at the exact wave SHA by a separate non-implementing session (negative-test mutation check, fallback census over the diff, ACT-N11 path reconciliation); the implementer records no PASS.
- Rollback: revert the wave merge; the IIAC rebind must be redone (plan section 13).

The census of default-dependent tests (a bounded run over the 52 files of `census_files.txt`) is a pre-implementation diagnostic with its own authorization (A-05); it is not part of this test contract.

## Out of scope

Everything outside the write set above: the CONFIG-X scan extension (`T-IGX-043-F`); `scripts/`, `.github/`, `docs/`, `governance/` other than this ticket and its plan, the roadmap, the FRZ checklist, qualification and IIAC records, `pyproject.toml`, `uv.lock`; the files of FC-1 (Nexus routing, stores, `harness_host_runtime.py`; FC-3 only passes an existing keyword to it from the host factories), FC-2 (capacity, queue, memory, OTLP, sandbox: A4-032 parts a and b; creator of `wiring_posture.py`), FC-4 (skill pinning, KV durability), W-A/W-B (boundary inversion, resolver, composition roots); FRZ promotion; Docker, new vendor integrations and production configuration (C10/C15 freeze); live vendor tests; fabricated parse defaults in vendor vision adapters, the explicit in-memory vector store of `legal_product()` and the Hugging Face host literal (recorded as observations in the plan, section 16).

## Lifecycle state (superseded by settlement below)

plan_approved

## RW traceability (GA.2R)

Traceability only (RW-PLAN-AUDIT finding M7, ticket T-IGX-078-B). This section grants no write authority and changes no scope, plan, plan digest, approval or lifecycle state.

| FRZ row of the RW CODE_FIX inventory | Negative test ids in the RW plan |
|---|---|
| FRZ-CFG-02 | NT-CFG-02-1, NT-CFG-02-2 |
| FRZ-CFG-06 | NT-CFG-06-1, NT-CFG-06-2 |
| FRZ-SEC-04 | NT-SEC-04-1, NT-SEC-04-2 |

Binding: `docs/project/maintainers/qualification/arch_freeze_program/rw_plan_bindings.v1.json`. Lane, hard predecessors, merge position and mandatory tests of this ticket are in the live plan `docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md`.

## Execution settlement (AF-W3-PT-FC3-01, 2026-11-27)

Session: `INTEGRAX | ARCH-FREEZE | W3 | PT-FC3 | AF-W3-PT-FC3-01`. Base SHA: `f991794d8755b9a5df7a40959f1995dd4cefead6` (`HEAD` == `origin/development` at preflight). SCOPE=CLOSED; plan digest `475a6642fe356f678c3619abd5551f4fcc55bc80f55a3876406978514b2e0da4` unchanged.

### Parts A–F disposition

| Part | Status | Notes |
|------|--------|-------|
| A Embedding | ALREADY_IMPLEMENTED_VERIFIED | NT-FC3-E1..E6 green in Q1 (54 collected). |
| B Vector store | ALREADY_IMPLEMENTED_VERIFIED | NT-FC3-V1..V4 green in Q1. |
| C Notifications | ALREADY_IMPLEMENTED_VERIFIED | NT-FC3-N1..N13 green in Q1; O-08/L-1 preserved. |
| D Vision | CORRECTED | Residual: post-backend zero detections invoked LAB fallback in Triton/HF adapters (plan §3.4). Fixed; NT-FC3-VIS3/VIS4 extended with injected fallback + `assert_not_called`. |
| E Inbound verification | ALREADY_IMPLEMENTED_VERIFIED | NT-FC3-INT1..INT12 green in Q1. |
| F Guardrails | ALREADY_IMPLEMENTED_VERIFIED | NT-FC3-GR1..GR5 green in Q1. |

### Vision zero-detections evidence

- RED SHA: `f991794d8755b9a5df7a40959f1995dd4cefead6`, exit 1 (`test_nt_fc3_vis4_hf_zero_detections_empty`; HF fallback invoked).
- GREEN: post-fix local run, 11/11 in `test_fc3_vision_fail_closed.py`.
- Evidence logs: `.tmp/session/af-w3-pt-fc3-01/vision_zero_detections_red.log` (SHA256 `32B2733310B25A25DD2A1843818A14601C64DF3EE578FA7D137D3771F1DBBDAA`), `vision_zero_detections_green.log`, `q1_fc3.log`.

### Qualification (this session)

| Batch | Result |
|-------|--------|
| Q1 FC3 selectors | 54 passed (pytest `-o cache_dir=…`; `-p no:cacheprovider` aborts per DEV-03) |
| Q2 CONFIG-X | 77 passed |
| Q3 RW regression slice | 65 passed |
| Q4 plan §6 (78 nodes) | PARTIAL — canonical `regression_nodes.txt` not in repo; `regression_scope_fc3_bundle.txt` lists 28 paths; slice executed via Q1–Q3 only |
| FRZ-CFG-02/06, FRZ-SEC-04 | Owner NTs covered in Q1+Q3; prior red-first disposition preserved |

### Changed paths (this delta)

- `intergrax/model_inference/adapters/triton_vision.py`
- `intergrax/model_inference/adapters/huggingface_inference_vision.py`
- `tests/unit/model_inference/test_fc3_vision_fail_closed.py`

### Residuals (unchanged)

- FRZ-EXE-02 / FRZ-PLG-05: EVIDENCE_MISSING
- STRICT host authority: OPEN_RESIDUAL (PT-FC1)
- OP_B_PENDING=11; RED_FIRST_EVIDENCE=INCOMPLETE
- PT-Q02 handoff prepared in session artifacts; no independent G6 PASS

## Lifecycle state

quality_gate
