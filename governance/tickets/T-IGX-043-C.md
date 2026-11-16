# T-IGX-043-C — EQO-A remediation FC-3: vendor/provider selector fail-closed

## Scope

Make four silent vendor/backend selectors fail closed in PRODUCT and STRICT. Findings of the Campaign A dossier (PR #589, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`): EQOA-A4-017 (vision stub fabricates a 0.99 `object.stub` detection), EQOA-A4-019 (embedding provider defaults to `ollama`, vector store to in-memory, notification backend to `log`), EQOA-A4-026 (inbound interaction verification defaults to a no-op verifier; unset or invalid mode resolves to `none`), EQOA-A4-032 part c (guardrail factory pass-through stub and silent degradation to the pattern engine). Master-plan node `PT-FC3`; decisions DP-24 (hybrid: wiring layers fail closed on STRICT or PRODUCT, leaf selectors default-deny with an explicit typed LAB opt-in), DP-25, DP-28 (vision stub only through an explicit LAB profile with a provenance flag), DP-18, DP-14, DP-11; conditions C3, C5, C6.

The CONFIG-X scan extension (former Part G) is **not** part of this ticket. It is the separate draft ticket `T-IGX-043-F`, a gate-meaning change that needs the Operator's personal ruling and is not approvable by a delegate. This ticket does not depend on it for its own correctness (the existing CONFIG-X gates stay unchanged and must stay green); `T-IGX-043-F` merges after this ticket or in the same wave behind it.

Write set, separated (details in the plan, section 4):

- Product paths, ACT-N11 sensitive (any change after the IIAC-bound product tree breaks the activation until the single RB-2 re-pin): 32 files in `intergrax/` named by the design plus 30 callers (seven Tier-3 host factories, `local_workspace` host runtime composition, bootstraps, LAB surfaces, two LAB scripts, two scaffold files, two long-running modules, five files in `applications/_shared`) = 62 files; the Operator's accepted bound was "about 59"; the three above it were accepted (O-09 = option 1, transcription of 2026-11-16). The posture helper `intergrax/applications/_shared/wiring_posture.py` is imported, not created: it belongs to `T-IGX-043-B`, which merges first.
- Tests, not ACT-N11 paths but outside the EBH-7 identity allowlist (they merge before the final pin): six new test files with 51 negative tests, and the inversion of six tests that pin today's silent behaviour (plus default-dependent tests found by a bounded census; `applications/*/tests` files are ACT-N11 sensitive by prefix).
- Documents: none in this ticket (plan section 4 D).

Notifications (restored Q-FC3-02 option 1): no silent `log` default for HITL escalation and long-running notifications in PRODUCT/STRICT. The explicit adapter reaches `NexusLoop` and the scheduler through existing keywords (`notification_adapter` of `build_harness_host_runtime` and of `wire_harness_host_long_running_scheduler`), so no file of FC-1 or W-A/W-B is edited for it; the coordinator refuses an escalation with no injected adapter and no explicit channel. The impact class of the leg stays BLOCKS. Paths without a composition-time refusal are named in the plan (section 3.3); the Operator accepted the plan as drafted for them (O-08 = option 1: late refusal by the coordinator guard, class BLOCKS kept, the unrefused paths named in the audit).

The Operator's answers of 2026-11-16 (O-08, O-09, L-1, merge order B, A, C, F) are recorded in `OPEN_QUESTIONS.md` from the transcription `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md`, a session artefact and not yet a repository record. Implementation starts only when milestone M1 holds (whether PDC P9 is formally closed is not decided), the prerequisites of the plan (section 2) are met and the authorizations A-01..A-10 are given. The Operator's acceptance of the design questions Q-FC3-01..11 (recommended options) was relayed by the orchestrating session on 2026-11-16; the decision record is a session artefact under `.tmp`, not a repository record (`OPEN_QUESTIONS.md` states exactly what was accepted). No closure, PASS, FRZ promotion or independent-verification claim is made by this ticket; an independent audit at the exact wave SHA is part of the test contract.

## Plan digest

Plan digest: cd040d9b943b126757827c2656a3a1e4f4b71ea96c08e2932eeb94a9930fab76

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-C.md`](plans/PLAN_T-IGX-043-C.md). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T13:52:59+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@c98a0d7d67a0b52b8e7e0e279baa0d906bb9c45d

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

## Lifecycle state

plan_approved
