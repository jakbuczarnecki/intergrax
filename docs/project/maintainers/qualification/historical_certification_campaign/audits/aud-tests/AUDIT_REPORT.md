# AUDIT_REPORT - T-IGX-031-A (tests-only), independent auditor `aud-tests`

Auditor: fresh session, not plan-tests / rev-plan / cert-031a / impl-tests / MASTER. Pinned detached checkout `D:/Projekty/_wt/intergrax/aud-tests` @ `b28398026645093a455eeb19b2808e7998d238e2` (impl branch hist-cert/impl-tests, base `af323c005`). All runs below are my own; implementer logs were not reused.

Read scope: brief, ticket, plan, implementer handoff, both authorization records, full `git diff af323c005 b28398026`, `security_runtime_bridge.py`, `SecretsStoreIntegrationContract`, `runtime_event_test_identity`, HIST-03 pin records.
Edit scope: only two uncommitted scratch edits in my checkout (probes A and B, both restored); this report and logs under `audits/aud-tests/`. No commit, push, merge, rebase, reset, amend.
End state of checkout: `git status --short` empty, `git diff` empty, HEAD = b28398026.

## VERDICT: PASS_SCOPED

Scope that passes: the five-file tests-only diff `af323c005..b28398026` (helper + three W5 autouse fixtures + `test_sec_ent.py` ledger S1-S11/S1b) is within plan scope, weakens no gate, and does what the plan claims; the 4-file/34-test isolation proof holds in both orders; the 27-path selection shows no `already registered` and none of the 15 pin failures that belong to this batch.

Not covered by this PASS: (a) step 11 exit code was not captured (see "Could not verify" - the process did not exit after the pytest summary); (b) F-015 (6 P2-C tests) stays red, out of scope; (c) R1/R2/R3(partly)/R7 and the `no_ci` exclusion stay open; (d) pre-existing ruff/format debt is accepted, not fixed; (e) this is not G6 closure / FRZ promotion of anything.

## Per-step evidence (all exact SHA b28398026, `uv run --extra dev-ci`, heavy lock, one process at a time)

| Step | Command | Result | Log |
|---|---|---|---|
| 0 scope | `git diff --name-only af323c005 b28398026` | exactly the 5 plan files; nothing under intergrax/ agents/ applications/ scripts/, no governance/ticket/plan/conftest/pyproject/manifest path; `no_ci` still in `pytestmark` (context line of hunk); `git diff --check` clean; 0 CR bytes in diff; numstat 30/10/9/9/69+17- (no whole-file rewrite) | - |
| 0 digest | `python scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-031-A.md` | `a83842545c6e...7267509` = approved digest in ticket L18; plan file unchanged between base and impl SHA | - |
| 1 | SE alone | 14 passed, exit 0 (67.7 s) | step1_se.log |
| 2 | B2 C D | 20 passed, exit 0 | step2_w5.log |
| 3 | iso A: SE B2 C D, one process | 34 passed, exit 0 | step3_isoA.log |
| 4 | iso B: D C B2 SE, one process | 34 passed, exit 0 | step4_isoB.log |
| 5 | single node `test_security_spine_counters_increment` | 1 passed, exit 0 (no event-kind fallback needed) | step5_node.log |
| 6 | PROBE A (scratch, restored) | 4 failed / 10 passed, exit 1. Actual red set: `..._fails_closed_on_resolution_error` (S6, DID NOT RAISE), `..._fails_closed_on_conformance_error` (S4, `TypeError ... is object, expected SecretsStoreIntegrationContract`), `test_harness_envelope_encryptor_not_selected_by_resolver` (S7, DID NOT RAISE), `test_security_spine_counters_increment` (S10, `RuntimeError: active execution identity required`). Exactly the plan prediction | probeA.log |
| 7 | PROBE B (scratch, restored) | 12 failed / 22 passed, exit 1. Red: SE S6, S7, S8 `uses_secrets_store_when_resolvable`, S9 `register_security_hooks_wires_spine_subscriber`; b2 x5 (`production_composition_creates_bounded_sink`, `published_event_reaches_bounded_sink`, `shutdown_ordering_closes_bus_and_sink`, `no_singleton_leakage_between_environments`, `legacy_composition_without_sink_preserves_behavior`); c x2 (`shutdown_flush_and_close_exporter`, `instance_isolation_between_environments`); d x1 (`ten_runtime_instances_isolated_exporters`). Exactly the prediction; proves the fixtures (not luck) make step 3 green | probeB.log |
| 8 | `verify_iiac_policy_activation --require-in-force` | exit 0, `IN_FORCE @ IIAC-2026-11-05.1` | step8a_verify.log |
| 8 | governance tests (activation_gate, negative_guards, evol_x_real_repo_fresh_clone, g0_inventory) | 65 passed, 1 skipped (`test_iiac_activation_gate.py:439` symlink creation unavailable on this host), exit 0, 679.8 s. The new `tests/support/` file did not trip ACT-N11 / G0; inline fallback not needed | step8b_gov.log |
| 9 | ruff check / format --check, HEAD vs base `git show af323c005:<f> \| ruff ... --stdin-filename` | see below | step9_ruff.log, step9_ruffdiff.log |
| 11 | 27-path single process (see below) | 6 failed, 347 passed, 1 warning, 158.9 s; **exit code not captured** | step11_27path.log |

Probe restore: Probe A via `git restore -- tests/unit/runtime/security/test_sec_ent.py`; probe B via `git restore -- tests/`; `git status --short` and `git diff --stat` empty after each and at the end.

### Step 9 - lint (the Operator's `--no-verify` acceptance covers ONLY the base set)
| File | HEAD ruff check | BASE ruff check | HEAD format | BASE format |
|---|---|---|---|---|
| tests/support/integration_catalog_isolation.py (new) | clean | n/a | clean | n/a |
| w5_b2 | I001 | I001 | clean | clean |
| w5_c | I001, E402 (L100) | I001, E402 (L91) | would reformat | would reformat |
| w5_d | I001 | I001 | would reformat | would reformat |
| test_sec_ent.py | I001, F401 `RuntimeEventBus` (L265) | I001, F401 (L223) | clean | clean |

Same rules, same count (6 findings, 2 format files) at base and HEAD. **No NEW finding.** Plan step 9 expectation "clean" does not hold, because the base is already red (the plan was wrong on that, the implementer correctly reported it). Informational: in b2/c/d the new `from tests.support...` import lands inside the already-flagged I001 block and adds one more line to the autofix diff (ruff would order it before `lab_application`, like the existing `testing_support` line); `test_sec_ent.py` new imports need no move. Residual lint debt is untouched and needs its own ticket.

### Step 11 - 27-path run (AUTH-HIST-27PATH-2026-11-08; sha256 of the record verified = 1cf2c0a8...1184c6b2)
Command: the 27 paths of `hist03/run1.sh`, in the same order (extracted by grep, 27 lines), only the pin dir replaced by my checkout: `uv run --directory D:/Projekty/_wt/intergrax/aud-tests --extra dev-ci pytest <27 paths> -q`, serial, one process, heavy lock, run last after steps 1-9, no retry used.

| | Pin (ba07eba, HIST-03) | This batch (b28398026) |
|---|---|---|
| result | 21 failed, 331 passed (352) | **6 failed, 347 passed (353 = 352 + new S5)** |
| `already registered` | 54 occurrences in the log (redis x17+, ollama x1 in tracebacks) | **0** |
| `test_sec_ent.py` failures | 7 | 0 |
| W5 b2/c/d failures | 8 | 0 |
| P2-C `p2_c_http_execution_admission` | 6 failed | the same 6 failed (test_p2c_01..06); all 12 traceback hits are `ValueError: wire_skill_execution_pinning_store accepts kv_store or document_store, not both`, the error recorded for F-015 in `HIST03_FINDINGS_AND_RESIDUALS.json` L417 - genuine, pre-existing, NOT caused by this batch |

The set difference is exactly: -7 (SE) -8 (W5) fixed, +1 new passing test (S5); the 6 P2-C failures are identical test ids. No failure is attributable to the clean-after fixtures leaving the catalogs empty: every file after the four fixtured ones in the selection (and the W5 a/b/g/h observability files, hosting/readiness files, ctrl_x, prod_q) passed.

## Ledger table (plan section 5) - hunk by hunk
| Row | Verdict | Evidence |
|---|---|---|
| S1 `_clean_catalogs` autouse in SE | conforms | before/after `reset_integration_tool_skill_catalogs()`; no assertion touched |
| S1b `_reset_store` also clears `_PrebuiltFakeSecretsStore.stored` | conforms | one added line |
| S2 `_PrebuiltFakeSecretsStore(SecretsStoreIntegrationContract)` | conforms | `stored: ClassVar[dict]`, same method bodies as `_FakeSecretsStore` (plus `version` kwarg on get_secret); `_FakeSecretsStore` unchanged |
| S3 valid secrets store | conforms | only `store = ...for_provider(provider_id="sec-ent-fake")`; isinstance / `__secret_ref` / envelope / stored-value assertions byte-identical (not in the diff) |
| S4 conformance error | conforms | input `object()` -> category contract without `get_secret/put_secret/delete_secret`; same exception type, same message assertion; passes only through the conformance gate (probe A: reverting to `object()` goes red with the guard TypeError) |
| S5 new type-guard test | conforms | pure addition; pins raw `TypeError` (= R7, documented) |
| S6 resolution error | conforms, strengthened | `_bootstrap_catalog()`; patch target now `...security_runtime_bridge.resolve_from_profile` (correct: bridge does `from ...factory import resolve_from_profile`, L14); all old assertions kept + `str(__cause__) == "secrets backend unavailable"` |
| S7 harness envelope not selected | conforms, strengthened | same two replacements; `as exc_info` + `__cause__ == "resolution failed"` added |
| S8 / S9 | conforms | only the bootstrap line replaced; assertions untouched |
| S10 spine counters | conforms | identity bound via `runtime_event_test_identity` + `bind_active_execution_identity` in try / `reset_active_execution_identity(token)` in finally; payload, `await bus.publish(event)` and `counters.defense_blocked == 1` kept (inside the try) |
| S11 `_bootstrap_catalog()` | conforms | one-line helper calling `bootstrap_application_integration_catalog()` |
| W5 b2/c/d fixtures | conforms | identical 5-line autouse fixture + `Iterator` and helper imports; zero test-body change |
| helper file | conforms | the 8 public reset/clear functions in the planned order; no private names; clean under ruff |

Gate-weakening checklist: no `skip`/`xfail`/marker added or removed; `no_ci` kept; no assertion deleted, loosened, widened to set/`in`; no swallowing try/except (the only new try/finally restores the identity); every "kept" item is absent from the diff; no effect/ledger/replay check dropped (S3 still checks the vaulted value; S6/S7 gained cause checks); negative probes kept or added (S4, S5; S6/S7 now inject the fault for real, proven by probe A); added code is isolation and test doubles only. **No gate-weakening found.** The `unused RuntimeEventBus` and the unused-import F401 pre-exist and were not touched.

## Findings
| ID | Sev | Location | Finding | Fix |
|---|---|---|---|---|
| AUD-1 | Low (process) | step 11 execution | After pytest printed its summary the `uv`/`pytest` process did not exit (observed ~35 min, `runlocked` wrapper never wrote `.exit`; a `coroutine 'DependencyAttemptExecutionBoundary.acquire.<locals>._acquire_coro' was never awaited` RuntimeWarning is in the log, plausibly an unjoined thread at interpreter exit). The pin run (HIST-03) exited normally, so this is either environmental or a hang introduced/uncovered by the new selection state. Not shown to be caused by this batch (SE/W5/step 1-5 runs all exited 0 normally). | Operator/MASTER: stop the lingering process, release `locks/heavy.lock` (held by me, `holder.txt` "aud-tests auditor step11_27path"), and decide whether to re-run once (the authorization allows one identical retry only for an INFRASTRUCTURE error; I did NOT retry). Treat the hang as a separate observation to look at, not as a batch defect without further evidence. |
| AUD-2 | Info | plan section 7 step 9 | Expected "ruff clean" is false at base for 4 of 5 files; 6 check findings + 2 format files pre-exist. Accepted by AUTH-HIST-NOVERIFY for these only. | Separate lint-debt ticket if wanted. Do not fold into this batch. |
| AUD-3 | Info | b2/c/d imports | New `tests.support` import is placed per the files' own manual order, inside the existing I001 block. No new rule. | none |
| AUD-4 | Info | handoff | Handoff claim "S3 works, escape clause not triggered": verified (step 1). Claim "step 8 619 s": mine 679.8 s, same result. BEFORE numbers (7 failed/6 passed; 20 passed; 15 failed/18 passed) were NOT re-measured by me; they match the planner's and pin records, not a basis of this verdict. | none |

Nothing must be corrected before merge on the strength of this audit.

## F-016 status the evidence supports
F-016 may be called **resolved for the 27-path selection of `hist03/run1.sh` at b28398026**: step 11 shows **0** `already registered` (pin: 54), 0 SE failures, 0 W5 failures, and the isolation proof passes in both orders plus single node. Conditions of that statement: the exit code of step 11 was not captured (pass/fail counts come from pytest's own summary line, `6 failed, 347 passed`); the claim is limited to this selection - R2 (about 72 other bare `register_default_integrations()` callers) stays unmeasured, and R1 (duplicate `ollama` slug) is a product defect that is merely side-stepped by the tests via `bootstrap_application_integration_catalog()`. R3: the "separate processes" workaround in selection manifests may be lifted for this selection after merge of the audited batch; no manifest was touched here.

## Residuals judged
R1 open (product, W1-PLAN step 3). R2 open (unmeasured). R3 supported as above, conditional on AUD-1 exit-code gap. R7 open (S5 pins raw `TypeError`; must be updated together with the product decision). `test_sec_ent.py` remains `no_ci`, so GitHub CI will not run it even though it is green. R4 untouched (the other F-006 file already passes: it is in the 27-path run, green).

## Honesty of the handoff
Every checkable claim was verified or measured within 10% (see AUD-4); no claim refuted. The `--no-verify` deviation is real and disclosed; the new content is lint-neutral. The handoff correctly said it was not proof of F-016 and left step 11 to me.

## Could not verify
- Real exit code of step 11 (process stayed alive after the summary; process inspection via PowerShell was refused by the harness classifier and I did not work around it). pytest semantics imply exit 1 for `6 failed`.
- The BEFORE measurements at base (not required by the brief).
- Clean-checkout equivalence beyond my pinned worktree (the `.venv` in it was created by my first `uv run`, gitignored).
- Anything about CI (`no_ci`), Docker, vendors (not touched).
