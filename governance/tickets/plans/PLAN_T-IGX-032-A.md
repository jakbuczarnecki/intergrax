## PLAN `T-IGX-032-A` - HIST03-F-015: strict production host composition crashes in `wire_skill_execution_pinning_store` when it supplies both a KV store and a document store (PROD-Q P2-C corrective)

**HELD - ACT-N11 sequencing (DR-12=B): do not approve or implement until MASTER confirms the IIAC re-audit/re-activation batch.**

Campaign: Integrax Historical Certification Closure, W1-PLAN step 3 (accepted as a SEQUENCE only; it approves no ticket plan). Authority for the scope: DR-6 (Q-11 commissioned; a ticket with an owner may follow; plan approval is a separate act), W1-PLAN step 3 ("the P2-C corrective with the KV store authoritative and SKILL-X as owner"), DR-12 = B. Planner != implementer != auditor != plan approver. This plan claims no PASS, no closure, no FRZ promotion and changes no FRZ status. No test was run by the planner: the failure is already measured by Q-11 (and HIST-03 iso runs).

Base: `origin/development` `af323c0057748bcaf472ae5290ceaeeba9bac97d` (planner worktree `plan-p2c`, branch `hist-cert/plan-p2c`). Product-path diff `ae467c8c4..HEAD` under `intergrax/ agents/ applications/` is EMPTY at this base (git diff, re-checked by the planner), so this fix is the first product-path change after the IIAC audited SHA.

### 1. Problem and evidence

`tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py`: 6 of 7 tests fail with `ValueError: wire_skill_execution_pinning_store accepts kv_store or document_store, not both` (Q-11, measured: first failure `test_p2c_01_positive_http_admission_continuity`, `1 failed, 1 passed`, EXIT_CODE=1 at `ba07ebaa3`). The net product-path diff `ae467c8c4..HEAD` is empty (above), and the chain below was re-read at HEAD, so the defect is present at HEAD.

Chain, each link READ by the planner at HEAD:

| # | Fact | Where | Status |
|---|---|---|---|
| 1 | Strict composition yields both stores: `ProductionPlatformPersistence` declares `kv_store` and `document_store` mandatory; `resolve_harness_host_profile_persistence_kwargs` returns both | `intergrax/applications/_shared/production_platform_persistence.py:33-38`, `:108-113` | VERIFIED (read) |
| 2 | Callers forward both to the host runtime: governed contractor strict branch and the `elif document_store is not None` branch (passes `document_store` and `key_value_cache`); research app both branches | `applications/governed_contractor_application/host/factory.py:199-209,392-393`; `applications/research_application/host/factory.py:88-114` | VERIFIED (read) |
| 3 | Host runtime coerces to `kv_store` / `doc_store` (both non-None) and calls the pin resolver with both | `intergrax/applications/_shared/harness_host_runtime.py:365-368`, `:419-427` | VERIFIED (read) |
| 4 | Resolver forwards both to the leaf wire function (`elif kv_store is not None or document_store is not None`) | `intergrax/skills/skill_execution_pinning_wiring.py:35-39` | VERIFIED (read) |
| 5 | Leaf rejects both | `intergrax/skills/execution_pinning_persistence.py:490-493` (raise at :491) | VERIFIED (read) |
| 6 | Introduced together with its only product caller by `d1040f485` (SKILL-X Batch C): `git log -S"resolve_skill_execution_pinning_wiring" -- harness_host_runtime.py` lists only that commit | git history | VERIFIED (git log); "green before it" rests on recorded evidence only (Q-11: accepted SHA `8e43b1604`, 6/6 HTTP; P3-A evidence L78), NOT re-run -> THESIS-level for the "was green" claim |
| 7 | The same host runtime already resolves coexisting stores to KV for the effective-profile stores: `profile_document_store = document_store if kv_store is None else None`, while `wire_effective_profile_execution_pinning_store` (the leaf) still rejects both (`persistence.py:344-348`) | `intergrax/applications/_shared/profile_resolution/wiring.py:68`, `persistence.py:339-348` | VERIFIED (read). This is the convention: the RESOLVER picks one store, the LEAF stays strict |
| 8 | Other siblings: provenance reader tests `kv_store` first, document store only as fallback | `intergrax/applications/_shared/integrations/integration_configuration_provenance_reader.py:52-67` | VERIFIED (read) |
| 9 | Durability validator is a type-level claim, not a backing check: it only reads `pinning_store.is_durable`; both `KvSkillExecutionPinningStore` and `DocumentStoreSkillExecutionPinningStore` return a hard-coded `True` | `intergrax/skills/execution_pinning_durability.py:22-30`; `execution_pinning_persistence.py:291`, `:388` | VERIFIED (read). Q-11 had not read this body |
| 10 | The strict reference composition builds a sqlite KV (`db_path`) plus `create_reference_in_memory_document_store()` | `production_platform_persistence.py:55-57`; Q-11 report | Partly VERIFIED (`db_path=` seen); the in-memory document store is Q-11-reported, not re-read |

Existing contract test that matters: `tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py:503` `test_wire_skill_execution_pinning_store_rejects_dual_backends` asserts the LEAF rejects both. It must stay green and untouched (see 2).

Not measured (planner ran nothing): the exact set of other tests that fail for the same reason. Any strict or both-store host build goes through the same line (research app, the governed-contractor `prod_q_r1_*` strict unit tests, the lab/non-composition governed-contractor branch). They are candidates, UNMEASURED; the implementer measures a bounded named set before and after (section 7, step 6).

### 2. Design alternatives

Q-11 recommended owner SKILL-X and KV authoritative as recommendations, not decisions. Evaluated here, then a recommendation.

| Alt | Change | Blast radius | Durability semantics | Fail-closed | Compatibility |
|---|---|---|---|---|---|
| **A2 (RECOMMENDED)** resolver resolves both to KV; leaf stays strict | `resolve_skill_execution_pinning_wiring`: pass `document_store=document_store if kv_store is None else None` to the leaf (same line shape as `wiring.py:68`) | 1 product file (`skill_execution_pinning_wiring.py`). Callers of the resolver: only `harness_host_runtime.py:423` (grep, product code) plus unit tests `test_skill_x_batch_c...py:442,450`. Callers of the leaf: only the resolver plus the unit tests at `:507` (grep). Covers every host that supplies both (governed contractor strict and non-strict, research app, scaffolds) with one edit | KV wins when present: the only store in strict composition that is genuinely durable (sqlite). Document store is used only when no KV exists | Unchanged: neither store + production -> in-memory pin store -> `SkillResolutionError` (durability validator); doc-only non-conditional -> `TypeError` (leaf); explicit `pinning_store` still wins. New: a document store that is not `ConditionalDocumentStore` is ignored for pinning when a KV exists (same as the profile sibling) | Leaf signature and its dual-backend rejection unchanged; existing leaf test untouched. Resolver signature unchanged; only the "both" case changes from `ValueError` to KV |
| A1 leaf accepts both and resolves to KV | `wire_skill_execution_pinning_store` drops the `not both` raise, prefers KV | 1 product file, but inverts `test_wire_skill_execution_pinning_store_rejects_dual_backends` (an assertion would be deleted or reversed: gate-weakening optics, needs a ruling) and diverges from the sibling leaf, which still rejects both | same as A2 | the explicit-API misuse guard is lost: a caller passing both by mistake silently gets KV | breaks a published contract test |
| B host runtime passes only one | `harness_host_runtime.py:423-427`: `document_store=doc_store if kv_store is None else None` | 1 product file, but the resolver stays a trap for any future caller; the policy lives at one call site and not in the resolver the profile sibling uses | same as A2 | same as A2 | no test change; acceptable fallback if the Operator wants the resolver left untouched |
| C strict composition supplies only one store | `resolve_harness_host_profile_persistence_kwargs` (or `ProductionPlatformPersistence`) stops returning `document_store` | `production_platform_persistence.py` plus governed contractor and research factories; `document_store` is also forwarded to `wire_application_environment` (`harness_host_runtime.py:411`) and used by other consumers (collaborative-work profile, integration wiring) that were NOT traced | would drop the document store from strict hosts globally | fixes only the composition branches; the `elif document_store is not None` governed-contractor branch and the research non-composition branch still pass both | `ProductionPlatformPersistence` declares both mandatory; large unverified blast radius. REJECTED |

Recommendation: **A2**. Reasons: (1) it is the existing convention in the same host runtime (`wiring.py:68`: resolver chooses, leaf strict); (2) one product file, three changed lines, no existing test inverted; (3) it fixes every caller that supplies both, not only strict composition; (4) KV authoritative is the right durability answer because `is_durable` is hard-coded `True` on both pin stores (row 9): if the in-memory reference document store were ever chosen in strict mode, the validator would still pass and pins would be lost on restart. Confirmed Q-11 recommendations: owner SKILL-X, KV authoritative. Refinement of Q-11's wording ("the pin wiring accepts both"): the resolver accepts both; the leaf keeps rejecting.

### 3. Exact edit scope (phase 2 may change ONLY these)

| # | File | Action |
|---|---|---|
| 1 | `intergrax/skills/skill_execution_pinning_wiring.py` | PRODUCT, ~3 lines: pass `document_store` to the leaf only when `kv_store is None`; keep the branch conditions; add a one-line comment that KV is authoritative; leave `wire_skill_execution_pinning_store`, the durability validator, `harness_host_runtime.py` and all factories untouched |
| 2 | `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py` | NEW test file (RED-first, section 5) |

Forbidden, absolute: `execution_pinning_persistence.py` (leaf and its `not both` guard), `execution_pinning_durability.py`, `harness_host_runtime.py`, any factory, the P2-C test file, `test_skill_x_batch_c_versioning_provenance_recovery.py` (including `...rejects_dual_backends`), pytest markers, `conftest.py`, `pyproject.toml`, selection manifests, the six `MATERIAL_GOVERNANCE_PATHS`, `scripts/`, IIAC / roadmap / FRZ records.

Commit trailer for the product commit: `Ref: governance/tickets/T-IGX-032-A.md`. Two commits recommended: (1) the new test file only (RED proof on this tree), (2) the product change (GREEN). Commit order gives the auditor a clean before/after.

### 4. ACT-N11 and governance paths

- Product paths touched (under `intergrax/ agents/ applications/`): exactly ONE: `intergrax/skills/skill_execution_pinning_wiring.py`. Nothing under `agents/` or `applications/`. The test file is under `tests/` (not a product prefix: `PRODUCT_PATH_PREFIXES` in `scripts/governance/iiac_common.py:22`).
- None of the six `MATERIAL_GOVERNANCE_PATHS` (`scripts/governance/iiac_activation_gate.py:40-49`) is touched; no `scripts/` file is touched.
- Consequence: `verify_iiac_policy_activation --require-in-force` goes red with one `ACT-N11 product path changed after the audited SHA` line for this path (plus the other held product tickets' paths). This is EXPECTED and is resolved only by the single IIAC re-audit/re-activation batch (W1-PLAN step 4). The implementer records the line verbatim, does not fix it, and does not touch gate or IIAC records.
- Lifecycle: ticket `plan_pending`, HELD, no `Approved:` line. Neither approval nor implementation starts before MASTER confirms the re-activation batch (DR-12 = B).
- File-disjoint from `T-IGX-030-B` (`contracts/runtime_error_code.py`, `error_codes.py`, `tool_operation_recovery.py`) and `T-IGX-030-C` (`invoker.py`); no ordering dependency; one IIAC cycle can cover all.

### 5. Test design: RED-FIRST

New file `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py` (positive and negative), calling `resolve_skill_execution_pinning_wiring` directly. Doubles: local KV and `ConditionalDocumentStore` doubles (same shape as `test_skill_x_batch_c...py:132,173`, copied, not imported from a test module); at least one case should use the real reference stores built like strict composition (`build_reference_production_platform_persistence`, signature not verified by the planner) to mirror the production shape.

| Test | Input | Expected | Before fix | After fix |
|---|---|---|---|---|
| T1 both -> KV | `production_mode=True`, `kv_store=KV`, `document_store=conditional doc` | result is a `KvSkillExecutionPinningStore`, `is_durable` True, no exception | RED: `ValueError ... not both` (exactly the P2-C error) | green |
| T2 both, non-conditional doc store -> KV | same, doc store is a plain `DocumentStore` | KV store, no `TypeError` | RED: `ValueError ... not both` | green |
| T3 KV is the write target | resolve with both, bind one pin through the result, read it from the KV double | pin readable from KV; doc double saw no write | RED (ValueError) | green |
| T4 doc store is the fallback | `document_store=conditional doc` only, production | `DocumentStoreSkillExecutionPinningStore` | green | green |
| T5 non-conditional doc only | plain `DocumentStore` only | `TypeError` ("requires ConditionalDocumentStore") | green | green (fail-closed kept) |
| T6 nothing, production | no stores, `production_mode=True` | `SkillResolutionError` with `DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG` | green | green |
| T7 explicit pin store wins | `pinning_store=X` plus both stores | result store is `X` | green | green |
| T8 leaf still strict | `wire_skill_execution_pinning_store(kv_store=KV, document_store=doc)` | `ValueError` "not both" | green | green (proves the fix is in the resolver, not the leaf) |

The RED set (T1, T2, T3) fails for exactly one reason, the `not both` `ValueError`; the assertions of T4-T8 pass on the unfixed tree, so a red outside T1-T3 would mean a wrong test, not a wrong fix.

Negative probes (uncommitted scratch edits on the final tree, then restored; `git diff` empty afterwards):
- P-A: revert only the product line (pass both again) -> exactly T1, T2, T3 red with `not both`; T4-T8 green.
- P-B: invert precedence (document store wins when both) -> T1 and T3 red (wrong store type / wrong write target), T2 red; T4-T8 green. Shows the tests pin KV authority and not merely "no exception".
- P-C: the leaf guard in `execution_pinning_persistence.py` removed (scratch only) -> T8 and the existing `...rejects_dual_backends` go red. Shows the strict leaf is still pinned by two tests.

No gate weakening: the P2-C file is byte-identical to HEAD (`git diff --stat -- <P2-C file>` empty); no `skip`/`xfail`/marker added; the existing `...rejects_dual_backends` test is untouched and green; the only new code is the 3-line resolver change and a new test file.

### 6. Roles

Implementer, independent auditor and plan approver are three different fresh sessions, none of them the planner. `Approved:` is written only by the Operator or by a commissioned `operator-delegate` session (README forms); never by the implementer or the planner. The auditor: (1) re-runs section 7 on the exact final SHA; (2) checks the diff guard (one product file, one new test file, ticket/plan files); (3) runs probes P-A, P-B, P-C itself; (4) checks that the P2-C file and `...rejects_dual_backends` are byte-identical; (5) re-checks the A2-vs-B choice if the Operator did not rule on it; (6) independently verifies that the strict-composition KV is the sqlite store and that the document store there is the in-memory reference one (row 10); (7) reviews the unmeasured wider set (section 7, step 6) before and after.

### 7. Test contract (phase 2)

Common rules: ONE pytest / ruff / uv process at a time; each under the heavy lock `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/locks/heavy.lock` (acquire with `mkdir`, write `holder.txt`, release with `rm holder.txt; rmdir`; never remove a lock you do not hold; on BUSY wait); pytest's own exit code captured with `; echo EXIT=$?` into the log, not through a pipe; logs under `.tmp/session/t-igx-032-a/`; always `uv run --extra dev-ci`; no xdist, no `--basetemp` outside the repo.

Paths: `NEW=tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py`, `P2C=tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py`, `BC=tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py`.

| Step | Command | Before the product commit | After |
|---|---|---|---|
| 1 | `uv run --extra dev-ci pytest $NEW -q` (RED proof, tests commit only) | T1, T2, T3 failed with `not both`; T4-T8 passed | all passed, exit 0 |
| 2 | `uv run --extra dev-ci pytest $P2C -q` | MEASURED by Q-11/HIST-03: 6 failed, 1 passed (`g00`); the implementer re-measures and records | **7 passed**, exit 0 |
| 3 | pinning tests, ONE process: `uv run --extra dev-ci pytest $BC tests/unit/skills/test_p1_10_execution_binding.py tests/unit/skills/test_skill_x_batch_d_trust_boundary_qualification.py tests/unit/skills/test_skill_x_batch_e_integrated_authority_tenant_qualification.py tests/unit/applications/test_effective_profile_persistence_wiring.py -q` | record | all passed (including `...rejects_dual_backends`) |
| 4 | probes P-A, P-B, P-C (section 5) on the final tree, one `$NEW` run each (P-C also runs `$BC -k dual`) | n/a | the predicted red sets; actual sets recorded, any other result reported not explained away |
| 5 | `uv run --extra dev-ci ruff check` then `ruff format --check` on the 2 files | n/a | clean |
| 6 | bounded wider measurement, ONE process, named files only, run before and after the product commit and compared: `applications/research_application/tests/host/test_research_factory_registry_projection_e2e.py` and the strict governed-contractor unit tests `tests/unit/applications/governed_contractor/test_prod_q_r1_r6_strict_gca_lifecycle.py`, `test_prod_q_r1_r2_gca_startup_continuation.py`, `test_prod_q_r1_dur_02_gca_composition_binding.py` (candidates; the implementer may adjust the list but may not widen it to a directory or the fast tier) | record which fail with `not both` | those failures gone; any other red is reported with its own cause and NOT fixed here |
| 7 | diff guard: `git diff --name-only af323c005..HEAD` | n/a | only the two files plus `governance/tickets/T-IGX-032-A.md` and `governance/tickets/plans/PLAN_T-IGX-032-A.md`; P2C and BC byte-identical |
| 8 | ACT-N11 line, informational: `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` | n/a | EXPECTED red with the one product path; record verbatim, do NOT fix |

Test contract statement: the 7 tests of `$P2C` are the acceptance contract of this ticket (7 passed, file unmodified); the pinning tests of steps 1 and 3 are the regression contract. A run of `$P2C` is HEAVY (about 30 s per Q-11) but bounded and is named here; broad diagnostic suites and `igx_gov ticket` (broad `fast` tier) need a separate Operator authorization (C10/C15, `agent-env/rules/intergrax-env-vendor-qualification.md`).

Stop rules: if step 2 stays red after the fix with a DIFFERENT error, STOP, record the exact test and error, do not widen the edit scope, report to MASTER (another defect hides behind the first; Q-11 saw only the first failure of `-x`). Any need to edit a forbidden file leaves this batch.

### 8. Rollback

`git revert` of the product commit restores the pre-change (P2-C red) state; the test file can stay (T1-T3 would go red again, which is the intended RED). No data, no schema, no gate change: pin records written through the KV store are the same as the ones the KV path already wrote in single-store hosts.

### 9. Risks and residuals

- R1 (Operator question, security-relevant, NOT fixed here): `is_durable` is a hard-coded `True` on both pin stores, so `validate_skill_execution_pinning_durability_for_composition` cannot detect a non-durable backing. A host that supplies only an in-memory document store in strict mode passes the validator and loses pins on restart. KV-first avoids it for strict composition, but a doc-only strict host is still accepted. Candidate for a separate SKILL-X finding; the profile pinning sibling should be checked for the same property (not read by the planner).
- R2: with KV present, a non-`ConditionalDocumentStore` document store is silently ignored for skill pinning (as the profile sibling does). Accepted by the convention; the auditor confirms.
- R3: the "was green at `8e43b1604` / the parent of `d1040f485`" claim is documentary; Q-11 asked MASTER for a parent-commit run and it was not done. The fix does not depend on it; P2-C green after the fix is the evidence that matters.
- R4: unmeasured wider blast radius (section 1, section 7 step 6). If the research app or other strict hosts fail for this same reason today, the fix also repairs them; if they fail for other reasons, those are separate findings.
- R5: PROD-Q P2-C is the victim, not the owner (Q-11). SEC-04, PRD-08, PRD-09 stay downgraded until the independent audit and the P2-C re-run on the final SHA; this ticket changes no FRZ status and no qualification record.
- R6: ACT-N11 red after this change until the IIAC re-activation batch; sequencing is the Operator's (DR-12 = B).

### 10. Questions for the Operator

1. Placement: confirm A2 (resolver resolves both to KV; the leaf guard and its test stay) over B (host runtime call site) and A1 (leaf accepts both, test inverted). Default if no answer: A2.
2. Owner SKILL-X and KV authoritative (Q-11 recommendations; W1-PLAN step 3 names both): confirm as decisions for this ticket.
3. Step 6 of the test contract measures five named extra files before and after. Confirm this bounded run under C10/C15, or reduce it to `$P2C` plus the pinning tests only.
4. Whether R1 (hard-coded `is_durable`) should become a separate finding and ticket now.

Could not verify: any "after" column (planner writes no test and ran none); that `$P2C` has no second defect behind the first (Q-11 ran `-x`); the strict KV being sqlite and the strict document store being in-memory (taken from Q-11 / partial read); the extra files' current status; whether one IIAC re-activation also covers the ENT-AUDIT-X stream (W1-PLAN step 2); the profile-pinning durability validator's body.

### 11. Out of scope

Everything outside the two files of section 3; the leaf `wire_skill_execution_pinning_store` and its dual-backend rejection; the durability validator and R1; `harness_host_runtime.py` and the factories; `ProductionPlatformPersistence` and strict composition; the P2-C test file and any edit of its roster; PROD-Q / SKILL-X records, roadmap and FRZ statuses; the six IIAC material governance paths and any IIAC record or activation; `T-IGX-030-B`, `T-IGX-030-C`, `T-IGX-031-A`, BND-03, `T-IGX-021-B/C`; PASS / closure claims; Docker, vendor and production configuration (C10/C15); push, merge, rebase, PR.
