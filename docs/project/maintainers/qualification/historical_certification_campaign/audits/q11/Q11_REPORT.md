# Q-11 REPORT - root cause of HIST03-F-015 (P2-C HTTP admission red)

Investigator: fresh, read-only. Pin worktree `pin-ba07eba` @ ba07ebaa3, 0 tracked changes. Test runs used: 1 of 2. Heavy lock acquired and released (locks dir empty afterwards).

## Verdicts

| # | Question | Verdict |
|---|----------|---------|
| 1 | Code defect vs config/fixture conflict | **CONFIRMED: code defect.** Env/fixture-leak alternative **REFUTED**. |
| 2 | Green where P2-C was accepted? | **CONFIRMED by recorded evidence** (accepted SHA `8e43b1604`, "6/6"; again 6/6 in P3-A evidence). My own re-run at that SHA is **INCONCLUSIVE** (no worktree available; request below). |
| 3 | Commit that made it red | **CONFIRMED: `d1040f485` SKILL-X Batch C** (2026-10-23), by code reading plus traceback. A green run at its parent `97a0771fc` is **INCONCLUSIVE** (same request). |

## Evidence

### Q1 - code defect

Command (pin worktree, serial, under heavy lock):
`uv run --extra dev-ci pytest tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py -x -q`
Result: `1 failed, 1 passed in 29.07s`, `EXIT_CODE=1` (log: `audits/q11/run1.log`).
First failure: `test_p2c_01_positive_http_admission_continuity`.
Innermost error: `ValueError: wire_skill_execution_pinning_store accepts kv_store or document_store, not both` at `intergrax/skills/execution_pinning_persistence.py:491` (the `raise` statement; the brief's :492 is the same statement).

Call chain that produces the double store (all product code; nothing from the test or the environment):

1. `applications/governed_contractor_application/host/strict_production_serving.py:40` `create_governed_contractor_strict_production_asgi_app`
2. `host/main.py:59` `create_governed_contractor_process_app`
3. `host/factory.py:365` `create_governed_contractor_backend_app`.
   - Strict branch `process_composition is not None` (factory.py:199) calls `resolve_harness_host_profile_persistence_kwargs_from_composition`.
   - That returns, from `intergrax/applications/_shared/production_platform_persistence.py:111-114`, BOTH `key_value_cache=platform_persistence.kv_store` AND `document_store=platform_persistence.document_store`.
   - `ProductionPlatformPersistence` (same file, :33-38) declares both as mandatory fields. `build_reference_production_platform_persistence` (:55) builds a sqlite KV plus `create_reference_in_memory_document_store()`. So strict production composition always supplies both.
   - factory.py:392-393 forwards both to the host runtime.
4. `intergrax/applications/_shared/harness_host_runtime.py:365-368` coerces them to `kv_store` and `doc_store` (both non-None). The block added by Batch C (:423, +16 lines) calls `resolve_skill_execution_pinning_wiring(kv_store=kv_store, document_store=doc_store)`.
5. `intergrax/skills/skill_execution_pinning_wiring.py:30-37`: `elif kv_store is not None or document_store is not None:` forwards both to `wire_skill_execution_pinning_store`.
6. `intergrax/skills/execution_pinning_persistence.py:489-491`: `if kv_store is not None and document_store is not None: raise ValueError(...)`.

Why config/fixture is refuted:
- The ValueError proves both objects are non-None, and both are created inside production composition code, not by the test.
- The test patches only GCA manifest/profile wiring, the LLM stub and the cursor secret.
- The three Batch C modules (wiring, persistence, durability) contain no `environ`/`getenv` reads.
- The failure is deterministic. It reproduced again in this run, and matches HIST-03's iso runs.

Contract inconsistency inside the same host runtime. Sibling wirings already accept "both" and resolve it to KV:
- `intergrax/applications/_shared/profile_resolution/wiring.py` (~:70): `profile_document_store = document_store if kv_store is None else None`. KV wins; the doc store is used only when no KV exists.
- `intergrax/applications/_shared/integrations/integration_configuration_provenance_reader.py:52-67`: tests `kv_store` first; the document store is only a fallback.
- `production_platform_persistence.py:70-90` `resolve_reference_production_strict_host_environment`: defaults providers to KV explicitly "when KV and document stores coexist".

Only the new Batch C pin wiring rejects both.

### Q2 - state at acceptance

- Accepted SHA: `8e43b160447ca43de6631fe2a1b7f99948b64f2f` (`PROD_Q_R1_R7_Q_P2_STRICT_PRODUCTION_INTEGRATED_E2E.md` L14, L147, L230, L249). Recorded as "HTTP 6/6 @ 8e43b1604", user-reported, partial log `6 passed in 16.40s`, 2026-10-15.
- `git merge-base --is-ancestor 8e43b1604 d1040f485` is true (323 commits between them).
- Later recorded green: `PROD_Q_R1_R7_Q_P3_A_CROSS_TENANT_STRICT_QUALIFICATION_EVIDENCE.md` L78: "P2-C HTTP regression ... 6/6 passed (same run)", before Batch C.
- The test file was last modified 2026-10-19 (`e1312dcae`, roster fix), also before Batch C (2026-10-23).
- Caveat: every green claim is doc-recorded or user-reported. I did not re-run at those SHAs. "6/6" refers to the HTTP tests; the file now has 7 tests including the `g00` ancestor check.

### Q3 - which commit

- `git show d1040f485` adds `skill_execution_pinning_wiring.py` (new) and `execution_pinning_persistence.py` (new, containing the "not both" guard). It also adds the 16-line block in `harness_host_runtime.py`, the only caller that passes both stores.
- Guard and caller arrive in the SAME commit, so there is no intermediate state. Red starts exactly at `d1040f485` for any strict-production path that supplies both stores.
- Later commits touching those files (963d652ff, 595eb8191, 4a466a568 Batch C remediations; 8d3c6a21d, 34777f9c7, 134f90dd1 TOOL-LIFE-X) did not remove the guard. It still raises at ba07ebaa3 (run above).
- Nothing was bisected by running tests, as required. Commits between acceptance and Batch C that touched factory/runtime (for example 495b001d2, 0bb8fcc8c, cae02d4cc, 08f1ef069) are not excluded by a run of mine. They are excluded only by the recorded green evidence (L78 above) and by the traceback pointing solely at Batch C code.

## Owner and authority recommendation (no code)

- **Owner: SKILL-X.** The defect is in the Batch C wiring contract (`wire_skill_execution_pinning_store` and `resolve_skill_execution_pinning_wiring`) and the SKILL-X-authored host runtime block. PROD-Q P2-C is the victim (SEC-04, PRD-08, PRD-09 downgraded). It should be the re-verification gate after the fix, not the owner of the change. The `research_application` factory (`factory.py:90`) uses the same persistence helper and should be included in that verification. Per memory, product-path changes under `intergrax/` interact with IIAC ACT-N11, so the ticket design must account for it.
- **Authoritative store: KV.** Reasons:
  - Strict composition's durable store is the sqlite KV. The document store there is `create_reference_in_memory_document_store()`, which is non-durable.
  - Every sibling wiring already resolves coexisting stores to KV.
  - The document store should remain only the fallback when no KV exists, and it must be a `ConditionalDocumentStore`.
  - `validate_skill_execution_pinning_durability_for_composition` guards strict-mode durability. This is an inference from its name and its place in the wiring; I did not read its body.

## Request for MASTER (I cannot create worktrees)

To close the empirical gap in Q2/Q3, create a detached worktree at `97a0771fc` (Batch C parent; ideally also at `8e43b1604`) and run the single P2-C file once under the heavy lock. Expected: HTTP tests green at the parent, red at `d1040f485`. This is optional; the static chain above is already conclusive.

## Notes

- The dossier path `docs/project/maintainers/qualification/historical_certification_campaign/dossiers/hist03/` does not exist in the repo or the main working tree. I read the HIST-03 material from `.tmp/session/hist-cert-campaign/hist03/` instead (`HIST03_ACCEPTANCE_HANDOFF.md`, `HIST03_PRODQ_RESIDUAL_QUALIFICATION.md`, `WORKLOG.md`). Its F-015 text is consistent with this report.
- Nothing in the repository was modified, committed, pushed or reset.
