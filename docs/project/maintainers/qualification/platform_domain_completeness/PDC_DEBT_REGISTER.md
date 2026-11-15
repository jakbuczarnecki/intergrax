# PDC-X — Debt register (PDC-P9-01 input) — DRAFT

> **DRAFT — proposal for the Operator, not a verdict.** This register lists LOW and INFO findings of the eight P7 audits, the debt items of `DELEGATION-PDC-WAVE2.md` decision 5, and the debt table of ticket `T-IGX-036-Z`. "Debt" means documented and not repaired in PDC-X; it is not a waiver, not an acceptance and not a rating. Nothing is closed by being listed. Declared exceptions (HIGH and MEDIUM findings and their rows) are in [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md).

| Item | Value |
| --- | --- |
| Drafted at | `origin/development` `cba32d2479cd134911d2fdf4fb1db4587c8a94d8` (2026-11-15) |
| Audited SHA of the findings | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (pre-repair) |
| Rule | `DELEGATION-PDC-P7-REPAIR.md` option B: "LOW and INFO findings go to the debt register of the P9 closure package unless a repair ticket closes them as a by-product." |
| Disposition vocabulary | **DEBT** = not repaired; **REPAIRED** = closed as a by-product by the named repair (not debt); **SPLIT** = part repaired, remainder debt; **RECORD** = INFO, positive or process note, nothing to repair |
| Verification | Dispositions are read from the signed plans' tables and the merged commits; the repairs were **not re-audited** (P8 pending). Findings repaired by a by-product are marked REPAIRED on the plan's word and need the P8 auditor's confirmation: `<P8 confirmation>`. |

Counts of source findings (audit headers): D1 7 LOW + 4 INFO; D2 10 LOW + 2 INFO; D3 6 LOW + 3 INFO; D4 7 LOW + 4 INFO; D5 6 LOW + 4 INFO; D6 9 LOW + 5 INFO; D7 7 LOW + 3 INFO; D8 9 LOW + 4 INFO. Recount against the lists below at signing: `<recount>`.

## 1. D1 AGENT-LIFECYCLE (plan `PLAN_T-IGX-041-A.md:33-34, 46-47`)

| Id | Sev | Title | Owner path | Disposition and why |
| --- | --- | --- | --- | --- |
| F-D1-09 | LOW | `upgrade_skill` restores only on `Exception` | `intergrax/skills/host_lifecycle.py` | DEBT: the skill verbs have no production caller (EX-D1-06) |
| F-D1-10 | LOW | Skill upgrade ignores pre-release tags | `intergrax/skills/host_lifecycle.py`, `intergrax/contracts/semver.py` | DEBT: same reason; `contracts/` is a hotspot |
| F-D1-12 | LOW | Gates are syntax- or name-based and miss simple bypasses; 3 of 10 mutants survive | `tests/qualification/pdc_x/d1/` | SPLIT: mutants M2, M7, M10 pinned by test (REPAIRED, plan A:33); the broader "syntax-based gates" claim is not addressed (R004/R020) |
| F-D1-13 | LOW | Upgrade admission edge cases (0.x rule, replay before authorization, leftover VERIFIED) | `semver_compat.py`, `upgrade_admission.py`, `admin_service.py` | SPLIT: leftover-VERIFIED part repaired through F-D1-06(a); rest DEBT |
| F-D1-14 | LOW | Named identity-uniqueness gate does not exist | `agent_registry.py`, `agent_contract_meta.py` | REPAIRED (test-only by-product, plan A:34) |
| F-D1-15 | LOW | Plan doc still says READY_FOR_INDEPENDENT_AUDIT | `docs/project/maintainers/plans/AGENT_DISTRIBUTION.md` | DEBT: left to the closure package so no status claim is edited before the verdict (plan A:46). **Closure-package action**, see the acceptance report |
| F-D1-16 | LOW | LLM profile pin and plugin provenance are opt-in | `admin_service.py`, `intergrax/core/plugins/admission.py`, `intergrax/core/catalog_bootstrap.py` | DEBT: strict-host default is an Operator-level profile decision (see item W-04) |
| F-D1-17 | INFO | No test for cross-host reuse | `agent_manager_command_facade.py` | RECORD, but row PDC-D1-R015 stays PARTIAL for exactly this reason (see exception register section 4) |
| F-D1-18 | INFO | Positive record | n/a | RECORD |
| F-D1-19 | INFO | Positive record | n/a | RECORD |
| F-D1-20 | INFO | One pytest process including xd_04 did not exit | `tests/qualification/pdc_x/xd` | RECORD: same class as R-05; the leak guard of T-IGX-036-Z addresses it (see item Z-1) |

## 2. D2 CONTEXT-KNOWLEDGE (plan `PLAN_T-IGX-041-B.md:44-63`)

| Id | Sev | Title | Owner path | Disposition and why |
| --- | --- | --- | --- | --- |
| F-D2-09 | LOW | Neo4j qualification test has an IndentationError | `tests/integration/rag/test_neo4j_live_qualification.py` | REPAIRED (plan B:31) |
| F-D2-10 | LOW | Stale pre-context policy wiring gate; docstring-coupled test | `intergrax/runtime/policy/pre_context_policy_audit.py`, `scripts/maintenance/check_pre_context_policy_wiring.py` | REPAIRED (plan B:32) |
| F-D2-11 | LOW | Raising reindex hook breaks retrieval; hooks are global and tenant-blind | `intergrax/rag/governance/embedding_version_policy.py:142-159` | SPLIT: isolation part REPAIRED; tenant part DEBT with EX-D2-03 (no composition) |
| F-D2-12 | LOW | Freshness identity: absolute path stamped, weak mtime+size revision | `intergrax/rag/ingest/ingest_pipeline.py`, `rag/governance/source_freshness.py` | DEBT with EX-D2-03: no observer is wired; not exploitable today (plan B:56) |
| F-D2-13 | LOW | Retention sweep: payload `as_of`, payload user, unbounded join | `intergrax/memory/retention_sweep.py:296-346` | SPLIT: clock and deadline part REPAIRED; payload-user part DEBT with EX-D2-03 |
| F-D2-14 | LOW | Tenant guard depends on `isinstance` and a private attribute | `intergrax/memory/default_memory_control_plane.py`, `memory/contracts/memory_control.py` | SPLIT: protocol-based guard REPAIRED; fail-closed-when-absent part is EX-D2-06 |
| F-D2-15 | LOW | All-blocked required source fails with an untyped `ValueError` naming the wrong cause | `intergrax/context/context_engine.py:321-334`, `context/policy/context_assembly_policy.py:40` | DEBT: fail-closed holds through a second gate; only the error class and reason code lose the egress cause (plan B:59) |
| F-D2-16 | LOW | Orphan `tool_call_id` accepted | `intergrax/context/provider_lifecycle.py`, `context/formatter.py` | DEBT: availability only; drop versus fail-closed needs an Operator decision (EX-D2-07) |
| F-D2-17 | LOW | Last-resort drop removes newest messages first | `intergrax/runtime/nexus/context/context_compiler.py:337-370` | REPAIRED (plan B:26) |
| F-D2-18 | LOW | Weak negative tests in the compaction round trip | `tests/qualification/pdc_x/d2/test_d2_compaction_roundtrip.py`, `test_d2_context_fidelity.py` | SPLIT: negatives strengthened; the "reconstruct from the durable store" assertion was **not done** (PR #575 deviation 2, absent from the ticket) |
| F-D2-19 | INFO | Matrix ownership citations do not name the carrying code | `PDC_DOMAIN_REQUIREMENTS_MATRIX.json` | DEBT: next matrix revision (plan B:62) |
| F-D2-20 | INFO | Scoped result and environment notes (basis of R008 PASS_SCOPED) | `intergrax/runtime/nexus/tools/plan_context_invocation.py` | RECORD |

## 3. D3 MODEL-RUNTIME (plan `PLAN_T-IGX-041-C.md:32, 42-43`)

| Id | Sev | Title | Owner path | Disposition and why |
| --- | --- | --- | --- | --- |
| F-D3-08 | LOW | LangChain Ollama stream silently re-issues the call after a mid-stream failure | `intergrax/llm_adapters/providers/ollama_adapter.py` | REPAIRED (by-product, plan C:29) |
| F-D3-09 | LOW | VBN classification of provider defaults missing | `tests/qualification/ebh_6/_findings_register.py` | DEBT: Option B rule; keeps row PDC-D3-R019 PARTIAL |
| F-D3-10 | LOW | Attribution records the requested model, not the served model | `intergrax/llm_adapters/contracts/adapter_response.py`, `_shared/openai_completion_mapping.py` | DEBT: "served model attribution" is out of scope (plan C:130); keeps row PDC-D3-R005 PARTIAL |
| F-D3-11 | LOW | Duplicate catalog record; loader accepts duplicates and non-positive windows | `intergrax/llm_adapters/registry/model_catalog.py`, `model_catalog.yaml` | DEBT |
| F-D3-12 | LOW | Tenant quota is one process-wide cap that disables itself on a malformed value | `intergrax/llm_adapters/governance/quota.py` | DEBT: quota module out of scope (plan C:129); contributes to row PDC-D3-R018 |
| F-D3-13 | LOW | Test strength gaps (mutants M03, M12 survive; `_declares_true` accepts any call) | `tests/qualification/pdc_x/d3/` | REPAIRED (by-product, plan C:30; M12 killed per ticket C:54) |
| F-D3-14 | INFO | Positive: F-49/F-50 resolved, shared-slug class closed, EBH-5 green | `integrations/providers/embedding_provider/openai/register.py` | RECORD |
| F-D3-15 | INFO | Environment notes and known reds | n/a | RECORD |
| F-D3-16 | INFO | Doc drift: architecture and plan docs lack the declared-capability contract | `docs/project/architecture/LLM_ADAPTERS.md` and its plan | DEBT: plan C:43 says "Doc lane of D3", but no D3 doc file is in the 041-C write set, so no ticket owns it |

## 4. D4 HOST-RUNTIME (plan `PLAN_T-IGX-041-D.md:34-58`)

| Id | Sev | Title | Owner path | Disposition and why |
| --- | --- | --- | --- | --- |
| F-D4-10 | LOW | `HostDeploymentProfile` extra=allow; wide bind accepted | `intergrax/applications/contracts/environment_profile/sub_profiles.py`, `_shared/lkw_hybrid_daemon_wiring.py` | DEBT: fail-closed direction; a bind-host validator needs a new opt-in key; reopens with the ownership decision EX-D4-05 |
| F-D4-11 | LOW | Instance guard: PID-reuse metadata rejects a legitimate start | `intergrax/hosting/instance/file_guard.py` | DEBT: fail-closed; the fix needs a process start time from the probe (protocol change) |
| F-D4-12 | LOW | Unbounded downstream publishes in the shutdown path | `intergrax/hosting/engine/engine.py`, `eventing.py`, `supervisor/supervisor.py` | REPAIRED (by-product, plan D:34) |
| F-D4-13 | LOW | Lifecycle metrics exported only with an enabled policy; bridge test covers 3 of 6 | `intergrax/hosting/runner.py`, `tests/qualification/pdc_x/d4/test_d4_lifecycle_metrics.py` | REPAIRED (test and doc only, plan D:35) |
| F-D4-14 | LOW | D4 suites and `host_01` are in no CI workflow or impact rule | `scripts/ci/run_regression_gate_ci.py`, `.github/workflows/unit-tests.yml`, `impact_map.py` | DEBT: needs a CI profile with the `mcp` extra; `.github/` is not delegated |
| F-D4-15 | LOW | Cited evidence `host_01` does not exercise R001/R002 | `PDC_DOMAIN_REQUIREMENTS_MATRIX.json` | DEBT: data correction belongs to the P9 closure package (plan D:53) |
| F-D4-16 | LOW | `run_bounded_phase` is cooperative; stubborn work overruns and is reported COMPLETED | `intergrax/hosting/shutdown.py` | DEBT: cooperative by design (`shutdown.py:131-145`) |
| F-D4-17 | INFO | Queue recovery redelivers with `idempotency_store=None` (routed F-13) | `intergrax/queueing/providers/document_store/colocated_worker.py` | RECORD: routed to the D7 owner |
| F-D4-18 | INFO | Impact-map root rule and `openai` catalog dispatch are outside D4 | n/a | RECORD |
| F-D4-19 | INFO | Environment and out-of-row red tests | `tests/unit/hosting/instance/`, `tests/unit/applications/architecture/` | SPLIT: (b) symlink tests REPAIRED; (a) environment extras and (c) out-of-row architecture gates DEBT with their owning domains |
| F-D4-20 | INFO | R010 semantics note (digest compares definition strings) | n/a | RECORD |

## 5. D5 EVALUATION (plan `PLAN_T-IGX-041-E.md:29, 44-49`)

| Id | Sev | Title | Owner path | Disposition and why |
| --- | --- | --- | --- | --- |
| F-D5-09 | LOW | Dataset asset digest optional and unenforced | `intergrax/runtime/architecture/evaluation_assets.py` | DEBT: `GoldenDatasetAsset` is not on a production path; keeps row PDC-D5-R002 PARTIAL |
| F-D5-10 | LOW | Digest hygiene (`default=str`, mutable containers, `result_digest` scope) | `intergrax/contracts/evaluation_contracts.py`, `contracts/canonical_payload_hash.py` | DEBT: a digest change re-pins every stored digest and needs coordination with the TENANT-X re-pin; hotspot |
| F-D5-11 | LOW | Evidence integrity self-attested; `$` regex accepts a trailing newline | `intergrax/runtime/evidence/eval_evidence_contracts.py`, `contracts/evaluation_contracts.py` | SPLIT: regex part REPAIRED; self-attestation DEBT (design decision: signature or anchor, D6 evidence plane) |
| F-D5-12 | LOW | Registry id oracle, id collision, lock-lease edges | `intergrax/runtime/architecture/online_evaluation_registry.py` | SPLIT: id collision and legacy `clear()` scope REPAIRED; lease owner token and existence oracle DEBT (not reproduced across processes) |
| F-D5-13 | LOW | Human review queue in-process, tenant-less, no reviewer identity | `intergrax/runtime/evaluation/human_review_sample_queue.py` | DEBT: a feature, not a repair (durable queue) |
| F-D5-14 | LOW | Safety evaluator context lacks tenant and agent; `risk_level` ignored | `intergrax/eval/evaluators.py`, `contracts/evaluation_contracts.py` | DEBT: a feature (tenant on `EvalSubject`) |
| F-D5-15 | INFO | `Co-Authored-By` trailer in a B-D5 commit | history | RECORD: predates the no-attribution rule; history is not rewritten |
| F-D5-16 | INFO | Stale pointers (matrix cites a nonexistent CRITIC_VERIFICATION path) | matrix, docs | DEBT: no ticket fixes them; the matrix row text is unchanged at HEAD |
| F-D5-17 | INFO | Judge isolation is by convention | `intergrax/runtime/architecture/evaluation_automation.py` | RECORD |
| F-D5-18 | INFO | Positives and non-reproductions | n/a | RECORD |

## 6. D6 RUNTIME-SECURITY (plan `PLAN_T-IGX-041-F.md:25, 35-36`)

| Id | Sev | Title | Owner path | Disposition and why |
| --- | --- | --- | --- | --- |
| F-D6-03 | LOW | Audit-trail list returns the oldest 1000 only; secret-shaped values stored | `intergrax/runtime/security/security_audit_trail.py` | DEBT: Option B |
| F-D6-08 | LOW | Tool-input guard is a four-regex heuristic; empty `tool_id` allowed | `intergrax/runtime/security/tool_injection_guard.py` | DEBT |
| F-D6-09 | LOW | Poisoning default differs by path | `tools/providers/rag/service.py`, `plan_context_invocation` | DEBT |
| F-D6-10 | LOW | Strict-config validator accepts weak or disabled settings | `intergrax/applications/_shared/security_assembly_resolver.py` | DEBT: hotspot; keeps row PDC-D6-R024 PARTIAL |
| F-D6-11 | LOW | Tenant-less resource open to any scoped tenant; identity not normalised | `intergrax/runtime/security/tenant_scope.py`, `contracts/agent_run.py` | DEBT: hotspot |
| F-D6-12 | LOW | Credential resolver keeps provider exception text as `__cause__` | `intergrax/integrations/credentials/secrets_store_resolver.py` | REPAIRED (by-product, plan F:23) |
| F-D6-13 | LOW | Security-defense plugin admission defaults to structural-only and `isolate` | `intergrax/core/plugins/`, `catalog_bootstrap` | DEBT (see W-04) |
| F-D6-14 | INFO | `raw_secret_gate` evasions | `intergrax/core/security/secret_safety.py` | DEBT/RECORD: heuristic gate disclosure; keeps row PDC-D6-R007 PARTIAL |
| F-D6-15 | INFO | Lab profiles use the public demo seed implicitly | `intergrax/runtime/execution_evidence/attestor.py` | RECORD: accepted lab path |
| F-D6-16 | LOW | Satellite wording stale or imprecise (also the `security_audit_trail_wiring.py` docstring) | `docs/.../EXECUTION_ENGINE_SECURITY_BOUNDARY*` | DEBT |
| F-D6-17 | INFO | Forbidden `Co-Authored-By` in B-D6 commits | history | RECORD |
| F-D6-18 | INFO | `prod_q` pytest process does not exit after the summary | `tests/qualification/prod_q` | RECORD: same class as R-05 (item Z-1), outside the `pdc_x` guard |
| F-D6-19 | INFO | MCP security not exercised (R017 blocked, `fastmcp` absent) | `intergrax/applications/_shared/mcp_application_server.py` | RECORD with EX-D6-04 |
| F-D6-20 | LOW | R015/R019 proven for credentials, allow-list and three XD pairs, not for model-provider tenant data | `intergrax/runtime/security/`, `tests/qualification/pdc_x/xd` | DEBT: keeps row PDC-D6-R015 PARTIAL |

## 7. D7 BACKGROUND-SCHED (plan `PLAN_T-IGX-041-G.md:35, 44-45`)

| Id | Sev | Title | Owner path | Disposition and why |
| --- | --- | --- | --- | --- |
| F-D7-14 | LOW | Consumers do not treat DEAD_LETTERED uniformly; terminal duplicate acked SUCCEEDED | `applications/local_workspace_application/serving/background_task_proof_routes.py`, `colocated_worker.py`, `worker_runtime.py` | DEBT |
| F-D7-15 | LOW | `RetryPolicy.max_retry_window_seconds` TypeError with `max_backoff_seconds=None` | `intergrax/queueing/worker/retry_policy.py` | DEBT: keeps row PDC-D7-R009 from a clean rating |
| F-D7-16 | LOW | Ownership record names the wrong composition root; registries diverge on duplicates | `PDC_OWNERSHIP_MATRIX.md`, `intergrax/background_tasks/registry.py` | DEBT: the only finding on row PDC-D7-R022, which therefore stays PARTIAL; a small documentation fix (see exception register section 4) |
| F-D7-17 | LOW | Retry/dead-letter silent; idempotency key pins a dead-lettered task; no purge | `intergrax/queueing/providers/document_store/`, `background_tasks/events.py` | DEBT: keeps row PDC-D7-R013 from a clean rating |
| F-D7-18 | LOW | Unsatisfiable cron and overflowing intervals accepted | `intergrax/long_running/recurrence.py` | REPAIRED (by-product of S5; `recurrence.py:93-95, 242, 288` per extractor) |
| F-D7-19 | LOW | `ScheduledResumePersistence.cancel` not tenant-scoped | `intergrax/long_running/scheduled_resume.py`, `store.py` | DEBT: port-signature change used by fakes of cited suites |
| F-D7-20 | LOW | `DocumentStoreTaskWorker.stop()` abandons the running thread | `intergrax/queueing/providers/document_store/colocated_worker.py` | DEBT: belongs with F-D7-03 (EX-D7-01) |
| F-D7-21 | INFO | DST gap docstring versus code | `intergrax/long_running/recurrence.py` | REPAIRED (by-product, `recurrence.py:19`) |
| F-D7-22 | INFO | Weak or missing negative tests | `tests/qualification/pdc_x/d7/` | REPAIRED "by the tests of this plan" (plan G:35); not a separate deliverable |
| F-D7-23 | INFO | Re-verification of routed items | n/a | RECORD |

## 8. D8 ARTIFACT-LIFECYCLE (plan `PLAN_T-IGX-041-H.md:30-33, 47-52`)

| Id | Sev | Title | Owner path | Disposition and why |
| --- | --- | --- | --- | --- |
| F-D8-08 | LOW | `_guard` ignores the authorization decision value | `intergrax/runtime/nexus/artifacts/governed_access.py` | REPAIRED (by-product, plan H:30) |
| F-D8-09 | LOW | Sweep robustness (starvation, poisoned row, naive datetimes, negative limit) | `retention.py`, both artifact stores, `models.py` | SPLIT: negative limit REPAIRED; protected-row starvation, poisoned row, naive datetimes DEBT (touches the query shape of both stores) |
| F-D8-10 | LOW | v1 retention class typo silently becomes the default | `intergrax/contracts/artifact_ref.py` | DEBT: hotspot file, not needed by any in-scope fix |
| F-D8-11 | LOW | Digest validators accept a trailing newline | `intergrax/contracts/artifact_ref.py`, `contracts/task_artifacts.py` | DEBT: hotspot |
| F-D8-12 | LOW | Backend failures leave no outcome event | `governed_access.py` | REPAIRED (by-product, plan H:30) |
| F-D8-13 | LOW | `reconstruct_artifact_lineage` does not check ancestor tenant | `intergrax/contracts/artifact_ref.py` | DEBT: hotspot |
| F-D8-14 | LOW | Legal hold overwritable and lossy | `sqlite_artifact_store.py`, `in_memory_artifact_store.py` | DEBT: needs a schema change of the holds table |
| F-D8-15 | LOW | Bundle builder follows symlinks; empty sha256 on entry types | `intergrax/orchestration/run_artifact_bundle_builder.py`, `contracts/task_artifacts.py` | DEBT: code-read only (the host cannot create symlinks); hotspot |
| F-D8-16 | LOW | No hostile-source or composition test | `tests/qualification/pdc_x/d8/test_d8_governed_port.py`, `p6_repair/test_p6r_d8_worker_bootstrap.py` | SPLIT: hostile-source test REPAIRED; composition test DEBT, tied to EX-D8-01 |
| F-D8-17 | INFO | Documentation and ownership inconsistencies, residual alias | `docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md`, `emitter.py`, `taxonomy.py` | SPLIT: documentation part REPAIRED; matrix wording, emitter annotation, taxonomy note DEBT (no action for the verdict) |
| F-D8-18 | INFO | B-D8 commit carries an AI-attribution trailer | history | RECORD |
| F-D8-19 | INFO | Adjacent partial-results test red | `tests/unit/runtime/long_running/test_partial_results.py` | RECORD: no action (plan H:52) |
| F-D8-20 | INFO | Known reds not reproduced in D8 runs | n/a | RECORD |

## 9. Debt items of `DELEGATION-PDC-WAVE2.md` decision 5

Decision 5: "One omnibus ticket (see scope); the rest recorded as documented debt." The omnibus ticket is `T-IGX-036-Z` (PR #549, merge `622a61b962ee9941e4d8d4e4823bdd54063a3e87`, batch B-P6-I, plan digest `da824e28...`).

| Id | Item | State | Owner path | Why debt / reopen |
| --- | --- | --- | --- | --- |
| W-01 | `pdc_x` pytest process that does not exit after its summary line (finding R-05) | **Repaired by `T-IGX-036-Z`** (commit `df436b7d1`: module-scoped leak guard in `tests/qualification/pdc_x/conftest.py`, closing the leaked boundary in `xd/test_xd_03_agent_governance.py`, exit test `p6_debt/test_p6debt_thread_exit.py`). **The ticket has no implementation checkpoint** (`T-IGX-036-Z.md` ends at `plan_approved`) so the whole-suite exit proof (Deliverable 6, once, `timeout 1500`) is **UNVERIFIED**. The D1 audit (F-D1-20) and D6 audit (F-D6-18, `prod_q`) still observed non-exiting processes before the repair. | `tests/qualification/pdc_x/` | Verify at the global audit SHA: `<exit proof result>` |
| W-02 | Tests red on the baseline for environment reasons (missing `INTERGRAX_PGVECTOR_TENANT_ID`, missing `sentence_transformers`, stale persisted-trace and `TaskResult` fixtures) | **Repaired by `T-IGX-036-Z`** (test-side only; HF test skips with its reason when the extra is missing). Same checkpoint caveat as W-01. | `tests/unit/integrations/providers/`, `tests/unit/rag/embedding/`, `tests/integration/runtime/nexus/tracing/`, `applications/local_workspace_application/tests/` | Verify: `<result>` |
| W-03 | Unused `embedding_provider/ollama/manifest.py` | **Repaired by `T-IGX-036-Z`** (file deleted; diff stat of `df436b7d1` shows `manifest.py | 17 ----`). | `intergrax/integrations/providers/embedding_provider/ollama/` | Stop rule: if the IIAC inventory matrix obliged the file, the deletion would have been reverted; no revert is visible in the diff stat. |
| W-04 | `compat_x` counters (item Z-D1: 18 R3-R1 violations and an unexpected-shim R5 gate) | **Planned for ticket `T-IGX-041-L`** (`DELEGATION-PDC-P7-REPIN.md` (a): re-pin at the last product change without weakening a gate). **No ticket file `T-IGX-041-J/K/L` exists on `development` at drafting: state UNVERIFIED.** The D6 ticket recorded 7 `r3_r1_gates` reds as "the known COMPAT-X counter drift class", also red at the pin. Cause recorded by Z: counters pinned to the accepted COMPAT-X lineage (7356 raw candidates, replay fingerprint at `05549f9acb`) no longer match; B-D6's `security_audit_trail.py` adds an unclassified event writer and persistence append. | `tests/qualification/compat_x/`, inventories of the closed COMPAT-X stage | Reopen: after `T-IGX-041-L` is merged and its re-pin recorded: `<result>` |
| W-05 | Celery, Kafka and RabbitMQ identity and dead-letter mapping | Documented debt, **no ticket** (decision 5; `T-IGX-036-Z` out of scope). Related finding F-D7-09 (identity adoption) is EX-D7-03. | `intergrax/integrations/providers/message_bus/{celery,kafka,rabbitmq}`, `intergrax/queueing/providers/rabbitmq/` | Reopen: a message-bus identity/DLQ ticket; C10/C15 limits live-broker proof |
| W-06 | Strict-host `plugin_provenance` | Documented debt, **no ticket**. Related: F-D6-05 / EX-D6-03, F-D1-16, F-D6-13, `OD-PDC-SCOPE-07` (implement; plugin provenance binding done in part by T-IGX-041-F). | `intergrax/core/plugins/admission.py`, `intergrax/applications/_shared/security_assembly_resolver.py` | Reopen: strict-profile decision and installed-file digest (EX-D6-03) |
| W-07 | `ent_audit_x` `launcher_05` FREEZE timeout | Documented debt, **no ticket**. Test: `tests/qualification/ent_audit_x/integration/test_audit_launchers_cross_platform.py` (exact test id UNVERIFIED). | `tests/qualification/ent_audit_x/` | Reopen: a launcher timeout ticket; listed among the known red classes of the P6 acceptance record (not run there) |
| W-08 | `roadmap_replay_x` dispositions | Owned by MCP-SERVER-X P7, see EX-G-05 (UNVERIFIED whether resolved). | `tests/qualification/roadmap_replay_x/` | See EX-G-05 |

## 10. Debt table of `T-IGX-036-Z` (plan `PLAN_T-IGX-036-Z.md`, "Documented debt")

| Id | Red or debt | Reproduced cause (from the plan) | Owner path | Why not repaired |
| --- | --- | --- | --- | --- |
| Z-D1 | COMPAT-X counters: `test_cx_r3_r1_combined_gate_passes_on_current_head`, `test_r5_p0_closed_world_gate_passes_on_current_head` | See W-04 | `tests/qualification/compat_x/` | Re-pinning changes accepted facts of a closed stage; `mcp_host_mount.py` is under the `_shared` hotspot; counters move with every later product repair, so re-pin once after the last product change (now planned as T-IGX-041-L) |
| Z-D2 | `test_document_store_vendor_category_cutover::test_vendor_sdk_imports_stay_inside_provider_packages` | `intergrax.integrations.providers.document_store.mongodb.client` imports `pymongo` outside `opens` | `intergrax/integrations/providers/document_store/mongodb/` | Product change in a provider the batch does not own, or a gate-meaning change |
| Z-D3 | `test_ebh_2d_d_application_contract_surface_recertification::test_whole_surface_forbids_contract_to_implementation_edges` | `intergrax/applications/contracts/environment_profile/root.py` imports `intergrax.applications._shared.strict_host_dependency_concurrency_admission` | `intergrax/applications/_shared/` | `_shared` hotspot (batch B-D4 owner) |
| Z-D4 | `test_dep1...imports_provider_only_on_selection` and the two registry-first strict xfail cases of `p6_repair` | Second import cycle `observability_otlp -> export_boundary -> runtime.events.store -> sqlite.register -> plugin_register`; passes in isolation (order-dependent catalog state); cheap fix named in the plan (remove two redundant module-level imports in `intergrax/runtime/observability/export_boundary.py`) | `intergrax/runtime/observability/export_boundary.py` | A second product path and strict xfail flips; offered as a one-line follow-up on Operator request |
| Z-D5 | trace_x model-call surface gates (4), `test_audit_ideal_28_3_lkw_hybrid_daemon`, a `runtime_state_event_bus` case, the `d4` LKW stub path | Reported as pre-existing at `f8ed234f8` (PR 517); registry or product-path mismatches (`stream_with_tools`, `scripts/lkw-host.py` versus `scripts/maintenance/lkw-host.py`) | `tests/qualification/trace_x/` and others | Product defects or gate registries; not re-reproduced by the plan |
| Z-D6 | `duckdb` cases of ECAP-F3, `anthropic`, `langchain_core`, `fastmcp`, `celery` skips | Optional extras not synced in the venv (the `dev-ci` sync passes them) | environment | No test change; the extra is the fix (also explains the D2 RAG skips and EX-D6-04) |
| Z-D7 | Celery, Kafka, RabbitMQ identity and DLQ; strict-host `plugin_provenance`; `ent_audit_x` `launcher_05` FREEZE timeout | Operator decision 5 | see W-05..W-07 | Documented debt without a ticket |

## 11. Other known reds and process debt seen in the sources (not findings)

| Id | Item | Source | Note |
| --- | --- | --- | --- |
| K-01 | IIAC activation staleness (ACT-N11) after product changes | `DELEGATION-PDC-CLOSE.md` item 3 | Known red class until the P8-04 rebind: `<state after P8-04>` |
| K-02 | EBH-7 identity gate until the final re-pin is recorded | `DELEGATION-PDC-P7-REPIN.md` | Final re-pin ticket J/K/L not on `development` yet |
| K-03 | TENANT-X carrier count (F-D2-03, F-D3-06, F-D6-06) | `T-IGX-041-I`, PR #576 | Re-pinned by classification; whole-gate green at the audited SHA UNVERIFIED |
| K-04 | 27 `test_agent_distribution_materialization*.py` reds from the worktree MAX_PATH, 8 classified D2 RAG reds, about 99 incident-scenario reds | PR bodies #574, #575; ticket C | Environment or exception class; see EX-D3-04 and the D2 gap |
| K-05 | Pyright errors added or pre-existing in touched files (D6 sandbox test 10 to 12; D3 adapters about 204 pre-existing) | tickets F, C | Not a gate of PDC-X |
| K-06 | Held dependency-bot / `uv.lock` pull requests | `DELEGATION-PDC-WAVE2.md` decision 7 | Merged in one batch after the final re-pin: `<batch state>` |
