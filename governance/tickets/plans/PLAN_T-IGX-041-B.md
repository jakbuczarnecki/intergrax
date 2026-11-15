# PLAN T-IGX-041-B — PDC-X batch B-P7-D2: repair of the IA-D2 audit findings for domain D2 (CONTEXT-KNOWLEDGE), repair loop 1 of 2

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P7-D2**, wave W13). Origin: independent audit `PDC_P7_D2_AUDIT_REPORT.md` (IA-D2, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, overall verdict PARTIAL: 0 HIGH, 8 MEDIUM, 10 LOW, 2 INFO). Authority for the scope: `operator_decisions/DELEGATION-PDC-P7-REPAIR.md` (Operator option B, 2026-11-15): real defects in existing code (bugs, tenant-scope gaps, fail-open paths, missing validation, product bugs behind red cited suites, stale or red cited tests, overstating documents) and test hygiene are repaired; findings of the kind "component exists but has no production caller or composition" (F-04 family) and "requirement claims a feature that has no implementation" are not repaired in PDC-X but declared as exceptions with a reopen condition (P9 exception register). Gates are only strengthened, classified or re-pinned, never weakened or deleted.

**This plan is not a certification verdict.** It certifies nothing, promotes no FRZ criterion, changes no requirement row and issues no `Approved:` line; a separate delegate session signs and a fresh IA-D2 session re-audits the domain afterwards (loop 1 of 2). The batch changes product paths (`intergrax/runtime/nexus/context/`, `intergrax/context/budget/`, `intergrax/memory/`, `intergrax/rag/`, `intergrax/runtime/policy/`). Consequences accepted by the Operator in the merge gate of `DELEGATION-PDC-P7-REPAIR.md` (known red classes) and in decision 6 of `DELEGATION-PDC-WAVE2.md` (product paths changed by repairs are re-pinned once after all P7 repairs): the IIAC activation (ACT-N11) stays stale after product changes until the P8-04 rebind, and the EBH-7 executable-identity gate stays red until the final re-pin of the campaign.

Every file:line reference and factual sentence below was re-checked against the code of this branch (base `f87df3b2a`, the merge of PR 567); the referenced files are unchanged on `origin/development` at `aa3b3e189`, so the references hold at both; the audit report is a claim to check, not an authority, and a claim that this plan could not re-check itself is labelled "audit". Where this plan disagrees with the audit or the orchestrator triage it says so.

## Problem

The B-D2 repairs hold where they were aimed (the auditor killed nine runtime mutants of the D2 guards). Re-reading the code confirms the following real defects in code that already runs, and a red evidence base:

1. **The hard model-window budget counts only `message.content` (F-D2-01).** The enforcement chain sizes a message as `count_tokens(message.content or "")` at: `classify_candidates` (`context_compiler.py:68-70`); `ContextCompiler._enforce_hard_budget` (`total` at `:282-283`, the last-user check at `:295`); `_hard_trim_messages` (`degradation_ladder.py:86`) and the `TOKENIZER_HARD_TRIM` accounting (`:265`); the three mandatory-reserve estimators `estimate_mandatory_provider_message_tokens`, `estimate_mandatory_protected_message_tokens` and `estimate_mandatory_base_message_tokens` (`context/budget/mandatory_base_messages.py:172,191,208`; the last one also feeds `estimate_mandatory_reserve_tokens` in `mandatory_reserve.py`); `CharEstimateContextTokenCounter.count_messages` (`context/budget/token_counter.py:35`, no caller today); and the content-only loop `count_message_tokens` (`context_preflight.py:28-32`). That loop matters: the engine's final window check passes `count_tokens=active_compiler.count_tokens` (`context_engine.py:548-553`), which selects the loop, so the D3 estimator change alone does not reach it (only `compile_service.py:39` and `context_validator.py:32` take the adapter path, `context_preflight.py:27`). An assistant message with empty content and a 200,000-character `tool_calls[].function.arguments` costs 0 on every one of these paths, so neither the ladder, the hard trim nor the final preflight ever sees it (audit probe P5: `total_tokens=4`, nothing trimmed). Content-only truncation cannot shrink such a payload, so the hard trim must also drop whole tool-call groups (or fail closed for mandatory ones) when the non-content payload alone does not fit. The planning ledger (`context/planner.py`, `count_tokens(message.content)` at `:165-449`, `:585`, `:620`) is also content-only; it feeds the plan record, not the enforcement, and is excluded (table below).
2. **The last-resort hard budget drops the newest messages first (F-D2-17).** `_enforce_hard_budget` iterates `range(len(working) - 1, -1, -1)` (`context_compiler.py:345`) and keeps the stalest history; it contradicts the oldest-first ladder step `TRUNCATE_OLDEST_HISTORY`.
3. **TASK scope of the memory control plane is tenant-blind and ungoverned (F-D2-02, F-D2-14).** `_remember_task` and `_forget_task` (`default_memory_control_plane.py:846-887`) are reached after `_assert_scope_authorized` only (`remember` at `:394`, dispatch to `_remember_task` at `:399`; `forget` at `:462`, dispatch to `_forget_task` at `:467`); they call `task_memory.write/delete(namespace, key)` with no tenant, identity or governance evaluation. USER scope runs `_assert_identity_tenant_is_bound_tenant` (def `:379-386`, called at `:396`, `:410`, `:425`, `:464`, `:479`, never on a TASK branch) and `_enforce_governance` (`:554` remember, `:767` forget). `TaskMemoryCapability` (`contracts/memory_control.py:295`) has no tenant. Two tenants sharing one capability overwrite and delete each other's `(namespace, key)` (audit probe P3). The bound-tenant guard checks only `isinstance(capability, UserProfileManagerMemoryCapability)` and reads the wrapper's `bound_tenant_id` property (`:213-217`); a replacement capability is unguarded. This is latent: `build_default_memory_control_plane` accepts `task_memory` (`memory_control_wiring.py:35,56`) but no caller supplies it (`memory_wiring.py:619-622` passes `user_profile_manager` only; no `task_memory=` argument to it exists in `intergrax/`, `agents/` or `applications/`), so TASK scope is reachable only from a host that composes its own capability. It is repaired as a tenant-scope gap in existing code (option B), not as composition.
4. **A raising reindex hook breaks retrieval (F-D2-11).** `_notify_reindex_needed` (`embedding_version_policy.py:142-159`) calls every registered hook in a bare loop (`:157-158`). It is reached from `assess_retrieval_freshness` (`source_freshness.py:241-251`, `request_reindex` defaults to `True`), which `RetrievalService._assess_freshness` calls at `retrieval_service.py:253` (invoked from `:226`). Only the observer call is guarded (`:246-250`); the assessment, and with it every hook, is not, so a hook exception turns a successful retrieval into an exception (audit probe P4). The same function is also called from the embedding-version evaluation (`embedding_version_policy.py:81,91`). Isolating the loop in `_notify_reindex_needed` fixes every caller.
5. **The retention sweep task trusts the producer's clock and blocks without a bound (F-D2-13).** `retention_sweep.py:337` takes `as_of` from the payload (a far-future value expires every entry with a validity or policy window; forget is irreversible); `:288-290` start and join the worker thread without a timeout, so cancellation and deadlines cannot interrupt it.
6. **The D2-cited RAG evidence is red (F-D2-08, F-D2-09, F-D2-10).** Audit run 5: 48 failures and 13 collection errors across 35 test files (audit; this plan does not re-run the suites, the implementer's first step does). What was re-checked here: `IngestPipeline._list_current_source_ids` (`ingest_pipeline.py:585`) calls `list_source_record_ids(..., root_document_id=...)` at `:596`; the current `BaseVectorstoreManager.list_source_record_ids` (`vectorstore_manager.py:401`) has that parameter (added by `ae14377c9`, 2026-08-13, after `9dd2a233a`), and doubles such as `_NoopVectorstore.list_source_record_ids(self, *, source_id, scope)` (`tests/unit/rag/ingest/test_ingest_lci2b_loader_boundary.py:54`) do not: a stale double, not a product defect. The default `uv run` environment lacks the optional extras of the existing CI profile (`rag-guard.yml:36`: `uv sync --extra dev-ci --extra dev-ci-rag`). `tests/integration/rag/test_neo4j_live_qualification.py` has an `IndentationError` (audit, line 613) and is lost from every run. The static pre-context policy gate scans `intergrax/agents/uaep.py` (`pre_context_policy_audit.py:19`), which was relocated by `42cbe7e1a` (2026-09-20, "relocate agent engine and UAEP into runtime nexus"), so the gate is red since that date and `test_ideal_w2_w2_script_gates` stops at the first failing script (its `assert` is inside the loop); `test_typed_sources_documented_on_provider_context` asserts a docstring phrase of `context/contracts.py` that is gone.
7. **Weak negative tests (F-D2-18).** `test_d2_compaction_roundtrip.py` (`except ValueError: return`, `except Exception: return`) and `test_d2_context_fidelity.py` (accepts "error or intact") can pass for the wrong reason.

## In-scope findings

| Finding | Sev | Rows | Repair |
| --- | --- | --- | --- |
| F-D2-01 (compiler, ladder and mandatory-reserve half) | MEDIUM | R010, R011 | One per-message cost function (new `context/budget/message_token_cost.py`, below the runtime layer so the estimators can use it) counts content plus serialized `tool_calls`, `name` and `tool_call_id`; every enforcement size site of Problem 1 uses it, including the final preflight loop; hard trim drops whole tool-call groups when the non-content payload cannot fit; mandatory messages that cannot fit fail closed (`ContextBudgetUnsatisfiableError`); the planning ledger is excluded |
| F-D2-17 | LOW | R011, R010 | Last-resort drop is oldest-first among non-mandatory units (tool-call groups still atomic) |
| F-D2-02 + F-D2-14 (bounded) | MEDIUM / LOW | R002, R003 | TASK namespace is tenant-qualified by the plane (injective encoding); TASK remember/forget run the same bound-tenant guard and a governance evaluation as USER scope; the guard becomes protocol-based (any capability that exposes `bound_tenant_id`) |
| F-D2-11 (isolation half) | LOW | R005 | Each reindex hook runs isolated in `_notify_reindex_needed` (failure logged, never raised; the function reports `False` when no hook accepted the request); retrieval does not fail because of a hook |
| F-D2-13 (clock and join half) | LOW | R015 | Payload `as_of_iso` is accepted only up to `clock() + skew`; the sweep runs under `asyncio.wait_for` with a deadline and the worker thread join is bounded |
| F-D2-08 | MEDIUM | R004 | First step: reproduce and classify every red id of audit run 5 into stale double / contract drift, removed API, missing extra, product bug; repair stale doubles, port or delete tests of removed APIs (deletion only with the removal commit named), mark env-dependent modules with an explicit `importorskip` reason and run them in the CI RAG profile; real product bugs found are fixed in scope |
| F-D2-09 | LOW | R004 | Fix the `IndentationError`; add a syntax gate over the D2-cited test trees so a syntax error cannot hide behind a skip marker |
| F-D2-10 | LOW | R006, R016 | Point the pre-context audit at the relocated roots (derived from the repository layout, not hard-coded), make `test_ideal_w2_w2_script_gates` report every failing script, replace the docstring assertion with a typed check |
| F-D2-18 | LOW | R012, R007 | Negative tests assert the specific exception type or typed refusal; reconstruction asserted against the durable store |

### Dependency on the D3 ticket (T-IGX-041-C, PR 569, F-D2-01 estimator half)

The adapter estimator (`BaseLLMAdapter.estimate_tokens_for_messages`, `llm_adapters/base/base_llm_adapter.py:632`; it joins only `m.content`, `:650-657`) is owned by T-IGX-041-C. Its PR 569 diff appends `json.dumps(m.tool_calls, sort_keys=True, default=str)` (with a `str()` fallback on `TypeError`/`ValueError`), `m.name` and `m.tool_call_id` to the counted parts. **This ticket does not touch `intergrax/llm_adapters/`** (the file is named here as a reference only, never in the write set). The compiler path of this ticket does not call the adapter estimator at runtime (`ContextEngine` builds `ContextCompiler(count_tokens=strategy_snapshot.token_counter.count_text, ...)`, `context_engine.py:347-351`; `ContextCompiler()` elsewhere defaults to a char estimate), so the two halves are code-independent. The dependency is the shared serialization recipe and the window check:

- `compile_service.py:39`, `context_validator.py:32` and `engine_history_layer.py:113` size messages through `adapter.count_messages_tokens` and become `tool_calls`-aware through the D3 change alone; no D2 edit is needed there.
- The engine's own final window check does **not** use the adapter: `context_engine.py:548-553` passes `count_tokens=active_compiler.count_tokens`, which selects the content-only loop of `count_message_tokens` (`context_preflight.py:28-32`). That loop is repaired by this ticket (write set), otherwise the D3 change leaves the main path content-only.
- The per-message cost function of this ticket uses the same recipe as the D3 estimator (same JSON options and fallback, plus `name` and `tool_call_id`). A parity test pins that for a 200,000-character `tool_calls` message the compiler cost and the adapter estimate are of the same order (ratio between 0.5 and 2 with the char fallback); it needs the D3 change.
- **Order of merge: D3 (T-IGX-041-C, PR 569) first, then this ticket.** The implementer rebases onto the merged D3 change before opening the PR and runs the parity test then. If D3 is not merged when the implementer is ready, the PR is held; the parity test is not skipped or deleted to bypass the order.

## Excluded findings

Declared exceptions (not repaired in PDC-X); the orchestrator records them in the P9 exception register.

| Id | Reason | Reopen condition |
| --- | --- | --- |
| F-D2-03 | TENANT-X carrier re-pin (`tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py`, 831 != 819, audit figures) belongs to T-IGX-041-I, the campaign's single re-pin; this ticket's new tenant-carrying fields are listed to that ticket in the hand-over (the TASK namespace helper adds no class-field carrier; the retention sweep fields are not touched) | T-IGX-041-I is planned; the implementer reports any carrier this ticket adds |
| F-D2-04 | Not a bounded formatter fix. `DefaultContextFormatter` emits `role=system` with the `[context:<source>:<id>]` prefix, and that contract is consumed by `llm/messages.py:373` (compatibility-shape check of the assembled window, requires system messages to start with `[context:`), `runtime/nexus/context/graph_assembly.py:68` (compatibility projection of `[context:` system messages) and 12 test modules (grep for `[context:` under `tests/`). `ContextFragmentSource` (`contracts.py:39-55`) is a plain origin enum without an untrusted/trusted classification; fragments carry a provider-assigned `authority_class` (default `UNASSIGNED`) and a numeric `trust_score` (`contracts.py:286-287`) that the formatter does not read, so "untrusted source" would be a new concept; a non-system role also changes provider message ordering rules for every host. This is a design change with an Operator-visible model-facing effect, not a repair of existing logic | Operator decision on the trust-class model (which sources are untrusted, which role or envelope the model sees) and a provider-compatibility check; then a ticket that extends the contract and migrates the consumers |
| F-D2-05 | Missing feature versus claim: no classifier, no redaction path, no producer of fragment sensitivity (grep for `sensitivity=` in `intergrax/` finds only copies of an existing value, `provider_lifecycle.py:325`, artifact references and a calendar field); the egress ceiling defaults to `INTERNAL` (`policy/egress.py:23`) and the engine never passes one (`context_engine.py:331-333`). Needs an Operator-level design decision (classification source, default for unclassified fragments, redaction) | Operator decision on classification source and default; then a feature ticket |
| F-D2-06 | F-04 family: sweep, freshness observer and reindex admission exist and are mutation-checked but no shipped composition registers them (`memory_wiring.py:619-622` is a hotspot file under `intergrax/applications/_shared/`); composing them is new composition, not repair | The Operator accepts the composition as a feature (host or worker root registers the sweep task and a recurrence, the RAG composition root passes `source_observer`); a composition test that fails when the task is not registered |
| F-D2-07 | Missing feature versus claim: `CitationChainLink` has no consumer and no enforcing code; the audit offers two alternatives (bind to the assembled-context ledger, or reclassify R016 as RAG-citation-only) and the choice is a requirement decision, not a code repair | Operator picks bind or reclassify; then a feature or a matrix revision |
| F-D2-11 (tenant part) | `ReindexQueueRequest` has no tenant or scope and the hook list is a process-global; per-instance, tenant-attributed hooks are a new capability on an uncomposed component (F-D2-06) | With F-D2-06 |
| F-D2-12 | The stamped id (`ingest_pipeline.py:273,279-284`: absolute `source_path=str(path)`, callback metadata wins over the stamp) and the `<mtime_ns>-<size>` revision (`source_freshness.py:130-141`) are coupled to `observe_file_source` and to the re-ingest generation logic; changing the identifier format needs a decision and has no composition that reads it (no observer is wired); not exploitable today | With F-D2-06 (freshness observer composition) |
| F-D2-13 (payload user) | Restricting the payload user to the admitted principal or a service identity needs an identity design for the (uncomposed) sweep | With F-D2-06 |
| F-D2-14 (fail-closed when the capability has no `bound_tenant_id`) | Needs a strict-profile notion in the plane that does not exist; the repair here is the protocol-based guard for capabilities that do expose the tenant | Operator decision on strict profiles for memory capabilities |
| F-D2-15 | Fail-closed already holds (second gate raises); only the error class and reason code lose the egress cause. Diagnostic quality, no unsafe path | Retry or diagnosis classification needs the typed cause |
| F-D2-16 | Availability only (a provider can supply an orphan `tool_call_id`; not an instruction-promotion path). Drop versus fail-closed is a behavior decision that interacts with the ACP iterative path, whose base messages are produced by the caller | Operator decision drop versus fail closed |
| F-D2-01 (planning ledger) | `context/planner.py` sizes fragments and turns for the plan record with `count_tokens(message.content)` (`:165-449`, `:585`, `:620`, about 15 sites); the plan estimate does not enforce the window (the compiler and the preflight do) and changing it moves every plan-ledger figure of the ce_02 and planner suites | The Operator wants ledger fidelity for tool-call payloads; then a planner ticket with the same cost function |
| F-D2-19 | INFO, documentation of ownership citations in `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`; the matrix is revised by the orchestrator, not by a product ticket | Next matrix revision |
| F-D2-20 | INFO, no repair suggested | none |

## Deliverables

### 1. Pinned regression tests (first, red on the unchanged code for the right reason, each with negatives)

New directory `tests/qualification/pdc_x/d2_repair/`. Every test names the finding id in its docstring.

1. `test_d2r_message_budget_tool_calls.py` (F-D2-01, F-D2-17).
   - Positive: an assistant message with empty content and a 200,000-character `tool_calls[].function.arguments`, compiled with `input_budget_tokens=200`, is trimmed or dropped (`trimmed=True`, `total_tokens` includes the payload); today it returns `total_tokens=4`, all messages kept (audit probe P5; reproduced red first).
   - Final window check: `verify_context_preflight(..., count_tokens=<char counter>)` and `count_message_tokens` raise or count above the window for the same message (today they count 0).
   - The per-message cost counts `name` and `tool_call_id`; a message without them costs exactly what it cost before (frozen table of inputs and expected counts from the unchanged code is committed first).
   - Negatives: a `tool_calls` group whose payload alone exceeds the budget is dropped with its tool results atomically (no orphan call or result, linkage preserved); the same payload on a mandatory message raises `ContextBudgetUnsatisfiableError` (fail closed); a non-serializable `tool_calls` value does not raise (falls back to `str`); `mandatory_reserve` counts the payload of a mandatory assistant tool-call message; deterministic cost (two runs equal).
   - Ordering (F-D2-17): six history messages with budget 130 keep the newest history and drop the oldest first (today it keeps `h0,h1` and drops `h2..h5`); mandatory and final-user messages are never dropped; tool-call groups still leave atomically.
   - Parity: for a 200,000-character `tool_calls` message the compiler cost and `estimate_tokens_for_messages` of a minimal adapter are of the same order (ratio between 0.5 and 2; runs after the D3 change; see "Dependency on the D3 ticket").
2. `test_d2r_task_scope_tenant.py` (F-D2-02, F-D2-14). Two tenants on one capability: a write by tenant A is not readable, overwritable or deletable by tenant B or C (today B overwrites and C deletes); the namespace encoding is injective (a tenant id or namespace containing the separator cannot collide with another pair, test over adversarial pairs); a TASK remember/forget with a governance policy that denies is denied and no capability call happens; a TASK call with an identity whose tenant differs from a capability that exposes `bound_tenant_id` is denied before any capability call; a duck-typed capability that exposes `bound_tenant_id` is guarded the same way as the shipped wrapper; USER-scope behavior is unchanged (existing `test_xd02_*` stay green). If the governance service cannot evaluate a TASK request without a record snapshot, the implementer stops and reports (decision below) instead of inventing a snapshot.
3. `test_d2r_freshness_hook_isolation.py` (F-D2-11). A hook that raises does not raise through `RetrievalService` (stale source observed, `request_reindex` default) nor through `evaluate_embedding_version`; the retrieval result is returned and the failure is observable (logged); the other hooks still run; negative: a healthy hook still receives the request.
4. `test_d2r_retention_task_hardening.py` (F-D2-13). A payload `as_of_iso` beyond `clock() + skew` is refused with a typed failure (`memory_retention_sweep.as_of_out_of_range`) and nothing expires; a past `as_of_iso` still works (existing sweep tests stay green); a sweep that exceeds the deadline returns a typed timeout failure and the coroutine is cancelled (a fake sweep that never completes); negative: tenant mismatch tests of the existing suite stay green.
5. `test_d2r_test_tree_syntax_gate.py` (F-D2-09). `ast.parse` over every `*.py` under the D2-cited test trees (`tests/unit/rag`, `tests/integration/rag`, `tests/integration/context`, `tests/unit/context`, `tests/qualification/pdc_x/d2`, `tests/qualification/pdc_x/d2_repair`); the test is red until the Neo4j file is repaired and fails again if any file in these trees is broken.
6. `test_d2r_pre_context_policy_gate.py` (F-D2-10). The audit finds its roots (no "missing scan root"); a negative case with a scan root that does not exist fails the audit; `test_ideal_w2_w2_script_gates` (edited) reports every failing script.

### 2. Test hygiene (F-D2-08, F-D2-09, F-D2-10, F-D2-18)

- Step 1 (first act of the implementer): one `uv run pytest tests/unit/rag tests/integration/rag tests/integration/context tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py -q --continue-on-collection-errors` with `--extra dev-ci --extra dev-ci-rag` (the existing CI profile, see Operator decisions) and once without extras; log under `.tmp/session/t041b/`. Every red id is classified in `tests/qualification/pdc_x/d2_repair/RAG_RED_CLASSIFICATION.md` (id, class, action); ids that are product bugs are listed and fixed in scope (file named in the report, edit scope limited to the cited owner module).
- Stale doubles are repaired in the test files that define them (the doubles are inline per file: `FakeVectorstore`, `_NoopVectorstore`, `_BridgeHandler`, `_DummyHandler`, `FakeVectorStoreManager`, `InmemoryVectorStoreIntegration`); tests of removed APIs (`PipelineDocumentHandler`, `RagProfile(graph_store_backend=...)`) are ported to the current API; a test is deleted only when the API it covered was removed on purpose, and the report names the removal commit.
- Env-dependent modules (langchain_core, qdrant_client, docx, pandas, fitz, sentence_transformers) get a module-level `pytest.importorskip(<module>, reason=...)` naming the extra; the classification file states that the CI RAG profile (`rag-guard.yml`) runs them and that an environment without the extra reports them as skipped, not green.
- F-D2-18: assert the specific exception type or typed refusal in the cited tests; reconstruction assertions use the durable store, not the in-test originals.
- F-D2-10: `pre_context_policy_audit.py` derives its scan roots from the repository layout (locate the relocated UAEP modules) and keeps failing closed when a root is missing; `check_pre_context_policy_wiring.py` is adjusted only as needed; the gate script is strengthened, never relaxed.

### 3. Product fixes (after the tests of section 1 are red for the right reason)

1. **F-D2-01.** New module `intergrax/context/budget/message_token_cost.py` (below the runtime layer: `runtime/nexus/context` already imports `context/budget`, so no new upward import) with one pure function `message_token_cost(message, count_text)` (content plus deterministic serialization of `tool_calls`, `name`, `tool_call_id`, same recipe as the D3 estimator) and a helper for the non-content share. Users: `classify_candidates`, `_enforce_hard_budget` (including the last-user check), `_hard_trim_messages`, the `TOKENIZER_HARD_TRIM` accounting, the three estimators of `mandatory_base_messages.py` (so `estimate_mandatory_reserve_tokens` follows without an edit), `CharEstimateContextTokenCounter.count_messages` and `count_message_tokens` in `context_preflight.py`. Hard trim: content is truncated against `limit - non_content_cost`; a group whose non-content cost alone does not fit is dropped as a unit; a mandatory message that cannot fit fails closed.
2. **F-D2-17.** `_enforce_hard_budget` scans oldest-first for the drop unit.
3. **F-D2-02, F-D2-14.** In `default_memory_control_plane.py`: a private `_task_namespace_for_tenant(tenant_id, namespace)` (length-prefixed, injective) applied to write and delete; the bound-tenant guard and `_enforce_governance` (operation `REMEMBER` / `DELETE`) run for TASK scope; the guard checks `bound_tenant_id` through a small runtime-checkable protocol added to `memory/contracts/memory_control.py` (additive; `TaskMemoryCapability` signature unchanged, so existing implementers keep working). The namespace seen by a capability changes; the `memory_behavior` TASK scenarios that assert the raw namespace are updated and each edit is reported.
4. **F-D2-11.** The hook loop in `_notify_reindex_needed` (`embedding_version_policy.py:157-158`) isolates each hook (log with `exc_info`, continue, return `False` if no hook accepted); `retrieval_service.py` needs no edit.
5. **F-D2-13.** `retention_sweep.py`: `as_of` bound, deadline through `asyncio.wait_for`, bounded join.
6. **F-D2-10.** `pre_context_policy_audit.py` roots.

Each product fix is one commit with its pinned tests green. Constraint: ruff and pyright on every touched file; no new import from `agents/` or `applications/` into `intergrax/`.

## Write set

The block below is the complete code and test write set, one path per line. It is equal, line for line, to the `writes` tuple of batch `B-P7-D2` in `scripts/qualification/pdc_x_wbs.py` and therefore to the row in `PDC_IMPLEMENTATION_WBS.md` (the implementer and the delegate compare them with the one-liner in Protocol item 7). Anything else is a write-set amendment, which is not delegated and returns to the Operator.

```text
intergrax/context/budget/message_token_cost.py
intergrax/context/budget/mandatory_base_messages.py
intergrax/context/budget/token_counter.py
intergrax/runtime/nexus/context/context_compiler.py
intergrax/runtime/nexus/context/degradation_ladder.py
intergrax/runtime/nexus/context/context_preflight.py
intergrax/memory/default_memory_control_plane.py
intergrax/memory/contracts/memory_control.py
intergrax/memory/retention_sweep.py
intergrax/rag/governance/embedding_version_policy.py
intergrax/runtime/policy/pre_context_policy_audit.py
scripts/maintenance/check_pre_context_policy_wiring.py
tests/qualification/pdc_x/d2_repair/
tests/unit/rag/document_loaders/parsers/test_parser_pipeline.py
tests/unit/rag/document_loaders/registry/test_document_handler_registry.py
tests/unit/rag/document_loaders/test_document_loader.py
tests/unit/rag/document_splitters/test_native_strategies.py
tests/unit/rag/indexing/test_indexing_manager_indexes_documents.py
tests/unit/rag/indexing/test_single_index_strategy.py
tests/unit/rag/ingest/test_hierarchical_dual_index_wiring.py
tests/unit/rag/ingest/test_ingest_lci2b_loader_boundary.py
tests/unit/rag/ingest/test_ingest_pipeline_graph_wiring.py
tests/unit/rag/retrieval/test_ebh_2g_r1_retrieval_contract_typing_gate.py
tests/unit/rag/retrievers/test_hybrid_retriever.py
tests/unit/rag/vectorstore/test_hybrid_capability_routing.py
tests/unit/rag/vectorstore/test_source_ownership_contract.py
tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py
tests/integration/rag/document_loaders/parsers/test_doc_smart_document_handler.py
tests/integration/rag/document_loaders/parsers/test_document_handlers_framework.py
tests/integration/rag/document_loaders/parsers/test_excel_smart_document_handler.py
tests/integration/rag/document_loaders/parsers/test_html_smart_document_handler.py
tests/integration/rag/document_loaders/parsers/test_image_smart_document_handler.py
tests/integration/rag/document_loaders/parsers/test_pdf_smart_document_handler.py
tests/integration/rag/document_loaders/parsers/test_video_smart_document_handler.py
tests/integration/rag/document_loaders/test_fixture_dataset_ingestion.py
tests/integration/rag/embedding/test_hf_embedding_pipeline_integration.py
tests/integration/rag/embedding/test_ollama_embedding_pipeline_integration.py
tests/integration/rag/embedding/test_vllm_embedding_pipeline_integration.py
tests/integration/rag/rerankers/test_cross_encoder_reranker.py
tests/integration/rag/rerankers/test_embedding_cosine_reranker.py
tests/integration/rag/rerankers/test_semantic_reranker.py
tests/integration/rag/retrievers/test_retrieval_integration.py
tests/integration/rag/retrievers/test_retriever_pipeline_basic.py
tests/integration/rag/test_graph_reingest_qualification.py
tests/integration/rag/test_neo4j_live_qualification.py
tests/integration/rag/test_same_source_reingest_serialization.py
tests/integration/context/test_mem_xint6_cross_layer_e2e_certification.py
tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py
tests/qualification/pdc_x/d2/test_d2_compaction_roundtrip.py
tests/qualification/pdc_x/d2/test_d2_context_fidelity.py
tests/qualification/memory_behavior/fixtures.py
tests/qualification/memory_behavior/scenarios/task.py
tests/unit/runtime/nexus/context/test_context_compiler.py
tests/unit/context/test_ce_02_r1d_safe_degradation.py
tests/unit/context/budget/test_ce02_r1i_mandatory_provider_hard_budget.py
tests/qualification/ce_02/test_ce_02_qualification_batch.py
tests/qualification/pdc_x/d2/test_d2_memory_retention.py
```

Classification of the block:

- Product (12 files, one new): the first twelve lines (`message_token_cost.py` is new).
- New tests: `tests/qualification/pdc_x/d2_repair/` (the six test files of Deliverables 1 and `RAG_RED_CLASSIFICATION.md`).
- Hygiene edits (F-D2-08/09/10/18): the 35 files of audit run 5 (14 under `tests/unit/rag`, 19 under `tests/integration/rag`, `tests/integration/context/test_mem_xint6_cross_layer_e2e_certification.py`, `tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py`) plus `tests/qualification/pdc_x/d2/test_d2_compaction_roundtrip.py`, `tests/qualification/pdc_x/d2/test_d2_context_fidelity.py` (F-D2-18), `tests/qualification/memory_behavior/fixtures.py` and `tests/qualification/memory_behavior/scenarios/task.py` (TASK namespace expectation).
- Conditional (the last five lines; edited only if a pinned expectation must change because the cost of `tool_calls` messages or the drop order changed; every such edit is reported with the old and the new expectation): `test_context_compiler.py`, `test_ce_02_r1d_safe_degradation.py`, `test_ce02_r1i_mandatory_provider_hard_budget.py`, `test_ce_02_qualification_batch.py`, `test_d2_memory_retention.py`.
- Governance and documents (not part of the row): `governance/tickets/T-IGX-041-B.md`, this plan, `scripts/qualification/pdc_x_wbs.py` and the generated `PDC_IMPLEMENTATION_WBS.md` / `PDC_DEPENDENCY_DAG.md`.

Not edited: `intergrax/llm_adapters/` (T-IGX-041-C owns the estimator half), every file under the hotspots `intergrax/applications/_shared/` (including `memory_wiring.py`, `memory_control_wiring.py`, `task_memory_wiring.py`) and `intergrax/contracts/`, `intergrax/context/formatter.py`, `provider_lifecycle.py`, `planner.py` and `mandatory_reserve.py`, `intergrax/rag/retrieval/retrieval_service.py` and `source_freshness.py`, `intergrax/runtime/context/citation_chain.py`, `tests/qualification/tenant_x/` (T-IGX-041-I), `scripts/governance/`, `.github/`, `pyproject.toml` and `uv.lock` (no dependency change; extras are used from the existing CI profile only), the IIAC activation record, the EBH-7 pins and allowlist, the requirements matrix.

## Protocol

1. Token-budget preflight (read scope, edit scope, tests) before each step; stop and ask when more files than this plan names are needed.
2. Tests of Deliverables 1 are committed first and recorded red under `.tmp/session/t041b/` (a red run per file, for the right reason: the unchanged code fails the assertion, not an import or fixture error); then the fixes.
3. One `uv run pytest` per verification step, several paths in one command; no parallel `uv` or pytest. Logs under `.tmp/session/t041b/`.
4. Mutation check of the new guards by runtime monkeypatch (no file edit), one at a time: (a) message cost reverted to content-only, (b) drop order reverted to newest-first, (c) tenant qualification removed from the TASK namespace, (d) hook isolation removed. Each new test file must fail for its mutant; the results are recorded in the PR body.
5. Commit with exactly one `Ref: governance/tickets/T-IGX-041-B.md` trailer, no AI attribution, no `Approved:` line (the implementer never writes one).
6. Open one PR after the D3 change (T-IGX-041-C, PR 569) is merged and this branch is rebased on it; bind the PR; no merge by the implementer (merge gate: `DELEGATION-PDC-P7-REPAIR.md`).
7. Write-set equality check before every commit and before the PR: the lines of the Write set block of this plan equal the `writes` tuple of `B-P7-D2` (`uv run python scripts/qualification/pdc_x_wbs.py --check` for conflicts; the equality itself is compared by reading both lists). A needed path outside the block stops the work and returns to the Operator.

## Acceptance

- Every in-scope finding has a pinned test that is red on the unchanged code and green after the fix, plus the negatives listed; mutants (a)-(d) are killed.
- The 35 red files of audit run 5 are each green, repaired, or explicitly classified (`RAG_RED_CLASSIFICATION.md`): zero unclassified reds; env-dependent modules are named as skipped, never silently green; the Neo4j file collects.
- `test_ideal_w2_w2_script_gates` and `check_pre_context_policy_wiring.py` pass; the audit still fails closed for a missing root.
- ce_02, `pdc_x/d2`, `tests/unit/context`, `tests/unit/memory`, `tests/qualification/memory_behavior` stay green (each changed expectation reported).
- `git diff --stat` lists only the write set; no hotspot path is touched; `uv run python -m scripts.governance.igx_gov ticket` passes.
- The PR body states: not a certification verdict; the IIAC activation stays stale until the P8-04 rebind; the EBH-7 identity gate stays red until the final re-pin; F-D2-03 carriers hand-over to T-IGX-041-I.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d2_repair tests/qualification/pdc_x/d2 tests/qualification/ce_01 tests/qualification/ce_02 tests/qualification/memory_behavior tests/unit/context tests/unit/memory tests/unit/runtime/nexus/context tests/unit/rag tests/integration/rag tests/integration/context tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py -q --continue-on-collection-errors
uv run ruff check intergrax/runtime/nexus/context intergrax/context/budget intergrax/memory intergrax/rag/governance/embedding_version_policy.py intergrax/runtime/policy/pre_context_policy_audit.py tests/qualification/pdc_x/d2_repair
uv run pyright intergrax/context/budget/message_token_cost.py intergrax/runtime/nexus/context/context_compiler.py intergrax/runtime/nexus/context/degradation_ladder.py intergrax/runtime/nexus/context/context_preflight.py intergrax/memory/default_memory_control_plane.py intergrax/memory/retention_sweep.py tests/qualification/pdc_x/d2_repair
uv run python -m scripts.governance.igx_gov ticket
```

(`--continue-on-collection-errors` stays only until the classification step has converted every collection error into a repair or an explicit skip; the final run has zero collection errors. The RAG part runs with the CI profile extras when the Operator authorizes them.)

## Out of scope

- Every row of "Excluded findings"; F-D2-03 and the TENANT-X carrier re-pin (T-IGX-041-I); the adapter estimator half of F-D2-01 (T-IGX-041-C); composition of the retention sweep or the freshness observer; the requirements matrix and ownership citations; hotspot files; re-pinning EBH-7; the IIAC audit, activation and rebind; FRZ promotion; OD signatures; certification.

## Operator decisions needed

1. **Extras for the RAG suites.** Authorize `uv run --extra dev-ci --extra dev-ci-rag` (existing `rag-guard.yml` profile: langchain-core, opentelemetry, qdrant-client; no Docker, no new vendor integration, no `pyproject.toml` change) in the implementer worktree to classify the extras-dependent reds. Default if not authorized: classification by reading only, the affected modules become explicit skips, and F-D2-08 stays partly unverified.
2. **TASK governance without a record snapshot.** If `MemorySecurityGovernanceService` cannot evaluate a TASK request that has no `MemoryGovernanceRecordSnapshot`, the implementer stops and reports; the fallback (tenant qualification and the bound-tenant guard only, governance for TASK scope declared as an exception) needs the Operator.
3. **Hotspot and wave note.** This batch writes no hotspot file. The D1 and D5 plans name files under `intergrax/applications/_shared/` in the same wave W13 (the generator allows one batch per wave per hotspot); that conflict is theirs, not this batch's.
