# T-IGX-051-D — RW train, node T-AF2-RW-RUNTIME: LLM runtime resilience, typed error contract v1, served model, refusal matrix and SDK boundary (EX-L-08, EX-D3-01, EX-D3-03, EX-L-07, D3-R005, D3-R019, A4-018)

## Scope

Product-path ticket of the single RW product train (MEP-AF2 node `T-AF2-RW-RUNTIME`, RW-PLAN v13 lane 3; branch `arch-freeze/rw-train`). Drafted from the plan of record section E (row "REMEDIATION — runtime / LLM" and answer WP12-Q02 for A4-018), RW-PLAN v13 appendix A1 block `T-AF2-RW-RUNTIME (T-IGX-051-D)` and its negative-test rows NT-ROW-005, -036, -037, -039 and -103..-108. Facts were read at `origin/development` `2727a2c2364a97492202b931375772a400279a9b`, whose product tree equals AUDIT_BASELINE_2 `7960ae7f6` and the train branch on every path of the write set.

Owner per RW-PLAN (exactly-once ownership): A4-018, EX-D3-01, EX-D3-03, EX-L-07, EX-L-08, FRZ-CTR-01, FRZ-CTR-02, FRZ-CTR-06, PDC-D3-R005, PDC-D3-R019. Contributes to: FRZ-CFG-02, FRZ-CFG-06, FRZ-SEC-04, PDC-D3-R018 (RW-PLAN dedupe D-07 reassigned R018 to `T-AF2-RW-SUPPLY`; see plan question Q-RUN-01).

- Part A (EX-L-08): the stream is opened and its first event pulled inside the physical attempt, so the circuit breaker and the retry loop see stream-open errors.
- Part B (EX-D3-01): the effective call timeout is forwarded to the provider SDK call through the mechanism each SDK offers; adapters that cannot forward it fail closed.
- Part C (EX-D3-03): typed provider-error contract v1 (`intergrax/llm_adapters/contracts/provider_error.py`, new) mapped at the adapter boundary; existing typed errors pass through unchanged.
- Part D (EX-L-07): a parametrized refusal and finish normalization matrix over every shipped adapter method and the Bedrock invoke paths; product edits only where a cell exposes a defect.
- Part E (PDC-D3-R005): an additive `served_model` field on the adapter response; call accounting records the served model.
- Part F (FRZ-CTR-01, PDC-D3-R019): capability-flag parity over every shipped adapter; VBN register rows for the 8 provider `DEFAULT_MODEL` constants.
- Part G (A4-018, FRZ-CTR-02, FRZ-CTR-06): no `ollama` import outside the provider package (`images_loader.py`, `qualification_types.py`); the vision call moves into a provider module and the qualifier classifies the typed errors.

Write set (exact list in the plan, section "Write set"; it equals `RW_PLAN.md` appendix A1 block `T-AF2-RW-RUNTIME`, with no difference): 17 firm paths (9 product: 6 files and the 3 globs `intergrax/llm_adapters/contracts/*`, `intergrax/llm_adapters/providers/*`, `intergrax/runtime/llm/*`; 1 existing support file `tests/qualification/ebh_6/_findings_register.py`; 7 new test files) and 0 conditional paths. Product commits go only to branch `arch-freeze/rw-train`. A need for any file outside this list (the plan's forbidden list names the likely ones) stops the work and goes to the orchestrator as an amendment. ACT-N11: every product path above breaks the IIAC activation until the single RB-2 re-pin; this ticket does not perform it and must not merge outside the wave. Contract surfaces get no new `Any`, `object` or reflection dispatch (OD-AF-11); the ticket edits no path under `intergrax/contracts/**`.

Prerequisites before implementation starts (not assumed): PT-WA and PT-WB merged into the train, the delegate approval block of this ticket, the product-path authorization of the node. Nothing in the repository may be edited before the wave of this ticket.

No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan. The items above are not closed by the draft; closure conditions are in plan section 15.

Commit trailer for the implementer: `Ref: governance/tickets/T-IGX-051-D.md` on every commit that touches `intergrax/`, `agents/`, `applications/` or `scripts/`. No AI-tool attribution in commits or the pull request.

## Plan digest

Plan digest: 5297254e206eeb5966f7b88f9538f4cac8ac67c249258612163a91b767ef1ca0

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-051-D.md` (drafted at `.tmp/session/arch-freeze/w3-drafts/rw-runtime/PLAN_T-IGX-051-D.md`; the plan is copied unchanged into `governance/tickets/plans/` when the ticket is committed). The digest was computed with `scripts/governance/plan_digest.py` (a copy extracted from `origin/development`) on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T15:17:24+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_DELEGATION_RECORD_4.md@5ba8c4928a06345d6885a59cc2519a793de73caf
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Independent audit

Audit line (RW-PLAN RWA-05, audit flag A4-018 bound to node `PT-Q02`): the independent audit of this ticket is a separate session at the exact wave head; the implementer of this ticket does not PASS. Mutation checks for the auditor are in plan section 16.

## Test contract

Nothing was run when this ticket was drafted. One process at a time, one `uv run pytest` per verification step with multiple paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock` (only the creator of the lock removes it); no broad or diagnostic suite locally, no live or network-marked test, no Docker, no vendor integration (C10/C15 freeze). Exact commands are in plan section 11:

- Red-first (S0): the seven new test files on a detached worktree at the pre-change commit, log kept under `.tmp/session/arch-freeze/rw-runtime/red-first/` for the auditor. Nine negative-test rows are red-first (basis B: NT-ROW-103..107; basis S: NT-ROW-005, -037, -039, -108); NT-ROW-036 is basis C (characterization, result recorded at step 0). A row that is not red at base is recorded as not red and re-disposed by the orchestrator; it is never reported as red.
- After the change (S8): the same seven files green; then `tests/unit/llm_adapters`, `tests/unit/multimedia`, the `tests/qualification/ebh_5` and `ebh_6` directories (full runs are CI at M3; a local restriction is recorded), the L1 gates (`check_intergrax_no_applications_imports.py`, `check_agent_registry_bypass.py`, `check_agents_no_tier3_imports.py`, `check_agents_no_vendor_sdk_imports.py`, `check_integration_vendor_imports.py`, `check_dependency_ownership.py`, `igx_gov ticket`, `verify_arch_freeze_plan_traceability.py`, `arch_freeze_b1_closed_world_inventory.py --strict-certify`), `uv run ruff check` and `uv run pyright` on the changed Python files. Gates that are expected red on the train (RW-PLAN table 1) stay expected.
- Ratchet: the wave-packet contract check (`rw-plan/work/typ05/measure_typ05.py` plus `typ_measure.py`) must report nothing new for the new and changed code in `intergrax/llm_adapters/contracts/*`.
- Proposed regression additions beyond the RW-PLAN paths (plan question Q-RUN-10): the PDC-D3 stream-guard and refusal tests, the provider-cancellation and dependency-isolation architecture tests, the harness_02 gates, the deadline-authority qualification tests and the roadmap-replay ledger tests.

## Out of scope

- `intergrax/llm_adapters/_shared/resilience.py`, `_shared/retry.py`, `_shared/call_lifecycle.py`, `_shared/adapter_response_builders.py`, `_shared/stream_finish_signal.py`, `base/usage_log.py`, `governance/quota.py` (PDC-D3-R018 root cause F-D3-12, `T-AF2-RW-SUPPLY`), `registry/*` including `LLMProfile.lab`, `agents/model_routing_qualifier/steps/model_routing_job.py`, the rag default embedding provider, and the Tier-3 benchmark import of the SDK in `applications/local_workspace_application/benchmarks/local_model_qualification/environment.py`.
- The banned-list extension of the two vendor-SDK gate scripts (`T-AF2-RW-GATES`), the VBN dossier text (`T-AF2-VBN-DOSSIER`), the W-F decisions on the direct SDK seam (moved here by dedupe D-05), consumer migration to the typed errors outside the write set, live-provider qualification of any adapter.
- The IIAC re-pin (RB-2), the EBH-7 pin, FRZ status promotion, stage closure, GitHub settings and rulesets, validator code, `scripts/`, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, PDC-X and IIAC record directories, Docker or vendor work, production configuration.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_db61e386-18ef-4102-952a-f87126b440ff

## Implementation record (implementer; not an approval, no PASS)

Step 0, SDK timeout mechanisms (signatures of the SDK versions in `uv.lock`):

| Adapter | Mechanism | Class |
|---|---|---|
| OpenAIChatCompletionsAdapter (with Azure and the OpenAI-compatible family) | `chat.completions.create(timeout=)` | PER_CALL |
| OpenAIChatResponsesAdapter | `responses.create(timeout=)`, `responses.stream(timeout=)` | PER_CALL |
| ClaudeChatAdapter | `messages.create(timeout=)` | PER_CALL |
| MistralChatAdapter | `chat.complete(timeout_ms=)` | PER_CALL |
| CohereNativeChatAdapter | `request_options={"timeout_in_seconds": ...}` | PER_CALL |
| GeminiChatAdapter (with Vertex) | `GenerateContentConfig(http_options=HttpOptions(timeout=<ms>))` per call | PER_CALL |
| NativeOllamaAdapter | `ollama.Client(timeout=)` built for the call when the adapter built its client | PER_CALL_DERIVED_CLIENT; NONE for an injected client |
| LangChainOllamaAdapter | `ChatOllama(client_kwargs={"timeout": ...})` built for the call when the adapter built its chat model; its HTTP client is closed after the call | PER_CALL_DERIVED_CLIENT; NONE (fail closed) for an injected `chat` |
| BedrockChatAdapter | `boto3.client(config=botocore.config.Config(connect_timeout=, read_timeout=))` built for the call when the adapter built its client; closed after the call | PER_CALL_DERIVED_CLIENT; NONE (fail closed) for an injected client |

Step 0, other records: capability-flag parity (NT-ROW-036) has no mismatch at base, recorded NOT_REPRODUCED; refusal matrix (NT-ROW-106) red at base in 5 cells (Claude `stream_with_tools` x3, Bedrock InvokeModel Mistral length x2); served model reported by OpenAI chat / Responses / Mistral / Anthropic (`model`), Gemini (`model_version`), Ollama (`model`), LangChain Ollama (`response_metadata["model"]`), Bedrock InvokeModel Anthropic body (`model`); not reported by Bedrock Converse and Cohere v2 (`served_model` stays `None`). Red-first and green logs: `.tmp/session/arch-freeze/T-IGX-051-D/` (main checkout).
