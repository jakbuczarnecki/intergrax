# PLAN T-IGX-043-C — EQO-A remediation FC-3: vendor/provider selector fail-closed (product part; the CONFIG-X gate extension is T-IGX-043-F)

Status: plan approved by the commissioned signing delegate (see the `## Approved` section of `T-IGX-043-C`); the approval covers the plan text and is NOT an implementation go (implementation waits for the ARCH-FREEZE gate `SCOPE` and the Operator's product-path go). Drafted 2026-11-16 by a planning session (read-only; no branch, no worktree, no commit, no test run). Nothing here is applied or audited. The orchestrating session relayed on 2026-11-16 that the Operator accepted the recommended option of every open question (Q-FC3-01..11); the decided answers are recorded in `OPEN_QUESTIONS.md` and the design below states only the decided design. No Operator-written record of those answers was seen by this session. Master-plan node `PT-FC3` (`WP-12/TICKETS_DRAFT.md`, `WP-12/DAG.md`).

Revision 2 (after the independent delegate review `FC/REVIEW/FC_DELEGATE_REVIEW.md`, 2026-11-16, section 4 and X-1, X-2, X-3, X-4, X-5, X-8): Part G (CONFIG-X scan extension) is split into the separate draft ticket `T-IGX-043-F`, a gate-meaning change that needs the Operator's personal ruling; the notification carrier was first judged unavailable and a recorded exception was drafted (withdrawn in revision 3); the census command is a bounded file list, not a directory run; merge order and file ownership are stated in section 9; the posture helper is imported from T-IGX-043-B; IIAC wording follows the record. The Operator's acceptance of Q-FC3-01..11 is a relay of 2026-11-16 by the orchestrating session; the decision record is a session artefact under `.tmp`, not a repository record (see `OPEN_QUESTIONS.md`).

Revision 3 (after the delta review `FC/REVIEW/FC_DELTA_REVIEW_1.md`, finding N-1, 2026-11-16): the premise of revision 2 that no carrier exists from a host to `NexusLoop` was wrong. At H the carrier exists as the existing keyword `notification_adapter` of `build_harness_host_runtime` (`intergrax/applications/_shared/harness_host_runtime.py@B:312`, forwarded at `:605`) through `host_orchestration_loop_init_spec.py@B:84` and `environment_orchestration_materialization.py@B:83` into `NexusLoop(notification_adapter=...)`, and as the keyword of `wire_harness_host_long_running_scheduler` (`harness_host_auxiliary_wiring.py@B:50,60`) into the scheduler, which hands it to the coordinator (`runtime/long_running/scheduler.py@B:97,458`). Passing an explicit adapter through these existing keywords needs no edit of `harness_host_runtime.py`, `nexus_loop.py`, the spec or the materialization. The Operator's accepted answer Q-FC3-02 (option 1: no silent `log` default for HITL escalation and long-running notifications in PRODUCT/STRICT; hosts pass an explicit adapter) is therefore restored, the recorded exception is withdrawn, and the impact class of the leg stays BLOCKS (A4-019, effective). Paths not covered by composition-time refusal are named in section 3.3 and put to the Operator.

Revision 4 (the Operator's answers of 2026-11-16 as transcribed in `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a transcription by the orchestrating session of the Operator's chat answers of 2026-11-16, a session artefact, not yet a repository record)): O-09 = option 1 (62 product files accepted), O-08 = the plan as drafted (late refusal by the coordinator guard, class BLOCKS kept, unrefused paths named in the audit), L-1 = as drafted (section 3.3, product hosts whose profile has no notification channel refuse at composition), merge order B, A, C, F confirmed. Not decided by that file and still open there: whether PDC P9 is formally closed (M1), test authorisations other than the scratch measurement of `T-IGX-043-F`, and the publication of the transcription.

Notation. `B` = `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (Campaign A baseline, PR #589). `H` = `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (`origin/development` after one `git fetch origin` on 2026-11-16). `path@B:line` and `path@H:line` mean the full SHAs above; `FACT_CHECK.md` spells them out in every row. Head status per file: `git diff --name-only B H -- intergrax agents applications` lists exactly three files (`intergrax/rag/graph/generation_visibility.py`, `.../indexer/heuristic_graph_indexer.py`, `.../providers/inmemory_graph_store.py`); none is in this ticket, so every file below is `UNCHANGED` at H (checked again per directory, see FACT_CHECK F-01). The only test change since B in the touched trees is one new graph test file.

## 1. Purpose, findings, decisions

Make every vendor/backend/provider selector in the FC-3 set fail closed in PRODUCT and STRICT (OD-REM-03, C3). The product part must be correct and auditable on its own: it does not depend on the CONFIG-X gate extension. The gate extension (the dossier shows the CONFIG-X scan cannot see these defaults although FRZ-CFG-02/06/08 are PASS under CONFIG-X at `fac9a43f` per the checklist) is `T-IGX-043-F`, a separate ticket that changes what a gate reports and is the Operator's personal decision (section 9).

| Finding | Severity (dossier) | Defect at B | ARCH-FREEZE impact (WP-2 / master plan) |
|---|---|---|---|
| EQOA-A4-017 | MEDIUM | Triton/HF vision adapters delegate to `StubVisionInferenceAdapter` (full-frame `object.stub`, confidence 0.99) when unconfigured AND when the real backend answered with zero detections; stub slug is `yolo_ultralytics`; two more silent substitutions (`as_extended_adapter`, `task_runner`) | CONDITIONS |
| EQOA-A4-019 | MEDIUM | Embedding provider unset -> `ollama`; vector store slug unset -> in-memory; notification backend unset or unparsable -> `log` | BLOCKS (WP-2, effective class) |
| EQOA-A4-026 | MEDIUM | Inbound verifier mode unset or unparsable -> `none` (no-op verifier); also mode `slack`/`teams` without secret or enable flag is a silent no-op; `InteractionIntakeService` has a second implicit Null default | CONDITIONS; BLOCKS if probe V3 shows the intake route reachable unauthenticated in a STRICT host |
| EQOA-A4-032 part c | LOW | Guardrail factory returns a pass-through stub for an unknown slug; vendor guardrails silently degrade to the regex `pattern` engine when the SDK is not importable | CONDITIONS (WP-2; dossier said NONE) |

Decisions applied (`OPERATOR_DECISIONS_REMEDIATION_WAVE1.md`): OD-REM-03, C3 (fail-closed, negative tests, regression, independent audit at the exact SHA), C5/C6; **DP-24** hybrid (wiring layers fail closed on STRICT or PRODUCT; leaf selectors are default-deny with an explicit typed LAB opt-in); **DP-25** (guide defaults are LAB-scoped; A4-019 and A4-026 are fixed in code, not by documenting them); **DP-28** (vision stub only through an explicit LAB profile with a provenance flag; production Triton/HF fail closed; adds a field to `VisionInferenceResult`); **DP-18** (closed list of provider composition roots, list may not grow: this ticket adds no provider factory module and no composition root); **DP-14/DP-20** where the gate scope overlaps (this ticket contains no gate change; the CONFIG-X scan extension is the separate ticket `T-IGX-043-F`, the EBH-3 scan extension and boundary gates belong to `PT-G01`, see section 9); **DP-10** (a gate that is red on day one lands only together with its fixes; this governs `T-IGX-043-F`); **DP-11** (one integrated wave); **DP-06** (a PASS row contradicted by a finding keeps PASS with an annotation and counts as BLOCKS in the entry check; applied here to FRZ-CFG-02/06/08 by its wording, confirmation requested in OPEN_QUESTIONS).

## 2. Prerequisites and current PDC state (reported, not decided here)

- Wave node: `PT-FC3`, hard prerequisites `SCOPE` (= M1 + DS-1..DS-4 + DS-EXC + S1 diagnostics + Campaign B-1/B-2 alpha) and `DS-3` (`WP-12/DAG.md` row `PT-FC3`); authorizations AR-41 (wave package) and AR-27 for the negative-test part only (delegable through the ticket approval line); the gate extension is `T-IGX-043-F` and is not covered by AR-27 (corrected in `AUTHORIZATION_REQUESTS.md`).
- Ticket approval line: written by the commissioned signing delegate (AR-38; form `Approved: operator-delegate @ <timestamp>`, see the `## Approved` section of `T-IGX-043-C`; approves the plan, not an implementation go); a ticket-specific branch and worktree authorization, the single-process pytest lock, and the S1/probe results (all listed as A-01..A-10 in `AUTHORIZATION_REQUESTS.md`).
- PDC status at H, as written in the files: `PDC_INDEX.md@H` rows 44-49 still read "Answers recorded in chat; awaiting the Operator's final confirmation" for the P9 documents; `PDC_SHA_LADDER.md@H:39` records `FINAL_CLOSURE_PUBLICATION_SHA = feaa2a87d41c3c8ae95724590291f360d0232016` (PR #603), and `git log origin/development` lists PRs #601..#604 (closure package, closure text, final SHA) as merged. The master plan states M1 is reached. The index text and the ladder/commit history are not consistent with each other at H; this plan does not decide whether P9 is closed; the Operator has not decided it and it stays open (OPEN_QUESTIONS Q-FC3-08).

## 3. Design

### 3.0 Cross-cutting rule (DP-24 applied)

1. **Leaf selector** (no posture available): default-deny. With no explicit selection it raises a typed configuration error; the only way to a lab default is a typed `LabSelectorOptIn(selector, reason)` argument (frozen dataclass, non-empty `reason`, same shape as the existing `LabEphemeralSigningOptIn@B:intergrax/applications/_shared/critical_action_signing_wiring.py:31-45`). An invalid value (typo) raises in every posture; a LAB opt-in never legitimises an invalid value.
2. **Wiring layer** (has `ApplicationEnvironmentProfile`): fails closed when `wiring_fails_closed(env)` holds (`env.execution_mode is STRICT` or `env.application_profile is PRODUCT`; the helper lives in `intergrax/applications/_shared/wiring_posture.py`, which is created by T-IGX-043-B, merges first and is only imported here; this ticket creates no equivalent); a LAB opt-in handed to a PRODUCT profile raises (precedent: `resolve_critical_action_signing_wiring@B:intergrax/applications/_shared/critical_action_signing_wiring.py:72-96`, reason code `LAB_OPT_IN_FORBIDDEN_IN_PRODUCT@B:intergrax/core/security/security_configuration.py:11`). Only for `LAB` and not `STRICT` the wiring derives the opt-in itself from the explicit LAB profile.
3. **Errors reuse existing types**: `IntegrationConfigurationError@B:intergrax/integrations/contracts/base.py:110` for selectors (embedding, vector store, notification, guardrail), `SecurityConfigurationError@B:intergrax/core/security/security_configuration.py:20` (new reason codes) for the inbound verifier, one new `VisionBackendUnconfiguredError(ValueError)` next to `MediaAuthorizationError@B:intergrax/model_inference/contracts.py:48`. No new composition root (DP-18).
4. **Placement of `LabSelectorOptIn`** (decided, Q-FC3-05 option A): one shared type in `intergrax/integrations/contracts/base.py`, next to `IntegrationConfigurationError`, so that no new file enters the closed-world inventories and so that Tier-2/3 may import it as an `integrations.contracts` leaf (DP-14).
5. **No posture is invented**: the seven posture signals of WP-2 section 2 are not changed here (FC-1 owns `production_mode` on Nexus objects).

### 3.1 Part A — embedding provider (A4-019a)

| Item | Current code @B | Change |
|---|---|---|
| Default slug | `rag/embedding/registry/provider_authority.py:15,19-20` | Remove the implicit `default_embedding_provider_slug()`; keep the value `ollama` only as a named LAB constant reachable through the opt-in path |
| Env profile | `rag/embedding/registry/profile.py:18,47-57` (`os.getenv(f"{prefix}_PROVIDER", _DEFAULT...)`) | unset `INTERGRAX_EMBEDDING_PROVIDER` raises unless `lab_opt_in` is passed; unknown slug already raises through the validator (`profile.py:29-34`) |
| Resolver | `rag/embedding/runtime/resolver.py:73-82` (last branch returns the default) | last branch raises `EmbeddingProviderConfigurationError` (`rag/embedding/contracts/runtime_binding.py`, imported at `resolver.py:18-23`) unless `lab_opt_in` |
| Bootstrap chain | `rag/embedding/bootstrap/default_embedding_engine.py:34-36,54,79-95` | thread an optional `lab_opt_in` through `create_default_embedding_manager/engine/pipeline`; no new default |
| Tier-3 wiring | `applications/_shared/environment_wiring.py:288-290` calls `create_default_embedding_manager()` with no profile although `resolved_integration` is in scope | pass `integration_profile=resolved_integration`; STRICT/PRODUCT with no `embedding_provider` binding raises; LAB profile derives the opt-in |
| Other callers | `rag/bootstrap/entry_point_load.py:281`, `rag/bootstrap/rag_stack_bootstrap.py:92`, `rag/document_splitters/bootstrap/default_chunking_engine.py:48`, `rag/rerankers/bootstrap/reranker_bootstrap.py:70,137,171`, `rag/retrievers/bootstrap/retriever_bootstrap.py:87,192,237,286,323` | forward the optional `integration_profile`/`lab_opt_in`; they gain no default (a lab caller passes the opt-in) |
| Scripts | `applications/local_workspace_application/scripts/lkw_ollama_embedding_bootstrap.py:12-13`, `.../run-lkw-file-watcher-e2e-proof.py:759-761` (inline `embedding_profile_from_env()` strings) | pass the LAB opt-in explicitly (they are LAB scripts) |

Observation kept out of scope: `IntegrationProfile.legal_product()` selects `vector_store=INMEMORY` explicitly (`intergrax/integrations/contracts/integration_profile.py@B:287-296`), while `RAG.md@B:510` calls InMemory "harness use only"; that is an explicit selection, not a silent default; decided (Q-FC3-07 option 1): a separate finding and ticket, not part of FC-3.

### 3.2 Part B — vector store (A4-019b)

`rag/vectorstore/bootstrap/integration_vectorstore.py@B:107,120-127`: `if not slug:` builds an in-memory store. Change: raise `IntegrationConfigurationError` when no `vector_store` slug is resolved, unless `lab_opt_in` is given; the tenant check (`:123-124`) stays and applies to the opt-in path too. `create_vectorstore_manager` (`:144-173`) and the "backward-compatible alias" `create_default_vectorstore_manager` (`:176-182`) forward `lab_opt_in`. Callers to adapt (they currently rely on the fallback): `rag/bootstrap/{entry_point_load.py:282,rag_stack_bootstrap.py:87,hierarchical_bootstrap.py:41}`, `rag/retrievers/bootstrap/retriever_bootstrap.py:84,189,283,320`, `tools/providers/rag/scope.py:94`, `applications/local_workspace_application/workspaces/indexed_vector_verifier.py:39`. An explicit slug `inmemory` keeps working (explicit selection).

### 3.3 Part C — notifications (A4-019c)

| Item | Current code @B | Change |
|---|---|---|
| Settings | `runtime/notifications/backend_contract.py:29` (`backend: NotificationBackend = LOG`), `:42-49` (unset or unparsable -> LOG) | `backend` becomes a required field; unset raises unless `lab_opt_in`; unparsable raises always |
| Factory tail | `runtime/notifications/factory.py:98` `return create_log_notification_channel()` | delete; raise for an unsupported backend. Fact: the name is imported only inside the `LOG` branch (`:62`), so the line would raise `UnboundLocalError` if ever reached, i.e. it is an unreachable fallback today |
| Channel resolver | `factory.py:112-116` (unknown channel -> LOG), `runtime/long_running/notification.py:79-91` (wrapper; `channel or "log"` at `:84`) | in `factory.resolve_notification_adapter` an unknown channel raises and `None` raises unless `lab_opt_in`; in the long-running wrapper `None` raises (the `or "log"` default is removed); an explicit channel name (`slack`, `teams`, `webhook`, `log`, ...) stays an explicit selection and resolves |
| HITL and long-running notifications (Q-FC3-02 option 1, restored) | `runtime/long_running/coordinator.py:249-250,278-279,298-299,317-318` (`notify_channel or "log"`, then `adapter or resolve_notification_adapter(channel)`); the adapter reaches the coordinator from `NexusLoop._notification_adapter` (`runtime/nexus/nexus_loop.py:253,324,484,1079,1098` into `runtime/nexus/orchestration/hitl_runner.py:60,236-240` and `long_running_bridge.py:46,87`) and from the scheduler (`runtime/long_running/scheduler.py:97,458`); both are fed by existing keywords (see revision 3) and all Tier-3 hosts pass `notification_adapter=None` to the scheduler wiring (`applications/{dispute_sim:143,governed_contractor:438,intergrax_assistant:92,legal:133,local_workspace:225,poc_template:81,research:131}_application/host/factory.py`) and none passes it to `build_harness_host_runtime` | **Coordinator** (`coordinator.py`, four `notify_*` methods): resolution order is injected `adapter`, then an explicit `notify_channel`, otherwise `IntegrationConfigurationError`; the `or "log"` default stays only as the label of the message when an adapter is injected. **Hosts**: a new helper `resolve_host_notification_adapter(env)` in `applications/_shared/notification_wiring.py` returns the adapter of the profile's `notification_channel` through the existing `create_notification_adapter_from_profile`; when the profile has none, it raises if `wiring_fails_closed(env)` holds, and in a LAB profile that is not STRICT it derives a `LabSelectorOptIn` and returns the log adapter. Each host factory passes the one resulting adapter to both `build_harness_host_runtime(notification_adapter=...)` and `wire_harness_host_long_running_scheduler(notification_adapter=...)`. No edit of `harness_host_runtime.py`, `nexus_loop.py`, the spec or the materialization |
| Catalog wiring | `applications/_shared/notification_wiring.py:62-104` resolves through the profile; its `resolve_notification_settings()` calls (`:71,:90`) read the environment backend, which is now required. The profile slug is the explicit selection: when it is a `NotificationBackend` member the wiring passes it as `backend=`; when it is a catalog slug that is not a member (`discord`, `twilio`, `email_smtp`) the wiring resolves through `resolve_from_profile` without consulting the environment backend (UNVERIFIED trace; the implementer keeps the existing slack/teams/webhook tests green, nodes in section 6) | adapt `notification_wiring.py` so that an explicit profile never needs the env backend, a missing profile slug still refuses, and add `resolve_host_notification_adapter` |

Paths that this ticket does not cover with a composition-time refusal (named, not softened; the class of the leg stays BLOCKS): the builds of `build_harness_host_runtime` outside the seven factories and `lab_application` are `applications/attestation_demo/host/factory.py:62`, `applications/local_workspace_application/model_runtime_proof/runtime.py:146`, `intergrax/harness/app.py:170`, the two background-worker callers of `build_local_workspace_harness_host_runtime` (`background_worker_factory.py:85`, `background_worker_main.py:68`) unless the composition module passes the argument through, and the scaffold output not edited. They are still protected by the coordinator guard: with no injected adapter and no explicit channel the escalation raises instead of delivering to `log` (late and visible, not silent). Decided (O-08 = option 1, the plan as drafted): no composition-time refusal is added at `build_harness_host_runtime` (a file owned by T-IGX-043-A); the late refusal by the coordinator guard stays, the class of the leg stays BLOCKS, and the audit names the unrefused paths above. `lab_application` already passes `integrations.notification_adapter` to both calls (`applications/lab_application/host/factory.py:122,147`). The scheduler wiring was traced from the host keyword to `scheduler.py:97,458`; the end-to-end behaviour of the other HITL paths is UNVERIFIED (reachability probes V4/V5 belong to Campaign B).

### 3.4 Part D — vision (A4-017, DP-28)

| Item | Current code @B | Change |
|---|---|---|
| Provenance flag | `model_inference/contracts.py:68-71` (`VisionInferenceResult`), `:80-84` (`VisionSegmentationResult`), `:92-95` (`VisionOcrResult`) | add a backward-compatible `provenance` field (enum `engine` default, `stub`) to `VisionInferenceResult` (DP-28); the same field on the other two result types (decided, Q-FC3-04 option 1) |
| Stub adapter | `model_inference/adapters/stub_vision.py:22` slug `yolo_ultralytics` (same string as the real adapter, `ultralytics_vision.py:24`); `:24-61` fabricated detection (0.99), segment (0.95), OCR text `[stub ocr]` | new non-impersonating slug (e.g. `stub_vision`); every result it returns carries `provenance=stub`; the class is documented LAB-only |
| Triton | `triton_vision.py:30,45-48,51-52,74-75` | constructor stays permissive (the registry builds unconfigured instances, see below); `detect()` raises `VisionBackendUnconfiguredError` before reading any media when no URL is set; the default `fallback or StubVisionInferenceAdapter()` is removed (an explicitly injected `fallback` stays as the LAB opt-in and its result keeps `provenance=stub`); zero parsed detections return an empty result in all modes |
| Hugging Face | `huggingface_inference_vision.py:41-45,48-49,61-62` | same; `DEFAULT_MODEL` and host literal (`:31,:54`) are FRZ-CFG-05 matters and stay out of scope (noted) |
| Registry placeholders | `registry/vision_profile.py:92-99` registers `TritonVisionServingAdapter()` and `HuggingFaceInferenceVisionAdapter()` with no configuration (`include_remote_placeholders=True`) | no change needed because construction does not raise; a negative test pins this |
| `as_extended_adapter` | `tools/providers/vision/inference_support.py:120-123` returns a new stub for a non-extended adapter; used at `tools/providers/vision/service.py:70,:102` | raise a typed unsupported-operation error |
| Worker | `model_inference/workers/task_runner.py:44,59` same substitution; `_run_vision` (`:89-100`) converts exceptions into `ModalityJobResult(error=...)` | raise inside the callback so the job result carries the error |
| Tool output | `tools/providers/vision/service.py:63,95,127` build outputs from `request_id` plus the list only; `tools/providers/vision/contracts.py:17-19,29-31,41-43` | the provenance flag must reach the tool outputs, otherwise the agent-visible result stays indistinguishable; add the field to the three output models |

Not changed (recorded): parse defaults that invent a confidence of 0.9 or a full-frame box when the vendor payload omits them (`triton_vision.py:92-102`, `huggingface_inference_vision.py:83-93`), and the OpenCV OCR placeholder text (`opencv_vision.py:123`); both are recorded for Campaign B-2 and a later ticket (decided, Q-FC3-06 option 2), not dropped. The `VisionInferenceResult` in `integrations/contracts/vision_serving.py:21` is a different class with the same name and is not touched.

### 3.5 Part E — inbound interaction verifier (A4-026)

| Item | Current code @B | Change |
|---|---|---|
| Settings | `runtime/interactions/verification/factory.py:37` (`mode = NONE`), `:52-56` (unset or `ValueError` -> NONE) | unset raises unless `lab_opt_in`; an invalid mode raises always |
| Factory | `factory.py:66-88`; tail returns `NullInboundRequestVerifier` | `none` and the tail require `lab_opt_in`; for `slack`/`teams` the built verifier must report `enabled` (secret present and enable flag true): `SlackSignatureVerifier._enabled = enabled and bool(secret)` (`slack_signature.py:38-39`), Teams the same (`teams_signature.py:35-36`), so mode `slack` without secret or flag is today a no-op; a verifier that would accept without verifying requires `lab_opt_in` |
| Null verifier | `verification/contract.py:20-24` ("default for laboratory paths") | keep the class; constructing it outside the opt-in path is refused by the factory and by the service |
| Service | `runtime/interactions/intake_service.py:45` `verifier or NullInboundRequestVerifier()` | `verifier=None` raises unless `lab_opt_in` |
| Wiring | `applications/_shared/interaction_wiring.py:18-38` (no posture, `create_inbound_verifier()` with no arguments, `:32`) | gains `env: ApplicationEnvironmentProfile` (typed) and optional `lab_opt_in`; when `wiring_fails_closed(env)` holds (STRICT or PRODUCT) it requires an explicit verifier mode with a secret and refuses `none` and any LAB opt-in; LAB and not STRICT derives the opt-in; the existing signature tests (`test_interaction_wiring_public_surface.py`) must still pass |
| Harness path | `applications/_shared/harness_host_auxiliary_wiring.py:92-105` receives `runtime` and discards it (`_ = runtime`) | derive `env` from `runtime.environment` (`HarnessHostRuntime.environment: ApplicationEnvironmentProfile`, `intergrax/applications/_shared/harness_host_runtime.py@B:251`) so `intergrax/scaffold/new_application*.py:535/600` generated code keeps its call shape |
| Direct callers | `applications/{dispute_sim:198,governed_contractor:442,intergrax_assistant:96,legal:213,local_workspace:386,poc_template:85,research:157}_application/host/factory.py` (each has `env` in scope), `applications/lab_application/host/factory.py:154`, `intergrax/debug/app.py:112,132`, `intergrax/lab/organization_worker.py:112` | pass `env`; the three LAB surfaces (`lab_application`, `debug/app.py`, `lab/organization_worker.py`) pass an explicit opt-in |

Decided (Q-FC3-03 option 1): PRODUCT and STRICT never accept `none`, a disabled verifier or a degraded guardrail engine; there is no typed acknowledgement path (the dossier's alternative wording is not adopted). Mode `slack`/`teams` requires the secret and the enable flag everywhere except behind a LAB opt-in.

### 3.6 Part F — guardrail (A4-032 part c)

| Item | Current code @B | Change |
|---|---|---|
| Unknown slug | `integrations/providers/llm_guardrail/_factory.py:45-47` returns `create_stub_guardrail(slug)`; `_stub_backend.py:11-52` is a pass-through whose `scan_tool_call` allows everything except a literal `BLOCK_TOOL` marker | unknown slug raises `IntegrationConfigurationError`; the stub is reachable only with `lab_opt_in` |
| Silent engine degradation | `bundles/llm_guard.py:16-33` and the six other bundles call `scan_patterns` when the vendor SDK import fails (`_vendor_opens.py:20-30` returns `None`); the only trace is `audit_payload={"engine": "pattern"}` (`_pattern_scanner.py:66`) | a typed `engine_degraded` signal on `GuardrailScanResult` (`integrations/contracts/llm_guardrail.py`) set by every pattern fallback; the default factory path raises when the vendor engine is unavailable unless the caller passes the opt-in |
| Wiring | `applications/_shared/guardrail_runtime_bridge.py:55-73` has `env`; no posture check | STRICT/PRODUCT never passes the opt-in; there is no profile-level allowance for the pattern engine (Q-FC3-03 option 1) |

### 3.7 Part G (CONFIG-X scan extension): moved to `T-IGX-043-F`

The extension of `tests/qualification/config_x/_config_x_semantic_production_scan.py` changes the verdict function of the gate that is the evidence for FRZ-CFG-02/06/08 (the scan today approves every string default except tenant `default` and localhost, `_classify_resolved_semantic_string_default@B:282-299`). That is a gate-meaning change and is personal to the Operator; a delegate cannot approve it. It is therefore not part of this ticket (design, tests and write set are in `PLAN_T-IGX-043-F.md`).

Dependency order. The product part (this ticket) does not depend on Part G for its own correctness: the existing CONFIG-X gates are not modified and must stay green on the product tree (command 2 in section 7); if one of them goes red, implementation stops and the plan is amended. Part G depends on this ticket: its repo-level test is green only after these fixes, so Part G merges after this ticket is merged, or in the same wave behind it with the Operator's ruling, never before it (DP-10). If the Operator rules Part G out, the closure condition of A4-019 that names the gate extension is not met by this ticket alone (section 10).

## 4. Write set

Separation required by the brief: **P/C are product paths (ACT-N11 sensitive); T are test paths (not in `PRODUCT_PATH_PREFIXES@H:scripts/governance/iiac_common.py:22` except `applications/*/tests/`, which match the `applications/` prefix); D are documents outside this ticket.**

### P — product paths named by the master plan or required by the design (ACT-N11 sensitive)

```text
intergrax/integrations/contracts/base.py                         (LabSelectorOptIn; Q-FC3-05 option A)
intergrax/integrations/contracts/llm_guardrail.py                (typed engine_degraded signal)
intergrax/core/security/security_configuration.py                (new reason codes only)
intergrax/rag/embedding/registry/provider_authority.py
intergrax/rag/embedding/registry/profile.py
intergrax/rag/embedding/runtime/resolver.py
intergrax/rag/embedding/bootstrap/default_embedding_engine.py
intergrax/rag/vectorstore/bootstrap/integration_vectorstore.py
intergrax/runtime/notifications/backend_contract.py
intergrax/runtime/notifications/factory.py
intergrax/model_inference/contracts.py
intergrax/model_inference/adapters/stub_vision.py
intergrax/model_inference/adapters/triton_vision.py
intergrax/model_inference/adapters/huggingface_inference_vision.py
intergrax/model_inference/workers/task_runner.py
intergrax/tools/providers/vision/inference_support.py
intergrax/tools/providers/vision/contracts.py
intergrax/tools/providers/vision/service.py
intergrax/runtime/interactions/verification/factory.py
intergrax/runtime/interactions/verification/contract.py          (documentation of the LAB-only class; no behaviour)
intergrax/runtime/interactions/intake_service.py
intergrax/applications/_shared/interaction_wiring.py
intergrax/integrations/providers/llm_guardrail/_factory.py
intergrax/integrations/providers/llm_guardrail/_stub_backend.py
intergrax/integrations/providers/llm_guardrail/_pattern_scanner.py
intergrax/integrations/providers/llm_guardrail/bundles/{llm_guard,presidio,guardrails_ai,llama_guard,nemo_guardrails,bedrock_guardrails,http_guardrail}.py   (7 files; each calls scan_patterns)
```
25 single paths plus 7 bundle files (every bundle that calls `scan_patterns`: `bedrock_guardrails`, `guardrails_ai`, `http_guardrail`, `llama_guard`, `llm_guard`, `nemo_guardrails`, `presidio`) = 32 files.

### C — callers that must change because the leaves stop defaulting (not named in the master-plan row; ACT-N11 sensitive)

```text
intergrax/applications/_shared/environment_wiring.py             (:290 embedding)
intergrax/applications/_shared/guardrail_runtime_bridge.py
intergrax/applications/_shared/harness_host_auxiliary_wiring.py  (:101)
intergrax/applications/_shared/notification_wiring.py            (explicit profile slug as backend; :71,:90; new `resolve_host_notification_adapter(env)`)
intergrax/model_inference/registry/vision_profile.py             (adds the explicit LAB stub profile helper `VisionProfile.lab_stub()`, DP-28)
intergrax/rag/bootstrap/{entry_point_load,rag_stack_bootstrap,hierarchical_bootstrap}.py
intergrax/rag/document_splitters/bootstrap/default_chunking_engine.py
intergrax/rag/rerankers/bootstrap/reranker_bootstrap.py
intergrax/rag/retrievers/bootstrap/retriever_bootstrap.py
intergrax/tools/providers/rag/scope.py
intergrax/runtime/long_running/{notification,coordinator}.py     (no `log` default; Q-FC3-02 option 1)
applications/local_workspace_application/host/host_runtime_composition.py   (pass `notification_adapter` through `build_local_workspace_harness_host_runtime` to `build_harness_host_runtime`, :109,:143)
intergrax/scaffold/{new_application,new_application_product}.py  (generated hosts pass an explicit adapter instead of `notification_adapter=None`, :526-531, :591-596)
intergrax/debug/app.py                                           (LAB opt-in at :112,:132)
intergrax/lab/organization_worker.py                             (LAB opt-in at :112)
applications/lab_application/host/factory.py                     (LAB opt-in at :154)
applications/{dispute_sim,governed_contractor,intergrax_assistant,legal,local_workspace,poc_template,research}_application/host/factory.py   (7 files: `env` passed to `wire_interaction_intake_service`; the explicit notification adapter of `resolve_host_notification_adapter(env)` passed to `build_harness_host_runtime` and to `wire_harness_host_long_running_scheduler`, section 3.3)
applications/local_workspace_application/workspaces/indexed_vector_verifier.py
applications/local_workspace_application/scripts/{lkw_ollama_embedding_bootstrap.py,run-lkw-file-watcher-e2e-proof.py}   (pass the LAB opt-in)
```
30 further files (listed above: the 25 of revision 2 plus the two long-running modules, `host_runtime_composition.py` and the two scaffold files, all added back by the restored Q-FC3-02 option 1). Size: 62 product files (32 + 30), against the eight areas named in the master-plan row (it also omits `model_inference/contracts.py`, which carries the DP-28 field). The Operator accepted Q-FC3-01 with an upper bound of "about 59 product files"; 62 is three above that figure (the composition module and the two scaffold files); decided (O-09 = option 1): 62 product files are accepted. Any `applications/*/tests/` file that the census (section 8, step 3) shows to be default-dependent on these hosts is edited in this ticket and is also ACT-N11 sensitive (prefix `applications/`).

Imported, not in this write set: `intergrax/applications/_shared/wiring_posture.py` (`wiring_fails_closed(env)`), created by T-IGX-043-B (FC-2), which merges first; this ticket creates no equivalent helper. If FC-2 does not merge first, FC-3 waits or branches from FC-2's branch.

### T — tests (not ACT-N11 paths, but not on the EBH-7 executable-identity allowlist `scripts/qualification/verify_ebh7_p03_immutable_evidence.py@H:20-37`: the tests merge before the EBH-7 pin)

New: `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py`, `tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py`, `tests/unit/runtime/notifications/test_fc3_notification_selector_fail_closed.py`, `tests/unit/model_inference/test_fc3_vision_fail_closed.py`, `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py`, `tests/unit/integrations/providers/llm_guardrail/test_fc3_guardrail_fail_closed.py`.
Not edited here: everything under `tests/qualification/config_x/` (Part G is `T-IGX-043-F`); the existing CONFIG-X gates stay unchanged and must stay green.
Inverted or updated (defect-pinning or default-dependent, section 6): `test_remote_vision_adapters.py`, `test_model_inference_registry.py`, `test_embedding_profile.py`, `test_integration_vectorstore_bootstrap.py`, `test_integration_vectorstore_scope.py`, `test_slack.py`, `test_teams.py`, `test_dep1_dependency_foundation.py`, and `tests/qualification/pdc_x/reg_p/test_reg_p_ollama_single_row.py` (a PDC-X test, edited inside this ticket, Q-FC3-09 option 1, with a note for the RB-2 and EBH-7 reviewers); the remaining default-dependent tests are found by the census in step 3 of section 8 and are edited in this ticket.

### D — documents (not in this write set)

`docs/project/technical/guides/PLATFORM_CONFIGURATION.md` (lines 74, 82, 84, 749, 841, 865-867 at H), `docs/project/architecture/MODALITY.md` (lines 317, 320-321, 372, 376, 455), `docs/project/architecture/RAG.md:510`, `intergrax/integrations/providers/llm_guardrail/USAGE.md`. DP-25 rewrites the guide as LAB-scoped (`PT-D07`, W2, before the code). Decided (Q-FC3-10 option 1): the W2 text states the current behaviour as LAB-scoped, and a one-line follow-up documentation patch rides with this wave; that patch is not in this write set.

### Forbidden paths

`scripts/`, `.github/`, `governance/` except this ticket and plan, `docs/` (all), `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, `docs/project/maintainers/qualification/**` (incl. `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json`, which lists `intergrax/model_inference/contracts.py`), and every file owned by another wave ticket: FC-1 (`runtime/nexus/**`, `runtime/execution/nexus_host_execution.py`, `task_memory`, `events/store.py`, `harness_host_runtime.py`), FC-2 (`runtime/capacity/**`, `applications/_shared/{production_capacity,queue_worker,memory,runtime_event_delivery,sandbox}_wiring.py`, `memory_provider_admission.py`), FC-4 (`skills/execution_pinning_*`, `distributed/contracts/kv_store.py`), PT-WA/PT-WB (resolver, profile projection, boundary inversion, composition roots). `PT-WB` also lists `applications/_shared` and "model profile classes": confirm file ownership before the first edit (Risk R5).

## 5. Negative tests (new; each must be red at B and green at the audited wave SHA)

Configuration "PRODUCT/STRICT" below means the wiring-level tests build an `ApplicationEnvironmentProfile` with `execution_mode=STRICT` or `application_profile=PRODUCT`; leaf tests carry no posture.

| ID | Exercises | Expected refusal |
|---|---|---|
| NT-FC3-E1 | `resolve_embedding_provider_slug()`, env unset, no profile | `EmbeddingProviderConfigurationError` |
| NT-FC3-E2 | `embedding_profile_from_env()`, `INTERGRAX_EMBEDDING_PROVIDER` unset | raises |
| NT-FC3-E3 | `create_default_embedding_manager()`, env unset | raises before any provider import |
| NT-FC3-E4 | same with `lab_opt_in` | positive control: resolves the LAB slug explicitly |
| NT-FC3-E5 | invalid slug with and without `lab_opt_in` | raises in both cases |
| NT-FC3-E6 | `wire_application_environment` with STRICT env, RAG enabled, no `embedding_provider` binding | raises (wiring layer) |
| NT-FC3-V1 | `create_vectorstore_from_integration(profile=IntegrationProfile(), tenant_id="t")` | `IntegrationConfigurationError` |
| NT-FC3-V2 | same with `lab_opt_in` | in-memory store, tenant-scoped |
| NT-FC3-V3 | `lab_opt_in` without tenant | still raises (tenant rule intact) |
| NT-FC3-V4 | `create_vectorstore_manager` / `create_default_vectorstore_manager`, env and profile empty | raises |
| NT-FC3-N1 | `resolve_notification_settings()`, backend env unset | raises |
| NT-FC3-N2 | backend `slakc` (typo) with and without opt-in | raises in both cases |
| NT-FC3-N3 | `NotificationSettings()` | `TypeError` (no default backend) |
| NT-FC3-N4 | `create_notification_adapter` with an unsupported enum member | raises, never a log adapter |
| NT-FC3-N5 | `resolve_notification_adapter("lgo")` | raises |
| NT-FC3-N6 | `resolve_notification_adapter(None)` without opt-in | raises |
| NT-FC3-N7 | same with opt-in | log adapter (positive control) |
| NT-FC3-N8 | `LongRunningCoordinator.notify_escalation` (and the other three `notify_*`) with no injected adapter and no `notify_channel` | `IntegrationConfigurationError`; no delivery to `log` |
| NT-FC3-N9 | same with an unknown explicit channel (`lgo`) and no adapter | raises (a typo no longer degrades to `log`) |
| NT-FC3-N10 | same with an injected adapter | the injected adapter receives the message (control) |
| NT-FC3-N11 | `resolve_host_notification_adapter(env)` with `execution_mode=STRICT`, profile without `notification_channel` | raises |
| NT-FC3-N12 | same with `application_profile=PRODUCT` | raises |
| NT-FC3-N13 | LAB profile (not STRICT) without a channel binding; and a PRODUCT profile with a bound channel | LAB: log adapter through the derived opt-in; PRODUCT: the profile's adapter (controls) |
| NT-FC3-VIS1 | `TritonVisionServingAdapter(base_url="").detect(...)` | `VisionBackendUnconfiguredError`, no media read |
| NT-FC3-VIS2 | `HuggingFaceInferenceVisionAdapter(api_key="").detect(...)` | same |
| NT-FC3-VIS3 | Triton configured, mocked response with zero detections | empty `detections`, no `object.stub` |
| NT-FC3-VIS4 | HF configured, zero detections | empty `detections` |
| NT-FC3-VIS5 | stub `detect`, `segment`, `ocr_regions` | every result has `provenance=stub` |
| NT-FC3-VIS6 | explicit `fallback=StubVisionInferenceAdapter()` on an unconfigured Triton | result flagged `stub` (LAB opt-in is visible in the result) |
| NT-FC3-VIS7 | stub slug | not equal to `yolo_ultralytics`; registry lookup of `yolo_ultralytics` never returns the stub |
| NT-FC3-VIS8 | `as_extended_adapter(<non-extended adapter>)` | raises (no stub substitution) |
| NT-FC3-VIS9 | `run_modality_job` segment/ocr with a non-extended adapter | `ModalityJobResult.error` set, no stub segments |
| NT-FC3-VIS10 | `vision_detect` through the stub profile | tool output carries the provenance flag |
| NT-FC3-VIS11 | `VisionProfile(...).build_registry()` with no Triton/HF configuration | still builds (placeholder construction does not raise) |
| NT-FC3-INT1 | `create_inbound_verifier()`, env unset | `SecurityConfigurationError` |
| NT-FC3-INT2 | mode `slak` (typo) with and without opt-in | raises in both cases |
| NT-FC3-INT3 | mode `slack`, no secret | raises (verifier would be disabled) |
| NT-FC3-INT4 | mode `slack`, secret but verification not enabled | raises |
| NT-FC3-INT5 | mode `teams`, no token | raises |
| NT-FC3-INT6 | explicit mode `none` without opt-in | raises |
| NT-FC3-INT7 | `none` with opt-in | `NullInboundRequestVerifier` (positive control) |
| NT-FC3-INT8 | `InteractionIntakeService(verifier=None)` | raises |
| NT-FC3-INT9 | `wire_interaction_intake_service(env=STRICT, no verifier configured)` | raises |
| NT-FC3-INT10 | same with `application_profile=PRODUCT` | raises |
| NT-FC3-INT11 | PRODUCT env plus a LAB opt-in | raises (`LAB_OPT_IN_FORBIDDEN_IN_PRODUCT`) |
| NT-FC3-INT12 | probe-V3 analogue at the router level: `wire_interaction_intake_service` with a STRICT env and no verifier raises, so `create_interaction_intake_router` (`intergrax/runtime/interactions/router.py:15`) has no service to mount; positive control: a LAB-derived service mounted on a FastAPI app with `TestClient` accepts unsigned JSON | no unauthenticated intake route exists under STRICT; the LAB route works only through the derived opt-in |
| NT-FC3-GR1 | `create_guardrail_backend("no_such_slug")` | raises |
| NT-FC3-GR2 | same with opt-in | flagged stub |
| NT-FC3-GR3 | vendor SDK import forced to fail for `llm_guard` | raises without the pattern-engine opt-in |
| NT-FC3-GR4 | same with opt-in | result has `engine_degraded` |
| NT-FC3-GR5 | `resolve_guardrail_backend(env=STRICT)` with a degraded vendor | raises |

Count: 51 (E 6, V 4, N 13, VIS 11, INT 12, GR 5). The CONFIG-X negative tests moved to `T-IGX-043-F`. Unit tests use monkeypatched env and mocked `httpx`; no live vendor, Docker or network (C10/C15). `NT-FC3-INT12` needs no host fixture: it uses the wiring function, the router factory and `TestClient`; `NT-FC3-N11..N13` call the wiring helper directly, no host factory is built.

## 6. Regression scope (78 existing node IDs, each verified to exist at B and at H; `regression_nodes.txt` in this directory)

Grouped (full IDs in the file and in `FACT_CHECK.md` F-40):

- Vision, 15 nodes (3 of them pinned, see below): `test_remote_vision_adapters.py` (3), `test_vision_profile.py` (5), `test_model_inference_registry.py` (1), `test_modality_execution.py::test_thread_pool_offloads_heavy_slug_only`, `test_celery_modality_execution.py::test_run_modality_detect_job_uses_harness_registry`, `test_media_trust_boundary.py` (4).
- Embedding, 16 nodes (1 pinned; 2 default-dependent): `test_embedding_profile.py` (6), `test_p2_002_b3_embedding_runtime_binding.py` (6), `test_p2_002_b3_runtime_binding_extension.py` (2), `test_dep1_dependency_foundation.py::test_default_embedding_bootstrap_imports_provider_only_on_selection` and `test_reg_p_ollama_single_row.py::test_default_embedding_manager_builds_on_fresh_catalog` (both call `create_default_embedding_manager()` with no argument, see `test_dep1_dependency_foundation.py` lines 53-70 and `test_reg_p_ollama_single_row.py` lines 79-86: they depend on the default and must be updated).
- Vector store and RAG scope, 9 nodes (2 pinned): `test_integration_vectorstore_bootstrap.py` (4), `test_integration_vectorstore_scope.py` (3), `test_rag_scope.py` (2).
- Notifications, 5 nodes: `test_log_notification_factory.py` (1), `test_slack.py` and `test_teams.py` (runtime factory and long-running resolution, 2 each).
- Interactions, 14 nodes: the two `*_inbound_verifier_*_mode_uses_integration` tests (they pass a secret without the enable flag; they need `slack_verify_enabled=True` / `teams_verify_enabled=True`), debug intake tests (5), `test_interaction_intake_service.py` (2), `test_interaction_wiring_public_surface.py` (3: signature and source-token tests), `test_rag_authoritative_plugin_evidence.py` (2).
- Guardrail, 8 nodes: bridge (2), chained backend (1), contract (2), harness wiring (2), `test_nexus_loop_guardrail_wiring.py` (1).
- CONFIG-X, 11 nodes (existing gates, not modified by this ticket; they must stay green): FRZ-CFG-02/05/06/08 gates (4), `test_cx_q01`, `q16`, `q18` (3), constant-resolution (3), `test_cx_a_*` (1).
Total 15 + 16 + 9 + 5 + 14 + 8 + 11 = 78.

**Defect-pinning tests to invert** (they assert today's silent behaviour): `test_remote_vision_adapters.py::test_triton_without_url_falls_back_to_stub`, `::test_hf_without_key_falls_back`, `test_model_inference_registry.py::test_default_registry_runs_stub_vision_and_ml` (uses slug `yolo_ultralytics` for the stub), `test_embedding_profile.py::test_embedding_profile_from_env_defaults_to_ollama`, `test_integration_vectorstore_bootstrap.py::test_create_vectorstore_from_integration_falls_back_to_inmemory`, `test_integration_vectorstore_scope.py::test_explicit_tenant_binds_in_memory_fallback`. Not enumerated by name (UNVERIFIED): every test that calls a Tier-0 bootstrap or a host factory and relies on the lab defaults (static symbol census: 52 test files, listed in `census_files.txt`; the bounded run in step 3 of section 8 shows which of them fail).

## 7. Test commands (one process at a time; no diagnostic or broad suite; no live infrastructure)

Take the same lock as the docs batches before each run (`mkdir D:/Projekty/_wt/_locks/pytest.lock`, `rmdir` immediately after; AUTHORIZATION_REQUESTS A-03). Logs under `.tmp/session/eqo-a-remediation/IMPL/FC-3/`.

1. New negative tests (51), red-first at `B` and again at the fix commit, one command:
   `uv run pytest tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py tests/unit/runtime/notifications/test_fc3_notification_selector_fail_closed.py tests/unit/model_inference/test_fc3_vision_fail_closed.py tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py tests/unit/integrations/providers/llm_guardrail/test_fc3_guardrail_fail_closed.py -p no:cacheprovider`
2. Existing CONFIG-X directory, unchanged gates (six files): `uv run pytest tests/qualification/config_x -p no:cacheprovider`
3. Regression: one `uv run pytest <78 node IDs from regression_nodes.txt> -p no:cacheprovider` (after the inversions).
4. Governance wrapper before push: `uv run python -m scripts.governance.igx_gov ticket`.

Not part of the approvable test contract: the census run (pre-implementation diagnostic, authorization A-05). It is one process over the exact, bounded list of 52 files in `census_files.txt` (static symbol census over `tests/` and `applications/*/tests` at `B`, 52 files; the list is a lower bound because tests that reach the selectors only through a host factory are not found by symbol): `uv run pytest <the 52 files of census_files.txt> -p no:cacheprovider --maxfail=50`, run once after the leaf changes and before the caller sweep, non-certifying, log kept, no directory arguments.

## 8. Order of work inside the ticket

1. Preflight: ticket approval line; worktree outside the repository (`<parent>/_wt/intergrax/t-igx-043-c`, slug under 24 characters, full path under 60); read S1/V3 results; the Operator's answers to Q-FC3-01..11 are recorded in `OPEN_QUESTIONS.md`.
2. Foundation: `LabSelectorOptIn` in `intergrax/integrations/contracts/base.py`, new reason codes in `security_configuration.py`, `VisionBackendUnconfiguredError`; the posture helper `wiring_fails_closed(env)` is imported from `intergrax/applications/_shared/wiring_posture.py` (T-IGX-043-B merged first), not created here. Commit.
3. Census: after the leaf changes and before the caller sweep, run the bounded census of section 7 (A-05) to list the default-dependent tests; no gate is edited in this ticket (Part G is `T-IGX-043-F`).
4. Parts A, B (rag), C (notifications), D (vision), E (interactions), F (guardrail), each: negative tests red -> fix -> green; parts are file-disjoint after step 2 and may use separate worktrees only if the Operator authorizes more than one (default: sequential, one worktree).
5. Caller sweep: hosts, bootstraps, LAB surfaces; each LAB opt-in is added only where a LAB profile or an explicit lab entry point exists.
6. Invert/update pinned tests (section 6); run commands 1-3.
7. Quality gate (`igx_gov ticket`), diff reconciliation of ACT-N11 paths against section 4, hand-off note listing every `LabSelectorOptIn(` construction site for the auditor.
8. Set lifecycle `quality_gate`; the independent audit runs at the wave SHA (section 11); the implementer never records PASS.

## 9. ACT-N11 / IIAC / EBH-7 consequence

- Every P and C path matches `PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")` (`scripts/governance/iiac_common.py@H:22`). The IIAC activation record at H is `IN_FORCE`; its bound product tree is `corrective_implementation_sha = 969cbd964c679d563b02e55f065b452d46fdf90b` and `audit_execution_sha`/evidence publication is `a73d15ab341b30e9748aad03894f6faa75ad4e09`, a later commit with the same product tree (`IIAC_POLICY_ACTIVATION_RECORD.json@H:4,8,9`). Any product-path change after that product tree breaks the activation until the single RB-2 re-pin (PT-Q03). The WP-2 plan quotes `ae467c8c...` as the audit SHA; that value is not in the record at H (contradiction C-05).
- T paths are not product paths, but the test files of this ticket are not on the EBH-7 executable-identity allowlist, so they merge in the same train and before the final EBH-7 pin (DP-61). `T-IGX-043-F` (tests only) is in the same position.
- Overlap with gate tickets: `PT-G01` (EBH-3 scan extension, DP-20) and `PT-T01` own `scripts/maintenance/check_*`, `governance/architecture` and `tests/unit/architecture`; this ticket touches none of them, and after the split it touches no gate at all (`T-IGX-043-F` edits only `tests/qualification/config_x/*`, which none of those tickets lists). A4-025 (B-2 probe `R23` on `tests/qualification/config_x`) is not claimed.
- `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` lists product files by path (for example `intergrax/model_inference/contracts.py`, line 6021 at H); whether a changed or new file requires a matrix update is a question for the RB-2 session (UNVERIFIED), not for this ticket.

### 9.1 Merge order and ownership (stated now, not at first edit)

- Order inside the wave: T-IGX-043-B (FC-2, creates `wiring_posture.py`) merges first; T-IGX-043-A (FC-1) next; this ticket (FC-3) rebases onto both; `T-IGX-043-F` (gate) after this ticket; FC-4 (T-IGX-043-D) is file-disjoint and may merge at any point. The order is a rule of this plan; confirmed by the Operator as B, then A, then C, then F, with D at any time and W-A/W-B before A on `host_orchestration_environment_spec_builder.py`, `harness_host_runtime.py` and `nexus_loop.py` (`OPERATOR_ANSWERS_TRANSCRIPTION.md`, item 6).
- Shared `applications/*/host/factory.py` call sites: FC-1 edits (conditionally, up to eight) the call sites of `build_harness_host_runtime`; this ticket edits the call sites of `wire_interaction_intake_service` in seven of the same files (`dispute_sim`, `governed_contractor`, `intergrax_assistant`, `legal`, `local_workspace`, `poc_template`, `research`) and `lab_application/host/factory.py`. FC-3 additionally edits two calls in the same files: it adds the argument `notification_adapter=` to the `build_harness_host_runtime(...)` call that FC-1 also edits (FC-1 adds `human_decision_store` and the acknowledgement keyword), and replaces `notification_adapter=None` in the scheduler call. Rule: FC-1 merges first, FC-3 rebases; the edits touch different arguments of the same call expression, so a textual conflict is resolved by keeping both argument sets.
- Posture helper: owned by T-IGX-043-B (`wiring_fails_closed(env)`); FC-3 imports it and never defines it.
- `applications/_shared`: `PT-WB` lists this directory. This ticket edits exactly five files there (`interaction_wiring.py`, `environment_wiring.py`, `guardrail_runtime_bridge.py`, `harness_host_auxiliary_wiring.py`, `notification_wiring.py`) and for the wave owns those five; FC-1 owns `harness_host_runtime.py`, FC-2 owns the `*_wiring.py` files of capacity, queue, memory, event delivery and sandbox. `PT-WB` rebases onto FC-1/FC-2/FC-3 or coordinates through the orchestrator before touching those five files.
- Gates: `PT-G01` owns the EBH-3 scan extension and boundary gates; this ticket contains no gate and `T-IGX-043-F` touches only `tests/qualification/config_x/*`.
- Combined test blast radius (review finding X-4): the PRODUCT/STRICT test environments will meet the FC-2, FC-3 and FC-4 guards together, so after the first of these tickets merges the later ones repeat their census on the integrated wave branch and the first-merging ticket carries the opt-ins of the shared fixtures; the later ticket rebases onto them.

## 10. ARCH-FREEZE impact and what closes it

| Finding | Class | Closure condition (none is asserted by this plan) |
|---|---|---|
| A4-019 | BLOCKS (effective; unchanged) | the three selectors raise by default; `NT-FC3-E*`, `V*`, `N*` red at B and green at the wave SHA; independent audit at the exact SHA; notification leg: the settings and factory path refuses, the coordinator refuses an escalation without adapter and explicit channel, and the seven Tier-3 factories compose an explicit adapter; the paths without composition-time refusal named in section 3.3 are open items put to the Operator, not exceptions; the closure condition that the CONFIG-X scan flags vendor-slug defaults is met only by `T-IGX-043-F` (the Operator's personal ruling); FRZ-CFG-02/06/08 stay PASS with annotation (DP-06) until the Operator accepts the audit |
| A4-017 | CONDITIONS | `NT-FC3-VIS*` green; no result without provenance; stub slug no longer impersonates; audit at the wave SHA |
| A4-026 | CONDITIONS (BLOCKS if V3 shows the route reachable) | `NT-FC3-INT*` green incl. INT12; V3 result recorded by Campaign B alpha (`PT-B01` R03); audit; FRZ-SEC-04 stays OPEN (A/B split is a separate decision) |
| A4-032 part c | CONDITIONS | `NT-FC3-GR*` green; parts a and b are FC-2 |

## 11. Independent audit design

Exact SHA: the wave head `RW` frozen after all `FC-*`, `W-*` merges (node `PT-Q02`); any later product change invalidates it. Auditor: a separate non-implementing session that also did not review the implementation and is not an IIAC signing or activation session, using `/intergrax-independent-audit`; the implementer records no PASS. The auditor replays: (1) every `NT-FC3-*` at `B` (expected red) and at the wave SHA (expected green), plus one mutation per part (re-introduce the default, expect red); (2) the fallback census over the diff (`getenv`/`environ.get` defaults, `or "log"`, `or Null`, `fallback or Stub`, `except ImportError: return None` feeding a pattern engine) over the P and C files; (3) the existing CONFIG-X directory is green on the wave tree (the CONFIG-X delta audit belongs to `T-IGX-043-F`); (4) ACT-N11 reconciliation: the product-path files touched by the commits that carry the trailer `Ref: governance/tickets/T-IGX-043-C.md` (`git log --grep` with `--name-only`, restricted to `intergrax/ agents/ applications/`) equal section 4 P+C, and `git diff --name-only 969cbd964c679d563b02e55f065b452d46fdf90b..<wave SHA> -- intergrax agents applications` (the IIAC-bound product tree) contains no file of this ticket that is missing from the list; (5) every `LabSelectorOptIn(` construction site is in the hand-off list and sits in a LAB-only entry point; none is reachable from a PRODUCT/STRICT host factory; (6) docs and behaviour agree (section 4 D). Label: `PASS_SCOPED` only for the scope actually executed.

## 12. Risks

- R1 Test blast radius: dozens of tests rely on the lab defaults; red census first, no unbounded edits.
- R2 Host startup: every Tier-3 host that relied on defaults must pass posture or an adapter; mistakes surface as startup errors in LAB apps. Mitigation: derive the opt-in only from an explicit LAB profile. PRODUCT consequence (L-1, decided as drafted): `IntegrationProfile.legal_product()` and `research_product()` set no `notification_channel` (`intergrax/integrations/contracts/integration_profile.py@B:287-309`; only the lab presets set one, `:240,256,278`), and `legal_product()` is the profile of the legal, dispute_sim and governed_contractor manifests. Once this ticket lands, `resolve_host_notification_adapter(env)` makes those hosts refuse at composition in STRICT/PRODUCT until a channel binding exists; that is the intended fail-closed effect of the accepted Q-FC3-02. The binding is supplied by the host environment or profile owner, not by an edit in this ticket (the profile file is not in the write set; `PT-WB` territory).
- R3 `tests/qualification/pdc_x/reg_p/test_reg_p_ollama_single_row.py` is a PDC-X record test; editing it may touch PDC evidence (decided: edited in this ticket, Q-FC3-09 option 1; the reviewers are told).
- R4 Closed-world inventories (IIAC impact matrix) may reject changed or new paths (UNVERIFIED).
- R5 File overlap with `PT-WB` (`applications/_shared`, model profile classes), FC-1 (host factories, `harness_host_runtime.py`) and FC-2 (`wiring_posture.py`, `*_wiring.py`): ownership and merge order are stated in section 9.1; remaining risk is a rebase conflict, handled by the later ticket.
- R6 Provenance field adds a field to three tool-output models; consumers of `vision.*` tools (`applications/lab_application/host/tool_wiring.py`, `applications/legal_application/host/{tool_wiring,wiring}.py`, `tests/unit/runtime/modality/test_modality_profile.py`) must tolerate it (backward-compatible default).
- R7 Slack/Teams semantics: requiring `enabled` for mode `slack` changes lab behaviour that documentation calls "opt-in" (`PLATFORM_CONFIGURATION.md@H:863-867`).
- R8 Docs ahead of or behind code (decided, Q-FC3-10 option 1: follow-up one-line patch rides with the wave).
- R9 `notification_wiring.py` currently calls `resolve_notification_settings()` and relies on the `log` default when the profile selects another channel; once the backend is required the wiring must pass the explicit profile selection (section 3.3; trace UNVERIFIED).
- R10 Late-refusal-only paths for notifications (section 3.3): hosts and builds that this ticket does not edit are refused at the first escalation, not at composition; a composition-time refusal in `build_harness_host_runtime` (T-IGX-043-A's file) was not chosen (decided, O-08 option 1).

## 13. Rollback

Revert the wave merge (one commit train); the IIAC rebind must then be redone (revert alone does not restore activation, AR-41). The gate extension is not in this ticket, so no part of it reverts independently of the product fixes (AR-27 is split in `AUTHORIZATION_REQUESTS.md`; for `T-IGX-043-F` a revert is tests-only and restores the old gate verdict). If only Part E (hosts) proves unsafe, revert that part before the audit SHA is frozen, never after.

## 14. Effort (planning estimate, not measured)

Foundation S; A embedding M; B vector store S; C notifications L (explicit adapter through the host factories); D vision M; E interactions L (host sweep); F guardrail M; test sweep L; audit hand-off S. Total L to XL, against `L` in the master plan.

## 15. Contradictions with the master plan, WP-2 and the dossier

C-01 The master-plan row lists eight areas; the design needs 27 more product files (section 4 C) and `model_inference/contracts.py`. C-02 AR-27 rollback text ("docs/tests only") is wrong for a ticket that also changes product paths; a corrected row is in `AUTHORIZATION_REQUESTS.md`. C-03 WP-2 says "interaction intake tests of the three Tier-3 hosts"; seven Tier-3 hosts, `lab_application`, `debug/app.py` and `lab/organization_worker.py` build verifiers. C-04 The dossier describes A4-026 as unset or invalid mode; at B mode `slack`/`teams` without secret or enable flag is also a no-op, and `factory.py:98` (A4-019) is an unreachable tail. C-05 WP-2 quotes IIAC audit SHA `ae467c8c...`; the record at H says `a73d15ab3...`. C-06 `PDC_INDEX.md@H` still shows P9 as awaiting confirmation while the ladder and commit history show closure PRs merged. C-07 WP-2 expects target-state docs to ride with the code batch; the master plan schedules `PT-D07` (W2) before the wave.

## 16. Out-of-scope observations (recorded, not fixed here)

`legal_product()` selects the in-memory vector store explicitly; vendor parse defaults invent confidence 0.9 or full-frame boxes; OpenCV OCR placeholder text; `CeleryModalityExecutor` falls back to its delegate without a broker (`test_celery_modality_execution.py::test_celery_without_broker_falls_back_to_delegate`); Hugging Face host and default model literals (FRZ-CFG-05); `INTERGRAX_TRITON_URL`/`HUGGINGFACE_API_KEY` stay environment-supplied.
