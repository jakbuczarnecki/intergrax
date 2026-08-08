# SKILLS — closed §6 queues

**Parent hub:** [`SKILLS.md`](../SKILLS.md)

### 6.1ci Harness implementation queue — skill catalog expansion (closed)

**Purpose:** Tier-0 skill packs for agent and Tier-3 authors. **Closed 2026-06-08** — SK-EXP + SK-EXP2 + SK-EXP3 + SK-PRESET.1/2/3 **Done**. Residual: **SK-BRIDGE.*** (prompt/policy runtime merge). **Not** Band 3 business agents (K.1/K.2).

**Catalog:** **149** skills · **41** bundles — see [`architecture/SKILLS.md`](../architecture/SKILLS.md#first-party-catalog-149-skills--41-bundles).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **SK-DOC.1** | Docs | **Done** | Engine pipeline in `architecture/SKILLS.md`; §7.1.8 + Appendix J sync | Arch doc + §7.1.8 |
| 2 | **SK-BRIDGE.1** | Code | **Done** | `skill_bridge_wiring.py` — `skill_prompt_metadata()` | `test_skill_bridge_wiring.py` |
| 3 | **SK-BRIDGE.2** | Code | **Done** | `merge_skill_policy_fragments()` → `RuntimePolicyBundle` | `test_skill_bridge_wiring.py` |
| 4 | **SK-EXP-P0** | Code | **Done** | Wave P0 — 6 universal packs | `test_sk_exp_skill_bundles.py` |
| 5 | **SK-EXP-P1** | Code | **Done** | Wave P1 — 7 ops/dev/productivity packs | Same |
| 6 | **SK-EXP-P2** | Code | **Done** | Wave P2 — 5 domain/platform packs | Same |
| 7 | **SK-PRESET.1** | Code | **Done** | Tier-3 presets in `skill_wiring.py` | `lkw_skill_profile`, `dispute_skill_profile`, … |
| 8 | **SK-EXP2-P0** | Code | **Done** | Wave P0 — 6 platform-governance packs | `test_sk_exp2_skill_bundles.py` |
| 9 | **SK-EXP2-P1** | Code | **Done** | Wave P1 — 6 async/modality/eval packs | Same |
| 10 | **SK-EXP2-P2** | Code | **Done** | Wave P2 — 6 domain/platform extension packs | Same |
| 11 | **SK-PRESET.2** | Code | **Done** | SK-EXP2 presets in `skill_wiring.py` | `sandbox_skill_profile`, `hitl_skill_profile`, … |
| 12 | **SK-EXP3-P0** | Code | **Done** | Wave P0 — 7 platform governance packs | `test_sk_exp3_skill_bundles.py` |
| 13 | **SK-EXP3-P1** | Code | **Done** | Wave P1 — 7 eval/RAG/ops extension packs | Same |
| 14 | **SK-EXP3-P2** | Code | **Done** | Wave P2 — 6 domain/productivity packs | Same |
| 15 | **SK-PRESET.3** | Code | **Done** | SK-EXP3 presets in `skill_wiring.py` | `cost_skill_profile`, `metrics_skill_profile`, … |
| 16 | **SK-EXP4-P0** | Code | **Done** | Wave P0 — 11 platform/integration packs | `test_sk_exp4_skill_bundles.py` |
| 17 | **SK-EXP4-P1** | Code | **Done** | Wave P1 — 10 eval/harness/ops extension packs | Same |
| 18 | **SK-EXP4-P2** | Code | **Done** | Wave P2 — 9 domain/destructive-admin packs | Same |
| 19 | **SK-PRESET.4** | Code | **Done** | SK-EXP4 presets in `skill_wiring.py` | `catalog_skill_profile`, `interaction_skill_profile`, … |
| 20 | **SK-EXP5** | Code | **Done** | 50 compositional vertical packs (no new bundles) | `test_sk_exp5_skill_bundles.py` |
| 21 | **SK-PRESET.5** | Code | **Done** | Vertical presets: oncall, legal_ops, research_lab, … | `skill_wiring.py` |

**Suggested PR order (complete):** SK-EXP → SK-EXP5 → SK-PRESET.1–5. **SK-BRIDGE.*** optional follow-up.

**Explicitly excluded:** K.1, K.2, new Tier-2 agents, workflow-sized fake tools, unvalidated filesystem skill discovery.

#### SK-EXP — Proposed skill register (18 packs)

Priority for **platform users** (agent authors, Tier-3 hosts, extension authors). Each row = one `SkillManifest` + `SkillPlugin` bundle entry (new bundle or extend existing).

**Wave P0 — Universal reuse (ship first)**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK-P0.1 | `rag.hybrid_qa` | `rag` | `rag.retrieve`, `rag.get_document`, `memory.read` | Default Q&A over index + session — LKW, Legal, Research, IAA |
| SK-P0.2 | `rag.document_ingest` | `rag` | `document.parse`, `rag.ingest_document`, `rag.describe_collection` | Ingestion pipeline without per-agent tool lists |
| SK-P0.3 | `research.web_evidence` | `research` | `websearch.query`, `websearch.read_url`, `websearch.fetch_batch` | Web-grounded evidence pack (complements `literature_scan`) |
| SK-P0.4 | `workspace.authoring` | `workspace` | `workspace.read_file`, `workspace.write_file`, `workspace.search`, `memory.write` | Shadow workspace drafts — LKW synthesizer, coding assistants |
| SK-P0.5 | `memory.task_scratchpad` | `memory` | `memory.read`, `memory.write`, `memory.list_keys` | Cross-step task KV — dispute sim, multi-turn agents |
| SK-P0.6 | `knowledge.wiki_navigator` | `knowledge` | `knowledge.search`, `knowledge.get_page`, `confluence.search` | Internal docs + wiki — complements `openai_strict` |

**Wave P1 — Ops, dev, collaboration**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK-P1.1 | `ops.trace_debug` | `ops` | `observability.query_traces`, `logs.search`, `errors.capture` | Harness/developer incident debugging |
| SK-P1.2 | `ops.incident_dispatch` | `ops` | `pagerduty.trigger_incident`, `notify.send`, `logs.search` | On-call + notification wiring |
| SK-P1.3 | `ops.security_audit` | `ops` | `security.scan`, `workspace.search`, `notify.send` | CI/security gate for agent workspaces |
| SK-P1.4 | `ops.workflow_runner` | `ops` | `workflow.trigger`, `workflow.poll`, `workflow.fetch_logs` | Batch eval / RAG refresh orchestration |
| SK-P1.5 | `dev.issue_triage` | `dev` | `issues.search`, `issues.get_issue`, `issues.add_comment`, `notify.send` | Provider-agnostic Jira/GitLab triage |
| SK-P1.6 | `browser.research_fetch` | `browser` | `browser.fetch_page`, `websearch.read_url`, `document.parse_preview` | JS-heavy pages + extraction |
| SK-P1.7 | `collaboration.outreach` | `collaboration` | `collaboration.send_mail`, `collaboration.list_messages`, `collaboration.get_message` | Email drafting / thread context |

**Wave P2 — Domain depth + platform hub**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK-P2.1 | `legal.clause_compare` | `legal` | `rag.retrieve`, `workspace.write_file`, `websearch.query` | `requires_skills`: `legal.contract_review` |
| SK-P2.2 | `legal.case_research` | `legal` | `rag.retrieve`, `knowledge.search`, `websearch.query` | Dispute sim + regulatory lookup |
| SK-P2.3 | `research.citation_synthesis` | `research` | `rag.retrieve`, `websearch.query`, `workspace.write_file` | SummaryAgent / report pipelines |
| SK-P2.4 | `data.sql_analyst` | `data` | `database.query`, `database.describe_schema`, `workspace.write_file` | Structured data Q&A |
| SK-P2.5 | `platform.concierge` | `platform` | `rag.retrieve`, `websearch.query`, `memory.read`, `skill.resolve` | `intergrax_assistant` hub — introspection + retrieval |

**Deferred to SK-EXP2 (now shipped):** `hitl.approval_gate`, `graph.entity_explorer`, `sandbox.code_exec` — see §6.1cj2 below.

**ADR:** no ADR for doc-only SK-DOC.1. New bundles follow existing Phase R pattern — **no ADR** unless a skill models a multi-step workflow as one tool (forbidden). SK-BRIDGE.* may need `docs/project/technical/adr/` entry if context merge semantics change Nexus contracts.

#### SK-EXP2 — Proposed skill register (18 packs)

Second wave after SK-EXP: platform governance, async/modality, and domain extensions. **9 new bundles** + **7 extended bundles**.

**Wave P0 — Platform governance (ship first)**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK2-P0.1 | `rag.index_admin` | `rag` | `rag.list_collections`, `rag.describe_collection`, `rag.check_index_status`, `rag.list_documents` | Operator index introspection without destructive ops |
| SK2-P0.2 | `rag.collection_lifecycle` | `rag` | `rag.search_by_metadata`, `rag.delete_documents`, `rag.purge_collection` | HIGH-risk controlled purge — admin hosts only |
| SK2-P0.3 | `sandbox.code_exec` | `sandbox` | `sandbox.exec`, `workspace.read_file`, `workspace.write_file` | Coding agents with isolated exec |
| SK2-P0.4 | `hitl.approval_gate` | `hitl` | `hitl.list_pending`, `hitl.submit_response`, `hitl.get_decision`, `notify.send` | Governed HITL without per-agent wiring |
| SK2-P0.5 | `graph.entity_explorer` | `graph` | `graph.run_query`, `graph.get_node`, `rag.retrieve` | Knowledge graph + RAG grounding |
| SK2-P0.6 | `storage.artifact_sync` | `storage` | `storage.get`, `storage.put`, `workspace.export_artifact`, `workspace.import_artifact` | Durable artifacts across runs |

**Wave P1 — Async, cache, modality, eval**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK2-P1.1 | `message_bus.async_runner` | `message_bus` | `message_bus.enqueue`, `message_bus.get_status`, `message_bus.get_result` | Long-running tasks off sync Nexus loop |
| SK2-P1.2 | `cache.session_cache` | `cache` | `cache.get`, `cache.set`, `memory.read` | Session acceleration with memory fallback |
| SK2-P1.3 | `eval.score_logger` | `eval` | `braintrust.log_eval`, `observability.query_traces` | Eval harness score + trace correlation |
| SK2-P1.4 | `modality.speech_io` | `modality` | `speech.transcribe`, `speech.synthesize` | Voice agents without vendor SDK in Tier-2 |
| SK2-P1.5 | `modality.vision_ocr` | `modality` | `vision.ocr_regions`, `vision.detect`, `document.parse_preview` | Multimodal OCR before ingest |
| SK2-P1.6 | `notify.scheduled_alerts` | `notify` | `notify.schedule`, `notify.list_scheduled`, `notify.cancel_scheduled`, `notify.send` | Deferred alerts for long workflows |

**Wave P2 — Domain and platform extensions**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK2-P2.1 | `collaboration.calendar` | `collaboration` | `collaboration.list_calendar`, `collaboration.create_event`, `collaboration.get_user` | Meeting scheduling complement to outreach |
| SK2-P2.2 | `platform.secrets_flags` | `platform` | `platform.get_secret`, `platform.evaluate_feature_flag` | Governed secrets/flags for trusted hosts |
| SK2-P2.3 | `platform.cicd_inspector` | `platform` | `platform.list_workflow_runs`, `platform.get_workflow_run`, `platform.list_check_suites` | CI visibility for release automation |
| SK2-P2.4 | `data.records_query` | `data` | `records.query`, `records.get`, `records.describe_collection` | NoSQL complement to `data.sql_analyst` |
| SK2-P2.5 | `dev.issue_creator` | `dev` | `issues.create_issue`, `issues.search`, `notify.send` | Discovery-to-ticket loop |
| SK2-P2.6 | `memory.session_cleanup` | `memory` | `memory.list_keys`, `memory.delete_key`, `memory.read` | Session hygiene for long multi-turn runs |

#### SK-EXP2 — Master register (shipped)

| ID | skill_id | Bundle | Status |
|----|----------|--------|--------|
| SK2-P0.1 | `rag.index_admin` | `rag` | **Done** |
| SK2-P0.2 | `rag.collection_lifecycle` | `rag` | **Done** |
| SK2-P0.3 | `sandbox.code_exec` | `sandbox` | **Done** |
| SK2-P0.4 | `hitl.approval_gate` | `hitl` | **Done** |
| SK2-P0.5 | `graph.entity_explorer` | `graph` | **Done** |
| SK2-P0.6 | `storage.artifact_sync` | `storage` | **Done** |
| SK2-P1.1 | `message_bus.async_runner` | `message_bus` | **Done** |
| SK2-P1.2 | `cache.session_cache` | `cache` | **Done** |
| SK2-P1.3 | `eval.score_logger` | `eval` | **Done** |
| SK2-P1.4 | `modality.speech_io` | `modality` | **Done** |
| SK2-P1.5 | `modality.vision_ocr` | `modality` | **Done** |
| SK2-P1.6 | `notify.scheduled_alerts` | `notify` | **Done** |
| SK2-P2.1 | `collaboration.calendar` | `collaboration` | **Done** |
| SK2-P2.2 | `platform.secrets_flags` | `platform` | **Done** |
| SK2-P2.3 | `platform.cicd_inspector` | `platform` | **Done** |
| SK2-P2.4 | `data.records_query` | `data` | **Done** |
| SK2-P2.5 | `dev.issue_creator` | `dev` | **Done** |
| SK2-P2.6 | `memory.session_cleanup` | `memory` | **Done** |

#### SK-EXP3 — Proposed skill register (20 packs)

Third wave: platform governance (cost, identity, health, context), eval/RAG depth, and product ops. **9 new bundles** + **9 extended bundles**.

**Wave P0 — Platform governance**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK3-P0.1 | `cost.budget_guardian` | `cost` | `cost.check_quota`, `cost.get_run_budget`, `cost.forecast_spend` | Run budget enforcement |
| SK3-P0.2 | `identity.access_checker` | `identity` | `identity.verify_token`, `identity.get_user`, `identity.list_tenants` | Multi-tenant access checks |
| SK3-P0.3 | `health.integration_probe` | `health` | `health.check_integration`, `health.check_profile`, `health.check_relational_store` | Operator health probes |
| SK3-P0.4 | `context.token_planner` | `context` | `context.estimate_tokens`, `context.summarize`, `memory.read` | Context budget planning |
| SK3-P0.5 | `memory.ltm_curator` | `memory` | `ltm.write_fact`, `ltm.search`, `memory.read` | LTM fact curation |
| SK3-P0.6 | `agent.roster_introspect` | `agent` | `agent.list_agents`, `agent.get_contract`, `skill.resolve` | Hub agent introspection |
| SK3-P0.7 | `vector_store.admin` | `vector_store` | `vector_store.list_collections`, `vector_store.count`, `vector_store.health` | Vector backend admin |

**Wave P1 — Eval, RAG, ops extensions**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK3-P1.1 | `eval.trajectory_judge` | `eval` | `eval.judge`, `eval.record_observation`, `eval.trajectory` | Trajectory-level eval |
| SK3-P1.2 | `eval.release_compare` | `eval` | `eval.compare_releases`, `eval.summarize_release`, `eval.export_observations` | Release regression compare |
| SK3-P1.3 | `rag.retrieval_tuner` | `rag` | `rag.preview_retrieval`, `rag.rerank`, `rag.retrieve` | Retrieval tuning loop |
| SK3-P1.4 | `workspace.snapshot_manager` | `workspace` | `workspace.snapshot`, `workspace.list_files`, `workspace.delete_file` | Workspace lifecycle |
| SK3-P1.5 | `message_bus.task_admin` | `message_bus` | `message_bus.list_tasks`, `message_bus.cancel`, `message_bus.purge_completed` | Queue administration |
| SK3-P1.6 | `ops.workflow_admin` | `ops` | `workflow.list_runs`, `workflow.cancel_run`, `workflow.fetch_logs` | Workflow run admin |
| SK3-P1.7 | `hitl.queue_manager` | `hitl` | `hitl.list_for_task`, `hitl.summarize_queue`, `hitl.list_pending` | HITL queue visibility |

**Wave P2 — Domain and product ops**

| ID | skill_id | Bundle | `tool_ids` (core) | Value |
|----|----------|--------|-------------------|-------|
| SK3-P2.1 | `crm.account_lookup` | `crm` | `crm.get_account`, `crm.list_contacts`, `crm.list_tickets` | CRM account research |
| SK3-P2.2 | `billing.usage_tracker` | `billing` | `billing.list_usage`, `billing.record_usage`, `harness.get_run_cost` | Usage metering |
| SK3-P2.3 | `metrics.run_observer` | `metrics` | `metrics.query_instant`, `metrics.query_range`, `observability.query_traces` | Metrics + trace join |
| SK3-P2.4 | `dev.issue_updater` | `dev` | `issues.update_issue`, `issues.add_comment`, `issues.get_issue` | Issue update loop |
| SK3-P2.5 | `collaboration.thread_reply` | `collaboration` | `collaboration.reply_message`, `collaboration.get_message`, `collaboration.list_messages` | Email thread follow-up |
| SK3-P2.6 | `ops.findings_review` | `ops` | `security.summarize_findings`, `security.scan`, `notify.send` | Security findings triage |

#### SK-EXP3 — Master register (shipped)

| ID | skill_id | Bundle | Status |
|----|----------|--------|--------|
| SK3-P0.1 | `cost.budget_guardian` | `cost` | **Done** |
| SK3-P0.2 | `identity.access_checker` | `identity` | **Done** |
| SK3-P0.3 | `health.integration_probe` | `health` | **Done** |
| SK3-P0.4 | `context.token_planner` | `context` | **Done** |
| SK3-P0.5 | `memory.ltm_curator` | `memory` | **Done** |
| SK3-P0.6 | `agent.roster_introspect` | `agent` | **Done** |
| SK3-P0.7 | `vector_store.admin` | `vector_store` | **Done** |
| SK3-P1.1 | `eval.trajectory_judge` | `eval` | **Done** |
| SK3-P1.2 | `eval.release_compare` | `eval` | **Done** |
| SK3-P1.3 | `rag.retrieval_tuner` | `rag` | **Done** |
| SK3-P1.4 | `workspace.snapshot_manager` | `workspace` | **Done** |
| SK3-P1.5 | `message_bus.task_admin` | `message_bus` | **Done** |
| SK3-P1.6 | `ops.workflow_admin` | `ops` | **Done** |
| SK3-P1.7 | `hitl.queue_manager` | `hitl` | **Done** |
| SK3-P2.1 | `crm.account_lookup` | `crm` | **Done** |
| SK3-P2.2 | `billing.usage_tracker` | `billing` | **Done** |
| SK3-P2.3 | `metrics.run_observer` | `metrics` | **Done** |
| SK3-P2.4 | `dev.issue_updater` | `dev` | **Done** |
| SK3-P2.5 | `collaboration.thread_reply` | `collaboration` | **Done** |
| SK3-P2.6 | `ops.findings_review` | `ops` | **Done** |

#### SK-EXP4 — Proposed skill register (30 packs)

Fourth wave: close remaining tool-catalog gaps — catalog introspection, interaction, vendor-specific trackers (Jira/GitLab), harness run ops, destructive admin paths. **10 new bundles** + **15 extended bundles**.

**Wave P0 — Platform integration (11 packs)**

| ID | skill_id | Bundle | Value |
|----|----------|--------|-------|
| SK4-P0.1 | `catalog.tool_introspect` | `catalog` | Tool catalog discovery + skill.resolve |
| SK4-P0.2 | `cloud_platform.resolver` | `cloud_platform` | Cloud endpoint resolution |
| SK4-P0.3 | `code.runner` | `code` | Script/code exec with sandbox listing |
| SK4-P0.4 | `filesystem.local_io` | `filesystem` | Trusted-host local filesystem IO |
| SK4-P0.5 | `http.api_client` | `http` | Outbound HTTP with observability |
| SK4-P0.6 | `interaction.session_handler` | `interaction` | User session list/history/reply |
| SK4-P0.7 | `interaction.input_capture` | `interaction` | Last input capture + memory write |
| SK4-P0.8 | `jira.task_navigator` | `jira` | Native Jira task navigation |
| SK4-P0.9 | `gitlab.issue_creator` | `gitlab` | GitLab issue create loop |
| SK4-P0.10 | `ml.explain_predict` | `ml` | ML predict + explain |
| SK4-P0.11 | `openai.vector_admin` | `openai` | OpenAI vector store lifecycle |

**Wave P1 — Harness, eval, ops depth (10 packs)**

| ID | skill_id | Bundle | Value |
|----|----------|--------|-------|
| SK4-P1.1 | `browser.interactive_run` | `browser` | Interactive browser automation |
| SK4-P1.2 | `cache.key_admin` | `cache` | Cache key list/delete admin |
| SK4-P1.3 | `knowledge.confluence_navigator` | `knowledge` | Confluence page deep nav |
| SK4-P1.4 | `eval.observation_browser` | `eval` | Eval observation listing |
| SK4-P1.5 | `harness.run_comparator` | `harness` | Compare harness runs |
| SK4-P1.6 | `harness.run_exporter` | `harness` | Export run bundles |
| SK4-P1.7 | `health.full_stack_probe` | `health` | Multi-backend health sweep |
| SK4-P1.8 | `ops.log_tail` | `ops` | Live log tail + search |
| SK4-P1.9 | `memory.semantic_search` | `memory` | Semantic memory + LTM search |
| SK4-P1.10 | `notify.batch_dispatch` | `notify` | Batch + due notification dispatch |

**Wave P2 — Destructive admin + domain (9 packs)**

| ID | skill_id | Bundle | Value |
|----|----------|--------|-------|
| SK4-P2.1 | `data.sql_mutator` | `data` | SQL execute (mutations) |
| SK4-P2.2 | `data.records_admin` | `data` | Records put/delete/count |
| SK4-P2.3 | `ops.incident_ack` | `ops` | PagerDuty acknowledge loop |
| SK4-P2.4 | `platform.secret_admin` | `platform` | Secret put/delete admin |
| SK4-P2.5 | `platform.workflow_cancel` | `platform` | CI workflow cancellation |
| SK4-P2.6 | `storage.object_lifecycle` | `storage` | Object delete/presigned URL |
| SK4-P2.7 | `vector_store.purge` | `vector_store` | Vector delete/purge |
| SK4-P2.8 | `modality.vision_segment` | `modality` | Vision segmentation pipeline |
| SK4-P2.9 | `research.web_cache_admin` | `research` | Web search cache invalidation |

#### SK-EXP4 — Master register (shipped)

All 30 rows **Done** — see [`architecture/SKILLS.md`](../architecture/SKILLS.md#first-party-catalog-99-skills--41-bundles).

#### SK-EXP — Master register (shipped)

| ID | skill_id | Bundle | Status |
|----|----------|--------|--------|
| SK-P0.1 | `rag.hybrid_qa` | `rag` | **Done** |
| SK-P0.2 | `rag.document_ingest` | `rag` | **Done** |
| SK-P0.3 | `research.web_evidence` | `research` | **Done** |
| SK-P0.4 | `workspace.authoring` | `workspace` | **Done** |
| SK-P0.5 | `memory.task_scratchpad` | `memory` | **Done** |
| SK-P0.6 | `knowledge.wiki_navigator` | `knowledge` | **Done** |
| SK-P1.1 | `ops.trace_debug` | `ops` | **Done** |
| SK-P1.2 | `ops.incident_dispatch` | `ops` | **Done** |
| SK-P1.3 | `ops.security_audit` | `ops` | **Done** |
| SK-P1.4 | `ops.workflow_runner` | `ops` | **Done** |
| SK-P1.5 | `dev.issue_triage` | `dev` | **Done** |
| SK-P1.6 | `browser.research_fetch` | `browser` | **Done** |
| SK-P1.7 | `collaboration.outreach` | `collaboration` | **Done** |
| SK-P2.1 | `legal.clause_compare` | `legal` | **Done** |
| SK-P2.2 | `legal.case_research` | `legal` | **Done** |
| SK-P2.3 | `research.citation_synthesis` | `research` | **Done** |
| SK-P2.4 | `data.sql_analyst` | `data` | **Done** |
| SK-P2.5 | `platform.concierge` | `platform` | **Done** |

#### SK-EXP — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-08 | SK-DOC.1 | Engine pipeline documented; §7.1.8 + Appendix J contract fix |
| 2026-06-08 | SK-EXP-P0–P2, SK-PRESET.1 | 18 skill packs + 9 new bundles; 31 total skills; `test_sk_exp_skill_bundles.py` |
| 2026-06-08 | SK-DOC.2 | Per-skill `USAGE.md` (31 files) + bundle indexes; gate `test_skill_usage_docs.py`; scaffold emits skill USAGE template |
| 2026-06-08 | SK-EXP2-P0–P2, SK-PRESET.2 | 18 skill packs + 9 new bundles; 49 total skills; `test_sk_exp2_skill_bundles.py`; SK-EXP2 presets in `skill_wiring.py` |
| 2026-06-08 | SK-DOC.3 | Per-skill `USAGE.md` for SK-EXP2 (18 files); gate count 49 |
| 2026-06-08 | SK-EXP3-P0–P2, SK-PRESET.3 | 20 skill packs + 9 new bundles; 69 total skills; `test_sk_exp3_skill_bundles.py`; SK-EXP3 presets in `skill_wiring.py` |
| 2026-06-08 | SK-DOC.4 | Per-skill `USAGE.md` for SK-EXP3 (20 files); gate count 69 |
| 2026-06-08 | SK-EXP4-P0–P2, SK-PRESET.4 | 30 skill packs + 10 new bundles; 99 total skills; `test_sk_exp4_skill_bundles.py` |
| 2026-06-08 | SK-DOC.5 | Per-skill `USAGE.md` for SK-EXP4 (30 files); gate count 99 |
| 2026-06-08 | SK-EXP5, SK-PRESET.5 | 50 compositional packs; 149 total skills; vertical presets |
| 2026-06-08 | SK-DOC.6 | Per-skill `USAGE.md` for SK-EXP5 (50 files); gate count 149 |

#### SK-EXP5 — Compositional register (50 packs, shipped)

Third wave after tool-coverage saturation (185/190 tools). Value = **vertical compositions** and **operator specializations** across 25 extended bundles — no new bundles.

| Wave | Count | Examples |
|------|-------|----------|
| P0 Domain depth | 18 | `legal.redline_draft`, `rag.semantic_qa`, `research.deep_dive`, `workspace.artifact_exporter` |
| P1 Ops/platform | 16 | `ops.oncall_runbook`, `ops.postmortem_writer`, `platform.deploy_inspector`, `hitl.escalation_router` |
| P2 Dev/data/sandbox | 16 | `dev.sprint_planner`, `data.pipeline_probe`, `sandbox.refactor_loop`, `agent.capability_mapper` |

All 50 rows **Done** — full skill_id list in `scripts/maintenance/scaffold_sk_exp5.py`.

---

### 6.1c Harness implementation queue — tools/skills closeout (closed)

**Purpose:** Single ordered list for **Phase TS** (Band 2k). **Closed 2026-06-02** — all TS rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **TS-DOC.1–2** | Docs | **Done** | Appendix J + cross-refs | Author map complete |
| 2 | **TS-1** | Code | **Done** | `catalog_runtime_bridge` + `RuntimeConfig.skill_profile` | `test_catalog_runtime_bridge.py` |
| 3 | **TS-2** | Code | **Done** | Harness host `resolve_llm_adapter` wiring | `test_harness_host_runtime_llm.py` |
| 4 | **TS-3** | Code | **Done** | `SkillResolverProtocol` | skill resolver tests green |

**Suggested PR order (complete):** TS-1 → TS-2 → TS-3 → TS-DOC.*.

**Explicitly excluded:** K.1, K.2, new product tools/skills, business agent packs — [§6.3a](#63a-business-backlog-register-consolidated).### 6.1aa Harness implementation queue — memory platform (closed)

**Purpose:** Phase MEM execution queue — **closed 2026-06-02** (48/48 Done). Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **MEM-1.1–MEM-1.4** | Code | **Done** | H-APP `MemoryProfile` + `ContextProfile.budget` + SQLite session → `RuntimeConfig` | MEM-1.5 gate test green |
| 2 | **MEM-2.1–MEM-2.3** | Code | **Done** | `SQLiteUserProfileStore` + bundle wiring + unit tests | LTM survives restart on sqlite profile |
| 3 | **MEM-1.6** | Docs/status | **Done** | H-APP.4.3 → **Done** | Bridge complete |
| 4 | **MEM-4.1–MEM-4.3** | Test | **Done** | Session + LTM + full-stack memory gates | acceptance/integration green |
| 5 | **MEM-5.1–MEM-5.2** | Test/Docs | **Done** | `engine_history_layer` tests + compression docs | unit + guide |
| 6 | **MEM-3.1–MEM-3.3** | Code | **Done** | Memory store plugin EP + reference fixture | bootstrap + gate |
| 7 | **MEM-0.3–MEM-DOC.*** | Docs | **Done** | Author cookbooks + Appendix G sync | guide updated |
| 8 | **MEM-6.*–MEM-7.*** | Code | **Done** | Retention enforcement + memory hooks | P2 after P0/P1 |
| 9 | **MEM-8.*–MEM-9.*** | RFC | **Done (RFC)** | Product memory layer + entity graph design | §6.3 gate for implementation |

**Suggested PR order:** See [Phase MEM — Suggested PR order](#mem--paydown-log).

**Explicitly excluded:** K.1, K.2, Mem0 SaaS product, entity graph ship (RFC only), business agent memory.

---
