# EBH-7 — B6 blocker register @ HEAD

**Open RPL concerns (semantic not QUALIFIED):** 51

| ID | Semantic | Verified | Defect | Operator path |
| --- | --- | --- | --- | --- |
| agent.effective_profile | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| application.composition_root | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| distributed.sqlite_kv | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| env.integration_slug | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| execution.integration_bound_resolution | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| integration.billing_meter | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (stripe) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.browser_automation | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.ci_cd | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.cloud_platform | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.collaboration_suite | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.conversation_channel | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.crm | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.document_parser | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.document_store | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.embedding_provider | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.external_work | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | No providers/external_work category — cannot register alternate backend | OD-RPL-G2 |
| integration.feature_flag | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.graph_store | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.identity_provider | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.issue_tracker | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.llm_guardrail | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.managed_retrieval | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (openai) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.message_bus | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.ml_inference_host | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (replicate) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.model_serving_runtime | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (ollama) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.notification_channel | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.object_storage | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.observability_backend | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.relational_store | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.rerank_provider | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.sandbox_host | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.search_provider | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.secrets_store | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.security_scanner | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.speech_provider | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.vector_store | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.vision_serving | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (triton) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.wiki_knowledge | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| integration.workflow_orchestrator | PARTIAL | DUAL_SLUG_REGISTRY_ONLY | Package C v2 dual-slug integration executed; RPL-C/D/E operational equivalence n | OD-SCOPE / qualify |
| llm.chat_completion | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| llm.embedding | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| llm.fallback_policy | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| llm.model_routing_qualifier | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| llm.rerank | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| memory.sqlite.conversational | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| plugin.integration_catalog | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| plugin.llm_adapter_registry | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| prodq.colocated_runtime_adapter_isinstance | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| speech.provider | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| tool.marketplace_activation | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| tool.observability_backend | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |

**TEN/XDM:** disposition EXACT + evidence SSOT (0 COMPOSED_PROXY); not a substitute for RPL B6 PASS.

**Not approved:** B6 PASS, OD-RPL-G2, OD-EBH7-SCOPE-RESIDUAL, OD-EBH7-08.
