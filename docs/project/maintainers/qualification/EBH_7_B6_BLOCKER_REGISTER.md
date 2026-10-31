# EBH-7 — B6 blocker register @ HEAD

**Open RPL concerns (semantic not QUALIFIED):** 23

| ID | Semantic | Verified | Defect | Operator path |
| --- | --- | --- | --- | --- |
| agent.effective_profile | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| application.composition_root | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| distributed.sqlite_kv | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| env.integration_slug | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| execution.integration_bound_resolution | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| integration.billing_meter | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (stripe) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.external_work | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | No providers/external_work category — cannot register alternate backend | OD-RPL-G2 |
| integration.managed_retrieval | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (openai) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.ml_inference_host | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (replicate) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.model_serving_runtime | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (ollama) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.vision_serving | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (triton) — alternate vendor sandbox required | OD-RPL-G2 |
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
