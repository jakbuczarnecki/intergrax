# EBH-7 — B6 blocker register @ HEAD

**Open RPL concerns (semantic not QUALIFIED):** 11

| ID | Semantic | Verified | Defect | Operator path |
| --- | --- | --- | --- | --- |
| agent.effective_profile | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| application.composition_root | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| integration.billing_meter | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (stripe) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.external_work | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | IntegrationCategory.external_work exists; zero catalog providers — dual provider | OD-RPL-G2 |
| integration.managed_retrieval | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (openai) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.ml_inference_host | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (replicate) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.model_serving_runtime | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (ollama) — alternate vendor sandbox required | OD-RPL-G2 |
| integration.vision_serving | BLOCKED | FIXED_SUBSTRATE_PENDING_OD | Single catalog provider (triton) — alternate vendor sandbox required | OD-RPL-G2 |
| memory.sqlite.conversational | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| prodq.colocated_runtime_adapter_isinstance | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |
| tool.marketplace_activation | PARTIAL | FACTORY_ONLY | Factory reachability — no distinct alternate backend proof | OD-SCOPE / qualify |

**TEN/XDM:** disposition EXACT + evidence SSOT (0 COMPOSED_PROXY); not a substitute for RPL B6 PASS.

**Not approved:** B6 PASS, OD-RPL-G2, OD-EBH7-SCOPE-RESIDUAL, OD-EBH7-08.
