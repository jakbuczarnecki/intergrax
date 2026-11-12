# PDC-X — red-gate diagnosis (wave A, PDC-P5-REG)

**Ticket:** T-IGX-036-G (P4 deliverable 6) · **Head diagnosed:** `dcdb35601305a2ad9a3ee22894b26c29800a9640` (`origin/development`) · **Method:** three read-only subagent investigations, each re-running only its failing node IDs once, then bisecting the change history.

The PDC-P2-09 replay found 13 failing tests that already fail on `development`. **12 of them are stale gates**: the protected invariants still hold, and the tests did not follow later, deliberate refactors. **One is a real product regression.**

| Requirement | Suite | Tests | Verdict | Introducing change | Fix scope | Batch |
| --- | --- | ---: | --- | --- | --- | --- |
| PDC-D3-R024 | `tests/qualification/ebh_5` | 4 | STALE_GATE | `f913944ff` (EBH-7 closure wave: RPL registry 5 → 56) and `d4937889f` (EBH-7 terminal: 8 concerns reclassified INTERNAL / FIXED_SUBSTRATE) | Tests only (3 files); optional MD summary row. The 8 IDs must stay in the registry, because ebh_7 depends on them | B-REG-T |
| PDC-D4-R020 | `tests/qualification/governance/gr12` | 4 | STALE_GATE | `49911bb3a` (P2-C: STRICT host requires revision-bound contract authority) | Test support `tests/unit/applications/task_control_product_host_test_support.py` must pass `agent_contract_authority`. Production composition already does (`governed_contractor_application/host/factory.py`) | B-REG-T |
| PDC-D6-R028 | `tests/qualification/governance/strategy` (GR-10) | 4 | STALE_GATE | `716ed746f` (STATE-X ACP checkpoint trust boundary) and `6700ff756` (EBH6-BLK-L9-GOV-01: PRE_MODEL wiring moved into governance) | Tests only. Text and AST gates are re-pointed to the new structure; pre-model enforcement and evidence persistence remain fail-closed | B-REG-T |
| PDC-D1-R022 | `tests/qualification/plug_03` (inner `tests/unit/rag/test_rag_plugin_discovery.py`) | 1 | **REGRESSION** | `7aa9cfa18` (EBH-7 gap closure: `embedding_provider/ollama` registers a second catalog row with slug `ollama`, which `model_serving_runtime/ollama` already registered) | **Product path** `intergrax/integrations/providers/embedding_provider/ollama/` (restore augment registration, or merge into the single `ollama` row) | B-REG-P |

## Impact of the regression

`register_default_integrations(preset="full")` raises `ValueError: Integration slug 'ollama' is already registered.` on a fresh catalog. Every default `create_default_embedding_manager()` therefore fails, which means any default semantic chunking or embedding bootstrap fails too.

The subagent also saw a separate circular-import `ImportError` in `runtime/integrations/observability.py` when `register_default_integrations` is the first import. That issue is noted here and has not been triaged.

## Sequencing consequence

- **B-REG-T** changes tests only. It does not touch IIAC ACT-N11 product paths, so it may merge before MCP-SERVER-X P8.
- **B-REG-P** changes `intergrax/`. Merging it invalidates the IIAC activation that MCP-SERVER-X P7/P8 rely on (ACT-N11). OD-PDC-SEQ-01 therefore holds it until MCP-SERVER-X P8, unless the Operator decides to fix it earlier through a rebind.

## Process note

None of `tests/qualification/ebh_5`, `governance` or `plug_03` runs in a CI workflow. That is why these failures were not noticed. PDC-P6 should add them to a CI tier, or to the ENT-AUDIT check manifest.
