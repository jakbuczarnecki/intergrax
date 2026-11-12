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

## Wave B results (T-IGX-036-I)

| Suite | Before | After batch B-REG-T |
| --- | --- | --- |
| `tests/qualification/ebh_5` | 239 passed / 4 failed | **243 passed / 0 failed** |
| `tests/qualification/governance` | 395 passed / 8 failed | **403 passed / 0 failed** |
| `tests/qualification/ebh_7` (regression check) | 11 failed | 11 failed, the same set: no new failures (group B below) |

The GR-12 repair needed one more fix besides supplying the authority. Once STRICT PRODUCT composition succeeded, the tests never closed the host runtime, and its background workers kept the pytest process alive after the tests passed. The four GR-12 tests now close every runtime they build through a fixture.

## Other pre-existing red suites (read-only diagnosis, wave B)

One bounded run at `origin/development` `f573e24e8`: `ebh_7`, `dist_scale_x`, `roadmap_replay_x` → 18 failed, 410 passed, 15 skipped. No fixes were made in this wave.

| Group | Suite / tests | Verdict | Cause | Fix scope | Product path |
| --- | --- | --- | --- | --- | --- |
| A | none of these suites at this run (still visible in `tests/unit/runtime/integrations/test_contract_registry_v2.py` and `tests/unit/test_dep1_dependency_foundation.py`) | REGRESSION | Duplicate `ollama` catalog slug (`7aa9cfa18`). Earlier ebh_7 failures from it disappeared, most likely through test-order effects on the catalog (not proven) | Batch B-REG-P | **yes** |
| B1 | `ebh_7` P03-B5 executable identity, B6-F10 normative check and its rendered matrix | PRODUCT_DELTA_GATE | By design it fails whenever paths outside the docs/CI allowlist change after the pinned TEST_EXECUTION_SHA `dc18e36d9` (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py:152`, allowlist lines 20–37). First violating commit: `9f9cc8b57` | Operator policy: re-pin the replay SHA, or widen the allowlist, then re-render | no |
| B2 | `ebh_7` P03 immutable digests; cascades into the closure evidence audit (4 tests), the F09 registry integrity check and the certification SSOT bundle | STALE_GATE | `ARTIFACT_DIGESTS.json` still pins `EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md` at its `df498b7a8` digest, although `5a4e922ff` changed the matrix | Re-pin the digest under the same policy decision as B1 | no |
| C | `roadmap_replay_x` (5 tests) | PRODUCT_DELTA_GATE | 17 MCP-SERVER-X product paths changed since `DUP_X_ACCEPTANCE_SHA` `a7d83925b` have no disposition row. Two mechanical counters (2 and 1) differ from the zeros recorded in ECAP | Disposition rows with regression evidence in `tests/qualification/roadmap_replay_x/_pre_ent_l3_prod_delta.py` (and the VBN counter if needed); natural owner MCP-SERVER-X P7 | no |
| D | `dist_scale_x` P4 and P6 batches | ENVIRONMENT | PostgreSQL via Docker is unavailable on this machine (`docker info` fails); P4/P6 use `pytest.fail` where P3 uses `pytest.skip` | Requires Docker/PostgreSQL, which the C10/C15 restriction forbids; it stays blocked until the Operator lifts it | no |

## Operator decisions on the remaining reds (2026-11-12)

Chat decisions of the Operator, transcribed in English.

- **Group A / batch B-REG-P (`ollama` slug):** wait for MCP-SERVER-X P8. The fix changes `intergrax/` and would invalidate the IIAC activation (ACT-N11) that MCP P7/P8 rely on, so it merges with the other P5 product batches.
- **Groups B1 and B2 (`ebh_7`):** re-pin once, after PDC-X P6 at `INTEGRATED_BASELINE_SHA`. The pin `dc18e36d9` sits in the signed EBH-7 certification documents, `MANIFEST.json` and CI defaults, so a re-pin is a fresh L3 replay with a new manifest, acceptance record and re-rendered `ARTIFACT_DIGESTS.json`, not a text substitution. Until then the 11 failures are a known state.
- **Group C and D:** unchanged (MCP P7 owns the disposition rows; Docker stays forbidden by C10/C15).
- **Publication:** wave B (ticket T-IGX-036-I) merged as PR #471 at `5836a5a5b`.

