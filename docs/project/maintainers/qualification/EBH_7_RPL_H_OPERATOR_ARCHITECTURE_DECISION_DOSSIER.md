# EBH-7 — RPL-H operator architecture decision dossier (Batch C preserved)

| Field | Value |
| --- | --- |
| Batch C impl SHA | `473462bae2bc508dec1230817d107bf802f59b57` |
| Semantic SSOT | `tests/qualification/ebh_7/_rpl_h_semantic_evidence.py` |
| Counts | **6 QUALIFIED · 29 PARTIAL · 21 BLOCKED** |
| Reclassification | **None** without new qualifying evidence or operator ADR |

## A — Architecture work required (INTEGRATIONS-2E cutover)

**Root cause:** `resolve_from_profile` requires `PlatformIntegrationContract`; many catalog factories still return legacy runtime types.

**Concerns (BLOCKED @ Batch C):** `integration.document_store`, `message_bus`, `notification_channel`, `object_storage`, `graph_store`, `collaboration_suite`, `issue_tracker`, `wiki_knowledge`, `browser_automation`, `cloud_platform`, `secrets_store`, `sandbox_host`, `speech_provider`, `embedding_provider`.

**Remediation route:** Complete contract cutover per category (no parallel registry); re-run Package C v2 dual-slug + operational sandbox proofs.

## B — Single-provider / missing category (operator vendor or scope)

**BLOCKED:** `integration.external_work` (no provider category), `billing_meter`, `managed_retrieval`, `ml_inference_host`, `model_serving_runtime`, `vision_serving` (single slug).

**Remediation route:** Register alternate production provider or operator FIXED-substrate ADR with evidence.

## C — Environment / sandbox limitation (29 PARTIAL)

**Root cause:** Dual slug **selection** proven @ dev-ci; operational equivalence requires vendor sandbox or heavy deps (e.g. `vector_store` + chromadb).

**Concerns:** Remaining integration PARTIAL rows in semantic evidence + non-integration Package C factory-only rows.

**Remediation route:** Sanctioned extras (`vector-chroma`, vendor qual profiles) or scoped Testcontainers — not Batch D TEN/XDM scope.

## D — Legitimate fixed substrate (proposal only — not authorized)

Categories with one production backend may be proposed as FIXED after operator ADR; **no agent reclassification**.

## E — Later canonical stage

PROD-Q live slice, WL-7 four-hour soak, plugin host `continuation_aware_catalog_tool_host` harness (TEN-13 / XDM-018) — **Path B / operator**.

## Relational store QUALIFIED integrity

- Mandatory `duckdb` via `dev-ci`; `require_python_module` fails closed (no skip).
- Negative: `UnknownIntegrationError` / `IntegrationConfigurationError` only.
- Evidence: `test_ebh5_pkgc_v2_relational_sqlite_duckdb_dual_slug` + lifecycle connect/close both slugs.
