# EBH-7 — Fixed substrate ratifications (OD-RPL-G2)

**Operator authorization:** 2026-10-31 @ `fd3bc0104c9273f30262946afb16c15c04c8493a`  
**SSOT:** [`tests/qualification/ebh_5/_fixed_substrate_registry.py`](../../../tests/qualification/ebh_5/_fixed_substrate_registry.py)

These substrates are **not** vendor-equivalent and **not** REPLACEABLE — QUALIFIED.

## `ebh7-conversational-memory-sqlite-lab`

| Field | Value |
| --- | --- |
| **Concern** | `memory.sqlite.conversational` only |
| **Owner** | `intergrax/memory/stores/sqlite_conversational_memory_store.py` |
| **Constraint** | SQLite lab conversational persistence — does not ratify entire memory abstraction as fixed |
| **Multi-instance** | Single-node / lab scope |
| **Review** | 2026-10-31 |

## `ebh7-integration-single-slug-vendor-slice`

| Concern | Production slug (catalog) | Path B |
| --- | --- | --- |
| `integration.billing_meter` | stripe | Sandbox replay package — operator |
| `integration.managed_retrieval` | openai | Sandbox replay package — operator |
| `integration.ml_inference_host` | replicate | Sandbox replay package — operator |
| `integration.model_serving_runtime` | ollama | Sandbox replay package — operator |
| `integration.vision_serving` | triton | Sandbox replay package — operator |

**Constraint:** No in-repo fake second provider; fail-closed contract tests remain authoritative for dev-ci.

## `integration.external_work`

Resolved by **profile-bound dual in-process providers** (not catalog slug). See `test_ebh_5_factory_rpl_one_pass` + `_external_work_dual_provider_proof.py`.
