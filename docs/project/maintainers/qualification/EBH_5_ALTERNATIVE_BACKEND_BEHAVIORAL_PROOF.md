# EBH-5 — ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF (index)

| Proof | Pytest node (under `tests/qualification/ebh_5/`) | Concern |
| --- | --- | --- |
| Relational sqlite via IntegrationProfile | `test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile` | `persistence.relational_sqlite_lab` |
| INT-CONFIG generic service + sqlite strategy | `test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service` | `int_config.configure_existing` |
| OTLP via composition wiring | `test_ebh5_behavioral_otlp_transport_via_sanctioned_composition_wiring` | `observability.otlp_transport` |
| Tokenizer via bootstrap composition | `test_ebh5_behavioral_tokenizer_selection_via_bootstrap_composition` | `tokenizer.selection` |
| KV external plugin | `test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition` | `integration.key_value_cache` |

**Negatywne (F01/F02):** `test_ebh5_behavioral_otlp_transport_negative_missing_injected_transport`, `test_ebh5_behavioral_tokenizer_selection_negative_unknown_default`.

**Suite:** [`test_ebh_5_behavioral_proofs.py`](../../../tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py)

**Registry SSOT:** [`_behavioral_proof_registry.py`](../../../tests/qualification/ebh_5/_behavioral_proof_registry.py)

**Historical supplemental (not matrix-bound):** INT-CONFIG-REAL-X @ `a597445…`; HARNESS-W5 @ `5a87046…`.
