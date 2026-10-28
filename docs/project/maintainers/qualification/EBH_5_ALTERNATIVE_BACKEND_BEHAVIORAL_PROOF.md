# EBH-5 — ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF (index)

| Proof | Pytest node (under `tests/qualification/ebh_5/`) | Concern |
| --- | --- | --- |
| Relational sqlite via IntegrationProfile | `test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile` | `persistence.relational_sqlite_lab` |
| INT-CONFIG generic service + sqlite strategy | `test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service` | `int_config.configure_existing` |
| Tokenizer alternate | `test_ebh5_behavioral_tokenizer_alternate_without_consumer_change` | `tokenizer.selection` |
| OTLP transport injection | `test_ebh5_behavioral_otlp_transport_port_injection` | `observability.otlp_transport` |
| KV external plugin | `test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition` | `integration.key_value_cache` |

**Suite:** [`test_ebh_5_behavioral_proofs.py`](../../../tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py)

**Registry SSOT:** [`_behavioral_proof_registry.py`](../../../tests/qualification/ebh_5/_behavioral_proof_registry.py)

**Historical supplemental (not matrix-bound):** INT-CONFIG-REAL-X @ `a59744517b92847f55def1db22826d17d89ee155`; HARNESS-W5 @ `5a87046…` — see registry `HISTORICAL_SUPPLEMENTAL_PROOF_NOTES`.
