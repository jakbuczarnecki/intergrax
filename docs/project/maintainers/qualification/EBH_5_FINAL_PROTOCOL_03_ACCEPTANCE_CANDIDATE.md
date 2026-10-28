# EBH-5 ? Final Protocol 03 acceptance candidate

| Field | Value |
| --- | --- |
| **Disposition** | **EBH-5 ? FINAL RPL-F EVIDENCE COMPLETE / READY FOR INDEPENDENT CLOSURE AUDIT** |
| **START_HEAD** | `ac64b8cc0e76b2a25adff3d2208a50389006ef05` |
| **IMPLEMENTATION_SHA** | `c723934998e2a3baae7299b472995f9ba96bd8bb` |
| **TEST_EXECUTION_SHA** | `c723934998e2a3baae7299b472995f9ba96bd8bb` |
| **CERTIFICATION_HANDOFF_SHA** | `c723934998e2a3baae7299b472995f9ba96bd8bb` |
| **PUBLICATION_SHA** | `7d5792eab95b423a03f6538033bd5cab1e667958` |

## Tests @ TEST_EXECUTION_SHA

```text
67 passed, 1 skipped (reachability ? graph integration module absent)
Log: .tmp/session/ebh5-rpl-f-final/pytest-ebh5.log
```

## RPL-F ? active provider failure (runtime)

Evidence SHA for pytest nodes = **TEST_EXECUTION_SHA**. SSOT: `_rpl_f_evidence_registry.py`.

| concern_id | RPL-F | Active failure scenario | pytest node |
| --- | --- | --- | --- |
| `observability.otlp_transport` | **PASS** | Active transport raises `OtlpTransportError` on export; propagates via `OtlpEventExportSink` | `test_ebh5_rpl_f_otlp_active_transport_export_failure_propagates` |
| `tokenizer.selection` | **PASS** | Active default tokenizer raises on `count_tokens` | `test_ebh5_rpl_f_tokenizer_active_provider_tokenization_failure` |
| `persistence.relational_sqlite_lab` | **PASS** | Connected store rejects invalid SQL at `execute` | `test_ebh5_rpl_f_relational_active_store_execute_failure` |
| `int_config.configure_existing` | **PASS** | Active strategy raises `REALIZATION_FAILED` after admission | `test_ebh5_rpl_f_int_config_active_strategy_realization_failure` |
| `integration.key_value_cache` | **PASS** | Active plugin client raises on `set` | `test_ebh5_rpl_f_kv_active_plugin_operation_failure` |

## RPL-A..H (five QUALIFIED)

| Gate | Status |
| --- | --- |
| RPL-A..E | **PASS** |
| RPL-F | **PASS** (table above) |
| RPL-G | **PASS** |
| RPL-H | **QUALIFIED** × 5 |

## Inventory / matrix / SAC

| Item | Value |
| --- | --- |
| Inventory | 69 concerns, unclassified 0 |
| QUALIFIED / NOT YET | 5 / 51 |
| SAC-EBH-5-v1 | **FROZEN** @ 2026-10-28 (unchanged) |
| Fixed substrates | **RATIFIED** (OD-EBH5-04, unchanged) |

## Mandatory blockers

| Gate | Status |
| --- | --- |
| OD-EBH5-07 Protocol 03 PASS | **OPEN** |
| OD-EBH5-08 CLOSED publication | **OPEN** |

**STOP** ? no agent-issued AUDIT PASS or EBH-5 CLOSED.
