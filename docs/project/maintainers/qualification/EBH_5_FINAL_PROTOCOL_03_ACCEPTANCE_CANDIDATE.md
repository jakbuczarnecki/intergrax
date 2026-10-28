# EBH-5 ? Final Protocol 03 acceptance candidate

| Field | Value |
| --- | --- |
| **Disposition** | **EBH-5 ? SQLITE RPL-F FINAL CORRECTION READY FOR INDEPENDENT ACCEPTANCE** |
| **START_HEAD** | `f2ec11ae60bef3f44d5c4415adecce8f629e44d9` |
| **IMPLEMENTATION_SHA** | `33ecf19cc020b112d9e586c87b49e819202f5d6d` |
| **TEST_EXECUTION_SHA** | `33ecf19cc020b112d9e586c87b49e819202f5d6d` |
| **CERTIFICATION_HANDOFF_SHA** | `33ecf19cc020b112d9e586c87b49e819202f5d6d` |
| **PUBLICATION_SHA** | `3386a453692d1c914f048dd3f1eaeb55b5437f07` |

## Tests @ TEST_EXECUTION_SHA

```text
67 passed, 1 skipped (reachability ? graph integration module absent)
Log: .tmp/session/ebh5-sqlite-rpl-f-correction/pytest-ebh5-postcommit.log
```

## RPL-F ? active provider failure (runtime)

Evidence SHA for pytest nodes = **TEST_EXECUTION_SHA**. SSOT: `_rpl_f_evidence_registry.py`.

| concern_id | RPL-F | Active failure scenario | pytest node |
| --- | --- | --- | --- |
| `observability.otlp_transport` | **PASS** | Active transport raises `OtlpTransportError` on export; propagates via `OtlpEventExportSink` | `test_ebh5_rpl_f_otlp_active_transport_export_failure_propagates` |
| `tokenizer.selection` | **PASS** | Active default tokenizer raises on `count_tokens` | `test_ebh5_rpl_f_tokenizer_active_provider_tokenization_failure` |
| `persistence.relational_sqlite_lab` | **PASS** | After sanctioned connect + control DML, operational loss of active `sqlite3` connection surfaces at `execute` (valid SQL); no provider swap | `test_ebh5_rpl_f_relational_active_store_operational_failure_after_connect` |
| `int_config.configure_existing` | **PASS** | Active strategy raises `REALIZATION_FAILED` after admission | `test_ebh5_rpl_f_int_config_active_strategy_realization_failure` |
| `integration.key_value_cache` | **PASS** | Active plugin client raises on `set` | `test_ebh5_rpl_f_kv_active_plugin_operation_failure` |

### SQLite RPL-F evidence ( `persistence.relational_sqlite_lab` )

| Item | Value |
| --- | --- |
| Active provider | `sqlite` (`SQLITE_RELATIONAL_STORE_PROVIDER_ID`) via `register_sqlite_integration()` |
| Production contract | `RelationalStoreIntegrationContract` ? `resolve_from_profile` + `IntegrationProfile.relational_store=SQLITE` |
| Fault injection | Controlled close of live `sqlite3.Connection` on active adapter client after successful control operations (not invalid SQL) |
| Expected exception | `sqlite3.ProgrammingError` (`closed database`) on subsequent valid `INSERT` |
| Silent fallback | **None** ? same resolved integration instance; no alternate relational provider |
| Recovery | **Not asserted** ? contract does not guarantee transparent reconnect after operational connection loss; test ends with `integration.close()` |
| Lifecycle | register ? profile ? resolve ? connect ? control ops ? fault ? close |

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
| Matrix | 56 rows |
| QUALIFIED / NOT YET | 5 / 51 |
| SAC-EBH-5-v1 | **FROZEN** @ 2026-10-28 (unchanged) |
| Fixed substrates | **RATIFIED** (OD-EBH5-04, unchanged) |

## G06 enterprise integrity

| Gate | Status |
| --- | --- |
| G06-01..04 | **PASS** @ prior operator batch (unchanged by SQLite RPL-F correction) |

## Mandatory blockers

| Gate | Status |
| --- | --- |
| OD-EBH5-07 Protocol 03 independent acceptance | **OPEN** ? corrective SQLite RPL-F evidence @ **CERTIFICATION_HANDOFF_SHA** |
| OD-EBH5-08 CLOSED publication | **OPEN** |

**STOP** ? no agent-issued AUDIT PASS or EBH-5 CLOSED.
