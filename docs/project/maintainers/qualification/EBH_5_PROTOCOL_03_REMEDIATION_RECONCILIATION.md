# EBH-5 — Protocol 03 remediation reconciliation (OD-EBH5-07)

| Field | Value |
| --- | --- |
| **Disposition** | **EBH-5 — OD-EBH5-07 CORRECTIVE EVIDENCE READY FOR INDEPENDENT RE-AUDIT** |
| **START_HEAD** | `34d6e664461a1fdc357c578b8971f4223b486bc8` |
| **IMPLEMENTATION_SHA** | `7b4259f3467555ee3bd4bed603a773807c8a56ea` |
| **TEST_EXECUTION_SHA** | `7b4259f3467555ee3bd4bed603a773807c8a56ea` |
| **CERTIFICATION_HANDOFF_SHA** | `7b4259f3467555ee3bd4bed603a773807c8a56ea` |
| **PUBLICATION_SHA** | b87690c50454e9673bc63192bb70bd7cecdecdbb |

## Tests @ TEST_EXECUTION_SHA

```text
62 passed, 1 skipped (reachability — graph integration module absent)
Log: .tmp/session/ebh5-od-ebh5-07/pytest-ebh5.log
```

## RPL-F — five QUALIFIED (SSOT: `_rpl_f_evidence_registry.py`)

| concern_id | RPL-F | Representative failure/recovery proof |
| --- | --- | --- |
| `observability.otlp_transport` | **PASS** | Invalid distributed OTLP config › `ConfigurationError`; empty OTLP endpoint fail-closed export |
| `tokenizer.selection` | **PASS** | Unknown default tokenizer id › `ValueError` at registry boundary |
| `persistence.relational_sqlite_lab` | **PASS** | Unregistered SQLite catalog › `UnknownIntegrationError` on `resolve_from_profile` |
| `int_config.configure_existing` | **PASS** | No matching strategy › `ExistingCapabilityConfigurationRealizationError` (`UNSUPPORTED_STRATEGY`) |
| `integration.key_value_cache` | **PASS** | Unregistered plugin slug › `UnknownIntegrationError` on sanctioned composition |

## Six roadmap N/A domains

SSOT: `_roadmap_domain_na_evidence.py` — `discovery_limitations` + inventory cross-check for `covered_concern_ids`.

## Counts

| Metric | Value |
| --- | --- |
| QUALIFIED | 5 |
| NOT YET (matrix) | 51 |
| Inventory concerns | 69 |

## Preserved operator state

- OD-EBH5-02 / SAC **FROZEN** @ 2026-10-28.
- OD-EBH5-04 **APPROVED** / substrates **RATIFIED**.
- OD-EBH5-07 **OPEN** (independent re-audit @ **CERTIFICATION_HANDOFF_SHA**).
- OD-EBH5-08 **OPEN**.

**Not recorded:** Protocol 03 PASS, EBH-5 CLOSED.
