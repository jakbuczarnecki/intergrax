# EBH-5 — Protocol 03 remediation reconciliation (OD-EBH5-07)

| Field | Value |
| --- | --- |
| **Disposition** | _set after publication_ |
| **START_HEAD** | `34d6e664461a1fdc357c578b8971f4223b486bc8` |
| **IMPLEMENTATION_SHA** | _implementation commit_ |
| **TEST_EXECUTION_SHA** | _same as IMPLEMENTATION_ |
| **CERTIFICATION_HANDOFF_SHA** | _same as IMPLEMENTATION_ |
| **PUBLICATION_SHA** | _publication commit_ |

## RPL-F — five QUALIFIED (SSOT: `_rpl_f_evidence_registry.py`)

| concern_id | RPL-F | Representative failure/recovery proof |
| --- | --- | --- |
| `observability.otlp_transport` | **PASS** | Invalid distributed OTLP config → `ConfigurationError`; empty OTLP endpoint fail-closed export |
| `tokenizer.selection` | **PASS** | Unknown default tokenizer id → `ValueError` at registry boundary |
| `persistence.relational_sqlite_lab` | **PASS** | Unregistered SQLite catalog entry → `IntegrationConfigurationError` on sanctioned `resolve_from_profile` |
| `int_config.configure_existing` | **PASS** | No matching strategy → `ExistingCapabilityConfigurationRealizationError` (`UNSUPPORTED_STRATEGY`) |
| `integration.key_value_cache` | **PASS** | Unregistered plugin slug → `IntegrationConfigurationError` on sanctioned composition |

No blanket `RPL-F = N/A` on QUALIFIED rows.

## Six roadmap N/A domains

SSOT: `_roadmap_domain_na_evidence.py` — each row includes `discovery_limitations` and inventory cross-check for `covered_concern_ids`.

## Preserved operator state

- OD-EBH5-02 / SAC **FROZEN** @ 2026-10-28 — criteria unchanged.
- OD-EBH5-04 **APPROVED** — fixed substrates **RATIFIED** in SSOT.
- OD-EBH5-07 **OPEN** — independent re-audit required.
- OD-EBH5-08 **OPEN**.

**Not recorded:** Protocol 03 PASS, EBH-5 CLOSED.
