# EBH-5 — Protocol 03 final audit correction (F01/F02)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD (pre-final)** | `ffe0aae8d2b64a558a18226cc1435f0058ff561b` |
| **AUDIT_BASELINE_SHA (wave)** | `83a376b3e1b312e2e4f88da548c99c328f3589f7` |
| **Disposition** | Final corrective batch — **not** parent CLOSED |

## F01 — `observability.otlp_transport`

| Element | Ustalenie |
| --- | --- |
| Produkcyjny odbiorca | `ApplicationRuntimeEventDeliveryWiring` → `OtlpEventExportSink` (bounded event delivery) |
| Kontrakt | `OtlpTransportPort` + `EventExportSinkPort` |
| Punkt wyboru | Tier-3 `resolve_application_runtime_event_delivery_wiring` + `ObservabilityExportProfile` / governance observability bundle; opcjonalnie `export_sink_factory` z `ObservabilityExportSinkFactory(otlp_transport=…)` |
| Alternatywy | `OtlpTransport` (OTLP) vs `CollectorTransport` (DISTRIBUTED_OTLP) via profil; test portu przez factory DI |
| Dowód | `test_ebh5_behavioral_otlp_transport_via_sanctioned_composition_wiring` — **bez** patch `_create_export_transport` |
| Negatywny | `test_ebh5_behavioral_otlp_transport_negative_invalid_distributed_configuration`; `test_ebh5_behavioral_otlp_transport_negative_otlp_without_endpoint_fail_closed` |
| RPL-F | **NA** (brak obowiązkowego failure/recovery w zakresie EBH-5) |

### F01 — semantyka wyłączenia vs błąd konfiguracji

| Scenariusz | Zachowanie (owner: `runtime_event_delivery_wiring`) | Interpretacja |
| --- | --- | --- |
| `ExporterKind.OTLP` + pusty `otlp_export_endpoint` | `_create_export_transport` → `None`; `OtlpEventExportSink.export` kończy bez wyjątku i **bez** wywołania transportu | **Wyłączenie transportu** przy profilu OTLP — nie tworzy `OtlpTransport`; brak pozornego sukcesu na injekowanym porcie (test) |
| `ExporterKind.DISTRIBUTED_OTLP` + brak endpoint / service name | `ConfigurationError` w `resolve_application_runtime_event_delivery_wiring` | **Fail-closed** dla aktywnego eksportera klastrowego |
| Brak nieautoryzowanego fallback | Brak alternatywnego transportu poza profilem / `export_sink_factory` | Dowód pozytywny: real `_create_export_transport` + factory DI |

**Known limitation (auditor disposition):** asymetria OTLP (ciche wyłączenie) vs DISTRIBUTED_OTLP (wyjątek) — jawny kontrakt; ewentualna unifikacja poza EBH-5 wymaga ADR + operatora.

## F02 — `tokenizer.selection`

| Element | Ustalenie |
| --- | --- |
| Produkcyjny odbiorca | `TokenizerManager` / RAG bootstrap via `create_default_tokenizer_engine` |
| Kontrakt | `TokenizerRegistry.default_tokenizer_id` + `TokenizerEngine.count_tokens` |
| Punkt wyboru | `create_default_tokenizer_engine(registry=…)` (bootstrap DI) |
| Alternatywa | `default_tokenizer_id="simple"` vs `"tiktoken"` bez zmiany `TokenizerManager` |
| Dowód | `test_ebh5_behavioral_tokenizer_selection_via_bootstrap_composition` |
| Negatywny | `test_ebh5_behavioral_tokenizer_selection_negative_unknown_default` |

## SSOT counts (unchanged classification)

| Metric | Count |
| --- | --- |
| QUALIFIED | 5 |
| NOT YET (matrix) | 51 |

## Verification

```powershell
uv run pytest tests/qualification/ebh_5/ -q
```

**Expected after final correction:** 49 passed, 1 skipped.
