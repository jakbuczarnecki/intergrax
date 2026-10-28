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
| Punkt wyboru | Tier-3 `resolve_application_runtime_event_delivery_wiring` + `ObservabilityExportSinkFactory(otlp_transport=…)` |
| Alternatywy | `OtlpTransport` / `CollectorTransport` via profile; test dowodzi wstrzyknięcia przez factory |
| Dowód | `test_ebh5_behavioral_otlp_transport_via_sanctioned_composition_wiring` |
| Negatywny | `test_ebh5_behavioral_otlp_transport_negative_missing_injected_transport` |
| RPL-F | **NA** (brak obowiązkowego failure/recovery w zakresie EBH-5) |

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

**Expected after final correction:** 48 passed, 1 skipped.
