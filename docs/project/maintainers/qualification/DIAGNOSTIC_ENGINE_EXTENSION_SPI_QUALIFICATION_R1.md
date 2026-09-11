# DIAGNOSTIC-ENGINE-EXTENSION-SPI — Qualification R1

**Task:** `DIAGNOSTIC-ENGINE-EXTENSION-SPI-ARCHITECTURE-R1`

## Matrix

| ID | Scenario | Expected | Test |
| -- | -------- | -------- | ---- |
| R5-A1 | Plugin contributes evidence | Evidence stored; Problem count unchanged | `test_r5_a1_*` |
| R5-A2 | Analyzer finds pattern | `DiagnosticExtensionFindingView`; lifecycle owns Problem | `test_r5_a2_*` |
| R5-A3 | Analyzer crash | Diagnostics continue; `DEGRADED` + `PLUGIN_UNAVAILABLE` | `test_r5_a3_*` |
| R5-A4 | Tenant isolation | Cross-tenant evidence rejected | `test_r5_a4_*` |
| R5-A5 | Duplicate plugin | `DiagnosticExtensionConfigurationError` | `test_r5_a5_*` |
| R5-A6 | Malicious plugin | No `ProblemPersistence` on extension service | `test_r5_a6_*` |
| R5-A7 | No extensions | `extension_enrichment is None` | `test_r5_a7_*` |

Harness: `testing_support/runtime/diagnostic_extension_spi_r5_harness.py` (extends R2 execution failure closure harness).

Contract tests: `tests/unit/contracts/test_diagnostic_extension_spi.py`.

Architecture gate symbols: `test_r5_quality_gates_forbidden_symbols_and_single_engine`.

## Verdict

PASS when:

```bash
uv run pytest tests/unit/contracts/test_diagnostic_extension_spi.py tests/unit/runtime/diagnostics/test_diagnostic_extension_spi_r5_qualification.py
```

succeeds on `development`.
