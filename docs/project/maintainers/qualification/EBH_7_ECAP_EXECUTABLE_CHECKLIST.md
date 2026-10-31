# EBH-7 — ECAP executable checklist (C1–C5)

**Purpose:** Independent ECAP verification — not counter-only reports.

| ECAP | Executable check | Gate / evidence |
| --- | --- | --- |
| C1 | Mechanism registry integrity @ HEAD | `test_ebh_7_mechanical_exit_gates.py` |
| C2 | DUP-X subset + tier import boundaries | `test_ebh_7_w4_gov_dup_gates.py` |
| C3 | PROHIBITED unresolved = 0 + EBH-7 delta bundles | `test_ebh6_prohibited_unresolved_zero` + `test_ebh_7_int2e_bundle_regression.py` |
| C4 | TEN 26 + disposition + SSOT evidence | `test_ebh_7_certification_ssot_gates.py` + appendix |
| C5 | XDM 34 — zero COMPOSED_PROXY @ SSOT | `test_ebh_7_certification_ssot_gates.py` + appendix |

**Deterministic L3 profile (`ci`) — full EBH-7 suite + mandatory regressions:**

```text
uv run pytest \
  tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero \
  tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py \
  tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py \
  tests/qualification/ebh_7/test_ebh_7_rpl_dual_slug_catalog_coverage.py \
  tests/qualification/ebh_7/test_ebh_7_certification_ssot_gates.py \
  tests/qualification/ebh_7/test_ebh_7_certification_mutation_fail_closed.py \
  tests/qualification/ebh_7/ \
  -q --tb=line
```

**Non-claims:** Global NPSC allowlist FAIL retained (Path B). B6 / OD-EBH7-08 OPEN until Wave 7–8.
