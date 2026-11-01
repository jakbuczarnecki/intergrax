# EBH-7 — ECAP executable checklist (C1–C5)

**Purpose:** Map ECAP-§2.C steps to **executable** proof. Verdicts are computed — never assigned here — by `test_b6_f10_normative_certification_computed` from closed normative sets; generated detail in [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md) and the §C computed block of [`EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md`](EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md).

| ECAP | Closed normative set (canonical source) | Executable evidence |
| --- | --- | --- |
| C1 | 26 FRZ families (freeze checklist IDs) + Tier-0..3 (`AGENTS.md`) | EBH-6 family L9 gate nodes per family · AST tier-import scan (in-process) · `test_hardening_3_layer_boundary_gate.py` |
| C2 | DUP-01..14 + 15 DUP-X responsibility families + 12 normative surfaces | DUP-X closed-world / mechanical / parent-closure gates · in-process census (composition roots, Tier-3 host factory, env slug reads, tools base imports) · owner delta bindings |
| C3 | Roadmap §3.0.8 domains 1–31 + every inventory concern + discovered mechanisms | EBH-5 inventory / qualified proof integrity · `test_ebh6_prohibited_unresolved_zero` · `test_ebh6_findings_unclassified_zero` · CONFIG-X 54/54 (separate obligations) |
| C4 | Every path changed since accepted baseline `5a3a0534…` (ROADMAP-REPLAY-X) | Change class per path (UNKNOWN = BLOCKER) · per-class required regressions (contract, owner, composition, runtime, security/tenant, state-recovery, vendor) |
| C5 | ECAP §14 report | Computed block == generated · no PASS outside block · matrix == generated (read-only) |

**Deterministic L3 profile:** MANIFEST `pytest_profile.command` (full `tests/qualification/ebh_7/` directory; starts with `uv run pytest tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero …`). Post-run: `scripts/qualification/attest_ebh7_f10_execution.py` (JUnit — every PROFILE_BOUND F10 node executed and passed).

**Generator (deliberate operation, not a test):** `uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py` (`--check` = read-only).

**Non-claims:** B6 PASS · OD-EBH7-08 · EBH-7 CLOSED · global FRZ PASS.
