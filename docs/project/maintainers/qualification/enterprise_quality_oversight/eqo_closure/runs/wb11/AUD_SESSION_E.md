# EQO-W3-WB11-AUD — session E verdict (wave AUD-2)

**Verdict (WB11): `FAIL`**
**Auditor session:** independent session E (not the EXE agent; no EXE/DIS artefact in this WB authored by this session).
**Date:** 2026-11-21
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469` (source read at the pin, read-only)
**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Verdict vocabulary

| Verdict | Meaning |
|---------|---------|
| `PASS` | Evidence at the pin supports the stated disposition. |
| `FAIL` | Source at the pin confirms the defect or contradicts the stated disposition. The finding stays OPEN. |
| `BLOCKED` | The pytest slice does not produce evidence to decide, or an owner/operator decision is a prerequisite. |

Aggregate rule: any `FAIL` gives `FAIL`; otherwise any `BLOCKED` gives `BLOCKED`; otherwise `PASS`.

## Packet checks

| Check | Result |
|-------|--------|
| TRAIN_PIN in manifest / DES_PREREG / packet = `28c27cb8…` | YES (object exists locally) |
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `088a4be9…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 1 failed, 68 passed in 150.74s |
| Log records command, node ids or its own hash | NO for the command and hash. The log does name the failing node id and assertion. |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** XD-12 EVOL/ENT-audit mapping, ENT-AUDIT "cannot exit clean with skipped mandatory", COMPAT-X inventory and persisted-R2 final gates, the PROD-Q R1 durable seal store test and the EBH-7 W4 governance/DUP gates. 68 pass. The single failure is `test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check` (`test_xd_12_evol_ent_audit.py:97`): "3 PDC-touched product modules map to no check", for example `intergrax/background_tasks/active_work.py`, `intergrax/background_tasks/worker_runtime.py` and `intergrax/runtime/capacity/production_adapters.py`. That failure is the one executed test that bears directly on a finding (A6-007). None of the other nine findings has a covering test in the slice. The slice names no P0/P2/P3 closed-world replay, no full COMPAT suite and no FRZ checklist reconciliation.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A6-007 | BLOCKED | **FAIL** | Hypothesis confirmed by execution at the pin. The XD-12 gate loads the default impact map (`load_default_impact_map`, `intergrax/platform/enterprise_audit/impact_map.py:309`) and finds product modules touched since PDC that select no check, in `intergrax/background_tasks/` and `intergrax/runtime/capacity/`. A change in those paths under CHANGED-SINCE therefore selects no mapped check, so only the invariant floor runs. Route: owner extends the impact map (or the check manifest) for the three modules and the packages they belong to, then replays. |
| EQOA-A7-037 | BLOCKED | **BLOCKED** | Source count at the pin: 143 skip/xfail marker lines in 114 files under `tests/qualification` and `tests/architecture` (the dossier says 164; the counting rule differs). Only 3 of those 114 files are named in the EBH-7 skip disposition register (`EBH_7_SKIP_DISPOSITION_REGISTER.md` or `_ebh7_skip_disposition_register.py`). The register is built from a real L3 `junit.xml`, so it lists skips that occurred, not every marker. Whether the other markers need a register is a scope decision for the owner, together with the 15 markers the dossier says carry no reason (not re-derived here). |
| EQOA-A1-008 | BLOCKED | **BLOCKED** | Roadmap-to-checklist link reconciliation is a documentation comparison. No test in the slice does it. Closure needs an owner reconciliation or a recorded acceptance. |
| EQOA-A1-012 | BLOCKED | **BLOCKED** | `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:456` records COMPAT-X-R5 as "CLOSED / independently accepted" at `84b8739b…`. The slice runs the COMPAT-X inventory and persisted-R2 final gates (pass) but not R5 and not the full COMPAT suite, which the dossier says was NOT RUN. The slice cannot confirm or refute the single-gate basis. |
| EQOA-A1-015 | BLOCKED | **BLOCKED** | The PROD-Q-P0 replay result (438/436/0/2) is not reproduced. The slice runs only `test_prod_q_r1_durable_seal_store.py`. No tracked log for the P0 replay, and no P2/P3 replay, is in the slice or the packet. |
| EQOA-A2-023 | BLOCKED | **BLOCKED** | The question is whether P0/P2/P3 B6 closures have an independent exact-SHA audit. A pytest slice cannot answer it, and the packet cites none. Closure needs the audit records or an owner acceptance. |
| EQOA-A2-024 | BLOCKED | **BLOCKED** | The P0 closed-world inventory gate is not in the slice, so there is no replay after the 202 product-path commits the dossier cites. |
| EQOA-A2-051 | BLOCKED | **BLOCKED** | `COMPAT_X_CONTRACT_SCHEMA_EVOLUTION_CERTIFICATION.md:443` still records the targeted-suite failures as "mandatory future debt; not resolved". The slice's persisted-R2 final gates pass, but they are not the 47-failure targeted suite, and the packet cites no later disposition. A replay of that targeted suite or an owner disposition is needed. |
| EQOA-A7-003 | BLOCKED | **BLOCKED** | The FRZ-DEBT-04/05 question (zero inventory of existing PARTIAL/DEFERRED entries) is a register reconciliation, not covered by any test in the slice. |
| EQOA-A7-007 | BLOCKED | **BLOCKED** | The EBH-7 W4 governance/DUP gates pass at the pin, but they do not show the zero-PROHIBITED and zero-duplicate counts were recomputed against the 275 changed product files. The WB12 verdict notes that the DUP-X closed-world tests are structural. Closure needs a recount at a current SHA. |

## Rationale for the aggregate

`FAIL` because A6-007 is confirmed by a gate that fails at the pin. The other nine findings are not covered by the slice and need owner or operator input or a replay. Nine of ten findings stay open with no evidence either way.

## Required follow-up (no product fix here)

1. A6-007: extend the impact map for `intergrax/background_tasks/` and `intergrax/runtime/capacity/` (and any other PDC-touched product package), then re-run XD-12.
2. A7-037: owner decides whether markers outside the L3 junit skips need a register, and fixes the reasonless ones.
3. A1-012, A1-015, A2-023, A2-024, A2-051, A7-007: replay the named gates or suites at a current pin, or record scoped acceptances with owner and reopen condition.
4. A1-008, A7-003: owner reconciliation of the roadmap, the FRZ checklist and the FRZ-DEBT register.

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
