# Enterprise Architecture Audit Report — EBH-7 B6-F10 certification correction

**Platform scope banner:** This report covers the **Intergrax Agent OS / Harness AI** four-tier platform (`intergrax/` · `intergrax/runtime/` · `agents/` · `applications/`) at EBH-7 closure scope. It is **pre-audit agent output** for independent B6 acceptance — **not** independent B6 PASS, **not** OD-EBH7-08, **not** global FRZ PASS.

Gate verdicts in this report are **computed** by `tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py` from closed normative sets and inserted by `scripts/qualification/render_ebh7_f10_normative_matrix.py`. Obligation-level detail: [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md).

## A. HEAD and baseline

- **TEST_EXECUTION_SHA (certified executable tree):** MANIFEST `TEST_EXECUTION_SHA` — see [`EBH_7_SHA_LADDER.md`](EBH_7_SHA_LADDER.md)
- **Last independently accepted baseline (C4):** `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92` (ROADMAP-REPLAY-X B6 PASS)
- **EBH-7 authorized entry baseline:** `81679e67225f42dfbc1c508516e19488b6fbf453`
- **DUP-X historical acceptance (C2):** `a7d83925b375b5f08eb0f175ced2c17fa026bba9` + current delta census
- **Audit mode:** FULL-HEAD closed-world obligations + DELTA vs accepted baseline

## B. Roadmap progress

- **Current parent stage:** EBH-7 — B6-F10 certification correction; independent B6 acceptance pending
- **Downstream (not entered):** EVOL-X · ENT-AUDIT-X · ARCH-FREEZE · SCENARIO-READINESS-X

## C. Enterprise compliance coverage

Canonical closed sets: 26 FRZ families (freeze checklist IDs), Tier-0..3 (`AGENTS.md`), DUP-01..14 + 15 DUP-X responsibility families + 12 normative C2 surfaces, roadmap §3.0.8 domains 1–31 + every inventory concern, every path changed since the accepted baseline.

ECAP-C5 (this report) is evaluated against the block below and the F10 matrix by the read-only gate; its computed verdict is published only in the F10 matrix.

<!-- F10-COMPUTED:BEGIN (generated — scripts/qualification/render_ebh7_f10_normative_matrix.py) -->
**Computed from closed normative sets @ TEST_EXECUTION_SHA `74e2191dd4eaad544b2146f505fe691d459997b3`** (C5 verified by read-only gate; no hand-assigned verdicts).

| ECAP gate | Computed verdict | Obligations | PASS | N/A WITH EVIDENCE | FAIL |
| --- | --- | --- | --- | --- | --- |
| ECAP-C1 | PASS | 32 | 28 | 4 | 0 |
| ECAP-C2 | PASS | 43 | 43 | 0 | 0 |
| ECAP-C3 | PASS | 105 | 99 | 6 | 0 |
| ECAP-C4 | PASS | 9 | 9 | 0 | 0 |

**C1–C4 computed disposition:** all obligations PASS or N/A WITH EVIDENCE
<!-- F10-COMPUTED:END -->

## D. Architecture findings

| Finding ID | Class | Scope | Disposition |
| --- | --- | --- | --- |
| B6-F10 | Evidence gap — declarative matrix | ECAP C1–C5 | Rebuilt as computed certification; verdict per block in §C |
| F10-C3-REG-01 | Qualification regression (EBH-5 Q07) | `integration.relational_store`, `tool.marketplace_activation`, `agent.effective_profile` | EBH-7 registered composite / parametrized proof refs rejected by EBH-5 validator; validator now resolves every node and binds parameter to `concern_id` |
| F10-C2-OWNER-DELTA | Stale DUP-X owner pointers | RF-EXE/BGW/GOV/RCV/STA/TEN/CMP/T3 | Current-HEAD owner delta bindings validated by path resolution / discovery census (DUP-X SSOT unchanged) |
| F10-C1-TIER-EXC-01 | Ratified tier exception | `intergrax/cli/external_work.py` → Tier-3 demo fixture | Lazy CLI demo import ratified in AC1–AC6 architecture audit; scan fails on any other violation or stale exception |
| F10-C1-STALE-GATE-REG | Stale family gate node | FRZ-REG (EBH-6 `_family_l9_evidence.py`) | Gate pointed to removed `test_aj_evidence_required_honest_not_zero`; rebound to its renamed successor `test_aj_evidence_required_matches_t2_bundle_gaps`; EBH-6 gate-node test now requires the function to exist (previously file-only) |
| F10-C4-STATE-REG-01 | Production regression (state-recovery) | `intergrax/memory/stores/sqlite_conversational_memory_store.py` | `load_memory` called `typing.Literal` `MessageRole(...)` → `TypeError` on every reload (introduced by `8afff322`); fixed with closed-role validation @ `a803191ad`; C4 regression `test_ebh7_fixed_substrate_rpl_positive[memory.sqlite.conversational]` |
| F10-P03-PROFILE-SHADOW | Replay coverage gap | MANIFEST `pytest_profile` | Four `ebh_7` files listed before `tests/qualification/ebh_7/` made pytest 8 collect only those four (195-node B5 runs); shadowing members removed; coverage helper no longer credits a shadowed directory (adversarial gate) |
| F10-CFG-ISO-01 | Test isolation | CONFIG-X FRZ-CFG-07 gates | `clear_catalog()` without bootstrap reset broke later W5-B2 composition tests; fixture restores default integration catalog state |

Historical corrective findings F01–F09 and F11: unchanged (see `_gate_findings_register.py`).

## E. Vendor/backend dependency changes

- C3 domain coverage is computed per roadmap domain (inventory concern or N/A WITH EVIDENCE); CONFIG-X 54/54 and PROHIBITED = 0 are separate obligations and do not substitute domain coverage.
- Production vendor-backend paths changed since the accepted baseline are classified in C4 (`vendor-backend`) with bound regressions.

## F. Cross-layer ownership and authority

- Tier import boundaries: AST scan of Tier-0/Tier-2 sources executed inside the F10 gate + hardening-3 boundary gate.
- DUP-X: class/family/surface identity, counters and closed-world census re-executed @ executed HEAD.

## G. Runtime reliability and failure semantics

- Runtime-behavior / security-tenant / state-recovery changes since the accepted baseline are bound to invoker, tenant-scope, tenant E2E and fixed-substrate RPL regressions (C4).

## H. Impact on historical certification

- B6-F10 code (qualification infrastructure + the F10-C4-STATE-REG-01 production fix) → new TEST_EXECUTION_SHA `74e2191dd4eaad544b2146f505fe691d459997b3` (ladder) and new GitHub Actions replay; prior B5 run `41214574614` retained as historical B5 evidence only — it never executed the full `ebh_7` directory (F10-P03-PROFILE-SHADOW).

## I. Scenario readiness / architecture-change risk

- SCENARIO-READINESS-X: downstream, not in scope.

## J. Blockers vs missing evidence

- Exact missing obligations, if any, are listed in the computed block (§C) and in the F10 matrix — no hand-assigned READY.
- Independent B6 acceptance remains with the external auditor.

## K. Closure verdict and next action

- **Agent disposition:** the computed disposition in §C (C1–C4) and the F10 matrix (C1–C5) — never overridden by text.
- **Non-claims:** B6 PASS · OD-EBH7-08 · EBH-7 CLOSED · EVOL-X · global FRZ PASS
