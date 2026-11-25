# ec-v6-ec02-gap-closure

**Status: DRAFT** — not Approved. Operator-facing orchestration companion to factory completion at integration SHA `35137143e209afd0b0000f8ff95c6542a07a9eb4` (`factory/ec-v6-ec02-closure/int`). `development` remains `530d9fa48`. No push planned by this document.

Factory reference: [`agent-env/plans/ec-v6-ec02-closure.md`](ec-v6-ec02-closure.md) (no QUAL/CLOSE steps; policy pin `BLOCKED_AUTHORITY` noted there).

**046-H verification (ticket read):** `T-IGX-046-H.md` contains `Approved: operator @ 2026-11-24T18:00:00+01:00` with authorization for AST rewrite of `scripts/maintenance/check_agent_registry_bypass.py` on branch **`arch-freeze/rw-train`** only — not `factory/ec-v6-ec02-closure/int` or EC-02 integration.

**051-N verification:** Train draft block lists `Approved: operator-delegate @ 2026-11-19T17:37:29+01:00`; conditional write set includes `tests/qualification/compat_x/_compat_x_event_evolution_inventory.py` and `tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py` under ruling GM-AB-1. Development tor header still shows `PENDING` approval — EC-02 factory step S05E cited train-approved ticket.

**043-A / 043-C / 043-M:** No valid `Approved:` line (placeholders `PENDING` only).

---

## Phase 0 — Authority (operator or independent certifier)

Decisions that cannot be coded away on the integration branch. Until resolved, implementers must **not**: edit `governance/TRUSTED_POLICY_PIN.yaml`, run `--write-pin`, weaken or skip tests, bypass hooks, force-push, or claim CLOSED / IIAC PASS / FRZ.

| Decision | Owner | Must not do meanwhile |
|----------|-------|------------------------|
| Policy-registry blob drift vs pin (`d676336c…` expected, `14692b2c…` at HEAD; commit `1a63ee8f0` / `T-IGX-054-B`, no restore ticket) | Operator authority | Re-pin, `--write-pin`, or treating pin/IIAC pytest reds as product defects |
| EQO terminal closure, IIAC PASS, FRZ, C01–C25 FULLY_MET, `EQO_CLOSED_INDEPENDENTLY_VERIFIED` per `EQO_ROADMAP_REMAINING.md` §11 | Independent certifier / Operator | Any status vocabulary CLOSED, IIAC PASS, or FRZ in EC-02 execution artifacts |
| `T-IGX-046-H` AST rewrite scope vs EC-V6 factory omission (`check_agent_registry_bypass.py` verified-only on EC-02; ticket targets `arch-freeze/rw-train`) | Operator / campaign authority | Implementing 046-H on `int` or `development` without explicit branch re-authorization |
| Missing `Approved:` on `T-IGX-043-A`, `T-IGX-043-C`, `T-IGX-043-M` | Operator wave packet | Product or docs commits under those ticket Refs |
| Interpretation of 62 QUAL failures (10 pin/IIAC formal, 49 env/out-of-scope; no test weakening) | Operator / tester disposition | Skipping pin/IIAC suites or weakening ENT-AUDIT promotion tests |

---

## Phase 1 — Approved-ticket product work (factory-eligible)

Only items with a valid `Approved:` line whose write set and branch match the execution target (`factory/ec-v6-ec02-closure/int` @ post-factory SHA unless operator re-targets).

1. **GM-AB-1 COMPAT-X residual (S05E gap)** — `T-IGX-051-N` conditional paths; implement inventory/gate updates not landed in S05E ACCEPT slice.
2. **ENT-AUDIT engine Parts A–D + first wave-diff instalment** — `T-IGX-043-I` (Approved); factory S07A — **verify** on `35137143` (status `EXISTING_VERIFIED` if green; `RECOVERED` if fix needed within write set).

**Excluded from Phase 1:** `T-IGX-046-H` (wrong branch for EC-02 integration; Phase 0). `T-IGX-043-M` Part E (no Approved). `T-IGX-043-A` / `T-IGX-043-C` (no Approved).

---

## Phase 2 — Approval required before code

Operator adds `Approved:` (and wave-packet record where required) before any implementer commit:

- `T-IGX-043-A` — W0 PT-FC1 documentation (decision register, SSOT reconciliation part A).
- `T-IGX-043-C` — PDC-X residual qualification package (parked).
- `T-IGX-043-M` — Part E manifest consumers / final wave-diff; prerequisite: `T-IGX-043-I` Part C (`digests.py`, FRZ attribution, stage-gate-only block) present on the tor named in the approval (043-M ticket: Part C **absent on `development`**; S07A may satisfy on `int` only).

No implementation steps drafted here.

---

## Phase 3 — Qualification (single SHA gate)

Run after Phase 1 (and Phase 2 if in scope) on **one** integration SHA — **not** the 83-minute full governance sweep as the EC-02 gate.

**In-scope suites:**

- DF5: `tests/unit/runtime/architecture/test_diag_foundation_5_destructive_proof.py` (and S03-listed DF5 fixtures if touched).
- ENT-AUDIT disposition: `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py` (S07C).
- Branch protection: `tests/unit/governance/test_verify_ent_audit_branch_protection.py`, `tests/qualification/ent_audit_x/unit/test_od_ent_02_verifiers.py` (S07B).
- ENT-AUDIT engine negatives (043-I contract): the four `test_wd_*` modules from S07A.
- Three FAST promotion disclosure tests: run with **short pytest `basetemp`** on Windows (MAX_PATH mitigation); failures under deep basetemp are **environment**, not S07C product bugs — do not weaken tests.

**Formal (non-product):** Pin and IIAC-related failures remain **formal** until Phase 0 pin authority resolves; record as `BLOCKED_AUTHORITY`, not open product defects.

Target status after green in-scope run: `QUALIFIED` (not CLOSED / not IIAC PASS).

---

## Phase 4 — CLOSE documentation only

After Phase 3 on the same SHA: update session/checklist docs under qualification orchestration paths only — statuses `INTEGRATED` / `QUALIFIED` as appropriate. Forbidden in execution: CLOSED, IIAC PASS, FRZ. Independent certifier owns terminal EQO gates.

---

## Twelve-item execution matrix

| ID | Phase | Owner role | Prerequisite | Exit criterion | Implementable now? |
|----|-------|------------|--------------|----------------|-------------------|
| P0-PIN | 0 | Operator | — | Recorded authority decision on registry blob / pin (no implementer re-pin) | **No** — `BLOCKED_AUTHORITY` |
| P0-EQO | 0 | Independent certifier | Phase 3 | EQO roadmap operator gates satisfied externally | **No** — `BLOCKED_AUTHORITY` |
| P0-046H | 0 | Operator | — | EC-02 vs `arch-freeze/rw-train` scope decision for AST gate | **No** — branch/campaign conflict |
| P0-APPR | 0 | Operator | — | `Approved:` on 043-A, 043-C, 043-M | **No** — authority |
| P1-COMPAT | 1 | igx-implementer | `T-IGX-051-N` Approved + GM-AB-1 | COMPAT-X two files updated; S05E tests green | **Yes** |
| P1-043I | 1 | igx-tester | S07A merged @ int SHA | Four `test_wd_*` modules pass; Part C artifacts present on int | **Yes** (verify) |
| P2-043A | 2 | — | P0-APPR + plan digest | Operator `Approved:` on 043-A | **No** — `BLOCKED_DEPENDENCY` |
| P2-043C | 2 | — | P0-APPR | Operator `Approved:` on 043-C | **No** — `BLOCKED_DEPENDENCY` |
| P2-043M | 2 | — | P0-APPR + 043-I Part C on approved tor | Operator `Approved:` on 043-M; Part E tests per plan | **No** — `BLOCKED_DEPENDENCY` |
| P3-QUAL | 3 | igx-tester | Phase 1 complete | In-scope suites green one SHA; pin/IIAC reds logged formal | **Yes** (after P1) |
| P3-WINPATH | 3 | igx-tester | P3-QUAL | Promotion tests pass short basetemp | **Yes** (harness only) |
| P4-DOC | 4 | igx-implementer | P3-QUAL | CLOSE docs updated; no forbidden statuses | **No** until P3 |

---

## Risks and escalation

- **Re-pin / hook bypass / test skip:** out of scope; escalate `BLOCKED_AUTHORITY`.
- **043-M on `development` while Part C only on `int`:** escalate scope/tor mismatch (`/igx-autonomy` scope).
- **Parallel QUAL on moving SHAs:** forbidden; one SHA for Phase 3.

## Tests (Phase 3 summary)

```text
uv run pytest tests/unit/runtime/architecture/test_diag_foundation_5_destructive_proof.py
uv run pytest tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py
uv run pytest tests/unit/governance/test_verify_ent_audit_branch_protection.py tests/qualification/ent_audit_x/unit/test_od_ent_02_verifiers.py
uv run pytest tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py
# promotion trio: same modules, pytest --basetemp=<short path> on Windows
```