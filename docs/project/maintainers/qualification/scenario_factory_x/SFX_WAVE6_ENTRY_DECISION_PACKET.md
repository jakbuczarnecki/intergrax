# Wave 6 Entry Operator Decision Packet (`SFX-X` · `SEF-X` · `SCENARIO-READINESS-X`)

| Field | Value |
|---|---|
| **Prepared By** | Enterprise Campaign Wave 2 (`T-IGX-090-D`, `OP-EC-01` / `OD-W2-SFX-PREP=A`) |
| **Date** | 2026-11-27 |
| **Purpose** | Single consolidated Operator decision packet to unblock `GATE-SFX-ENTRY`, `GATE-SEF-CONSUME`, and Day-1 Wave 6 launch immediately upon `ARCH-FREEZE = CLOSED` |
| **Product Code Delta** | `0` (decision-prep documentation only) |
| **Pre-Condition to Implementation** | **`ARCH-FREEZE = CLOSED`** (`FREEZE_CERTIFICATION_SHA` pinned) + Operator signatures on Tier A decisions below |

> **Non-Claims:** This packet authorizes **NO** implementation, declares **NO** `GATE-SFX-ENTRY PASS`, **NO** `SFX_PRE_IMPLEMENTATION_READY`, **NO** `CLOSED`, **NO** `IIAC PASS`, **NO** `FRZ PASS`, and **NO** `G6 PASS`. `ADR-SFX-001..006` and `ADR-SEF-001` remain **`Proposed`** until Operator signature.

---

## 1. Tier A — Hard `GATE-SFX-ENTRY` Blockers (MUST Sign Before Wave 6 `SFX-W3`)

Signing these 4 decisions (after `ARCH-FREEZE = CLOSED` and inventory re-pin `E-6`) satisfies the operator-decision gates for `GATE-SFX-ENTRY`:

| ID | Decision Topic | Recommended Option (SSOT) | Readiness State | Blocks |
|---|---|---|---|---|
| **`OD-SFX-05`** | Node LTS toolchain, `npm` + committed `web/package-lock.json`, and IIAC G0 treatment of `package-lock.json` (`GAP-24`, `ADR-SFX-003 §8 S-5`) | **Option (a):** Adopt Node LTS + `npm` + committed `applications/scenario_studio_application/web/package-lock.json`, registered as an enumerated IIAC neutral row (zero `scripts/` change, zero `C-AF-01` impact). Path-filtered `.github/workflows/scenario-studio-web.yml`; `unit-tests.yml` stays Node-free. | Ready for signature | `GATE-SFX-ENTRY`, `SFX-P5-06`, `SFX-P10-04` |
| **`OD-SFX-06`** | Formal acceptance of **`ADR-SFX-001`..`ADR-SFX-006`** (`Proposed` -> `Accepted`), settling `C-ECAP-01` (no ECAP §16 revision under Option B), `C-OD-02` (sidecar catalog), `ME-1/2/3`, `OQ-002-1..6`, `UX-Q-04/05` | **Accept all 6 ADRs.** Pre-acceptance prerequisites completed in `T-IGX-090-D`: independent `W2-SFX-REV` review recorded (`ACCEPT_WITH_MINOR_DOC_PATCH`, 0 BLOCKER/CRITICAL/HIGH), all 7 MEDIUM + 5 LOW patches applied (incl. `GAP-23` citation & `WT-12` 30-array + 4-tightening schema closure). See [`SFX_ADR_001_006_INDEPENDENT_REVIEW_RECORD.md`](SFX_ADR_001_006_INDEPENDENT_REVIEW_RECORD.md). | Ready for signature | `GATE-SFX-ENTRY`, all `SFX-W3+` |
| **`OD-SFX-09`** | IIAC per-stage activation & per-wave `REBIND-Wn` scope (coordinated with `OD-SFX-05` lockfile row and `C-IIAC-01`) | **Approve per-wave `REBIND-Wn`:** ECAP A1–A8 + D1–D6 + IIAC ACT-N11 product-path re-bind at close of each wave `W3..W8`. | Ready for signature | `GATE-SFX-ENTRY`, every `W3..W8` wave |
| **`OD-SFX-14`** | PR-pipeline SAST & dependency scanning (`semgrep`, `ruff S`, `pip-audit`, `npm audit`) (`GAP-13`) | **Option (a):** SFX-scoped PR checks first (generated scenario packages + `applications/scenario_studio_application/web/`), without disrupting global repository PR gates. | Ready for signature | `GATE-SFX-ENTRY`, `SFX-P9-03`, `SFX-P10-04` |

---

## 2. Tier B — High-Priority Pre-W3 & Cross-Track Enablers (Sign in Same Operator Session)

| ID | Decision Topic | Recommended Option (SSOT) | Why Sign Before W3 | Blocks |
|---|---|---|---|---|
| **`OD-SFX-10`** | One-way cross-track dependency edge `SFX -> SEF-X` + register SFX as a named SEF intake consumer (`C-SEF-01`) | **Approve one-way edge now:** SFX consumes SEF-X (`CON-N-01..05`, `CON-E-01/02/04/05`) by reference only with HO-2 human submit preserved; SEF-X takes zero reverse dependency on SFX. | Lets `SEF-X` P2/P3 include the SFX intake consumer contract from Day 1 without rework. | **`GATE-SEF-CONSUME`**, `SFX-P8-03`, W7 agentic runs, SFX closure |
| **`OD-SFX-17`** *(proposed)* | Approval & HO-2 recording doors and MCP API-key principal attribution (`SFX-GAP-25`, `UX-Q-01`, `ADR-SFX-004` `R-004-03`) | **Option (b):** Every door may act ONLY with a human-attributable IdP bearer principal for L2 execution, artifact approvals, and HO-2 submissions; API-key MCP callers remain strictly capped at L0/L1; fail-closed auth (`fails_closed=True`, `HARNESS_BOUND`, `tenant_required=True`). | Locks `SFX-CON-13` and PC-3/PC-6 operation map in **`SFX-P5-03`** (W3). | `SFX-P5-03` (W3), `SFX-P10-02` |
| **`OD-SFX-18`** *(proposed)* | Self-approval ban & distinct human gate for `SANDBOX_CERTIFIED -> ADMISSION_DOSSIER_READY` (`DOSSIER_EXPORT`) (`UX-Q-02/03`, `ADR-SFX-002`, `ADR-SFX-006`) | **Approve:** Enforce `requester != approver` (no self-approval) on gated transitions and require a distinct human approver binding for `DOSSIER_EXPORT` (`SANDBOX_CERTIFIED -> ADMISSION_DOSSIER_READY`). | Locks `SFX-CON-13`/`14`/`17` FSM checks in **`SFX-P5-03`** (W3). | `SFX-P5-03` (W3), `SFX-P11-04` |

### Secondary Tier B / Deferred Tier C SFX Items (For Operator Awareness / Batch Triage)

- **Tier B (optional in same session):** `OD-SFX-07` (numeric targets for TTFS, SUS, task success, `G-1..G-5`), `OD-SFX-08` (include `REF-2` with non-overlap guard against READINESS settlement per `C-RDY-01`, plus `REF-3`), `OD-SFX-20` (docs-only handoff of 4 platform-owned seams: `GAP-15` S1–S6, `GAP-18` NoOp JWT, `GAP-19` scaffold wildcard CORS, `GAP-26/27` Cursor F-01/F-02).
- **Tier C (post-W3 / later):** `OD-SFX-19` (synthetic-only document upload behind `SFX-P9-05`), `OD-SFX-R2-01` (bounded L2 autonomy envelope — paired with `OD-SEF-R2`, NOT required for assisted REF-1 POC), `OD-SFX-R2-03` (confirm REF-1 first commercial slice), `OD-SFX-R2-04` (internal sandbox demo with `NOT PRODUCTION` label), `OD-SFX-13` (per-scenario production admission owner — external to SFX), `OD-SFX-11` (SFX closure at W8), `OD-SFX-12` (**decline** pulling `SFX-P4-B` pre-freeze; keep in W3).
- **Bookkeeping:** Optionally record formal signature on already-recorded `OD-SFX-21` and `OD-SFX-R2-GOV-01` (`T-IGX-070-C`).

---

## 3. `SEF-X` Cross-Track Conflicts (`C-SEF-01..04`) & Open SEF/ECAP Decisions

`SFX-W3`, `SFX-W4-CORE`, `SFX-W5-BASE` (excl. `R2-W5-03`), and `SFX-W6` do **NOT** wait on `GATE-SEF-CONSUME` (~75% of SFX runs in parallel with `SEF-X` P2–P12). However, `GATE-SEF-CONSUME`, `SFX-P8-03` (`AGENTIC_ENGINEERING`), `REF1_ASSISTED_SANDBOX_POC`, and `GATE-SFX-FULL` require resolving the 4 conflicts below in `SEF-X`:

| Conflict ID | Nature of Conflict | Required Resolution | Owner / Target Decision |
|---|---|---|---|
| **`C-SEF-01`** | SEF `ASF-SEF-MASTER-PLAN.md` names no external intake consumer (`CON-N-01` is human submit only). | Register SFX as a named intake consumer under **`OD-SEF-07`** + **`OD-SFX-10`**, preserving HO-2 human submit (`operator_id` = human approver; SFX never auto-submits). | SEF + Operator (`OD-SEF-07`, `OD-SFX-10`) |
| **`C-SEF-02`** | SEF `T-P5-004` (network egress policy, `GAP-MED-07`) is currently `OPTIONAL` and excluded from SEF certification. | Promote `T-P5-004` from `OPTIONAL` to **MANDATORY** for `GATE-SEF-CONSUME` via a small docs amendment to `ASF-SEF-MASTER-PLAN.md` before SEF W2. | SEF + Operator |
| **`C-SEF-03`** | SEF `CON-E-04` (`VerificationRecord`, `ASF-SEF-MASTER-PLAN.md:372`) has no assigned WBS task among the 82 SEF tasks. | Assign an explicit SEF WBS task for `CON-E-04` in SEF P2/P3 before `T-IGX-SEF-B` approval. | SEF + Operator |
| **`C-SEF-04`** | `GAP-HIGH-07` (`SEF <-> READINESS` Delivery Kit coupling) is OPEN. | Keep SFX deterministic scaffold strictly on existing scaffold tools (`intergrax/scaffold/cli.py`, `scripts/proof/init_scenario_implementation.py`) until `SCENARIO-READINESS-X` closes its Delivery Kit analysis (`C-RDY-01`). | SEF + READINESS |

### Open SEF / ECAP Operator Decisions Paired with Wave 6 Entry

- **`ADR-SEF-001` Acceptance** (`Proposed` -> `Accepted`) — unblocks `SEF-X` P2/P3 (`T-IGX-SEF-B`).
- **`OD-SEF-04`** (Engineering FSM <-> DevOS ticket binding, `T-P3-001`) and **`OD-SEF-05`** (provider-neutral `CodingExecutionBackend` allowlist, `T-P4`).
- **`OD-SEF-07`** (READINESS handoff + `C-SEF-01` SFX intake consumer registration).
- **`OD-ECAP-16`** (`C-ECAP-01`: confirm ECAP §16 does not apply to Tier-2 SFX under Option B; may be confirmed directly inside `OD-SFX-06`).
- *(Later, not blocking Wave 6 start or assisted REF-1 POC:)* `OD-SEF-06` (scoped B6 auditor), `OD-SEF-08` (SEF closure), `OD-SEF-R2` (delegated `CON-N-01` v2 grant for `BOUNDED_AUTONOMOUS`).

---

## 4. Remaining Operator Pre-Freeze Hygiene Prerequisites (Zero Product Delta)

Before declaring `GATE-SFX-ENTRY` after `ARCH-FREEZE = CLOSED`:

1. **`PF-3` (Operator IDE Toolkit v0 Walkthrough):** Execute the Operator walkthrough of IDE toolkit v0 on REF-1 problem text and record sign-off in [`SFX_IDE_TOOLKIT_V0_WALKTHROUGH.md`](SFX_IDE_TOOLKIT_V0_WALKTHROUGH.md) (`SFX-P4-A01` acceptance criterion).
2. **`PF-4` (`GEN_SET.json` Custody Relocation):** Relocate `D:\Projekty\_wt\_scratch\sfx-gen-set\GEN_SET.{json,md}` to an operator-restricted directory outside all agent workspaces, re-verify SHA-256 `8b6ff8eb...3e6e0d`, and update [`SFX_GEN_SET_SEAL.md`](SFX_GEN_SET_SEAL.md).
3. **`E-6` Inventory Re-Pin at `FREEZE_CERTIFICATION_SHA`:** Re-pin [`SFX_CAPABILITY_INVENTORY.json`](SFX_CAPABILITY_INVENTORY.json) (currently pinned at `eadb1c21`, pre-PDC-X closure) at `FREEZE_CERTIFICATION_SHA`, verifying Collaborative Work (`CAP-08, 30, 31`) and `CAP-15` (`GAP-22`) qualification status.
4. **`SCENARIO-READINESS-X` Parallel Fast-Start (`T-IGX-RDY-A`):** Upon `ARCH-FREEZE = CLOSED`, Operator approves [`governance/tickets/T-IGX-RDY-A.md`](../../../../../governance/tickets/T-IGX-RDY-A.md) (`PLAN_T-IGX-RDY-A.md`) to immediately start the sole mandatory predecessor of **`SCENARIO-GATE`** in parallel with `SEF-X` W1 and `SFX-W3`.
