> **AF-W2-FINAL-01:** OD-AF-CLOSE-B6=DECIDED (variant A); RW_PLAN_AUDIT=NOT_PERFORMED; SCOPE=OPEN; G2=OPEN.
# SCOPE hard predecessor index (WP-12)

> **Obowiązujący stan (AF-W2-CLOSURE-01, Appendix V6):** gałąź wykonawcza **`development`** @ `f4e8444551994d9033c00fd6884d5e6f822d3ed7`; **`SCOPE=OPEN`**; **`RED_FIRST_EVIDENCE=INCOMPLETE`**; **`RW_PLAN_AUDIT=NOT_PERFORMED`**. Wpis 2026-11-26 z `RED_FIRST_EVIDENCE=COMPLETE` jest historyczny. **`SCOPE_MILESTONE: READY_TO_RECORD` nie obowiązuje.**

**Tor wykonawczy:** `origin/development` (W3). **Historyczne tory** `arch-freeze/fc-train` / `arch-freeze/rw-train` — cytat only @ `689b277e6` / starsze piny.
**DAG SSOT:** [`WP12_DAG.md`](../WP12_DAG.md) — **12** hard SCOPE predecessors (`M1`, `DS-1`..`DS-4`, `DS-EXC`, `PT-V04`, `PT-B01`, `PT-B02`, `REV-MASTER`, `T-AF2-RW-PLAN`, `T-AF2-RW-PLAN-AUDIT`).
**Ancestry:** dla dowodów na `development`, `merge-base --is-ancestor <sha> f4e8444`; dla cytatów train — ON_TRAIN względem wskazanego train tip.

This index cites in-repo evidence only. It does not set `SCOPE`, `G2`, `FCS`, `M3`, or `ARCH_FREEZE_CLOSED`.

## Hard predecessors

| Node | Evidence file(s) | Evidence SHA | Train ancestry | Disposition |
| --- | --- | --- | --- | --- |
| **M1** | [`WP12_DAG.md`](../WP12_DAG.md) L67; [`ARCH_FREEZE_NODE_LEDGER.md`](../ARCH_FREEZE_NODE_LEDGER.md) L69 | `feaa2a87d41c3c8ae95724590291f360d0232016` (PDC P9 closure publication) | ON_TRAIN | Formal pass recorded (M1 REACHED 2026-11-16) |
| **DS-1** | [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](../../ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md) L82; [`eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md`](../../eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md) (PR #615) | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | Formal pass (DECIDED, DP-01..12) |
| **DS-2** | ODR L83; Operator record session 2 (PR #615) | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | Formal pass (DECIDED, DP-13..23) |
| **DS-3** | ODR L84; Operator record session 3 (PR #615) | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | Formal pass (DECIDED, DP-24..33) |
| **DS-4** | [`DS4_DECISION_RECONCILIATION_RECORD.md`](../DS4_DECISION_RECONCILIATION_RECORD.md); ODR L85 | `8aaa178a5d45736ed51efe7c03b638ce5e253e37` | ON_TRAIN | Formal pass (DECIDED + DS4_DAG_COMPLETE AF-02D) |
| **DS-EXC** | ODR L30 (DECIDED); [`PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md`](../../PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md) | `743087d57c674100dc9119959dc875271f70e72c` (decision); package intro `4e256292bc7f15d38fd53311d43ad39ad5665ba2` | ON_TRAIN | Formal pass (Operator disposition per residual package) |
| **PT-V04** | [`continuation/PT_V04_REPLAY.md`](PT_V04_REPLAY.md) | `29fedd73f17deeb8b5cbc83f12269e80335813eb` (train replay record) | ON_TRAIN | Completed diagnostic (non-certifying S1): F10/STATE-X/TRACE-X/tenant_x **PASS**; ROADMAP-REPLAY-X **3 failures** (`test_replay_x_prod_delta_empty_since_dup_x`, `test_replay_x_closure_counters_computed`, `test_replay_x_b6_r2_reconciliation_zero`) — S1 baseline for PT-Q04/PT-Q05; **no product fix** on this commit |
| **PT-B01** | [`continuation/CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md); [`continuation/CAMPAIGN_B_MISSING_CASE_MAP.md`](CAMPAIGN_B_MISSING_CASE_MAP.md) (28/28 matrix) | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (baseline **B** `HISTORICAL_ALPHA`); HEAD supplements @ `c94745b` (R01a/R02/R05 only) | ON_TRAIN (B cited); HEAD supplementary | **ACCEPTED_DIAGNOSTIC_INPUT** @ B (AF-W2-FINAL-01 record); full case matrix; **not** formally CLOSED |
| **PT-B02** | [`continuation/CAMPAIGN_B_V5_ALPHA.md`](CAMPAIGN_B_V5_ALPHA.md); [`continuation/CAMPAIGN_B_MISSING_CASE_MAP.md`](CAMPAIGN_B_MISSING_CASE_MAP.md); [`continuation/CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md`](CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md) | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (baseline **B**); S03 node **exit 0** / 127 passed (closure-02) | ON_TRAIN (B) | **PARTIAL_DIAGNOSTIC** @ B — **9/9** cases in matrix; HEAD_DIAGNOSTIC (e.g. alpha replay @ `3991186b1`) remains **supplementary**; **not** formally CLOSED |
| **REV-MASTER** | [`REV_MASTER_FORMAL_CLOSEOUT_RECORD.md`](../REV_MASTER_FORMAL_CLOSEOUT_RECORD.md) | `deea67621120403edb2d9bdd33e3eed8903199de` | ON_TRAIN | Formal pass (`REV_MASTER_FORMAL_PASSED=YES`, AF-02F) |
| **T-AF2-RW-PLAN** | [`RW_PLAN.md`](../RW_PLAN.md); [`rw_plan.v1.json`](../rw_plan.v1.json) | `bd6a5e190ef2dec7227fefb2c9304b60cb7fca9f` (canonical publish); execution cite `deea67621120403edb2d9bdd33e3eed8903199de` in `RW_PLAN.md` | ON_TRAIN | Published canonical machine + human output (W2 DOC node); ledger still **PARTIAL** on RED_FIRST row inventory |
| **T-AF2-RW-PLAN-AUDIT** | [`continuation/RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md); GA.2R [`continuation/RED_FIRST_GA2R_RESULT.md`](RED_FIRST_GA2R_RESULT.md) | `c94745b11b35a3be55e4e9cb6502befc60262e26` (`development`); historyczny RM-03 @ `ebe45c269` (strukturalny tylko) | ON_TRAIN (historyczny RM-03); GA.2R + OP-B on `development` | **NOT_PERFORMED** — RM-03 **nie jest** końcową akceptacją; independent round 2 required |

## Supplementary / branch-only cites (not primary train evidence)

| Cite | SHA | Train ancestry | Note |
| --- | --- | --- | --- |
| Campaign B alpha branch replay | `e4857f515b4271966963675f2b04234ec1907a58` (`af-fc/b-alpha`) | **EVIDENCE_NOT_ON_TRAIN** | Superseded for index by in-repo [`CAMPAIGN_B_ALPHA_REPLAY.md`](CAMPAIGN_B_ALPHA_REPLAY.md) @ `3991186b1` (ON_TRAIN); file present on train — no copy required |

## PT-V04 ROADMAP-REPLAY-X S1 baseline (train @ `29fedd73f`)

Failures recorded in [`PT_V04_REPLAY.md`](PT_V04_REPLAY.md) remain the S1 baseline for PT-Q04/PT-Q05. They are not repaired on `29fedd73f`.

## Campaign B S03 scaffold @ train tip (`689b277e6`)

S03 scaffold node `test_harness_01_new_agent_scaffold_generates_nexus_free_authoring_package`: **PASS_WITH_UV_ON_PATH** @ `689b277e62db18613a415e6b2e404864412112b8` (evidence [`CAMPAIGN_B_B02_UV_RERUN.md`](CAMPAIGN_B_B02_UV_RERUN.md); measured from base `66d0aef63af462535bac75ef377497b436844b8f`, 19.02s, exit 0). The outer phase **b2** campaign process has **no exit-0 record** and is not cited as a hard predecessor. No G2, FCS, M3, or ARCH-FREEZE CLOSED claim.

## Phase-b2 campaign process waiver (historical only — not current authorization)

The killed phase-b2 `run_campaign_b_w2_alpha.py` process (exit `4294967295`, ~56 minutes, no campaign summary) was previously recorded as a **process waiver**. **AF-W2-CLOSURE-02:** that waiver is **historical orchestrator text** and **not** current SCOPE authorization. **PT-B02** is cited via baseline **B** diagnostic matrix ([`CAMPAIGN_B_MISSING_CASE_MAP.md`](CAMPAIGN_B_MISSING_CASE_MAP.md)) and targeted S03 measurement ([`CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md`](CAMPAIGN_B_W2_CLOSURE_02_TARGETED.md)). **SCOPE remains OPEN.**

**SCOPE_MILESTONE:** not set — **SCOPE remains OPEN**.

## SCOPE predecessor reconciliation (AF-W2-FINAL-01)

| Metric | Value |
| --- | ---: |
| Hard predecessors (declared set) | **12** |
| Accounted in this index (evidence cited) | **12** |
| Awaiting independent acceptance | **3** (`PT-B01`, `PT-B02`, `T-AF2-RW-PLAN-AUDIT`) |
| **SCOPE** | **OPEN** |

`rw_plan.v1.json` `dependencies.SCOPE` matches the same 12-node set. No DAG edges added or removed in this package.
