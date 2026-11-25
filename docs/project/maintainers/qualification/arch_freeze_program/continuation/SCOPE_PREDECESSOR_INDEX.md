# SCOPE hard predecessor index (WP-12)

**Train:** `arch-freeze/fc-train` @ `689b277e62db18613a415e6b2e404864412112b8`  
**DAG SSOT:** [`WP12_DAG.md`](../WP12_DAG.md) SCOPE row (hard predecessors).  
**Ancestry rule:** `git merge-base --is-ancestor <evidence_sha> 689b277e6` → **ON_TRAIN**; else **EVIDENCE_NOT_ON_TRAIN**.

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
| **PT-B01** | [`continuation/CAMPAIGN_B_ALPHA_REPLAY.md`](CAMPAIGN_B_ALPHA_REPLAY.md) | `3991186b1e388efef28e6a1be09aabc32e7fc04d` (HEAD_DIAGNOSTIC execution) | ON_TRAIN | Completed diagnostic: 9/9 manifest cases PASS |
| **PT-B02** | [`continuation/CAMPAIGN_B_ALPHA_REPLAY.md`](CAMPAIGN_B_ALPHA_REPLAY.md); [`continuation/CAMPAIGN_B_S03_DISPOSITION.md`](CAMPAIGN_B_S03_DISPOSITION.md); [`continuation/CAMPAIGN_B_B02_UV_RERUN.md`](CAMPAIGN_B_B02_UV_RERUN.md); S03 product gate fix | `3991186b1e388efef28e6a1be09aabc32e7fc04d` (alpha replay); S03 remediation `4b7acac4ba32326b32f446e23d624f91393d963e`; S03 scaffold **PASS_WITH_UV_ON_PATH** @ `689b277e62db18613a415e6b2e404864412112b8` | ON_TRAIN | Completed diagnostic at replay SHA (6/7 PASS, S03 **FAIL** at `3991186b1`); pip setup **ERROR** nodes classed **missing_extra** (environment); Nexus importer product fix on train @ `4b7acac4` — **no alpha re-run** after fix in-repo; scoped S03 scaffold remeasure **PASS_WITH_UV_ON_PATH** when `%USERPROFILE%\.local\bin` is on `PATH` (train @ `689b277e6`, no product delta); full phase **b2** `run_campaign_b_w2_alpha.py` campaign has **no exit-0 record** (killed prior run only — not a qualification predecessor; **not** rerun) |
| **REV-MASTER** | [`REV_MASTER_FORMAL_CLOSEOUT_RECORD.md`](../REV_MASTER_FORMAL_CLOSEOUT_RECORD.md) | `deea67621120403edb2d9bdd33e3eed8903199de` | ON_TRAIN | Formal pass (`REV_MASTER_FORMAL_PASSED=YES`, AF-02F) |
| **T-AF2-RW-PLAN** | [`RW_PLAN.md`](../RW_PLAN.md); [`rw_plan.v1.json`](../rw_plan.v1.json) | `bd6a5e190ef2dec7227fefb2c9304b60cb7fca9f` (canonical publish); execution cite `deea67621120403edb2d9bdd33e3eed8903199de` in `RW_PLAN.md` | ON_TRAIN | Published canonical machine + human output (W2 DOC node); ledger still **PARTIAL** on RED_FIRST row inventory |
| **T-AF2-RW-PLAN-AUDIT** | [`continuation/RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md); [`continuation/RW_PLAN_AUDIT_EVIDENCE_INDEX.md`](RW_PLAN_AUDIT_EVIDENCE_INDEX.md) | `ebe45c26930a9291a3df2d6a75028c35c036cee2` (AR-12-07 independent session) | ON_TRAIN | Formal pass (RM-03 structural **PASS**); RED_FIRST evidence class disposition in [`RED_FIRST_G2_DISPOSITION.md`](RED_FIRST_G2_DISPOSITION.md) @ `8c0158ac8152a7ff97734b34faf37c25682a77be` (9× RED_AT_PREFIX, 8× RED_FIRST_UNREPRODUCIBLE — **do not block** SCOPE measurement) |

## Supplementary / branch-only cites (not primary train evidence)

| Cite | SHA | Train ancestry | Note |
| --- | --- | --- | --- |
| Campaign B alpha branch replay | `e4857f515b4271966963675f2b04234ec1907a58` (`af-fc/b-alpha`) | **EVIDENCE_NOT_ON_TRAIN** | Superseded for index by in-repo [`CAMPAIGN_B_ALPHA_REPLAY.md`](CAMPAIGN_B_ALPHA_REPLAY.md) @ `3991186b1` (ON_TRAIN); file present on train — no copy required |

## PT-V04 ROADMAP-REPLAY-X S1 baseline (train @ `29fedd73f`)

Failures recorded in [`PT_V04_REPLAY.md`](PT_V04_REPLAY.md) remain the S1 baseline for PT-Q04/PT-Q05. They are not repaired on `29fedd73f`.

## Campaign B S03 scaffold @ train tip (`689b277e6`)

S03 scaffold node `test_harness_01_new_agent_scaffold_generates_nexus_free_authoring_package`: **PASS_WITH_UV_ON_PATH** @ `689b277e62db18613a415e6b2e404864412112b8` (evidence [`CAMPAIGN_B_B02_UV_RERUN.md`](CAMPAIGN_B_B02_UV_RERUN.md); measured from base `66d0aef63af462535bac75ef377497b436844b8f`, 19.02s, exit 0). The outer phase **b2** campaign process has **no exit-0 record** and is not cited as a hard predecessor. No G2, FCS, M3, or ARCH-FREEZE CLOSED claim.

SCOPE_MILESTONE: NOT_SET
