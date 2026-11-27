> **AF-W2-GATE-BIND-01:** OD-AF-CLOSE-B6=DECIDED (variant A); RW_PLAN_AUDIT=NOT_PERFORMED; SCOPE=OPEN; G2=OPEN.

# SCOPE hard predecessor index (WP-12)

> **Obowiązujący stan (Appendix V6):** gałąź wykonawcza **`development`** @ `a92a023e1a238b95635dcecdc4e51adecdcbc916`; **`SCOPE=OPEN`**; **`RED_FIRST_EVIDENCE=INCOMPLETE`**; **`RW_PLAN_AUDIT=NOT_PERFORMED`**.

**Tor wykonawczy:** `origin/development` (W3). **Historyczne tory** `arch-freeze/fc-train` / `arch-freeze/rw-train` — cytat only; nie obowiązujący tor wykonawczy.
**DAG SSOT:** [`WP12_DAG.md`](../WP12_DAG.md) — **12** hard SCOPE predecessors (`M1`, `DS-1`..`DS-4`, `DS-EXC`, `PT-V04`, `PT-B01`, `PT-B02`, `REV-MASTER`, `T-AF2-RW-PLAN`, `T-AF2-RW-PLAN-AUDIT`).
**Ancestry:** dla dowodów na `development`, `merge-base --is-ancestor <sha> a92a023e`.

This index cites in-repo evidence only. It does not set `SCOPE`, `G2`, `FCS`, `M3`, or `ARCH_FREEZE_CLOSED`.

## Hard predecessors

| Node | Evidence file(s) | Evidence SHA | Train ancestry | Disposition |
| --- | --- | --- | --- | --- |
| **M1** | [`WP12_DAG.md`](../WP12_DAG.md) L67; [`ARCH_FREEZE_NODE_LEDGER.md`](../ARCH_FREEZE_NODE_LEDGER.md) L69 | `feaa2a87d41c3c8ae95724590291f360d0232016` | ON_TRAIN | Formal pass recorded (M1 REACHED 2026-11-16) |
| **DS-1** | ODR + eqo_a remediation | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | Formal pass |
| **DS-2** | ODR session 2 | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | Formal pass |
| **DS-3** | ODR session 3 | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | Formal pass |
| **DS-4** | DS4 record + ODR | `8aaa178a5d45736ed51efe7c03b638ce5e253e37` | ON_TRAIN | Formal pass |
| **DS-EXC** | ODR + residual package | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | Formal pass |
| **PT-V04** | [`continuation/PT_V04_REPLAY.md`](PT_V04_REPLAY.md) | `29fedd73f17deeb8b5cbc83f12269e80335813eb` | ON_TRAIN | Completed diagnostic (non-certifying S1) |
| **PT-B01** | [`continuation/CAMPAIGN_B_DIAGNOSTIC_ACCEPTANCE_RECORD.md`](CAMPAIGN_B_DIAGNOSTIC_ACCEPTANCE_RECORD.md); alpha + missing-case map | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` | ON_TRAIN (B) | **ACCEPTED_DIAGNOSTIC_INPUT** (planning only) |
| **PT-B02** | [`continuation/CAMPAIGN_B_DIAGNOSTIC_ACCEPTANCE_RECORD.md`](CAMPAIGN_B_DIAGNOSTIC_ACCEPTANCE_RECORD.md); closure-02 targeted | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` | ON_TRAIN (B) | **ACCEPTED_DIAGNOSTIC_INPUT** (planning only) |
| **REV-MASTER** | [`REV_MASTER_FORMAL_CLOSEOUT_RECORD.md`](../REV_MASTER_FORMAL_CLOSEOUT_RECORD.md) | `deea67621120403edb2d9bdd33e3eed8903199de` | ON_TRAIN | Formal pass |
| **T-AF2-RW-PLAN** | [`RW_PLAN.md`](../RW_PLAN.md); [`rw_plan.v1.json`](../rw_plan.v1.json) | `a92a023e1a238b95635dcecdc4e51adecdcbc916` | ON_TRAIN | Canonical publish; red-first inventory INCOMPLETE |
| **T-AF2-RW-PLAN-AUDIT** | [`continuation/RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md) | `a92a023e1a238b95635dcecdc4e51adecdcbc916` | ON_TRAIN | **NOT_PERFORMED** |

## SCOPE predecessor reconciliation (AF-W2-GATE-BIND-01)

| Metric | Value |
| --- | ---: |
| Hard predecessors (canonical set) | **12** |
| Predecessors with assigned materials | **12** |
| Planning / formal passes accounted | **9** |
| Accepted diagnostic input only | **2** (`PT-B01`, `PT-B02`) |
| Awaiting independent audit | **1** (`T-AF2-RW-PLAN-AUDIT`) |
| Formal closure still outstanding | **SCOPE**, **RW_PLAN_AUDIT**, **G2** |

11/12 material index coverage is **not** 11/12 formal gate satisfaction.

`rw_plan.v1.json` `dependencies.SCOPE` matches the same 12-node set.