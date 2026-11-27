> **AF-W2-SCOPE-CLOSE-01:** **SCOPE=CLOSED**; **SCOPE_PREDECESSORS_SATISFIED=12/12**; **RW_PLAN_AUDIT=PASS** @ `73835a889fcebe4393113abd25d9cf9aeceda7c1`; **W3_ENTRY=AUTHORIZED**; **G2=OPEN**; **RED_FIRST_EVIDENCE=INCOMPLETE**.

# SCOPE hard predecessor index (WP-12)

**Tor wykonawczy:** `origin/development`. **DAG SSOT:** [`WP12_DAG.md`](../WP12_DAG.md) — **12** hard SCOPE predecessors.
**Formal closeout:** [`SCOPE_FORMAL_CLOSEOUT_RECORD.md`](../SCOPE_FORMAL_CLOSEOUT_RECORD.md)

This index cites in-repo evidence only. It records predecessor settlement for **SCOPE** closure; it does **not** set `G2`, `FCS`, `M3`, or `ARCH_FREEZE_CLOSED`.

## Hard predecessors

| Node | Evidence file(s) | Evidence SHA | Train ancestry | Disposition |
| --- | --- | --- | --- | --- |
| **M1** | [`WP12_DAG.md`](../WP12_DAG.md) L67; [`ARCH_FREEZE_NODE_LEDGER.md`](../ARCH_FREEZE_NODE_LEDGER.md) | `feaa2a87d41c3c8ae95724590291f360d0232016` | ON_TRAIN | **SATISFIED** |
| **DS-1** | ODR + eqo_a remediation | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | **SATISFIED** |
| **DS-2** | ODR session 2 | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | **SATISFIED** |
| **DS-3** | ODR session 3 | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | **SATISFIED** |
| **DS-4** | DS4 record + ODR | `8aaa178a5d45736ed51efe7c03b638ce5e253e37` | ON_TRAIN | **SATISFIED** |
| **DS-EXC** | ODR + residual package | `743087d57c674100dc9119959dc875271f70e72c` | ON_TRAIN | **SATISFIED** |
| **PT-V04** | [`continuation/PT_V04_REPLAY.md`](PT_V04_REPLAY.md) | `29fedd73f17deeb8b5cbc83f12269e80335813eb` | ON_TRAIN | **SATISFIED** — diagnostic |
| **PT-B01** | [`continuation/CAMPAIGN_B_DIAGNOSTIC_ACCEPTANCE_RECORD.md`](CAMPAIGN_B_DIAGNOSTIC_ACCEPTANCE_RECORD.md) | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` | ON_TRAIN (B) | **SATISFIED** — diagnostic |
| **PT-B02** | [`continuation/CAMPAIGN_B_DIAGNOSTIC_ACCEPTANCE_RECORD.md`](CAMPAIGN_B_DIAGNOSTIC_ACCEPTANCE_RECORD.md) | `7960ae7f6c82da60ca9acdc0ceebf63810002a42` | ON_TRAIN (B) | **SATISFIED** — diagnostic |
| **REV-MASTER** | [`REV_MASTER_FORMAL_CLOSEOUT_RECORD.md`](../REV_MASTER_FORMAL_CLOSEOUT_RECORD.md) | `deea67621120403edb2d9bdd33e3eed8903199de` | ON_TRAIN | **SATISFIED** |
| **T-AF2-RW-PLAN** | [`RW_PLAN.md`](../RW_PLAN.md); [`rw_plan.v1.json`](../rw_plan.v1.json) | `73835a889fcebe4393113abd25d9cf9aeceda7c1` | ON_TRAIN | **SATISFIED** |
| **T-AF2-RW-PLAN-AUDIT** | [`continuation/RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md) (AF-W2-INDEPENDENT-AUDIT-02) | `73835a889fcebe4393113abd25d9cf9aeceda7c1` | ON_TRAIN | **SATISFIED** — independent PASS |

## SCOPE predecessor reconciliation (AF-W2-SCOPE-CLOSE-01)

| Metric | Value |
| --- | ---: |
| Hard predecessors (canonical set) | **12** |
| Predecessors formally satisfied | **12** |
| Predecessors pending | **0** |
| `RW_PLAN_AUDIT` (published) | **PASS** |
| `SCOPE` milestone | **CLOSED** |
| `W3_ENTRY` | **AUTHORIZED** |
| `OP_B_PENDING` | **12** (downstream; not SCOPE predecessors) |
| `RED_FIRST_EVIDENCE` | **INCOMPLETE** |

**Historical:** AF-W2-GATE-BIND-01 reconciliation (SCOPE=OPEN) superseded by this publication after independent audit round 2.

`rw_plan.v1.json` `dependencies.SCOPE` matches the same 12-node set.
