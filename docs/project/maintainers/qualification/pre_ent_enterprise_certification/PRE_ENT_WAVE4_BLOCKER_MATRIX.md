# PRE-ENT Wave 4 — blocker matrix

**Campaign:** `pre_ent_enterprise_certification`  
**Snapshot @ CERT_DOCS_HEAD:** `03ce1d6f380f92bc86cfadac9a88fcd6473e5ec3`  
**Product evidence pin:** `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` (`AUDIT_HEAD`)  
**Date:** 2026-11-10  

**Purpose:** rows that **block** or **scope-limit** formal PRE-ENT closure and **T-IGX-034-I** matrix publication (Wave 5). Wave 4 certifier sessions on the five ready predecessors proceed **in spite of** these rows for ticket-scoped ACCEPT only — they do **not** clear global blockers.

Operator §7 baseline: [`PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md`](PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md) §4, [`PRE_ENT_CAMPAIGN_PROGRESS.md`](PRE_ENT_CAMPAIGN_PROGRESS.md).

| ID | Blocker | Owner | Campaign state | Wave 4 certifier impact | Close criterion |
| --- | --- | --- | --- | --- | --- |
| **G-SKILL** | SKILL-X §3.0.4 (upgrade/disable/remove) beyond 032-A pinning | **Operator** | **BLOCKED** — D-PREENT-02=defer | 032-A certifier: **scoped** KV pinning only; no SKILL-X parent ACCEPT | **G-SKILL-A** or **G-SKILL-B** + signed `Approved` line |
| **F10** | TES ladder pin `496016c0cf6cc5cf1f9a29d885677605dd0415d3`; 182 TREE_DRIFT / 0 SEMANTIC @ campaign tree | **Operator + EBH-7** | **BLOCKED** — D-PREENT-04=block | No 034-F certifier row; EBH-7 rollup blocked | Controlled TES advance or formal scoped N/A — [`WAVE1_F10_PINNED_REPLAY.md`](WAVE1_F10_PINNED_REPLAY.md), `T-IGX-034-F.md` |
| **OD-HIST04-A** | EVOL-X formal ratification absent | **Operator** | **DEFER** — D-PREENT-06 | `FORMAL_AUTHORITY_INCOMPLETE`; 030-C scoped ACCEPT OK | Operator ratification record (no PRE-ENT material IIAC edits) |
| **OD-HIST04-B** | Amendment 01 vs IIAC material paths | **Governance** | **OPEN** | Campaign does not edit 6 IIAC paths; certifier notes limitation | Governance decision; IIAC record `8045915a` |
| **OD-HIST04-C** | TOOL-LIFE-X roadmap label vs open disposition | **Operator** | **DEFER** — D-PREENT-06 | 030-C certifier: ACCEPT **SCOPED**; no parent TOOL-LIFE-X stage CLOSED | Operator label alignment or scoped acceptance |
| **Pop E** | ENT-AUDIT-X enterprise audit Population E | **ENT-AUDIT-X** | **OUT OF PRE-ENT** — D-PREENT-03=b | 034-G limited to Pop **B–D**; no Pop E cert in PRE-ENT | ENT owner rollup + FRZ-AUD disposition |
| **034-C** | HIST-F-DUR-01 / durable pinning semantics | **Operator** | **DEFER** — D-PREENT-07 | Out of Wave 4 five-row queue; blocks matrix honesty for DUR family | Implement / accept residual / defer decision |
| **034-D** | TOOL-LIFE invoker qual-only (030-C on HEAD) | **Campaign** | **NOT STARTED** | Optional qual batch before matrix | Qual log + auditor if material gap found |
| **034-G** | QUAL-X Pop B–D pinned HEAD | **Campaign** | **NOT STARTED** | Required before Wave 5 matrix for qual populations | Serial pytest per `T-IGX-034-G.md` |
| **034-H** | DIST-SCALE §3.0.6 vs §5 + Q-03 audit | **Campaign** | **NOT STARTED** | Matrix row BLOCKED until audit | Independent audit or scoped disposition |

## Wave 3 → Wave 4 dependency

| Wave 3 auditor PASS (5 tickets) | Clears blocker? |
| --- | --- |
| 034-B, 034-E, 030-B, 030-C, 032-A | **No** — enables **certifier** queue only |
| Consolidated register | [`wave3_audits/WAVE3_AUDIT_CONSOLIDATED_REGISTER.md`](wave3_audits/WAVE3_AUDIT_CONSOLIDATED_REGISTER.md) |

## SSOT

- Certifier queue: [`PRE_ENT_WAVE4_CERTIFICATION_HANDOFF_QUEUE.md`](PRE_ENT_WAVE4_CERTIFICATION_HANDOFF_QUEUE.md)  
- Waves 5–6: [`PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md`](PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md)  
- Wave 0 routing: [`WAVE0_PRESENCE_RECONCILIATION.md`](WAVE0_PRESENCE_RECONCILIATION.md)
