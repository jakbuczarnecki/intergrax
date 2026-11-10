# PRE-ENT enterprise certification — final predecessor matrix (Wave 5)

**Campaign:** `pre_ent_enterprise_certification`  
**Ticket:** `governance/tickets/T-IGX-034-I.md`  
**Parent:** `governance/tickets/T-IGX-034-A.md`  
**Date:** 2026-11-10  
**Campaign status:** **NOT CLOSED** — terminal label **`PRE_ENT_PREDECESSORS_EXACT_BLOCKERS`** assigned @ W6-G3 (Wave 6 audit); **not** full Enterprise certification; `T-IGX-034-A` **not** CLOSED.

**Ceiling (unchanged):** Five Wave 4 certifier rows are **ACCEPT_SCOPED** only — not full independent certification, not FRZ promotion, not `ALL_PRE_ENT_PREDECESSORS_INDEPENDENTLY_CERTIFIED`. Terminal campaign label belongs to Wave 6 (W6-G3).

---

## 1. SHA ladder

| Pin | SHA | Role |
| --- | --- | --- |
| **BASE (W5 branch root)** | `6399a16574e3784261dd2473770f3bb8dc922716` | `development` @ Wave 5 entry |
| **AUDIT_HEAD** | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` | Product + Wave 3 auditor replay |
| **CERT_DOCS_HEAD** | `d00df592037404cd6c2b1bc95c00757544d551d8` | Wave 4 certifier handoff SSOT |
| **WAVE4_PUBLICATION_SHA** | `c4a3b751bf7d6716bb0003bed23ae20b1e8f910f` | Wave 4 register publication |
| **WAVE5_PUBLICATION_SHA** | `0f3b4976726dcbd3046cd888aa91703214f27997` | Wave 5 matrix + SSOT publication |
| **IIAC effective_from_sha** | `661978df191d9356058a1312a712fcfa3c890e9e` | IIAC `IN_FORCE` (record `8045915a` — not edited) |
| **F10 TES pin** | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` | D-PREENT-04=block replay |

---

## 2. Wave 4 certifier predecessors (ACCEPT_SCOPED — not promoted)

Source: [`wave4_certs/WAVE4_CERT_CONSOLIDATED_REGISTER.md`](wave4_certs/WAVE4_CERT_CONSOLIDATED_REGISTER.md) @ **AUDIT_HEAD**.

| Ticket | Verdict (ticket scope) | Wave 5 matrix note |
| --- | --- | --- |
| **T-IGX-034-B** | **ACCEPT SCOPED** | Does not clear **G-SKILL** or global BND parent |
| **T-IGX-034-E** | **ACCEPT SCOPED** | Does not clear ROADMAP-REPLAY-X parent closure |
| **T-IGX-030-B** | **ACCEPT SCOPED** | Does not clear EBH-4 parent |
| **T-IGX-030-C** | **ACCEPT SCOPED** | Does not clear **OD-HIST04-A/C** or TOOL-LIFE-X §3 parent label |
| **T-IGX-032-A** | **ACCEPT SCOPED** | KV pinning only — does not clear **G-SKILL** / SKILL-X §3.0.4 |

---

## 3. Blocker and scope rows (honest disposition)

Operator routing preserved: **D-PREENT-02=defer**, **04=block**, **06=defer**, **07=defer**, **03=b**.

| ID | Wave 5 disposition | Owner | Evidence (read-only) | Close criterion |
| --- | --- | --- | --- | --- |
| **G-SKILL** | **EXACT_BLOCKER** | Operator | §3 **SKILL-X** = `CLOSED — INDEPENDENTLY ACCEPTED` with F-04 (disable/remove/upgrade unsupported); `T-IGX-032-A` only **ACCEPT_SCOPED** | **G-SKILL-A** or **G-SKILL-B** + signed `Approved` |
| **F10 / T-IGX-034-F** | **EXACT_BLOCKER** | Operator + EBH-7 | Pin `496016c0…`; 182 `TREE_DRIFT` / 0 `SEMANTIC` — [`WAVE1_F10_PINNED_REPLAY.md`](WAVE1_F10_PINNED_REPLAY.md); D-PREENT-04=block | Controlled TES advance or formal scoped N/A — no allowlist weakening |
| **OD-HIST04-A** | **EXACT_BLOCKER** | Operator | §3 **EVOL-X** — `[ ] FORMAL PROTOCOL 03 B6 — PENDING_AUTHORITY`; no PRE-ENT IIAC path edits | Operator ratification record |
| **OD-HIST04-B** | **EXACT_BLOCKER** | Governance | Amendment 01 vs six material IIAC paths; campaign does not edit IIAC | Governance decision; record `8045915a` |
| **OD-HIST04-C** | **EXACT_BLOCKER** | Operator | §3 **TOOL-LIFE-X** = `CLOSED — INDEPENDENTLY ACCEPTED` (DR-3: audit exact-SHA was **CONDITIONAL**); `T-IGX-030-C` **ACCEPT_SCOPED** only | Operator label alignment or scoped acceptance — **§3 not edited by this wave** |
| **Population E** | **N/A (PRE-ENT)** | ENT-AUDIT-X | §3 **ENT-AUDIT-X** — formal closure pending; **FRZ-AUD-01..10** OPEN (D-PREENT-03=b) | ENT owner rollup — out of PRE-ENT |
| **T-IGX-034-C** | **EXACT_BLOCKER** | Operator | D-PREENT-07=defer; **HIST-F-DUR-01** open; no ticket implementation | Implement / defer decision + new scope |
| **T-IGX-034-D** | **EXACT_BLOCKER** | Campaign | Qual-only; no dedicated ticket; module list not frozen; 030-C scoped cert does not cover 034-D | Frozen qual list + execution |
| **T-IGX-034-G** | **EXACT_BLOCKER** | Campaign | D-PREENT-03=b; [`T-IGX-034-G.md`](../../../../governance/tickets/T-IGX-034-G.md) checkpoint **Not started**; QUAL-X inventory has **Population A** closure only — **no frozen Pop B–D pytest module tuple** (see §4) | Serial pytest on frozen B–D module list |
| **T-IGX-034-H** | **EXACT_BLOCKER** | Campaign | Q-03 **TO_SCHEDULE**; §3 **DIST-SCALE-X** = `CLOSED — OPERATOR SCOPED ACCEPTANCE` @ `df485a96…` is not Q-03 on current HEAD; WL-7 deferral ≠ Q-03 closure | Independent Q-03 audit session |

**Silent OPEN:** none — every row is **EXACT_BLOCKER**, **N/A (PRE-ENT)**, or documented **ACCEPT_SCOPED** (predecessors table).

---

## 4. §3 roadmap citation notes (status unchanged)

This wave **does not edit** §3 stage rows in [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md). Citations inform blockers only:

| Stage (§3) | Cited label | Matrix use |
| --- | --- | --- |
| **SKILL-X** | `CLOSED — INDEPENDENTLY ACCEPTED` (conditional F-04) | **G-SKILL** remains **EXACT_BLOCKER** vs 032-A scoped cert |
| **TOOL-LIFE-X** | `CLOSED — INDEPENDENTLY ACCEPTED` | **OD-HIST04-C** — scoped 030-C does not reconcile §3 vs open disposition |
| **DIST-SCALE-X** | `CLOSED — OPERATOR SCOPED ACCEPTANCE` | **034-H** — not substitute for Q-03 fresh auditor |
| **EVOL-X** | `CLOSED — INDEPENDENTLY ACCEPTED (AUTOMATION_FIRST SCOPED)` | **OD-HIST04-A** — formal Protocol 03 B6 authority pending per §3 checkbox |
| **ENT-AUDIT-X** | Formal B6 **PENDING** | **Population E** — **N/A (PRE-ENT)** |
| **QUAL-X** | `CLOSED — POPULATION A ONLY` | **034-G** — B–D not closed in inventory |

---

## 5. FRZ checklist

[`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) was **not edited** in Wave 5. FRZ rows remain **OPEN** / **PASS** / **BLOCKED** / **N/A — WITH EVIDENCE** per checklist vocabulary only — campaign disposition labels are not imported into the checklist.

---

## 6. T-IGX-034-G qualification inventory check

Searched [`QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md`](../QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md): formal closure record exists for **Population A** only (`QUAL-X CLOSED — POPULATION A ONLY`). **No** committed frozen pytest module tuple for Populations **B–D**. Representative collect-only placeholder in `T-IGX-034-G.md` is not a frozen contract.

**Action:** pytest **not run** (no tuple). Row stays **EXACT_BLOCKER**.

---

## 7. Wave 6 cross-audit (complete — republication)

| Gate | Result |
| --- | --- |
| **W6-G0** | **PASS** — consistency (Wave 3/4/5 vs §3 citations) |
| **W6-G1** | **PASS** — SHA ladder @ `f4ca1c864` |
| **W6-G2** | **BOUNDED PASS only** — 41 passed blocking bundle; **full parent regression NOT RUN** |
| **W6-G3** | **`PRE_ENT_PREDECESSORS_EXACT_BLOCKERS`** |
| **W6-G4** | Operator sign-off pending — see [`PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md`](PRE_ENT_WAVE6_CROSS_AUDIT_REPORT.md) |

**AUDITOR_SESSION_ID:** `W6-INDEP-AUDIT-2026-11-10-f4ca1c864`.

---

## 8. Roadmap §5 ledger

One **non-closing** ledger row added (or refusal documented here if governance verifier rejects §5 edit): points to this file and **WAVE5_PUBLICATION_SHA**. **Not** stage closure; **not** independent exact-SHA audit.

---

## SSOT cross-links

- Closure path: [`PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md`](PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md)  
- Blocker matrix: [`PRE_ENT_WAVE4_BLOCKER_MATRIX.md`](PRE_ENT_WAVE4_BLOCKER_MATRIX.md)  
- Campaign progress: [`PRE_ENT_CAMPAIGN_PROGRESS.md`](PRE_ENT_CAMPAIGN_PROGRESS.md)  
- Formal checkpoint: [`PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md`](PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md)
