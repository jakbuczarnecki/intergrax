# Operator decision records W1 - W1-R1, W1-R2, W1-R3, W1-PLAN

Written by the designated certifying session `cert-w1` on 2026-11-08, under IIAC Amendment 01 (R-1..R-6, S-1 and S-3 struck, S-2 kept) and the delegation record `docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W1-2026-11-08.md`.
Every `Approved` line below is a **delegate line**. It is not written by the Operator and must never be cited as an Operator-written line. Assurance level: `process_enforced`. Each line ends with `@cbb511ec7ded7b55f068444e326158fcf807960a`: the first commit of this branch, which carries the byte-identical copy of the delegation record. Each line carries its own time stamp. Nothing was amended. The pull request must be merged with a merge commit so that that commit stays reachable. Corrections after the independent review are in the final section (erratum).

## 0. Session statements (R-2) and method

- `cert-w1` is (1) a **new** session, (2) **dedicated** solely to certifying these four acceptances, (3) **not a continuation** of any other session, (4) **not an implementation session**. It is not MASTER, not `cert-w0`, not a reviewer or auditor of this stage, and not a HIST worker.
- Delegation verified before any work: SHA-256 of `DELEG-HIST-W1-2026-11-08.md` = `747dce743947fa2d88d00b85d66631b2d562fc5939d3c8e83e0a44e89a4c9a1c` (equals the value in the brief and in the `.sha256` file). The published copy is byte-identical: `sha256sum` of the working file and of the staged blob (`git cat-file blob :<path> | sha256sum`) give the same value.
- Baseline: worktree `cert-w1`, branch `hist-cert/cert-w1`, base `origin/development` = `70cceae21a4ee5e9bdbbc554565d96d8cbdd9115`. The first commit is `cbb511ec7ded7b55f068444e326158fcf807960a`.
- Method: read-only verification of repository files and git history; worker and MASTER statements treated as theses. **No test was run.** No governance script was run (`iiac_activation_gate.py` and `iiac_common.py` were read, not executed).
- Naming: W1-R1..W1-R3 are the three rulings of this delegation. They are not the IIAC Amendment 01 rules R-1..R-6.
- Consent is not pre-agreed as a result. `cert-w1` signs an item only where its own verification supports the accepted content, and says what it could not verify. All four items are SIGNED; each with the residuals listed under it.

| Item | Verdict | Effect |
|---|---|---|
| W1-R1 (SEC-05) | SIGNED | record only; row stays OPEN; L66 unchanged |
| W1-R2 (Decision-03 / D03-A) | SIGNED | record only; four rows stay OPEN; L66 unchanged |
| W1-R3 (FRZ-REG-02, 03, 06, 09) | SIGNED | record only; no promotion |
| W1-PLAN | SIGNED, SEQUENCE ONLY | no ticket plan approved; no FRZ promotion |

---

## W1-R1 - FRZ-SEC-05: scope of the sandbox/isolation evidence

**Accepted content.** Scope = contract + reference substrate (Linux netns + nftables). Per-host isolation is an explicit stated limitation, not proof. The row stays OPEN. Checklist L66 unchanged.

**Verification (read-only).**
- Row text. `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` line 222: criterion "sandbox/isolation boundaries qualified", acceptance owners "CTRL-X, PROD-Q", status `OPEN`. The note says AW-7C-P0-3B (`a597445...`) "reference Linux substrate contributes scoped platform capability evidence (kernel netns egress)" and AW-7C-P0-3B-PHYSQ (qualified baseline `d44a238dbe0d23dd9b864297175f54e663cfee3d`, evidence anchor `deaafee1a7612824bbe06dd7f0d403c3e490124b`) shows "real netns + nftables enforcement; allowed endpoint reachable; alive denied endpoint physically blocked; redirect escape blocked; effective policy reconstructed from kernel state". It ends: "Scoped capability evidence only; global criterion remains OPEN - PROD-Q still required. new global FRZ PASS = 0."
- The row text therefore already is a scope-limited statement. Recording "contract + reference substrate, limitation stated, row stays OPEN" is not wider than the row; it does not promote anything.
- Stage row. Checklist line 384 (AW-7C-P0-3B-PHYSQ): "PHYSQ CLOSED at child scope; ... all listed global FRZ criteria remain OPEN; new global FRZ PASS = 0". SEC-05 is among the listed criteria.
- Evidence file. `AW_7C_P0_3B_PHYSQ_EVIDENCE.json` (106 lines, read in full): `provider_identity` = `reference-substrate-qualification`; scenario `reference-physical-egress-causal-proof`; one allowed target (`allowed.test:18080`, reachable in the control and qualified phases) and one denied target (`denied.test:18081`, reachable in the control phase, `reachable: false` in the qualified phase); redirect to the denied target not escaped; `attestation_correlation.correlation_result` = `PASS`; three sessions destroyed; timestamp 2026-09-29. The JSON does not itself name netns or nftables; those words come from the checklist row. Both `deaafee1a` and `d44a238db` are ancestors of the base (`git merge-base --is-ancestor`, rc 0 for both). The file's history is two commits of 2026-09-29 ("record AW-7C physical egress proof").
- Scope in the campaign. `HIST03_ARCHITECTURE_VS_LIVE_SCOPE_MATRIX.md` line 26 classes SEC-05 as architecture-scoped (A): architecture obligation "Provider-neutral isolation/egress contract + reference enforcement"; live obligation "Real provider sandboxes (see SEC-06)". Line 70: "evidence is Linux reference substrate only; scope meaning at architecture level needs certifier/operator ruling". L66 (checklist line 66) does not list SEC-05, and W1-R1 does not add it.
- C10/C15 freeze. `agent-env/rules/intergrax-env-vendor-qualification.md`: "Contract stubs, fakes, or in-process doubles are not real vendor integration evidence - label scope accordingly in qualification records"; no Docker migration, no new vendor integrations, no production configuration changes; R7-Q, PROD-Q and `FRZ-*` statuses unchanged by tooling or rule edits alone. W1-R1 adds no vendor integration and no configuration; it is the labelling the rule asks for. The reference substrate is not a real provider, so the label "reference substrate, not provider qualification" is required, not optional.

**Reading of "per-host isolation" (disclosed).** The repository does not define the term. I read the stated limitation as: the evidence covers one reference substrate with one allow/deny endpoint pair and one policy; it does not prove isolation per real provider sandbox or per production host. This matches the "live obligation" column of the HIST-03 matrix and the row's "PROD-Q still required". If the Operator meant something narrower or wider, the Operator may correct this reading.

**Residual risks.**
1. Tension inside the row, not resolved by this record: HIST-03 classes SEC-05 as architecture-scoped (it must reach PASS or N/A-with-evidence for ARCH-FREEZE), while the row itself says "PROD-Q still required" for the global criterion. W1-R1 does not decide whether an architecture-scoped PASS can ever be granted on the contract + reference substrate scope alone. That is for the independent audit and a later ruling.
2. The evidence is from 2026-09-29 on a base that has moved; it is not replayed at current HEAD.
3. The checklist is not edited. If its text should carry the limitation more explicitly, that is a separately reviewed change.

**Could not verify.** That the netns + nftables enforcement actually ran as described (the JSON does not state the mechanism; I read the checklist's words). Nothing was executed.

**Verdict: SIGNED.**
```text
Approved W1-R1: operator-delegate cert-w1 @ 2026-11-08T15:51:46+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W1-2026-11-08.md@cbb511ec7ded7b55f068444e326158fcf807960a
```

---

## W1-R2 - Decision-03 / D03-A split (FRZ-SEC-01, FRZ-REL-03, FRZ-REL-05, FRZ-PRD-08)

**Accepted content.** Split into (a) what binds the contract and the trust/PIS semantics (testable in Dimension A) and (b) infrastructure/live (Kubernetes, Spanner + Azure witness; Dimension B). Rows stay OPEN; each qualification record must say which part it covers. No new entry in checklist L66.

**The ratified text (found).**
- `PROD_Q_R1_R7_Q_TRUSTED_HISTORY_ANCHOR_MECHANISM_DECISION_03.md` (68 lines, read in full). Status line: "RATIFIED ARCHITECTURAL TARGET / NOT IMPLEMENTED / DEPLOYMENT BLOCKED PENDING AUTHORITY & STORAGE DECISIONS"; operator selection 2026-10-18: Kubernetes, multiple replicas. Section 8: "RATIFIED FOR TARGET REQUIREMENTS. Deployment backend/anchor trust domain, global PIS/dispatch, and cross-store continuity protocol NOT RATIFIED, NOT IMPLEMENTED, NOT QUALIFIED. IMPL-01, R7-Q, PROD-Q and FRZ-* remain OPEN / BLOCKED."
- `PROD_Q_R1_R7_Q_D03_A_ELCR_IIRW_MECHANISM_CLASS_OPERATOR_RATIFICATION.md` (35 lines, read in full): OD-D03-A-01 ratifies the mechanism class ELCR-IIRW only; "no service selection or infrastructure authorization"; D03-B and D03-C stay mandatory independent gates.
- `PROD_Q_R1_R7_Q_D03_A_CONDITIONAL_INFRASTRUCTURE_DIRECTION_OPERATOR_DECISION.md` (lines 1-70 read): the Spanner Enterprise Plus `eur3` primary and Azure Blob RA-GZRS North Europe witness are "accepted only as a preferred service choice"; OD-D03-A-03..08 are not ratified; cloud execution NOT AUTHORIZED; an offline provider child (`...OFFLINE-PROVIDER-IMPL-01`) is "CLOSED - SCOPED / INDEPENDENTLY ACCEPTED (offline-only; not cloud-qualified)".
- Checklist line 218 (SEC-01) quotes all of these and ends each with "FRZ remains OPEN".

**What maps cleanly (as far as the text allows).**
- (a) contract and trust/PIS semantics: Decision-03 sections 1, 2, 5 (the four-part never-call-unless rule, as a fail-closed contract with offline or simulated evidence only - proof against a real continuity authority or real replicas is part (b); one semantic owner; the existing `GovernedProviderInvocationDispatchGate` as the only CREATE enforcer, `create_work` only; fail-closed deny on invalid/stale/revoked evidence; accept/cancel keep separate semantics; all strict HTTP/AC3/lifecycle paths share the guarded implementation); the vendor-neutral platform contracts and fail-closed requirement (conditional direction, "Accepted" items 3-4); the mechanism class ELCR-IIRW as a contract; the offline provider child as offline evidence.
- (b) infrastructure/live: the Kubernetes multi-replica deployment itself; the actual anchor backend and trust domain (Spanner + Azure witness: project/subscription identities, IAM independence, retention, credentials, cost, SLO); OD-D03-A-03..08; disaster-recovery and region-failover evidence; cloud execution. The text names these as NOT RATIFIED and cloud execution as NOT AUTHORIZED, which is consistent with placing them in Dimension B.
- Per row, as far as the text reaches:
  - FRZ-SEC-01: the only row whose own text names Decision-03 and D03-A (line 218: `Decision-03` once, `D03-A` five times, `Spanner` once, `Kubernetes` once). Tenant/caller chain evidence (P3-A, P3-A2) is a different, already-scoped contribution. The anchor-contract semantics are (a); the anchor backend and Kubernetes behaviour are (b).
  - FRZ-REL-03: Decision-03 section 1(b) and section 4 (one authoritative PIS concurrency domain; the stable CREATE identity comes from the prior Stable CREATE Identity Decision 01 cited in the Prior field of Decision-03). The PIS ownership and no-second-dispatch-under-known-intent semantics are (a); "all replicas share one first-writer domain" (gate D03-B) proven on real replicas is (b).
  - FRZ-REL-05: Decision-03 section 4 last paragraph (no automatic second dispatch under known intent is not provider exactly-once) and the unchanged accept/cancel semantics are (a); provider-native termination and exactly-once behaviour are (b).
  - FRZ-PRD-08: the guard-sharing / non-bypassable CREATE enforcement (section 5) is (a); the production-graph bypass proof for operator profiles is already (b) by checklist L66.

**What I could not map from the text.**
1. Only SEC-01's row text names Decision-03 / D03-A. The rows for REL-03, REL-05 and PRD-08 (checklist lines 230, 232, 258) contain none of `Decision-03`, `D03-A`, `Spanner` or `Kubernetes` (counted per row). Their link to Decision-03 comes from the HIST-03 matrix (rows at lines 22 and 34 name Decision-03; for REL-05 (line 36) and PRD-08 (line 50) only the narrative at line 71 does), which is a worker thesis, and from my reading of Decision-03 sections 1-5. The per-row lists above are therefore my reading, not text.
2. D03-B (globally coordinated PIS/dispatch) and D03-C (continuity write protocol) are called "independent mandatory gates". The text does not say whether offline or simulated evidence for them counts as Dimension A or must be Kubernetes evidence. Decision-03 section 7 requires thirteen behavioural proofs "before strict production admission" and says to "distinguish simulations from actual multi-process/Kubernetes tests"; it does not say which of them belong to the architecture freeze. Under W1-R2, any such evidence is part (a) only if it is labelled as offline or simulated and makes no Kubernetes or exactly-once claim.
3. Decision-03 itself contains no Dimension A / Dimension B vocabulary. The split is a new ruling of this delegation that is consistent with the text; it is not derived from the text.

**Residual risks.**
1. L66 is unchanged, as accepted. For SEC-01, REL-03 and REL-05, which are not on L66, the checklist rule still reads that `OPEN = 0` applies to architecture-scoped criteria. W1-R2 does not exempt the (b) parts of those rows from that rule; it only requires each qualification record to say which part it covers. Whether the (b) part of those three rows may stay OPEN at ARCH-FREEZE needs a later checklist ruling (an L66 change was explicitly not made here).
2. All four rows stay OPEN and, per `OPERATOR_DECISION_RECORDS_W0.md` section DR-9d, were `AUTHORITY_REQUIRED`; this record answers the scope question only. It authorizes no backend, no global PIS, no deployment, and no FRZ change.
3. Part (a) work still needs implementation and tests that do not yet exist (Decision-03 says "NOT IMPLEMENTED"); signing the split does not claim any of it is done.

**Could not verify.** Whether the offline provider child's evidence is current at HEAD (not replayed). Nothing was executed.

**Verdict: SIGNED.**
```text
Approved W1-R2: operator-delegate cert-w1 @ 2026-11-08T15:51:49+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W1-2026-11-08.md@cbb511ec7ded7b55f068444e326158fcf807960a
```

---

## W1-R3 - FRZ-REG-02, FRZ-REG-03, FRZ-REG-06, FRZ-REG-09: promotion scope

**Accepted content.** The strictest variant (a) of DR-4: promotion only after the ENT-AUDIT-X Population E rollup.

**Verification (read-only).**
- Rows. Checklist lines 264, 265, 268, 271: all four are `OPEN`, owner QUAL-X (REG-02 also EBH-5). Only the REG-06 row text names Population E: "does not promote this checklist row to PASS (global/platform rollup remains QUAL-X / Population E obligation)". The REG-02, REG-03 and REG-09 rows do not use the words Population E or ENT-AUDIT-X (checked by pattern search on the row text).
- Where Population E is defined. `QUAL_X_ENTERPRISE_QUALIFICATION_BASELINE_AND_GATE_INVENTORY.md` line 2361, section "Population boundaries": "E | Full freeze checklist rollup | REG-01 PLATFORM BLOCKED - ENT-AUDIT-X". Populations A (QUAL-X parent), B (historical stage waves), C (future stages), D (PROD-Q Dimension B) are on lines 2357-2360. The campaign documents use the label but do not define it.
- HIST-01. `dossiers/hist01/HIST01_ACCEPTANCE_HANDOFF.md` lines 18-22: option (a) "REG rows promote only on the ENT-AUDIT-X Population E rollup (slowest, strictest)"; HIST-01 recommends (b) for REG-07 and REG-10 and (a) for REG-01, REG-04 and REG-05, and says nothing about REG-02, REG-03, REG-06, REG-09. Their per-id matrix entries (`HIST01_FRZ_REG_PER_ID_MATRIX.json`) class all four `QUALIFICATION_GAP` / `QUALIFICATION_REQUIRED` with only Population A evidence at `d9ea0d6e8` and per-stage audits "not rolled up".
- WBS. `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md` line 60: `T-B7-PROMOTE-REG` covers REG-01..10, "Promote OPEN REG FRZ rows after FREEZE audit", depends on `T-B7-L3-FREEZE`. Line 38: `T-B7-L3-FREEZE` = "run_enterprise_audit.py --mode FREEZE", "FREEZE PASS". `.github/workflows/enterprise-audit.yml` is named "Enterprise audit (ENT-AUDIT-X)" and runs `scripts/enterprise_audit/run_enterprise_audit.py`. So the WBS promotion step is the same audit family that the QUAL-X inventory calls the Population E rollup. This identification (FREEZE-mode run = Population E rollup) is my reading of three documents, not a sentence in any one of them.
- Consistency with DR-4 as recorded. `OPERATOR_DECISION_RECORDS_W0.md` DR-4 (read in full): ruling (b) for REG-07 and REG-10, (a) for REG-01, REG-04, REG-05, REG-08 open, no REG promotion now, and "Residual risks / gaps (1): REG-02, REG-03, REG-06 and REG-09 are not covered by the HIST-01 split ... need their own ruling (or the Operator's explicit reading that they follow (a))". W1-R3 is exactly that ruling. It changes nothing in the DR-4 ruling and does not contradict it. DR-4's rule that a later promotion also needs an exact SHA, code, tests and an independent audit that did not author or certify the stage stays in force.
- Consistency with the 19 candidates. None of REG-02, 03, 06, 09 is among the 19 `VALID_SCOPED_PASS_CANDIDATE` rows (list taken from `HISTORICAL_FRZ_85_DISPOSITION_MATRIX.json`); the only REG row among them is REG-10, which DR-4 puts under (b), not under W1-R3. So W1-PLAN step 6 does not touch the rows of W1-R3.

**Residual risks.**
1. (a) is the slowest route. If the ENT-AUDIT-X Population E rollup is never run, or covers these rows differently, they stay OPEN. That is the cost the Operator accepted.
2. Three of the four row texts do not mention Population E; the link rests on the HIST-01 recommendation shape and on the REG-06 row. W1-R3 is a scope ruling, not a statement that the rows' text already says it.
3. `needs_head_replay` for REG-02, 03, 06, 09 (HIST-01 matrix) is unaffected: the DR-10 replay is still needed as evidence, but under (a) it does not by itself promote anything.

**Could not verify.** The scope of the ENT-AUDIT-X Population E rollup itself (another stream's material, out of read scope) and whether it will include these four rows.

**Verdict: SIGNED.**
```text
Approved W1-R3: operator-delegate cert-w1 @ 2026-11-08T15:51:50+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W1-2026-11-08.md@cbb511ec7ded7b55f068444e326158fcf807960a
```

---

## W1-PLAN - six-step plan, accepted as a SEQUENCE only

**Accepted content.** (1) test-only batch for HIST03-F-006 / F-016 with an audit proving no gate is weakened; (2) agreement with the ENT-AUDIT-X stream on whether ONE IIAC re-activation covers both streams; (3) one product batch (T-IGX-030-B, T-IGX-030-C, the P2-C corrective with the KV store authoritative and SKILL-X as owner, the Tier-0 -> Tier-3 move for FRZ-BND-03, validator alignment T-IGX-021-B/C), each with implementer then fresh auditor; (4) one IIAC re-activation after audit at the final SHA; (5) the one QUAL-X Dimension A replay on the final SHA (already approved by DR-10); (6) certification of the 19 candidate rows that only wait for audit and replay.

**This record certifies a sequence. It approves no ticket plan.** Every ticket named in the plan still needs its own plan and its own `Approved:` act. Nothing here promotes an FRZ row.

**Verification (read-only).**
1. *Tests-only change and ACT-N11.* `scripts/governance/iiac_common.py` line 22: `PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")`; `product_paths_changed_between` (lines 95-110) runs `git diff --name-only <audit_sha> <head>` and keeps only paths with those prefixes. `iiac_activation_gate.py` lines 567-576 report `ACT-N11 product path changed after the audited SHA` for each such path, and `ACT-N11 stale audit after material governance change` for each path in `MATERIAL_GOVERNANCE_PATHS` (lines 40-49; `_material_changes`, lines 329-344, compares exactly that set). The six members are `verify_iiac_policy_activation.py`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py` and `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`. A change confined to `tests/` (and `docs/`, `governance/tickets/`) matches neither rule. **Claim confirmed for this gate.** Not checked: whether another verifier (for example `verify_iiac_audit_manifest.py`) or the ACT-N13 test-matrix bytes bind specific test files.
2. *DR-12 = B and DR-10.* `HISTORICAL_AUTHORITY_AND_OPERATOR_DECISIONS.md` lines 124-127 (DR-12): B = tests-only now, product edits batched with the next IIAC re-activation; "Whether one re-activation covers both streams must be confirmed by the ENT-AUDIT-X stream - it is not assumed." Step 1 (tests only, now) and step 3 (product batch, held) match B; step 2 is the confirmation that DR-12 says must not be assumed. Lines 101-103 (DR-10): one bounded replay "on the final integrated SHA after the product batch ... with DR-12 = B this is after the IIAC re-activation batch". Steps 3, 4, 5 are in that order. **Consistent.**
3. *Ticket files.* In `governance/tickets/`: `T-IGX-030-B.md` and `T-IGX-030-C.md` exist, both with lifecycle state `plan_pending`, both marked HELD for ACT-N11 sequencing, both with no approval line. `T-IGX-030-A.md` exists and is approved (delegate line by `cert-030a`); it is the earlier tests-only batch and explicitly excludes `test_sec_ent.py` (F-006) and the W5 b2/c/d isolation batch (F-016). Not existing: any ticket for the F-006 / F-016 test-only batch, for the P2-C corrective (no ticket names `wire_skill_execution_pinning_store` or P2-C), for the FRZ-BND-03 move (no ticket names `external_work.py` or BND-03; one unrelated plan row mentions `intergrax/contracts/external_work.py`). **Also not existing: `T-IGX-021-B` and `T-IGX-021-C`**; `T-IGX-021-A.md` lines 47-48 name them only as follow-up tickets (021-B code incl. "material verifiers", 021-C independent audit). The brief listed the first four; the 021-B/C absence is an addition from my check. **Matches the statement, plus 021-B/C.**
4. *No promotion before an independent audit.* Steps 1, 3 each require an audit; step 4 is the IIAC re-activation after audit; step 5 is a replay; step 6 says "certification of the 19 candidate rows that only wait for audit and replay". The 19 rows are defined in `HISTORICAL_CERTIFICATION_CAMPAIGN_FINAL_REPORT.md` line 63 as "an independent audit and an exact-SHA replay are the only thing missing, not passed". Step 6 is placed after the replay and none of the earlier steps promotes a row. **No step claims a promotion before an independent audit.** But the wording of step 6 ("certification") is ambiguous; it is read here as the later act that needs its own exact-SHA independent audit (DR-4 rule), not as a promotion by this acceptance. The delegation's MASTER reading 2 says the same.
5. *Ordering point worth stating.* `T-IGX-021-B` is described as code touching "material verifiers"; any edit to the six material paths after the audited SHA triggers `ACT-N11 stale audit after material governance change`. Placing 021-B/C inside step 3, before the single re-activation of step 4, is therefore required, and the plan does that.

**Residual risks.**
1. Whether F-006 and F-016 can be fixed by tests alone is a thesis, not verified: HIST03-F-006 is "OPEN - run pending" with a static observation that product code binds `resolve_from_profile` by name; HIST03-F-016 is a test-isolation finding. If either needs a product edit, it leaves step 1 and moves into step 3.
2. Step 2 is outside this repository's control: the ENT-AUDIT-X stream may decide that one re-activation does not cover both streams. The sequence then needs a revision, not a new certification of this one.
3. The plan batches seven ticket-sized pieces in step 3 with no stated internal order; their relative order and dependencies (for example 030-C needs 030-A on its branch) are for their plans.
4. The certified sequence is not an approval, a schedule, or a budget. The DR-10 replay approval is the only execution approval it cites, and it is DR-10's, not this record's.

**Could not verify.** Anything about the ENT-AUDIT-X stream's plans; the IIAC verifiers other than the two gate files read; the feasibility of any listed ticket.

**Verdict: SIGNED (sequence acceptance only).**
```text
Approved W1-PLAN: operator-delegate cert-w1 @ 2026-11-08T15:51:53+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W1-2026-11-08.md@cbb511ec7ded7b55f068444e326158fcf807960a
```

---

## Final checks and what this session did not do

- Files written by `cert-w1`: the delegation copy (first commit), this file (second commit), and the report and flag under the campaign `.tmp` folder (not tracked). Nothing under `intergrax/ agents/ applications/ scripts/ tests/ governance/`. No checklist, roadmap, WBS or ticket edit.
- No plan approval, FRZ promotion, IIAC activation, PR, push, merge, rebase, reset or amend. No test was run.
- Open items for MASTER are listed in `CERT_W1_REPORT.md`.

## Erratum after the independent review rev-w1 (MASTER, 2026-11-08)

Applied in a follow-up commit; no `Approved` line was changed or re-stamped, and none of the lines carries a hash of this body.
- F-1: the list of rows with `needs_head_replay` true now includes FRZ-REG-03 (residual risk 3).
- F-2: the stable CREATE identity is cited as a prior decision of Decision-03, not as a requirement of its section 4 (REL-03 mapping).
- F-3: the never-call-unless rule under (a) is limited to a fail-closed contract with offline or simulated evidence; proof against a real continuity authority or real replicas belongs to (b).
- F-4: in (a), "trust" means only the contract-level fail-closed predicate and the ELCR-IIRW class as a contract. Decision-03 Gate D03-A (anchor backing and trust domain, section 6) is itself part (b).
- F-5: evidence citation for REL-05 and PRD-08 made precise (only the narrative at HIST-03 matrix line 71 names Decision-03 for them).
- F-6: the merge method is stated in the header.
- F-7: the four time stamps are signing times, not verification times.
