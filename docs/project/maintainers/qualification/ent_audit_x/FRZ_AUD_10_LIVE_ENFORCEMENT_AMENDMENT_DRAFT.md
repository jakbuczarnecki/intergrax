<!-- Draft added by governance/tickets/T-IGX-044-A.md (EQO-A remediation node PT-E00, documentation only). Approval state: see the Approved section of that ticket; approval of the ticket does not make this text a LIVE disposition. -->

# FRZ-AUD-10 - live enforcement amendment (DRAFT)

Status: **DRAFT - proposal text, not a LIVE disposition, no effect.** This page is a proposal text. It is **not** an Operator decision record (`OD-ENT-02-R1` does not exist), it does **not** change the FRZ-AUD-10 disposition, it edits neither the freeze checklist nor the roadmap, and it claims no PASS, closure or independent verification. FRZ-AUD-10 stays as the checklist row reads today (section 2). Facts were checked at `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2` (`origin/development` when this draft was written).

## 1. Purpose and origin

Operator decision OD-REM-07 (EQO-A remediation planning, 2026-11-16) sets the target for FRZ-AUD-10 to **live enforcement** of the ENT-AUDIT-X floor, delivered by a separate, controlled plan; condition C9 requires that plan to carry preflight, negative trials, a false-positive assessment and a rollback procedure, and that no branch setting or validator code is changed within the audit campaign itself. Decision DP-64 fixes the first stage: a new ruleset labelled `LIVE (ENT floor)`, no bypass actors, and a `pull_request` rule only after trial N3; each ruleset change needs its own authorisation. These decisions are session records of the planning conversation; at the head named above they are not yet a tracked repository file, so this draft cites them by identifier only.

This draft prepares the **text** that a later amendment would need. Applying any part of it is a separate act (section 5).

## 2. Current wiring (verified at the head named above)

| Item | Fact | Source |
|---|---|---|
| Checklist row | FRZ-AUD-10 reads `PASS - COMPENSATING (SCOPED)` with the evidence note "OD-ENT-02 compensating + `SUFFICIENT_COMPENSATING_SCOPED`; **OPEN-BP-01** remains". The criterion text offers two routes: LIVE branch protection or COMPENSATING (SCOPED) per the IIAC package, "not equivalence without evidence" | `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md:325` |
| Status mode | `od_ent_02_mode` is `compensating` | `ent_audit_x/ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json:51` |
| Preflight coupling | The formal-closure preflight accepts the modes `live` and `compensating` only, and requires the FRZ-AUD-10 row text to contain `COMPENSATING` in compensating mode and `LIVE` in live mode | `scripts/qualification/ent_audit_x_formal_closure_preflight.py:42`, `:261-264` |
| Mode branch | In `compensating` mode the preflight runs the compensating verifier with the branch rules read from GitHub; in `live` mode it runs `verify_ent_audit_branch_protection.verify_branch_protection()` | `scripts/qualification/ent_audit_x_formal_closure_preflight.py:280-301` |
| Compensating verifier | Requires `mode == "compensating"` in the published sub-manifest `od_ent_02_compensating.v1.json` and states that no mode treats compensating controls as equivalent to live branch protection | `scripts/governance/verify_od_ent_02_compensating_control.py:10`, `:121` |
| Live verifier | `verify_ent_audit_branch_protection.py` requires the two job names `ent-audit-fast` and `ent-audit-pr-gate` on `development` | `scripts/governance/verify_ent_audit_branch_protection.py:25-28` |
| Open backlog item | OPEN-BP-01 (live branch protection on `development` not enabled; owner repo admin / ARCH-FREEZE) is recorded in the handoff, the compensating package and the threat model | `ent_audit_x/ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md:48,52`; `ent_audit_x/OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md:5`; `ent_audit_x/OD_ENT_02_THREAT_MODEL.md:5,32` |
| Live GitHub state | A read-only `GET repos/<owner>/intergrax/rules/branches/development` on 2026-11-16 returned two rules only, `deletion` and `non_fast_forward`, both from ruleset 26771061; no required-status-check rule | read-only API call during this change; not a tracked artefact |

## 3. What a LIVE disposition would need (proposal, nothing here is performed)

| Step | Content | Authority and owner |
|---|---|---|
| L1 | A new ruleset on `development` (never an edit of ruleset 26771061) requiring `ent-audit-fast` and `ent-audit-pr-gate`; stage plan and thresholds per the separate controlled activation plan (C9) | Operator authorisation per stage (DP-64); executor named by the Operator |
| L2 | Same-repository trial on a disposable ruleset restricted to `refs/heads/enforcement-trial/**`, with negative trials and a false-positive assessment before any ruleset on `development` (DP-67) | Operator authorisation; independent session reads the results |
| L3 | A new Operator decision record `OD-ENT-02-R1` that sits beside, and does not edit, the signed delegate record | Operator |
| L4 | Status JSON `od_ent_02_mode: live` (the sealed status of that file must be checked before editing) | docs implementer, after L1-L3 |
| L5 | Checklist row text containing `LIVE` (required by the preflight coupling in section 2) | Operator-only file; changed in the later FRZ/roadmap batch, not by this ticket |
| L6 | Close OPEN-BP-01 in the handoff, the threat model and the debt register; keep the compensating package as history | docs implementer, after the independent audit |
| L7 | Rollback exactly as designed in the controlled plan: disable or delete the new ruleset, then confirm that only `deletion` and `non_fast_forward` apply again | Operator pre-authorises triggers; anyone with admin may execute |

## 4. Gate for the claim

Until an independent audit at one exact SHA has examined the ruleset JSON, the live verifier output, the trial evidence and the record edits, the FRZ-AUD-10 row stays `PASS - COMPENSATING (SCOPED)`; if the ruleset is enabled earlier, the only permitted wording is a note such as "LIVE enabled, audit pending". The implementer of any step does not certify it.

## 5. Statements the future amendment must carry

1. The OD-ENT-02 sufficiency `SUFFICIENT_COMPENSATING_SCOPED` is not rescinded by this draft. Whether the compensating control was holding in practice is a question for evidence: the planning session analysed recent merged pull requests against check timing and result, but that analysis is a session artefact and is not restated here. Before adoption it has to be published as evidence at an exact SHA and cited.
2. LIVE as targeted by DP-64 covers the ENT floor (`ent-audit-fast`, `ent-audit-pr-gate`) only, as the live verifier requires (section 2). Any further required check, and the fact that a ruleset can be disabled by whoever holds admin rights on the repository, must be stated in the amendment as limits of the claim.
3. OPEN-BP-01 stays open until the LIVE disposition is independently verified.
4. The related findings A2-037 and A3-007 (FRZ-AUD-10 part) keep their EQO-A condition status until then; this draft closes neither.

## 6. What this draft does not do

- It does not change the FRZ-AUD-10 disposition, the checklist, the roadmap, the status JSON, any script, workflow or GitHub setting.
- It does not authorise any of L1-L7; each needs its own Operator authorisation.
- It does not record a decision of the Operator beyond identifying OD-REM-07, C9, DP-64 and DP-67 as the origin.

## 7. Open points for the Operator

- Where the OD-REM-07 / DP-64 / DP-67 decisions are published as a tracked record, so that this page can link to it instead of citing identifiers.
- Which session writes `OD-ENT-02-R1` and when (after the independent audit, per section 4).
