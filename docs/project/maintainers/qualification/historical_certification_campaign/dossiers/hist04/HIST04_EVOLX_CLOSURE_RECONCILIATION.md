# HIST04 - EVOL-X closure reconciliation (read-only; baseline ba07ebaa3)

Scope: mission items 1-4 and the EVOL-X part of 8/18. No file of the ENT-AUDIT-X dossier or the published handoff was touched. Nothing was executed.

## 1. Latest formal EVOL-X closure SHA (item 1) - VERIFIED
| Ladder role | SHA | git (cat-file / merge-base --is-ancestor HEAD) |
|---|---|---|
| B6_ACCEPTANCE_PUBLICATION_SHA (phase 1) | 1dd769331506149a1ee2f411c2bc62e85e5bbf25 | commit, ancestor; committed 2026-11-07T16:09:47+01:00; phase 2 b288ba756 (16:10:34) |
| OD-EVOL-12 + FRZ-EVOL promotion | 109029b2ca29fe0ba1b95a0f66ae4c243fdf9e5b | commit, ancestor; 18:18:13 |
| OD-EVOLX-08 | 5f97270320... | commit, ancestor; 18:21:33 |
| certifier blocker B-5 resolved | 50ddc04fbabd24ac062267d5d33f14ccef89840c | commit, ancestor; 18:21:34 |
| **FORMAL_CLOSURE_PUBLICATION_SHA (phase 1)** | **be77b3a5fb97f10a3bd2df6ee038cb92aad77eca** | commit, ancestor; 18:27:25; phase 2 = 9f4042110 (ancestor) |
| W7_06_HANDOFF_PUBLICATION_SHA (phase 1) | 7a4f279fd8c0986431941c77283d872ea5b0c1ba | commit, ancestor; 2026-11-08T06:54:57+01:00; phase 2 = 360a4ed91 |
Latest formal EVOL-X closure SHA = be77b3a5f (ladder row FORMAL_CLOSURE_PUBLICATION_SHA; handoff record sections 1/11 cite the same value). The ladder deliberately holds the phase-1 SHA (non-self-referential); phase-2 commits bind it in `EVOL_X_CLOSURE_EXECUTION_STATUS.json`. I did not open that JSON (not needed).

## 2. EVOL-X B6 and OD-EVOL-12 / OD-EVOLX-08 binding (item 2)
- B6: IIAC record `iiac_evidence/EVOL-X/b6_acceptance/EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md`, PASS_SCOPED, 12 limitations bound by digest (ladder). Signature line `Approved OD-EVOLX-B6-ACCEPT-2026-11-07: operator @ 2026-11-07T15:41:06+01:00` - plain form (see authority note).
- OD-EVOL-12 (`...EVOL_X_OD_EVOL_12_DECISION_RECORD.md`): signature 18:17:34. Delegation file `evol_x/EVOL_X_OD_EVOL_12_SIGNING_DELEGATION_2026-11-07.md` records sha256 `5c9da7e4...de2f` of the text before signing. **Recomputed by me** (record at 109029b2c minus its last line): `5c9da7e431b1e6116a8be0fddd5ab4e1c09530113ac85d001bbee37ecc18de2f` - MATCH.
- OD-EVOLX-08 (`...EVOL_X_OD_EVOLX_08_DECISION_RECORD.md`): signature 18:21:06. Delegation records `1befe97c...3354`. **Recomputed** (record at 5f9727032 minus last line): `1befe97c946609cb041bafe5c3df86f6855bc342b48e36321ac28ad932ce3354` - MATCH. The signing commit added the whole file in one commit (no earlier blob in git), so the digest proves only that the signature line is the sole delta against the text the delegation named, not that this text existed before.
- Ordering: OD-EVOL-12 (18:17:34) < OD-EVOLX-08 (18:21:06) < closure publication (18:27:25): consistent with the W7-04 precondition "OD-EVOL-12 signed, FRZ rows promoted".
- Binding verdict: content binding VERIFIED; **authority binding NOT verifiable from the repository** (AUTHORITY_GAP, F-H4-01).
- FRZ: FRZ-EVOL-01..10 are PASS / PASS-SCOPED in the checklist (rows 273-282), LIM-01..07 carried. FRZ-EVOL-07 qualification-layer only (LIM-05). HIST-04 does not re-open them.

## 3. Stale EVOL-X PENDING text (item 3)
| Where | Stale text | Truth at baseline | Patch |
|---|---|---|---|
| roadmap L489 body | "Protocol 03 B6 PENDING_AUTHORITY; FRZ-EVOL-01..10 OPEN; FORMAL_CLOSURE_PUBLICATION_SHA unset; not CLOSED - INDEPENDENTLY ACCEPTED" | closed/published, FRZ promoted | PATCH_H4_01 |
| roadmap L489 status cell | "[ ] FORMAL PROTOCOL 03 B6 - PENDING_AUTHORITY" | CLOSED - INDEPENDENTLY ACCEPTED (AUTOMATION_FIRST SCOPED) - PUBLISHED (L561, L563, L1565 already say so) | PATCH_H4_01 |
| roadmap L2065 ledger | "W7-06 formal handoff pin and D10 attestation pending" | published 7a4f279fd | PATCH_H4_01 |
| EVOL_X_SHA_LADDER L81 | "closure SHAs remain pending" (OD-EVOL-18 note) | published | PATCH_H4_01 |
`git apply --check` of PATCH_H4_01 against the worktree: OK. L490 (ENT-AUDIT-X row) and L2066 mention "Protocol 03 B6 PENDING" for ENT-AUDIT-X: those are ENT-AUDIT-X facts (other session) - not touched. IIAC protocol header "NOT_IN_FORCE" stays (material path, F-H4-07).

## 4. EVOL-X -> ENT-AUDIT-X handoff (item 4) - read only
`evol_x/EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md` status "FORMAL HANDOFF PUBLISHED (W7-06, 2026-11-08)"; section 1 cites formal closure be77b3a5f and B6 1dd769331; section 11 formal checklist all [x]; D10 digest-chain attestation = `EVOL_X_D10_DIGEST_CHAIN_ATTESTATION.md` (process_enforced). Consistent with the ladder. Exclusions kept (section 8: PROD-Q live, M-09H NOT_MEASURED, FRZ-EVOL-07 qualification-only, ENT-AUDIT-X/ARCH-FREEZE not closed). Never altered. The handoff text itself says the signatures are delegated under Amendment 01, `process_enforced`, "operator-adopted, not audited, not mechanically enforced" - so any downstream consumer already has the caveat; HIST-04's added caveat is the authority-scope conflict (F-H4-01), which the handoff does not mention.

## 5. Conclusion
EVOL-X closure is internally consistent in SHAs, digests and ordering; remaining obligations: (a) SSOT drift patch (PATCH_H4_01), (b) operator personal ratification of the delegated plain-form signatures. Until (b), cite EVOL-X as closed with "delegated, process_enforced signatures; operator ratification pending".
