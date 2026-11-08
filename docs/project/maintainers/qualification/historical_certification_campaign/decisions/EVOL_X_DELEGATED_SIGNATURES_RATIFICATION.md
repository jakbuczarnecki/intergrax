# EVOL-X delegated signatures - ratification record (DR-1 / OD-HIST04-A, option 1, candidate C1)

Decision ID: OD-EVOLX-RATIFY-2026-11-08
Stage: EVOL-X
Policy version: IIAC-2026-11-05.1
Written by: the designated certifying session `cert-w0` (IIAC Amendment 01, R-1..R-6), under the delegation `docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md`. This is a delegate record. It is not written by the Operator and must not be cited as an Operator-written line. Assurance level: `process_enforced` (no cryptographic proof of the Operator's identity or of any session's identity).
Signed line: see `OPERATOR_DECISION_RECORDS_W0.md`, section DR-1 (the one disclosed-form line for this decision lives there, not here).

## 1. Transcription of the commission

The Operator's 2026-11-08 chat commission (English rendering by the MASTER orchestrator, as transcribed in the delegation record above; the Polish original is held in the chat transcript and in that record): the commissioned session certifies the seven open decisions, DR-1, DR-2, DR-3, DR-4, DR-6, DR-8 and DR-9, to which the Operator consents. For DR-1 the consented answer is: "Option 1: ratify the delegated EVOL-X plain-form signatures per candidate C1, effective from the original timestamps (Amendment 01 section 1.2, 'Earlier chains')."

The original EVOL-X commissions that this record ratifies are the ones transcribed in the three signing-delegation records already published in the repository (the Polish verbatim quotes are kept in those files and are not repeated here; English renderings below are by `cert-w0` and are labelled as renderings, not as the Operator's words):

| Delegation record (tracked) | Operator quote, English rendering | Added in commit |
|---|---|---|
| `evol_x/EVOL_X_P3_SIGNING_DELEGATION_2026-11-07.md` | "apply the new rule and start a separate session that will sign, in my name, everything that blocks certification" | `27d905027` (2026-11-07T15:42:21+01:00) |
| `evol_x/EVOL_X_OD_EVOL_12_SIGNING_DELEGATION_2026-11-07.md` | "start a signing session for OD-EVOL-12 and OD-EVOLX-08"; path choice "A: reviewer first, then signature" | `109029b2c` (2026-11-07T18:18:13+01:00) |
| `evol_x/EVOL_X_OD_EVOLX_08_SIGNING_DELEGATION_2026-11-07.md` | same commission and path choice; step 4 (a second, separate signing session for OD-EVOLX-08) | `5f9727032` (2026-11-07T18:21:33+01:00) |

All paths above are relative to `docs/project/maintainers/qualification/`.

## 2. What is ratified

Under Amendment 01 section 1.2 ("Earlier chains"), `cert-w0` records, on the Operator's consent to DR-1 option 1, that the following five plain-form lines are treated as written on the Operator's authority, effective from their original timestamps. The timestamps are not changed and no earlier record is rewritten.

| # | Decision ID | Plain-form line (verbatim, unchanged in its record) | Record | Signing commit |
|---|---|---|---|---|
| 1 | OD-EVOLX-P3-B6-2026-11-07 | `Approved OD-EVOLX-P3-B6-2026-11-07: operator @ 2026-11-07T15:41:06+01:00` | `iiac_evidence/EVOL-X/operator_decisions/EVOL_X_P3_B6_AUTHORIZATION_RECORD.md` | `27d905027` |
| 2 | OD-EVOLX-PRE-P3-2026-11-07 | `Approved OD-EVOLX-PRE-P3-2026-11-07: operator @ 2026-11-07T15:41:06+01:00` | `evol_x/EVOL_X_PRE_P3_OPERATOR_DECISIONS_AND_PUBLISHER_DELEGATION.md` | `27d905027` |
| 3 | OD-EVOLX-B6-ACCEPT-2026-11-07 | `Approved OD-EVOLX-B6-ACCEPT-2026-11-07: operator @ 2026-11-07T15:41:06+01:00` | `iiac_evidence/EVOL-X/b6_acceptance/EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md` | `1dd769331` (2026-11-07T16:09:47+01:00) |
| 4 | OD-EVOL-12 | `Approved OD-EVOL-12: operator @ 2026-11-07T18:17:34+01:00` | `iiac_evidence/EVOL-X/operator_decisions/EVOL_X_OD_EVOL_12_DECISION_RECORD.md` | `109029b2c` |
| 5 | OD-EVOLX-08 | `Approved OD-EVOLX-08: operator @ 2026-11-07T18:21:06+01:00` | `iiac_evidence/EVOL-X/operator_decisions/EVOL_X_OD_EVOLX_08_DECISION_RECORD.md` | `5f9727032` |

## 3. Evidence checked by cert-w0 (all read-only, on `origin/development` 991a3e410)

- **Content binding (recomputed by `cert-w0`).** For each record the SHA-256 of the committed text with the `Approved ...` line removed equals the digest named in the corresponding delegation record: lines 1-3 (P3 delegation): `a1c14c35...87b4`, `7bf1ad16...db7d`, `520bf25a...ff65`; line 4: `5c9da7e4...de2f`; line 5: `1befe97c...3354`. Five of five match. The delegation records bind the text that was signed; they do not prove that this text existed before the signing commit (the signing commits add the files whole).
- **Ancestry.** `1dd769331`, `109029b2c`, `5f9727032` and the formal closure `be77b3a5f` are ancestors of the base.
- **Order and expiry.** The signature times (15:41:06, 18:17:34, 18:21:06 on 2026-11-07) are before the `Expires` value of every delegation record (2026-11-07T23:59:59+01:00) and in the order OD-EVOL-12 < OD-EVOLX-08 < formal closure publication (18:27:25), as the W7 sequence requires.
- **Form.** Each delegation record names the plain form `Approved <id>: operator @ <time>` (Amendment 01 S-2: the plain form is allowed where a validator accepts only it, if the commission names it). The B6 preflight accepts only that form (`scripts/qualification/evol_x_closure_publication_preflight.py:368-369`).
- **FRZ-EVOL.** On the base all ten rows FRZ-EVOL-01..10 are `PASS` / `PASS - SCOPED` in the checklist (rows 273-282). This record does not touch them.
- **Campaign delegations exclude EVOL-X** (`DELEG-CERTCAMPAIGN-SIGN-2026-11-07`, `DELEG-2026-11-08`): verified only through the HIST-04 quotation in `dossiers/hist04/HIST04_EARLY_AUTHORITY_NOTE.md`; the originals are gitignored session files outside the read scope of `cert-w0`. The three EVOL-X delegation records above are separate instruments and were checked directly.

## 4. Limits of this ratification

1. It ratifies the **authority basis** of the five lines only. It makes no statement about the technical correctness of the EVOL-X closure and does not substitute an independent audit.
2. It rewrites no record, does not touch the EVOL-X records, the SHA ladder, the preflight, the handoff to ENT-AUDIT-X or the IIAC activation, and changes no FRZ status (FRZ-EVOL-01..10 stay as they are on the base).
3. It does not repair what the repository cannot show: that the Operator gave the quoted chat commissions, that the signing sessions were new and dedicated (R-2), and that the delegation records existed before the signatures. Those facts rest on the Operator's consent to this ratification and remain `process_enforced`.
4. The five lines are still plain-form lines. Anyone citing EVOL-X must keep the caveat "signatures written by operator-designated sessions, ratified by a delegate record on 2026-11-08, `process_enforced`", and must not cite them as lines written by the Operator.
5. **Re-binding of closure SHAs.** Not required by option 1: this record is a new file outside the EVOL-X evidence tree (`iiac_evidence/EVOL-X/`), the SHA ladder binds publication commits and not this file, and the digests above are computed on the unchanged pre-signature texts. This was determined by reading the preflight path handling (`evol_x_closure_publication_preflight.py` lines 153, 1415, 1678) and the ladder; the preflight was not run (no tests in this session). Option 2 (re-signing) would have needed a separate assessment; it was not chosen.
