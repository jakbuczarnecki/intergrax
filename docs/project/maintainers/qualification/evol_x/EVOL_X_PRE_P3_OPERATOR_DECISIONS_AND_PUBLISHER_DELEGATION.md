# EVOL-X pre-P3 operator decisions and publisher delegation (UNSIGNED - the operator adds the signature line)

Source of the choices: the operator's own answers in the Claude Code session of 2026-11-07, given through the question dialog. The agent recorded them; the agent did not sign. Assurance level: process_enforced.

Decision ID: OD-EVOLX-PRE-P3-2026-11-07
Stage: EVOL-X
Policy version: IIAC-2026-11-05.1
Decision scope: Pre-P3 review of OD-1 and OD-2, interpretation of OD-3 condition 3, and narrow publisher delegation for EVOL-X B6 acceptance publication
Decision status: AUTHORIZED

## 1. OD-1 re-opened: option C
Residual W401-RA-08 is accepted again without a material change. P2 pin stays in scripts/qualification/evol_x_closure_publication_preflight.py. Any change to that file, its tests, the P2 constant or .gitattributes needs its own ticket and independent audit.

## 2. OD-2 re-opened, Amendment 2 confirmed as the operator's own decision
python -m scripts.qualification... is NOT a supported command form (finding CERT-01, reproduced on 2026-11-07 at 2f6664d53). The only supported form is: python scripts/qualification/evol_x_closure_publication_preflight.py --gate <gate> on a fresh clone of the pinned SHA in a clean environment outside the clone (OD-2 Amendment 1). A result obtained with -m is not evidence.

## 3. OD-3 condition 3 interpretation
Condition 3 refers to the six MATERIAL_GOVERNANCE_PATHS of iiac_activation_gate.py, the preflight script, its P2 constant, and the preflight tests. Changes to scripts/governance/igx_gov/* (G0 category 2) and the new scripts/governance/sync_agent_env.py do not void OD-3.

## 4. Publisher delegation (narrow)
The agent session that prepares this record may create and push the publication commits of the P3/B6 chain (operator decision record, certification record resolution of B6-P3, B6 acceptance record phase 1, phase 2 SHA binding) on branch development, ONLY after the operator has signed the P3 record and the B6 record with their own signature lines. Excluded: FRZ promotion, OD-EVOL-12, OD-EVOLX-08, formal closure, any change to material governance paths, activation record, protocol or scripts.

## Limits
This record is not the P3 decision, not a B6 acceptance, and does not change any gate result.

## Addendum 2026-11-07 (Amendment 01, R-6)
The signature lines of this record, of the P3 record and of the B6 record are written in the operator's name by a designated signing session commissioned by the operator in chat on 2026-11-07, see the delegation record EVOL_X_P3_SIGNING_DELEGATION_2026-11-07.md.
Approved OD-EVOLX-PRE-P3-2026-11-07: operator @ 2026-11-07T15:41:06+01:00
