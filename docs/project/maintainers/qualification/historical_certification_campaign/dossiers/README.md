# Dossiers - how to read the `.diff` files

The `dossiers/hist01..hist04/` directories hold the substantive artifacts of the four campaign workers (per-ID matrices, findings registers, reports, handoffs). The `.diff` files in them are **working candidates produced by the workers. They are NOT the published changes.**

The published changes are three merged pull requests:
- PR 405 (head `6d1434796`): documentation-only evidence and citation corrections.
- PR 406 (head `937c39820`): revision of IIAC Amendment 01 and related governance files.
- PR 407 (head `90fae38aa`): T-IGX-030-A, tests only.

## Do not apply these files
- `hist04/PATCH_H4_01_evolx_ssot_reconciliation.diff` and `hist04/PATCH_H4_01_REBASED_64ce093.diff`: both replace the EVOL-X roadmap stage cell `[ ] FORMAL PROTOCOL 03 B6 - PENDING_AUTHORITY` with `[x] CLOSED - INDEPENDENTLY ACCEPTED`. The independent review found this stage-label change; PR 405 reverted it.
- `hist04/PATCH_RULES_UPDATE_AMENDMENT01_AND_TICKETS_15347255.diff` and `hist04/PATCH_RULES_UPDATE_v2_15347255.diff`: superseded by PR 406 (which also contains the review corrections).
- `hist01/PATCH_H01_NONGATED_492e361.diff`, `hist02/HIST02_SSOT_CANDIDATE_PATCH_BND06_PLG06.diff`, `hist03/PATCH_H03_P01_P02_P07_492e361.diff`: their content was published through PR 405, with the review corrections applied in commit `6d1434796`. The scope notes next to them (`PATCH_H01_SCOPE_NOTE.md`, `PATCH_H03_SCOPE_NOTE.md`) say what was included and excluded and why.

## What is deliberately not included
Raw logs, status files, work logs and the local snapshot of earlier-campaign audit material (`hist01/inputs`, which is unpublished local evidence) were left out. No claim in the package depends exclusively on them; see section 4 of `HISTORICAL_STAGE_SHA_AND_EVIDENCE_REGISTRY.md`.
