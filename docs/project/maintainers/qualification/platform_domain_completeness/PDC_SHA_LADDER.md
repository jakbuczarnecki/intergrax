# PDC-X — SHA ladder

| Role | SHA | Notes |
| --- | --- | --- |
| BASELINE_SHA | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` | `origin/development` at PDC-P0 start (PR #459 merge) |
| ROADMAP_AMENDMENT_SHA | `bc9ef52bd690738e71638a599eec64c76ac356c9` | PR #461 merge on `development` (T-IGX-036-A); descends from BASELINE_SHA; contains P0 head `142190b313d304bdc6c6362a1d7484b88a802920` |
| P1_PUBLICATION_SHA | `dd2055e19c64ef35fd80dde770f3de924ddb39d1` | Closed-world inventory (PR #463 merge, T-IGX-036-B) |
| P2_PUBLICATION_SHA | `dd2055e19c64ef35fd80dde770f3de924ddb39d1` | Requirements matrix, gap register, FRZ crosswalk (PR #463 merge, T-IGX-036-C) |
| P3_PUBLICATION_SHA | `dd2055e19c64ef35fd80dde770f3de924ddb39d1` | Ownership matrix, ADR and OD drafts (PR #463 merge, T-IGX-036-D) |
| P3_LOCK_PUBLICATION_SHA | `dcdb35601305a2ad9a3ee22894b26c29800a9640` | Architecture lock record (T-IGX-036-E, PR #465 merge) |
| P4_PUBLICATION_SHA | `f573e24e851a82a7ed4bab72396eecd7d3e45a1b` | Implementation WBS, dependency DAG, cross-domain test matrix (PR #466 merge, T-IGX-036-G) |
| P5_WAVE_A_PUBLICATION_SHA | `2bd529785cef3eeece55afa588946b410022294b` | B-DOC documentation reconciliations (PR #467 merge, T-IGX-036-H) |
| P5_B_REG_T_PUBLICATION_SHA | `5836a5a5bafab1cf819c34666afaf1789825dd52` | B-REG-T repair of 12 stale gates (PR #471 merge, T-IGX-036-I); ticket closed in PR #472 (`af7c72ac5f82f568381fbe0ad19f35a80673755e`) |
| P5_PLAN_PUBLICATION_SHAS | see Notes | Signed plans are docs-only merges: wave 1 J/K/L `895809e7070fd27373623cb28d04b2b6ba4d05cb` (PR #473); wave 2 N/O `36b466da26f7d6253ecc0025ef21e18747fc49bb` (PR #484), B-D2 amendment `9d2f8bfc212e75305fb4ae0d1b779ce1c81566a8` (PR #491); wave 3a P `c4ee8b198b6c5a51f94b7d5e9e6286721167242a` (PR #492), B-D6 amendment `1ce6eb6283f36e19fb03370a286ce2fae219a1ca` (PR #495); wave 3b Q `95dd41b150c492d9b3ea1dc721391d44a52909e9` (PR #497), B-D8 amendment `364b3892544535e0969c28cfdcc0d308d565af3f` (PR #502); wave 3c R `2b27322cb6134a6bdd69d3aafbe835173621bb8e` (PR #499); standing authorizations `6e533709a599e0bc453fef5c012a11bdbc060843` (PR #503); B-REG-P M `48200c48e064c0b3e13e5999763961d06addcb22` (PR #504) |
| D1_IMPLEMENTATION_SHA | `bc7739c0700502afdf998301a2b87c393b3ce821` | Branch `pdc-x-d1` (T-IGX-036-J). Draft PR #478 closed by the Operator; branch intact. Pre-integration head, not merged |
| D2_IMPLEMENTATION_SHA | `bf9bad971900449ce77d6f88aed06f6f4f21a7af` | Branch `pdc-x-d2` (T-IGX-036-N), draft PR #489. Pre-integration head, not merged |
| D3_IMPLEMENTATION_SHA | `ff469148d6b7f7976be88c74ee85f03bc0cba6da` | Branch `pdc-x-d3` (T-IGX-036-K). Draft PR #476 closed by the Operator; branch intact. Pre-integration head, not merged |
| D4_IMPLEMENTATION_SHA | — | Branch `pdc-x-d4` (T-IGX-036-R); implementation in progress |
| D5_IMPLEMENTATION_SHA | `85a8c441d8e3f37e3f2d09451686dcd2f556d490` | Branch `pdc-x-d5` (T-IGX-036-O), draft PR #488. Pre-integration head, not merged |
| D6_IMPLEMENTATION_SHA | `453cc10ced15fc5b2d2d308aa172a83602570b85` | Branch `pdc-x-d6` (T-IGX-036-P), draft PR #494. Pre-integration head, not merged |
| D7_IMPLEMENTATION_SHA | `52f35a41fcd32e552edb68401aefb2ec3f565daf` | Branch `pdc-x-d7` (T-IGX-036-L). Draft PR #475 closed by the Operator; branch intact. Pre-integration head, not merged |
| D8_IMPLEMENTATION_SHA | `9c7d59f88036b86407dc61373ce64513679d8934` | Branch `pdc-x-d8` (T-IGX-036-Q), draft PR #501. Pre-integration head, not merged |
| B_REG_P_IMPLEMENTATION_SHA | `c3d5b7005cee1b93c9e2a98211f977d77da752a6` | Branch `pdc-x-regp` (T-IGX-036-M), draft PR #505. Pre-integration head, not merged |
| Dn_PUBLICATION_SHA | — | One per P5 lane (D1..D8) and B-REG-P; publication = merge commit on `development`, set after MCP-SERVER-X P8 |
| INTEGRATED_BASELINE_SHA | — | P6; must descend from every Dn publication |
| TEST_EXECUTION_SHA | — | Must equal the audited SHA of each audit |
| Dn_AUDITED_SHA | — | P7 per domain; INTEGRATED_BASELINE_SHA or a corrected descendant |
| AUDIT_EXECUTION_SHA | — | IIAC `audit_execution_sha`; no product delta to HEAD at activation (ACT-N11) |
| EVIDENCE_PUBLICATION_SHA / CERTIFICATION_PUBLICATION_SHA | — | Evidence-only descendants (docs paths only) |
| GLOBAL_AUDITED_SHA | — | P8; must contain the MCP-SERVER-X closure publication |
| FINAL_CLOSURE_PUBLICATION_SHA | — | P9 merge |

Historical accepted SHAs of predecessor stages are reuse evidence only. Each reuse needs change-delta evidence from the accepted SHA to the audited SHA (PDC-P2-09). Being an ancestor does not make a commit a current-HEAD PASS.
