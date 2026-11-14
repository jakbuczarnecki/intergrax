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
| D4_IMPLEMENTATION_SHA | `becdd16cf4eeb9ec1864a36d1be01f42b5ed8039` | Branch `pdc-x-d4` (T-IGX-036-R), PR #508. Pre-integration head |
| D5_IMPLEMENTATION_SHA | `85a8c441d8e3f37e3f2d09451686dcd2f556d490` | Branch `pdc-x-d5` (T-IGX-036-O), draft PR #488. Pre-integration head, not merged |
| D6_IMPLEMENTATION_SHA | `453cc10ced15fc5b2d2d308aa172a83602570b85` | Branch `pdc-x-d6` (T-IGX-036-P), draft PR #494. Pre-integration head, not merged |
| D7_IMPLEMENTATION_SHA | `52f35a41fcd32e552edb68401aefb2ec3f565daf` | Branch `pdc-x-d7` (T-IGX-036-L). Draft PR #475 closed by the Operator; branch intact. Pre-integration head, not merged |
| D8_IMPLEMENTATION_SHA | `9c7d59f88036b86407dc61373ce64513679d8934` | Branch `pdc-x-d8` (T-IGX-036-Q), draft PR #501. Pre-integration head, not merged |
| B_REG_P_IMPLEMENTATION_SHA | `c3d5b7005cee1b93c9e2a98211f977d77da752a6` | Branch `pdc-x-regp` (T-IGX-036-M), draft PR #505. Pre-integration head, not merged |
| Dn_PUBLICATION_SHA | see Notes | One per P5 lane; publication = merge commit on `development`. D1 `aaa93faecf7482eb23407318c05a148de609ce30` (PR #478); D2 `7bdd304493c405ddab82855d9a446ad63fecd81e` (PR #489); D3 `b5dd083938f266c2d223f9a3924173a8c0d01fe7` (PR #476); D4 `214092e87e8a55c18468ab52dd09a8290b092c4a` (PR #508); D5 `0d86fd4797cecdec6bbbf168534a3266810879c9` (PR #488); D6 `1f6dc3c4f93f76c32ad2838ac3b2e0df962a5853` (PR #494); D7 `2665dafc41dc3f647935964dd11365f6cf1f53e8` (PR #475); D8 `ffcf055a4267b3d8cdea2a58e20c2e5ce6b291da` (PR #501); B-REG-P `e8368a4344f7a1f1347e62d1b5cf9e88a797085a` (PR #505); RAG fix `b26668ca6d49d81e56d334839e8f3e3882a092e5` (PR #510) |
| P6_REPAIR_PUBLICATION_SHAS | see Notes | Integration repairs from the B-P6 report: S (XD scenarios, ECAP) `fbad6a8dea2087d0a3122101f405d40f875fdc03` (PR #514); T `fa6d4dedcbc386aa1de17ee37f76ca9f540c044f` (PR #517), `87f4d4f417050fec40c32e4051bf76fb1124d58b` (PR #520), `9d8e1ee0e54985f0028fd330802e00e548021113` (PR #523); U `3dd991336233cea1662518df8b35e5e1ea901e00` (PR #524); V `a156cbbd5ff521dc7e99dd02ab6bebd4ca234a73` (PR #525) |
| INTEGRATED_BASELINE_SHA | `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` | P6 product tree: PR #537 merge (T-IGX-036-X), the last product-path change. Previous value `a156cbbd5ff521dc7e99dd02ab6bebd4ca234a73` (PR #525 merge) was superseded because T-IGX-036-X changed a product path (`intergrax/integrations/providers/embedding_provider/ollama/register.py`). Descends from every Dn publication and every P6 repair publication. The P6 acceptance record covers `a156cbbd5` / `884c177ad` (Part A, PR #532, with an addendum for #537); the EBH-7 replay at `146e7acf1` is pending. This row fixes the SHA, it records no PASS. Docs-only descendants do not change it |
| T_IGX_036_W_X_PUBLICATION_SHAS | see Notes | T-IGX-036-W and T-IGX-036-X publication facts (merge commits on `development`): PR #530 `28e370a37` (W plan); PR #531 `884c177ad` (W sign); PR #532 `e5f5bec08` (W Part A acceptance record); PR #533 `434249534` (X plan); PR #534 `3a7694cb7` (X sign); PR #535 `e417f7a89` (X amendment); PR #536 `8ce6233f0` (X re-sign); PR #537 `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` (X product fix). W1 (EBH-7 re-pin prep) and the replay are not yet published |
| TEST_EXECUTION_SHA | — | Must equal the audited SHA of each audit |
| Dn_AUDITED_SHA | — | P7 per domain; INTEGRATED_BASELINE_SHA or a corrected descendant |
| AUDIT_EXECUTION_SHA | — | IIAC `audit_execution_sha`; no product delta to HEAD at activation (ACT-N11) |
| EVIDENCE_PUBLICATION_SHA / CERTIFICATION_PUBLICATION_SHA | — | Evidence-only descendants (docs paths only) |
| GLOBAL_AUDITED_SHA | — | P8; must contain the MCP-SERVER-X closure publication |
| FINAL_CLOSURE_PUBLICATION_SHA | — | P9 merge |

Historical accepted SHAs of predecessor stages are reuse evidence only. Each reuse needs change-delta evidence from the accepted SHA to the audited SHA (PDC-P2-09). Being an ancestor does not make a commit a current-HEAD PASS.
