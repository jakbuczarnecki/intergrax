# PDC-X — SHA ladder

| Role | SHA | Notes |
| --- | --- | --- |
| BASELINE_SHA | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` | `origin/development` at PDC-P0 start (PR #459 merge) |
| ROADMAP_AMENDMENT_SHA | `bc9ef52bd690738e71638a599eec64c76ac356c9` | PR #461 merge on `development` (T-IGX-036-A); descends from BASELINE_SHA; contains P0 head `142190b313d304bdc6c6362a1d7484b88a802920` |
| P1_PUBLICATION_SHA | — | Closed-world inventory |
| P2_PUBLICATION_SHA | — | Requirements matrix, gap register, FRZ crosswalk |
| P3_PUBLICATION_SHA | — | Ownership matrix, ADRs, ODs (architecture lock) |
| P4_PUBLICATION_SHA | — | Implementation WBS, dependency DAG, cross-domain test matrix |
| Dn_IMPLEMENTATION_SHA / Dn_PUBLICATION_SHA | — | One pair per P5 lane (D1..D8); publication = merge commit on `development` |
| INTEGRATED_BASELINE_SHA | — | P6; must descend from every Dn publication |
| TEST_EXECUTION_SHA | — | Must equal the audited SHA of each audit |
| Dn_AUDITED_SHA | — | P7 per domain; INTEGRATED_BASELINE_SHA or a corrected descendant |
| AUDIT_EXECUTION_SHA | — | IIAC `audit_execution_sha`; no product delta to HEAD at activation (ACT-N11) |
| EVIDENCE_PUBLICATION_SHA / CERTIFICATION_PUBLICATION_SHA | — | Evidence-only descendants (docs paths only) |
| GLOBAL_AUDITED_SHA | — | P8; must contain the MCP-SERVER-X closure publication |
| FINAL_CLOSURE_PUBLICATION_SHA | — | P9 merge |

Historical accepted SHAs of predecessor stages are reuse evidence only. Each reuse needs change-delta evidence from the accepted SHA to the audited SHA (PDC-P2-09). Being an ancestor does not make a commit a current-HEAD PASS.
