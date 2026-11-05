# EVOL-X — master plan Part VII mechanical coverage matrix (Session C auditor results)

Each ID below carries the auditor's own result. `PASS — SCOPED` rows name their limitation (LIM-*) and finding (F-C-*) in the dossier. Rows marked PROCEDURAL are certifier/operator gates, not audit evidence.

| Section | ID | Tests replayed (TES+HEAD) | Auditor result | Note |
|---|---|---:|---|---|
| VII-A EVOL-01..10 | EVOL-01 | 11 | PASS — SCOPED | Graph-derived impact; unknown -> fail closed; 0 production consumers missed vs independent AST closure. |
| VII-A EVOL-01..10 | EVOL-02 | 8 | PASS | S1 escape=0; escapes carry witness. |
| VII-A EVOL-01..10 | EVOL-03 | 20 | PASS | INCOMPATIBLE detected, consumers enumerated, hidden consumer found. |
| VII-A EVOL-01..10 | EVOL-04 | 19 | PASS | 128-input exhaustive: unsafe_adoption=0. |
| VII-A EVOL-01..10 | EVOL-05 | 25 | PASS — SCOPED | 8 fault classes fail closed; scope = EVOL_CONSERVATIVE_FULL. |
| VII-A EVOL-01..10 | EVOL-06 | 21 | PASS — SCOPED | Qualification-layer DM-01..07 (384 combinations). |
| VII-A EVOL-01..10 | EVOL-07 | 9 | PASS — SCOPED | Passive Q1-Q6 dossier; deterministic; no runtime coupling. |
| VII-A EVOL-01..10 | EVOL-08 | 9 | PASS — SCOPED | M-10 satisfied: 0 unauthorized core edits; registry limited to 9 surfaces. |
| VII-A EVOL-01..10 | EVOL-09 | 4 | PASS — SCOPED | T1..T9 executable; freshness counter. |
| VII-A EVOL-01..10 | EVOL-10 | 6 | PASS | 8-stage pipeline; CLI e2e; failures return non-zero. |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-01 | 3 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-02 | 3 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-03 | 2 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-04 | 2 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-05 | 2 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-06 | 2 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-07 | 2 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-08 | 2 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-09 | 2 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-B FRZ-EVOL-01..10 | FRZ-EVOL-10 | 2 | SEE frz_evol_acceptance_matrix.json | Criterion-level result is recorded only in the dedicated FRZ matrix (no duplication, no mass PASS). |
| VII-C CW-01..10 | CW-01 | 2 | PASS | unclassified=0; negative injection detected |
| VII-C CW-01..10 | CW-02 | 1 | PASS | dual-owner conflicts=0 |
| VII-C CW-01..10 | CW-03 | 1 | PASS | orphan imports=0 |
| VII-C CW-01..10 | CW-04 | 2 | PASS | exclusion predicates X-03/X-06/X-07 proven |
| VII-C CW-01..10 | CW-05 | 2 | PASS | pytest-collect vs AST reconciled; fails closed without collect |
| VII-C CW-01..10 | CW-06 | 1 | PASS | text oracle parity (M-02) |
| VII-C CW-01..10 | CW-07 | 1 | PASS | 10/10 mutation probes flip |
| VII-C CW-01..10 | CW-08 | 2 | PASS | dual build digest parity — Windows local replay + Ubuntu CI both green against the same committed digests |
| VII-C CW-01..10 | CW-09 | 1 | PASS | QUAL/COMPAT parity explained |
| VII-C CW-01..10 | CW-10 | 3 | PASS | incremental == full; stale fact cache detected |
| VII-D D1..D10 | D1 | 3 | PASS | universe scan complete |
| VII-D D1..D10 | D2 | 8 | PASS | symbol extraction |
| VII-D D1..D10 | D3 | 2 | PASS | 19 REL typed; dynamic -> UNKNOWN not dropped |
| VII-D D1..D10 | D4 | 9 | PASS | owner rules load/validate |
| VII-D D1..D10 | D5 | 1 | PASS | X register |
| VII-D D1..D10 | D6 | 1 | PASS | CW suite runs ten gates |
| VII-D D1..D10 | D7 | 3 | PASS | UNKNOWN register explicit |
| VII-D D1..D10 | D8 | 2 | PASS | digest stable and tamper-evident |
| VII-D D1..D10 | D9 | 2 | PASS | incremental is development-only; equals full |
| VII-D D1..D10 | D10 | 4 | PASS | digest chain MANIFEST -> ARTIFACT_DIGESTS -> VL3 independently verified (27/27 digests recomputed); ENT handoff pin W7-06 belongs to certifier |
| VII-E S1..S8 | S1 | 3 | PASS | positive: blast_apps=['app_a'] escapes=[] / negative: escapes=['module:applications.app_b.escape'] naive_blast=['app_a'] blast=['app_a', 'app_b'] |
| VII-E S1..S8 | S2 | 4 | PASS | positive: consumers=['module:agents.fx_planner.plan', 'module:applications.app_a.billing', 'module:applications.app_b.ship', 'module:applications.app_c.hidden'] orphans=0 breaking=INCOMPATIBLE / negative: hidden=['module:applications.app_c.hidden'] naive_missed=['module:applications.app_c.hidden'] |
| VII-E S1..S8 | S3 | 3 | PASS | positive: matrix_rows=5 pin_preserved=True / negative: production_rejects_repin=True unsafe_adoptions=0 |
| VII-E S1..S8 | S4 | 3 | PASS | positive: rewrite_set=[] extension=EXTENSION_ONLY / negative: rewrite_required=['module:applications.app_a.use_concrete'] |
| VII-E S1..S8 | S5 | 3 | PASS | positive: migrate=MIGRATION_REQUIRED additive=COMPATIBLE / negative: no_migration=INCOMPATIBLE reasons=['type changed:total', 'no migration registered (silent break)'] |
| VII-E S1..S8 | S6 | 3 | PASS | positive: detected=8/8 baseline=SELECTED / negative: bypass_count=0 of 64 compound injections |
| VII-E S1..S8 | S7 | 3 | PASS — SCOPED | positive: dm_outcomes=7/7 ad_rollback_ok=True / negative: dm04=REFUSE false_safe=0 ad_layer_alone_succeeds=True |
| VII-E S1..S8 | S8 | 3 | PASS | positive: scaffold=EXTENSION_ONLY real_poc_files=35 / negative: core_edit=CORE_TOUCH_UNAUTHORIZED unauthorized_surfaces=1 |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-01 | — | PASS | MEASURED 40/40 threshold=0.95 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-02 | — | PASS | MEASURED 11528/11528 threshold=0.9 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-02C | — | PASS | MEASURED 0/11528 threshold=0.0 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-03 | — | PASS — SCOPED | MEASURED 2082/2082 threshold=0.98 — Independent AST oracle; arcs A1..A6 are self-surface arcs; identical results independently re-derived. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-03R | — | PASS — SCOPED | MEASURED 2082/2082 threshold=1.0 — Independent AST oracle; arcs A1..A6 are self-surface arcs; identical results independently re-derived. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-03C | — | PASS — SCOPED | MEASURED 342/342 threshold=1.0 — Independent AST oracle; arcs A1..A6 are self-surface arcs; identical results independently re-derived. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-03E | — | PASS — SCOPED | MEASURED 0/2082 threshold=0.0 — Independent AST oracle; arcs A1..A6 are self-surface arcs; identical results independently re-derived. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-04 | — | PASS | MEASURED 8/8 threshold=1.0 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-05 | — | PASS | MEASURED 2/2 threshold=1.0 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-06 | — | PASS | MEASURED 0/1 threshold=0.0 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-07 | — | PASS | MEASURED 4/4 threshold=1.0 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-08 | — | PASS | MEASURED 2/2 threshold=1.0 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-09A | — | PASS — SCOPED | MEASURED 3/3 threshold=1.0 — Independently re-derived by Mode A (registry oracle, diagnostic digests, adversarial control). |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-09H | — | NOT_MEASURED — scoped by OD-EVOL-13 (not a failure, not a pass) | NOT_MEASURED 0/0 threshold=1.0 — Human drill not performed; verifier forbids forged M-09H (probe T6/T12 detected). |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-10 | — | PASS | MEASURED 0/2 threshold=0.0 — Independently re-derived at TES: numerator/denominator/population digest identical. |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-09T | — | PASS | MEASURED 3/3 threshold=1.0 — independently re-derived |
| VII-F EVOL-M-01..11 (+supplements) | EVOL-M-11 | — | PASS | MEASURED 9/9 threshold=1.0 — independently re-derived |
| VII-G findings F-P0/F-P1/F-P2 | F-P0-01 | 1 | PASS | leave-one-out detection real |
| VII-G findings F-P0/F-P1/F-P2 | F-P0-02 | 2 | PASS | EVOL manifest cannot be shrunk |
| VII-G findings F-P0/F-P1/F-P2 | F-P0-03 | 52 | PASS | 52 negatives + auditor Mode A (12) and Mode B (5) tamper probes; HEAD-only replay refused |
| VII-G findings F-P0/F-P1/F-P2 | F-P0-04 | 8 | PASS | anti-vacuity guards for all S1..S8 negatives |
| VII-G findings F-P0/F-P1/F-P2 | F-P0-05 | 22 | PASS | FRZ tests behavioral |
| VII-G findings F-P0/F-P1/F-P2 | F-P0-06 | 22 | PASS | real populations; independent re-derivation identical |
| VII-G findings F-P0/F-P1/F-P2 | F-P1-01 | 1 | PASS | semantic RELs beyond imports |
| VII-G findings F-P0/F-P1/F-P2 | F-P1-02 | 23 | PASS — SCOPED | TESTED_BY direct-import only (F-C-03); in-scope under-selection probe: 0 |
| VII-G findings F-P0/F-P1/F-P2 | F-P1-03 | 23 | PASS | deterministic selection decision identity |
| VII-G findings F-P0/F-P1/F-P2 | F-P1-04 | 4 | PASS | flag smokes replaced by harness arms |
| VII-G findings F-P0/F-P1/F-P2 | F-P1-05 | 4 | PASS | fixture trees exist (collected and executed) |
| VII-G findings F-P0/F-P1/F-P2 | F-P1-06 | 15 | PASS | all ten CW gates present |
| VII-G findings F-P0/F-P1/F-P2 | F-P1-07 | — | PASS — OD-EVOL-01..10,13 signed; OD-EVOL-11/12/EVOLX-08 handled per IIAC migration (see dossier §10) | register verified; legacy OD-11/12/EVOLX-08 remain PENDING in the register |
| VII-G findings F-P0/F-P1/F-P2 | F-P1-08 | — | PASS | RC-7 formal exclusion (OD-EVOL-10) verified; the ImportError still reproduces at HEAD (tracked platform defect, outside EVOL scope) |
| VII-G findings F-P0/F-P1/F-P2 | F-P2-01 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | roadmap §5 ledger sync is a certifier publication step (W7-05); auditor edits no roadmap |
| VII-G findings F-P0/F-P1/F-P2 | F-P2-02 | 1 | PASS | semantic graph digest |
| VII-G findings F-P0/F-P1/F-P2 | F-P2-03 | 74 | PASS | expanded negatives in junit |
| VII-H mandatory evidence artifacts | MANIFEST.json | — | PASS | bound to TES; no self-referential field |
| VII-H mandatory evidence artifacts | ARTIFACT_DIGESTS.json | — | PASS | 27/27 digests independently recomputed over git blobs at FINAL |
| VII-H mandatory evidence artifacts | evol_test_execution_manifest.v1 | — | PASS | 290 node ids; content digest verified; 2 documented exclusions (OD-EVOL-10) |
| VII-H mandatory evidence artifacts | evol_junit_outcomes.v1.json | — | PASS | 290 passed; equals auditor's independent replay outcome map |
| VII-H mandatory evidence artifacts | inventory/graph/selection manifests | — | PASS | inventory_binding + m03_causal_audit bound to TES |
| VII-H mandatory evidence artifacts | junit XML archive | — | PASS | CI artifact sha256 69c6a9ad… (not byte-comparable to local replay by design); auditor local replay sha256 fe8abfce… |
| VII-H mandatory evidence artifacts | evol_metrics.v1 | — | PASS — SCOPED | M-09H NOT_MEASURED by OD-EVOL-13 (plan row says 'no NOT_MEASURED at closure': operator-scoped deviation) |
| VII-H mandatory evidence artifacts | 16 experiment verdicts | — | PASS | 8x2 arms, all passed |
| VII-H mandatory evidence artifacts | evol_frz_evidence.v1 | — | PASS — SCOPED | 10 rows; FRZ-EVOL-06 scope_footnote empty (F-C-02) |
| VII-H mandatory evidence artifacts | B6 PASS record | — | PENDING — this audit's pass record is the auditor deliverable; certification publication belongs to certifier | see EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md |
| VII-H mandatory evidence artifacts | ENT handoff | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | W7-06 ENT-AUDIT-X handoff pin is a certifier publication step |
| VII-I operator decision gates | OD-EVOLX-01 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED (register) |
| VII-I operator decision gates | OD-EVOL-01 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED |
| VII-I operator decision gates | OD-EVOL-09 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED |
| VII-I operator decision gates | OD-EVOL-10 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED |
| VII-I operator decision gates | OD-EVOL-02 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED |
| VII-I operator decision gates | OD-EVOL-06 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED |
| VII-I operator decision gates | OD-EVOL-05a/c | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED |
| VII-I operator decision gates | OD-EVOL-05b | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED — Option 1 (qualification-only) |
| VII-I operator decision gates | OD-EVOL-03/07/08 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | SIGNED / APPROVED (+OD-EVOL-08-M1) |
| VII-I operator decision gates | OD-EVOL-11 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | Human auditor identity NOT NAMED — superseded by IIAC auditor_session_id (migration matrix); IIAC handoff record operator attestation PENDING |
| VII-I operator decision gates | OD-EVOL-12 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | PENDING in register — IIAC migration: operator FRZ row mass promotion -> certifier gate + per-row evidence |
| VII-I operator decision gates | OD-EVOLX-08 | — | PROCEDURAL — NOT AN AUDIT-EVIDENCE ITEM (certifier/operator gate) | PENDING in register — IIAC migration: operator stage closure -> certifier + FORMAL_CLOSURE_PUBLICATION_SHA + preflight |
