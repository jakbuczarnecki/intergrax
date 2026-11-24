# ARCH-FREEZE node ledger (R0)

Packet: P-R0-A (factual recount). Prior packet: P-R0-3W draft (2026-11-24).
DAG SSOT: [`WP12_DAG.md`](./WP12_DAG.md).

**Recount pins (2026-11-24, PT-Q02 audit delta):** `origin/development` = `c9ce6175f9a0da6a1dec622c4e4bb4b81683ea4e`; `origin/arch-freeze/rw-train` = `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a`. Local `arch-freeze/rw-train` ref may be stale â€” use `origin/` only.

Status vocabulary: `DONE-FORMAL`, `REQ`, `PARTIAL`, `UNK`, `OPEN`, `VOID` (optional DAG nodes not activated).
REQ flag: `Y` iff status is `REQ`. SHA: acceptance or merge SHA when cited in tracked evidence; otherwise `NONE`.
Abbreviations: ODR = `ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`; LAD = `ARCH_FREEZE_SHA_LADDER.md`; GAP = `ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md`.

Row count: 149

| id | status | ticket | SHA | evidence | REQ flag | FRZ family |
|---|---|---|---|---|---|---|
| AUTH-PRE | DONE-FORMAL | - | NONE | ODR L37,L43; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| DS-1 | DONE-FORMAL | NONE | NONE | ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md L82; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| DS-2 | DONE-FORMAL | NONE | NONE | ODR L83; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| DS-3 | DONE-FORMAL | NONE | NONE | ODR L84; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| DS-4 | DONE-FORMAL | - | AF-02D | ODR L85; DS4_DECISION_RECONCILIATION_RECORD.md L21,L49; DAG L39; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| DS-5 | DONE-FORMAL | NONE | NONE | ODR L86; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| DS-6 | DONE-FORMAL | NONE | NONE | ODR L87; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| DS-ADM | UNK | - | NONE | DAG L158; ODR L26; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| DS-EXC | DONE-FORMAL | T-IGX-050-C | NONE | ODR L30 (DECIDED); @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-050-C=plan_approved | N | - |
| DS-Q | DONE-FORMAL | NONE | NONE | ODR L36; MEP L517; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | EVOL, FRZ |
| ECAP-EXEC | UNK | T-IGX-052-E | NONE | ODR L63 (DP-54); @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-052-E=draft | N | - |
| ENTRY | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| FCS | UNK | - | NONE | LAD L28 (FCS NOT SET); train product @ be0ce70d not merged to dev (P-R0-A) | N | - |
| M-NOW | REQ | NONE | NONE | eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md; WP12_DAG.md; @dev 51edb2a7 / train be0ce70d (P-R0-A) | Y | - |
| M1 | DONE-FORMAL | - | feaa2a87d | DAG L67; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| M3 | OPEN | - | NONE | LAD L29 (M3 NOT SET); OPEN ladder (P-R0-A) | N | FRZ |
| PDC-TAIL | DONE-FORMAL | - | feaa2a87d | DAG L51; ARCH_FREEZE_SHA_LADDER.md L25; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| PT-AUD-DOC | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-AUD-ENF1 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | AUD, REG |
| PT-AUD-ENF12 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | AUD |
| PT-AUD-ENFA | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-B01 | PARTIAL | - | NONE | plan section 1.4 (tooling only); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | SEC |
| PT-B02 | PARTIAL | - | NONE | plan section 1.4 (tooling only); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| PT-B1W | UNK | T-IGX-052-F | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-B2W | UNK | T-IGX-052-F | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-BA | UNK | T-IGX-052-F | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-C01 | UNK | T-IGX-052-G | NONE | CHECKPOINT.md L42-47 (prep only); CHECKPOINT header stale (e69e1ff58); P-R0-A annex; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-052-G=plan_pending | N | - |
| PT-D01 | PARTIAL | T-IGX-042-A | NONE | governance/tickets/T-IGX-042-A.md; WBS L36; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-042-A=plan_approved | N | DOC |
| PT-D02 | UNK | T-IGX-042-D | NONE | MEP L519 (#607); @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-042-D=plan_approved | N | DEBT |
| PT-D03 | UNK | T-IGX-046-A | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-D04 | UNK | T-IGX-046-E | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-D05 | REQ | T-IGX-042-C | NONE | WBS L36; MEP L519 (#606); @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-042-C=plan_approved | Y | DOC |
| PT-D06 | UNK | T-IGX-046-B | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-D07 | REQ | T-IGX-042-B | NONE | WBS L36; MEP L519 (#608); @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-042-B=plan_approved | Y | DOC |
| PT-D08 | REQ | T-IGX-046-D | 397580bfe | PR #665 merge (agents/README.md, neo4j/USAGE.md); git log origin/arch-freeze/rw-train; @train e9bebe0ac; T-IGX-046-D=plan_approved | Y | - |
| PT-D09 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-D10 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | PRD |
| PT-D11 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-D12 | REQ | T-IGX-046-C | NONE | WBS L36; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-046-C=plan_approved | Y | DOC |
| PT-D13 | UNK | T-IGX-052-F | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-E00 | REQ | T-IGX-044-A | NONE | ent_audit_x/ENT_AUDIT_X_IIAC_MANIFEST_DRAFT_POINT_IN_TIME_RECORD.md; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-044-A=plan_approved | Y | AUD |
| PT-E01 | REQ | T-IGX-044-B | NONE | .github/workflows/enterprise-audit.yml; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-044-B=plan_approved | Y | REG |
| PT-E02 | REQ | T-IGX-044-C | NONE | .github/workflows/ticket-governance.yml; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-044-C=plan_approved | Y | REG |
| PT-E03 | UNK | - | NONE | RC0 L20-32 (NOT_MERGED); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | REG |
| PT-E04 | UNK | T-IGX-044-E | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-E05 | REQ | T-IGX-044-F | NONE | plan section 1.4; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-044-F=plan_approved | Y | - |
| PT-FC1 | REQ | T-IGX-043-A/N | 65ef1f196 | PR #680 train merge SHA; CHECKPOINT.md L120 (#680); CHECKPOINT header stale (e69e1ff58) â€” P-R0-A annex; T-IGX-043-A=plan_approved; @dev 51edb2a7 / train be0ce70d (P-R0-A) | Y | PRD |
| PT-FC2 | PARTIAL | T-IGX-043-B/E | 711ee4a11 | plan section 1.4; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-043-B=quality_gate | N | REL |
| PT-FC3 | PARTIAL | T-IGX-043-C/F | c443865e9 | PR #688 @ train; vision follow-up PR #812 e9bebe0ac; dev regression 413fb9186 cite-only; @train e9bebe0ac; T-IGX-043-C=plan_approved | N | PRD |
| PT-FC4 | REQ | T-IGX-043-D | 33a91bfdd | PR #668 merge intergrax/skills/execution_pinning_durability.py; git log origin/arch-freeze/rw-train; @train e9bebe0ac; T-IGX-043-D=plan_approved | Y | - |
| PT-FRZ-RR | DONE-FORMAL | T-IGX-073-A/B/C | aaabfc6c8 | PT_FRZ_RR_FORMAL_CLOSEOUT_RECORD.md L11,L23,L24; ODR L95; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-073-A=closed | N | all FRZ rows |
| PT-G01 | PARTIAL | T-IGX-046-H | NONE | T-IGX-046-H L56-58; CHECKPOINT.md L134; CHECKPOINT header stale (e69e1ff58); P-R0-A annex; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-046-H=plan_pending | N | BND |
| PT-G02 | PARTIAL | T-IGX-046-G | NONE | T-IGX-046-G L15-17; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-046-G=draft | N | OWN, CTR, PLG |
| PT-PB | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-Q02 | UNK | - | NONE | LAD L26 (G2 NOT SET); wave audit not PASS (P-R0-A) | N | all |
| PT-Q03 | UNK | - | NONE | LAD L27; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| PT-Q04 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | TEN |
| PT-Q05 | UNK | T-IGX-052-G | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | OWN |
| PT-Q06 | UNK | T-IGX-052-G | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | HIST, OBS |
| PT-Q07 | UNK | T-IGX-052-G | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | TEN |
| PT-Q08 | UNK | T-IGX-052-F | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-Q09 | UNK | T-IGX-052-F | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-Q10 | VOID | - | NONE | DAG L145 (OPTIONAL); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | HIST |
| PT-Q11 | UNK | T-IGX-052-H | NONE | LAD L16-18; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-052-H=plan_pending | N | - |
| PT-Q12 | UNK | T-IGX-052-G | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | HIST |
| PT-Q13 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | HIST |
| PT-R01 | UNK | T-IGX-047-A | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | DEBT |
| PT-R02 | UNK | T-IGX-047-A | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | DEBT |
| PT-R03 | UNK | T-IGX-046-E | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | DEBT |
| PT-S01 | UNK | - | NONE | ODR L40; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | AUD |
| PT-S02 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | AUD |
| PT-S03 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | AUD |
| PT-S04 | UNK | T-IGX-053-I | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | AUD |
| PT-S05 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-S06 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-S07 | UNK | - | NONE | DAG L60 (OPTIONAL, DP-67); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | AUD |
| PT-T01 | UNK | T-IGX-065-A | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| PT-V02 | UNK | - | NONE | continuation/RC0_EVIDENCE.md L68; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| PT-V03 | UNK | - | NONE | RC0 L59-73 (BLOCKED_EVIDENCE); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| PT-V04 | PARTIAL | T-IGX-073-E | NONE | T-IGX-073-E L1,L9; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-073-E=in_progress | N | - |
| PT-V05 | UNK | - | NONE | RC0_EVIDENCE.md L71; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| PT-V06 | UNK | - | NONE | RC0 L59-73; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | REG |
| PT-VBN | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | RPL |
| PT-WA | REQ | T-IGX-043-A/G | 65ef1f196 | CHECKPOINT.md L120 (#680); CHECKPOINT header stale (e69e1ff58); P-R0-A annex; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-043-A=plan_approved | Y | BND |
| PT-WB | PARTIAL | T-IGX-043-H | NONE | plan section 1.4; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-043-H=plan_approved | N | OWN |
| PT-WD | PARTIAL | T-IGX-043-I/M | NONE | CHECKPOINT.md L40; RC0 L20-32; CHECKPOINT header stale (e69e1ff58); P-R0-A annex; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-043-I=plan_approved | N | - |
| PT-WE | VOID | - | NONE | DAG L104 (OPTIONAL, conditional on DP-43/RA-1); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| PT-WF | PARTIAL | T-IGX-043-J | db69a9e44 | G-09 product e6b1d97f7 via PR #718; E2-E4 absent @ train (PLAN_T-IGX-043-J, DEF_CB_B2_S03_PT_WF_DISPOSITION.md); dev dc083f6ea not on train; T-IGX-043-J=in_progress | N | SEC |
| REV-MASTER | PARTIAL | - | NONE | REV_MASTER_CORRECTIVE_CLOSURE_RECORD.md L22; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| RW | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| SCOPE | PARTIAL | T-IGX-050-N | ebe45c269 | AR-12-07 RM-03 PASS @ ebe45c26930a9291a3df2d6a75028c35c036cee2; continuation/RW_PLAN_AUDIT_STATUS.md; RED_FIRST INCOMPLETE; G2/FCS/M3 NOT SET | N | - |
| T-AF2-BASELINE-PIN | DONE-FORMAL | - | 7960ae7f6 | LAD L25 (AUDIT_BASELINE_2 PINNED); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| T-AF2-BND-CLOSURE | UNK | - | NONE | RC0 L59-73; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | BND |
| T-AF2-CI-HEAD-RUN | PARTIAL | - | 43858457516 | GAP L52; KNOWN_RED_REGISTER.md L12; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | REG |
| T-AF2-CLOSED-WORLD | UNK | T-IGX-050-E (absent) | NONE | RC0 L59-73; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| T-AF2-DAG-REGEN | REQ | - | NONE | DAG L85-87; @dev 51edb2a7 / train be0ce70d (P-R0-A) | Y | - |
| T-AF2-DEPS-BATCH | REQ | T-IGX-051-H | NONE | plan section 1.4; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-051-H=plan_approved | Y | - |
| T-AF2-DS-EXC-PACKAGE | REQ | T-IGX-050-C | NONE | GAP L42-45 (PR #612); @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-050-C=plan_approved | Y | - |
| T-AF2-ECAP-REVISION | UNK | T-IGX-050-I / T-IGX-052-D | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| T-AF2-ENT-BASELINE-PREFREEZE | UNK | T-IGX-052-A | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| T-AF2-EXPECTED-RED-PROBE | UNK | - | NONE | MEP L456; KRR L47,L60; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | REG |
| T-AF2-FRZ-MATRIX | UNK | T-IGX-050-D | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | all |
| T-AF2-MAIN-SYNC | VOID | T-IGX-050-K (absent) | N/A | MEP L845 (NOT REQUIRED); GAP L52; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| T-AF2-MANIFESTS | UNK | T-IGX-050-G / T-IGX-052-B | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | BND, CTR, DOC, FRZ |
| T-AF2-OD-RECORD | REQ | T-IGX-050-A | NONE | ODR exists; GAP L41 (PR #613); @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-050-A=plan_approved | Y | - |
| T-AF2-PLAN-PUBLISH | REQ | T-IGX-050-A | NONE | WP12_DAG.md; MEP-AF2_v2.1.md tracked; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-050-A=plan_approved | Y | - |
| T-AF2-POST-BASELINE | UNK | - | NONE | ODR L88-94; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| T-AF2-RED-CLASSES | REQ | T-IGX-050-J | b3e89fd69 | GAP L49; KRR; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-050-J=plan_approved | Y | CMP, REG |
| T-AF2-REQUAL-HIST-PASS | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | HIST, CMP, TEN, EVOL, FRZ-11, AUD |
| T-AF2-RM499-ENTRY | UNK | T-IGX-050-F | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| T-AF2-RW-GATES | UNK | T-IGX-051-F | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | OBS |
| T-AF2-RW-PLAN | PARTIAL | T-IGX-073-D | NONE | T-IGX-073-D L5; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-073-D=closed | N | all CODE_FIX rows |
| T-AF2-RW-PLAN-AUDIT | UNK | T-IGX-050-N | NONE | continuation/RW_PLAN_AUDIT_STATUS.md L11,L13 (BLOCKED); T-IGX-050-N; no audit PASS @dev 51edb2a7 (P-R0-A) | N | - |
| T-AF2-RW-RECOVERY | PARTIAL | T-IGX-051-E | NONE | RC0 L34-42; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-051-E=plan_approved | N | REL |
| T-AF2-RW-RUNTIME | REQ | T-IGX-051-D | NONE | plan section 1.4; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-051-D=plan_approved | Y | CTR, REL |
| T-AF2-RW-SUPPLY | REQ | T-IGX-051-B/N | 711ee4a11 | CHECKPOINT.md L120-124; CHECKPOINT header stale (e69e1ff58); P-R0-A annex; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-051-B=plan_approved | Y | PLG, SEC |
| T-AF2-RW-TENANT | UNK | T-IGX-051-A | 876af7e5d | PR #811 partial slice only (full A1 write set absent @ e9bebe0ac); git log origin/arch-freeze/rw-train; T-IGX-051-A=plan_approved | N | SEC |
| T-AF2-RW-TRUST | PARTIAL | T-IGX-051-C | dbaebfa8c | Trust slices via PR #699 train-sync (e28de385f); NT-TRUST-06 trace @ 26b6115a1 (#717); T-IGX-051-C=in_progress | N | SEC |
| T-AF2-RW-TYP | UNK | T-IGX-051-G | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | TYP |
| T-AF2-SSOT-RECONCILE-A | REQ | T-IGX-050-A | NONE | GAP L41 (merged PR #613); @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-050-A=plan_approved | Y | - |
| T-AF2-SSOT-RECONCILE-B | UNK | T-IGX-050-A2 | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| T-AF2-TYP-MEASURE | UNK | - | NONE | RC0 L59-73; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | TYP |
| T-AF2-VALIDATOR-AMEND | REQ | T-IGX-050-B | NONE | T-IGX-050-L L1; @dev 51edb2a7 / train be0ce70d (P-R0-A); T-IGX-050-B=plan_approved | Y | - |
| T-AF2-VALIDATOR-INDEP-REVIEW | PARTIAL | - | NONE | T-IGX-050-L L1; @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | - |
| T-AF2-VBN-DOSSIER | UNK | T-IGX-050-H / T-IGX-052-C | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | RPL |
| T-AF2-W8-INDEP-CONFIRM | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | FRZ (01/02/09/10/13) |
| T-B7-CMP-REVAL | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | CMP |
| T-B7-COORD-ARCH-FRZ | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| T-B7-ECAP-2C | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | all |
| T-B7-FRZ-FRZ | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | FRZ |
| T-B7-LOCK-FREEZE-SHA | OPEN | - | NONE | LAD L29 (= M3); @dev 51edb2a7 / train be0ce70d (P-R0-A) | N | FRZ |
| T-B7-PFP-ACTIVATE | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | EVOL |
| T-B7-PFP-VERIFY | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | EVOL |
| T-B7-PROMOTE-BND | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | BND |
| T-B7-PROMOTE-CTR | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | CTR |
| T-B7-PROMOTE-DEBT | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | DEBT |
| T-B7-PROMOTE-DOC | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | DOC |
| T-B7-PROMOTE-OBS | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | OBS |
| T-B7-PROMOTE-OWN | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | OWN |
| T-B7-PROMOTE-PLG | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | PLG |
| T-B7-PROMOTE-PRD | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | PRD |
| T-B7-PROMOTE-REG | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | REG |
| T-B7-PROMOTE-REL | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | REL |
| T-B7-PROMOTE-RPL | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | RPL |
| T-B7-PROMOTE-SEC | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | SEC |
| T-B7-PROMOTE-TYP | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | TYP |
| T-B7-TEN-REVAL | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | TEN |
| T-B8-ECAP-16 | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
| T-B8-ROADMAP | UNK | - | NONE | no in-repo closure cite @dev 51edb2a7 (P-R0-A) | N | - |
