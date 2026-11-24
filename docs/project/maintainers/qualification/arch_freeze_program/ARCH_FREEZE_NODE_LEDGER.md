# ARCH-FREEZE node ledger (R0)

Packet: P-R0-3W. Maintainer-facing closure ledger for WP-12 DAG nodes and MEP-only identifiers.
Source draft: P-R0-3 node ledger (2026-11-24). DAG SSOT: [`WP12_DAG.md`](./WP12_DAG.md).

Status vocabulary: `DONE-FORMAL`, `REQ`, `PARTIAL`, `UNK`, `OPEN`, `VOID` (optional DAG nodes not activated).
REQ flag: `Y` iff status is `REQ`. SHA: acceptance or merge SHA when cited in tracked evidence; otherwise `NONE`.
Abbreviations: ODR = `ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`; LAD = `ARCH_FREEZE_SHA_LADDER.md`; GAP = `ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md`.

Row count: 149

| id | status | ticket | SHA | evidence | REQ flag | FRZ family |
|---|---|---|---|---|---|---|
| AUTH-PRE | DONE-FORMAL | - | NONE | ODR L37,L43 | N | - |
| DS-1 | DONE-FORMAL | NONE | NONE | ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md L82 | N | - |
| DS-2 | DONE-FORMAL | NONE | NONE | ODR L83 | N | - |
| DS-3 | DONE-FORMAL | NONE | NONE | ODR L84 | N | - |
| DS-4 | DONE-FORMAL | - | AF-02D | ODR L85; DS4_DECISION_RECONCILIATION_RECORD.md L21,L49; DAG L39 | N | - |
| DS-5 | DONE-FORMAL | NONE | NONE | ODR L86 | N | - |
| DS-6 | DONE-FORMAL | NONE | NONE | ODR L87 | N | - |
| DS-ADM | UNK | - | NONE | DAG L158; ODR L26 | N | - |
| DS-EXC | DONE-FORMAL | T-IGX-050-C | NONE | ODR L30 (DECIDED) | N | - |
| DS-Q | DONE-FORMAL | NONE | NONE | ODR L36; MEP L517 | N | EVOL, FRZ |
| ECAP-EXEC | UNK | T-IGX-052-E | NONE | ODR L63 (DP-54) | N | - |
| ENTRY | UNK | - | NONE | - | N | - |
| FCS | UNK | - | NONE | LAD L28 (FCS NOT SET) | N | - |
| M-NOW | REQ | NONE | NONE | eqo_a_remediation/OPERATOR_DECISIONS_EQOA_REMEDIATION.md; WP12_DAG.md | Y | - |
| M1 | DONE-FORMAL | - | feaa2a87d | DAG L67 | N | - |
| M3 | OPEN | - | NONE | LAD L29 (M3 NOT SET) | N | FRZ |
| PDC-TAIL | DONE-FORMAL | - | feaa2a87d | DAG L51; ARCH_FREEZE_SHA_LADDER.md L25 | N | - |
| PT-AUD-DOC | UNK | - | NONE | - | N | - |
| PT-AUD-ENF1 | UNK | - | NONE | - | N | AUD, REG |
| PT-AUD-ENF12 | UNK | - | NONE | - | N | AUD |
| PT-AUD-ENFA | UNK | - | NONE | - | N | - |
| PT-B01 | PARTIAL | - | NONE | plan section 1.4 (tooling only) | N | SEC |
| PT-B02 | PARTIAL | - | NONE | plan section 1.4 (tooling only) | N | - |
| PT-B1W | UNK | T-IGX-052-F | NONE | - | N | - |
| PT-B2W | UNK | T-IGX-052-F | NONE | - | N | - |
| PT-BA | UNK | T-IGX-052-F | NONE | - | N | - |
| PT-C01 | UNK | T-IGX-052-G | NONE | CHECKPOINT.md L42-47 (prep only) | N | - |
| PT-D01 | PARTIAL | T-IGX-042-A | NONE | governance/tickets/T-IGX-042-A.md; WBS L36 | N | DOC |
| PT-D02 | UNK | T-IGX-042-D | NONE | MEP L519 (#607) | N | DEBT |
| PT-D03 | UNK | T-IGX-046-A | NONE | - | N | - |
| PT-D04 | UNK | T-IGX-046-E | NONE | - | N | - |
| PT-D05 | REQ | T-IGX-042-C | NONE | WBS L36; MEP L519 (#606) | Y | DOC |
| PT-D06 | UNK | T-IGX-046-B | NONE | - | N | - |
| PT-D07 | REQ | T-IGX-042-B | NONE | WBS L36; MEP L519 (#608) | Y | DOC |
| PT-D08 | REQ | T-IGX-046-D | NONE | plan section 1.4 | Y | - |
| PT-D09 | UNK | - | NONE | - | N | - |
| PT-D10 | UNK | - | NONE | - | N | PRD |
| PT-D11 | UNK | - | NONE | - | N | - |
| PT-D12 | REQ | T-IGX-046-C | NONE | WBS L36 | Y | DOC |
| PT-D13 | UNK | T-IGX-052-F | NONE | - | N | - |
| PT-E00 | REQ | T-IGX-044-A | NONE | ent_audit_x/ENT_AUDIT_X_IIAC_MANIFEST_DRAFT_POINT_IN_TIME_RECORD.md | Y | AUD |
| PT-E01 | REQ | T-IGX-044-B | NONE | .github/workflows/enterprise-audit.yml | Y | REG |
| PT-E02 | REQ | T-IGX-044-C | NONE | .github/workflows/ticket-governance.yml | Y | REG |
| PT-E03 | UNK | - | NONE | RC0 L20-32 (NOT_MERGED) | N | REG |
| PT-E04 | UNK | T-IGX-044-E | NONE | - | N | - |
| PT-E05 | REQ | T-IGX-044-F | NONE | plan section 1.4 | Y | - |
| PT-FC1 | REQ | T-IGX-043-A/N | NONE | CHECKPOINT.md L120 (#680) | Y | PRD |
| PT-FC2 | PARTIAL | T-IGX-043-B/E | NONE | plan section 1.4 | N | REL |
| PT-FC3 | PARTIAL | T-IGX-043-C/F | NONE | CHECKPOINT.md L33 (413fb9186); RC0 L44-55 | N | PRD |
| PT-FC4 | REQ | T-IGX-043-D | NONE | plan section 1.4 | Y | - |
| PT-FRZ-RR | DONE-FORMAL | T-IGX-073-A/B/C | aaabfc6c8 | PT_FRZ_RR_FORMAL_CLOSEOUT_RECORD.md L11,L23,L24; ODR L95 | N | all FRZ rows |
| PT-G01 | PARTIAL | T-IGX-046-H | NONE | T-IGX-046-H L56-58; CHECKPOINT.md L134 | N | BND |
| PT-G02 | PARTIAL | T-IGX-046-G | NONE | T-IGX-046-G L15-17 | N | OWN, CTR, PLG |
| PT-PB | UNK | - | NONE | - | N | - |
| PT-Q02 | UNK | - | NONE | ARCH_FREEZE_SHA_LADDER.md L26 (G2 NOT SET) | N | all |
| PT-Q03 | UNK | - | NONE | LAD L27 | N | - |
| PT-Q04 | UNK | - | NONE | - | N | TEN |
| PT-Q05 | UNK | T-IGX-052-G | NONE | - | N | OWN |
| PT-Q06 | UNK | T-IGX-052-G | NONE | - | N | HIST, OBS |
| PT-Q07 | UNK | T-IGX-052-G | NONE | - | N | TEN |
| PT-Q08 | UNK | T-IGX-052-F | NONE | - | N | - |
| PT-Q09 | UNK | T-IGX-052-F | NONE | - | N | - |
| PT-Q10 | VOID | - | NONE | DAG L145 (OPTIONAL) | N | HIST |
| PT-Q11 | UNK | T-IGX-052-H | NONE | LAD L16-18 | N | - |
| PT-Q12 | UNK | T-IGX-052-G | NONE | - | N | HIST |
| PT-Q13 | UNK | - | NONE | - | N | HIST |
| PT-R01 | UNK | T-IGX-047-A | NONE | - | N | DEBT |
| PT-R02 | UNK | T-IGX-047-A | NONE | - | N | DEBT |
| PT-R03 | UNK | T-IGX-046-E | NONE | - | N | DEBT |
| PT-S01 | UNK | - | NONE | ODR L40 | N | AUD |
| PT-S02 | UNK | - | NONE | - | N | AUD |
| PT-S03 | UNK | - | NONE | - | N | AUD |
| PT-S04 | UNK | T-IGX-053-I | NONE | - | N | AUD |
| PT-S05 | UNK | - | NONE | - | N | - |
| PT-S06 | UNK | - | NONE | - | N | - |
| PT-S07 | UNK | - | NONE | DAG L60 (OPTIONAL, DP-67) | N | AUD |
| PT-T01 | UNK | T-IGX-065-A | NONE | - | N | - |
| PT-V02 | UNK | - | NONE | continuation/RC0_EVIDENCE.md L68 | N | - |
| PT-V03 | UNK | - | NONE | RC0 L59-73 (BLOCKED_EVIDENCE) | N | - |
| PT-V04 | PARTIAL | T-IGX-073-E | NONE | T-IGX-073-E L1,L9 | N | - |
| PT-V05 | UNK | - | NONE | RC0_EVIDENCE.md L71 | N | - |
| PT-V06 | UNK | - | NONE | RC0 L59-73 | N | REG |
| PT-VBN | UNK | - | NONE | - | N | RPL |
| PT-WA | REQ | T-IGX-043-A/G | NONE | CHECKPOINT.md L120 (#680) | Y | BND |
| PT-WB | PARTIAL | T-IGX-043-H | NONE | plan section 1.4 | N | OWN |
| PT-WD | PARTIAL | T-IGX-043-I/M | NONE | CHECKPOINT.md L40; RC0 L20-32 | N | - |
| PT-WE | VOID | - | NONE | DAG L104 (OPTIONAL, conditional on DP-43/RA-1) | N | - |
| PT-WF | PARTIAL | T-IGX-043-J | NONE | CHECKPOINT.md L34 (dc083f6ea) | N | SEC |
| REV-MASTER | PARTIAL | - | NONE | REV_MASTER_CORRECTIVE_CLOSURE_RECORD.md L22 | N | - |
| RW | UNK | - | NONE | - | N | - |
| SCOPE | UNK | - | NONE | DAG L85; ODR L54; continuation/RW_PLAN_AUDIT_STATUS.md | N | - |
| T-AF2-BASELINE-PIN | DONE-FORMAL | - | 7960ae7f6 | LAD L25 (AUDIT_BASELINE_2 PINNED) | N | - |
| T-AF2-BND-CLOSURE | UNK | - | NONE | RC0 L59-73 | N | BND |
| T-AF2-CI-HEAD-RUN | PARTIAL | - | 43858457516 | GAP L52; KNOWN_RED_REGISTER.md L12 | N | REG |
| T-AF2-CLOSED-WORLD | UNK | T-IGX-050-E (absent) | NONE | RC0 L59-73 | N | - |
| T-AF2-DAG-REGEN | REQ | - | NONE | DAG L85-87 | Y | - |
| T-AF2-DEPS-BATCH | REQ | T-IGX-051-H | NONE | plan section 1.4 | Y | - |
| T-AF2-DS-EXC-PACKAGE | REQ | T-IGX-050-C | NONE | GAP L42-45 (PR #612) | Y | - |
| T-AF2-ECAP-REVISION | UNK | T-IGX-050-I / T-IGX-052-D | NONE | - | N | - |
| T-AF2-ENT-BASELINE-PREFREEZE | UNK | T-IGX-052-A | NONE | - | N | - |
| T-AF2-EXPECTED-RED-PROBE | UNK | - | NONE | MEP L456; KRR L47,L60 | N | REG |
| T-AF2-FRZ-MATRIX | UNK | T-IGX-050-D | NONE | - | N | all |
| T-AF2-MAIN-SYNC | VOID | T-IGX-050-K (absent) | N/A | MEP L845 (NOT REQUIRED); GAP L52 | N | - |
| T-AF2-MANIFESTS | UNK | T-IGX-050-G / T-IGX-052-B | NONE | - | N | BND, CTR, DOC, FRZ |
| T-AF2-OD-RECORD | REQ | T-IGX-050-A | NONE | ODR exists; GAP L41 (PR #613) | Y | - |
| T-AF2-PLAN-PUBLISH | REQ | T-IGX-050-A | NONE | WP12_DAG.md; MEP-AF2_v2.1.md tracked | Y | - |
| T-AF2-POST-BASELINE | UNK | - | NONE | ODR L88-94 | N | - |
| T-AF2-RED-CLASSES | REQ | T-IGX-050-J | b3e89fd69 | GAP L49; KRR | Y | CMP, REG |
| T-AF2-REQUAL-HIST-PASS | UNK | - | NONE | - | N | HIST, CMP, TEN, EVOL, FRZ-11, AUD |
| T-AF2-RM499-ENTRY | UNK | T-IGX-050-F | NONE | - | N | - |
| T-AF2-RW-GATES | UNK | T-IGX-051-F | NONE | - | N | OBS |
| T-AF2-RW-PLAN | PARTIAL | T-IGX-073-D | NONE | T-IGX-073-D L5 | N | all CODE_FIX rows |
| T-AF2-RW-PLAN-AUDIT | UNK | T-IGX-050-N | NONE | continuation/RW_PLAN_AUDIT_STATUS.md; RC0 L59-73 | N | - |
| T-AF2-RW-RECOVERY | PARTIAL | T-IGX-051-E | NONE | RC0 L34-42 | N | REL |
| T-AF2-RW-RUNTIME | REQ | T-IGX-051-D | NONE | plan section 1.4 | Y | CTR, REL |
| T-AF2-RW-SUPPLY | REQ | T-IGX-051-B/N | NONE | CHECKPOINT.md L120-124 | Y | PLG, SEC |
| T-AF2-RW-TENANT | UNK | T-IGX-051-A | NONE | CHECKPOINT.md L131 | N | SEC |
| T-AF2-RW-TRUST | PARTIAL | T-IGX-051-C | NONE | CHECKPOINT.md L35 | N | SEC |
| T-AF2-RW-TYP | UNK | T-IGX-051-G | NONE | - | N | TYP |
| T-AF2-SSOT-RECONCILE-A | REQ | T-IGX-050-A | NONE | GAP L41 (merged PR #613) | Y | - |
| T-AF2-SSOT-RECONCILE-B | UNK | T-IGX-050-A2 | NONE | - | N | - |
| T-AF2-TYP-MEASURE | UNK | - | NONE | RC0 L59-73 | N | TYP |
| T-AF2-VALIDATOR-AMEND | REQ | T-IGX-050-B | NONE | T-IGX-050-L L1 | Y | - |
| T-AF2-VALIDATOR-INDEP-REVIEW | PARTIAL | - | NONE | T-IGX-050-L L1 | N | - |
| T-AF2-VBN-DOSSIER | UNK | T-IGX-050-H / T-IGX-052-C | NONE | - | N | RPL |
| T-AF2-W8-INDEP-CONFIRM | UNK | - | NONE | - | N | FRZ (01/02/09/10/13) |
| T-B7-CMP-REVAL | UNK | - | NONE | - | N | CMP |
| T-B7-COORD-ARCH-FRZ | UNK | - | NONE | - | N | - |
| T-B7-ECAP-2C | UNK | - | NONE | - | N | all |
| T-B7-FRZ-FRZ | UNK | - | NONE | - | N | FRZ |
| T-B7-LOCK-FREEZE-SHA | OPEN | - | NONE | LAD L29 (= M3) | N | FRZ |
| T-B7-PFP-ACTIVATE | UNK | - | NONE | - | N | EVOL |
| T-B7-PFP-VERIFY | UNK | - | NONE | - | N | EVOL |
| T-B7-PROMOTE-BND | UNK | - | NONE | - | N | BND |
| T-B7-PROMOTE-CTR | UNK | - | NONE | - | N | CTR |
| T-B7-PROMOTE-DEBT | UNK | - | NONE | - | N | DEBT |
| T-B7-PROMOTE-DOC | UNK | - | NONE | - | N | DOC |
| T-B7-PROMOTE-OBS | UNK | - | NONE | - | N | OBS |
| T-B7-PROMOTE-OWN | UNK | - | NONE | - | N | OWN |
| T-B7-PROMOTE-PLG | UNK | - | NONE | - | N | PLG |
| T-B7-PROMOTE-PRD | UNK | - | NONE | - | N | PRD |
| T-B7-PROMOTE-REG | UNK | - | NONE | - | N | REG |
| T-B7-PROMOTE-REL | UNK | - | NONE | - | N | REL |
| T-B7-PROMOTE-RPL | UNK | - | NONE | - | N | RPL |
| T-B7-PROMOTE-SEC | UNK | - | NONE | - | N | SEC |
| T-B7-PROMOTE-TYP | UNK | - | NONE | - | N | TYP |
| T-B7-TEN-REVAL | UNK | - | NONE | - | N | TEN |
| T-B8-ECAP-16 | UNK | - | NONE | - | N | - |
| T-B8-ROADMAP | UNK | - | NONE | - | N | - |
