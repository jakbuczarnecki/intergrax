# SFX R2 — Publication handoff (IGX-SFX-R2-PUBLISH-001)

**Campaign closeout:** `IGX-SFX-R2-DOC-FINALIZATION-001`
**Status:** `PR_OPEN` · **NOT** merged to `development` · **Branch:** `integ/sfx-r2-corr-001`
**PR:** [#756](https://github.com/jakbuczarnecki/intergrax/pull/756)

## Publication lanes (separated)

| Lane | State | SHA / pointer |
| --- | --- | --- |
| R2 documentation **preparation** (T-IGX-070-A) | **Published on `origin/development`** | `10cf95df89e806191d5ff801917ece91ff144bad` |
| R2 **corrective** documentation (T-IGX-070-B) | **Integrator branch; PR open** | see `PUBLICATION_CANDIDATE_SHA` below |
| Ticket **070-B plan approval** | **Closed on `development`** (not in PR diff) | `3e5dedf26` |
| Ticket **070-A plan approval** | **Open** | `plan_pending` |
| **PR #756 merge** | **Blocked** — platform CI | not canonical R2 adoption |
| **`CANONICAL_R2_APPROVED`** | **NO** | operator decision |
| **`IMPLEMENTATION_AUTHORIZED`** | **NO** | ARCH-FREEZE + gates |

## Git pins

| Field | SHA / value |
| --- | --- |
| `CURRENT_REMOTE_BASE_SHA` | `3e5dedf267bc783a795ef20974a4e70809f05fbd` |
| `PUBLICATION_CANDIDATE_SHA` | _finalization commit on this branch_ |
| `REMOTE_PUBLICATION_SHA` | **NONE** until PR #756 merge |
| Prior corrective HEAD | `42fe39fbf1b2acf5108cc52ba4890d1b0d33e7ac` |

## Validation @ publication candidate

| Check | Result |
| --- | --- |
| JSON (6 schemas) | PASS |
| DAG (37 tasks) | PASS |
| REQ-R2 (18) | PASS |
| Readiness matrix | PASS |
| Independent audit | PASS_WITH_FINDINGS |
| Global link pytest | GLOBAL_PREEXISTING_FAILURE |

## Merge blockers (out of SFX scope)

| Check | Owner |
| --- | --- |
| `ent-audit-fast` ENT-A-PRD-01/02, ENT-A-REG-03 | Enterprise audit / platform qualification |
| `governance-fast` IIAC ACT-N10/N11 | IIAC governance platform |

## Operator next steps

1. Review dossiers + `r2/SFX_R2_IMPLEMENTATION_READINESS_MATRIX.md`.
2. Merge PR #756 when platform CI is green.
3. Pin `REMOTE_PUBLICATION_SHA` on `development` after merge.
4. Separate: `CANONICAL_R2_APPROVED`, HO-2, ECAP §16, PLAN_T-IGX-070-A approval.
