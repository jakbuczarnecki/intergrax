# SFX R2 — Publication handoff (IGX-SFX-R2-PUBLISH-002 + adoption)

**Campaign:** `IGX-SFX-R2-FINAL-PUBLICATION-002` · **Adoption:** `IGX-SFX-R2-CANONICAL-ADOPTION-001`
**Status:** `PUBLICATION_COMPLETE_ON_DEVELOPMENT` · **Ticket:** [`T-IGX-070-C`](../../../../../governance/tickets/T-IGX-070-C.md)

## Publication lanes

| Lane | State |
| --- | --- |
| T-IGX-070-A initial R2 package | `development` @ `10cf95df89e806191d5ff801917ece91ff144bad` (historical) |
| PR #756 corrective (partial) | **MERGED** @ `67b127f3b4a919c13d5d156ef792f359943cfe3f` |
| PR #757 finalization remainder (7 paths) | **MERGED** @ `7ddd53266d32ce47f059456e110e414b92248453` |
| T-IGX-070-C G2 plan approval | **APPROVED** @ `f33501780a62ae82a718834b6521adc856ec9bfb` |
| T-IGX-070-C G3 operator adoption | **APPROVED** @ `81c51263ea672e0f4d886dd0dc85bcf646849108` |
| R2 overlay in master plan (§18) | **In current documentation candidate** — not yet a new `development` publication SHA |
| `CANONICAL_R2_APPROVED` | **YES** (G3 decision authority) |
| `IMPLEMENTATION_AUTHORIZED` | **NO** |
| G6 + remote verification | **Required** before final master-plan publication on `development` |

## Git pins (immutable)

| Field | Value |
| --- | --- |
| PR #756 merge SHA | `67b127f3b4a919c13d5d156ef792f359943cfe3f` |
| PR #757 reviewed head SHA | `e2b2a2b7e72f35f1e6da4dad4d850d278a4f3e5d` |
| PR #757 merge / `REMOTE_PUBLICATION_SHA` | `7ddd53266d32ce47f059456e110e414b92248453` |
| SFX path diff (review head → merge) | **empty** (content-equivalent) |

## Independent review

| Stage | SHA / identity | Verdict |
| --- | --- | --- |
| Corrective (T-IGX-070-B) | `e5cd9879e` · igx-reviewer | PASS_WITH_FINDINGS |
| Finalization F-01..F-05 | `360cb4887f5045ddf0f8461f038fb593a2b61292` | PASS |
| Remainder #757 (7 files) | Reviewed @ `e2b2a2b7`; evidence `.tmp/proof/sfx-r2-publish-002/review.json` | ACCEPT_WITH_FINDINGS |

## Authority separation

| Concern | Status |
| --- | --- |
| Documentation publication | **COMPLETE** on `development` |
| Architecture overlay (`§18`) | **Candidate text** (T-IGX-070-C); G3 adoption **APPROVED** @ `81c51263`; final remote publication **after G6** |
| Implementation | **NOT AUTHORIZED** |
| Enterprise certification | **NOT CLAIMED** |

## Platform CI (unchanged; not retrograded to PASS)

At PR #757 HEAD: `governance-fast` and `ent-audit-fast` **failure**; `ent-audit-pr-gate` success. At merge base `67b127f3`: `governance-fast` already **failure**. GitHub ruleset *Protect development* did not require green checks. Owners: IIAC (governance-fast), ENT-AUDIT-X (ent-audit-fast).
