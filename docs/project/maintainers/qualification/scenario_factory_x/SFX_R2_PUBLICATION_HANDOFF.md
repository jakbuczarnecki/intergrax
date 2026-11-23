# SFX R2 — Publication handoff (IGX-SFX-R2-PUBLISH-002 + adoption)

**Campaign:** `IGX-SFX-R2-FINAL-PUBLICATION-002` · **Adoption:** `IGX-SFX-R2-CANONICAL-ADOPTION-001`  
**Status:** `PUBLICATION_COMPLETE_ON_DEVELOPMENT` · **Ticket:** [`T-IGX-070-C`](../../../../../governance/tickets/T-IGX-070-C.md)

## Publication lanes

| Lane | State |
| --- | --- |
| T-IGX-070-A initial R2 package | `development` @ `10cf95df89e806191d5ff801917ece91ff144bad` (historical) |
| PR #756 corrective (partial) | **MERGED** @ `67b127f3b4a919c13d5d156ef792f359943cfe3f` |
| PR #757 finalization remainder (7 paths) | **MERGED** @ `7ddd53266d32ce47f059456e110e414b92248453` |
| R2 overlay in master plan | Prepared under T-IGX-070-C; **effective when OD signed** |
| `CANONICAL_R2_APPROVED` | **NO** until `OD-SFX-R2-CANONICAL-ADOPTION` signed |
| `IMPLEMENTATION_AUTHORIZED` | **NO** |

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
| Architecture overlay (`§18`) | **PUBLISHED text**; binding adoption **PENDING OD signature** |
| Implementation | **NOT AUTHORIZED** |
| Enterprise certification | **NOT CLAIMED** |

## Platform CI (unchanged; not retrograded to PASS)

At PR #757 HEAD: `governance-fast` and `ent-audit-fast` **failure**; `ent-audit-pr-gate` success. At merge base `67b127f3`: `governance-fast` already **failure**. GitHub ruleset *Protect development* did not require green checks. Owners: IIAC (governance-fast), ENT-AUDIT-X (ent-audit-fast).
