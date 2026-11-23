# SFX R2 — Publication handoff (IGX-SFX-R2-PUBLISH-002 + adoption)

**Campaign:** `IGX-SFX-R2-FINAL-PUBLICATION-002` · **Adoption:** `IGX-SFX-R2-CANONICAL-ADOPTION-001`
**Status:** `R2_OVERLAY_PUBLISHED_ON_DEVELOPMENT` · **Ticket:** [`T-IGX-070-C`](../../../../../governance/tickets/T-IGX-070-C.md)

## Publication lanes

| Lane | State |
| --- | --- |
| T-IGX-070-A initial R2 package | `development` @ `10cf95df89e806191d5ff801917ece91ff144bad` (historical) |
| PR #756 corrective (partial) | **MERGED** @ `67b127f3b4a919c13d5d156ef792f359943cfe3f` |
| PR #757 finalization remainder (7 paths) | **MERGED** @ `7ddd53266d32ce47f059456e110e414b92248453` |
| T-IGX-070-C G2 plan approval | **APPROVED** @ `f33501780a62ae82a718834b6521adc856ec9bfb` |
| T-IGX-070-C G3 operator adoption | **APPROVED** @ `81c51263ea672e0f4d886dd0dc85bcf646849108` |
| T-IGX-070-C overlay commit (§18 package) | **PUBLISHED** @ `1cd4c621ad7eee35526fb24eaedbfdc265a20404` |
| T-IGX-070-C G6 whitespace candidate | **PUBLISHED** @ `9e4eb2d07df8621378830efd26e6c9db6a0f337a` |
| `development` HEAD (post-overlay, non-SFX) | `9d3a042fcb7219ab344e526e036ae3e3ca3dde9e` |

## Git pins (immutable)

| Field | Value |
| --- | --- |
| PR #756 merge SHA | `67b127f3b4a919c13d5d156ef792f359943cfe3f` |
| PR #757 reviewed head SHA | `e2b2a2b7e72f35f1e6da4dad4d850d278a4f3e5d` |
| PR #757 merge / `REMOTE_PUBLICATION_SHA` | `7ddd53266d32ce47f059456e110e414b92248453` |
| SFX path diff (review head → merge) | **empty** (content-equivalent) |
| G2 plan approval | `f33501780a62ae82a718834b6521adc856ec9bfb` |
| G3 OD signing (BASE) | `81c51263ea672e0f4d886dd0dc85bcf646849108` |
| First T-IGX-070-C overlay on `development` | `1cd4c621ad7eee35526fb24eaedbfdc265a20404` |
| G6 documentation candidate | `9e4eb2d07df8621378830efd26e6c9db6a0f337a` |

## Campaign posture (current)

| State | Value |
| --- | --- |
| `R2_ARCHITECTURE_APPROVED` | **YES** (G3 @ `81c51263`) |
| `R2_OVERLAY_PUBLISHED_ON_DEVELOPMENT` | **YES** (`9e4eb2d07` on `development`) |
| `G6_CONTENT_VERDICT` | **PASS_WITH_FINDINGS** @ audited `9e4eb2d07` |
| `G6_GOVERNANCE_VERDICT` | **NONCOMPLIANT** (publication before G6) |
| `FORMAL_TICKET_CLOSURE` | **PENDING** |
| `IMPLEMENTATION_AUTHORIZED` | **NO** |
| `ENTERPRISE_PRODUCT_CERTIFIED` | **NO** |

## Independent review

| Stage | SHA / identity | Verdict |
| --- | --- | --- |
| Corrective (T-IGX-070-B) | `e5cd9879e` · igx-reviewer | PASS_WITH_FINDINGS |
| Finalization F-01..F-05 | `360cb4887f5045ddf0f8461f038fb593a2b61292` | PASS |
| Remainder #757 (7 files) | Reviewed @ `e2b2a2b7`; evidence `.tmp/proof/sfx-r2-publish-002/review.json` | ACCEPT_WITH_FINDINGS |
| T-IGX-070-C G6 (post-publication) | Audited `9e4eb2d07` · host `local_039b59d4-5f1c-4360-8fd7-28315b244478` | CONTENT PASS_WITH_FINDINGS; GOVERNANCE NONCOMPLIANT |

## Authority separation

| Concern | Status |
| --- | --- |
| R2 dossier publication (PR #756/#757) | **COMPLETE** on `development` |
| Architecture overlay (`§18`) | **PUBLISHED** on `development` @ `9e4eb2d07` (G3 adoption **APPROVED** @ `81c51263`) |
| Implementation | **NOT AUTHORIZED** |
| Enterprise product certification | **NOT CLAIMED** |

## Governance — historical procedure gap (F-GOV-01)

Overlay commits (`1cd4c621a`, `9e4eb2d07`) reached `development` **before** independent G6 sign-off. This is a **historical procedure violation**; it is **not** reversed, rewritten in git, or automatically accepted. Operator disposition on F-GOV-01 and formal ticket lifecycle remain **open** (`FORMAL_TICKET_CLOSURE = PENDING`). G6 corrective documentation does **not** close `T-IGX-070-C`.

## Platform CI (unchanged; not retrograded to PASS)

At PR #757 HEAD: `governance-fast` and `ent-audit-fast` **failure**; `ent-audit-pr-gate` success. At merge base `67b127f3`: `governance-fast` already **failure**. GitHub ruleset *Protect development* did not require green checks. Owners: IIAC (governance-fast), ENT-AUDIT-X (ent-audit-fast).