# EVOL-X — exact formats accepted by the closure preflight (B6 record and promoted FRZ rows)

**Ticket:** `T-IGX-019-A` (finding W401-RA-06) · **Source of truth:** `scripts/qualification/evol_x_closure_publication_preflight.py`
(`B6_RECORD_TITLE`, `B6_RECORD_FIELDS`, `_b6_record_exactness_errors`, `render_promoted_row`, `_gate_b_frz`).
This page only documents what the code already enforces. It authorises nothing, creates no record and promotes no FRZ row. A test
(`test_w401_ra_06_format_document_matches_the_code`) fails when this page and the code diverge.

## 1. B6 acceptance record (`iiac_evidence/EVOL-X/b6_acceptance/EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md`)

The record is a plain text file made of exactly these line kinds; anything else (extra prose, fenced blocks, other fields) is rejected by Gate B (`B-11`):

1. The title line, exactly: `# EVOL-X IIAC B6 acceptance record`
2. `Field: value` lines whose field name is one of the allow-listed names below (each stated once; blank lines are ignored).
3. One signature line: `Approved <Decision ID>: operator @ <ISO-8601 timestamp>` (written by the operator, never by an agent).

Allow-listed fields:

| Field | Required value |
|---|---|
| `Decision ID` | a bare identifier (letters, digits, `.`, `_`, `-`; at most 80 characters) |
| `Stage` | `EVOL-X` |
| `Policy version` | the IIAC policy version in force |
| `Decision scope` | exactly `B6 acceptance for EVOL-X (<verdict>, evidence-bound)`, with `<verdict>` the audited verdict (for example `PASS_SCOPED`) |
| `Decision status` | `AUTHORIZED` (any negated, revoked or non-affirmative status is rejected) |
| `Accepted verdict` | the audited verdict of the independent audit manifest |
| `Accepted limitations` | one comma-separated list of `<limitation id>:<sha256 of the limitation text>` entries; the set must equal the limitations certified at P2 (the content is bound, not only the ids) |
| `Independent audit manifest digest` | SHA-256 of the manifest blob |
| `Audit report digest` | SHA-256 from the manifest |
| `Findings digest` | SHA-256 from the manifest |
| `Audited implementation SHA` | 40-hex SHA from the manifest |
| `Certification publication SHA` | the pinned P2 certification publication |
| `Certifier session` | the certifier session id from the manifest |
| `Certifier agent` | `certification-governance-agent` |

## 2. Promoted FRZ-EVOL checklist row (`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`)

A row is promoted only after a signed B6 acceptance and a separate operator decision. Gate B then requires the row to have **exactly six cells**:

`| <FRZ-EVOL-nn> | <area> | <canonical requirement text> | <owner> | <status> | <notes> |`

* The requirement text must equal the canonical literal.
* `<status>` and `<notes>` must equal the machine rendering of the disposition certified at P2 (`render_promoted_row`):

| Certified disposition | Status cell | Notes cell |
|---|---|---|
| `ELIGIBLE_PASS` | `PASS` | `—` |
| `ELIGIBLE_PASS_SCOPED(a; b)` | `PASS — SCOPED` | `a; b` |
| `ELIGIBLE_PASS_EVOL_X_SCOPED(a)` | `PASS (EVOL-X SCOPED)` | `a` |

* The Evidence column of older rows cannot be kept: the row is the six-cell form above. Evidence is carried by the B6 record and the certification record.
* A scoped disposition can never be promoted to an unrestricted `PASS`, and the limitation ids in the disposition must appear in the row.
