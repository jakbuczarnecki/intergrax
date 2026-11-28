# ANNEX — T-IGX-079-E scoped qualification (R1)

**Parent plan (immutable):** `governance/tickets/plans/PLAN_T-IGX-079-E.md`
**Parent plan digest (binding):** `b40163343535234fc2c241faa4f2b423232147e0463e4e6d77cb7f35bce9632c`
**Ticket:** `governance/tickets/T-IGX-079-E.md`
**Operator authority:** `OD-II-W0-B-SCOPED-QUALIFICATION-01 — APPROVED` (preparation only)
**Annex status:** `PROPOSED_PENDING_CANONICAL_AUTHORITY_AND_INDEPENDENT_ACCEPTANCE`

This annex is supplemental scoped qualification authority for the documentation-only II-W0-05 corrective batch. It does **not** replace the parent plan, architecture WBS, or mandatory lifecycle gates.

## Mandatory qualification contract (preserved)

Q01–Q06, exact plan approval, canonical lifecycle, approved 50-path write-set plus this annex path, protected blob verification, link/anchor validation, PDC inventory checks, focused documentation tests, candidate-range `git diff --check`, independent review, formal independent G6, and separate Operator closure authority remain binding.

## Scope-specific historical qualification results (recorded exactly)

| Domain | Result | Notes |
|---|---|---|
| EVOL-X A5 | 354 executed / 309 passed / 45 failed | Population evidence preserved at historical candidate SHA; not re-run in this corrective batch |
| EVOL-X A8 | FAIL — accepted manifest unavailable | Does **not** assert causal link to P03 integrity failures |
| ENT-AUDIT-X | BLOCKED — two failed controls | Matched-baseline external finding; owners retain remediation |
| Governance smoke | PASS at historical candidate | Does not automatically qualify a new SHA |

## External qualification findings (unresolved)

- Failed P03 artifact-integrity tests (external domain)
- TEN-FINAL (external qualification finding)
- Two reported matched-baseline ENT-AUDIT-X control failures (signatures preserved in domain owners)

## Proportional acceptance proposal (documentation-only II-W0-05)

For the documentation-only package:

- Preserve actual EVOL-X impact/selection/safety evidence from Phase A diagnostic runs
- Do **not** require a new accepted global baseline solely to complete scoped documentary adoption
- Do **not** claim EVOL-X A8 PASS, ENT-AUDIT-X PASS, or external-domain certification
- Route external-domain findings to proper owners
- Require evidence-backed confirmation that scoped documentary acceptance is permitted by canonical governance before annex activation

This proposal does **not** supersede mandatory higher-order gates. If canonical governance requires separate owner acceptance, a revised approved plan digest, or a specific exception record, that requirement blocks annex activation until satisfied.

## Canonical authority requirements

| Requirement | Status |
|---|---|
| Independent review of corrective candidate | **Pending** |
| Formal independent G6 for II-W0-05 | **Pending** |
| Operator formal closure / acceptance | **Pending** |
| Annex activation by implementer declaration | **Prohibited** |

## Activation

The implementing agent must **not** activate this annex. Activation requires canonical authority and independent acceptance explicit in ticket lifecycle.

## Digest

Compute with: `uv run python scripts/governance/plan_digest.py governance/tickets/plans/ANNEX_T-IGX-079-E_SCOPED_QUALIFICATION_R1.md`

**Recorded after commit:** see terminal report `QUALIFICATION_ANNEX_DIGEST`.
