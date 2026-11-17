# T-IGX-043-F — EQO-A remediation: CONFIG-X semantic scan extension for vendor-slug defaults (gate-meaning change)

## Scope

Extend the CONFIG-X semantic production scan so that silent vendor/backend slug defaults (EQOA-A4-019 blind-spot statement; FRZ-CFG-02, FRZ-CFG-06 and FRZ-CFG-08 are PASS under CONFIG-X per the checklist) become visible to the gate: one edit of `tests/qualification/config_x/_config_x_semantic_production_scan.py` and one new test file `tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py`, 12 negative tests (NT-FC3-CX1..12). Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589). Split out of `T-IGX-043-C` after the independent delegate review.

**Classification: gate-meaning change.** It changes what the gate reports about FRZ-CFG-02/06/08. It is personal to the Operator: it is not approvable by a delegate or a commissioned session, and no implementer writes its approval line. Dependency order: it depends on `T-IGX-043-C` (its repo-level test is green only after the product fixes) and merges after C, or in the same wave behind C with the Operator's ruling, never before C; C does not depend on it for its own correctness.

Tests only, no product path: not ACT-N11 sensitive; outside the EBH-7 identity allowlist, so it merges before the final EBH-7 pin. No closure, PASS, FRZ promotion or independent-verification claim is made by this ticket. The Operator's ruling of 2026-11-16 (extend the scan, scoped to the eight FC-3 paths with a recorded finding per hit outside them; same wave behind `T-IGX-043-C`; scratch measurement F-A2 authorised and not yet run) is recorded in the plan, section 14, from `FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a session artefact, not a repository record); the approval line stays the Operator's own and is not written; the authorization entries (including the scratch measurement) are in `AUTHORIZATION_REQUESTS_T-IGX-043-F.md`. The LAB exemption is syntactic (a default reachable only under a `lab_opt_in` argument), with no module or path exemption.

## Plan digest

Plan digest: 60d2a16a20e23b167bcfebfe7977b72d869e26b09512904d630e93d8c5d303cf

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-F.md`](plans/PLAN_T-IGX-043-F.md). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line goes here after plan review and is written by the Operator personally, because this ticket is a gate-meaning change; it is not written by a commissioned session and never by the implementer. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

One process at a time; none was run when this ticket was drafted.

- `uv run pytest tests/qualification/config_x/test_config_x_fc3_vendor_slug_defaults.py -p no:cacheprovider` (12 negative tests; red at the baseline where stated; with `T-IGX-043-C` in the tree the repo-level test asserts an explicit expected-residual list, initially empty).
- `uv run pytest tests/qualification/config_x -p no:cacheprovider` (whole directory).
- Regression: one command over the 11 existing CONFIG-X node IDs listed in `regression_nodes_F.txt` of the plan package (verified to exist at the baseline and at the current head).
- `uv run python -m scripts.governance.igx_gov ticket` before push.
- Independent audit at the exact wave SHA by a separate non-implementing session (replay of the 12 tests, mutation, comparison of scan findings before and after); the implementer records no PASS.
- Rollback: revert the two-file commit (tests only).

## Out of scope

Every product path, `scripts/`, `.github/`, `docs/`, `governance/` other than this ticket and its plan, the inventory and classification modules of CONFIG-X, the EBH-3 scan extension and boundary gates (`PT-G01`), all files of `T-IGX-043-A..D`, any allowlist of providers, FRZ promotion, Docker, new vendor integrations and production configuration (C10/C15 freeze).

## Lifecycle state

`draft`
