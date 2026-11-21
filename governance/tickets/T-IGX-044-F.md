# T-IGX-044-F — EQO-A remediation node PT-E05 (T-ENF-5): IIAC validator hardening on the material governance paths

Implementer session: local_f49f5be3-807a-4da8-b1cd-38eed5ae38d3

## Scope

Edits five of the six `MATERIAL_GOVERNANCE_PATHS` of `iiac_activation_gate.py`, edits the docstring of one PT-E04 file that becomes material, edits one existing test and adds one test file. Paths: `scripts/governance/iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, `iiac_activation_gate.py`, the IIAC certification protocol document; `scripts/governance/iiac_verifier_common.py` (module docstring only); `tests/unit/governance/test_iiac_new_verifiers.py` (only its PT-E04 scope guard, whose three assertions the wiring makes false and which is replaced by an exact-set check); and the new `tests/unit/governance/test_e05_role_separation_identity.py`. Content (details in the plan): a v2 manifest marker with `draft`/`final` state and strict rules for new manifests only; normalised identity comparison and per-role identity provenance with host receipts; an operator countersignature line checked mechanically (digest over the canonical JSON of the manifest without `handoff_record_ref`, fully specified in the plan); operator line kinds (`personal`, `delegate`) with a delegation resolver in the activation gate; wiring of the PT-E04 verifiers through the gate (the artifact verifier in `_companion_errors`, the corpus verifier once in `evaluate_activation` through `_corpus_errors`, skipped when the evaluated repository lacks `scripts/governance/iiac_known_draft_manifests.v1.json` or `docs/project/maintainers/qualification/`, with the exact call given in the plan), a gate check that a v2 manifest that is not final cannot back an activation, and the four PT-E04 files they depend on added to `MATERIAL_GOVERNANCE_PATHS`. The five-SHA strict chain and the binding of all four digests to their file bytes for a v2 final manifest run through the gate only; direct callers of `verify_iiac_audit_manifest.py`, in code and in three procedure documents, do not get them (plan section 2, Operator decision Q-E05-12). Source: EQO-A Campaign A findings A2-038, A2-006 (consumed through PT-E04), WP-7 plan B designs V1, V2, V3; decisions DP-58, DP-68; RW-PLAN v13 Appendix A1 block `PT-E05` and row NT-ROW-010. Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`.

Consequence recorded here: until the single RB-2 re-pin, the activation gate reports `ACT-N11 stale audit after material governance change` for every edited or newly material path (9 lines). This ticket does not suppress that and does not re-pin.

Out of scope: `verify_iiac_policy_activation.py`, the activation record file, every published manifest, the handoff template, `iiac_g0_inventory.py`, every workflow, hook and `igx_gov` command, the ticket validators (PT-E02), the PT-E04 files themselves (read and executed; only the module docstring of `iiac_verifier_common.py` is edited, see Scope; no code line of a PT-E04 file changes), every product path, the IIAC re-pin (RB-2), the PDC-X and EBH-7 pins, GitHub settings and rulesets, and any PASS, closure, FRZ promotion or certification claim.

## Plan digest

Plan digest: 3b5e5d4f776f747a6d509a6cdd11770428bda019dad682ff12840d17c2691de3

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-044-F.md`](plans/PLAN_T-IGX-044-F.md) (amendment 5, drafted at `.tmp/session/arch-freeze/w3-packet-2/044-F/PLAN_T-IGX-044-F_amend5.md`; copied unchanged when the ticket is committed). The digest is computed with `scripts/governance/plan_digest.py` and must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-18T09:48:29+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_STANDING_DELEGATION_RECORD.md@881428fb231f34215a67364aa5788a4f2f986e83
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

One process at a time under the shared pytest lock, at most 15 minutes (plan section 8):

- `uv run pytest tests/unit/governance/test_e05_role_separation_identity.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_g0_inventory.py tests/unit/governance/test_iiac_new_verifiers.py -o cache_dir=<scratch dir>`
- `ruff check` and `ruff format --check` over the changed Python files.
- Red-first run of the new test file and of the inverted guard at the pre-change commit: 20 of the 27 measured negative tests are measured in their final form (reds and the two controls as logged by the implementer; NT-E05-06 and NT-E05-19..24 are structural reds, not behavioural); seven tests (NT-E05-11, 12, 14, 17, 18, 25, 26) are rewritten for the amended design; step 0: before any code line is aligned the amended test file and the renamed guard alone are run once at `7fa785485`, and implementation stops if any outcome differs from the plan (plan sections 6a and 8); `NT-E05-28` and control C3 are regression guards, not red-first; `test_canonical_policy_is_in_force_and_verifies` and `test_n12_policy_is_in_force_only_with_valid_activation_evidence` are already red at the train head (ACT-N11 product-path lines for 62 distinct paths, registered expected-red set, KR-08) and this ticket adds 9 ACT-N11 stale-governance lines to them, compared by node id (the 9 added lines are informational); reviewer check `git diff --stat origin/arch-freeze/rw-train...HEAD` lists exactly the paths of the plan's write set.
- Independent audit AUD-ENF-2 at the exact head SHA by the IIAC re-audit session (not the implementer, not the certifier).

## Out of scope

See Scope. No Docker, vendor integration or production configuration.

## Lifecycle state

plan_approved
