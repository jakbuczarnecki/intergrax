# T-IGX-044-F — EQO-A remediation node PT-E05 (T-ENF-5): IIAC validator hardening on the material governance paths

Implementer session: local_f49f5be3-807a-4da8-b1cd-38eed5ae38d3

## Scope

Edits five of the six `MATERIAL_GOVERNANCE_PATHS` of `iiac_activation_gate.py` and adds one test file: `scripts/governance/iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, `iiac_activation_gate.py`, the IIAC certification protocol document, and `tests/unit/governance/test_e05_role_separation_identity.py`. Content (details in the plan): a v2 manifest marker with `draft`/`final` state and strict rules for new manifests only; normalised identity comparison and per-role identity provenance with host receipts; an operator countersignature line checked mechanically; operator line kinds (`personal`, `delegate`) with a delegation resolver in the activation gate; wiring of the PT-E04 corpus and artifact verifiers into the gate's companion calls, with the four files they depend on added to `MATERIAL_GOVERNANCE_PATHS`. Source: EQO-A Campaign A findings A2-038, A2-006 (consumed through PT-E04), WP-7 plan B designs V1, V2, V3; decisions DP-58, DP-68; RW-PLAN v13 Appendix A1 block `PT-E05` and row NT-ROW-010. Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`.

Consequence recorded here: until the single RB-2 re-pin, the activation gate reports `ACT-N11 stale audit after material governance change` for every edited material path. This ticket does not suppress that and does not re-pin.

Out of scope: `verify_iiac_policy_activation.py`, the activation record file, every published manifest, the handoff template, `iiac_g0_inventory.py`, every workflow, hook and `igx_gov` command, the ticket validators (PT-E02), the PT-E04 files themselves (read and executed, not edited), every product path, the IIAC re-pin (RB-2), the PDC-X and EBH-7 pins, GitHub settings and rulesets, and any PASS, closure, FRZ promotion or certification claim.

## Plan digest

Plan digest: 531ef8e1a60f1e4e1c42011935fcc847ca5e289336970466e7a0351846b5fb0e

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-044-F.md`](plans/PLAN_T-IGX-044-F.md) (drafted at `.tmp/session/arch-freeze/w3-packet-2/044-F/PLAN_T-IGX-044-F.md`; copied unchanged when the ticket is committed). The digest is computed with `scripts/governance/plan_digest.py` and must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T13:28:10+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_DELEGATION_RECORD_3.md@62dcc99faf7dc2ab95b83e8e5dae2ce845366ec2
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

One process at a time under the shared pytest lock, at most 15 minutes (plan section 8):

- `uv run pytest tests/unit/governance/test_e05_role_separation_identity.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_g0_inventory.py tests/unit/governance/test_iiac_new_verifiers.py -o cache_dir=<scratch dir>`
- `ruff check` and `ruff format --check` over the changed Python files.
- Red-first run of the new test file at the pre-change commit; reviewer check `git diff --stat origin/development...HEAD` lists exactly the paths of the plan's write set.
- Independent audit AUD-ENF-2 at the exact head SHA by the IIAC re-audit session (not the implementer, not the certifier).

## Out of scope

See Scope. No Docker, vendor integration or production configuration.

## Lifecycle state

plan_approved
