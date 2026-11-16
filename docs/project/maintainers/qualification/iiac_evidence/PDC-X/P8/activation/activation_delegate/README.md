# PDC-X P8-04 activation delegate: scripts and logs

Written by the activation delegate session (`DELEGATION-PDC-P8-ACT.md` (c)); it is not an implementer, auditor or certifier. No pytest beyond the four named targeted tests below was run, one process at a time. Interpreter: the main checkout `.venv` (Python 3.12.11, has `jsonschema`) with the worktree as cwd.

| Log | What | Result |
| --- | --- | --- |
| `act_01_confirmations_1_3_4` | confirmations 1, 3 and 4 (script `confirm_p8_04_act.py`) at `3ad1b4363` | exit 0, all OK |
| `act_02a`, `act_02b`, `act_02c` | `verify_iiac_role_separation --strict-product-diff`, `verify_iiac_audit_manifest --require-handoff-template` and the certifier's manifest dry run on the binding manifest at `3ad1b4363` (confirmation 2) | exit 0 each |
| `act_02d_policy_activation_current_record` | `verify_iiac_policy_activation --require-in-force` on the record before the rebind | exit 1: ACT-N11 for the previous binding (the staleness this rebind removes) |
| `act_03_phase1_policy_activation` | the same verifier at the phase-1 commit `9f05749bc` | exit 1, exactly the four expected lines (missing `effective_from_sha`, missing `policy_activation_sha`, two 40-char SHA lines) |
| `act_04_phase2_policy_activation` | the same verifier at the phase-2 commit `293bb5b3c` | exit 0, `IN_FORCE @ IIAC-2026-11-05.1 from 9f05749bc23592868f3d0a7f505f9edbb2cfd025` |
| `act_05`, `act_06` | role separation and audit manifest validators on the binding manifest at phase 2 | exit 0 each |
| `act_07_targeted_pytest` | four tests: `test_canonical_policy_is_in_force_and_verifies`, `test_act_n12_effective_sha_pre_activation_blob`, `test_act_n12_self_referential_head` (`tests/unit/governance/test_iiac_activation_gate.py`) and `test_n12_policy_is_in_force_only_with_valid_activation_evidence` (`test_iiac_negative_guards.py`) | `4 passed` with `PYTHONPATH` set to the worktree |

Environment note on `act_07`: the first two attempts of this call are not in the log (it was overwritten). The first used an unsupported `-p no:cacheprovider` flag (exit 4, no tests ran). The second, without `PYTHONPATH`, failed two tests with ACT-N11 errors for the previous pin: those tests start the verifier as `python scripts/governance/verify_iiac_policy_activation.py`, so `scripts` then resolves through the main checkout's editable install and the verifier reads the main checkout's old record, not this worktree. With `PYTHONPATH` set to the worktree the same four tests pass. CI checks out this tree directly and is not affected.

`record_edit.py` is the script that produced the two record edits (phase 1 and phase 2).
