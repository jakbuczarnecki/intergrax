# T-IGX-043-D — EQO-A code wave FC-4: skill execution pin durability and KV durability attestation (EQOA-A4-002)

## Scope

Product-path fix for the Campaign A finding EQOA-A4-002 (baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589; related HIST-F-DUR-01 and FRZ-PRD-07; master-plan node `PT-FC4`, wave package W3). The two skill execution pin stores report `is_durable` as a literal `True`, so the production guard cannot refuse a non-durable backing store. The fix derives the flag from the backing store: a default-False `survives_process_restart` capability on `DistributedKVStore` (Sqlite declares True; Redis only through an explicit typed `KVDurabilityAttestation`, DP-31), the document store through its existing process-restart capability read strictly, and a guard that accepts only `is_durable is True` in `production_mode` (DP-24). Six product files change: `intergrax/distributed/contracts/kv_store.py`, `intergrax/distributed/providers/sqlite_kv_store.py`, `intergrax/distributed/providers/redis_kv_store.py`, `intergrax/integrations/providers/key_value_cache/redis/bundle.py`, `intergrax/skills/execution_pinning_persistence.py`, `intergrax/skills/execution_pinning_durability.py`. Tests: four new files and two edited files under `tests/unit/`. Docs: the attestation procedure in `docs/project/architecture/SKILLS.md`. Fourteen negative tests (a PRODUCT/STRICT configuration that must now be refused), 25 unchanged regression nodes plus six migrated nodes. The ticket joins the single integrated wave (DP-11) and is implemented only after milestone `M1` and the wave scope freeze; any product-path change breaks the IIAC activation until the single RB-2 re-pin (ACT-N11). It makes no PASS, closure, FRZ promotion or independent-verification claim; the independent audit at the exact wave SHA is a separate, non-implementing session.

Decided design: DP-31, DP-24, DP-11, DP-44 and the Operator's acceptance of the recommended answer to every draft question: a non-durable KV is refused even when a durable document store is supplied (no fallback); both Redis AOF modes (`always`, `everysec`) are attestable and the attestation is declared, not probed; the sibling stores with the same literal `True` (27 bodies in product code per the delegate review) are not fixed here and become rows of the Enterprise Debt & Exceptions Register (DP-46..DP-53), owner per family under DP-47/DP-48; the answers were relayed on 2026-11-16 and are transcribed in the published `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md` (an orchestrator transcription, not a signature); the Redis composition root `bundle.py` is in the write set; Sqlite's reference-host default path is an accepted, documented limitation; the audit records HIST-F-DUR-01 and the widened SKILL-X delta. OPEN-ATT-1 is resolved by fallback (b), as recorded in item 4 of the Operator-answers transcription named above: the `SKILLS.md` procedure states the attestation shape only and the ticket is approvable under (b). Still open (not decided): who may attest Redis durability (owner: the Operator; no authority grant is recorded), whether PDC P9 is formally closed, and the test authorisations. A plan approval is not an implementation go: implementation waits for the ARCH-FREEZE gate `SCOPE` and the Operator's product-path go. Train position: node `PT-FC4`, lane 2 of the RW product train, after PT-WA and PT-WB; no path of its write set is shared with another train ticket.

## Plan digest

Plan digest: 2dfe0ad9f28e5b583c3288ba079a272ad6df940fbc2c2abd1382bc0b3466dd30

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-D.md`](plans/PLAN_T-IGX-043-D.md). The digest was computed with `scripts/governance/plan_digest.py` on the plan text; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T13:28:10+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_DELEGATION_RECORD_3.md@62dcc99faf7dc2ab95b83e8e5dae2ce845366ec2
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

One process at a time, under the shared pytest lock; none run when this ticket was drafted:

- Red-first, at the pre-change commit: `uv run pytest tests/unit/skills/test_skill_execution_pinning_durability_guard.py tests/unit/distributed/contracts/test_kv_durability_attestation.py tests/unit/distributed/test_sqlite_kv_store_durability.py tests/unit/distributed/test_redis_kv_store_durability.py -p no:cacheprovider -q` (expected red set is listed in the plan, section 4.1); the same command green after the change.
- Regression, one command: `uv run pytest tests/unit/skills/test_skill_x_batch_c_versioning_provenance_recovery.py tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py tests/unit/skills/test_p1_10_execution_binding.py tests/unit/skills/test_skill_x_batch_d_trust_boundary_qualification.py tests/unit/skills/test_skill_x_batch_e_integrated_authority_tenant_qualification.py tests/unit/skills/test_version_pinning.py tests/unit/distributed tests/unit/integrations/providers/key_value_cache/test_redis.py tests/integration/marketplace/test_me15_marketplace_skill_composition_e2e.py tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py -p no:cacheprovider -q`.
- Static: `uv run ruff check` and `uv run ruff format --check` on the changed files; `uv run pyright` on the six product files and the new tests; `uv run python scripts/docs/check_docs_domain_pairs.py`.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the six product files, the six test files, the docs file(s) of the plan, and this ticket with its plan.
- Wave-level (not run by the FC-4 implementer): EBH-7 F10 classifier probe, COMPAT-X persisted inventory for the `distributed.kv_entry` family, ROADMAP-REPLAY-X product-delta dispositions, RB-2 IIAC re-pin, independent audit at the exact wave SHA with the mutation check of plan section 7.

## Out of scope

`intergrax/skills/skill_execution_pinning_wiring.py` and `intergrax/applications/_shared/harness_host_runtime.py` (unchanged by design); the shared helper `document_store_survives_process_restart`; the sibling KV or document-backed stores with a literal `is_durable` `True` (profile resolution, attempt lifecycle, execution terminal, delegated correlation, integration configuration) (decided: recorded separately, not fixed here); `production_platform_persistence.py` (default-path limitation accepted and documented); a live Redis probe and any live Redis evidence; reading the attestation from environment or config; edits to the SKILL-X Batch C/D/E evidence documents or any AUD-SKILLX custody record (DP-44); the roadmap, the FRZ checklist, IIAC records, `scripts/`, `.github/`, `pyproject.toml`, `uv.lock`; FRZ promotion, Docker, vendor integration, production configuration; any claim of closure, PASS or independent verification.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_e96f6733-22dd-4b36-9377-9f415abe77b2
