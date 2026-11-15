# T-IGX-041-F — PDC-X batch B-P7-D6: repair of the independent D6 (RUNTIME-SECURITY) audit findings, loop 1

## Scope

Product and test repair after the independent audit IA-D6 (`PDC_P7_D6_AUDIT_REPORT.md`, verdict PARTIAL, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`), under the Operator's option B (`DELEGATION-PDC-P7-REPAIR.md`): real defects in existing code and test hygiene only. In scope: F-D6-01 (the evidence-plane security audit trail normalizes its persistence port with `as_evidence_persistence_port`, so its append goes through the validating port that the F-35 COMPAT-X classification assumes), the relabel part of F-D6-02 (the "immutable, replicated" wording of the multi-region trail is in-process bookkeeping), the binding part of F-D6-05 (plugin provenance requires the attested distribution to equal the entry point's distribution), F-D6-07 (the sandbox governance-deny unit test gets the governance identity it needs; a negative test pins the fail-closed contract), and by-product F-D6-12 (no provider text in the credential-unavailable error chain). Excluded with reason and reopen condition (plan, table "Excluded findings"): tamper evidence of the audit trail, F-D6-04 (production composition, F-04 family), the installed-file digest of F-D6-05, F-D6-06 (ticket T-IGX-041-I), and all other LOW/INFO findings. Product code: yes (`intergrax/runtime/security/`, `intergrax/core/plugins/`, `intergrax/integrations/credentials/`). No hotspot path is written. This ticket is not a certification verdict; IIAC activation stays stale until the P8-04 rebind and the EBH-7 identity gate stays red until the final re-pin. Plan: `PLAN_T-IGX-041-F.md`.

## Plan digest

Plan digest: 65cb0343e67e9193f424d0362e3a26b7e44ac51df6e524f6ae0fae5534a57574

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-F.md`](plans/PLAN_T-IGX-041-F.md).

## Approved

Approved: operator-delegate @ 2026-11-15T10:32:00+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPAIR.md@8f4b139c1f5352c9861872f0b5077996b8f0f891

Delegate checks (scope of `DELEGATION-PDC-P7-REPAIR.md`, all passed before this line was written; delegation read on `origin/development`, last changed by `8f4b139c1`; plan read at commit `277bc2d0f` of the batch branch). A first signing attempt at `e3979479e` was refused because the plan stated "1 failed, 10 passed" for the sandbox test (actual: 1 failed, 12 passed, 13 tests); the planner amended the plan and the checks were redone from scratch:

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`65cb0343e67e9193f424d0362e3a26b7e44ac51df6e524f6ae0fae5534a57574`); the amendment changed only plan wording and this digest line.
- The write set of the plan (6 entries: 4 exact product files, the new directory `tests/qualification/pdc_x/d6_repair/`, one existing test file) equals the write set of batch row `B-P7-D6` in the regenerated `PDC_IMPLEMENTATION_WBS.md` (lists compared as ordered lists, equal, empty symmetric difference).
- No hotspot path is in the write set; `intergrax/applications/_shared/` (including `security_audit_trail_wiring.py`) and `intergrax/contracts/` are out of scope and F-D6-04 is excluded for that reason. The product files lie in `intergrax/runtime/security/`, `intergrax/core/plugins/` and `intergrax/integrations/credentials/`.
- In-scope findings (F-D6-01, F-D6-02 relabel, F-D6-05 binding, F-D6-07, F-D6-12) and excluded findings (F-D6-02 tamper evidence and F-D6-05 installed-file digest = missing feature versus claim, F-D6-04 = F-04 family, F-D6-06 -> T-IGX-041-I, LOW and INFO to the debt register) are listed with reason and reopen condition.
- Code claims re-verified against the code at this head (the D6 product, COMPAT-X, `testing_support/` and `runtime/nexus/tools` paths are unchanged since the first attempt): `security_audit_trail.py:158-159` stores the port unchanged and `:213` calls `self._evidence.append`; `evidence_persistence_adapter.py:155` is `as_evidence_persistence_port` with the three branches described; `ValidatingEvidencePersistencePort.append` runs `prepare_canonical_production_write_event` and `assert_runtime_event_schema` and provides `list_for_task`; `security_audit_trail_wiring.py:67` passes the port through; `resolve_runtime_event_persistence` returns `ValidatingRuntimeEventPersistence` (store.py:67,71); `multi_region_audit_trail.py:23-26` report class has no docstring or field descriptions, `:43` `_region_entry_ids`, `:69` `verify_replication`, `:91` `seal_prefix`; `admission.py:259-268` matches by normalized mapping key and `:297` resolves the version of `attestation.distribution`; `secrets_store_resolver.py:48` `from exc` versus `:40` `from None`; `invoker.py:749` calls `record_governance_policy_decision_evidence_for_active_identity` inside `_require_current_attempt_authorization` (`:572`), reached from `_prepare_invocation` (`:457`, called at `:318` from `invoke`); the module docstring of the F-35 test carries the "wraps that port in the validating port" sentence. The sandbox test was re-run once by the delegate: 1 failed, 12 passed (13 tests), the failure being `RuntimeError("active execution governance identity required")` with a traceback through `invoke:318` -> `_prepare_invocation:457` -> `_require_current_attempt_authorization:749`, as the plan states. Claims the plan marks "(audit)" (probes A1, A2, PV1, Windows egress skips, `test_audit_ideal_depth_gate` reds) were not re-run by the delegate.
- COMPAT-X rule of T-IGX-040-D intact: the rule in `_compat_x_event_evolution_inventory.py` (path + qualname `append` + callee `self._evidence.append`) and its F-35 tests are not in the write set and are required to stay byte-identical; the planned change is in `__init__` only, so the rule keeps matching.
- No gate is weakened or deleted: the new tests are written first and shown failing; the only existing test edited is `test_cap_c1c_sandbox_evidence_and_contract_purity.py`, where the stale setup of one test gains the governance identity it needs with its assertions kept, and a new negative test pins the fail-closed refusal without identity; `test_d6_plugin_provenance.py` and the other D6 files stay unedited. F-D6-05 only narrows admission (rejects a mismatched attestation).
- No registry, scheduler, engine, trust, policy or audit authority is added; nothing under `scripts/governance/` or `.github/` is in the write set; the plan states it is not a certification verdict; the implementer (branch `pdc-x-041f`) and this delegate session are different sessions, and the delegate edited nothing except this approval block and the lifecycle state.
- Merge gate (stated as the delegation record requires): product pull requests merge under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`, as `DELEGATION-PDC-P7-REPAIR.md` names them; WBS merge gate OD-PDC-SEQ-01 also applies. The delegate did not verify that the numbering matches the numbered items of `DELEGATION-PDC-STANDING.md`. Write-set amendments are not delegated.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d6_repair tests/qualification/pdc_x/d6 tests/qualification/pdc_x/xd/test_xd_02_agent_context.py tests/qualification/pdc_x/xd/test_xd_06_background_security.py tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py tests/unit/runtime/sandbox tests/integration/runtime/sandbox/reference_substrate tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py tests/unit/integrations/credentials -q -p no:randomly
uv run python scripts/maintenance/check_immutable_security_audit_trail.py
uv run ruff check <touched files>
uv run pyright <touched files>
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Known reds that may remain and are not repaired here: the two `test_audit_ideal_depth_gate.py` failures (`29_1`, `30_4`), the physical egress tests skipped on Windows, EBH-7 identity, IIAC activation.

## Out of scope

- The excluded findings (plan table) and the TENANT-X carrier re-pin (T-IGX-041-I).
- `intergrax/applications/_shared/` (including `security_audit_trail_wiring.py`), `intergrax/contracts/`, `intergrax/runtime/nexus/`, `testing_support/`, the COMPAT-X inventory and its tests, `scripts/governance/`, `.github/`, FRZ files and P8 originals.
- IIAC activation and rebind, the EBH-7 re-pin, OD signatures, certification and the re-audit.

## Lifecycle state

plan_approved
