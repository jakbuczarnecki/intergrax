# T-IGX-040-D — PDC-X finding F-35: classify the evidence-plane security audit trail as a COMPAT-X persistence site

## Scope

Test-infrastructure repair for finding F-35 (`OD-MCP-RC-04` section 2). Ten COMPAT-X event-evolution gates (four in `test_compat_x_event_evolution_r3_p0_gates.py`, six in `test_compat_x_event_evolution_r3_r1_gates.py`) are red on `development` because `EvidencePlaneSecurityAuditTrail` (`intergrax/runtime/security/security_audit_trail.py`, added by T-IGX-036-P) appends a canonical `RuntimeEvent` through the injected `EvidencePersistencePort` and constructs it at line 198; the gates' classifiers do not know the site (`canonical production persistence bypass count 1 (expected 0)`). The plan diagnoses it as a legitimate classified site (canonical validating evidence plane, registered typed payload, same pattern as the classified governance evidence persistence), not a bypass and not a product defect. The repair adds two narrow, typed classification rules on the test side (exact module + function + callee; no module-wide trust) and new pinned positive and negative tests written first; no gate is weakened or deleted, the two gate test files stay byte-identical. Product code: no. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-040-D.md`.

## Plan digest

Plan digest: 01717feb557c0cb983859fc539e4e0c6f3d0013157bb31a7e481bdfd4259d584

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-040-D.md`](plans/PLAN_T-IGX-040-D.md).

## Approved

Approved: operator-delegate @ 2026-11-15T06:53:04+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-F56-F35.md@88ee973db2f84a4be63493a40edf1ca0000a745b

Delegate checks (scope of `DELEGATION-PDC-F56-F35.md`, all passed before this line was written; read at the batch branch, base `61531828f`, commits `4ffdff976`, `9ca336a8f`, `88ee973db`):

- The plan text states "signing returns to the Operator"; that sentence predates the delegation. The Operator later answered "Tak, delegat podpisze" (2026-11-15, recorded in `DELEGATION-PDC-F56-F35.md`). The delegate treats the delegation record as the Operator's later, specific instruction; the plan text is not edited (it would change the digest).
- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`01717feb557c0cb983859fc539e4e0c6f3d0013157bb31a7e481bdfd4259d584`).
- The write set of the plan equals the diff of commit `9ca336a8f` (5 files: `_compat_x_event_evolution_inventory.py`, `_compat_x_event_evolution_r3_r1.py`, the new `test_compat_x_event_evolution_f35_security_audit_trail_site.py`, this ticket, the plan); it lies only in `tests/qualification/compat_x/` and batch documents; no path under a product root, `scripts/governance/`, `.github/`, `intergrax/contracts/` or `intergrax/applications/_shared/`. `test_compat_x_event_evolution_r3_p0_gates.py` and `test_compat_x_event_evolution_r3_r1_gates.py` are byte-identical to `origin/development` (`32728f8e9`).
- Diagnosis re-verified against `intergrax/runtime/security/security_audit_trail.py`: `EvidencePlaneSecurityAuditTrail` imports only the `EvidencePersistencePort` contract (no backend store); it appends through the injected port (`self._evidence.append`); the canonical composition (`as_evidence_persistence_port`, `resolve_runtime_event_persistence`) wraps the port in `ValidatingEvidencePersistencePort` / `ValidatingRuntimeEventPersistence`, both of which run `prepare_canonical_production_write_event` and `assert_runtime_event_schema` before delegating; the payload `SecurityAuditPayloadV1` is registered with `register_payload_schema(..., extension=True)`; the `RuntimeEvent(...)` constructor is built from live arguments (no `**raw`, no `json.loads` of a wire event, no wire `schema_version`). Not a persistence bypass. The residual observation (the trail itself accepts any port; the guarantee rests on the composition root) is accepted as out of scope, as in the plan.
- New rules are exact: the append rule needs module suffix `runtime/security/security_audit_trail.py`, enclosing function `append` and callee `self._evidence.append`; the constructor rule is one `(module suffix, "append", evidence)` tuple of `_CONSTRUCTOR_POSITIVE_EVIDENCE_RULES` (no module-wide `None` entry). Negative tests cover another function, other callees (`self._store.append`, `self._inner.append`), another module, a forced persisted-reconstruction shape, and injected unclassified append and constructor sites that still fail the P0 and R3-R1 gates.
- Normalization of `_compat_x_event_evolution_r3_r1.py`: the delegate compared the parsed syntax trees of `61531828f` and the batch commit; the import sets are equal, one import statement only changed position, and the only other differences are the three additions (enum member, frozenset member, rule tuple). The two `# noqa: UP042` comments are comments; no `StrEnum` switch.
- One `uv run pytest` over the 040-C and 040-D test files (log `.tmp/proof/pdc-x-f56/delegate-verify.log`): all four P0 gates, the F-35 tests and `test_d6_audit_trail_durable.py` pass. Seven R3-R1 closed-world inventory tests stay red (`raw_candidates 7412 != 7356`, `exclusions`, semantic-surface counts): the seven gates red at the pin named in the plan; no violation mentions `security_audit_trail`. The five `roadmap_replay_x` reds of the same run belong to 040-C's file set (unscoped prod delta since DUP-X) and are unrelated to this change.
- No registry, scheduler, engine, trust, policy or audit authority is introduced; the plan states it is not a certification verdict.
- Merge gate (stated as the delegation record requires): the test-only pull request that carries this batch merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`, as `DELEGATION-PDC-F56-F35.md` names them: green CI, only the known red classes, `headRefOid` equal to the verified SHA. The EBH-7 identity gate stays red until the campaign's final re-pin (accepted by the Operator). Not decided here: the delegate did not verify that the numbering used by the record matches the numbered items of `DELEGATION-PDC-STANDING.md`.
- The implementer session (`impl-f56`) and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state. Write-set amendments are not delegated and return to the Operator. This signature is not a certification verdict.

## Test contract

```text
uv run pytest tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py tests/qualification/pdc_x/d6/test_d6_audit_trail_durable.py -q
uv run ruff check tests/qualification/compat_x/_compat_x_event_evolution_inventory.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py
uv run pyright tests/qualification/compat_x/_compat_x_event_evolution_inventory.py tests/qualification/compat_x/_compat_x_event_evolution_r3_r1.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py
```

## Out of scope

- Any product path (including a runtime marker assertion in the audit trail), the two gate test files, the seven COMPAT-X gates that are red at the pin, `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`.
- Re-pinning EBH-7, the IIAC audit, activation and certification, and every other finding.

## Lifecycle state

plan_approved
